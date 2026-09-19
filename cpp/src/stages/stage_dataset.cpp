#include "actlab/stages/stage_dataset.hpp"

#include "actlab/dataset_schema.hpp"
#include "actlab/npy_io.hpp"
#include "actlab/path_utils.hpp"
#include "actlab/proc.hpp"
#include "actlab/stages/stage_evidence.hpp"

#include <chrono>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <vector>

namespace actlab {

StageResult run_stage_dataset(RunContext& ctx) {
  const auto start = std::chrono::steady_clock::now();
  const std::filesystem::path python = ctx.python_bin();
  const std::filesystem::path script = ctx.experiment_root() / "scripts" / "py" / "scan_dataset_ref.py";
  const std::filesystem::path data_root = ctx.resolve(ctx.config().paths.data_dir);
  const std::filesystem::path dataset_root = data_root / ctx.config().dataset.repo_id;
  const std::filesystem::path download_log = ctx.log_file("07_dataset_download.txt");
  const std::filesystem::path schema_log = ctx.log_file("08_dataset_schema.txt");
  const std::filesystem::path schema_md = ctx.log_file("08_dataset_schema.md");
  const std::filesystem::path viz_log = ctx.log_file("09_dataset_viz.txt");
  const std::filesystem::path viz_out_dir = ctx.experiment_root() / "results";
  ensure_dir(viz_out_dir);
  const std::filesystem::path samples_dir = ctx.experiment_root() / "artifacts" / "dataset_samples";

  if (!std::filesystem::exists(script)) {
    return StageResult::fail("dataset", "missing controlled python script: " + script.string());
  }
  ensure_dir(samples_dir);

  std::ostringstream download_text;
  download_text << "# Stage S4 dataset acquisition\n# timestamp: " << timestamp_iso8601() << "\n";
  download_text << "# repo_id: " << ctx.config().dataset.repo_id
                << "\n# source priority: " << ctx.config().dataset.source_priority
                << "\n# HF_LEROBOT_HOME: " << data_root.string() << "\n\n";

  // 受控 Python 调用点 (b)：数据加载/统计扫描（含真实样本导出，供 C++ 侧推理使用）
  ProcResult scan = ctx.run_stack(
      {python.string(), script.string(), "--repo-id", ctx.config().dataset.repo_id, "--root",
       data_root.string(), "--samples-out", samples_dir.string(), "--fallback-repo-id",
       ctx.config().dataset.fallback_repo_id, "--viz-out", viz_out_dir.string(), "--chunk-size",
       std::to_string(ctx.config().act.chunk_size)},
      "07_scan_dataset_ref.txt", 3600);
  download_text << scan.output << "\n[exit_code=" << scan.exit_code << " duration_ms="
                << static_cast<long long>(scan.duration_ms) << "]\n";
  {
    std::ofstream out(download_log, std::ios::trunc);
    out << download_text.str();
  }
  if (scan.exit_code != 0) {
    return StageResult::fail("dataset", "scan_dataset_ref.py failed (see " + download_log.string() + ")");
  }

  // S5：C++ 侧 schema 审计（独立于 Python 的第二次校验）
  const DatasetSchemaReport report =
      audit_dataset_schema(dataset_root, {}, ctx.config().dataset.expected_fps,
                           ctx.config().dataset.expected_state_dim,
                           ctx.config().dataset.expected_action_dim,
                           ctx.config().dataset.min_episodes);
  {
    std::ofstream out(schema_log, std::ios::trunc);
    out << "summary: " << report.summary() << "\n\n" << render_schema_markdown(report);
  }
  {
    std::ofstream out(schema_md, std::ios::trunc);
    out << render_schema_markdown(report);
  }
  if (!report.ok()) {
    std::ostringstream problems;
    for (const auto& problem : report.problems) {
      problems << problem << "; ";
    }
    return StageResult::fail("dataset", "schema audit failed: " + problems.str());
  }

  ctx.manifest().set_field("dataset", "repo_id", ctx.config().dataset.repo_id);
  ctx.manifest().set_field("dataset", "codebase_version", report.codebase_version);
  ctx.manifest().set_field("dataset", "episodes", report.total_episodes);
  ctx.manifest().set_field("dataset", "frames", report.total_frames);
  ctx.manifest().set_field("dataset", "fps", report.fps);
  ctx.manifest().set_field("dataset", "state_dim", report.state_dim);
  ctx.manifest().set_field("dataset", "action_dim", report.action_dim);
  ctx.manifest().set_field("dataset", "camera_keys",
                           report.camera_keys.empty() ? "" : report.camera_keys.front() + " (+" +
                                                                  std::to_string(report.camera_keys.size() - 1) + ")");
  ctx.manifest().add_artifact("S4", "log", download_log, "dataset acquisition");
  ctx.manifest().add_artifact("S5", "log", schema_log, "schema audit");

  // S6：可视化 —— 真实帧拼图（不依赖 GUI），并在可用时生成 rerun .rrd
  std::ostringstream viz_text;
  viz_text << "# Stage S6 dataset visualization\n# timestamp: " << timestamp_iso8601() << "\n\n";
  std::vector<std::string> artifacts;
  {
    // 证据拼图：读取 scan_dataset_ref.py 导出的真实帧（float32 CHW [0,1]）
    std::vector<NpyArray> images;
    const std::filesystem::path meta_path = samples_dir / "samples_meta.json";
    int sample_count = 8;
    if (std::filesystem::exists(meta_path)) {
      const Json meta = Json::parse_file(meta_path.string());
      if (meta.has("count")) {
        sample_count = static_cast<int>(meta.int_or_throw("count"));
      }
    }
    for (int index = 0; index < sample_count; ++index) {
      std::ostringstream name;
      name << "sample_" << std::setfill('0') << std::setw(3) << index << "_cam0.npy";
      const std::filesystem::path image_path = samples_dir / name.str();
      if (!std::filesystem::exists(image_path)) {
        continue;
      }
      const NpyArray chw = read_npy(image_path);
      NpyArray hwc;
      hwc.shape = {chw.shape[1], chw.shape[2], 3};
      hwc.dtype = "<u1";
      hwc.data = float_chw_to_uint8_hwc(chw);
      images.push_back(std::move(hwc));
    }
    if (!images.empty()) {
      const std::filesystem::path montage_ppm = samples_dir / "frames_montage.ppm";
      write_montage_ppm(images, montage_ppm, 4, 2);
      const std::string convert_bin = find_executable("convert");
      const std::filesystem::path montage_png = ctx.experiment_root() / "results" / "dataset_frames.png";
      if (!convert_bin.empty()) {
        const ProcResult conv =
            run_process({convert_bin, montage_ppm.string(), montage_png.string()}, ProcOptions{});
        viz_text << "montage from " << images.size() << " real dataset frames -> " << montage_png.string()
                 << " (exit=" << conv.exit_code << ")\n";
        if (conv.exit_code == 0) {
          artifacts.push_back(montage_png.string());
          ctx.manifest().add_artifact("S6", "figure", montage_png, "real dataset frames montage");
        }
      }
      viz_text << "montage source: " << montage_ppm.string() << "\n";
      artifacts.push_back(montage_ppm.string());
    } else {
      viz_text << "[WARN] no sample frames exported by scan_dataset_ref.py\n";
    }
  }

  // 官方可视化产物（rerun .rrd，真实回放数据；体积较大，故单独记录并标注）
  for (const auto& entry : std::filesystem::directory_iterator(viz_out_dir)) {
    if (entry.is_regular_file() && entry.path().extension() == ".rrd") {
      ctx.manifest().add_artifact("S6", "visualization", entry.path(), "official dataset visualization (rerun)");
      artifacts.push_back(entry.path().string());
    }
  }
  viz_text << "\n# viz 由受控脚本 scan_dataset_ref.py --viz-out 触发（同一调用点，避免新增 Python 进程）\n";
  {
    std::ofstream out(viz_log, std::ios::trunc);
    out << viz_text.str();
  }
  ctx.manifest().add_artifact("S6", "log", viz_log, "dataset visualization");
  ctx.manifest().save();

  const ScreenshotResult download_shot = capture_terminal_screenshot(
      ctx, "07_dataset_download", "s4_dataset", "cat " + download_log.string() + " | tail -25", 18);
  const ScreenshotResult viz_shot = capture_terminal_screenshot(
      ctx, "09_dataset_visualization", "s6_viz",
      "ls -la " + samples_dir.string() + " | head -15", 18);
  const ScreenshotResult schema_shot = capture_terminal_screenshot(
      ctx, "08_dataset_structure", "s5_schema", "head -34 " + schema_md.string(), 18);

  StageResult result = StageResult::pass("dataset", report.summary());
  result.artifacts.push_back(download_log.string());
  result.artifacts.push_back(schema_log.string());
  result.artifacts.insert(result.artifacts.end(), artifacts.begin(), artifacts.end());
  result.evidence.push_back(download_shot.captured ? download_shot.path.string()
                                                   : "SCREENSHOT_UNAVAILABLE: " + download_shot.note);
  result.evidence.push_back(viz_shot.captured ? viz_shot.path.string()
                                              : "SCREENSHOT_UNAVAILABLE: " + viz_shot.note);
  result.evidence.push_back(schema_shot.captured ? schema_shot.path.string()
                                                 : "SCREENSHOT_UNAVAILABLE: " + schema_shot.note);
  result.duration_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
  return result;
}

}  // namespace actlab

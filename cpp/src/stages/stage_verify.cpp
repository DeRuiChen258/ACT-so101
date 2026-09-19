#include "actlab/stages/stage_verify.hpp"

#include "actlab/path_utils.hpp"
#include "actlab/proc.hpp"
#include "actlab/sha256.hpp"
#include "actlab/stages/stage_evidence.hpp"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <sstream>
#include <vector>

namespace actlab {
namespace {

std::vector<std::filesystem::path> collect_checkpoint_dirs(const std::filesystem::path& output_dir) {
  std::vector<std::filesystem::path> dirs;
  const std::filesystem::path checkpoints = output_dir / "checkpoints";
  if (!std::filesystem::exists(checkpoints)) {
    return dirs;
  }
  for (const auto& entry : std::filesystem::recursive_directory_iterator(checkpoints)) {
    if (entry.is_directory() && entry.path().filename() == "pretrained_model") {
      dirs.push_back(entry.path());
    }
  }
  std::sort(dirs.begin(), dirs.end());
  return dirs;
}

}  // namespace

std::filesystem::path find_latest_checkpoint(const RunContext& ctx, const std::string& job_name) {
  const std::filesystem::path train_root = ctx.resolve(ctx.config().paths.outputs_dir) / "train";
  if (!std::filesystem::exists(train_root)) {
    return {};
  }
  // 1) 优先精确匹配 <job_name>（避免 "act_baseline" 误吞 "act_baseline_smoke"）
  {
    const std::filesystem::path exact = train_root / job_name;
    if (std::filesystem::exists(exact)) {
      const auto found = collect_checkpoint_dirs(exact);
      if (!found.empty()) {
        return found.back();
      }
    }
  }
  // 2) 再匹配 <job_name>_<timestamp> 形式的多次运行目录
  std::vector<std::filesystem::path> dirs;
  for (const auto& entry : std::filesystem::directory_iterator(train_root)) {
    if (!entry.is_directory()) {
      continue;
    }
    const std::string name = entry.path().filename().string();
    if (name.rfind(job_name + "_", 0) != 0) {
      continue;
    }
    const auto found = collect_checkpoint_dirs(entry.path());
    dirs.insert(dirs.end(), found.begin(), found.end());
  }
  std::sort(dirs.begin(), dirs.end());
  return dirs.empty() ? std::filesystem::path{} : dirs.back();
}

StageResult run_stage_verify(RunContext& ctx) {
  const auto start = std::chrono::steady_clock::now();
  const std::filesystem::path python = ctx.python_bin();
  const std::filesystem::path export_script =
      ctx.experiment_root() / "scripts" / "py" / "export_act_torchscript.py";
  const std::filesystem::path artifacts_dir = ctx.resolve(ctx.config().paths.artifacts_dir) /
                                              ctx.config().act.job_name;
  const std::filesystem::path torchscript_dir = artifacts_dir / ctx.config().paths.torchscript_rel;
  const std::filesystem::path samples_dir = ctx.experiment_root() / "artifacts" / "dataset_samples";
  const std::filesystem::path verify_log = ctx.log_file("15_checkpoint.txt");
  const std::filesystem::path reload_log = ctx.log_file("16_model_reload.txt");

  // S12：checkpoint 发现与完整性校验（不只看"文件存在"）
  std::vector<std::string> job_candidates = {ctx.config().act.job_name,
                                             ctx.config().act.job_name + "_smoke"};
  std::filesystem::path checkpoint;
  std::string chosen_job;
  for (const auto& job : job_candidates) {
    checkpoint = find_latest_checkpoint(ctx, job);
    if (!checkpoint.empty()) {
      chosen_job = job;
      break;
    }
  }
  std::ostringstream verify;
  verify << "# Stage S12 checkpoint verification\n# timestamp: " << timestamp_iso8601() << "\n";
  if (checkpoint.empty()) {
    verify << "no checkpoint found under "
           << (ctx.resolve(ctx.config().paths.outputs_dir) / "train").string() << "\n";
    std::ofstream out(verify_log, std::ios::trunc);
    out << verify.str();
    return StageResult::fail("verify", "no checkpoint found (train stage must run first)");
  }
  verify << "checkpoint: " << checkpoint.string() << "\njob: " << chosen_job << "\n\nfiles:\n";
  bool has_weights = false;
  bool has_config = false;
  std::vector<std::filesystem::path> files;
  for (const auto& entry : std::filesystem::directory_iterator(checkpoint)) {
    if (!entry.is_regular_file()) {
      continue;
    }
    files.push_back(entry.path());
    const std::string name = entry.path().filename().string();
    if (name.find("model") != std::string::npos && name.find(".safetensors") != std::string::npos) {
      has_weights = true;
    }
    if (name == "config.json" || name == "train_config.json") {
      has_config = true;
    }
    verify << "  - " << name << " (" << std::filesystem::file_size(entry.path())
           << " bytes, sha256 " << sha256_file(entry.path()).substr(0, 16) << ")\n";
    ctx.manifest().add_artifact("S12", "checkpoint", entry.path(), "checkpoint file");
  }
  verify << "\nhas_weights=" << (has_weights ? "true" : "false")
         << " has_config=" << (has_config ? "true" : "false") << "\n";
  {
    std::ofstream out(verify_log, std::ios::trunc);
    out << verify.str();
  }
  if (!has_weights || !has_config) {
    return StageResult::fail("verify", "checkpoint incomplete (weights=" + std::string(has_weights ? "yes" : "no") +
                                           ", config=" + std::string(has_config ? "yes" : "no") + ")");
  }

  // S13-a：TorchScript 导出（受控 Python 调用点 c）
  if (!std::filesystem::exists(export_script)) {
    return StageResult::fail("verify", "missing export script: " + export_script.string());
  }
  ensure_dir(torchscript_dir);
  const ProcResult export_result = ctx.run_stack(
      {python.string(), export_script.string(), "--checkpoint", checkpoint.string(), "--out-dir",
       torchscript_dir.string(), "--repo-id", ctx.config().dataset.repo_id, "--root",
       (ctx.resolve(ctx.config().paths.data_dir) / ctx.config().dataset.repo_id).string(),
       "--samples-dir", samples_dir.string(), "--device", ctx.options().device},
      "13_torchscript_export.txt", 3600);
  verify << "\n===== export =====" << "\n" << export_result.output
         << "\n[exit=" << export_result.exit_code << "]\n";
  {
    std::ofstream out(verify_log, std::ios::trunc);
    out << verify.str();
  }
  const std::filesystem::path torchscript = torchscript_dir / "act_policy.pt";
  if (export_result.exit_code != 0 || !std::filesystem::exists(torchscript)) {
    ctx.manifest().add_artifact("S13", "log", verify_log, "checkpoint/export log");
    ctx.manifest().save();
    capture_terminal_screenshot(ctx, "15_checkpoint", "s13_export_fail", "tail -30 " + verify_log.string(), 18);
    return StageResult::blocked("verify",
                                "TorchScript export failed; C++ inference gate cannot be satisfied "
                                "(see " + verify_log.string() + ")");
  }
  ctx.manifest().add_artifact("S13", "torchscript", torchscript, "exported TorchScript policy");

  // S13-b：新进程重载（硬门禁）——独立可执行文件 infer_act
  const std::filesystem::path infer_bin = ctx.experiment_root() / "build" / "gpu" / "infer_act";
  if (!std::filesystem::exists(infer_bin)) {
    return StageResult::fail("verify", "infer_act binary missing: " + infer_bin.string());
  }
  const ProcResult reload = run_process(
      {infer_bin.string(), "--torchscript", torchscript.string(), "--samples", samples_dir.string(),
       "--device", ctx.options().device, "--out", (ctx.experiment_root() / "artifacts" / "reload_output.csv").string()},
      ProcOptions{.log_path = reload_log, .echo = true, .timeout_seconds = 600});
  ctx.manifest().add_artifact("S13", "log", reload_log, "new-process reload evidence");
  ctx.manifest().save();
  const ScreenshotResult checkpoint_shot = capture_terminal_screenshot(
      ctx, "15_checkpoint", "s13_checkpoint", "cat " + verify_log.string() + " | tail -30", 18);
  const ScreenshotResult reload_shot = capture_terminal_screenshot(
      ctx, "16_model_reload", "s13_reload", "cat " + reload_log.string() + " | tail -25", 18);
  if (reload.exit_code != 0) {
    return StageResult::fail("verify", "new-process C++ reload failed (see " + reload_log.string() + ")");
  }

  StageResult result = StageResult::pass(
      "verify", "checkpoint=" + checkpoint.string() + "; torchscript=" + torchscript.string() +
                    "; new-process reload OK");
  result.artifacts = {verify_log.string(), torchscript.string(), reload_log.string()};
  result.evidence.push_back(checkpoint_shot.captured ? checkpoint_shot.path.string()
                                                     : "SCREENSHOT_UNAVAILABLE: " + checkpoint_shot.note);
  result.evidence.push_back(reload_shot.captured ? reload_shot.path.string()
                                                 : "SCREENSHOT_UNAVAILABLE: " + reload_shot.note);
  result.duration_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
  return result;
}

}  // namespace actlab

#include "actlab/stages/stage_evidence.hpp"

#include "actlab/path_utils.hpp"
#include "actlab/proc.hpp"
#include "actlab/sha256.hpp"

#include <atomic>
#include <chrono>
#include <fstream>
#include <sstream>
#include <thread>
#include <vector>

namespace actlab {
namespace {

// xwininfo -root -tree 输出中按窗口标题定位窗口 id
std::string find_window_id(const std::string& title) {
  const std::string xwininfo = find_executable("xwininfo");
  if (xwininfo.empty()) {
    return "";
  }
  ProcOptions quiet;
  quiet.echo = false;  // 窗口枚举细节不打印到控制台/日志
  const ProcResult result = run_process({xwininfo, "-root", "-tree"}, quiet);
  if (result.exit_code != 0) {
    return "";
  }
  std::istringstream stream(result.output);
  std::string line;
  while (std::getline(stream, line)) {
    if (line.find("\"" + title + "\"") == std::string::npos) {
      continue;
    }
    if (line.find("xterm") == std::string::npos && line.find("XTerm") == std::string::npos) {
      continue;
    }
    std::istringstream fields(line);
    std::string id;
    fields >> id;
    if (!id.empty() && id.rfind("0x", 0) == 0) {
      return id;
    }
  }
  return "";
}

// 截图所需的窗口保活脚本：执行命令后停留，便于抓取
std::filesystem::path write_hold_script(const RunContext& ctx, const std::string& command,
                                        int hold_seconds) {
  // 文件名必须唯一：同一秒内多次截图若共用文件名，后写入会截断正在运行的 bash 脚本，
  // 导致 xterm 立即退出（实测为僵尸进程）而使抓窗失败。
  static std::atomic<unsigned> counter{0};
  const std::filesystem::path script = ctx.log_file("screenshot_hold_" + timestamp_compact() + "_" +
                                                   std::to_string(counter.fetch_add(1)) + ".sh");
  std::ofstream out(script, std::ios::trunc);
  if (!out) {
    throw std::runtime_error("cannot write screenshot hold script: " + script.string());
  }
  // 回显行必须转义单引号：命令里带 ' 时若不转义，会破坏 echo 的引用并让 bash 语法错误退出，
  // 进而使 xterm 立即关闭（实测表现为抓不到窗口）。
  std::string escaped;
  for (const char ch : command) {
    if (ch == '\'') {
      escaped += "'\\''";
    } else {
      escaped.push_back(ch);
    }
  }
  out << "#!/usr/bin/env bash\n";
  out << "date -Is\n";
  out << "echo '$ " << escaped << "'\n";
  out << command << "\n";
  out << "echo\necho '[captured for evidence; window closes in " << hold_seconds << "s]'\n";
  out << "sleep " << hold_seconds << "\n";
  out.close();
  return script;
}

}  // namespace

std::string detect_screenshot_tool() {
  // 顺序与知识库一致：gnome-screenshot -> scrot -> spectacle -> import
  for (const char* candidate : {"gnome-screenshot", "scrot", "spectacle"}) {
    const std::string found = find_executable(candidate);
    if (!found.empty()) {
      return found;
    }
  }
  const std::string import_bin = find_executable("import");
  if (!import_bin.empty()) {
    return import_bin;
  }
  return "";
}

ScreenshotResult capture_terminal_screenshot(RunContext& ctx, const std::string& evidence_name,
                                             const std::string& title, const std::string& shell_command,
                                             int hold_seconds) {
  ScreenshotResult result;
  const std::string xterm = find_executable("xterm");
  const std::string import_bin = detect_screenshot_tool();
  if (xterm.empty() || import_bin.empty()) {
    result.note = "xterm or screenshot tool unavailable (xterm='" + xterm + "', tool='" + import_bin + "')";
    return result;
  }

  const std::filesystem::path script = write_hold_script(ctx, shell_command, hold_seconds);
  const std::string window_title = "actlab_" + evidence_name;
  // 实测：偶发窗口未及时注册，因此做一次重试（仍失败则如实标记，不伪造）
  for (int spawn_attempt = 0; spawn_attempt < 2 && !result.captured; ++spawn_attempt) {
    const int pid = spawn_detached({xterm, "-title", window_title, "-geometry", "110x32", "-e", "bash",
                                    script.string()});
    if (pid <= 0) {
      result.note = "failed to spawn xterm";
      return result;
    }

    std::string window_id;
    const int attempts = spawn_attempt == 0 ? 20 : 30;
    for (int attempt = 0; attempt < attempts && window_id.empty(); ++attempt) {
      std::this_thread::sleep_for(std::chrono::milliseconds(500));
      window_id = find_window_id(window_title);
    }

    if (!window_id.empty()) {
      const std::filesystem::path out_path = ctx.evidence_file(evidence_name + ".png");
      ProcOptions quiet;
      quiet.echo = false;
      const ProcResult shot =
          run_process({import_bin, "-window", window_id, out_path.string()}, quiet);
      if (shot.exit_code == 0 && std::filesystem::exists(out_path) &&
          std::filesystem::file_size(out_path) > 0) {
        result.captured = true;
        result.path = out_path;
        result.tool = import_bin;
        result.note = "real xterm window capture (window id " + window_id +
                      ", attempt " + std::to_string(spawn_attempt + 1) + ")";
      } else {
        result.note = "import failed with exit code " + std::to_string(shot.exit_code);
      }
    } else {
      result.note = "xterm window '" + window_title + "' not found via xwininfo (attempt " +
                    std::to_string(spawn_attempt + 1) + ")";
    }
    terminate_process(pid, 3000);
  }
  std::error_code ec;
  std::filesystem::remove(script, ec);

  std::ofstream note_out(ctx.evidence_file(evidence_name + ".capture.json"), std::ios::trunc);
  note_out << "{\"captured\": " << (result.captured ? "true" : "false") << ", \"tool\": \""
           << result.tool << "\", \"note\": \"" << result.note << "\", \"timestamp\": \""
           << timestamp_iso8601() << "\"}\n";
  return result;
}

StageResult run_stage_evidence(RunContext& ctx) {
  const auto start = std::chrono::steady_clock::now();
  const std::filesystem::path evidence_dir = ctx.experiment_root() / "evidence";
  const std::filesystem::path logs_dir = ctx.experiment_root() / "logs";
  ensure_dir(evidence_dir);
  ensure_dir(logs_dir);

  struct Row {
    std::string id;
    std::string stage;
    std::string png;
    std::string log;
    std::string note;
  };
  // 编号与工单 §4 的证据编号一致；png 存在性在运行期实测判定
  const std::vector<Row> rows = {
      {"00", "hardware_check", "00_hardware_check.png", "00_hardware_check.txt", "OS/GPU/CUDA/Python/conda/git/ffmpeg 探测"},
      {"01", "python_environment", "01_python_environment.png", "01_python_environment.txt", "conda env 与 python 版本"},
      {"02", "cuda_pytorch", "02_cuda_pytorch.png", "02_cuda_pytorch.txt", "torch CUDA / sm_120 可用性"},
      {"03", "repo_clone", "03_repo_clone.png", "03_repo_probe.txt", "LeRobot 源码 commit/branch"},
      {"04", "lerobot_install", "04_lerobot_install.png", "04_lerobot_install.txt", "editable 安装与 import 验证"},
      {"05", "lerobot_version", "05_lerobot_version.png", "05_lerobot_version.txt", "版本锁定（pip freeze / conda lock）"},
      {"06", "cli_check", "06_cli_check.png", "06_cli_check.txt", "lerobot-train --help 关键参数"},
      {"07", "dataset_download", "07_dataset_download.png", "07_dataset_download.txt", "数据集落盘与规模"},
      {"08", "dataset_structure", "08_dataset_structure.png", "08_dataset_schema.txt", "schema 审计结果"},
      {"09", "dataset_visualization", "09_dataset_visualization.png", "09_dataset_viz.txt", "可视化产物（.rrd / 帧拼图）"},
      {"10", "act_config", "10_act_config.png", "10_act_config.txt", "冻结的超参与配置哈希"},
      {"11", "act_model_init", "11_act_model_init.png", "11_act_model_init.txt", "最小初始化/前向/反向后端验证"},
      {"12", "training_start", "12_training_start.png", "12_training_smoke.txt", "smoke 训练启动（100 步）"},
      {"13", "training_progress", "13_training_progress.png", "13_training_release.txt",
       "release 训练（3000 步）"},
      {"14", "training_loss", "14_training_loss.png", "14_training_loss.txt", "loss 曲线（真实日志）"},
      {"15", "checkpoint", "15_checkpoint.png", "15_checkpoint.txt", "checkpoint 目录与哈希"},
      {"16", "model_reload", "16_model_reload.png", "16_model_reload.txt", "新进程 C++ 重载"},
      {"17", "inference", "17_inference.png", "17_inference.txt", "C++ 推理与延时统计"},
      {"18", "evaluation", "18_evaluation.png", "18_evaluation.txt", "Offline Evaluation 结果"},
      {"19", "final_environment", "19_final_environment.png", "19_final_environment.txt", "最终环境与产物总览"},
      {"20", "so101_mujoco_viz", "20_so101_mujoco_window.png", "20_so101_mujoco_window.log",
       "SO-101 MuJoCo 桌面仿真窗口（真实物体：抓取前静置 → 夹住随爪移动 → 松开落回平面）"},
  };

  // SO-101 MuJoCo 可视化的可留存产物（图/视频/抓取参数），存在即登记进 manifest
  const std::filesystem::path viz_dir = ctx.experiment_root() / "results" / "so101_mujoco";
  for (const auto& [file, kind, note] :
       std::vector<std::tuple<std::string, std::string, std::string>>{
           {"so101_render_montage_ep000.png", "figure", "SO-101 3D 回放关键帧拼图（含物体抓取）"},
           {"so101_ep000_replay.mp4", "video", "SO-101 3D 回放视频（真实轨迹驱动）"},
           {"so101_compare_montage_ep000.png", "figure", "真实相机帧 vs 3D 渲染 并排核对"},
           {"grasp_info.json", "metrics", "由真实轨迹 FK 推断的抓取点/水平面/阶段参数"},
           {"trajectory_compare.json", "metrics", "专家 vs ACT 预测 的夹爪轨迹偏差（EE MAE 等）"},
           {"physics_summary.md", "report", "物理抓取稳定性检查汇总表（真实接触/摩擦）"},
           {"physics_grasp_scale1.05_montage.png", "figure", "物理抓取过程关键帧（默认尺寸）"},
       }) {
    const std::filesystem::path path = viz_dir / file;
    if (std::filesystem::exists(path)) {
      ctx.manifest().add_artifact("so101_viz", kind, path, note);
    }
  }

  std::vector<std::string> missing_screenshots;
  std::vector<std::string> missing_logs;
  std::ostringstream index;
  index << "# 实验证据索引\n\n";
  index << "| 编号 | 阶段 | 截图 | 日志 | 说明 | 时间戳 | 截图哈希 |\n";
  index << "| --- | --- | --- | --- | --- | --- | --- |\n";
  for (const auto& row : rows) {
    const std::filesystem::path png = evidence_dir / row.png;
    const std::filesystem::path log = logs_dir / row.log;
    const bool has_png = std::filesystem::exists(png);
    const bool has_log = std::filesystem::exists(log);
    if (!has_png) missing_screenshots.push_back(row.png);
    if (!has_log) missing_logs.push_back(row.log);
    std::string png_cell = has_png ? row.png : "SCREENSHOT_UNAVAILABLE";
    std::string hash_cell = "-";
    if (has_png) {
      hash_cell = sha256_file(png).substr(0, 16);
      ctx.manifest().add_artifact("evidence", "screenshot", png, row.note);
    }
    std::string log_cell = has_log ? row.log : "LOG_MISSING";
    index << "| " << row.id << " | " << row.stage << " | " << png_cell << " | " << log_cell << " | "
          << row.note << " | " << (has_png ? timestamp_iso8601() : "-") << " | " << hash_cell << " |\n";
  }
  index << "\n- 截图工具: `" << detect_screenshot_tool() << "`\n";
  index << "- 截图方式: 真实 xterm 窗口运行真实命令后按窗口 id 抓取（"
           "本机 Wayland 下整屏抓取不可用，已在 evidence/README.md 说明）\n";
  index << "- 缺失截图数: " << missing_screenshots.size() << "\n";
  index << "- 缺失日志数: " << missing_logs.size() << "\n";
  if (!missing_screenshots.empty()) {
    index << "\n## SCREENSHOT_UNAVAILABLE\n\n";
    for (const auto& item : missing_screenshots) {
      index << "- " << item << "\n";
    }
  }
  if (!missing_logs.empty()) {
    index << "\n## LOG_MISSING（必须补齐，否则证据不完整）\n\n";
    for (const auto& item : missing_logs) {
      index << "- " << item << "\n";
    }
  }

  {
    std::ofstream out(evidence_dir / "index.md", std::ios::trunc);
    out << index.str();
  }
  {
    std::ofstream out(evidence_dir / "README.md", std::ios::trunc);
    out << "# Evidence\n\n"
        << "- 截图工具探测顺序: gnome-screenshot -> scrot -> spectacle -> import\n"
        << "- 本机实测: gnome-screenshot/scrot/spectacle 缺失；ImageMagick `import` 可用\n"
        << "- Wayland 限制: `import -window root` 与 `ffmpeg x11grab` 对本机 Xwayland root 均不可用\n"
        << "- 采用方案: 启动真实 xterm 运行真实命令，再 `import -window <window-id>` 抓取\n"
        << "- 不可抓取时: 显式写 SCREENSHOT_UNAVAILABLE，绝不使用合成/伪造图片\n";
  }

  ctx.manifest().add_artifact("evidence", "index", evidence_dir / "index.md", "evidence index");
  ctx.manifest().save();

  StageResult result = StageResult::pass("evidence", "index.md generated; screenshots present=" +
                                                        std::to_string(rows.size() - missing_screenshots.size()) +
                                                        ", missing=" + std::to_string(missing_screenshots.size()) +
                                                        ", logs missing=" + std::to_string(missing_logs.size()));
  result.artifacts.push_back((evidence_dir / "index.md").string());
  result.duration_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
  return result;
}

}  // namespace actlab

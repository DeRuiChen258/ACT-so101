#include "actlab/cli.hpp"

#include "actlab/config_loader.hpp"
#include "actlab/path_utils.hpp"
#include "actlab/stages/stage_dataset.hpp"
#include "actlab/stages/stage_env.hpp"
#include "actlab/stages/stage_eval.hpp"
#include "actlab/stages/stage_evidence.hpp"
#include "actlab/stages/stage_install.hpp"
#include "actlab/stages/stage_report.hpp"
#include "actlab/stages/stage_repo.hpp"
#include "actlab/stages/stage_train.hpp"
#include "actlab/stages/stage_verify.hpp"

#include <unistd.h>

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace actlab {
namespace {

constexpr const char* kStages[] = {"env",    "repo",  "install", "dataset", "train", "verify",
                                   "infer",  "eval",  "evidence", "report"};
constexpr int kStageCount = 10;

bool is_stage(const std::string& name) {
  for (const char* stage : kStages) {
    if (name == stage) {
      return true;
    }
  }
  return name == "all";
}

// 从可执行文件位置推导实验根：<root>/build/{gpu,local}/actlab
std::filesystem::path infer_experiment_root() {
  char buffer[4096] = {};
  const ssize_t length = ::readlink("/proc/self/exe", buffer, sizeof(buffer) - 1);
  if (length > 0) {
    std::filesystem::path exe(buffer);
    const std::filesystem::path build_dir = exe.parent_path().parent_path();
    if (build_dir.filename() == "build") {
      return build_dir.parent_path();
    }
  }
  return std::filesystem::current_path();
}

std::string next_step_for(const std::string& stage) {
  if (stage == "env") return "repo（源码获取与结构分析）";
  if (stage == "repo") return "install（LeRobot 安装与 CLI 校验）";
  if (stage == "install") return "dataset（数据集获取、schema 审计、可视化）";
  if (stage == "dataset") return "train（配置冻结 → 最小初始化 → smoke → release）";
  if (stage == "train") return "verify（checkpoint 校验 + TorchScript 导出 + 新进程重载）";
  if (stage == "verify") return "infer（C++ 主路径推理）";
  if (stage == "infer") return "eval（Offline Evaluation）";
  if (stage == "eval") return "evidence（证据索引与配对校验）";
  if (stage == "evidence") return "report（最终报告与总结块）";
  return "无";
}

}  // namespace

void print_usage() {
  std::cout << "actlab — LeRobot/ACT experiment driver (C++17)\n\n"
            << "usage: actlab <stage> [options]\n\n"
            << "stages:\n"
            << "  env        硬件/CUDA/Python/conda/git/ffmpeg 探测与门禁（S0）\n"
            << "  repo       源码获取与结构分析（S1–S2）\n"
            << "  install    LeRobot 安装与 CLI 校验（S3）\n"
            << "  dataset    数据集获取、schema 审计、可视化（S4–S6）\n"
            << "  train      配置冻结/最小初始化/smoke/正式训练（S7–S11）\n"
            << "  verify     checkpoint 校验 + TorchScript 导出 + 新进程重载（S12–S13）\n"
            << "  infer      C++ 主路径推理与延时统计（S14）\n"
            << "  eval       Offline Evaluation（S15）\n"
            << "  evidence   证据索引与配对校验（S16）\n"
            << "  report     最终 README 报告（S16）\n"
            << "  all        按顺序驱动全流程（遇到非 PASS 即停止）\n\n"
            << "options:\n"
            << "  --config=DIR      配置目录（默认 <root>/configs）\n"
            << "  --device=DEVICE   cuda|cpu（默认 cuda）\n"
            << "  --steps=N         覆盖训练步数（smoke 用 N<=runtime.smoke_steps）\n"
            << "  --dry-run         只校验配置，不产生副作用\n"
            << "  --resume          断点续训\n"
            << "  --overwrite       允许覆盖既有产物\n"
            << "  --env-root=DIR    环境层根目录（默认 /home/violet/Workspace/IDE/Physical_AI）\n"
            << "  --help            显示本帮助\n";
}

CliOptions parse_cli(int argc, char** argv) {
  CliOptions options;
  options.experiment_root = infer_experiment_root();
  options.config_dir = options.experiment_root / "configs";
  const char* env_root = std::getenv("ACTLAB_ENV_ROOT");
  options.env_root = env_root != nullptr ? std::filesystem::path(env_root)
                                         : std::filesystem::path("/home/violet/Workspace/IDE/Physical_AI");
  bool stage_seen = false;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      options.show_help = true;
    } else if (arg == "--dry-run") {
      options.run.dry_run = true;
    } else if (arg == "--resume") {
      options.run.resume = true;
    } else if (arg == "--smoke") {
      options.run.smoke = true;
    } else if (arg == "--overwrite") {
      options.run.overwrite = true;
    } else if (arg == "--schema" || arg == "--viz") {
      options.run.dataset_flag = arg.substr(2);
    } else if (arg.rfind("--config=", 0) == 0) {
      options.config_dir = arg.substr(9);
    } else if (arg.rfind("--device=", 0) == 0) {
      options.run.device = arg.substr(9);
    } else if (arg.rfind("--steps=", 0) == 0) {
      options.run.steps_override = std::stoll(arg.substr(8));
    } else if (arg.rfind("--env-root=", 0) == 0) {
      options.env_root = arg.substr(11);
    } else if (!arg.empty() && arg[0] == '-') {
      throw std::runtime_error("unknown option: " + arg);
    } else if (!stage_seen) {
      if (!is_stage(arg)) {
        throw std::runtime_error("unknown stage: " + arg);
      }
      options.run.stage = arg;
      stage_seen = true;
    } else {
      throw std::runtime_error("unexpected positional argument: " + arg);
    }
  }
  return options;
}

void report_stage_result(const StageResult& result, int stage_index, int stage_total,
                         const std::string& executed, const std::string& next_step) {
  std::cout << "\n[阶段 " << stage_index << "/" << stage_total << "] " << result.stage << "\n";
  std::cout << "状态：" << to_string(result.status) << "\n";
  std::cout << "实际执行：" << executed << "\n";
  std::cout << "关键结果：" << result.detail << "\n";
  std::cout << "产生文件：";
  for (size_t i = 0; i < result.artifacts.size(); ++i) {
    std::cout << (i > 0 ? ", " : "") << result.artifacts[i];
  }
  if (result.artifacts.empty()) {
    std::cout << "(none)";
  }
  std::cout << "\n截图：";
  if (result.evidence.empty()) {
    std::cout << "SCREENSHOT_UNAVAILABLE";
  } else {
    for (size_t i = 0; i < result.evidence.size(); ++i) {
      std::cout << (i > 0 ? ", " : "") << result.evidence[i];
    }
  }
  std::cout << "\n下一步：" << next_step << "\n" << std::endl;
}

namespace {

StageResult dispatch_stage(const std::string& stage, RunContext& ctx) {
  if (stage == "env") return run_stage_env(ctx);
  if (stage == "repo") return run_stage_repo(ctx);
  if (stage == "install") return run_stage_install(ctx);
  if (stage == "dataset") return run_stage_dataset(ctx);
  if (stage == "train") return run_stage_train(ctx);
  if (stage == "verify") return run_stage_verify(ctx);
#ifdef ACTLAB_WITH_TORCH
  if (stage == "infer") return run_stage_infer(ctx);
  if (stage == "eval") return run_stage_eval(ctx);
#else
  if (stage == "infer" || stage == "eval") {
    return StageResult::blocked(stage,
                                "stage requires LibTorch build (use build/gpu/actlab); "
                                "local build only covers pure logic");
  }
#endif
  if (stage == "evidence") return run_stage_evidence(ctx);
  if (stage == "report") return run_stage_report(ctx);
  throw std::runtime_error("unhandled stage: " + stage);
}

int to_exit_code(StageStatus status) {
  switch (status) {
    case StageStatus::Pass: return kExitPass;
    case StageStatus::Fail: return kExitFail;
    case StageStatus::Blocked: return kExitBlocked;
  }
  return kExitFail;
}

}  // namespace

int run_cli(int argc, char** argv) {
  CliOptions options;
  try {
    options = parse_cli(argc, argv);
  } catch (const std::exception& error) {
    std::cerr << "usage error: " << error.what() << "\n\n";
    print_usage();
    return kExitUsage;
  }
  if (options.show_help) {
    print_usage();
    return kExitPass;
  }

  ExperimentConfig config;
  try {
    config = load_experiment_config(options.config_dir);
  } catch (const std::exception& error) {
    std::cerr << "[FATAL] config load/validation failed: " << error.what() << std::endl;
    return kExitFail;
  }

  std::vector<std::string> sequence;
  if (options.run.stage == "all") {
    sequence = {"env", "repo", "install", "dataset", "train", "verify", "infer", "eval", "evidence",
                "report"};
  } else {
    sequence = {options.run.stage};
  }

  for (size_t index = 0; index < sequence.size(); ++index) {
    const std::string stage = sequence[index];
    RunOptions run_options = options.run;
    run_options.stage = stage;
    if (run_options.smoke && run_options.steps_override <= 0) {
      run_options.steps_override = config.runtime.smoke_steps;
    }
    RunContext ctx(run_options, config, options.experiment_root, options.env_root);
    StageResult result;
    const auto start = std::chrono::steady_clock::now();
    try {
      result = dispatch_stage(stage, ctx);
    } catch (const std::exception& error) {
      result = StageResult::fail(stage, std::string("exception: ") + error.what());
    }
    result.duration_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    ctx.logger().log(result.status == StageStatus::Pass ? LogLevel::Info : LogLevel::Error, "stage",
                     stage + " -> " + to_string(result.status) + ": " + result.detail,
                     result.duration_ms);

    std::string executed = "actlab " + stage;
    if (options.run.steps_override > 0) {
      executed += " --steps=" + std::to_string(options.run.steps_override);
    }
    if (options.run.dry_run) {
      executed += " --dry-run";
    }
    report_stage_result(result, static_cast<int>(index) + 1, static_cast<int>(sequence.size()), executed,
                        result.status == StageStatus::Pass ? next_step_for(stage) : "修正失败原因后重跑本阶段");

    if (result.status != StageStatus::Pass) {
      return to_exit_code(result.status);
    }
  }
  return kExitPass;
}

}  // namespace actlab

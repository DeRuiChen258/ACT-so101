// RunContext：把"配置 + 路径 + 日志 + 清单 + 受控子进程"装配成一个可传递对象，
// 使各 stage 只依赖抽象而非全局状态（DIP）。
#pragma once

#include "actlab/config_loader.hpp"
#include "actlab/logging.hpp"
#include "actlab/manifest.hpp"
#include "actlab/proc.hpp"

#include <filesystem>
#include <memory>
#include <string>

namespace actlab {

struct RunOptions {
  std::string stage = "env";       // env/repo/install/dataset/train/verify/infer/eval/evidence/report/all
  std::string device = "cuda";
  bool dry_run = false;            // 只校验配置，不产生副作用
  bool overwrite = false;          // 允许覆盖既有产物（默认拒绝）
  bool resume = false;             // 断点续训
  bool smoke = false;              // 使用 runtime.smoke_steps（脚本不写魔数）
  long long steps_override = -1;   // >0 时覆盖配置里的 steps（smoke 用）
  std::string dataset_flag;        // dataset 子动作：schema / viz
};

// 阶段结果：PASS / FAIL / BLOCKED（门禁语义，禁止用"看起来成功"代替）
enum class StageStatus { Pass, Fail, Blocked };

const char* to_string(StageStatus status);

struct StageResult {
  std::string stage;
  StageStatus status = StageStatus::Fail;
  std::string detail;
  std::vector<std::string> artifacts;
  std::vector<std::string> evidence;   // 截图路径或 SCREENSHOT_UNAVAILABLE
  double duration_ms = 0.0;

  static StageResult pass(std::string stage, std::string detail);
  static StageResult fail(std::string stage, std::string detail);
  static StageResult blocked(std::string stage, std::string detail);
};

class RunContext {
 public:
  RunContext(RunOptions options, ExperimentConfig config, fs::path experiment_root, fs::path env_root);

  const RunOptions& options() const { return options_; }
  const ExperimentConfig& config() const { return config_; }
  const fs::path& experiment_root() const { return experiment_root_; }
  const fs::path& env_root() const { return env_root_; }
  Logger& logger() { return *logger_; }
  Manifest& manifest() { return *manifest_; }

  // 相对实验根解析路径（配置里允许写相对路径）
  fs::path resolve(const std::string& maybe_relative) const;

  // 以实验根为基准的日志路径
  fs::path log_file(const std::string& name) const;
  fs::path evidence_file(const std::string& name) const;
  fs::path results_file(const std::string& name) const;

  // 统一执行训练栈命令：注册 PATH、LD_LIBRARY_PATH、HF_LEROBOT_HOME 与可控日志
  ProcResult run_stack(const std::vector<std::string>& argv, const std::string& log_name,
                       int timeout_seconds = 0);

  // 训练侧 python 解释器（conda env lerobot_act）
  std::string python_bin() const;

 private:
  RunOptions options_;
  ExperimentConfig config_;
  fs::path experiment_root_;
  fs::path env_root_;
  std::unique_ptr<Logger> logger_;
  std::unique_ptr<Manifest> manifest_;
};

}  // namespace actlab

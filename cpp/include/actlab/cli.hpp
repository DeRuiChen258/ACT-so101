// CLI 主入口：解析 --stage/--config/--device/--resume 等参数，分派子命令，统一退出码
#pragma once

#include "actlab/context.hpp"

#include <filesystem>
#include <string>

namespace actlab {

// 退出码约定：0=PASS, 1=FAIL, 2=BLOCKED, 64=用法错误
constexpr int kExitPass = 0;
constexpr int kExitFail = 1;
constexpr int kExitBlocked = 2;
constexpr int kExitUsage = 64;

struct CliOptions {
  RunOptions run;
  std::filesystem::path experiment_root;
  std::filesystem::path env_root;
  std::filesystem::path config_dir;
  bool show_help = false;
};

CliOptions parse_cli(int argc, char** argv);
void print_usage();

// 打印阶段实时汇报块（Prompt §5 要求的固定格式）
void report_stage_result(const StageResult& result, int stage_index, int stage_total,
                         const std::string& executed, const std::string& next_step);

int run_cli(int argc, char** argv);

}  // namespace actlab

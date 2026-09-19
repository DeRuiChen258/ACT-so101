// 阶段 S12–S13：checkpoint 完整性校验、TorchScript 导出编排、新进程 C++ 重载
#pragma once

#include "actlab/context.hpp"

namespace actlab {

// 最近一次训练输出目录中的最新 checkpoint
std::filesystem::path find_latest_checkpoint(const RunContext& ctx, const std::string& job_name);

StageResult run_stage_verify(RunContext& ctx);

}  // namespace actlab

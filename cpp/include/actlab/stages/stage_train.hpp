// 阶段 S7–S11：算法笔记/配置冻结、最小初始化校验、smoke 训练与正式训练编排
#pragma once

#include "actlab/context.hpp"

namespace actlab {

// S7–S8：生成算法笔记 + 配置冻结校验（--dry-run 路径）
StageResult run_stage_train_config(RunContext& ctx);

// S9：最小初始化（Dataset→Policy→Forward→Loss→Backward），由受控脚本执行并解析结果
StageResult run_stage_train_smoke_init(RunContext& ctx);

// S10–S11：smoke / release 训练
StageResult run_stage_train(RunContext& ctx);

}  // namespace actlab

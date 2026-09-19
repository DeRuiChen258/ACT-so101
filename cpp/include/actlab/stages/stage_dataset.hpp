// 阶段 S4–S6：数据集下载、schema 校验、可视化编排
#pragma once

#include "actlab/context.hpp"

namespace actlab {

StageResult run_stage_dataset(RunContext& ctx);

}  // namespace actlab

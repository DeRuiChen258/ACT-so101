// 阶段 S1–S2：源码获取（GitCode 镜像优先）与结构分析
#pragma once

#include "actlab/context.hpp"

namespace actlab {

StageResult run_stage_repo(RunContext& ctx);

}  // namespace actlab

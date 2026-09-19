// 阶段 S14–S15：C++ 推理（主路径）与 Offline Evaluation
#pragma once

#include "actlab/context.hpp"

namespace actlab {

StageResult run_stage_infer(RunContext& ctx);
StageResult run_stage_eval(RunContext& ctx);

}  // namespace actlab

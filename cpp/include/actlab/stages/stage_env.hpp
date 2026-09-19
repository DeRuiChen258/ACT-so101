// 阶段 S0：硬件、CUDA、Python、Conda、Git、FFmpeg、LibTorch 全项探测与门禁
#pragma once

#include "actlab/context.hpp"

namespace actlab {

StageResult run_stage_env(RunContext& ctx);

}  // namespace actlab

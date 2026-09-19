// 阶段 S3：编排 LeRobot 安装与 CLI 探测，产出安装证据与版本锁定
#pragma once

#include "actlab/context.hpp"

namespace actlab {

StageResult run_stage_install(RunContext& ctx);

}  // namespace actlab

// 证据管线：真实截图调用、日志配对校验、evidence/index.md 生成
#pragma once

#include "actlab/context.hpp"

#include <filesystem>
#include <string>

namespace actlab {

struct ScreenshotResult {
  bool captured = false;
  std::filesystem::path path;
  std::string tool;
  std::string note;
};

// 在真实 xterm 窗口中执行 shell_command，随后按 README 记录的方式抓取该窗口。
// 抓取失败时返回 captured=false，调用方必须写 SCREENSHOT_UNAVAILABLE（禁止伪造）。
ScreenshotResult capture_terminal_screenshot(RunContext& ctx, const std::string& evidence_name,
                                             const std::string& title, const std::string& shell_command,
                                             int hold_seconds = 20);

// 检测可用截图工具（本机实测：gnome-screenshot/scrot/spectacle 缺失，ImageMagick import 可用）
std::string detect_screenshot_tool();

// 校验"每阶段图 + 日志配对"，生成 evidence/index.md；缺失项写入 index 而非静默通过
StageResult run_stage_evidence(RunContext& ctx);

}  // namespace actlab

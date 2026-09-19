// 路径规范化 / 目录创建 / 覆盖保护 / 时间戳命名
#pragma once

#include <filesystem>
#include <string>

namespace actlab {

namespace fs = std::filesystem;

// 展开 "~" 前缀；不做 shell 展开
fs::path expand_user(const std::string& raw);

// 相对路径按 base 解析为绝对路径
fs::path resolve_path(const fs::path& raw, const fs::path& base);

// 递归创建目录；已存在则复用
void ensure_dir(const fs::path& dir);

// 覆盖保护：目标已存在且 !overwrite 时抛 std::runtime_error
void ensure_writable_target(const fs::path& target, bool overwrite);

// 本地时间戳，形如 20260919_185500
std::string timestamp_compact();

// ISO-8601 秒级时间戳，形如 2026-09-19T18:55:00+08:00
std::string timestamp_iso8601();

// 相对 base 的路径（用于日志可读性）；不在 base 下则返回原路径
std::string relative_to(const fs::path& path, const fs::path& base);

}  // namespace actlab

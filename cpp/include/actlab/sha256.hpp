// SHA-256（自实现，供 manifest 与证据审计使用；不引入外部依赖）
#pragma once

#include <cstdint>
#include <filesystem>
#include <string>

namespace actlab {

// 返回 64 位小写十六进制摘要
std::string sha256_hex(const std::string& data);

// 流式读取文件计算摘要；文件不可读时抛 std::runtime_error
std::string sha256_file(const std::filesystem::path& path);

}  // namespace actlab

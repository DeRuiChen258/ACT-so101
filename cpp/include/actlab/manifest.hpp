// experiment_manifest.json：版本/硬件/配置哈希/产物 SHA256 的唯一机器可读事实源
#pragma once

#include "actlab/json_min.hpp"

#include <filesystem>
#include <string>

namespace actlab {

class Manifest {
 public:
  explicit Manifest(std::filesystem::path path);

  // 写入/覆盖某个顶层字段（section 形如 "environment" / "dataset" / "training"）
  void set_field(const std::string& section, const std::string& key, const std::string& value);
  // 显式重载：避免字符串字面量被隐式转换为 bool（曾导致 evaluation.type 被写成 true）
  void set_field(const std::string& section, const std::string& key, const char* value) {
    set_field(section, key, std::string(value));
  }
  void set_field(const std::string& section, const std::string& key, double value);
  void set_field(const std::string& section, const std::string& key, long long value);
  void set_field(const std::string& section, const std::string& key, bool value);

  // 记录产物（计算 SHA256 与字节数）；path 不存在时抛错，绝不静默跳过
  void add_artifact(const std::string& stage, const std::string& kind, const std::filesystem::path& path,
                    const std::string& note = "");

  // 记录一条原始命令（可审计）
  void add_command(const std::string& stage, const std::string& command);

  void save();
  const std::filesystem::path& path() const { return path_; }
  std::size_t artifact_count() const { return artifact_count_; }

 private:
  std::filesystem::path path_;
  Json root_;
  std::size_t artifact_count_ = 0;
  bool dirty_ = false;

  void load_existing();
};

}  // namespace actlab

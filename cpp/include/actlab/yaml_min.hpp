// 极简 YAML 子集解析：仅支持本项目配置文件用到的语法
//   - 缩进表示嵌套映射；
//   - `key: value` 标量（string / number / bool / null）；
//   - 行内列表 `[a, b, c]`；
//   - `#` 行内注释；
//   - 块序列 `- item`（仅顶层标量序列）。
// 不支持的语法必须显式报错（禁止静默忽略）。
#pragma once

#include <map>
#include <string>
#include <vector>

namespace actlab {

class YamlNode {
 public:
  enum class Type { Null, Scalar, List, Map };

  Type type() const { return type_; }
  bool is_map() const { return type_ == Type::Map; }
  bool is_scalar() const { return type_ == Type::Scalar; }
  bool is_list() const { return type_ == Type::List; }

  bool has(const std::string& key) const;
  const YamlNode& at(const std::string& key) const;  // 缺失抛 std::out_of_range

  const std::string& scalar() const { return scalar_; }
  const std::vector<std::string>& list() const { return list_; }
  const std::vector<std::string>& keys() const { return keys_; }
  const std::map<std::string, YamlNode>& children() const { return children_; }

  // 类型化取值；类型不符或缺失即抛 std::runtime_error
  std::string as_string(const std::string& key) const;
  std::string as_string(const std::string& key, const std::string& fallback) const;
  long long as_int(const std::string& key) const;
  double as_double(const std::string& key) const;
  bool as_bool(const std::string& key) const;

  static YamlNode parse(const std::string& text);
  static YamlNode parse_file(const std::string& path);

 private:
  Type type_ = Type::Null;
  std::string scalar_;
  std::vector<std::string> list_;
  std::vector<std::string> keys_;
  std::map<std::string, YamlNode> children_;

  friend class YamlParser;
};

}  // namespace actlab

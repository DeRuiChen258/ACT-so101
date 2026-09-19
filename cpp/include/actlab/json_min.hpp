// 极简 JSON：值树 + 序列化 + 解析（仅覆盖 manifest / dataset meta 所需子集）
#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace actlab {

class Json;
using JsonPtr = std::shared_ptr<Json>;

class Json {
 public:
  enum class Type { Null, Bool, Number, String, Array, Object };

  Json() = default;
  static Json make_object();
  static Json make_array();
  static Json make_string(std::string value);
  static Json make_number(double value);
  static Json make_bool(bool value);

  Type type() const { return type_; }
  bool is_object() const { return type_ == Type::Object; }
  bool is_array() const { return type_ == Type::Array; }

  // 对象写入（保持插入顺序由 keys_ 记录）
  void set(const std::string& key, Json value);
  void set(const std::string& key, const std::string& value) { set(key, make_string(value)); }
  void set(const std::string& key, const char* value) { set(key, make_string(value)); }
  void set(const std::string& key, double value) { set(key, make_number(value)); }
  void set(const std::string& key, int value) { set(key, make_number(static_cast<double>(value))); }
  void set(const std::string& key, long long value) { set(key, make_number(static_cast<double>(value))); }
  void set(const std::string& key, bool value) { set(key, make_bool(value)); }

  // 数组追加
  void push(Json value);
  void push(const std::string& value) { push(make_string(value)); }
  void push(const char* value) { push(make_string(value)); }
  void push(double value) { push(make_number(value)); }
  void push(long long value) { push(make_number(static_cast<double>(value))); }
  void push(bool value) { push(make_bool(value)); }

  bool has(const std::string& key) const;
  const Json& at(const std::string& key) const;  // 缺失抛 std::out_of_range
  Json& at_mut(const std::string& key);          // 变异访问（用于追加数组元素）
  const std::vector<Json>& items() const { return array_; }
  const std::vector<std::string>& keys() const { return keys_; }

  const std::string& as_string() const { return string_; }
  double as_number() const { return number_; }
  bool as_bool() const { return bool_; }
  int as_int() const { return static_cast<int>(number_); }

  // 类型不符即抛 std::runtime_error（禁止静默取默认值）
  std::string string_or_throw(const std::string& key) const;
  double number_or_throw(const std::string& key) const;
  long long int_or_throw(const std::string& key) const;

  std::string dump(int indent = 2) const;

  // 解析失败抛 std::runtime_error，带偏移信息
  static Json parse(const std::string& text);
  static Json parse_file(const std::string& path);

 private:
  Type type_ = Type::Null;
  bool bool_ = false;
  double number_ = 0.0;
  std::string string_;
  std::vector<Json> array_;
  std::vector<std::string> keys_;
  std::map<std::string, Json> object_;

  void dump_to(std::string& out, int indent, int depth) const;
};

// 字符串转义（写入 JSON 文本时使用）
std::string json_escape(const std::string& raw);

}  // namespace actlab

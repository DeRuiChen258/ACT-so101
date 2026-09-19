#include "actlab/yaml_min.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace actlab {

bool YamlNode::has(const std::string& key) const {
  return type_ == Type::Map && children_.find(key) != children_.end();
}

const YamlNode& YamlNode::at(const std::string& key) const {
  if (type_ != Type::Map) {
    throw std::runtime_error("YAML: node is not a mapping, cannot read key '" + key + "'");
  }
  const auto it = children_.find(key);
  if (it == children_.end()) {
    throw std::out_of_range("YAML: missing key '" + key + "'");
  }
  return it->second;
}

std::string YamlNode::as_string(const std::string& key) const {
  const YamlNode& node = at(key);
  if (node.type() != Type::Scalar) {
    throw std::runtime_error("YAML: key '" + key + "' is not a scalar");
  }
  return node.scalar();
}

std::string YamlNode::as_string(const std::string& key, const std::string& fallback) const {
  return has(key) ? as_string(key) : fallback;
}

long long YamlNode::as_int(const std::string& key) const {
  const std::string raw = as_string(key);
  try {
    size_t consumed = 0;
    const long long value = std::stoll(raw, &consumed);
    if (consumed != raw.size()) {
      throw std::invalid_argument("trailing characters");
    }
    return value;
  } catch (const std::exception&) {
    throw std::runtime_error("YAML: key '" + key + "' value '" + raw + "' is not an integer");
  }
}

double YamlNode::as_double(const std::string& key) const {
  const std::string raw = as_string(key);
  try {
    size_t consumed = 0;
    const double value = std::stod(raw, &consumed);
    if (consumed != raw.size()) {
      throw std::invalid_argument("trailing characters");
    }
    return value;
  } catch (const std::exception&) {
    throw std::runtime_error("YAML: key '" + key + "' value '" + raw + "' is not a number");
  }
}

bool YamlNode::as_bool(const std::string& key) const {
  std::string raw = as_string(key);
  std::transform(raw.begin(), raw.end(), raw.begin(), [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
  if (raw == "true" || raw == "yes" || raw == "on") return true;
  if (raw == "false" || raw == "no" || raw == "off") return false;
  throw std::runtime_error("YAML: key '" + key + "' value is not a boolean");
}

namespace {

std::string strip_comment(const std::string& line) {
  bool in_single = false;
  bool in_double = false;
  for (size_t i = 0; i < line.size(); ++i) {
    const char ch = line[i];
    if (ch == '\'' && !in_double) in_single = !in_single;
    else if (ch == '"' && !in_single) in_double = !in_double;
    else if (ch == '#' && !in_single && !in_double) return line.substr(0, i);
  }
  return line;
}

std::string trim(const std::string& raw) {
  const size_t begin = raw.find_first_not_of(" \t\r\n");
  if (begin == std::string::npos) return "";
  const size_t end = raw.find_last_not_of(" \t\r\n");
  return raw.substr(begin, end - begin + 1);
}

std::string unquote(const std::string& raw) {
  if (raw.size() >= 2 && ((raw.front() == '"' && raw.back() == '"') || (raw.front() == '\'' && raw.back() == '\''))) {
    return raw.substr(1, raw.size() - 2);
  }
  return raw;
}

std::vector<std::string> split_flow_list(const std::string& raw) {
  std::vector<std::string> items;
  std::string current;
  int depth = 0;
  for (const char ch : raw) {
    if (ch == '[') ++depth;
    if (ch == ']') --depth;
    if (ch == ',' && depth == 0) {
      items.push_back(trim(current));
      current.clear();
      continue;
    }
    current.push_back(ch);
  }
  if (!trim(current).empty()) {
    items.push_back(trim(current));
  }
  return items;
}

}  // namespace

class YamlParser {
 public:
  explicit YamlParser(const std::string& text) : text_(text) {}

  YamlNode parse() {
    std::istringstream stream(text_);
    std::string line;
    YamlNode root;
    root.type_ = YamlNode::Type::Map;
    std::vector<YamlNode*> maps{&root};
    std::vector<int> indents{-1};
    int line_no = 0;

    while (std::getline(stream, line)) {
      ++line_no;
      const std::string without_comment = strip_comment(line);
      if (trim(without_comment).empty()) {
        continue;
      }
      const size_t first = without_comment.find_first_not_of(' ');
      if (first == std::string::npos) {
        continue;
      }
      if (without_comment.find('\t', 0) != std::string::npos) {
        throw std::runtime_error("YAML line " + std::to_string(line_no) + ": tab indentation is not supported");
      }
      const int indent = static_cast<int>(first);
      const std::string body = trim(without_comment);
      const size_t colon = body.find(':');
      if (colon == std::string::npos) {
        throw std::runtime_error("YAML line " + std::to_string(line_no) +
                                 ": expected 'key: value', got '" + body + "'");
      }
      const std::string key = trim(body.substr(0, colon));
      const std::string value_text = trim(body.substr(colon + 1));
      if (key.empty()) {
        throw std::runtime_error("YAML line " + std::to_string(line_no) + ": empty key");
      }

      // 同级或更浅的缩进都要回到父级映射（否则兄弟顶层键会被错误地挂到上一个块下）
      while (indents.size() > 1 && indent <= indents.back()) {
        indents.pop_back();
        maps.pop_back();
      }
      if (indent > indents.back() + 4) {
        throw std::runtime_error("YAML line " + std::to_string(line_no) + ": indentation jump too large");
      }
      YamlNode* current = maps.back();

      if (value_text.empty()) {
        YamlNode child;
        child.type_ = YamlNode::Type::Map;
        current->children_[key] = child;
        current->keys_.push_back(key);
        maps.push_back(&current->children_[key]);
        indents.push_back(indent);
        continue;
      }
      YamlNode scalar;
      scalar.type_ = YamlNode::Type::Scalar;
      if (!value_text.empty() && value_text.front() == '[' && value_text.back() == ']') {
        scalar.type_ = YamlNode::Type::List;
        for (const auto& item : split_flow_list(value_text.substr(1, value_text.size() - 2))) {
          if (!item.empty()) {
            scalar.list_.push_back(unquote(item));
          }
        }
      } else {
        scalar.scalar_ = unquote(value_text);
      }
      current->children_[key] = scalar;
      current->keys_.push_back(key);
    }
    return root;
  }

 private:
  const std::string& text_;
};

YamlNode YamlNode::parse(const std::string& text) {
  YamlParser parser(text);
  return parser.parse();
}

YamlNode YamlNode::parse_file(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("YamlNode::parse_file cannot open " + path);
  }
  std::ostringstream buffer;
  buffer << in.rdbuf();
  return parse(buffer.str());
}

}  // namespace actlab

#include "actlab/json_min.hpp"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace actlab {

Json Json::make_object() {
  Json value;
  value.type_ = Type::Object;
  return value;
}

Json Json::make_array() {
  Json value;
  value.type_ = Type::Array;
  return value;
}

Json Json::make_string(std::string text) {
  Json value;
  value.type_ = Type::String;
  value.string_ = std::move(text);
  return value;
}

Json Json::make_number(double number) {
  Json value;
  value.type_ = Type::Number;
  value.number_ = number;
  return value;
}

Json Json::make_bool(bool flag) {
  Json value;
  value.type_ = Type::Bool;
  value.bool_ = flag;
  return value;
}

void Json::set(const std::string& key, Json value) {
  if (type_ != Type::Object) {
    throw std::runtime_error("Json::set on non-object");
  }
  if (object_.find(key) == object_.end()) {
    keys_.push_back(key);
  }
  object_[key] = std::move(value);
}

void Json::push(Json value) {
  if (type_ != Type::Array) {
    throw std::runtime_error("Json::push on non-array");
  }
  array_.push_back(std::move(value));
}

bool Json::has(const std::string& key) const {
  return type_ == Type::Object && object_.find(key) != object_.end();
}

const Json& Json::at(const std::string& key) const {
  if (type_ != Type::Object) {
    throw std::runtime_error("Json::at on non-object");
  }
  const auto it = object_.find(key);
  if (it == object_.end()) {
    throw std::out_of_range("Json::at missing key: " + key);
  }
  return it->second;
}

Json& Json::at_mut(const std::string& key) {
  if (type_ != Type::Object) {
    throw std::runtime_error("Json::at_mut on non-object");
  }
  const auto it = object_.find(key);
  if (it == object_.end()) {
    throw std::out_of_range("Json::at_mut missing key: " + key);
  }
  return it->second;
}

std::string Json::string_or_throw(const std::string& key) const {
  const Json& value = at(key);
  if (value.type() != Type::String) {
    throw std::runtime_error("Json field '" + key + "' is not a string");
  }
  return value.as_string();
}

double Json::number_or_throw(const std::string& key) const {
  const Json& value = at(key);
  if (value.type() != Type::Number) {
    throw std::runtime_error("Json field '" + key + "' is not a number");
  }
  return value.as_number();
}

long long Json::int_or_throw(const std::string& key) const {
  return static_cast<long long>(number_or_throw(key));
}

std::string json_escape(const std::string& raw) {
  std::string out;
  out.reserve(raw.size() + 8);
  for (const char ch : raw) {
    switch (ch) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (static_cast<unsigned char>(ch) < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned char>(ch));
          out += buf;
        } else {
          out.push_back(ch);
        }
    }
  }
  return out;
}

void Json::dump_to(std::string& out, int indent, int depth) const {
  const std::string pad(static_cast<size_t>(indent * depth), ' ');
  const std::string pad_inner(static_cast<size_t>(indent * (depth + 1)), ' ');
  switch (type_) {
    case Type::Null: out += "null"; break;
    case Type::Bool: out += bool_ ? "true" : "false"; break;
    case Type::Number: {
      std::ostringstream stream;
      if (std::fabs(number_ - std::floor(number_)) < 1e-12 && std::fabs(number_) < 1e15) {
        stream << static_cast<long long>(number_);
      } else {
        stream.precision(17);
        stream << number_;
      }
      out += stream.str();
      break;
    }
    case Type::String: out += "\"" + json_escape(string_) + "\""; break;
    case Type::Array: {
      out += "[\n";
      for (size_t i = 0; i < array_.size(); ++i) {
        out += pad_inner;
        array_[i].dump_to(out, indent, depth + 1);
        out += (i + 1 < array_.size()) ? ",\n" : "\n";
      }
      out += pad + "]";
      break;
    }
    case Type::Object: {
      out += "{\n";
      for (size_t i = 0; i < keys_.size(); ++i) {
        const auto& key = keys_[i];
        out += pad_inner + "\"" + json_escape(key) + "\": ";
        object_.at(key).dump_to(out, indent, depth + 1);
        out += (i + 1 < keys_.size()) ? ",\n" : "\n";
      }
      out += pad + "}";
      break;
    }
  }
}

std::string Json::dump(int indent) const {
  std::string out;
  dump_to(out, indent, 0);
  return out;
}

namespace {

class Parser {
 public:
  explicit Parser(const std::string& text) : text_(text) {}

  Json parse() {
    skip_ws();
    Json value = parse_value();
    skip_ws();
    if (pos_ != text_.size()) {
      fail("trailing characters");
    }
    return value;
  }

 private:
  const std::string& text_;
  size_t pos_ = 0;

  [[noreturn]] void fail(const std::string& why) const {
    throw std::runtime_error("json parse error at offset " + std::to_string(pos_) + ": " + why);
  }

  void skip_ws() {
    while (pos_ < text_.size() && (text_[pos_] == ' ' || text_[pos_] == '\n' || text_[pos_] == '\r' ||
                                   text_[pos_] == '\t')) {
      ++pos_;
    }
  }

  char peek() {
    if (pos_ >= text_.size()) {
      fail("unexpected end of input");
    }
    return text_[pos_];
  }

  Json parse_value() {
    switch (peek()) {
      case '{': return parse_object();
      case '[': return parse_array();
      case '"': return Json::make_string(parse_string());
      case 't': expect_literal("true"); return Json::make_bool(true);
      case 'f': expect_literal("false"); return Json::make_bool(false);
      case 'n': expect_literal("null"); return Json();
      default: return parse_number();
    }
  }

  void expect_literal(const char* literal) {
    const std::string text(literal);
    if (text_.compare(pos_, text.size(), text) != 0) {
      fail("invalid literal");
    }
    pos_ += text.size();
  }

  Json parse_object() {
    ++pos_;  // '{'
    Json object = Json::make_object();
    skip_ws();
    if (peek() == '}') {
      ++pos_;
      return object;
    }
    while (true) {
      skip_ws();
      const std::string key = parse_string();
      skip_ws();
      if (peek() != ':') {
        fail("expected ':'");
      }
      ++pos_;
      skip_ws();
      object.set(key, parse_value());
      skip_ws();
      const char ch = peek();
      if (ch == ',') {
        ++pos_;
        continue;
      }
      if (ch == '}') {
        ++pos_;
        return object;
      }
      fail("expected ',' or '}'");
    }
  }

  Json parse_array() {
    ++pos_;  // '['
    Json array = Json::make_array();
    skip_ws();
    if (peek() == ']') {
      ++pos_;
      return array;
    }
    while (true) {
      skip_ws();
      array.push(parse_value());
      skip_ws();
      const char ch = peek();
      if (ch == ',') {
        ++pos_;
        continue;
      }
      if (ch == ']') {
        ++pos_;
        return array;
      }
      fail("expected ',' or ']'");
    }
  }

  std::string parse_string() {
    if (peek() != '"') {
      fail("expected string");
    }
    ++pos_;
    std::string out;
    while (true) {
      if (pos_ >= text_.size()) {
        fail("unterminated string");
      }
      const char ch = text_[pos_++];
      if (ch == '"') {
        return out;
      }
      if (ch != '\\') {
        out.push_back(ch);
        continue;
      }
      if (pos_ >= text_.size()) {
        fail("unterminated escape");
      }
      const char esc = text_[pos_++];
      switch (esc) {
        case '"': out.push_back('"'); break;
        case '\\': out.push_back('\\'); break;
        case '/': out.push_back('/'); break;
        case 'b': out.push_back('\b'); break;
        case 'f': out.push_back('\f'); break;
        case 'n': out.push_back('\n'); break;
        case 'r': out.push_back('\r'); break;
        case 't': out.push_back('\t'); break;
        case 'u': {
          if (pos_ + 4 > text_.size()) {
            fail("bad unicode escape");
          }
          const std::string hex = text_.substr(pos_, 4);
          pos_ += 4;
          const unsigned code = static_cast<unsigned>(std::stoul(hex, nullptr, 16));
          if (code < 0x80) {
            out.push_back(static_cast<char>(code));
          } else if (code < 0x800) {
            out.push_back(static_cast<char>(0xC0 | (code >> 6)));
            out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
          } else {
            out.push_back(static_cast<char>(0xE0 | (code >> 12)));
            out.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
          }
          break;
        }
        default: fail("unsupported escape");
      }
    }
  }

  Json parse_number() {
    const size_t start = pos_;
    while (pos_ < text_.size()) {
      const char ch = text_[pos_];
      if ((ch >= '0' && ch <= '9') || ch == '-' || ch == '+' || ch == '.' || ch == 'e' || ch == 'E') {
        ++pos_;
      } else {
        break;
      }
    }
    if (start == pos_) {
      fail("expected number");
    }
    try {
      return Json::make_number(std::stod(text_.substr(start, pos_ - start)));
    } catch (const std::exception&) {
      fail("invalid number");
    }
  }
};

}  // namespace

Json Json::parse(const std::string& text) {
  Parser parser(text);
  return parser.parse();
}

Json Json::parse_file(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("Json::parse_file cannot open " + path);
  }
  std::ostringstream buffer;
  buffer << in.rdbuf();
  return parse(buffer.str());
}

}  // namespace actlab

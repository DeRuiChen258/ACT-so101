// 极简测试框架：避免引入 gtest 等额外依赖（KISS/YAGNI）
#pragma once

#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace actlab_test {

struct TestCase {
  std::string name;
  std::function<void()> body;
};

inline std::vector<TestCase>& registry() {
  static std::vector<TestCase> tests;
  return tests;
}

struct Registrar {
  Registrar(const std::string& name, std::function<void()> body) {
    registry().push_back({name, std::move(body)});
  }
};

inline void fail(const std::string& file, int line, const std::string& message) {
  std::ostringstream out;
  out << file << ":" << line << ": " << message;
  throw std::runtime_error(out.str());
}

}  // namespace actlab_test

#define ACTLAB_TEST(name)                                                        \
  static void name();                                                            \
  static const actlab_test::Registrar registrar_##name(#name, name);             \
  static void name()

#define CHECK_TRUE(expr)                                                         \
  do {                                                                           \
    if (!(expr)) {                                                               \
      actlab_test::fail(__FILE__, __LINE__, "CHECK_TRUE failed: " #expr);        \
    }                                                                            \
  } while (false)

#define CHECK_EQ(lhs, rhs)                                                       \
  do {                                                                           \
    const auto left_value = (lhs);                                               \
    const auto right_value = (rhs);                                              \
    if (!(left_value == right_value)) {                                          \
      std::ostringstream message;                                                \
      message << "CHECK_EQ failed: " #lhs " == " #rhs " (got " << left_value     \
              << " vs " << right_value << ")";                                   \
      actlab_test::fail(__FILE__, __LINE__, message.str());                      \
    }                                                                            \
  } while (false)

#define CHECK_THROWS(expr)                                                       \
  do {                                                                           \
    bool threw = false;                                                          \
    try {                                                                        \
      expr;                                                                      \
    } catch (const std::exception&) {                                            \
      threw = true;                                                              \
    }                                                                            \
    if (!threw) {                                                                \
      actlab_test::fail(__FILE__, __LINE__, "CHECK_THROWS failed: " #expr);      \
    }                                                                            \
  } while (false)

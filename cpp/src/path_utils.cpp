#include "actlab/path_utils.hpp"

#include <chrono>
#include <ctime>
#include <cstdlib>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace actlab {

fs::path expand_user(const std::string& raw) {
  if (raw.empty() || raw[0] != '~') {
    return fs::path(raw);
  }
  const char* home = std::getenv("HOME");
  if (home == nullptr) {
    throw std::runtime_error("expand_user: $HOME is not set");
  }
  if (raw.size() == 1) {
    return fs::path(home);
  }
  if (raw[1] == '/') {
    return fs::path(home) / raw.substr(2);
  }
  return fs::path(raw);  // "~user" 不支持，按原样返回并由调用方校验
}

fs::path resolve_path(const fs::path& raw, const fs::path& base) {
  fs::path path = expand_user(raw.string());
  if (path.is_relative()) {
    path = base / path;
  }
  return path.lexically_normal();
}

void ensure_dir(const fs::path& dir) {
  std::error_code ec;
  fs::create_directories(dir, ec);
  if (ec) {
    throw std::runtime_error("ensure_dir failed for " + dir.string() + ": " + ec.message());
  }
}

void ensure_writable_target(const fs::path& target, bool overwrite) {
  std::error_code ec;
  if (fs::exists(target, ec) && !overwrite) {
    throw std::runtime_error("refusing to overwrite existing path: " + target.string() +
                             " (pass --overwrite to allow)");
  }
}

std::string timestamp_compact() {
  const auto now = std::chrono::system_clock::now();
  const std::time_t t = std::chrono::system_clock::to_time_t(now);
  std::tm tm{};
  localtime_r(&t, &tm);
  std::ostringstream out;
  out << std::put_time(&tm, "%Y%m%d_%H%M%S");
  return out.str();
}

std::string timestamp_iso8601() {
  const auto now = std::chrono::system_clock::now();
  const std::time_t t = std::chrono::system_clock::to_time_t(now);
  std::tm tm{};
  localtime_r(&t, &tm);
  std::ostringstream out;
  out << std::put_time(&tm, "%Y-%m-%dT%H:%M:%S") << std::put_time(&tm, "%z");
  std::string text = out.str();
  if (text.size() >= 5) {  // "+0800" -> "+08:00"
    text.insert(text.size() - 2, ":");
  }
  return text;
}

std::string relative_to(const fs::path& path, const fs::path& base) {
  std::error_code ec;
  const auto rel = fs::relative(path, base, ec);
  if (ec || rel.empty()) {
    return path.string();
  }
  return rel.string();
}

}  // namespace actlab

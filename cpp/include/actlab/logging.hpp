// 结构化日志：同时写控制台与 logs/*.jsonl（timestamp / level / stage / module / duration_ms）
#pragma once

#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>

namespace actlab {

enum class LogLevel { Debug, Info, Warn, Error };

const char* to_string(LogLevel level);

class Logger {
 public:
  Logger(std::string stage, std::filesystem::path jsonl_path, LogLevel console_level = LogLevel::Info);
  ~Logger();

  Logger(const Logger&) = delete;
  Logger& operator=(const Logger&) = delete;

  void log(LogLevel level, const std::string& module, const std::string& message,
           double duration_ms = -1.0);
  void debug(const std::string& module, const std::string& message, double duration_ms = -1.0) {
    log(LogLevel::Debug, module, message, duration_ms);
  }
  void info(const std::string& module, const std::string& message, double duration_ms = -1.0) {
    log(LogLevel::Info, module, message, duration_ms);
  }
  void warn(const std::string& module, const std::string& message, double duration_ms = -1.0) {
    log(LogLevel::Warn, module, message, duration_ms);
  }
  void error(const std::string& module, const std::string& message, double duration_ms = -1.0) {
    log(LogLevel::Error, module, message, duration_ms);
  }

  const std::string& stage() const { return stage_; }
  std::size_t record_count() const { return record_count_; }

 private:
  std::string stage_;
  std::filesystem::path jsonl_path_;
  std::ofstream sink_;
  LogLevel console_level_;
  std::mutex mutex_;
  std::size_t record_count_ = 0;
};

}  // namespace actlab

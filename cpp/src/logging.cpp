#include "actlab/logging.hpp"

#include "actlab/json_min.hpp"
#include "actlab/path_utils.hpp"

#include <iostream>
#include <stdexcept>

namespace actlab {

const char* to_string(LogLevel level) {
  switch (level) {
    case LogLevel::Debug: return "DEBUG";
    case LogLevel::Info: return "INFO";
    case LogLevel::Warn: return "WARN";
    case LogLevel::Error: return "ERROR";
  }
  return "UNKNOWN";
}

Logger::Logger(std::string stage, std::filesystem::path jsonl_path, LogLevel console_level)
    : stage_(std::move(stage)), jsonl_path_(std::move(jsonl_path)), console_level_(console_level) {
  ensure_dir(jsonl_path_.parent_path());
  const bool exists = std::filesystem::exists(jsonl_path_);
  sink_.open(jsonl_path_, std::ios::app);
  if (!sink_) {
    throw std::runtime_error("Logger: cannot open " + jsonl_path_.string());
  }
  if (!exists) {
    sink_ << "# actlab structured log (JSON Lines) — stage=" << stage_ << "\n";
    sink_.flush();
  }
}

Logger::~Logger() {
  if (sink_.is_open()) {
    sink_.flush();
  }
}

void Logger::log(LogLevel level, const std::string& module, const std::string& message, double duration_ms) {
  std::lock_guard<std::mutex> guard(mutex_);
  Json record = Json::make_object();
  record.set("ts", timestamp_iso8601());
  record.set("level", to_string(level));
  record.set("stage", stage_);
  record.set("module", module);
  record.set("message", message);
  if (duration_ms >= 0.0) {
    record.set("duration_ms", duration_ms);
  }
  sink_ << record.dump(0) << "\n";
  sink_.flush();
  ++record_count_;

  if (static_cast<int>(level) >= static_cast<int>(console_level_)) {
    std::ostream& out = (level == LogLevel::Error) ? std::cerr : std::cout;
    out << "[" << to_string(level) << "][" << stage_ << "][" << module << "] " << message;
    if (duration_ms >= 0.0) {
      out << " (" << duration_ms << " ms)";
    }
    out << std::endl;
  }
}

}  // namespace actlab

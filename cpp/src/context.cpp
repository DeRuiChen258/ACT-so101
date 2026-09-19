#include "actlab/context.hpp"

#include "actlab/path_utils.hpp"

#include <cstdlib>
#include <stdexcept>

namespace actlab {
namespace {

std::string getenv_or(const char* name, const std::string& fallback) {
  const char* raw = std::getenv(name);
  return raw == nullptr ? fallback : std::string(raw);
}

}  // namespace

const char* to_string(StageStatus status) {
  switch (status) {
    case StageStatus::Pass: return "PASS";
    case StageStatus::Fail: return "FAIL";
    case StageStatus::Blocked: return "BLOCKED";
  }
  return "UNKNOWN";
}

StageResult StageResult::pass(std::string stage, std::string detail) {
  return StageResult{std::move(stage), StageStatus::Pass, std::move(detail), {}, {}, 0.0};
}

StageResult StageResult::fail(std::string stage, std::string detail) {
  return StageResult{std::move(stage), StageStatus::Fail, std::move(detail), {}, {}, 0.0};
}

StageResult StageResult::blocked(std::string stage, std::string detail) {
  return StageResult{std::move(stage), StageStatus::Blocked, std::move(detail), {}, {}, 0.0};
}

RunContext::RunContext(RunOptions options, ExperimentConfig config, fs::path experiment_root,
                       fs::path env_root)
    : options_(std::move(options)),
      config_(std::move(config)),
      experiment_root_(std::move(experiment_root)),
      env_root_(std::move(env_root)) {
  ensure_dir(experiment_root_ / "logs");
  logger_ = std::make_unique<Logger>(options_.stage, experiment_root_ / "logs" /
                                                         ("events_" + options_.stage + ".jsonl"));
  manifest_ = std::make_unique<Manifest>(experiment_root_ / "experiment_manifest.json");
  logger_->info("context", "experiment_root=" + experiment_root_.string() +
                               " env_root=" + env_root_.string() + " config_sha256=" +
                               config_.config_sha256.substr(0, 16));
}

fs::path RunContext::resolve(const std::string& maybe_relative) const {
  if (maybe_relative.empty()) {
    throw std::runtime_error("RunContext::resolve: empty path");
  }
  return resolve_path(maybe_relative, experiment_root_);
}

fs::path RunContext::log_file(const std::string& name) const {
  ensure_dir(experiment_root_ / "logs");
  return experiment_root_ / "logs" / name;
}

fs::path RunContext::evidence_file(const std::string& name) const {
  ensure_dir(experiment_root_ / "evidence");
  return experiment_root_ / "evidence" / name;
}

fs::path RunContext::results_file(const std::string& name) const {
  ensure_dir(experiment_root_ / "results");
  return experiment_root_ / "results" / name;
}

std::string RunContext::python_bin() const {
  return (fs::path(config_.paths.conda_root) / "envs" / config_.paths.env_name / "bin" / "python")
      .string();
}

ProcResult RunContext::run_stack(const std::vector<std::string>& argv, const std::string& log_name,
                                 int timeout_seconds) {
  const std::string bin_dir =
      (fs::path(config_.paths.conda_root) / "envs" / config_.paths.env_name / "bin").string();
  const std::string libtorch_lib = (fs::path(config_.paths.libtorch_dir) / "lib").string();

  ProcOptions options;
  options.log_path = log_file(log_name);
  options.timeout_seconds = timeout_seconds > 0 ? timeout_seconds : config_.runtime.default_timeout_seconds;
  options.env["PATH"] = bin_dir + ":/usr/local/bin:/usr/bin:/bin";
  options.env["LD_LIBRARY_PATH"] = libtorch_lib + ":" + getenv_or("LD_LIBRARY_PATH", "");
  options.env["HF_LEROBOT_HOME"] = config_.paths.data_dir;
  options.env["PYTHONUNBUFFERED"] = "1";
  options.env["TOKENIZERS_PARALLELISM"] = "false";
  options.env["CUDA_VISIBLE_DEVICES"] = getenv_or("CUDA_VISIBLE_DEVICES", "0");

  manifest_->add_command(options_.stage, format_command(argv));
  logger_->info("proc", "exec: " + format_command(argv));
  const ProcResult result = run_process(argv, options);
  logger_->info("proc",
                "exit=" + std::to_string(result.exit_code) + " duration_ms=" +
                    std::to_string(static_cast<long long>(result.duration_ms)),
                result.duration_ms);
  return result;
}

}  // namespace actlab

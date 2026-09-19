#include "actlab/config_loader.hpp"

#include "actlab/sha256.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace actlab {
namespace {

std::string read_text(const fs::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("config file not found: " + path.string());
  }
  std::ostringstream buffer;
  buffer << in.rdbuf();
  return buffer.str();
}

void require(bool condition, const std::string& what) {
  if (!condition) {
    throw std::runtime_error("config validation failed: " + what);
  }
}

void require_positive(long long value, const std::string& what) {
  if (value <= 0) {
    throw std::runtime_error("config validation failed: " + what + " must be > 0 (got " +
                             std::to_string(value) + ")");
  }
}

void require_range(double value, double low, double high, const std::string& what) {
  if (value < low || value > high) {
    throw std::runtime_error("config validation failed: " + what + " out of range [" +
                             std::to_string(low) + ", " + std::to_string(high) + "] (got " +
                             std::to_string(value) + ")");
  }
}

}  // namespace

ExperimentConfig load_experiment_config(const fs::path& config_dir) {
  ExperimentConfig config;
  config.config_dir = config_dir.string();

  const fs::path act_path = config_dir / "act_baseline.yaml";
  const fs::path paths_path = config_dir / "paths.yaml";
  const fs::path runtime_path = config_dir / "runtime.yaml";

  const std::string act_text = read_text(act_path);
  const std::string paths_text = read_text(paths_path);
  const std::string runtime_text = read_text(runtime_path);
  config.config_sha256 = sha256_hex(act_text + "|" + paths_text + "|" + runtime_text);

  const YamlNode act = YamlNode::parse(act_text);
  const YamlNode paths = YamlNode::parse(paths_text);
  const YamlNode runtime = YamlNode::parse(runtime_text);

  const YamlNode& policy = act.at("policy");
  config.act.n_obs_steps = static_cast<int>(policy.as_int("n_obs_steps"));
  config.act.chunk_size = static_cast<int>(policy.as_int("chunk_size"));
  config.act.n_action_steps = static_cast<int>(policy.as_int("n_action_steps"));
  config.act.vision_backbone = policy.as_string("vision_backbone");
  config.act.dim_model = static_cast<int>(policy.as_int("dim_model"));
  config.act.dim_feedforward = static_cast<int>(policy.as_int("dim_feedforward"));
  config.act.n_heads = static_cast<int>(policy.as_int("n_heads"));
  config.act.n_encoder_layers = static_cast<int>(policy.as_int("n_encoder_layers"));
  config.act.n_decoder_layers = static_cast<int>(policy.as_int("n_decoder_layers"));
  config.act.use_vae = policy.as_bool("use_vae");
  config.act.latent_dim = static_cast<int>(policy.as_int("latent_dim"));
  config.act.n_vae_encoder_layers = static_cast<int>(policy.as_int("n_vae_encoder_layers"));
  config.act.dropout = policy.as_double("dropout");
  config.act.kl_weight = policy.as_double("kl_weight");
  config.act.temporal_ensemble_coeff = policy.as_double("temporal_ensemble_coeff");
  config.act.has_temporal_ensemble = config.act.temporal_ensemble_coeff > 0.0;

  const YamlNode& training = act.at("training");
  config.act.optimizer = training.as_string("optimizer");
  config.act.optimizer_lr = training.as_double("optimizer_lr");
  config.act.optimizer_weight_decay = training.as_double("optimizer_weight_decay");
  config.act.batch_size = static_cast<int>(training.as_int("batch_size"));
  config.act.steps = training.as_int("steps");
  config.act.num_workers = static_cast<int>(training.as_int("num_workers"));
  config.act.seed = training.as_int("seed");
  config.act.save_freq = training.as_int("save_freq");
  config.act.log_freq = training.as_int("log_freq");
  config.act.precision = training.as_string("precision");
  config.act.policy_type = training.as_string("policy_type");
  config.act.job_name = training.as_string("job_name");

  const YamlNode& path_node = paths.at("paths");
  config.paths.repo_dir = path_node.as_string("repo_dir");
  config.paths.data_dir = path_node.as_string("data_dir");
  config.paths.outputs_dir = path_node.as_string("outputs_dir");
  config.paths.artifacts_dir = path_node.as_string("artifacts_dir");
  config.paths.libtorch_dir = path_node.as_string("libtorch_dir");
  config.paths.conda_root = path_node.as_string("conda_root");
  config.paths.env_name = path_node.as_string("env_name");
  config.paths.screenshot_tool = path_node.as_string("screenshot_tool");
  config.paths.torchscript_rel = path_node.as_string("torchscript_rel");
  config.paths.onnx_rel = path_node.as_string("onnx_rel");

  const YamlNode& runtime_node = runtime.at("runtime");
  config.runtime.log_level = runtime_node.as_string("log_level");
  config.runtime.metric_sample_interval = static_cast<int>(runtime_node.as_int("metric_sample_interval"));
  config.runtime.smoke_steps = static_cast<int>(runtime_node.as_int("smoke_steps"));
  config.runtime.default_timeout_seconds = static_cast<int>(runtime_node.as_int("default_timeout_seconds"));
  config.runtime.train_timeout_seconds = static_cast<int>(runtime_node.as_int("train_timeout_seconds"));
  config.runtime.max_parallel_subprocess = static_cast<int>(runtime_node.as_int("max_parallel_subprocess"));
  config.runtime.oom_ladder = runtime_node.as_string("oom_ladder");

  const YamlNode& dataset_node = act.at("dataset");
  config.dataset.repo_id = dataset_node.as_string("repo_id");
  config.dataset.fallback_repo_id = dataset_node.as_string("fallback_repo_id");
  config.dataset.source_priority = dataset_node.as_string("source_priority");
  config.dataset.expected_fps = static_cast<int>(dataset_node.as_int("expected_fps"));
  config.dataset.expected_state_dim = static_cast<int>(dataset_node.as_int("expected_state_dim"));
  config.dataset.expected_action_dim = static_cast<int>(dataset_node.as_int("expected_action_dim"));
  config.dataset.min_episodes = static_cast<int>(dataset_node.as_int("min_episodes"));

  validate_config(config);
  return config;
}

void validate_config(const ExperimentConfig& config) {
  const ActConfig& act = config.act;
  require(act.n_obs_steps == 1, "policy.n_obs_steps must be 1 (current ACT implementation limit)");
  require_positive(act.chunk_size, "policy.chunk_size");
  require_positive(act.n_action_steps, "policy.n_action_steps");
  require(act.n_action_steps <= act.chunk_size, "policy.n_action_steps must be <= chunk_size");
  require(act.dim_model % act.n_heads == 0, "policy.dim_model must be divisible by n_heads");
  require(act.dim_model > 0 && act.n_heads > 0 && act.dim_feedforward > 0,
          "policy dims must be positive");
  require(act.n_encoder_layers > 0 && act.n_decoder_layers > 0 && act.n_vae_encoder_layers > 0,
          "policy layer counts must be positive");
  require(act.vision_backbone == "resnet18" || act.vision_backbone == "resnet34" ||
              act.vision_backbone == "resnet50",
          "policy.vision_backbone must be a resnet variant");
  require_range(act.dropout, 0.0, 0.9, "policy.dropout");
  require_range(act.kl_weight, 0.0, 1e4, "policy.kl_weight");
  if (act.use_vae) {
    require_positive(act.latent_dim, "policy.latent_dim (required when use_vae=true)");
  }
  if (act.has_temporal_ensemble) {
    require(act.n_action_steps == 1,
            "policy.n_action_steps must be 1 when temporal_ensemble_coeff is set "
            "(see configuration_act.__post_init__)");
    require_range(act.temporal_ensemble_coeff, 0.0, 1.0, "policy.temporal_ensemble_coeff");
  }
  require(act.optimizer == "adamw", "training.optimizer must be 'adamw' for the current ACT preset");
  require_range(act.optimizer_lr, 1e-7, 1e-1, "training.optimizer_lr");
  require_range(act.optimizer_weight_decay, 0.0, 1.0, "training.optimizer_weight_decay");
  require_positive(act.batch_size, "training.batch_size");
  require_positive(act.steps, "training.steps");
  require(act.num_workers >= 0, "training.num_workers must be >= 0");
  require(act.seed >= 0, "training.seed must be >= 0");
  require_positive(act.save_freq, "training.save_freq");
  require_positive(act.log_freq, "training.log_freq");
  require(act.precision == "fp32" || act.precision == "bf16" || act.precision == "fp16",
          "training.precision must be fp32|bf16|fp16");
  require(act.policy_type == "act", "training.policy_type must be 'act'");
  require(!act.job_name.empty(), "training.job_name must not be empty");

  require(!config.paths.repo_dir.empty(), "paths.repo_dir must not be empty");
  require(!config.paths.data_dir.empty(), "paths.data_dir must not be empty");
  require(!config.paths.outputs_dir.empty(), "paths.outputs_dir must not be empty");
  require(!config.paths.artifacts_dir.empty(), "paths.artifacts_dir must not be empty");
  require(!config.paths.libtorch_dir.empty(), "paths.libtorch_dir must not be empty");
  require(!config.paths.conda_root.empty(), "paths.conda_root must not be empty");
  require(!config.paths.env_name.empty(), "paths.env_name must not be empty");

  require(config.runtime.metric_sample_interval > 0, "runtime.metric_sample_interval must be > 0");
  require_positive(config.runtime.smoke_steps, "runtime.smoke_steps");
  require_positive(config.runtime.default_timeout_seconds, "runtime.default_timeout_seconds");
  require_positive(config.runtime.train_timeout_seconds, "runtime.train_timeout_seconds");
  require_positive(config.runtime.max_parallel_subprocess, "runtime.max_parallel_subprocess");
  require(config.runtime.oom_ladder == "batch,resolution,workers,precision",
          "runtime.oom_ladder must be the documented degradation order");

  require(!config.dataset.repo_id.empty(), "dataset.repo_id must not be empty");
  require_positive(config.dataset.expected_fps, "dataset.expected_fps");
  require_positive(config.dataset.expected_state_dim, "dataset.expected_state_dim");
  require_positive(config.dataset.expected_action_dim, "dataset.expected_action_dim");
  require_positive(config.dataset.min_episodes, "dataset.min_episodes");
}

}  // namespace actlab

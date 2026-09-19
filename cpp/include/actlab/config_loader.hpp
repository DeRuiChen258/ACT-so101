// 配置加载与字段级校验：缺失或越界必须拒绝启动（禁止静默默认）
#pragma once

#include "actlab/yaml_min.hpp"

#include <filesystem>
#include <string>

namespace actlab {

namespace fs = std::filesystem;

struct ActConfig {
  // 策略结构（以 src/lerobot/policies/act/configuration_act.py 实参为准）
  int n_obs_steps = 0;
  int chunk_size = 0;
  int n_action_steps = 0;
  std::string vision_backbone;
  int dim_model = 0;
  int dim_feedforward = 0;
  int n_heads = 0;
  int n_encoder_layers = 0;
  int n_decoder_layers = 0;
  bool use_vae = false;
  int latent_dim = 0;
  int n_vae_encoder_layers = 0;
  double dropout = -1.0;
  double kl_weight = -1.0;
  bool has_temporal_ensemble = false;
  double temporal_ensemble_coeff = 0.0;
  // 优化与训练
  double optimizer_lr = -1.0;
  double optimizer_weight_decay = -1.0;
  std::string optimizer;
  int batch_size = 0;
  long long steps = 0;
  int num_workers = -1;
  long long seed = -1;
  long long save_freq = 0;
  long long log_freq = 0;
  std::string precision;
  std::string policy_type;
  std::string job_name;
};

struct PathsConfig {
  std::string repo_dir;
  std::string data_dir;
  std::string outputs_dir;
  std::string artifacts_dir;
  std::string libtorch_dir;
  std::string conda_root;
  std::string env_name;
  std::string screenshot_tool;
  std::string torchscript_rel;   // 相对 artifacts_dir
  std::string onnx_rel;
};

struct RuntimeConfig {
  std::string log_level;
  int metric_sample_interval = 0;
  int smoke_steps = 0;
  int default_timeout_seconds = 0;
  int train_timeout_seconds = 0;
  int max_parallel_subprocess = 0;
  std::string oom_ladder;  // "batch,resolution,workers,precision"
};

struct DatasetConfig {
  std::string repo_id;
  std::string fallback_repo_id;
  std::string source_priority;  // "modelscope,hf"
  int expected_fps = 0;
  int expected_state_dim = 0;
  int expected_action_dim = 0;
  int min_episodes = 0;
};

struct ExperimentConfig {
  ActConfig act;
  PathsConfig paths;
  RuntimeConfig runtime;
  DatasetConfig dataset;
  std::string config_dir;      // 配置文件所在目录，用于相对路径解析
  std::string config_sha256;   // 三个 YAML 文件内容的合并哈希
};

// 读取 <config_dir>/{act_baseline,paths,runtime}.yaml 并校验
ExperimentConfig load_experiment_config(const fs::path& config_dir);

// 校验（load_experiment_config 内部调用，单独暴露以便 --dry-run 复用）
void validate_config(const ExperimentConfig& config);

}  // namespace actlab

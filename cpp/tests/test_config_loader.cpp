#include "actlab/config_loader.hpp"

#include "actlab_test.hpp"
#include "actlab/path_utils.hpp"

#include <filesystem>
#include <fstream>

namespace {

const char* kActYaml = R"(# act_baseline（测试用最小合法配置）
policy:
  n_obs_steps: 1
  chunk_size: 100
  n_action_steps: 16
  vision_backbone: resnet18
  dim_model: 512
  dim_feedforward: 3200
  n_heads: 8
  n_encoder_layers: 4
  n_decoder_layers: 1
  use_vae: true
  latent_dim: 32
  n_vae_encoder_layers: 4
  dropout: 0.1
  kl_weight: 10.0
  temporal_ensemble_coeff: 0.0

training:
  optimizer: adamw
  optimizer_lr: 1.0e-05
  optimizer_weight_decay: 1.0e-04
  batch_size: 8
  steps: 1000
  num_workers: 2
  seed: 1000
  save_freq: 500
  log_freq: 50
  precision: fp32
  policy_type: act
  job_name: act_baseline

dataset:
  repo_id: lerobot/svla_so101_pickplace
  fallback_repo_id: lerobot/pusht
  source_priority: modelscope,hf
  expected_fps: 30
  expected_state_dim: 6
  expected_action_dim: 6
  min_episodes: 10
)";

const char* kPathsYaml = R"(paths:
  repo_dir: /tmp/actlab_test/repo
  data_dir: /tmp/actlab_test/data
  outputs_dir: outputs
  artifacts_dir: artifacts
  libtorch_dir: /tmp/actlab_test/libtorch
  conda_root: /tmp/actlab_test/miniconda
  env_name: lerobot_act
  screenshot_tool: import
  torchscript_rel: torchscript
  onnx_rel: onnx
)";

const char* kRuntimeYaml = R"(runtime:
  log_level: info
  metric_sample_interval: 10
  smoke_steps: 100
  default_timeout_seconds: 600
  train_timeout_seconds: 7200
  max_parallel_subprocess: 2
  oom_ladder: batch,resolution,workers,precision
)";

std::filesystem::path write_config_dir(const std::string& act_body) {
  const std::filesystem::path dir =
      std::filesystem::temp_directory_path() / ("actlab_cfg_" + actlab::timestamp_compact());
  std::filesystem::create_directories(dir);
  {
    std::ofstream out(dir / "act_baseline.yaml", std::ios::trunc);
    out << act_body;
  }
  {
    std::ofstream out(dir / "paths.yaml", std::ios::trunc);
    out << kPathsYaml;
  }
  {
    std::ofstream out(dir / "runtime.yaml", std::ios::trunc);
    out << kRuntimeYaml;
  }
  return dir;
}

}  // namespace

ACTLAB_TEST(config_loader_accepts_valid_config) {
  const std::filesystem::path dir = write_config_dir(kActYaml);
  const auto config = actlab::load_experiment_config(dir);
  CHECK_EQ(config.act.chunk_size, 100);
  CHECK_EQ(config.act.n_action_steps, 16);
  CHECK_EQ(config.act.batch_size, 8);
  CHECK_EQ(config.act.steps, 1000LL);
  CHECK_TRUE(!config.config_sha256.empty());
  CHECK_TRUE(config.runtime.smoke_steps == 100);
  std::filesystem::remove_all(dir);
}

ACTLAB_TEST(config_loader_rejects_multiple_obs_steps) {
  std::string broken(kActYaml);
  broken.replace(broken.find("n_obs_steps: 1"), std::string("n_obs_steps: 1").size(),
                 "n_obs_steps: 2");
  const std::filesystem::path dir = write_config_dir(broken);
  CHECK_THROWS(actlab::load_experiment_config(dir));
  std::filesystem::remove_all(dir);
}

ACTLAB_TEST(config_loader_rejects_action_steps_over_chunk) {
  std::string broken(kActYaml);
  broken.replace(broken.find("n_action_steps: 16"), std::string("n_action_steps: 16").size(),
                 "n_action_steps: 128");
  const std::filesystem::path dir = write_config_dir(broken);
  CHECK_THROWS(actlab::load_experiment_config(dir));
  std::filesystem::remove_all(dir);
}

ACTLAB_TEST(config_loader_rejects_missing_field) {
  std::string broken(kActYaml);
  const std::string line = "  kl_weight: 10.0\n";
  broken.replace(broken.find(line), line.size(), "");
  const std::filesystem::path dir = write_config_dir(broken);
  CHECK_THROWS(actlab::load_experiment_config(dir));
  std::filesystem::remove_all(dir);
}

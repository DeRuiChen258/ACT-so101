// LeRobotDataset schema 审计：读取 meta/info.json 并校验必填 feature 与维度自洽
#pragma once

#include "actlab/json_min.hpp"

#include <filesystem>
#include <string>
#include <vector>

namespace actlab {

struct FeatureSpec {
  std::string key;
  std::string dtype;
  std::vector<long long> shape;
  bool required = true;
  bool present = false;
};

struct DatasetSchemaReport {
  std::string root;
  std::string codebase_version;
  std::string robot_type;
  long long total_episodes = -1;
  long long total_frames = -1;
  long long total_tasks = -1;
  double fps = -1.0;
  std::vector<std::string> camera_keys;
  long long state_dim = -1;
  long long action_dim = -1;
  std::vector<FeatureSpec> features;
  std::vector<std::string> problems;

  bool ok() const { return problems.empty(); }
  std::string summary() const;
};

// 校验规则：
//   - meta/info.json 存在且可解析；
//   - 至少一个 observation.images.*（dtype=video 或 image）与 observation.state / action；
//   - episode_index / frame_index / task_index / timestamp 存在；
//   - state 与 action 的 shape 一致，且等于期望维度（来自配置）；
//   - fps 与期望值一致；
//   - total_episodes >= 配置下限。
DatasetSchemaReport audit_dataset_schema(const std::filesystem::path& dataset_root,
                                         const std::vector<std::string>& required_extra_keys,
                                         int expected_fps, int expected_state_dim,
                                         int expected_action_dim, int min_episodes);

std::string render_schema_markdown(const DatasetSchemaReport& report);

}  // namespace actlab

#include "actlab/dataset_schema.hpp"

#include <algorithm>
#include <sstream>
#include <stdexcept>

namespace actlab {
namespace {

std::string join(const std::vector<std::string>& items, const std::string& sep) {
  std::string out;
  for (size_t i = 0; i < items.size(); ++i) {
    if (i > 0) out += sep;
    out += items[i];
  }
  return out;
}

std::string shape_to_string(const std::vector<long long>& shape) {
  std::ostringstream out;
  out << "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i > 0) out << ", ";
    out << shape[i];
  }
  out << "]";
  return out.str();
}

}  // namespace

std::string DatasetSchemaReport::summary() const {
  std::ostringstream out;
  out << "root=" << root << " codebase=" << codebase_version << " robot=" << robot_type
      << " episodes=" << total_episodes << " frames=" << total_frames << " fps=" << fps
      << " cameras=" << join(camera_keys, ",") << " state_dim=" << state_dim
      << " action_dim=" << action_dim << " problems=" << problems.size();
  return out.str();
}

DatasetSchemaReport audit_dataset_schema(const std::filesystem::path& dataset_root,
                                         const std::vector<std::string>& required_extra_keys,
                                         int expected_fps, int expected_state_dim,
                                         int expected_action_dim, int min_episodes) {
  DatasetSchemaReport report;
  report.root = dataset_root.string();

  const std::filesystem::path info_path = dataset_root / "meta" / "info.json";
  if (!std::filesystem::exists(info_path)) {
    report.problems.push_back("missing meta/info.json at " + info_path.string());
    return report;
  }
  const Json info = Json::parse_file(info_path.string());
  report.codebase_version = info.has("codebase_version") ? info.string_or_throw("codebase_version") : "unknown";
  report.robot_type = info.has("robot_type") ? info.string_or_throw("robot_type") : "unknown";
  report.total_episodes = info.int_or_throw("total_episodes");
  report.total_frames = info.int_or_throw("total_frames");
  report.total_tasks = info.has("total_tasks") ? info.int_or_throw("total_tasks") : -1;
  report.fps = info.number_or_throw("fps");

  if (report.total_episodes < min_episodes) {
    report.problems.push_back("total_episodes=" + std::to_string(report.total_episodes) +
                              " < required min_episodes=" + std::to_string(min_episodes));
  }
  if (expected_fps > 0 && static_cast<int>(report.fps) != expected_fps) {
    report.problems.push_back("fps=" + std::to_string(report.fps) + " != expected " +
                              std::to_string(expected_fps));
  }

  if (!info.has("features")) {
    report.problems.push_back("info.json has no 'features' section");
    return report;
  }
  const Json& features = info.at("features");

  for (const auto& key : features.keys()) {
    const Json& spec = features.at(key);
    FeatureSpec feature;
    feature.key = key;
    feature.dtype = spec.has("dtype") ? spec.string_or_throw("dtype") : "unknown";
    if (spec.has("shape")) {
      for (const auto& dim : spec.at("shape").items()) {
        feature.shape.push_back(static_cast<long long>(dim.as_number()));
      }
    }
    feature.present = true;
    report.features.push_back(feature);

    if (key.rfind("observation.images.", 0) == 0) {
      report.camera_keys.push_back(key);
      if (feature.shape.size() != 3) {
        report.problems.push_back("camera " + key + " shape is not [H, W, C]: " +
                                  shape_to_string(feature.shape));
      }
    } else if (key == "observation.state") {
      if (feature.shape.size() != 1) {
        report.problems.push_back("observation.state must be 1-D, got " + shape_to_string(feature.shape));
      } else {
        report.state_dim = feature.shape[0];
      }
    } else if (key == "action") {
      if (feature.shape.size() != 1) {
        report.problems.push_back("action must be 1-D, got " + shape_to_string(feature.shape));
      } else {
        report.action_dim = feature.shape[0];
      }
    }
  }

  if (report.camera_keys.empty()) {
    report.problems.push_back("no observation.images.* camera feature found");
  }
  if (report.state_dim <= 0) {
    report.problems.push_back("observation.state missing or invalid");
  }
  if (report.action_dim <= 0) {
    report.problems.push_back("action missing or invalid");
  }
  if (report.state_dim > 0 && report.action_dim > 0 && report.state_dim != report.action_dim) {
    report.problems.push_back("state_dim=" + std::to_string(report.state_dim) +
                              " != action_dim=" + std::to_string(report.action_dim));
  }
  if (expected_state_dim > 0 && report.state_dim > 0 && report.state_dim != expected_state_dim) {
    report.problems.push_back("state_dim=" + std::to_string(report.state_dim) + " != expected " +
                              std::to_string(expected_state_dim));
  }
  if (expected_action_dim > 0 && report.action_dim > 0 && report.action_dim != expected_action_dim) {
    report.problems.push_back("action_dim=" + std::to_string(report.action_dim) + " != expected " +
                              std::to_string(expected_action_dim));
  }

  std::vector<std::string> required = {"episode_index", "frame_index", "timestamp", "task_index", "index"};
  required.insert(required.end(), required_extra_keys.begin(), required_extra_keys.end());
  for (const auto& key : required) {
    if (!features.has(key)) {
      report.problems.push_back("required feature missing: " + key);
    }
  }
  return report;
}

std::string render_schema_markdown(const DatasetSchemaReport& report) {
  std::ostringstream out;
  out << "# Dataset schema audit\n\n";
  out << "- root: `" << report.root << "`\n";
  out << "- codebase_version: `" << report.codebase_version << "`\n";
  out << "- robot_type: `" << report.robot_type << "`\n";
  out << "- total_episodes: " << report.total_episodes << "\n";
  out << "- total_frames: " << report.total_frames << "\n";
  out << "- total_tasks: " << report.total_tasks << "\n";
  out << "- fps: " << report.fps << "\n";
  out << "- camera_keys: " << join(report.camera_keys, ", ") << "\n";
  out << "- state_dim: " << report.state_dim << "\n";
  out << "- action_dim: " << report.action_dim << "\n\n";
  out << "| feature | dtype | shape |\n| --- | --- | --- |\n";
  for (const auto& feature : report.features) {
    out << "| `" << feature.key << "` | " << feature.dtype << " | " << shape_to_string(feature.shape)
        << " |\n";
  }
  out << "\n**problems**: ";
  if (report.problems.empty()) {
    out << "none (schema PASS)\n";
  } else {
    out << "\n";
    for (const auto& problem : report.problems) {
      out << "- " << problem << "\n";
    }
  }
  return out.str();
}

}  // namespace actlab

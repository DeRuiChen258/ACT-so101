#include "actlab/dataset_schema.hpp"

#include "actlab_test.hpp"
#include "actlab/path_utils.hpp"

#include <filesystem>
#include <fstream>

namespace {

std::filesystem::path write_info(const std::string& body) {
  const std::filesystem::path root =
      std::filesystem::temp_directory_path() / ("actlab_schema_" + actlab::timestamp_compact());
  std::filesystem::create_directories(root / "meta");
  std::ofstream out(root / "meta" / "info.json", std::ios::trunc);
  out << body;
  return root;
}

const char* kValidInfo = R"({
  "codebase_version": "v3.0",
  "robot_type": "so100_follower",
  "total_episodes": 50,
  "total_frames": 11939,
  "total_tasks": 1,
  "fps": 30,
  "features": {
    "action": {"dtype": "float32", "shape": [6]},
    "observation.state": {"dtype": "float32", "shape": [6]},
    "observation.images.up": {"dtype": "video", "shape": [480, 640, 3]},
    "episode_index": {"dtype": "int64", "shape": [1]},
    "frame_index": {"dtype": "int64", "shape": [1]},
    "timestamp": {"dtype": "float32", "shape": [1]},
    "task_index": {"dtype": "int64", "shape": [1]},
    "index": {"dtype": "int64", "shape": [1]}
  }
})";

}  // namespace

ACTLAB_TEST(schema_valid_dataset_passes) {
  const std::filesystem::path root = write_info(kValidInfo);
  const auto report = actlab::audit_dataset_schema(root, {}, 30, 6, 6, 10);
  CHECK_TRUE(report.ok());
  CHECK_EQ(report.total_episodes, 50LL);
  CHECK_EQ(report.camera_keys.size(), static_cast<size_t>(1));
  CHECK_EQ(report.state_dim, 6LL);
  std::filesystem::remove_all(root);
}

ACTLAB_TEST(schema_detects_dimension_mismatch) {
  std::string broken(kValidInfo);
  const std::string from = "\"action\": {\"dtype\": \"float32\", \"shape\": [6]}";
  const std::string to = "\"action\": {\"dtype\": \"float32\", \"shape\": [5]}";
  broken.replace(broken.find(from), from.size(), to);
  const std::filesystem::path root = write_info(broken);
  const auto report = actlab::audit_dataset_schema(root, {}, 30, 6, 6, 10);
  CHECK_TRUE(!report.ok());
  CHECK_TRUE(report.problems.size() >= 1);
  std::filesystem::remove_all(root);
}

ACTLAB_TEST(schema_detects_missing_camera) {
  std::string broken(kValidInfo);
  const std::string from = "\"observation.images.up\": {\"dtype\": \"video\", \"shape\": [480, 640, 3]},";
  broken.replace(broken.find(from), from.size(), "");
  const std::filesystem::path root = write_info(broken);
  const auto report = actlab::audit_dataset_schema(root, {}, 30, 6, 6, 10);
  CHECK_TRUE(!report.ok());
  std::filesystem::remove_all(root);
}

ACTLAB_TEST(schema_missing_file_fails) {
  const std::filesystem::path root = std::filesystem::temp_directory_path() / "actlab_schema_absent";
  std::filesystem::remove_all(root);
  const auto report = actlab::audit_dataset_schema(root, {}, 30, 6, 6, 10);
  CHECK_TRUE(!report.ok());
}

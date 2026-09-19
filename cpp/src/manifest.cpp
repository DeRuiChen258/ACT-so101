#include "actlab/manifest.hpp"

#include "actlab/path_utils.hpp"
#include "actlab/sha256.hpp"

#include <fstream>
#include <stdexcept>

namespace actlab {

Manifest::Manifest(std::filesystem::path path) : path_(std::move(path)) {
  root_ = Json::make_object();
  root_.set("schema", "actlab.experiment_manifest/1");
  root_.set("generated_at", timestamp_iso8601());
  root_.set("artifacts", Json::make_array());
  root_.set("commands", Json::make_array());
  root_.set("environment", Json::make_object());
  root_.set("dataset", Json::make_object());
  root_.set("training", Json::make_object());
  root_.set("inference", Json::make_object());
  root_.set("evaluation", Json::make_object());
  load_existing();
}

void Manifest::load_existing() {
  if (!std::filesystem::exists(path_)) {
    return;
  }
  try {
    Json existing = Json::parse_file(path_.string());
    for (const auto& key : existing.keys()) {
      if (key == "artifacts" || key == "commands") {
        continue;
      }
      root_.set(key, existing.at(key));
    }
    // 已有产物与命令记录保持追加语义
    if (existing.has("artifacts")) {
      for (const auto& item : existing.at("artifacts").items()) {
        root_.at_mut("artifacts").push(item);
        ++artifact_count_;
      }
    }
    if (existing.has("commands")) {
      for (const auto& item : existing.at("commands").items()) {
        root_.at_mut("commands").push(item);
      }
    }
  } catch (const std::exception& error) {
    throw std::runtime_error("manifest: existing file is not readable JSON (" + path_.string() +
                             "): " + error.what());
  }
}

namespace {

Json& section_of(Json& root, const std::string& section) {
  if (!root.has(section)) {
    root.set(section, Json::make_object());
  }
  return const_cast<Json&>(root.at(section));
}

}  // namespace

void Manifest::set_field(const std::string& section, const std::string& key, const std::string& value) {
  section_of(root_, section).set(key, value);
  dirty_ = true;
}

void Manifest::set_field(const std::string& section, const std::string& key, double value) {
  section_of(root_, section).set(key, value);
  dirty_ = true;
}

void Manifest::set_field(const std::string& section, const std::string& key, long long value) {
  section_of(root_, section).set(key, value);
  dirty_ = true;
}

void Manifest::set_field(const std::string& section, const std::string& key, bool value) {
  section_of(root_, section).set(key, value);
  dirty_ = true;
}

void Manifest::add_artifact(const std::string& stage, const std::string& kind,
                            const std::filesystem::path& path, const std::string& note) {
  if (!std::filesystem::exists(path)) {
    throw std::runtime_error("manifest: artifact does not exist: " + path.string());
  }
  Json entry = Json::make_object();
  entry.set("stage", stage);
  entry.set("kind", kind);
  entry.set("path", std::filesystem::absolute(path).lexically_normal().string());
  entry.set("bytes", static_cast<long long>(std::filesystem::file_size(path)));
  entry.set("sha256", sha256_file(path));
  if (!note.empty()) {
    entry.set("note", note);
  }
  entry.set("recorded_at", timestamp_iso8601());
  root_.at_mut("artifacts").push(entry);
  ++artifact_count_;
  dirty_ = true;
}

void Manifest::add_command(const std::string& stage, const std::string& command) {
  Json entry = Json::make_object();
  entry.set("stage", stage);
  entry.set("command", command);
  entry.set("ts", timestamp_iso8601());
  root_.at_mut("commands").push(entry);
  dirty_ = true;
}

void Manifest::save() {
  ensure_dir(path_.parent_path());
  const std::filesystem::path tmp = path_.string() + ".tmp";
  {
    std::ofstream out(tmp, std::ios::trunc);
    if (!out) {
      throw std::runtime_error("manifest: cannot write " + tmp.string());
    }
    out << root_.dump(2) << "\n";
  }
  std::filesystem::rename(tmp, path_);
  dirty_ = false;
}

}  // namespace actlab

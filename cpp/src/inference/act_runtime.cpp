#include "actlab/inference/act_runtime.hpp"

#include "actlab/json_min.hpp"
#include "actlab/npy_io.hpp"
#include "actlab/tensor_utils.hpp"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace actlab {
namespace {

torch::Tensor npy_to_tensor(const std::filesystem::path& path) {
  const NpyArray array = read_npy(path);
  torch::ScalarType dtype = torch::kFloat32;
  if (array.dtype == "<u1" || array.dtype == "|u1") {
    dtype = torch::kUInt8;
  } else if (array.dtype == "<f4") {
    dtype = torch::kFloat32;
  } else if (array.dtype == "<f8") {
    dtype = torch::kFloat64;
  } else if (array.dtype == "<i8") {
    dtype = torch::kInt64;
  } else {
    throw std::runtime_error("act_runtime: unsupported npy dtype " + array.dtype + " in " +
                             path.string());
  }
  std::vector<int64_t> sizes;
  sizes.reserve(array.shape.size());
  for (const auto dim : array.shape) {
    sizes.push_back(dim);
  }
  auto options = torch::TensorOptions().dtype(dtype);
  torch::Tensor tensor = torch::from_blob(const_cast<uint8_t*>(array.data.data()), sizes, options);
  return tensor.clone();  // 复制到自有内存，避免引用已销毁的 buffer
}

}  // namespace

ActRuntime::ActRuntime(const std::filesystem::path& torchscript_path, const std::string& device)
    : device_(device) {
  if (!std::filesystem::exists(torchscript_path)) {
    throw std::runtime_error("act_runtime: TorchScript not found: " + torchscript_path.string());
  }
  module_ = torch::jit::load(torchscript_path.string());
  if (device_ == "cuda") {
    if (!torch::cuda::is_available()) {
      throw std::runtime_error("act_runtime: requested device=cuda but CUDA is unavailable");
    }
    module_.to(torch::kCUDA);
  } else {
    module_.to(torch::kCPU);
  }
  module_.eval();
  const auto schema = module_.get_method("forward").function().getSchema();
  for (const auto& argument : schema.arguments()) {
    input_names_.push_back(argument.name());
  }
}

torch::Tensor ActRuntime::forward(const ObservationSample& sample) {
  std::vector<torch::Tensor> images;
  images.reserve(sample.images.size());
  for (const auto& image : sample.images) {
    auto prepared = image.to(torch::kFloat32);
    if (prepared.dim() == 3) {
      prepared = prepared.unsqueeze(0);
    }
    if (device_ == "cuda") {
      prepared = prepared.to(torch::kCUDA);
    }
    images.push_back(prepared);
  }
  auto state = sample.state.to(torch::kFloat32);
  if (state.dim() == 1) {
    state = state.unsqueeze(0);
  }
  if (device_ == "cuda") {
    state = state.to(torch::kCUDA);
  }

  std::vector<torch::jit::IValue> inputs;
  inputs.emplace_back(images);
  inputs.emplace_back(state);
  const torch::Tensor output = module_.forward(inputs).toTensor();
  if (!all_finite(output)) {
    throw std::runtime_error("act_runtime: forward produced non-finite actions");
  }
  return output.detach().to(torch::kCPU);
}

LatencyStats ActRuntime::benchmark(const ObservationSample& sample, int warmup, int iterations) {
  LatencyStats stats;
  stats.warmup = warmup;
  stats.iterations = iterations;
  if (warmup <= 0 || iterations <= 0) {
    return stats;
  }
  if (device_ == "cuda") {
    torch::cuda::synchronize();
  }
  for (int i = 0; i < warmup; ++i) {
    forward(sample);
  }
  if (device_ == "cuda") {
    torch::cuda::synchronize();
  }
  std::vector<double> latencies;
  latencies.reserve(static_cast<size_t>(iterations));
  for (int i = 0; i < iterations; ++i) {
    const auto start = std::chrono::steady_clock::now();
    forward(sample);
    if (device_ == "cuda") {
      torch::cuda::synchronize();
    }
    latencies.push_back(
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
  }
  std::sort(latencies.begin(), latencies.end());
  double sum = 0.0;
  for (const double value : latencies) {
    sum += value;
  }
  stats.mean_ms = sum / static_cast<double>(latencies.size());
  stats.min_ms = latencies.front();
  stats.max_ms = latencies.back();
  stats.p50_ms = latencies[latencies.size() / 2];
  stats.p95_ms = latencies[std::min(latencies.size() - 1, static_cast<size_t>(latencies.size() * 0.95))];
  return stats;
}

std::vector<ObservationSample> load_observation_samples(const std::filesystem::path& samples_dir,
                                                        int max_samples) {
  const std::filesystem::path meta_path = samples_dir / "samples_meta.json";
  if (!std::filesystem::exists(meta_path)) {
    throw std::runtime_error("act_runtime: missing " + meta_path.string());
  }
  const Json meta = Json::parse_file(meta_path.string());
  std::vector<std::string> camera_keys;
  for (const auto& item : meta.at("camera_keys").items()) {
    camera_keys.push_back(item.as_string());
  }
  const long long count = meta.int_or_throw("count");
  std::vector<ObservationSample> samples;
  for (long long index = 0; index < count; ++index) {
    if (max_samples > 0 && static_cast<int>(samples.size()) >= max_samples) {
      break;
    }
    std::ostringstream prefix;
    prefix << "sample_" << std::setfill('0') << std::setw(3) << index;
    ObservationSample sample;
    sample.sample_id = prefix.str();
    bool complete = true;
    for (size_t cam = 0; cam < camera_keys.size(); ++cam) {
      const std::filesystem::path image_path =
          samples_dir / (prefix.str() + "_cam" + std::to_string(cam) + ".npy");
      if (!std::filesystem::exists(image_path)) {
        complete = false;
        break;
      }
      sample.images.push_back(npy_to_tensor(image_path));
    }
    const std::filesystem::path state_path = samples_dir / (prefix.str() + "_state.npy");
    if (!std::filesystem::exists(state_path)) {
      complete = false;
    }
    if (!complete) {
      continue;
    }
    sample.state = npy_to_tensor(state_path);
    samples.push_back(std::move(sample));
  }
  return samples;
}

}  // namespace actlab

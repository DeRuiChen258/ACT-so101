#include "actlab/tensor_utils.hpp"

#include "actlab/path_utils.hpp"

#include <c10/cuda/CUDACachingAllocator.h>
#include <cuda_runtime_api.h>
#include <torch/csrc/Export.h>

#include <fstream>
#include <sstream>
#include <stdexcept>

namespace actlab {

void assert_shape(const torch::Tensor& tensor, const std::vector<int64_t>& expected,
                  const std::string& what) {
  if (tensor.dim() != static_cast<int64_t>(expected.size())) {
    throw std::runtime_error("shape assertion failed for " + what + ": expected rank " +
                             std::to_string(expected.size()) + ", got " + tensor_shape_string(tensor));
  }
  for (size_t i = 0; i < expected.size(); ++i) {
    if (tensor.size(static_cast<int64_t>(i)) != expected[i]) {
      throw std::runtime_error("shape assertion failed for " + what + ": expected " +
                               std::to_string(expected[i]) + " at dim " + std::to_string(i) +
                               ", got " + tensor_shape_string(tensor));
    }
  }
}

bool all_finite(const torch::Tensor& tensor) {
  return torch::isfinite(tensor).all().item<bool>();
}

std::string tensor_shape_string(const torch::Tensor& tensor) {
  std::ostringstream out;
  out << "(";
  for (int64_t i = 0; i < tensor.dim(); ++i) {
    if (i > 0) out << ", ";
    out << tensor.size(i);
  }
  out << ")";
  return out.str();
}

Json tensor_stats(const torch::Tensor& tensor) {
  Json stats = Json::make_object();
  Json shape = Json::make_array();
  for (int64_t i = 0; i < tensor.dim(); ++i) {
    shape.push(static_cast<double>(tensor.size(i)));
  }
  stats.set("shape", shape);
  stats.set("dtype", std::string(c10::toString(tensor.scalar_type())));
  stats.set("finite", all_finite(tensor));
  const auto detached = tensor.detach().to(torch::kCPU);
  const double min_value = detached.min().item<double>();
  const double max_value = detached.max().item<double>();
  const double mean_value = detached.mean().item<double>();
  stats.set("min", min_value);
  stats.set("max", max_value);
  stats.set("mean", mean_value);
  return stats;
}

void write_tensor_csv(const torch::Tensor& tensor, const std::filesystem::path& path) {
  ensure_dir(path.parent_path());
  std::ofstream out(path, std::ios::trunc);
  if (!out) {
    throw std::runtime_error("write_tensor_csv: cannot write " + path.string());
  }
  torch::Tensor cpu = tensor.detach().to(torch::kCPU).to(torch::kFloat64);
  if (cpu.dim() > 2) {
    cpu = cpu.reshape({cpu.size(0), -1});
  }
  if (cpu.dim() == 1) {
    cpu = cpu.unsqueeze(0);
  }
  const auto accessor = cpu.accessor<double, 2>();
  for (int64_t row = 0; row < cpu.size(0); ++row) {
    for (int64_t col = 0; col < cpu.size(1); ++col) {
      if (col > 0) out << ",";
      out << accessor[row][col];
    }
    out << "\n";
  }
}

MemoryStats memory_stats() {
  MemoryStats stats;
  if (!torch::cuda::is_available()) {
    return stats;
  }
  const auto device_stats = c10::cuda::CUDACachingAllocator::getDeviceStats(0);
  stats.allocated_mib =
      static_cast<double>(device_stats.allocated_bytes[0].current) / 1024.0 / 1024.0;
  stats.reserved_mib =
      static_cast<double>(device_stats.reserved_bytes[0].current) / 1024.0 / 1024.0;
  stats.peak_mib = static_cast<double>(device_stats.allocated_bytes[0].peak) / 1024.0 / 1024.0;
  return stats;
}

void reset_peak_memory() {
  if (!torch::cuda::is_available()) {
    return;
  }
  try {
    c10::cuda::CUDACachingAllocator::resetPeakStats(0);
  } catch (const std::exception&) {
    // 允许在 CUDA context 尚未建立时失败：peak 值仍可用，只是包含更早的分配
  }
}

double tensor_max_abs_diff(const torch::Tensor& lhs, const torch::Tensor& rhs) {
  if (lhs.sizes() != rhs.sizes()) {
    throw std::runtime_error("tensor_max_abs_diff: shape mismatch " + tensor_shape_string(lhs) +
                             " vs " + tensor_shape_string(rhs));
  }
  return (lhs.detach().to(torch::kCPU) - rhs.detach().to(torch::kCPU)).abs().max().item<double>();
}

}  // namespace actlab

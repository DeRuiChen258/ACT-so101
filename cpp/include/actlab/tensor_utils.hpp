// torch tensor 与 JSON/CSV 互转、shape 断言、NaN/Inf 检查、显存统计
#pragma once

#include "actlab/json_min.hpp"

#include <torch/torch.h>

#include <filesystem>
#include <string>
#include <vector>

namespace actlab {

struct MemoryStats {
  double allocated_mib = 0.0;
  double reserved_mib = 0.0;
  double peak_mib = 0.0;
};

// 断言 shape 完全一致，否则抛 std::runtime_error
void assert_shape(const torch::Tensor& tensor, const std::vector<int64_t>& expected,
                  const std::string& what);

// 全部元素有限（无 NaN/Inf）
bool all_finite(const torch::Tensor& tensor);

std::string tensor_shape_string(const torch::Tensor& tensor);

// {"shape": [...], "dtype": "...", "finite": true, "min": x, "max": x, "mean": x}
Json tensor_stats(const torch::Tensor& tensor);

// 二维以下张量写 CSV；高维张量会展平为 (n, -1)
void write_tensor_csv(const torch::Tensor& tensor, const std::filesystem::path& path);

MemoryStats memory_stats();
void reset_peak_memory();
double tensor_max_abs_diff(const torch::Tensor& lhs, const torch::Tensor& rhs);

}  // namespace actlab

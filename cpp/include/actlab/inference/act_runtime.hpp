// C++ 推理核心：加载 TorchScript、设备选择、样本准备、批量与单样本前向、动作序列输出、延时统计
//
// 归一化契约：图像/状态归一化与动作反归一化全部由导出的 TorchScript 模块内部完成
// （导出时打包了 checkpoint 的 normalization statistics），C++ 侧不做任何手写归一化，
// 只负责把数据搬上设备并校验数值有限性（见 Prompt §2.4）。
#pragma once

#include <torch/script.h>
#include <torch/torch.h>

#include <filesystem>
#include <string>
#include <vector>

namespace actlab {

struct ObservationSample {
  std::vector<torch::Tensor> images;  // 每个相机一个 (1, C, H, W) float32
  torch::Tensor state;                // (1, state_dim) float32
  std::string sample_id;
};

struct LatencyStats {
  int warmup = 0;
  int iterations = 0;
  double mean_ms = 0.0;
  double p50_ms = 0.0;
  double p95_ms = 0.0;
  double min_ms = 0.0;
  double max_ms = 0.0;
};

class ActRuntime {
 public:
  ActRuntime(const std::filesystem::path& torchscript_path, const std::string& device);

  const std::vector<std::string>& input_names() const { return input_names_; }

  // 单样本前向（返回 (chunk_size, action_dim) 的真实动作序列）
  torch::Tensor forward(const ObservationSample& sample);

  // 延时统计：warmup 后重复 iterations 次
  LatencyStats benchmark(const ObservationSample& sample, int warmup, int iterations);

  const std::string& device() const { return device_; }

 private:
  torch::jit::script::Module module_;
  std::string device_;
  std::vector<std::string> input_names_;
};

// 读取受控脚本导出的样本目录（samples_meta.json + sample_XXX_camN/state/action.npy）
std::vector<ObservationSample> load_observation_samples(const std::filesystem::path& samples_dir,
                                                        int max_samples);

}  // namespace actlab

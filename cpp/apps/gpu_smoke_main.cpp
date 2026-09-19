// gpu_smoke：验证 torch::cuda::is_available()、capability，执行一次 GPU matmul 并统计显存
#include "actlab/gpu_check.hpp"
#include "actlab/tensor_utils.hpp"

#include <c10/cuda/CUDAFunctions.h>
#include <torch/torch.h>

#include <iostream>

int main() {
  try {
    const actlab::GpuInfo info = actlab::probe_gpu();
    std::cout << actlab::render_gpu_markdown(info) << std::endl;
    if (!info.cuda_available || info.device_count <= 0) {
      std::cerr << "gpu_smoke: CUDA unavailable" << std::endl;
      return 1;
    }
    c10::cuda::set_device(0);
    const auto options =
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::Device(torch::kCUDA, 0));
    // 先用一次极小分配建立 CUDA context，再重置峰值统计，保证 peak 只覆盖被测区间
    { auto warmup = torch::zeros({1}, options); (void)warmup; }
    actlab::reset_peak_memory();
    auto a = torch::rand({1024, 1024}, options);
    auto b = torch::rand({1024, 1024}, options);
    auto c = torch::matmul(a, b);
    torch::cuda::synchronize();
    const actlab::MemoryStats stats = actlab::memory_stats();
    std::cout << "matmul[1024x1024] finite=" << (actlab::all_finite(c) ? "true" : "false")
              << " checksum=" << c.sum().item<double>() << std::endl;
    std::cout << "memory: allocated=" << stats.allocated_mib << " MiB reserved=" << stats.reserved_mib
              << " MiB peak=" << stats.peak_mib << " MiB" << std::endl;
    if (info.capability_major != 12 || info.capability_minor != 0) {
      std::cerr << "gpu_smoke: unexpected capability " << info.capability_major << "."
                << info.capability_minor << std::endl;
      return 2;
    }
    std::cout << "GPU_SMOKE_PASS" << std::endl;
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "gpu_smoke: exception: " << error.what() << std::endl;
    return 3;
  }
}

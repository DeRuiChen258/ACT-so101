// GPU/CUDA 能力探测（LibTorch 侧）：设备、capability、显存、驱动与编译版本
//
// 说明（诚实性约束）：C++ 侧 LibTorch 未暴露 torch.cuda.get_arch_list()，
// 因此"sm_120 可用"在 C++ 侧以**功能性证据**给出：在同一进程内执行一次 CUDA matmul，
// 若二进制缺少该架构的内核会在运行时报 "no kernel image is available"。
// 架构列表（arch_list）由训练侧 Python 环境在 S0 日志中给出，两者共同构成证据。
#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace actlab {

struct GpuInfo {
  bool cuda_available = false;
  int device_count = 0;
  std::string device_name;
  int capability_major = -1;
  int capability_minor = -1;
  std::size_t total_memory_bytes = 0;
  std::size_t free_memory_bytes = 0;
  std::string torch_version;
  std::string cuda_runtime_version;  // 编译期 CUDA runtime 头版本
  std::string libtorch_build_version;  // 来自 <libtorch>/build-version（由调用方填充）
  std::string torch_cxx_abi;         // _GLIBCXX_USE_CXX11_ABI
  std::string driver_version;        // 取自 nvidia-smi（真实驱动）
  std::string cuda_umd_version;

  std::string summary() const;
};

GpuInfo probe_gpu();
std::string cuda_arch_list_string();  // 由调用方从训练侧 Python 日志读取，此处仅占位说明
std::string render_gpu_markdown(const GpuInfo& info);

}  // namespace actlab

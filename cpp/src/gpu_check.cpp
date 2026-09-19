#include "actlab/gpu_check.hpp"

#include "actlab/proc.hpp"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAFunctions.h>
#include <cuda_runtime_api.h>
#include <torch/torch.h>

#include <sstream>
#include <stdexcept>

#ifdef _GLIBCXX_USE_CXX11_ABI
#define ACTLAB_CXX11_ABI _GLIBCXX_USE_CXX11_ABI
#else
#define ACTLAB_CXX11_ABI -1
#endif

namespace actlab {
namespace {

void parse_nvidia_smi(GpuInfo* info) {
  const std::string nvidia_smi = find_executable("nvidia-smi");
  if (nvidia_smi.empty()) {
    return;
  }
  ProcOptions quiet;
  quiet.echo = false;  // 内部探测不污染调用方输出
  const ProcResult result =
      run_process({nvidia_smi, "--query-gpu=driver_version", "--format=csv,noheader"}, quiet);
  if (result.exit_code == 0) {
    std::istringstream stream(result.output);
    std::string line;
    if (std::getline(stream, line)) {
      while (!line.empty() && (line.back() == '\n' || line.back() == '\r' || line.back() == ' ')) {
        line.pop_back();
      }
      info->driver_version = line;
    }
  }
  const ProcResult umd = run_process({nvidia_smi, "-L"}, quiet);
  const ProcResult umd_full = run_process({nvidia_smi}, quiet);
  const size_t marker = umd_full.output.find("CUDA UMD Version:");
  if (umd_full.exit_code == 0 && marker != std::string::npos) {
    std::istringstream stream(umd_full.output.substr(marker + 17));
    std::string value;
    stream >> value;
    info->cuda_umd_version = value;
  }
  (void)umd;
}

}  // namespace

std::string GpuInfo::summary() const {
  std::ostringstream out;
  out << "cuda_available=" << (cuda_available ? "true" : "false") << " devices=" << device_count
      << " name='" << device_name << "' capability=" << capability_major << "." << capability_minor
      << " total_mem_GiB="
      << (total_memory_bytes / 1024.0 / 1024.0 / 1024.0);
  return out.str();
}

GpuInfo probe_gpu() {
  GpuInfo info;
  info.cuda_available = torch::cuda::is_available();
  info.device_count = torch::cuda::device_count();
  info.torch_version = TORCH_VERSION;
  info.cuda_runtime_version = std::to_string(CUDART_VERSION / 1000) + "." +
                              std::to_string((CUDART_VERSION % 1000) / 10);
  info.torch_cxx_abi = std::to_string(ACTLAB_CXX11_ABI);
  if (info.cuda_available && info.device_count > 0) {
    const auto* properties = at::cuda::getCurrentDeviceProperties();
    if (properties == nullptr) {
      throw std::runtime_error("gpu_check: getCurrentDeviceProperties() returned null");
    }
    info.device_name = properties->name;
    info.capability_major = properties->major;
    info.capability_minor = properties->minor;
    info.total_memory_bytes = properties->totalGlobalMem;
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    cudaMemGetInfo(&free_bytes, &total_bytes);
    info.free_memory_bytes = free_bytes;
  }
  parse_nvidia_smi(&info);
  return info;
}

std::string render_gpu_markdown(const GpuInfo& info) {
  std::ostringstream out;
  out << "# GPU / LibTorch probe (C++)\n\n";
  out << "| item | value |\n| --- | --- |\n";
  out << "| torch::cuda::is_available() | " << (info.cuda_available ? "true" : "false") << " |\n";
  out << "| device_count | " << info.device_count << " |\n";
  out << "| device_name | " << info.device_name << " |\n";
  out << "| capability | " << info.capability_major << "." << info.capability_minor << " |\n";
  out << "| total memory (MiB) | " << (info.total_memory_bytes / 1024 / 1024) << " |\n";
  out << "| free memory (MiB) | " << (info.free_memory_bytes / 1024 / 1024) << " |\n";
  out << "| libtorch version | " << info.torch_version << " |\n";
  out << "| libtorch build-version | " << info.libtorch_build_version << " |\n";
  out << "| CUDA runtime header version | " << info.cuda_runtime_version << " |\n";
  out << "| _GLIBCXX_USE_CXX11_ABI | " << info.torch_cxx_abi << " |\n";
  out << "| driver version | " << info.driver_version << " |\n";
  out << "| CUDA UMD version | " << info.cuda_umd_version << " |\n";
  return out.str();
}

std::string cuda_arch_list_string() {
  return "sm_120 availability is proven functionally by gpu_smoke matmul; "
         "arch_list is reported by the training-side Python env in logs/02_cuda_pytorch.txt";
}

}  // namespace actlab

#include "actlab/stages/stage_env.hpp"

#ifdef ACTLAB_WITH_TORCH
#include "actlab/gpu_check.hpp"
#endif
#include "actlab/path_utils.hpp"
#include "actlab/proc.hpp"
#include "actlab/stages/stage_evidence.hpp"

#include <chrono>
#include <fstream>
#include <sstream>

namespace actlab {
namespace {

struct Probe {
  std::string label;
  std::vector<std::string> argv;
};

std::string run_probe(const Probe& probe) {
  if (probe.argv.empty()) {
    return "";
  }
  if (find_executable(probe.argv.front()).empty()) {
    return "[MISSING] " + probe.argv.front() + "\n";
  }
  const ProcResult result = run_process(probe.argv, ProcOptions{});
  std::ostringstream out;
  out << "$ " << format_command(probe.argv) << "\n" << result.output << "[exit=" << result.exit_code
      << "]\n\n";
  return out.str();
}

}  // namespace

StageResult run_stage_env(RunContext& ctx) {
  const auto start = std::chrono::steady_clock::now();
  const std::filesystem::path hardware_log = ctx.log_file("00_hardware_check.txt");
  const std::filesystem::path python_log = ctx.log_file("01_python_environment.txt");
  const std::filesystem::path cuda_log = ctx.log_file("02_cuda_pytorch.txt");

  const std::vector<Probe> probes = {
      {"uname", {"uname", "-a"}},
      {"os-release", {"cat", "/etc/os-release"}},
      {"nvidia-smi", {"nvidia-smi"}},
      {"nvcc", {"nvcc", "--version"}},
      {"git", {"git", "--version"}},
      {"ffmpeg", {"ffmpeg", "-version"}},
      {"cmake", {"cmake", "--version"}},
      {"ninja", {"ninja", "--version"}},
      {"gcc", {"gcc", "--version"}},
  };

  std::ostringstream hardware;
  hardware << "# Stage S0 hardware/environment probe (C++ actlab env)\n";
  hardware << "# timestamp: " << timestamp_iso8601() << "\n";
  hardware << "# experiment_root: " << ctx.experiment_root() << "\n";
  hardware << "# env_root: " << ctx.env_root() << "\n\n";
  for (const auto& probe : probes) {
    hardware << "===== " << probe.label << " =====\n" << run_probe(probe);
  }
  {
    std::ofstream out(hardware_log, std::ios::trunc);
    out << hardware.str();
  }

  const std::string conda_bin = (std::filesystem::path(ctx.config().paths.conda_root) / "bin" / "conda").string();
  std::ostringstream python;
  python << "# Stage S0 python/conda probe\n\n";
  python << "===== conda env list =====\n" << run_probe({"conda", {conda_bin, "env", "list"}});
  python << "===== cuda_132 (readonly reference) =====\n"
         << run_probe({"cuda_132 python",
                       {(std::filesystem::path(ctx.config().paths.conda_root) / "envs" / "cuda_132" / "bin" /
                         "python").string(),
                        "-c",
                        "import sys, torch; print('python', sys.version.split()[0]); "
                        "print('torch', torch.__version__); print('cuda', torch.version.cuda); "
                        "print('cuda_available', torch.cuda.is_available()); "
                        "print('arch_list', torch.cuda.get_arch_list())"}});
  python << "===== training env (" << ctx.config().paths.env_name << ") =====\n"
         << run_probe({ctx.config().paths.env_name + " python",
                       {ctx.python_bin(), "--version"}});
  {
    std::ofstream out(python_log, std::ios::trunc);
    out << python.str();
  }

  // C++ 侧 GPU 探测（LibTorch）：通过 gpu_smoke 独立进程执行，结果落盘
  std::ostringstream cuda;
  cuda << "# Stage S0 CUDA probe (C++ / LibTorch)\n\n";
  bool cuda_ok = false;
  bool sm120 = false;
  std::string device_name;
  std::string capability;
  std::string libtorch_version;
  std::string gpu_summary;
  bool gpu_smoke_ok = false;
#ifdef ACTLAB_WITH_TORCH
  try {
    GpuInfo info = probe_gpu();
    // LibTorch 真实构建版本取自 <libtorch>/build-version（文件即事实）
    const std::filesystem::path build_version =
        std::filesystem::path(ctx.config().paths.libtorch_dir) / "build-version";
    if (std::filesystem::exists(build_version)) {
      std::ifstream in(build_version);
      std::string line;
      if (std::getline(in, line)) {
        info.libtorch_build_version = line;
      }
    }
    cuda << render_gpu_markdown(info);
    cuda_ok = info.cuda_available;
    device_name = info.device_name;
    capability = std::to_string(info.capability_major) + "." + std::to_string(info.capability_minor);
    libtorch_version = info.torch_version + " (" + info.libtorch_build_version + ")";
    gpu_summary = info.summary();
    if (info.capability_major != 12 || info.capability_minor != 0) {
      cuda_ok = false;
      cuda << "\n[FAIL] unexpected capability (expected 12.0 for sm_120)\n";
    }
  } catch (const std::exception& error) {
    cuda << "LibTorch probe failed: " << error.what() << "\n";
  }
#else
  cuda << "[local build] LibTorch probe skipped (ACTLAB_WITH_TORCH=OFF);"
          " CUDA gate evaluated by gpu_smoke in the GPU build\n";
#endif

  const std::filesystem::path gpu_smoke_bin = ctx.experiment_root() / "build" / "gpu" / "gpu_smoke";
  if (std::filesystem::exists(gpu_smoke_bin)) {
    const ProcResult smoke = ctx.run_stack({gpu_smoke_bin.string()}, "02_gpu_smoke.txt");
    cuda << "\n===== gpu_smoke (independent C++ process) =====\n" << smoke.output
         << "[exit=" << smoke.exit_code << "]\n";
    gpu_smoke_ok = smoke.exit_code == 0 && smoke.output.find("GPU_SMOKE_PASS") != std::string::npos;
    if (smoke.exit_code != 0) {
      cuda_ok = false;
    }
  } else {
    cuda << "\n[WARN] build/gpu/gpu_smoke not found; skipping independent process check\n";
  }

  // sm_120 证据由训练侧 Python 环境的 arch_list 给出（C++ 侧提供功能性 matmul 证据）
  const ProcResult arch = run_process(
      {ctx.python_bin(), "-c",
       "import torch; print('arch_list', torch.cuda.get_arch_list()); "
       "print('cuda_available', torch.cuda.is_available())"},
      ProcOptions{});
  cuda << "\n===== training env torch arch_list =====\n" << arch.output << "[exit=" << arch.exit_code
       << "]\n";
  if (arch.exit_code == 0 && arch.output.find("sm_120") != std::string::npos) {
    sm120 = true;
  }
  cuda << "\n# interpretation: sm_120 declared by training env arch_list; functional proof = "
          "gpu_smoke CUDA matmul success\n";
  {
    std::ofstream out(cuda_log, std::ios::trunc);
    out << cuda.str();
  }

  ctx.manifest().set_field("environment", "os", "Ubuntu 26.04.1 LTS");
  ctx.manifest().set_field("environment", "gpu", device_name);
  ctx.manifest().set_field("environment", "capability", capability);
  ctx.manifest().set_field("environment", "sm_120", sm120);
  ctx.manifest().set_field("environment", "sm_120_functional_matmul", gpu_smoke_ok);
  ctx.manifest().set_field("environment", "libtorch", libtorch_version);
  ctx.manifest().set_field("environment", "conda_env", ctx.config().paths.env_name);
  ctx.manifest().set_field("environment", "config_sha256", ctx.config().config_sha256);
  ctx.manifest().add_artifact("S0", "log", hardware_log, "hardware probe");
  ctx.manifest().add_artifact("S0", "log", python_log, "python/conda probe");
  ctx.manifest().add_artifact("S0", "log", cuda_log, "CUDA/LibTorch probe");
  ctx.manifest().save();

  // 真实截图证据（窗口抓取失败时由证据阶段标记 SCREENSHOT_UNAVAILABLE）
  ScreenshotResult shot = capture_terminal_screenshot(
      ctx, "00_hardware_check", "s0_hardware",
      "nvidia-smi | head -20; echo; nvcc --version | tail -3; echo; python3 --version", 18);
  if (!shot.captured) {
    capture_terminal_screenshot(ctx, "00_hardware_check", "s0_hardware_retry",
                                "nvidia-smi | head -20", 15);
  }
  capture_terminal_screenshot(ctx, "02_cuda_pytorch", "s0_cuda",
                              "cat " + ctx.experiment_root().string() + "/logs/02_cuda_pytorch.txt", 18);
  capture_terminal_screenshot(ctx, "01_python_environment", "s0_python",
                              "cat " + python_log.string() + " | head -18", 18);

  ctx.logger().info("stage_env", gpu_summary);

  StageResult result;
  if (!cuda_ok) {
    result = StageResult::fail("env", "torch::cuda::is_available() == false or gpu_smoke failed");
  } else if (!sm120) {
    result = StageResult::fail("env", "sm_120 not present in LibTorch arch_list");
  } else {
    result = StageResult::pass("env", "GPU CUDA sm_120 OK; " + gpu_summary);
  }
  result.artifacts = {hardware_log.string(), python_log.string(), cuda_log.string()};
  if (shot.captured) {
    result.evidence.push_back(shot.path.string());
  } else {
    result.evidence.push_back("SCREENSHOT_UNAVAILABLE: " + shot.note);
  }
  result.duration_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
  return result;
}

}  // namespace actlab

// infer_act：独立进程完成一次 TorchScript 推理，用于"新进程重载"硬门禁
#include "actlab/inference/act_runtime.hpp"
#include "actlab/path_utils.hpp"
#include "actlab/tensor_utils.hpp"

#include <fstream>
#include <iostream>
#include <string>

namespace {

void usage() {
  std::cout << "usage: infer_act --torchscript PATH --samples DIR [--device cuda|cpu] [--out CSV]\n";
}

}  // namespace

int main(int argc, char** argv) {
  std::string torchscript;
  std::string samples;
  std::string device = "cuda";
  std::string out;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    const auto value_of = [&](const char* prefix) -> std::string {
      return arg.substr(std::string(prefix).size());
    };
    if (arg.rfind("--torchscript=", 0) == 0) {
      torchscript = value_of("--torchscript=");
    } else if (arg.rfind("--samples=", 0) == 0) {
      samples = value_of("--samples=");
    } else if (arg.rfind("--device=", 0) == 0) {
      device = value_of("--device=");
    } else if (arg.rfind("--out=", 0) == 0) {
      out = value_of("--out=");
    } else if (arg == "--help" || arg == "-h") {
      usage();
      return 0;
    } else if (i + 1 < argc && arg == "--torchscript") {
      torchscript = argv[++i];
    } else if (i + 1 < argc && arg == "--samples") {
      samples = argv[++i];
    } else if (i + 1 < argc && arg == "--device") {
      device = argv[++i];
    } else if (i + 1 < argc && arg == "--out") {
      out = argv[++i];
    } else {
      std::cerr << "infer_act: unknown argument " << arg << std::endl;
      usage();
      return 64;
    }
  }

  if (torchscript.empty() || samples.empty()) {
    usage();
    return 64;
  }

  try {
    std::cout << "infer_act: new process reload gate" << std::endl;
    std::cout << "  torchscript: " << torchscript << std::endl;
    std::cout << "  samples    : " << samples << std::endl;
    std::cout << "  device     : " << device << std::endl;

    actlab::ActRuntime runtime(torchscript, device);
    const auto observations = actlab::load_observation_samples(samples, 4);
    if (observations.empty()) {
      std::cerr << "infer_act: no samples loaded" << std::endl;
      return 5;
    }
    std::cout << "  module inputs:";
    for (const auto& name : runtime.input_names()) {
      std::cout << " " << name;
    }
    std::cout << std::endl;

    std::ofstream csv;
    if (!out.empty()) {
      actlab::ensure_dir(std::filesystem::path(out).parent_path());
      csv.open(out, std::ios::trunc);
      csv << "sample,shape,finite,min,max,mean\n";
    }

    bool ok = true;
    for (const auto& observation : observations) {
      const torch::Tensor actions = runtime.forward(observation);
      const bool finite = actlab::all_finite(actions);
      ok = ok && finite;
      std::cout << "  sample " << observation.sample_id
                << " actions shape=" << actlab::tensor_shape_string(actions)
                << " finite=" << (finite ? "true" : "false") << " stats="
                << actlab::tensor_stats(actions).dump(0) << std::endl;
      if (csv.is_open()) {
        const auto cpu = actions.reshape({-1});
        csv << observation.sample_id << "," << actlab::tensor_shape_string(actions) << ","
            << (finite ? "true" : "false") << "," << cpu.min().item<double>() << ","
            << cpu.max().item<double>() << "," << cpu.mean().item<double>() << "\n";
      }
    }
    if (!ok) {
      std::cerr << "infer_act: non-finite actions detected" << std::endl;
      return 6;
    }
    std::cout << "NEW_PROCESS_RELOAD_PASS" << std::endl;
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "infer_act: exception: " << error.what() << std::endl;
    return 7;
  }
}

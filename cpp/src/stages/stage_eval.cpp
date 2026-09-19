#include "actlab/stages/stage_eval.hpp"

#include "actlab/inference/act_runtime.hpp"
#include "actlab/inference/temporal_ensemble.hpp"
#include "actlab/json_min.hpp"
#include "actlab/npy_io.hpp"
#include "actlab/path_utils.hpp"
#include "actlab/proc.hpp"
#include "actlab/stages/stage_evidence.hpp"
#include "actlab/tensor_utils.hpp"

#include <chrono>
#include <cmath>
#include <fstream>
#include <sstream>
#include <vector>

namespace actlab {
namespace {

struct EvalRow {
  std::string sample_id;
  double mae = 0.0;
  double mse = 0.0;
  double first_action_l2 = 0.0;
};

std::filesystem::path resolve_torchscript(const RunContext& ctx) {
  return ctx.resolve(ctx.config().paths.artifacts_dir) / ctx.config().act.job_name /
         ctx.config().paths.torchscript_rel / "act_policy.pt";
}

// 读取 Python 参考实现输出（export 阶段落盘），用于一致性比对
Json load_reference(const RunContext& ctx) {
  const std::filesystem::path path = ctx.resolve(ctx.config().paths.artifacts_dir) /
                                     ctx.config().act.job_name / "reference_outputs.json";
  if (!std::filesystem::exists(path)) {
    return Json();
  }
  return Json::parse_file(path.string());
}

std::string render_eval_summary(const std::vector<EvalRow>& rows, const LatencyStats& latency,
                                const Json& reference, double max_diff, bool has_reference) {
  std::ostringstream out;
  out << "# Offline Evaluation Summary\n\n";
  out << "> 评测类型：**Offline Evaluation**（数据集回放 + 离线动作误差）；"
         "仿真成功率/真机成功率：`NOT_MEASURED`（无真机、无仿真场景接入）。\n\n";
  out << "## C++ 推理延时（warmup=" << latency.warmup << ", iterations=" << latency.iterations << "）\n\n";
  out << "| 指标 | 值 (ms) |\n| --- | --- |\n";
  out << "| mean | " << latency.mean_ms << " |\n";
  out << "| p50 | " << latency.p50_ms << " |\n";
  out << "| p95 | " << latency.p95_ms << " |\n";
  out << "| min | " << latency.min_ms << " |\n";
  out << "| max | " << latency.max_ms << " |\n\n";
  out << "## 与数据集专家动作的差异（" << rows.size() << " 个真实样本）\n\n";
  out << "| 样本 | MAE | MSE | 首步动作 L2 |\n| --- | --- | --- | --- |\n";
  double mae_sum = 0.0;
  double mse_sum = 0.0;
  for (const auto& row : rows) {
    out << "| " << row.sample_id << " | " << row.mae << " | " << row.mse << " | " << row.first_action_l2
        << " |\n";
    mae_sum += row.mae;
    mse_sum += row.mse;
  }
  if (!rows.empty()) {
    out << "| **平均** | **" << mae_sum / static_cast<double>(rows.size()) << "** | **"
        << mse_sum / static_cast<double>(rows.size()) << "** | - |\n";
  }
  out << "\n## 与 Python 参考实现的一致性\n\n";
  if (has_reference) {
    out << "- Python 参考实现输出已加载（" << reference.keys().size() << " 个样本）\n";
    out << "- C++ 与 Python 前向输出的最大绝对差：`" << max_diff << "`\n";
    out << "- 结论：" << (max_diff < 1e-3 ? "数值一致（同源归一化 + 同源权重）"
                                          : "存在差异，需检查设备/精度差异来源")
        << "\n";
  } else {
    out << "- `NOT_MEASURED`：未找到 reference_outputs.json（Python 参考未导出）\n";
  }
  return out.str();
}

}  // namespace

StageResult run_stage_infer(RunContext& ctx) {
  const auto start = std::chrono::steady_clock::now();
  const std::filesystem::path torchscript = resolve_torchscript(ctx);
  const std::filesystem::path samples_dir = ctx.experiment_root() / "artifacts" / "dataset_samples";
  const std::filesystem::path log = ctx.log_file("17_inference.txt");
  std::ostringstream text;
  text << "# Stage S14 C++ inference (primary path)\n# timestamp: " << timestamp_iso8601() << "\n";
  text << "torchscript: " << torchscript.string() << "\ndevice: " << ctx.options().device << "\n";

  if (!std::filesystem::exists(torchscript)) {
    text << "TorchScript not found — run stage verify first\n";
    std::ofstream out(log, std::ios::trunc);
    out << text.str();
    return StageResult::fail("infer", "TorchScript missing: " + torchscript.string());
  }

  ActRuntime runtime(torchscript, ctx.options().device);
  const auto samples = load_observation_samples(samples_dir, 8);
  if (samples.empty()) {
    return StageResult::fail("infer", "no observation samples under " + samples_dir.string());
  }
  text << "module inputs: ";
  for (const auto& name : runtime.input_names()) {
    text << name << " ";
  }
  text << "\nloaded samples: " << samples.size() << "\n";

  const torch::Tensor first = runtime.forward(samples.front());
  text << "forward output shape: " << tensor_shape_string(first) << "\n";
  text << "action stats: " << tensor_stats(first).dump(0) << "\n";
  const LatencyStats latency = runtime.benchmark(samples.front(), 5, 30);
  text << "latency mean/p50/p95/min/max (ms): " << latency.mean_ms << " / " << latency.p50_ms << " / "
       << latency.p95_ms << " / " << latency.min_ms << " / " << latency.max_ms << "\n";

  // 延时明细 CSV（真实多次采样）
  {
    std::ofstream csv(ctx.results_file("inference_latency.csv"), std::ios::trunc);
    csv << "phase,warmup,iterations,mean_ms,p50_ms,p95_ms,min_ms,max_ms\n";
    csv << "cxx_torchscript," << latency.warmup << "," << latency.iterations << "," << latency.mean_ms
        << "," << latency.p50_ms << "," << latency.p95_ms << "," << latency.min_ms << ","
        << latency.max_ms << "\n";
  }

  // Temporal ensembling 一致性自检（配置启用时按源码算法融合；未启用时记录 disabled）
  if (ctx.config().act.has_temporal_ensemble) {
    TemporalEnsembler ensembler(ctx.config().act.temporal_ensemble_coeff, first.size(1));
    const torch::Tensor fused = ensembler.update(first.unsqueeze(0));
    text << "temporal ensemble fused action shape: " << tensor_shape_string(fused)
         << " finite=" << (all_finite(fused) ? "true" : "false") << "\n";
  } else {
    text << "temporal ensemble: disabled (temporal_ensemble_coeff not set)\n";
  }

  {
    std::ofstream out(log, std::ios::trunc);
    out << text.str();
  }
  ctx.manifest().set_field("inference", "torchscript", torchscript.string());
  ctx.manifest().set_field("inference", "device", ctx.options().device);
  ctx.manifest().set_field("inference", "output_shape", tensor_shape_string(first));
  ctx.manifest().set_field("inference", "latency_mean_ms", latency.mean_ms);
  ctx.manifest().set_field("inference", "latency_p95_ms", latency.p95_ms);
  ctx.manifest().set_field("inference", "samples_used", static_cast<long long>(samples.size()));
  ctx.manifest().add_artifact("S14", "log", log, "C++ inference");
  ctx.manifest().add_artifact("S14", "metrics", ctx.results_file("inference_latency.csv"),
                              "latency measurements");
  ctx.manifest().save();

  const ScreenshotResult infer_shot = capture_terminal_screenshot(
      ctx, "17_inference", "s14_infer", "cat " + log.string() + " | tail -12", 18);

  StageResult result = StageResult::pass("infer", "shape=" + tensor_shape_string(first) + " mean=" +
                                                      std::to_string(latency.mean_ms) + "ms");
  result.artifacts = {log.string()};
  result.evidence.push_back(infer_shot.captured ? infer_shot.path.string()
                                                : "SCREENSHOT_UNAVAILABLE: " + infer_shot.note);
  result.duration_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
  return result;
}

StageResult run_stage_eval(RunContext& ctx) {
  const auto start = std::chrono::steady_clock::now();
  const std::filesystem::path torchscript = resolve_torchscript(ctx);
  const std::filesystem::path samples_dir = ctx.experiment_root() / "artifacts" / "dataset_samples";
  const std::filesystem::path log = ctx.log_file("18_evaluation.txt");
  if (!std::filesystem::exists(torchscript)) {
    return StageResult::fail("eval", "TorchScript missing; run verify first");
  }

  ActRuntime runtime(torchscript, ctx.options().device);
  const auto samples = load_observation_samples(samples_dir, 8);
  if (samples.empty()) {
    return StageResult::fail("eval", "no observation samples available");
  }

  std::vector<EvalRow> rows;
  double max_diff = 0.0;
  const Json reference = load_reference(ctx);
  const bool has_reference = reference.is_object() && reference.keys().size() > 0;
  std::size_t reference_index = 0;

  for (const auto& sample : samples) {
    const torch::Tensor predicted = runtime.forward(sample);  // (1, chunk, action_dim)
    const std::filesystem::path action_path = samples_dir / (sample.sample_id + "_action.npy");
    if (!std::filesystem::exists(action_path)) {
      continue;
    }
    const NpyArray action_array = read_npy(action_path);
    std::vector<int64_t> sizes;
    for (const auto dim : action_array.shape) {
      sizes.push_back(dim);
    }
    torch::Tensor expert =
        torch::from_blob(const_cast<uint8_t*>(action_array.data.data()), sizes,
                         torch::TensorOptions().dtype(torch::kFloat32))
            .clone();
    torch::Tensor flat_predicted = predicted.reshape({-1});
    torch::Tensor flat_expert = expert.reshape({-1});
    const int64_t overlap =
        std::min<int64_t>(flat_predicted.numel(), flat_expert.numel());
    flat_predicted = flat_predicted.slice(0, 0, overlap);
    flat_expert = flat_expert.slice(0, 0, overlap);
    EvalRow row;
    row.sample_id = sample.sample_id;
    const auto diff = (flat_predicted - flat_expert).abs();
    row.mae = diff.mean().item<double>();
    row.mse = (flat_predicted - flat_expert).pow(2).mean().item<double>();
    row.first_action_l2 = predicted.select(1, 0).norm().item<double>();
    rows.push_back(row);

    if (has_reference && reference_index < reference.keys().size()) {
      const std::string& key = reference.keys()[reference_index];
      const Json& values = reference.at(key);
      std::vector<double> reference_values;
      for (const auto& item : values.items()) {
        reference_values.push_back(item.as_number());
      }
      const int64_t count =
          std::min<int64_t>(static_cast<int64_t>(reference_values.size()), predicted.numel());
      auto cpu = predicted.reshape({-1}).slice(0, 0, count).to(torch::kFloat64);
      const auto accessor = cpu.accessor<double, 1>();
      for (int64_t i = 0; i < count; ++i) {
        max_diff = std::max(max_diff, std::fabs(accessor[i] - reference_values[static_cast<size_t>(i)]));
      }
      ++reference_index;
    }
  }

  const LatencyStats latency = runtime.benchmark(samples.front(), 5, 30);
  const std::string summary = render_eval_summary(rows, latency, reference, max_diff, has_reference);
  {
    std::ofstream out(ctx.results_file("eval_summary.md"), std::ios::trunc);
    out << summary;
  }
  {
    std::ofstream out(log, std::ios::trunc);
    out << summary;
  }
  ctx.manifest().set_field("evaluation", "type", "Offline Evaluation");
  ctx.manifest().set_field("evaluation", "samples", static_cast<long long>(rows.size()));
  ctx.manifest().set_field("evaluation", "python_cxx_max_abs_diff", max_diff);
  ctx.manifest().set_field("evaluation", "sim_success_rate", "NOT_MEASURED");
  ctx.manifest().set_field("evaluation", "real_robot_success_rate", "NOT_MEASURED");
  ctx.manifest().add_artifact("S15", "report", ctx.results_file("eval_summary.md"), "offline evaluation");
  ctx.manifest().save();

  if (rows.empty()) {
    return StageResult::fail("eval", "no samples with expert action chunks were evaluated");
  }
  const ScreenshotResult eval_shot = capture_terminal_screenshot(
      ctx, "18_evaluation", "s15_eval", "head -32 " + ctx.results_file("eval_summary.md").string(), 18);
  StageResult result = StageResult::pass(
      "eval", "Offline Evaluation on " + std::to_string(rows.size()) + " real samples; python_max_diff=" +
                  std::to_string(max_diff));
  result.artifacts = {log.string(), ctx.results_file("eval_summary.md").string()};
  result.evidence.push_back(eval_shot.captured ? eval_shot.path.string()
                                               : "SCREENSHOT_UNAVAILABLE: " + eval_shot.note);
  result.duration_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
  return result;
}

}  // namespace actlab

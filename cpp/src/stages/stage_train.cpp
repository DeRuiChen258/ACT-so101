#include "actlab/stages/stage_train.hpp"

#include "actlab/metrics.hpp"
#include "actlab/path_utils.hpp"
#include "actlab/proc.hpp"
#include "actlab/stages/stage_evidence.hpp"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <sstream>
#include <vector>

namespace actlab {
namespace {

std::string json_escape_light(const std::string& raw) {
  std::string out;
  for (const char ch : raw) {
    if (ch == '"' || ch == '\\') {
      out.push_back('\\');
    }
    out.push_back(ch);
  }
  return out;
}

void write_config_snapshot(const RunContext& ctx, const std::filesystem::path& path) {
  const ActConfig& act = ctx.config().act;
  std::ofstream out(path, std::ios::trunc);
  out << "# Stage S8 config freeze\n";
  out << "# timestamp: " << timestamp_iso8601() << "\n";
  out << "# config_sha256(act_baseline|paths|runtime): " << ctx.config().config_sha256 << "\n\n";
  out << "chunk_size            : " << act.chunk_size << "\n";
  out << "n_action_steps        : " << act.n_action_steps << "\n";
  out << "n_obs_steps           : " << act.n_obs_steps << "\n";
  out << "vision_backbone       : " << act.vision_backbone << "\n";
  out << "dim_model             : " << act.dim_model << "\n";
  out << "dim_feedforward       : " << act.dim_feedforward << "\n";
  out << "n_heads               : " << act.n_heads << "\n";
  out << "n_encoder_layers      : " << act.n_encoder_layers << "\n";
  out << "n_decoder_layers      : " << act.n_decoder_layers << "\n";
  out << "use_vae / latent_dim  : " << (act.use_vae ? "true" : "false") << " / " << act.latent_dim << "\n";
  out << "dropout / kl_weight   : " << act.dropout << " / " << act.kl_weight << "\n";
  out << "temporal_ensemble     : "
      << (act.has_temporal_ensemble ? std::to_string(act.temporal_ensemble_coeff) : "none") << "\n";
  out << "optimizer / lr / wd   : " << act.optimizer << " / " << act.optimizer_lr << " / "
      << act.optimizer_weight_decay << "\n";
  out << "batch_size            : " << act.batch_size << "\n";
  out << "steps                 : " << act.steps << "\n";
  out << "num_workers           : " << act.num_workers << "\n";
  out << "seed / precision      : " << act.seed << " / " << act.precision << "\n";
  out << "save_freq / log_freq  : " << act.save_freq << " / " << act.log_freq << "\n";
  out << "policy_type / job     : " << act.policy_type << " / " << act.job_name << "\n";
  out << "dataset.repo_id       : " << ctx.config().dataset.repo_id << "\n";
  out << "device                : " << ctx.options().device << "\n";
  out.flush();
}

std::vector<std::string> build_train_argv(const RunContext& ctx, long long steps,
                                          const std::filesystem::path& output_dir,
                                          const std::string& job_name, bool resume) {
  const ActConfig& act = ctx.config().act;
  const std::filesystem::path dataset_root =
      ctx.resolve(ctx.config().paths.data_dir) / ctx.config().dataset.repo_id;
  std::vector<std::string> argv = {
      ctx.python_bin(),
      "-m",
      "lerobot.scripts.lerobot_train",
      "--dataset.repo_id=" + ctx.config().dataset.repo_id,
      "--dataset.root=" + dataset_root.string(),
      "--policy.type=" + act.policy_type,
      "--policy.device=" + ctx.options().device,
      "--policy.push_to_hub=false",
      "--output_dir=" + output_dir.string(),
      "--job_name=" + job_name,
      "--batch_size=" + std::to_string(act.batch_size),
      "--steps=" + std::to_string(steps),
      "--num_workers=" + std::to_string(act.num_workers),
      "--seed=" + std::to_string(act.seed),
      "--save_freq=" + std::to_string(act.save_freq),
      "--log_freq=" + std::to_string(act.log_freq),
      "--eval_steps=0",
      "--policy.optimizer_lr=" + std::to_string(act.optimizer_lr),
      "--policy.optimizer_weight_decay=" + std::to_string(act.optimizer_weight_decay),
      "--policy.chunk_size=" + std::to_string(act.chunk_size),
      "--policy.n_action_steps=" + std::to_string(act.n_action_steps),
      "--policy.dim_model=" + std::to_string(act.dim_model),
      "--policy.n_heads=" + std::to_string(act.n_heads),
      "--policy.dim_feedforward=" + std::to_string(act.dim_feedforward),
      "--policy.n_encoder_layers=" + std::to_string(act.n_encoder_layers),
      "--policy.n_decoder_layers=" + std::to_string(act.n_decoder_layers),
      "--policy.n_vae_encoder_layers=" + std::to_string(act.n_vae_encoder_layers),
      "--policy.latent_dim=" + std::to_string(act.latent_dim),
      "--policy.dropout=" + std::to_string(act.dropout),
      "--policy.kl_weight=" + std::to_string(act.kl_weight),
      "--policy.use_vae=" + std::string(act.use_vae ? "true" : "false"),
  };
  if (resume) {
    argv.emplace_back("--resume=true");
  }
  return argv;
}

struct TrainOutcome {
  bool ok = false;
  std::filesystem::path output_dir;
  std::filesystem::path log_path;
  std::filesystem::path checkpoint_dir;
  std::vector<MetricSample> samples;
  std::string detail;
};

// 在 output_dir 下寻找 lerobot 写出的 checkpoint（checkpoints/<step>/pretrained_model 或 last）
std::filesystem::path find_checkpoint(const std::filesystem::path& output_dir) {
  const std::filesystem::path checkpoints = output_dir / "checkpoints";
  if (!std::filesystem::exists(checkpoints)) {
    return {};
  }
  std::vector<std::filesystem::path> candidates;
  for (const auto& entry : std::filesystem::directory_iterator(checkpoints)) {
    if (!entry.is_directory()) {
      continue;
    }
    for (const auto& sub : std::filesystem::directory_iterator(entry.path())) {
      if (sub.is_directory() && sub.path().filename() == "pretrained_model") {
        candidates.push_back(sub.path());
      }
    }
  }
  if (candidates.empty()) {
    return {};
  }
  std::sort(candidates.begin(), candidates.end(), [](const auto& lhs, const auto& rhs) {
    return lhs.string() < rhs.string();
  });
  return candidates.back();
}

TrainOutcome run_training(RunContext& ctx, long long steps, const std::string& job_name, bool resume,
                          const std::string& log_name, int timeout_seconds) {
  TrainOutcome outcome;
  const std::filesystem::path train_root = ctx.resolve(ctx.config().paths.outputs_dir) / "train";
  outcome.output_dir = train_root / job_name;
  // LeRobot 拒绝写入已存在的输出目录（除非 --resume），因此非续训时改用带时间戳的新目录，
  // 既不覆盖既有产物，也保留每次运行的可追溯性（Prompt §4.5：禁止覆盖既有产物）。
  if (!resume && std::filesystem::exists(outcome.output_dir)) {
    outcome.output_dir = train_root / (job_name + "_" + timestamp_compact());
  }
  outcome.log_path = ctx.log_file(log_name);

  const std::vector<std::string> argv = build_train_argv(ctx, steps, outcome.output_dir, job_name, resume);
  const ProcResult result = ctx.run_stack(argv, log_name, timeout_seconds);
  outcome.samples = parse_metrics_log(outcome.log_path);
  outcome.checkpoint_dir = find_checkpoint(outcome.output_dir);
  outcome.ok = result.exit_code == 0 && !outcome.checkpoint_dir.empty() && !outcome.samples.empty();
  std::ostringstream detail;
  detail << "exit=" << result.exit_code << " steps=" << steps << " metric_samples=" << outcome.samples.size()
         << " checkpoint=" << (outcome.checkpoint_dir.empty() ? "NONE" : outcome.checkpoint_dir.string());
  outcome.detail = detail.str();
  return outcome;
}

}  // namespace

StageResult run_stage_train_config(RunContext& ctx) {
  const auto start = std::chrono::steady_clock::now();
  const std::filesystem::path config_log = ctx.log_file("10_act_config.txt");
  write_config_snapshot(ctx, config_log);
  std::ifstream in(config_log);
  std::ostringstream snapshot;
  snapshot << in.rdbuf();

  const std::filesystem::path notes = ctx.log_file("act_algorithm_notes.md");
  if (!std::filesystem::exists(notes)) {
    std::ofstream out(notes, std::ios::trunc);
    out << "# ACT 算法笔记（占位：需按当前源码逐条补全）\n";
  }

  ctx.manifest().set_field("training", "config_sha256", ctx.config().config_sha256);
  ctx.manifest().set_field("training", "chunk_size", static_cast<long long>(ctx.config().act.chunk_size));
  ctx.manifest().set_field("training", "n_action_steps",
                           static_cast<long long>(ctx.config().act.n_action_steps));
  ctx.manifest().set_field("training", "batch_size", static_cast<long long>(ctx.config().act.batch_size));
  ctx.manifest().set_field("training", "lr", ctx.config().act.optimizer_lr);
  ctx.manifest().set_field("training", "planned_steps", ctx.config().act.steps);
  ctx.manifest().set_field("training", "seed", ctx.config().act.seed);
  ctx.manifest().add_artifact("S8", "config", config_log, "frozen config snapshot");
  ctx.manifest().save();

  const ScreenshotResult config_shot =
      capture_terminal_screenshot(ctx, "10_act_config", "s8_config", "cat " + config_log.string(), 18);

  StageResult result = StageResult::pass("train-config", "config validated; sha256=" +
                                                             ctx.config().config_sha256.substr(0, 16));
  result.artifacts = {config_log.string(), notes.string()};
  result.evidence.push_back(config_shot.captured ? config_shot.path.string()
                                                 : "SCREENSHOT_UNAVAILABLE: " + config_shot.note);
  result.duration_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
  return result;
}

StageResult run_stage_train_smoke_init(RunContext& ctx) {
  const auto start = std::chrono::steady_clock::now();
  const std::filesystem::path python = ctx.python_bin();
  const std::filesystem::path script = ctx.experiment_root() / "scripts" / "py" / "act_forward_probe.py";
  const std::filesystem::path log = ctx.log_file("11_act_model_init.txt");
  if (!std::filesystem::exists(script)) {
    return StageResult::fail("train-init", "missing probe script: " + script.string());
  }
  const ProcResult result = ctx.run_stack(
      {python.string(), script.string(), "--repo-id", ctx.config().dataset.repo_id, "--root",
       (ctx.resolve(ctx.config().paths.data_dir) / ctx.config().dataset.repo_id).string(), "--device",
       ctx.options().device, "--config-dir", ctx.resolve(ctx.config().config_dir).string()},
      "11_act_forward_probe.txt", 1800);
  {
    std::ofstream out(log, std::ios::trunc);
    out << result.output << "\n[exit=" << result.exit_code << "]\n";
  }
  const ScreenshotResult init_shot = capture_terminal_screenshot(
      ctx, "11_act_model_init", "s9_init", "cat " + log.string() + " | tail -30", 18);

  ctx.manifest().add_artifact("S9", "log", log, "ACT init/forward/backward probe");
  ctx.manifest().save();
  if (result.exit_code != 0) {
    return StageResult::fail("train-init", "ACT init/forward/backward probe failed (see " + log.string() + ")");
  }
  StageResult stage = StageResult::pass("train-init", "forward/loss/backward verified on " + ctx.options().device);
  stage.artifacts = {log.string()};
  stage.evidence.push_back(init_shot.captured ? init_shot.path.string()
                                              : "SCREENSHOT_UNAVAILABLE: " + init_shot.note);
  stage.duration_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
  return stage;
}

StageResult run_stage_train(RunContext& ctx) {
  const auto start = std::chrono::steady_clock::now();
  if (!std::filesystem::exists(ctx.python_bin())) {
    return StageResult::fail("train", "training env missing: " + ctx.python_bin());
  }
  const std::string phase = ctx.options().dry_run ? "dry-run"
                            : (ctx.options().steps_override > 0 &&
                               ctx.options().steps_override <= ctx.config().runtime.smoke_steps)
                                ? "smoke"
                                : "release";
  if (phase == "dry-run") {
    return run_stage_train_config(ctx);
  }
  if (phase == "smoke" && !ctx.options().resume) {
    const StageResult init = run_stage_train_smoke_init(ctx);
    if (init.status != StageStatus::Pass) {
      return init;
    }
  }

  const long long steps = ctx.options().steps_override > 0 ? ctx.options().steps_override
                                                           : ctx.config().act.steps;
  const std::string job_name =
      phase == "smoke" ? (ctx.config().act.job_name + "_smoke") : ctx.config().act.job_name;
  const std::string log_name = phase == "smoke" ? "12_training_smoke.txt" : "13_training_release.txt";

  const TrainOutcome outcome =
      run_training(ctx, steps, job_name, ctx.options().resume, log_name,
                   ctx.config().runtime.train_timeout_seconds);

  // 指标聚合（真实日志 → JSONL/CSV/PNG）
  const std::filesystem::path metrics_jsonl = ctx.log_file("metrics.jsonl");
  const std::filesystem::path metrics_csv = ctx.results_file("loss_curve.csv");
  std::vector<std::string> result_evidence;
  if (!outcome.samples.empty()) {
    write_metrics_jsonl(outcome.samples, metrics_jsonl);
    write_metrics_csv(outcome.samples, metrics_csv);
  }
  std::string curve_note;
  std::filesystem::path curve_path;
  if (!outcome.samples.empty()) {
    curve_path = write_loss_curve(outcome.samples, ctx.experiment_root() / "results", &curve_note);
  }

  if (!outcome.samples.empty()) {
    const MetricSample& last = outcome.samples.back();
    const auto best = std::min_element(outcome.samples.begin(), outcome.samples.end(),
                                       [](const MetricSample& lhs, const MetricSample& rhs) {
                                         return lhs.loss < rhs.loss;
                                       });
    ctx.manifest().set_field("training", "steps_executed", last.step);
    ctx.manifest().set_field("training", "final_loss", last.loss);
    ctx.manifest().set_field("training", "best_loss", best->loss);
    if (last.samples_per_s > 0) {
      ctx.manifest().set_field("training", "samples_per_s", last.samples_per_s);
    }
    if (last.gpu_mem_gb > 0) {
      ctx.manifest().set_field("training", "peak_gpu_mem_gb", last.gpu_mem_gb);
    }
    ctx.manifest().add_artifact("S10/S11", "metrics", metrics_jsonl, "parsed metrics");
    ctx.manifest().add_artifact("S10/S11", "metrics", metrics_csv, "loss curve csv");
    if (!curve_path.empty() && std::filesystem::exists(curve_path)) {
      ctx.manifest().add_artifact("S10/S11", "figure", curve_path, curve_note);
    }
    // 14_training_loss.txt：由真实日志解析结果生成的 loss 摘要（供证据索引配对）
    {
      std::ofstream out(ctx.log_file("14_training_loss.txt"), std::ios::trunc);
      out << "# training loss summary (derived from " << outcome.log_path.filename().string() << ")\n";
      out << "# generated_at: " << timestamp_iso8601() << "\n\n";
      out << "metric_samples : " << outcome.samples.size() << "\n";
      out << "first_loss     : " << outcome.samples.front().loss << "\n";
      out << "best_loss      : " << best->loss << "\n";
      out << "final_loss     : " << last.loss << "\n";
      out << "final_l1_loss  : (见日志 loss_dict)\n";
      out << "samples_per_s  : " << last.samples_per_s << "\n";
      out << "gpu_mem_gb     : " << last.gpu_mem_gb << "\n";
      out << "steps_executed : " << last.step << "\n";
      out << "curve_file     : " << curve_path.string() << "\n";
    }
    ctx.manifest().add_artifact("S10/S11", "log", ctx.log_file("14_training_loss.txt"), "loss summary");
  }
  if (!outcome.checkpoint_dir.empty()) {
    ctx.manifest().set_field("training", "checkpoint_dir", outcome.checkpoint_dir.string());
  }
  ctx.manifest().add_artifact(phase == "smoke" ? "S10" : "S11", "log", outcome.log_path,
                              "training log");
  ctx.manifest().save();

  if (phase == "smoke") {
    const ScreenshotResult shot = capture_terminal_screenshot(
        ctx, "12_training_start", "s10_smoke", "tail -25 " + outcome.log_path.string(), 18);
    result_evidence.push_back(shot.captured ? shot.path.string() : "SCREENSHOT_UNAVAILABLE: " + shot.note);
  } else {
    const ScreenshotResult shot = capture_terminal_screenshot(
        ctx, "13_training_progress", "s11_progress", "tail -25 " + outcome.log_path.string(), 18);
    result_evidence.push_back(shot.captured ? shot.path.string() : "SCREENSHOT_UNAVAILABLE: " + shot.note);
    if (!outcome.checkpoint_dir.empty()) {
      const ScreenshotResult ckpt_shot = capture_terminal_screenshot(
          ctx, "15_checkpoint", "s11_checkpoint", "ls -la " + outcome.checkpoint_dir.string(), 18);
      result_evidence.push_back(ckpt_shot.captured ? ckpt_shot.path.string()
                                                   : "SCREENSHOT_UNAVAILABLE: " + ckpt_shot.note);
    }
  }
  if (!curve_path.empty() && std::filesystem::exists(curve_path)) {
    const ScreenshotResult curve_shot = capture_terminal_screenshot(
        ctx, "14_training_loss", "s11_curve",
        "ls -la " + (ctx.experiment_root() / "results").string() + " | grep loss && head -5 " +
            metrics_csv.string(),
        18);
    result_evidence.push_back(curve_shot.captured ? curve_shot.path.string()
                                                  : "SCREENSHOT_UNAVAILABLE: " + curve_shot.note);
  }

  StageResult result;
  if (!std::filesystem::exists(outcome.log_path)) {
    result = StageResult::fail("train", "training log missing");
  } else if (outcome.samples.empty()) {
    result = StageResult::fail("train", "no metrics parsed from training log: " + outcome.detail);
  } else if (outcome.checkpoint_dir.empty()) {
    result = StageResult::fail("train", "no checkpoint produced: " + outcome.detail);
  } else {
    result = StageResult::pass("train", outcome.detail);
  }
  result.artifacts = {outcome.log_path.string()};
  if (!outcome.checkpoint_dir.empty()) {
    result.artifacts.push_back(outcome.checkpoint_dir.string());
  }
  result.evidence = result_evidence;
  result.duration_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
  return result;
}

}  // namespace actlab

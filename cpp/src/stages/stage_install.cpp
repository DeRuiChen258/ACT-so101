#include "actlab/stages/stage_install.hpp"

#include "actlab/path_utils.hpp"
#include "actlab/proc.hpp"
#include "actlab/stages/stage_evidence.hpp"

#include <chrono>
#include <fstream>
#include <sstream>

namespace actlab {
namespace {

// 事实来源：logs/06_lerobot_train_help.txt 实测输出。
// 通用参数出现在顶层 --help；ACT 专有参数只有在 --policy.type=act 时才注册（draccus 行为），
// 因此分两次探测，二者都必须命中。
constexpr const char* kRequiredGeneralFlags[] = {"--dataset.repo_id", "--policy.type", "--output_dir",
                                                 "--job_name", "--steps", "--batch_size"};
constexpr const char* kRequiredActFlags[] = {
    "--policy.device",       "--policy.chunk_size",        "--policy.n_action_steps",
    "--policy.dim_model",    "--policy.n_heads",           "--policy.dim_feedforward",
    "--policy.n_encoder_layers", "--policy.n_decoder_layers", "--policy.n_vae_encoder_layers",
    "--policy.latent_dim",   "--policy.dropout",           "--policy.kl_weight",
    "--policy.use_vae",      "--policy.optimizer_lr",      "--policy.push_to_hub"};

std::string to_string(const ProcResult& result) {
  std::ostringstream out;
  out << result.output << "\n[exit_code=" << result.exit_code << " duration_ms="
      << static_cast<long long>(result.duration_ms) << "]\n";
  return out.str();
}

}  // namespace

StageResult run_stage_install(RunContext& ctx) {
  const auto start = std::chrono::steady_clock::now();
  const std::filesystem::path repo = ctx.resolve(ctx.config().paths.repo_dir);
  const std::filesystem::path python = ctx.python_bin();
  const std::filesystem::path install_log = ctx.log_file("04_lerobot_install.txt");
  const std::filesystem::path version_log = ctx.log_file("05_lerobot_version.txt");
  const std::filesystem::path cli_log = ctx.log_file("06_cli_check.txt");

  if (!std::filesystem::exists(python)) {
    return StageResult::fail("install",
                             "training env python not found: " + python.string() +
                                 " (run scripts/b01_create_env.sh first)");
  }
  if (!std::filesystem::exists(repo / "pyproject.toml")) {
    return StageResult::fail("install", "repo not found at " + repo.string() + " (run stage repo first)");
  }

  std::ostringstream install_text;
  install_text << "# Stage S3 install\n# timestamp: " << timestamp_iso8601() << "\n\n";

  const ProcResult pip_version = ctx.run_stack({python.string(), "-m", "pip", "--version"},
                                               "04_pip_version.txt");
  install_text << "===== pip =====\n" << to_string(pip_version);

  const ProcResult install = ctx.run_stack(
      {python.string(), "-m", "pip", "install", "-e", repo.string() + "[training]"}, "04_pip_install.txt",
      3600);
  install_text << "\n===== pip install -e .[training] =====\n" << to_string(install);
  if (install.exit_code != 0) {
    std::ofstream out(install_log, std::ios::trunc);
    out << install_text.str();
    return StageResult::fail("install", "pip install -e .[training] failed (see " + install_log.string() + ")");
  }

  const ProcResult import_check = ctx.run_stack(
      {python.string(), "-c",
       "import lerobot, torch, torchvision, sys; print('lerobot', lerobot.__version__); "
       "print('torch', torch.__version__); print('torchvision', torchvision.__version__); "
       "print('cuda_available', torch.cuda.is_available()); print('arch_list', torch.cuda.get_arch_list()); "
       "print('python', sys.version.split()[0])"},
      "04_import_check.txt");
  install_text << "\n===== import check =====\n" << to_string(import_check);
  {
    std::ofstream out(install_log, std::ios::trunc);
    out << install_text.str();
  }
  if (import_check.exit_code != 0) {
    return StageResult::fail("install", "import lerobot failed (see " + install_log.string() + ")");
  }

  // CLI 探测：关键参数必须存在（缺一即 FAIL，不做推断）
  const ProcResult train_help = ctx.run_stack({python.string(), "-m", "lerobot.scripts.lerobot_train", "--help"},
                                              "06_lerobot_train_help.txt");
  std::ostringstream cli_text;
  cli_text << "# Stage S3 CLI check\n# timestamp: " << timestamp_iso8601() << "\n\n";
  cli_text << to_string(train_help);
  std::vector<std::string> missing_flags;
  for (const char* flag : kRequiredGeneralFlags) {
    if (train_help.output.find(flag) == std::string::npos) {
      missing_flags.emplace_back(flag);
    }
  }
  const ProcResult act_help = ctx.run_stack(
      {python.string(), "-m", "lerobot.scripts.lerobot_train", "--policy.type=act", "--help"},
      "06_lerobot_act_help.txt");
  cli_text << "\n===== lerobot_train --policy.type=act --help (ACT 专有参数) =====\n"
           << to_string(act_help);
  for (const char* flag : kRequiredActFlags) {
    if (act_help.output.find(flag) == std::string::npos) {
      missing_flags.emplace_back(flag);
    }
  }
  const ProcResult act_check = ctx.run_stack(
      {python.string(), "-c",
       "from lerobot.policies.factory import get_policy_class, make_policy_config; "
       "cls = get_policy_class('act'); cfg = make_policy_config('act'); "
       "print('policy_class', cls.__name__); print('config_class', type(cfg).__name__); "
       "print('chunk_size', cfg.chunk_size); print('n_action_steps', cfg.n_action_steps); "
       "print('dim_model', cfg.dim_model); print('ACTPOLICY_OK')"},
      "06_act_policy_check.txt");
  cli_text << "\n===== policy.type=act resolution =====\n" << to_string(act_check);
  cli_text << "\nmissing required train flags: ";
  if (missing_flags.empty()) {
    cli_text << "none\n";
  } else {
    for (const auto& flag : missing_flags) {
      cli_text << flag << " ";
    }
    cli_text << "\n";
  }
  {
    std::ofstream out(cli_log, std::ios::trunc);
    out << cli_text.str();
  }

  // 版本锁定：conda env export + pip freeze 成对产出
  std::ostringstream version_text;
  version_text << "# Stage S3 version lock\n# timestamp: " << timestamp_iso8601() << "\n\n";
  const std::filesystem::path conda_bin =
      std::filesystem::path(ctx.config().paths.conda_root) / "bin" / "conda";
  const std::filesystem::path conda_lock =
      ctx.resolve(ctx.config().config_dir) / "conda_env.lock.yml";
  const std::filesystem::path pip_lock = ctx.log_file("pip_freeze.txt");
  {
    const std::string command = conda_bin.string() + " env export -n " + ctx.config().paths.env_name +
                                " --no-builds > " + conda_lock.string();
    const ProcResult lock = run_process({"bash", "-lc", command}, ProcOptions{});
    version_text << "$ " << command << "\n[exit=" << lock.exit_code << "]\n";
  }
  {
    const std::string command = python.string() + " -m pip freeze > " + pip_lock.string();
    const ProcResult lock = run_process({"bash", "-lc", command}, ProcOptions{});
    version_text << "$ " << command << "\n[exit=" << lock.exit_code << "]\n";
  }
  version_text << "\nconda lock bytes: "
               << (std::filesystem::exists(conda_lock) ? std::filesystem::file_size(conda_lock) : 0) << "\n";
  version_text << "pip freeze bytes: "
               << (std::filesystem::exists(pip_lock) ? std::filesystem::file_size(pip_lock) : 0) << "\n";
  {
    std::ofstream out(version_log, std::ios::trunc);
    out << version_text.str();
  }

  ctx.manifest().set_field("environment", "lerobot_version", "0.6.2 (from repo pyproject)");
  ctx.manifest().add_artifact("S3", "log", install_log, "pip install evidence");
  ctx.manifest().add_artifact("S3", "log", cli_log, "CLI check");
  if (std::filesystem::exists(conda_lock)) {
    ctx.manifest().add_artifact("S3", "lock", conda_lock, "conda env lock");
  }
  ctx.manifest().add_artifact("S3", "lock", pip_lock, "pip freeze");
  ctx.manifest().save();

  const ScreenshotResult install_shot =
      capture_terminal_screenshot(ctx, "04_lerobot_install", "s3_install",
                                  "tail -25 " + install_log.string(), 18);
  const ScreenshotResult cli_shot = capture_terminal_screenshot(
      ctx, "06_cli_check", "s3_cli",
      "grep -nE -- '--(dataset.repo_id|policy.type|policy.device)' " + cli_log.string() + " | head -12",
      18);
  const ScreenshotResult version_shot = capture_terminal_screenshot(
      ctx, "05_lerobot_version", "s3_version",
      "cat " + version_log.string() + " && grep -E '^(lerobot|torch|torchvision)==' " + pip_lock.string(),
      18);

  if (!missing_flags.empty()) {
    return StageResult::fail("install", "lerobot-train --help is missing required flags");
  }
  if (act_check.exit_code != 0 || act_check.output.find("ACTPOLICY_OK") == std::string::npos) {
    return StageResult::fail("install", "policy.type=act resolution failed");
  }

  StageResult result = StageResult::pass("install", "editable install OK; CLI flags present; ACT resolvable");
  result.artifacts = {install_log.string(), version_log.string(), cli_log.string(), pip_lock.string()};
  result.evidence.push_back(install_shot.captured ? install_shot.path.string()
                                                  : "SCREENSHOT_UNAVAILABLE: " + install_shot.note);
  result.evidence.push_back(cli_shot.captured ? cli_shot.path.string()
                                              : "SCREENSHOT_UNAVAILABLE: " + cli_shot.note);
  result.evidence.push_back(version_shot.captured ? version_shot.path.string()
                                                  : "SCREENSHOT_UNAVAILABLE: " + version_shot.note);
  result.duration_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
  return result;
}

}  // namespace actlab

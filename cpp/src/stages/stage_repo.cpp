#include "actlab/stages/stage_repo.hpp"

#include "actlab/path_utils.hpp"
#include "actlab/proc.hpp"
#include "actlab/sha256.hpp"
#include "actlab/stages/stage_evidence.hpp"

#include <chrono>
#include <fstream>
#include <sstream>

namespace actlab {
namespace {

constexpr const char* kGitcodeMirror = "https://gitcode.com/GitHub_Trending/le/lerobot.git";
constexpr const char* kGithubFallback = "https://github.com/huggingface/lerobot.git";

std::string git(const std::filesystem::path& repo, const std::vector<std::string>& args) {
  std::vector<std::string> argv = {"git", "-C", repo.string()};
  argv.insert(argv.end(), args.begin(), args.end());
  const ProcResult result = run_process(argv, ProcOptions{});
  std::string output = result.output;
  while (!output.empty() && (output.back() == '\n' || output.back() == '\r')) {
    output.pop_back();
  }
  return output;
}

std::string count_files(const std::filesystem::path& dir) {
  if (!std::filesystem::exists(dir)) {
    return "0";
  }
  size_t count = 0;
  for (const auto& entry : std::filesystem::recursive_directory_iterator(dir)) {
    if (entry.is_regular_file()) {
      ++count;
    }
  }
  return std::to_string(count);
}

std::string grep_line(const std::string& text, const std::string& needle) {
  std::istringstream stream(text);
  std::string line;
  while (std::getline(stream, line)) {
    if (line.find(needle) != std::string::npos) {
      return line;
    }
  }
  return "";
}

}  // namespace

StageResult run_stage_repo(RunContext& ctx) {
  const auto start = std::chrono::steady_clock::now();
  const std::filesystem::path repo = ctx.resolve(ctx.config().paths.repo_dir);
  const std::filesystem::path probe_log = ctx.log_file("03_repo_probe.txt");
  std::ostringstream log;
  log << "# Stage S1 repo probe\n# timestamp: " << timestamp_iso8601() << "\n\n";

  if (!std::filesystem::exists(repo / ".git")) {
    ensure_dir(repo.parent_path());
    log << "repo not found, cloning GitCode mirror: " << kGitcodeMirror << "\n";
    ProcOptions clone_options;
    clone_options.echo = true;
    clone_options.timeout_seconds = 1800;
    ProcResult clone =
        run_process({"git", "clone", kGitcodeMirror, repo.string()}, clone_options);
    if (clone.exit_code != 0) {
      log << "GitCode clone failed (exit=" << clone.exit_code << "); falling back to GitHub\n";
      clone = run_process({"git", "clone", kGithubFallback, repo.string()}, clone_options);
      if (clone.exit_code != 0) {
        StageResult result = StageResult::fail("repo", "clone failed from both GitCode and GitHub");
        result.duration_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        return result;
      }
      log << "cloned from GitHub fallback\n";
    }
  } else {
    log << "repo already present (reused, not re-cloned)\n";
  }

  const std::string commit = git(repo, {"rev-parse", "HEAD"});
  const std::string branch = git(repo, {"rev-parse", "--abbrev-ref", "HEAD"});
  const std::string remote = git(repo, {"config", "--get", "remote.origin.url"});
  const std::string describe = git(repo, {"describe", "--tags", "--always"});
  const std::string status = git(repo, {"status", "--porcelain"});

  const std::filesystem::path pyproject = repo / "pyproject.toml";
  std::ifstream pyproject_in(pyproject);
  std::ostringstream pyproject_buffer;
  pyproject_buffer << pyproject_in.rdbuf();
  const std::string pyproject_text = pyproject_buffer.str();
  const std::string version_line = grep_line(pyproject_text, "version =");
  const std::string python_line = grep_line(pyproject_text, "requires-python");
  const std::string torch_line = grep_line(pyproject_text, "\"torch>=");
  const std::string torchvision_line = grep_line(pyproject_text, "\"torchvision>=");
  const std::string training_extra = grep_line(pyproject_text, "training = [");

  const std::filesystem::path src_root = repo / "src" / "lerobot";
  const std::string policies_files = count_files(src_root / "policies");
  const std::string datasets_files = count_files(src_root / "datasets");
  const std::string scripts_files = count_files(src_root / "scripts");
  const std::filesystem::path act_dir = src_root / "policies" / "act";

  log << "commit      : " << commit << "\n";
  log << "branch      : " << branch << "\n";
  log << "describe    : " << describe << "\n";
  log << "remote      : " << remote << "\n";
  log << "status      : " << (status.empty() ? "(clean)" : status) << "\n";
  log << "pyproject   : " << version_line << " | " << python_line << "\n";
  log << "torch req   : " << torch_line << "\n";
  log << "tv req      : " << torchvision_line << "\n";
  log << "training    : " << training_extra << "\n";
  log << "src files   : policies=" << policies_files << " datasets=" << datasets_files
      << " scripts=" << scripts_files << "\n";
  log << "ACT impl    : " << (std::filesystem::exists(act_dir) ? act_dir.string() : "MISSING") << "\n";
  for (const auto& name : {"configuration_act.py", "modeling_act.py", "processor_act.py"}) {
    const std::filesystem::path file = act_dir / name;
    log << "  - " << name << " : " << (std::filesystem::exists(file) ? "present" : "MISSING");
    if (std::filesystem::exists(file)) {
      log << " (sha256 " << sha256_file(file).substr(0, 16) << ", " << std::filesystem::file_size(file)
          << " bytes)";
    }
    log << "\n";
  }
  {
    std::ofstream out(probe_log, std::ios::trunc);
    out << log.str();
  }

  // S2：结构分析文档（由真实扫描结果 + 源码模板生成）
  std::ostringstream analysis;
  analysis << "# LeRobot 源码结构分析（C++ actlab repo 阶段自动生成）\n\n";
  analysis << "- 生成时间: " << timestamp_iso8601() << "\n";
  analysis << "- 仓库: `" << repo.string() << "`\n";
  analysis << "- commit: `" << commit << "` (branch `" << branch << "`, " << describe << ")\n";
  analysis << "- remote: " << remote << "\n";
  analysis << "- pyproject: " << version_line << "; " << python_line << "\n";
  analysis << "- torch 约束: " << torch_line << "\n\n";
  analysis << "## 目录与职责\n\n";
  analysis << "```text\nlerobot/\n├── datasets/      # LeRobotDataset 本体、meta、视频解码与流式读取 (" << datasets_files << " 文件)\n";
  analysis << "├── policies/      # 策略实现，ACT 位于 policies/act/ (" << policies_files << " 文件)\n";
  analysis << "│   └── act/       # configuration_act.py / modeling_act.py / processor_act.py\n";
  analysis << "├── scripts/       # CLI 入口（lerobot_train / lerobot_dataset_viz 等） (" << scripts_files << " 文件)\n";
  analysis << "└── utils/         # 日志、训练循环等基础设施\n```\n\n";
  analysis << "## 数据流（BC + ACT 训练路径）\n\n";
  analysis << "```text\nLeRobotDataset (parquet + 视频)\n   ↓ DataLoader (batch, num_workers)\n";
  analysis << "observation.images.*  observation.state\n   ↓ ACT preprocessing (normalization, MEAN_STD)\n";
  analysis << "ACT Policy: ResNet18 backbone → transformer encoder/decoder（VAE 分支仅训练期）\n";
  analysis << "   ↓ action chunk [B, chunk_size, action_dim]\n";
  analysis << "Loss = L1(action) + kl_weight * KL(q(z|a) || N(0,1))\n";
  analysis << "   ↓ optimizer (AdamW)\nCheckpoint (pretrained_model/, optimizer state, 训练配置)\n```\n\n";
  analysis << "## 关键源码位置（实测存在性）\n\n";
  analysis << "| 文件 | 存在 | 大小 | sha256(前16) |\n| --- | --- | --- | --- |\n";
  for (const auto& name : {"configuration_act.py", "modeling_act.py", "processor_act.py"}) {
    const std::filesystem::path file = act_dir / name;
    if (std::filesystem::exists(file)) {
      analysis << "| `policies/act/" << name << "` | yes | " << std::filesystem::file_size(file)
               << " | " << sha256_file(file).substr(0, 16) << " |\n";
    } else {
      analysis << "| `policies/act/" << name << "` | NO | - | - |\n";
    }
  }
  analysis << "\n> 说明：本文件由 `actlab repo` 依据真实扫描结果生成；参数级细节见 "
              "`logs/act_algorithm_notes.md`（S7，逐条标注源码行号）。\n";
  const std::filesystem::path analysis_path = ctx.log_file("architecture_analysis.md");
  {
    std::ofstream out(analysis_path, std::ios::trunc);
    out << analysis.str();
  }

  ctx.manifest().set_field("environment", "lerobot_commit", commit);
  ctx.manifest().set_field("environment", "lerobot_branch", branch);
  ctx.manifest().set_field("environment", "lerobot_version_line", version_line);
  ctx.manifest().add_artifact("S1", "log", probe_log, "repo probe");
  ctx.manifest().add_artifact("S2", "analysis", analysis_path, "architecture analysis");
  ctx.manifest().save();

  const ScreenshotResult shot = capture_terminal_screenshot(
      ctx, "03_repo_clone", "s1_repo",
      "cd " + repo.string() + " && git log -1 --oneline && sed -n '24,32p' pyproject.toml", 18);

  StageResult result = StageResult::pass(
      "repo", "commit=" + commit + " branch=" + branch + " ACT impl=" +
                  (std::filesystem::exists(act_dir) ? "policies/act/" : "MISSING"));
  result.artifacts = {probe_log.string(), analysis_path.string()};
  result.evidence.push_back(shot.captured ? shot.path.string()
                                          : "SCREENSHOT_UNAVAILABLE: " + shot.note);
  result.duration_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
  return result;
}

}  // namespace actlab

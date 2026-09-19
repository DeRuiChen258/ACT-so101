// 受控子进程执行：fork/execvp + 实时流式抓取 stdout/stderr + 超时/中断 + 返回码校验 + 命令留存
#pragma once

#include <filesystem>
#include <map>
#include <string>
#include <vector>

namespace actlab {

struct ProcOptions {
  std::filesystem::path cwd;                       // 空 = 继承当前目录
  std::map<std::string, std::string> env;          // 追加/覆盖的环境变量
  std::filesystem::path log_path;                  // 原始输出留存（追加写）
  bool echo = true;                               // 是否同时回显到控制台
  int timeout_seconds = 0;                         // 0 = 不设超时
};

struct ProcResult {
  int exit_code = -1;
  bool timed_out = false;
  bool signaled = false;
  std::string output;                              // 合并后的 stdout+stderr
  double duration_ms = 0.0;
};

// 执行 argv（argv[0] 为可执行文件名或其绝对路径）；返回合并输出与退出码。
// 返回码非 0 不抛异常，由调用方按门禁决定是否失败。
ProcResult run_process(const std::vector<std::string>& argv, const ProcOptions& options = {});

// 语义化封装：非 0 退出码抛 std::runtime_error（附最后 800 字符输出）
ProcResult run_process_or_throw(const std::vector<std::string>& argv, const ProcOptions& options = {});

// 生成便于留档的可读命令行（含引号转义）
std::string format_command(const std::vector<std::string>& argv);

// 检查可执行文件是否在 PATH 中（which 语义，不依赖外部 which 命令）
std::string find_executable(const std::string& name);

// 分离式启动（用于 GUI 程序，如 xterm 证据窗口）；返回子进程 PID，-1 表示失败
int spawn_detached(const std::vector<std::string>& argv, const ProcOptions& options = {});

// 终止进程（先 SIGTERM，等待 grace_ms 后 SIGKILL）
void terminate_process(int pid, int grace_ms = 2000);

}  // namespace actlab

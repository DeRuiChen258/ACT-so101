#include "actlab/proc.hpp"

#include "actlab/path_utils.hpp"

#include <sys/wait.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace actlab {
namespace {

std::string quote(const std::string& raw) {
  if (raw.find_first_of(" '\"$\\") == std::string::npos) {
    return raw;
  }
  std::string out = "'";
  for (const char ch : raw) {
    if (ch == '\'') {
      out += "'\\''";
    } else {
      out.push_back(ch);
    }
  }
  out += "'";
  return out;
}

}  // namespace

std::string format_command(const std::vector<std::string>& argv) {
  std::string out;
  for (size_t i = 0; i < argv.size(); ++i) {
    if (i > 0) out += " ";
    out += quote(argv[i]);
  }
  return out;
}

std::string find_executable(const std::string& name) {
  const char* path_env = std::getenv("PATH");
  if (path_env == nullptr) {
    return "";
  }
  std::stringstream stream(path_env);
  std::string dir;
  while (std::getline(stream, dir, ':')) {
    if (dir.empty()) continue;
    const std::filesystem::path candidate = std::filesystem::path(dir) / name;
    if (::access(candidate.c_str(), X_OK) == 0) {
      return candidate.string();
    }
  }
  return "";
}

ProcResult run_process(const std::vector<std::string>& argv, const ProcOptions& options) {
  if (argv.empty()) {
    throw std::runtime_error("run_process: empty argv");
  }
  const auto start = std::chrono::steady_clock::now();

  int pipe_fd[2];
  if (::pipe(pipe_fd) != 0) {
    throw std::runtime_error(std::string("run_process: pipe() failed: ") + std::strerror(errno));
  }

  const pid_t pid = ::fork();
  if (pid < 0) {
    ::close(pipe_fd[0]);
    ::close(pipe_fd[1]);
    throw std::runtime_error(std::string("run_process: fork() failed: ") + std::strerror(errno));
  }

  if (pid == 0) {
    // 子进程：合并 stderr 到 stdout 管道
    ::setpgid(0, 0);
    ::dup2(pipe_fd[1], STDOUT_FILENO);
    ::dup2(pipe_fd[1], STDERR_FILENO);
    ::close(pipe_fd[0]);
    ::close(pipe_fd[1]);
    if (!options.cwd.empty()) {
      if (::chdir(options.cwd.c_str()) != 0) {
        std::cerr << "run_process: chdir failed: " << options.cwd << std::endl;
        ::_exit(127);
      }
    }
    for (const auto& [key, value] : options.env) {
      ::setenv(key.c_str(), value.c_str(), 1);
    }
    std::vector<char*> raw;
    raw.reserve(argv.size() + 1);
    for (const auto& item : argv) {
      raw.push_back(const_cast<char*>(item.c_str()));
    }
    raw.push_back(nullptr);
    ::execvp(raw[0], raw.data());
    std::cerr << "run_process: execvp failed for " << raw[0] << ": " << std::strerror(errno) << std::endl;
    ::_exit(127);
  }

  ::close(pipe_fd[1]);
  std::ofstream log_sink;
  if (!options.log_path.empty()) {
    ensure_dir(options.log_path.parent_path());
    log_sink.open(options.log_path, std::ios::app);
  }
  if (options.echo || log_sink.is_open()) {
    const std::string header = "$ " + format_command(argv) + "\n";
    if (options.echo) std::cout << header << std::flush;
    if (log_sink.is_open()) log_sink << header << std::flush;
  }

  ProcResult result;
  std::array<char, 4096> buffer{};
  bool timeout_hit = false;
  while (true) {
    const ssize_t got = ::read(pipe_fd[0], buffer.data(), buffer.size());
    if (got > 0) {
      const std::string chunk(buffer.data(), static_cast<size_t>(got));
      result.output += chunk;
      if (options.echo) std::cout << chunk << std::flush;
      if (log_sink.is_open()) log_sink << chunk << std::flush;
    } else if (got == 0) {
      break;
    } else if (errno != EINTR) {
      break;
    }

    if (options.timeout_seconds > 0 && !timeout_hit) {
      const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                               std::chrono::steady_clock::now() - start)
                               .count();
      if (elapsed > options.timeout_seconds) {
        timeout_hit = true;
        ::kill(-pid, SIGTERM);
      }
    }
  }
  ::close(pipe_fd[0]);

  int status = 0;
  while (::waitpid(pid, &status, 0) < 0 && errno == EINTR) {
  }
  if (WIFEXITED(status)) {
    result.exit_code = WEXITSTATUS(status);
  } else if (WIFSIGNALED(status)) {
    result.signaled = true;
    result.exit_code = 128 + WTERMSIG(status);
    if (WTERMSIG(status) == SIGTERM || WTERMSIG(status) == SIGKILL) {
      result.timed_out = timeout_hit;
    }
  }
  result.duration_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
  if (log_sink.is_open()) {
    log_sink << "[exit_code=" << result.exit_code << " duration_ms=" << result.duration_ms << "]\n";
  }
  return result;
}

ProcResult run_process_or_throw(const std::vector<std::string>& argv, const ProcOptions& options) {
  const ProcResult result = run_process(argv, options);
  if (result.exit_code != 0) {
    std::string tail = result.output;
    if (tail.size() > 800) {
      tail = tail.substr(tail.size() - 800);
    }
    throw std::runtime_error("command failed (exit=" + std::to_string(result.exit_code) + "): " +
                             format_command(argv) + "\n--- output tail ---\n" + tail);
  }
  return result;
}

int spawn_detached(const std::vector<std::string>& argv, const ProcOptions& options) {
  if (argv.empty()) {
    throw std::runtime_error("spawn_detached: empty argv");
  }
  const pid_t pid = ::fork();
  if (pid < 0) {
    throw std::runtime_error(std::string("spawn_detached: fork() failed: ") + std::strerror(errno));
  }
  if (pid == 0) {
    ::setsid();
    const std::string dev_null = "/dev/null";
    const int fd = ::open(dev_null.c_str(), O_RDWR);
    if (fd >= 0) {
      ::dup2(fd, STDIN_FILENO);
      ::dup2(fd, STDOUT_FILENO);
      ::dup2(fd, STDERR_FILENO);
      if (fd > STDERR_FILENO) {
        ::close(fd);
      }
    }
    if (!options.cwd.empty()) {
      if (::chdir(options.cwd.c_str()) != 0) {
        ::_exit(127);
      }
    }
    for (const auto& [key, value] : options.env) {
      ::setenv(key.c_str(), value.c_str(), 1);
    }
    std::vector<char*> raw;
    raw.reserve(argv.size() + 1);
    for (const auto& item : argv) {
      raw.push_back(const_cast<char*>(item.c_str()));
    }
    raw.push_back(nullptr);
    ::execvp(raw[0], raw.data());
    ::_exit(127);
  }
  return static_cast<int>(pid);
}

void terminate_process(int pid, int grace_ms) {
  if (pid <= 0) {
    return;
  }
  if (::kill(pid, SIGTERM) != 0) {
    return;
  }
  const int steps = grace_ms > 0 ? grace_ms / 50 : 0;
  for (int i = 0; i < steps; ++i) {
    int status = 0;
    if (::waitpid(pid, &status, WNOHANG) == pid) {
      return;
    }
    ::usleep(50 * 1000);
  }
  ::kill(pid, SIGKILL);
  int status = 0;
  ::waitpid(pid, &status, 0);
}

}  // namespace actlab

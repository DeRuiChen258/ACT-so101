// actlab 可执行文件入口：装配对象、注册子命令、顶层异常捕获与错误码映射
#include "actlab/cli.hpp"

#include <exception>
#include <iostream>

int main(int argc, char** argv) {
  try {
    return actlab::run_cli(argc, argv);
  } catch (const std::exception& error) {
    std::cerr << "[FATAL] unhandled exception: " << error.what() << std::endl;
    return actlab::kExitFail;
  } catch (...) {
    std::cerr << "[FATAL] unhandled non-standard exception" << std::endl;
    return actlab::kExitFail;
  }
}

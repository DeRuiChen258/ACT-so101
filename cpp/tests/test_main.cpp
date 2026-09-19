#include "actlab_test.hpp"

#include <iostream>
#include <string>

int main() {
  int failed = 0;
  int passed = 0;
  for (const auto& test : actlab_test::registry()) {
    try {
      test.body();
      ++passed;
      std::cout << "[PASS] " << test.name << std::endl;
    } catch (const std::exception& error) {
      ++failed;
      std::cout << "[FAIL] " << test.name << " -> " << error.what() << std::endl;
    }
  }
  std::cout << "\nunit tests: " << passed << " passed, " << failed << " failed" << std::endl;
  return failed == 0 ? 0 : 1;
}

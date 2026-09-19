# ==============================================================================
# actlab — LeRobot/ACT 实验工程（C++17 主语言）
#   build/local : 无 GPU 目标（纯逻辑 + 单元测试），ACTLAB_WITH_TORCH=OFF
#   build/gpu   : 链接 LibTorch（GPU 推理），ACTLAB_WITH_TORCH=ON
# 用法：
#   cmake -S cpp -B build/local -G Ninja -DACTLAB_WITH_TORCH=OFF
#   cmake -S cpp -B build/gpu   -G Ninja -DACTLAB_WITH_TORCH=ON \
#         -DCMAKE_PREFIX_PATH=/home/violet/Workspace/IDE/libtorch
# ==============================================================================
cmake_minimum_required(VERSION 3.20)
project(actlab LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(ACTLAB_WITH_TORCH "Link LibTorch for GPU inference targets" ON)

add_compile_options(-Wall -Wextra -Wpedantic)

# ---- 纯 C++ 核心（两种构建都编译） ----
set(ACTLAB_CORE_SOURCES
  src/sha256.cpp
  src/path_utils.cpp
  src/json_min.cpp
  src/yaml_min.cpp
  src/logging.cpp
  src/proc.cpp
  src/context.cpp
  src/config_loader.cpp
  src/metrics.cpp
  src/manifest.cpp
  src/dataset_schema.cpp
  src/npy_io.cpp
  src/cli.cpp
  src/stages/stage_env.cpp
  src/stages/stage_repo.cpp
  src/stages/stage_install.cpp
  src/stages/stage_dataset.cpp
  src/stages/stage_train.cpp
  src/stages/stage_verify.cpp
  src/stages/stage_evidence.cpp
  src/stages/stage_report.cpp
)

# ---- 依赖 LibTorch 的推理侧（仅 GPU 构建） ----
set(ACTLAB_TORCH_SOURCES
  src/gpu_check.cpp
  src/tensor_utils.cpp
  src/inference/act_runtime.cpp
  src/inference/temporal_ensemble.cpp
  src/stages/stage_eval.cpp
)

if(ACTLAB_WITH_TORCH)
  find_package(Torch REQUIRED)
  message(STATUS "actlab: LibTorch found at ${TORCH_INSTALL_PREFIX}")

  # 单一库：核心 + LibTorch 推理侧源码（避免两个静态库之间的循环依赖）
  add_library(actlab_core STATIC ${ACTLAB_CORE_SOURCES} ${ACTLAB_TORCH_SOURCES})
  target_include_directories(actlab_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${TORCH_INCLUDE_DIRS})
  target_compile_definitions(actlab_core PUBLIC ACTLAB_WITH_TORCH=1)
  target_link_libraries(actlab_core PUBLIC "${TORCH_LIBRARIES}")
  if(TORCH_CXX_FLAGS)
    target_compile_options(actlab_core PUBLIC ${TORCH_CXX_FLAGS})
  endif()

  add_executable(actlab apps/actlab_main.cpp)
  target_link_libraries(actlab PRIVATE actlab_core)

  add_executable(gpu_smoke apps/gpu_smoke_main.cpp)
  target_link_libraries(gpu_smoke PRIVATE actlab_core)

  add_executable(infer_act apps/infer_act_main.cpp)
  target_link_libraries(infer_act PRIVATE actlab_core)
  set_target_properties(infer_act PROPERTIES BUILD_RPATH "${TORCH_INSTALL_PREFIX}/lib")
  set_target_properties(actlab PROPERTIES BUILD_RPATH "${TORCH_INSTALL_PREFIX}/lib")
  set_target_properties(gpu_smoke PROPERTIES BUILD_RPATH "${TORCH_INSTALL_PREFIX}/lib")
else()
  message(STATUS "actlab: ACTLAB_WITH_TORCH=OFF (pure-logic local build)")
  add_library(actlab_core STATIC ${ACTLAB_CORE_SOURCES})
  target_include_directories(actlab_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
  add_executable(actlab_local apps/actlab_main.cpp)
  target_link_libraries(actlab_local PRIVATE actlab_core)
endif()

# ---- 单元测试（不依赖 GPU / 网络） ----
add_executable(actlab_tests
  tests/test_main.cpp
  tests/test_sha256.cpp
  tests/test_metrics_parse.cpp
  tests/test_schema_validate.cpp
  tests/test_config_loader.cpp
)
target_link_libraries(actlab_tests PRIVATE actlab_core)
target_include_directories(actlab_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)

enable_testing()
add_test(NAME actlab_unit_tests COMMAND actlab_tests)

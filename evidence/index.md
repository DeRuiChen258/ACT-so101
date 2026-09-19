# 实验证据索引

| 编号 | 阶段 | 截图 | 日志 | 说明 | 时间戳 | 截图哈希 |
| --- | --- | --- | --- | --- | --- | --- |
| 00 | hardware_check | 00_hardware_check.png | 00_hardware_check.txt | OS/GPU/CUDA/Python/conda/git/ffmpeg 探测 | 2026-09-19T20:25:02+08:00 | cb38433cba054f88 |
| 01 | python_environment | 01_python_environment.png | 01_python_environment.txt | conda env 与 python 版本 | 2026-09-19T20:25:02+08:00 | 78eedc89c56c40a1 |
| 02 | cuda_pytorch | 02_cuda_pytorch.png | 02_cuda_pytorch.txt | torch CUDA / sm_120 可用性 | 2026-09-19T20:25:02+08:00 | 83a27e5ad1defdfe |
| 03 | repo_clone | 03_repo_clone.png | 03_repo_probe.txt | LeRobot 源码 commit/branch | 2026-09-19T20:25:02+08:00 | ba377e07209d7cd7 |
| 04 | lerobot_install | 04_lerobot_install.png | 04_lerobot_install.txt | editable 安装与 import 验证 | 2026-09-19T20:25:02+08:00 | d3c04e25451d6cde |
| 05 | lerobot_version | 05_lerobot_version.png | 05_lerobot_version.txt | 版本锁定（pip freeze / conda lock） | 2026-09-19T20:25:02+08:00 | dd577c50dec2f27e |
| 06 | cli_check | 06_cli_check.png | 06_cli_check.txt | lerobot-train --help 关键参数 | 2026-09-19T20:25:02+08:00 | 37a60074d9fe117a |
| 07 | dataset_download | 07_dataset_download.png | 07_dataset_download.txt | 数据集落盘与规模 | 2026-09-19T20:25:02+08:00 | a3ab0f4773d19855 |
| 08 | dataset_structure | 08_dataset_structure.png | 08_dataset_schema.txt | schema 审计结果 | 2026-09-19T20:25:02+08:00 | 9703d233789ad70a |
| 09 | dataset_visualization | 09_dataset_visualization.png | 09_dataset_viz.txt | 可视化产物（.rrd / 帧拼图） | 2026-09-19T20:25:02+08:00 | 2d6de855fd2b2bdf |
| 10 | act_config | 10_act_config.png | 10_act_config.txt | 冻结的超参与配置哈希 | 2026-09-19T20:25:02+08:00 | f371f544d2c6a9e6 |
| 11 | act_model_init | 11_act_model_init.png | 11_act_model_init.txt | 最小初始化/前向/反向后端验证 | 2026-09-19T20:25:02+08:00 | 37e17497a16f03b1 |
| 12 | training_start | 12_training_start.png | 12_training_smoke.txt | smoke 训练启动（100 步） | 2026-09-19T20:25:02+08:00 | dd696c573ec47045 |
| 13 | training_progress | 13_training_progress.png | 13_training_release.txt | release 训练（3000 步） | 2026-09-19T20:25:02+08:00 | ebbd46c8c4355461 |
| 14 | training_loss | 14_training_loss.png | 14_training_loss.txt | loss 曲线（真实日志） | 2026-09-19T20:25:02+08:00 | f3d658d175817ad9 |
| 15 | checkpoint | 15_checkpoint.png | 15_checkpoint.txt | checkpoint 目录与哈希 | 2026-09-19T20:25:02+08:00 | f5de38bbc78abe22 |
| 16 | model_reload | 16_model_reload.png | 16_model_reload.txt | 新进程 C++ 重载 | 2026-09-19T20:25:02+08:00 | c1e1690141806d0a |
| 17 | inference | 17_inference.png | 17_inference.txt | C++ 推理与延时统计 | 2026-09-19T20:25:02+08:00 | 92f5149df678ddc0 |
| 18 | evaluation | 18_evaluation.png | 18_evaluation.txt | Offline Evaluation 结果 | 2026-09-19T20:25:02+08:00 | 0fac8242d2255acd |
| 19 | final_environment | 19_final_environment.png | 19_final_environment.txt | 最终环境与产物总览 | 2026-09-19T20:25:02+08:00 | 99c433f3775420ed |
| 20 | so101_mujoco_viz | 20_so101_mujoco_window.png | 20_so101_mujoco_window.log | SO-101 MuJoCo 桌面仿真窗口（真实物体：抓取前静置 → 夹住随爪移动 → 松开落回平面） | 2026-09-19T20:25:02+08:00 | db44194a7d6b57f6 |

- 截图工具: `/usr/bin/import`
- 截图方式: 真实 xterm 窗口运行真实命令后按窗口 id 抓取（本机 Wayland 下整屏抓取不可用，已在 evidence/README.md 说明）
- 缺失截图数: 0
- 缺失日志数: 0

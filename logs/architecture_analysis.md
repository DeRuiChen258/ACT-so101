# LeRobot 源码结构分析（C++ actlab repo 阶段自动生成）

- 生成时间: 2026-09-19T19:09:07+08:00
- 仓库: `/home/violet/Workspace/IDE/Physical_AI/repo`
- commit: `c73cec56b04f50ebe9286c79905103866d970169` (branch `main`, c73cec5)
- remote: https://gitcode.com/GitHub_Trending/le/lerobot.git
- pyproject: version = "0.6.2"; requires-python = ">=3.12"
- torch 约束:     "torch>=2.7,<2.12.0",

## 目录与职责

```text
lerobot/
├── datasets/      # LeRobotDataset 本体、meta、视频解码与流式读取 (27 文件)
├── policies/      # 策略实现，ACT 位于 policies/act/ (174 文件)
│   └── act/       # configuration_act.py / modeling_act.py / processor_act.py
├── scripts/       # CLI 入口（lerobot_train / lerobot_dataset_viz 等） (21 文件)
└── utils/         # 日志、训练循环等基础设施
```

## 数据流（BC + ACT 训练路径）

```text
LeRobotDataset (parquet + 视频)
   ↓ DataLoader (batch, num_workers)
observation.images.*  observation.state
   ↓ ACT preprocessing (normalization, MEAN_STD)
ACT Policy: ResNet18 backbone → transformer encoder/decoder（VAE 分支仅训练期）
   ↓ action chunk [B, chunk_size, action_dim]
Loss = L1(action) + kl_weight * KL(q(z|a) || N(0,1))
   ↓ optimizer (AdamW)
Checkpoint (pretrained_model/, optimizer state, 训练配置)
```

## 关键源码位置（实测存在性）

| 文件 | 存在 | 大小 | sha256(前16) |
| --- | --- | --- | --- |
| `policies/act/configuration_act.py` | yes | 8697 | 35a60c0dd3c811b1 |
| `policies/act/modeling_act.py` | yes | 35219 | 8d622935319af7d0 |
| `policies/act/processor_act.py` | yes | 1991 | 247d3b13c93568c1 |

> 说明：本文件由 `actlab repo` 依据真实扫描结果生成；参数级细节见 `logs/act_algorithm_notes.md`（S7，逐条标注源码行号）。

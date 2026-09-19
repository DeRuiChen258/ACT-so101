# TASK: LeRobot + ACT 具身智能实验闭环（C++ 为主语言）

> 创建时间：2026-09-19
> Workflow：unitree（机器人域）；训练/推理阶段按 contract 允许的 rl-infra / benchmark 技能执行，无需跨域申请
> 状态：**PLAN（待用户确认决策点 D1–D3 后进入执行）**

## 0. 输入事实源与路径覆盖

输入（优先级由高到低）：

1. 本机实测输出（本文件第 1 节，2026-09-19 复测）；
2. `Prompt/LeRobot_ACT_Cpp_Prompt.txt`（详细执行提示词 v1.0）+ `Prompt/LeRobot + ACT 具身智能实验搭建执行工单.txt`（原始工单）；
3. `Prompt/TASK.md`（上一轮「由工单生成提示词」任务的验收记录）；
4. 本机 clone 的 LeRobot 源码实测（`/tmp/lerobot_recon`，commit `c73cec56b04f50ebe9286c79905103866d970169`，v0.6.2）与官方 wheel/index 实测。

**路径覆盖**（用户指令优先于提示词中的默认路径 `/home/violet/Workspace/lerobot_act_experiment/`）：

| 层 | 路径 | 职责 |
| --- | --- | --- |
| 环境层 | `/home/violet/Workspace/IDE/Physical_AI/` | conda/venv 环境、LeRobot 源码 `repo/`、`third_party/libtorch`、数据集本体 `data/`、环境锁文件 |
| 实验层 | `/home/violet/Workspace/Code/Embedded_code/unitree_workspace/a_Visual_experiment/ACT/` | C++ 主工程、配置、训练产物、日志、证据、结果、报告 |

两层之间用**软链**衔接（禁止复制双份）：`ACT/data -> Physical_AI/data`、`ACT/repo -> Physical_AI/repo`。

## 1. 本机实测事实台账（2026-09-19，全部为本轮真实命令输出）

| 项 | 实测值 | 来源命令 |
| --- | --- | --- |
| OS / 内核 | Ubuntu 26.04.1 LTS / 7.0.0-31-generic | `grep PRETTY /etc/os-release` |
| GPU | RTX 5070 Laptop，8151 MiB，capability (12,0)=sm_120，空闲 65 MiB | `nvidia-smi` |
| Driver / CUDA UMD | 615.71.09 / 13.4 | `nvidia-smi` |
| CUDA Toolkit | 13.2，V13.2.86 | `nvcc --version` |
| gcc / cmake / ninja | 15.2.0 / 4.4.0-rc1 / 1.13.2 | 直接调用 |
| 内存 / 磁盘 | 30 G（可用 19 G）/ `/home` 可用 165 G | `free -g`、`df -h` |
| 已有 conda 环境 | base, agent, cuda_132, qml, rock_cpu, unitree_rt（当前激活） | `conda env list` |
| `cuda_132` | Python 3.12.13，torch 2.13.0+cu132，cuda_avail=True，arch_list 含 sm_120，torchvision/cv2/numpy/datasets 齐全，**lerobot 缺失** | 该 env 的 `python -c` |
| 全机 lerobot 现状 | `find /home/violet -iname "*lerobot*"` = **空**（无源码、无安装、无数据集缓存） | `find` |
| 已有 LibTorch | `/home/violet/Workspace/IDE/libtorch` = **2.12.0+cu132**（build-hash 7661cd9c…） | `cat build-version` |
| 截图工具 | `gnome-screenshot`/`scrot`/`spectacle` **缺失**；ImageMagick 7.1.2 `import`、`ffmpeg`、`xterm`、`xwininfo` 存在 | `command -v` |
| 会话 | Wayland（`XDG_SESSION_TYPE=wayland`, `DISPLAY=:0`） | 环境变量 |
| 网络 | gitcode 镜像 / ModelScope / PyPI / 清华源 / HF / GitHub 全部 200 | `curl -w %{http_code}` |

### 1.1 版本兼容链实测结论（本计划的技术命门）

| 事实 | 实测证据 | 影响 |
| --- | --- | --- |
| LeRobot v0.6.2 `requires-python>=3.12`，`torch>=2.7,<2.12.0`，`torchvision>=0.22,<0.27` | `pyproject.toml`（镜像 clone） | 训练侧 torch 上限被锁死 |
| v0.6.1/v0.6.0 同样为 `<2.12.0`；v0.5.1/v0.4.4 为 `<2.11.0` | `git show <tag>:pyproject.toml` | **没有任何 LeRobot 版本允许 torch 2.12/2.13** |
| `cuda_132` 内 torch=2.13.0+cu132；已有 LibTorch=2.12.0+cu132 | 实测 | 二者既互不匹配，也都不满足 LeRobot 约束 |
| cu132 索引只有 torch≥2.12；cu130 有 2.9.0/2.9.1/2.10.0/2.11.0/2.12+；cu128 有 2.7–2.11 | `download.pytorch.org/whl/<cu>/torch/` | torch 2.11.0+cu130 可满足约束且是 cu13x |
| torchvision 0.26.0+cu130 / +cu128 均存在 | 同上 | 与 torch 2.11.0 配对合法 |
| libtorch 可下载版本：cu132 → 2.12.0/2.12.1/2.13.0/2.14.0；cu130 → 2.9.x/2.10.0/2.11.0/2.12+ | `download.pytorch.org/libtorch/<cu>/` | 与训练侧对齐的 LibTorch 可获取（2.11.0+cu130 或 2.13.0+cu132 各 1.8 GB） |
| LeRobot 官方 `uv.lock` 锁定 torch 2.11.0+cu128 / torchvision 0.26.0+cu128 / torchcodec 0.11.1 | `uv.lock` | 2.11.0 是官方实际测试版本 |
| 数据集 `lerobot/svla_so101_pickplace`：codebase v3.0、so100_follower、50 episodes、11939 frames、fps 30、2 路相机 480×640×3(av1)、state/action 各 6 维 | HF `meta/info.json` 直读 | 与 v0.6.2（CODEBASE_VERSION=v3.0）同代，schema 可校验；ACT 输入为 2 图 + 6 维 state，输出 6 维 action |
| 截图可行性 | `import -window <xterm 窗口 id>` 成功产出真实截图（mean=0.0298，非黑屏）；`ffmpeg -f x11grab` 抓 Xwayland root 为全黑；GNOME Shell 截图 DBus 接口 AccessDenied | 本轮实测 | 证据方案改为「真实 xterm 窗口抓取」，非伪造、非补图 |

## 2. Objective

以 C++ 为主语言，在 Physical_AI 环境层搭好 LeRobot 训练栈与 LibTorch 推理栈，在 ACT 实验层真实跑通并留证：

`LeRobot → LeRobotDataset → BC → ACT → CUDA 训练 → Checkpoint → 新进程重载 → C++ 推理 → 离线评测 → 报告`

## 3. Requirements（硬约束，违反即失败）

1. **语言边界**：C++17 负责环境/硬件探测、配置校验、数据集 schema 审计、训练进程编排、资源监控、指标聚合、loss 曲线、证据与日志管线、checkpoint 校验、LibTorch 推理运行时、离线评测、清单与报告；Python 只允许 4 个受控调用点（`lerobot-train`、`scan_dataset_ref.py`、`export_act_torchscript.py`、ONNX 兜底），每次调用登记到 `logs/hybrid_boundary.md`。
2. **主入口 C++**：`actlab` 一条命令可驱动全流程；`main`/`CLI`/`进程编排`均为 C++，Python 只作为被调用子进程。
3. **C++ 推理硬门禁**：必须在 sm_120 上完成一次 C++ 侧前向；TorchScript 失败按 `torch.jit.trace → torch.export → ONNX Runtime(CUDA)` 降级并在报告记录降级链；全链失败标 BLOCKED，不得标 PASS。
4. **不动既有环境**：`cuda_132` 只读；任何新增包不得写入其中（复用只允许「继承 + 只读」方式，见 D1）。
5. **训练/推理同源**：C++ 侧归一化/反归一化必须复用 checkpoint 内 normalization statistics，禁止直觉重写。
6. **零虚构**：未实测项写 `NOT_MEASURED`；日志、loss、显存、延时、截图全部来自真实执行。
7. **评测类型**：无真机 → 只能声明 `Offline Evaluation`（+ 数据集回放），禁止声称真机或仿真成功率。
8. **不修改 LeRobot 业务代码**：需要绕过缺陷时在 `scripts/` 侧适配并记录原因/影响/回滚。
9. **既有产物保护**：实验层所在 git 仓库（root=`unitree_workspace`）当前有未提交改动，本任务只新增文件，不提交、不 reset、不覆盖他人改动。

## 4. 目录与职责（目标结构）

```text
Physical_AI/                     # 环境层（只放"可复用的环境资产"）
├── repo/                        # LeRobot 源码（GitCode 镜像 clone，只读使用）
├── data/                        # HF_LEROBOT_HOME：数据集本体 + hub 缓存
├── envs/                        # venv（复用 cuda_132 时的分层环境）或 env 记录
├── third_party/libtorch/        # 版本对齐的 LibTorch（仅在必须时下载）
├── configs/                     # conda_env.lock.yml / constraints.txt / env_manifest.json
└── logs/                        # 环境构建原始日志

ACT/                             # 实验层（实验的唯一事实源）
├── TASK.md  .agent/             # 任务台账、检查点、决策与失败记录
├── cpp/                         # C++17 主工程（include/ src/ apps/ tests/ CMakeLists.txt）
├── build/local  build/gpu/      # 双构建目标（无 GPU 逻辑 / 链接 LibTorch 推理）
├── configs/                     # act_baseline.yaml / paths.yaml / runtime.yaml（唯一超参源）
├── scripts/                     # b00–b06 受控 shell 脚本 + py/ 4 个受控 Python 脚本
├── data -> Physical_AI/data     # 软链，禁止复制双份
├── repo -> Physical_AI/repo     # 软链
├── artifacts/act_baseline/      # checkpoint、torchscript、onnx 兜底
├── outputs/train/               # lerobot-train 原始输出目录
├── logs/                        # 全量终端日志、指标 JSONL、分析文档（含 hybrid_boundary.md）
├── evidence/                    # 真实截图 + index.md + README.md（工具与限制声明）
├── results/                     # loss_curve.csv/png、inference_latency.csv、eval_summary.md
├── tests/                       # C++ 单元测试（不依赖 GPU/网络）
├── README.md                    # 最终实验报告正文
└── experiment_manifest.json     # 版本/硬件/配置哈希/产物 SHA256
```

## 5. 阶段计划（S0–S16，逐阶段门禁）

每个阶段固定节奏：`探测 → 校验 → 执行 → 验证退出码 → 存日志 → 存截图 → 更新 manifest → 汇报`。上一阶段非 PASS 不得进入下一阶段。

### P0 计划与门禁（当前阶段）

- [ ] 写 `TASK.md`、`.agent/{state,decisions,failures}.md`
- [ ] 复测事实台账（第 1 节），修正提示词中与实际不符的项
- [ ] 提交决策点 D1–D3 给用户确认
- 交付物：本文件 + `.agent/*`
- 门禁：用户确认 D1–D3

### P1 C++ 工程骨架与双构建（提示词 §4.2）

- [ ] 生成 `cpp/` 全模块：`cli/main/logging/proc/metrics/manifest/sha256/gpu_check/dataset_schema/tensor_utils/config_loader/path_utils/yaml_min` + `stages/{env,repo,install,dataset,train,verify,eval,evidence}` + `inference/{act_runtime,temporal_ensemble}` + `apps/{actlab_main,infer_act_main,gpu_smoke_main}` + `tests/*`
- [ ] `cpp/CMakeLists.txt`：C++17、`-Wall -Wextra`、`find_package(Torch REQUIRED)`、`TORCH_LIBRARIES`、`build/local`（无 GPU）与 `build/gpu` 双目标
- [ ] `scripts/b02_build.sh`：本地导出 `LD_LIBRARY_PATH`（不污染全局），两个 build 目录都编译通过
- [ ] 单测（`test_sha256` / `test_metrics_parse` / `test_schema_validate` / `test_config_loader`）全绿，且不依赖 GPU/网络
- 门禁：`build/local` + `build/gpu` 编译通过 + 单测全绿

### P2 S0 环境探测与门禁

- [ ] `actlab env`：复测 OS/GPU/nvcc/python/conda/git/ffmpeg，写 `logs/00_hardware_check.txt`
- [ ] `gpu_smoke`：`torch::cuda::is_available()`、capability、一次 GPU matmul、allocated/reserved 显存
- [ ] LibTorch 版本/ABI 与训练侧对齐判定（见 D2）
- [ ] 截图：真实 xterm 运行上述命令后抓窗（`import -window <id>`）
- 门禁：`torch::cuda::is_available()==true`、capability 12.0、LibTorch 判定结论落盘

### P3 S1–S2 源码获取与结构分析

- [ ] `actlab repo`：GitCode 镜像 clone 到 `Physical_AI/repo`（失败才用官方 GitHub），记录 commit/branch、解析 `pyproject.toml`
- [ ] `find src/lerobot -maxdepth 3 -type f` + ACT 实现定位（`policies/act/{configuration_act,modeling_act,processor_act}.py`）
- [ ] 产出 `logs/architecture_analysis.md`（Dataset→DataLoader→ACT Policy→Loss→Optimizer→Checkpoint 数据流图）
- 门禁：commit 已记录；ACT 实现路径已确认；分析与源码逐条对应

### P4 S3 训练环境安装与版本锁定（决策点 D1）

- [ ] 按 D1 选定路径建训练环境（复用优先 / 新建 `lerobot_act`）
- [ ] 安装 LeRobot（`pip install -e ".[training]"`，按缺失项追加 extra，禁止 `.`all`）
- [ ] 验证：`import lerobot`、`lerobot-train --help` 含 `--dataset.repo_id / --policy.type / --policy.device / --output_dir`、`policy.type=act` 可解析
- [ ] 锁定：`conda env export --no-builds > configs/conda_env.lock.yml` + `pip freeze > logs/pip_freeze.txt` 成对产出
- [ ] 登记 Python 调用点到 `logs/hybrid_boundary.md`
- 门禁：四项 CLI 参数存在、ACT 可解析、torch.cuda 可用且 arch_list 含 sm_120

### P5 S4–S6 数据集获取 / schema 校验 / 可视化

- [ ] 下载 `lerobot/svla_so101_pickplace`（ModelScope 优先，其次 HF 镜像）到 `Physical_AI/data`；失败才降级 `lerobot/pusht` 并写明理由
- [ ] `scan_dataset_ref.py` 实测 schema：长度、episode 数、fps、相机键、state/action 维度、图像 shape
- [ ] C++ `actlab dataset --schema`：必填 feature 与维度自洽校验（不符合即停）
- [ ] `actlab dataset --viz`：`lerobot-dataset-viz`（无 GUI 依赖路径：`--save` 产出 `.rrd`）+ 由真实帧生成 PNG 拼图（真实数据，非示意图）
- 门禁：数据集落盘 + 版本入 manifest；schema 校验通过；可视化有真实产物

### P6 S7–S8 算法笔记与配置冻结

- [ ] `logs/act_algorithm_notes.md`：BC 目标、ACT action chunking、Transformer 编解码、CVAE/KL 与重建损失、temporal ensembling（全部标注源码文件与行号；实测字段：`n_obs_steps=1, chunk_size=100, n_action_steps=100, dim_model=512, n_heads=8, dim_feedforward=3200, n_encoder_layers=4, n_decoder_layers=1, use_vae=True, latent_dim=32, n_vae_encoder_layers=4, kl_weight=10.0, dropout=0.1, vision_backbone=resnet18, temporal_ensemble_coeff=None`）
- [ ] `configs/act_baseline.yaml` 冻结超参（batch size、lr、optimizer、weight decay、chunk/action/obs steps、steps、seed、device、precision、num_workers、checkpoint 周期）
- [ ] `actlab train --dry-run`：字段级校验，缺失/越界拒绝启动；配置哈希写 manifest
- [ ] 截图：`10_act_config.png`
- 门禁：非法配置被拒绝；配置哈希入 manifest

### P7 S9–S11 训练（smoke → release）

- [ ] S9 最小初始化：Dataset→Policy→Forward→Loss→Backward，打印张量 shape、`isnan/isinf` 均为 false、梯度存在、显存 allocated/reserved
- [ ] S10 smoke 训练：100–500 steps，产出至少一个 checkpoint
- [ ] S11 正式训练：先观察显存/利用率/吞吐再定 batch size；OOM 按 `batch → 分辨率 → num_workers → AMP` 顺序降级并记录每次尝试
- [ ] 指标：`actlab metrics` 解析训练日志 → `logs/metrics.jsonl` + `results/loss_curve.csv` + `results/loss_curve.png`
- [ ] 截图：`12_training_start` / `13_training_progress` / `14_training_loss` / `15_checkpoint`
- 门禁：训练真实发生、loss 有真实数据、checkpoint 周期落盘、峰值显存与吞吐为实测值

### P8 S12–S13 checkpoint 校验与新进程重载（硬门禁）

- [ ] `actlab verify --artifact`：校验权重、配置、normalization statistics、训练状态，计算 SHA256
- [ ] `export_act_torchscript.py`（受控调用点 b）：重建 policy → 断言与训练同源 → 导出 TorchScript → 打印输入/输出签名
- [ ] `infer_act`（独立 C++ 新进程）：加载 TorchScript、设备 CUDA、前向成功、动作输出有限值
- [ ] 若 TorchScript 失败：按降级链执行并记录（`backup_export_fallback.py`）
- 门禁：新进程内完成加载与 GPU 前向，输出 shape 与训练侧一致；否则 BLOCKED

### P9 S14 推理（C++ 主路径）

- [ ] `actlab infer`：Dataset observation → ACT → action chunk → 预测；warmup 后多次采样统计延时（p50/p95、min/max、样本数）
- [ ] `offline_baseline_eval.py`（受控调用点 d）作为 Python 参考实现
- [ ] C++ 与 Python 输出差异量化（max abs/rel diff + 结论解释），写入 `results/eval_summary.md`
- [ ] 截图：`16_model_reload` / `17_inference`
- 门禁：输入/输出 shape、延时分布、差异量化均有实测值

### P10 S15 评测

- [ ] `actlab eval`：当前可用方案 = **Offline Evaluation**（dataset replay + 离线动作误差）
- [ ] 显式声明评测类型；仿真/真机成功率一律 `NOT_MEASURED`
- [ ] 截图：`18_evaluation.png`
- 门禁：评测类型显式声明、未测项为 NOT_MEASURED

### P11 S16 证据、报告与收尾

- [ ] `actlab evidence`：截图—日志配对校验、生成 `evidence/index.md`（编号/阶段/文件/说明/时间戳/哈希）；缺失即报错
- [ ] `evidence/README.md`：截图工具与可用性说明（本轮为真实窗口抓取；若某阶段不可用必须显式 `SCREENSHOT_UNAVAILABLE`）
- [ ] `actlab report`：生成 `README.md`（环境表/数据集表/超参表/训练表/推理表/C++ 模块表/证据统计/类型声明/最终状态）+ 结尾总结块
- [ ] `experiment_manifest.json`：版本、硬件、配置哈希、产物 SHA256
- [ ] `.agent/*` 记忆沉淀
- 门禁：每阶段至少一份日志；截图缺失有显式声明；指数与哈希可核验

## 6. 验证门禁（全绿才允许宣告完成）

- [ ] 构建：`build/local` 与 `build/gpu` 均通过；`tests/` 单测全绿（不依赖 GPU/网络）
- [ ] 环境：GPU/CUDA/PyTorch 实测可用且 sm_120 命中；训练侧 torch 与 C++ 侧 LibTorch 的版本/ABI 关系已复测并落盘
- [ ] 框架：`import lerobot` 成功；`lerobot-train --help` 关键参数存在；`policy.type=act` 可解析
- [ ] 数据：Dataset 可加载、schema 校验通过、可视化成功，三项各有证据
- [ ] ACT：模型可初始化、前向成功、loss 有限、反向有梯度、CUDA 执行成功
- [ ] 训练：真实发生、loss 有记录、checkpoint 周期落盘、吞吐与峰值显存有实测值
- [ ] 重载：新进程加载 TorchScript 成功并在 CUDA 前向成功
- [ ] 推理：C++ 主路径完成推理，shape 与延时均为实测
- [ ] 一致性：C++ 与 Python 参考实现差异有量化结论与解释
- [ ] 证据：图 + 日志配对齐全，索引与哈希可核验
- [ ] 诚实性：未测项为 `NOT_MEASURED`，评测类型显式声明

## 6.1 资源与工期预算（估算，执行时用实测值替换）

| 项目 | 估算 | 说明 |
| --- | --- | --- |
| 网络下载 | D1-A ≈ 3.0 GB（torch+torchvision+lerobot 依赖）+ 1.8 GB（LibTorch，若 D2-A 通过可省）；D1-B ≈ 1.0–1.5 GB（lerobot 依赖）+ 1.8 GB + 已有 torch | 已有资源全部复用，不重复下载 |
| 磁盘占用 | 数据集 ~0.1–1 GB（实测 tree 86 MB，下载后复核）+ 环境 5–8 GB + 产物（checkpoint ~200 MB/个） | `/home` 可用 165 GB，充裕 |
| 工期 | P1 骨架 1.5–2 h；P2–P3 环境 0.5–1 h；P4 安装 0.5–1.5 h（视 D1）；P5 数据 0.5–1 h；P6 笔记与配置 0.5 h；P7 训练 1.5–3 h；P8–P10 重载/推理/评测 1–1.5 h；P11 报告 0.5 h | 合计约 7–11 h 机器时间；每阶段结束实时汇报 |

## 7. 风险与降级矩阵

| 风险 | 触发信号 | 降级/处置 |
| --- | --- | --- |
| LeRobot 与 torch 2.13 不兼容（仅 D1-B 路径） | import / smoke 前向报 API 错误 | 立即按提示词 §3.2 切回新建 `lerobot_act`（torch 2.11.0+cu130） |
| 训练侧 torch 与已有 LibTorch 2.12.0 不匹配 | C++ 加载/前向失败或 schema 报错 | 下载对齐版 LibTorch（2.11.0+cu130 或 2.13.0+cu132）到 `Physical_AI/third_party/libtorch` |
| torchcodec 与 torch 版本/FFmpeg 8 不兼容 | 解码 import 或读取失败 | 不装 torchcodec，走 PyAV 后端（`get_safe_default_video_backend()` 已实现回退） |
| CUDA OOM（8151 MiB 上限） | 训练 step 抛 OOM | 按 batch → 分辨率 → workers → AMP 顺序降级并记录 |
| ModelScope/HF 下载失败 | 传输错误 | 换源重试 → 最后降级 `lerobot/pusht` 并写明理由 |
| TorchScript 导出失败 | `torch.jit.trace` 报错 | `torch.export` → ONNX Runtime(CUDA)，降级链写入报告与 manifest；全失败标 BLOCKED |
| Wayland 无法整屏截图 | `import -window root` 失败（已实测） | 只抓真实 xterm 窗口（已实测可行）；仍不可用则 `SCREENSHOT_UNAVAILABLE` + 日志/命令/输出/配置/时间戳 |
| 数据集与 LeRobot 版本代差 | CODEBASE_VERSION 不符 | 以 `scan_dataset_ref.py` 实测输出为准，必要时改用同代数据集 |
| viz extra（rerun/foxglove）装不上或无法起 GUI | `lerobot-dataset-viz` 报缺依赖/无显示 | 用 `--save` 产出 `.rrd` + 由 C++ 解码真实帧生成 PNG 拼图，并把「可视化工具降级」写入证据说明（真实数据，不伪造） |

## 8. 待确认决策点（阻塞执行）

| ID | 决策 | 备选 | 建议默认 |
| --- | --- | --- | --- |
| D1 | 训练环境策略 | A：新建 conda env `lerobot_act`（torch 2.11.0+cu130，官方锁定版本，满足 `<2.12`，干净可复现，需下载 ~3 GB）<br>B：**复用优先**：在 `cuda_132` 之上建 `--system-site-packages` venv，继承 torch 2.13.0+cu132，LeRobot 以 `--no-deps + 约束文件` 安装（不写入 cuda_132，下载最少，但与 LeRobot 声明约束不符，需 smoke 验证，失败即回退 A） | **B → 失败回退 A**（符合用户「已有不重复配置」与提示词「先复用后新建」） |
| D2 | C++ 侧 LibTorch | A：先实测已有 `/home/violet/Workspace/IDE/libtorch`（2.12.0+cu132）能否加载训练侧导出的 TorchScript（跨版本前向兼容测试）<br>B：直接下载与训练侧严格对齐的 LibTorch（1.8 GB） | **A → 失败回退 B** |
| D3 | 数据集与放置 | A：`lerobot/svla_so101_pickplace`（ModelScope 优先）放 `Physical_AI/data`，`ACT/data` 软链<br>B：放 `ACT/data` 实体目录 | **A**（环境层资产集中、避免双份） |

## 9. 计划自审（Plan Review）

对照提示词逐节核对，本轮自审发现并已修正的问题：

1. **提示词 §3.1 的数据有两处已过时**：真实 `nvcc` 为 13.2（一致），但「cuda_132 可复用于训练」与「LibTorch 2.12.0+cu132 与训练侧同版本」两条在 LeRobot 实际约束下**不可能同时成立**（第 1.1 节实测）。已把该冲突升级为决策点 D1/D2，而不是照抄提示词直接开工。
2. **提示词 §3.2 要求「cu13x 轮子」与 §3.3「同 major/minor」在 cu132 索引上无解**（无 torch<2.12 的 cu132 轮子）。方案：训练侧 cu130 + LibTorch 对齐，或复用 cuda_132 的 cu132 组合，两条都已列出并给出验证手段。
3. **提示词假设有可用截图工具**：实测 gnome-screenshot/scrot/spectacle 均缺失，整屏抓取失败。已实测出可行替代（真实窗口抓取）并写入证据方案与降级条款，避免执行期才发现拦路。
4. **提示词默认实验根目录**为 `~/Workspace/lerobot_act_experiment/`，与用户新指令冲突；已按用户指令改为两层布局并用软链保持提示词的目录语义。
5. **工作区是脏 git 仓库**（`unitree_workspace` 有未提交改动）：已加入约束「只新增、不提交、不覆盖」。
6. 覆盖度核对：提示词 §2（技术约束）→ 本文件 §3；§3（环境）→ P2/P4；§4（目录与文件职责）→ §4；§5（阶段门禁）→ §5；§6（验证门禁）→ §6；§7（异常）→ §7；§8（报告与产物）→ P11 + §6；§9（执行纪律）→ §5 节奏与阶段汇报要求。
7. 仍存在的不确定性（不隐瞒）：D1-B 下 LeRobot×torch 2.13 兼容性未知；跨版本 TorchScript 加载成功率未知；两者都已有明确回退路径，不会导致任务假 PASS。

## 10. Checklist

- [x] 任务理解与事实勘察（Stage 1）
- [x] 读取工作流 README + unitree workflow + unitree contract + registry 解析
- [x] 创建 `TASK.md` 与 `.agent/` 记录
- [ ] 用户确认 D1–D3
- [ ] 环境层构建（P3）与 C++ 骨架（P1）
- [ ] 数据/训练/重载/推理/评测全链执行
- [ ] 验证门禁全绿 + 报告与 manifest 产出
- [ ] 记忆沉淀与最终汇报

## Current Stage

`P0（计划与门禁）`：TASK.md 已落盘，事实台账已复测，等待用户对 D1–D3 的确认；确认后从 `P1 C++ 骨架` 继续。中断恢复：先读本文件 + `.agent/state.md`，从 P1 的第一个未勾选项继续，不重跑已完成阶段。

## Known Issues

- 提示词 §3.1 的「cuda_132 可复用 + 已有 LibTorch 同版本」组合与 LeRobot 实际约束冲突，已在 §1.1 与 §9 记录，等待 D1/D2 决策。
- `import -window root` 在本机 Wayland 下不可用（已实测），证据只走真实窗口抓取或 `SCREENSHOT_UNAVAILABLE`。
- 无真机：评测上限为 Offline Evaluation，仿真成功率同样不可测，将全部标 `NOT_MEASURED`。

## 备注

- 关键决策与理由：`.agent/decisions.md`
- 失败与解决记录：`.agent/failures.md`
- 检查点状态：`.agent/state.md`

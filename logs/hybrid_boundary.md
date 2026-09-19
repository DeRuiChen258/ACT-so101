# C++ / Python 边界登记表

> 契约来源：提示词 §2.1（C++ 为唯一主语言）与 §4.3（受控 Python 脚本清单）。
> 规则：每次 Python 进程调用都必须登记；未登记的调用视为违规。
> 说明：提示词 §2.1 列出 4 个允许调用点，§4.3 又列出 5 个脚本，两者不完全一致。
> 本表按"**进程级调用点 ≤4 + 库级复用**"解释并逐条登记，偏差在 `.agent/decisions.md` 记录。

## 1. 进程级 Python 调用点（全部由 C++ `actlab` 发起）

| # | 调用点 | 脚本 / 命令 | 理由 | 输入 | 输出 | 可否被 C++ 替代 |
| --- | --- | --- | --- | --- | --- | --- |
| (a) | S10–S11 训练 | `python -m lerobot.scripts.lerobot_train` | LeRobot 训练内核为 Python/PyTorch，禁止在 C++ 重写 | `configs/act_baseline.yaml` 派生的 CLI 参数 | `outputs/train/<job>/checkpoints/**`、逐行指标日志 | 否（框架内核） |
| (b) | S4–S6 数据 | `scripts/py/scan_dataset_ref.py` | 数据集加载/解码依赖 LeRobot 的 Python 栈（parquet + 视频解码） | `--repo-id/--root/--chunk-size/--viz-out` | schema 统计、真实样本 `.npy` 与 `samples_meta.json`、`.rrd` 可视化 | 否（解码栈） |
| (c) | S13 导出 | `scripts/py/export_act_torchscript.py` | 只有 Python 侧能读 safetensors 权重并 trace 出 TorchScript | checkpoint 目录、样本目录 | `act_policy.pt`、`equivalence.json`、`reference_outputs.json` | 否（导出器本身） |
| (d) | S13 兜底 | `scripts/py/backup_export_fallback.py` | TorchScript 失败时的降级链（`torch.export` → ONNX） | checkpoint、device | `act_policy.exported.pt2` 或 `act_policy.onnx` | 否 |
| (e) | S9 初始化探针 | `scripts/py/act_forward_probe.py` | Dataset→Policy→Loss→Backward 的最小验证必须在训练同源栈内执行 | 数据集、`configs/act_baseline.yaml` | shape/loss/梯度/显存实测打印 | 部分（可退化为 `--steps=1` 训练，但会失去逐项证据） |
| (f) | 可视化（非训练链） | `scripts/py/so101_mujoco_view.py`（经 `scripts/b07_so101_mujoco_viz.sh` 调用） | SO-101 的 3D 可视化：用 MuJoCo Menagerie 模型 + 数据集真实关节轨迹回放，并按抓取点放置物体 | 数据集 parquet、`configs/paths.yaml`、模型 XML | PNG 关键帧、拼图、mp4、`grasp_info.json`、桌面仿真窗口 | 否（可视化本身即目的；不参与训练/评测结论） |

> (e) 属于提示词 §5 阶段表明确要求（S9），但 §2.1 的 4 点清单未提供载体；
> 本任务将其作为第 4 个"必选"调用点登记，(d) 仅在失败降级时触发。
> (f) 属于事后追加的可视化能力（用户要求），**不进入训练/评测链路**，仅读取已落盘数据，
> 因此不改变 (a)–(d) 的边界契约；其产物登记在 `evidence/index.md` 第 20 行与 manifest 的 `so101_viz` 条目。

## 2. 库级复用（不单独起进程，故不额外占用调用点）

| 模块 | 被谁 import | 作用 |
| --- | --- | --- |
| `scripts/py/offline_baseline_eval.py` | 被 `export_act_torchscript.py` import | Python 参考前向实现，产出 `reference_outputs.json` 供 C++ `actlab eval` 交叉比对 |

## 3. C++ 侧职责（不调用 Python）

| 阶段 | C++ 模块 | 职责 |
| --- | --- | --- |
| S0 | `stage_env` + `gpu_check` + `apps/gpu_smoke_main.cpp` | 硬件/CUDA/LibTorch 探测与门禁 |
| S1–S2 | `stage_repo` | 源码获取、commit 记录、结构扫描、分析文档生成 |
| S3 | `stage_install` | pip 编排、`--help` 参数校验、版本锁定 |
| S4–S6 | `stage_dataset` + `dataset_schema` + `npy_io` | schema 审计、真实帧拼图（不依赖 GUI） |
| S8 | `stage_train`（dry-run 路径） | 配置字段级校验、配置哈希 |
| S12 | `stage_verify` | checkpoint 完整性校验与 SHA256 |
| S13 | `apps/infer_act_main.cpp` + `inference/act_runtime` | 新进程加载 TorchScript 并在 CUDA 前向（硬门禁） |
| S14 | `stage_eval` + `inference/*` | 推理、延时统计、与 Python 参考值比对 |
| S15–S16 | `stage_eval` / `stage_evidence` / `stage_report` | 离线评测、证据索引、报告与 manifest |

## 4. 复核方式

- 每次进程调用由 `RunContext::run_stack()` 写入 `experiment_manifest.json` 的 `commands` 数组（含时间戳）。
- 每个阶段的原始输出落盘到 `logs/`，与 `evidence/` 中的截图按编号配对（见 `evidence/index.md`）。

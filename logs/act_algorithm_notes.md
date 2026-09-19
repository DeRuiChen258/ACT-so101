# ACT 算法笔记（以当前仓库源码为准）

> 唯一事实来源：`/home/violet/Workspace/IDE/Physical_AI/repo` @ commit `c73cec56b04f50ebe9286c79905103866d970169`（LeRobot v0.6.2，GitCode 镜像）。
> 论文：*Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware*（ACT）。
> 本文件所有参数与公式均可回溯到下列文件与行号；未标注来源的推断一律不写。

## 1. Behavior Cloning（BC）基本形式

BC 学习的是条件策略 π_θ(a_t | o_t)：以专家演示数据集 D = {(o_t, a_t)} 为监督信号，
最小化预测动作与专家动作之间的偏差。训练循环（`src/lerobot/scripts/lerobot_train.py`）：

```text
LeRobotDataset → DataLoader → ACTPolicy.forward(batch) → loss → loss.backward() → AdamW.step()
```

关键点：loss 只在"有效动作位"上统计（见 §4 的 `action_is_pad` 掩码）。

## 2. ACT 的核心：Action Chunking

ACT 不预测单步动作 a_t，而是一次预测未来 k 步的动作序列：

```text
observation_t → [a_t, a_{t+1}, …, a_{t+k-1}]        (k = chunk_size = 100)
```

动机：逐步执行的传统 BC 会因误差累积与人类演示中的停顿而漂移；预测动作块让策略在
多步尺度上一致，显著缓解累积误差（论文 §III）。

本轮配置（`configs/act_baseline.yaml`，取自 `configuration_act.py` 的实际字段名与默认值）：

| 参数 | 值 | 源码位置 |
| --- | --- | --- |
| `n_obs_steps` | 1 | `policies/act/configuration_act.py:84`（且 `__post_init__` 对 ≠1 直接报错，见 `:171-174`） |
| `chunk_size` | 100 | `configuration_act.py:85` |
| `n_action_steps` | 16（本轮设置） | `configuration_act.py:86`（默认 100）；约束 `n_action_steps ≤ chunk_size` 见 `:180-184` |
| `vision_backbone` | resnet18 | `configuration_act.py:96`（默认）；实现见 `modeling_act.py:328-330` |

## 3. 结构与数据流（Transformer 编解码）

```text
图像 (2 路相机) ─→ ResNet18 backbone ─→ 特征图展平 + 位置编码 ┐
机器人状态 observation.state ─→ 线性投影 ─────────────────────┤→ Transformer Encoder
latent z（仅训练期，见 §5）───────────────────────────────────┘
                                                             ↓
                                       Transformer Decoder（action queries）
                                                             ↓
                                              action chunk [B, chunk_size, action_dim]
```

代码位置：

- `ACTPolicy`（`modeling_act.py:42`）：策略封装；`select_action`（`:103`）、`predict_action_chunk`（`:128`）、`forward`（`:139`）。
- `ACT` 主干（`modeling_act.py:260`，`__init__` `:295`，`forward` `:382`）。
- `ACTEncoder` / `ACTEncoderLayer`（`:517` / `:536`）、`ACTDecoder` / `ACTDecoderLayer`（`:575` / `:598`）。
- 位置编码：`ACTSinusoidalPositionEmbedding2d`（`:688`）。
- `predict_action_chunk` 会把每个相机键组装为 `batch[OBS_IMAGES]` 列表后送入主干（`:132-136`）。

## 4. 损失函数（逐行对齐实现）

`modeling_act.py:139-166`：

```python
actions_hat, (mu_hat, log_sigma_x2_hat) = self.model(batch)
abs_err    = F.l1_loss(batch[ACTION], actions_hat, reduction="none")
valid_mask = ~batch["action_is_pad"].unsqueeze(-1)
num_valid  = valid_mask.sum() * abs_err.shape[-1]
l1_loss    = (abs_err * valid_mask).sum() / num_valid.clamp_min(1)     # 重建项（L1）
mean_kld   = (-0.5 * (1 + log_sigma_x2_hat - mu_hat.pow(2) - log_sigma_x2_hat.exp())).sum(-1).mean()
loss       = l1_loss + mean_kld * self.config.kl_weight                # kl_weight = 10.0
```

即 ACT 论文中的 **L = L_reconstruction + β · D_KL(q(z|a, o) ‖ N(0, I))**，本轮 β=`kl_weight`=10.0
（`configuration_act.py:113` 附近默认值）。日志中的 `loss` 即上式总值，子项写入 `loss_dict`。

## 5. CVAE / 潜在变量

- `use_vae=True` 时额外训练一个 Transformer 编码器（VAE encoder），把专家动作块编码为
  `N(mu, sigma)`，训练时采样 z 参与解码；推理时 z 置零（先验均值），因此不需要示范动作。
- 相关实现：`use_vae`（`configuration_act.py:105`）、`latent_dim`（`:106`）、
  `n_vae_encoder_layers`（`:107`）；VAE 分支在 `ACT.forward`（`modeling_act.py:382` 起）中生效。
- 注意：LeRobot 明确说明该 VAE 编码器与主干 Transformer encoder 不是同一个模块
  （`configuration_act.py:69-71` 的文档字符串）。

## 6. Temporal Ensembling（本基线未启用）

实现：`ACTTemporalEnsembler`（`modeling_act.py:169-258`），权重 `w_i = exp(-coeff · i)`
（`i=0` 为最旧动作，见 `:213`），并按在线方式等价于加权平均（`:223-258`）：

```python
ensembled *= w_cumsum[count-1]; ensembled += actions[:, :-1] * w[count]
ensembled /= w_cumsum[count];  count = clamp(count+1, max=chunk_size)
ensembled = cat([ensembled, actions[:, -1:]], dim=1)
```

- 约束：使用该功能时 `n_action_steps` 必须为 1（`configuration_act.py:171-174`）。
- 本基线取 `temporal_ensemble_coeff: 0.0`（不启用），采用 action chunking（`n_action_steps=16`）执行。
  C++ 侧已按上述逐行等价实现（`cpp/src/inference/temporal_ensemble.cpp`），可在启用时复用。

## 7. 与 C++ 推理的同源边界

C++ 侧不重写归一化：`scripts/py/export_act_torchscript.py` 把 checkpoint 自带的
preprocessor（归一化）+ policy + postprocessor（反归一化）一起 trace 成 TorchScript，
并在导出时与官方 pipeline 做数值比对（`torchscript/equivalence.json`）。
因此 C++ 只需提供与训练一致量纲的输入：图像 float32 `[B,3,H,W]`（数据集侧的 [0,1] 浮点）
与状态 `[B,D]`，输出为真实单位动作块。

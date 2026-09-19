#!/usr/bin/env python
"""受控 Python 调用点：ACT 最小初始化探针（S9）。

只在真实数据集样本上做一次 Dataset → Policy → Forward → Loss → Backward，
打印张量 shape、loss 有限性、梯度存在性与显存占用，全部为实测值。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml


def load_config(config_dir: Path) -> dict:
    with (config_dir / "act_baseline.yaml").open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--root", required=True, help="HF_LEROBOT_HOME 父目录")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--config-dir", required=True)
    args = parser.parse_args()

    config = load_config(Path(args.config_dir))
    policy_cfg = config["policy"]
    training_cfg = config["training"]

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.factory import make_policy, make_pre_post_processors

    dataset_root = Path(args.root)
    dataset = LeRobotDataset(args.repo_id, root=dataset_root)
    fps = float(dataset.meta.fps)
    chunk_size = int(policy_cfg["chunk_size"])
    dataset = LeRobotDataset(
        args.repo_id,
        root=dataset_root,
        delta_timestamps={"action": [i / fps for i in range(chunk_size)]},
    )
    print(f"dataset: frames={len(dataset)} episodes={dataset.meta.total_episodes} fps={fps}")
    print(f"camera_keys={dataset.meta.camera_keys}")

    cfg = ACTConfig(
        n_obs_steps=int(policy_cfg["n_obs_steps"]),
        chunk_size=chunk_size,
        n_action_steps=int(policy_cfg["n_action_steps"]),
        vision_backbone=str(policy_cfg["vision_backbone"]),
        dim_model=int(policy_cfg["dim_model"]),
        dim_feedforward=int(policy_cfg["dim_feedforward"]),
        n_heads=int(policy_cfg["n_heads"]),
        n_encoder_layers=int(policy_cfg["n_encoder_layers"]),
        n_decoder_layers=int(policy_cfg["n_decoder_layers"]),
        use_vae=bool(policy_cfg["use_vae"]),
        latent_dim=int(policy_cfg["latent_dim"]),
        n_vae_encoder_layers=int(policy_cfg["n_vae_encoder_layers"]),
        dropout=float(policy_cfg["dropout"]),
        kl_weight=float(policy_cfg["kl_weight"]),
        temporal_ensemble_coeff=None,
        optimizer_lr=float(training_cfg["optimizer_lr"]),
        optimizer_weight_decay=float(training_cfg["optimizer_weight_decay"]),
    )
    cfg.device = args.device

    device = torch.device(args.device)
    policy = make_policy(cfg, ds_meta=dataset.meta)
    policy.to(device)
    policy.train()
    preprocessor, _ = make_pre_post_processors(policy_cfg=cfg, dataset_stats=dataset.meta.stats)

    item = dataset[0]
    batch = {key: value.unsqueeze(0) for key, value in item.items() if isinstance(value, torch.Tensor)}
    if "action_is_pad" not in batch:
        batch["action_is_pad"] = torch.zeros(
            batch["action"].shape[:2], dtype=torch.bool, device=batch["action"].device
        )
    for key in dataset.meta.camera_keys:
        if batch[key].dtype == torch.uint8:
            batch[key] = batch[key].to(torch.float32) / 255.0
    batch = preprocessor(batch)
    batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}

    print("\n=== input batch ===")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key:32s} shape={tuple(value.shape)} dtype={value.dtype} device={value.device}")

    torch.cuda.reset_peak_memory_stats(device) if device.type == "cuda" else None
    loss, loss_dict = policy(batch)
    print("\n=== forward ===")
    print(f"  loss = {float(loss):.6f}  finite={bool(torch.isfinite(loss))}")
    for key, value in loss_dict.items():
        print(f"  loss_dict[{key}] = {value}")

    loss.backward()
    grad_norm = 0.0
    with_grad = 0
    for param in policy.parameters():
        if param.grad is not None:
            grad_norm += float(param.grad.detach().norm() ** 2)
            with_grad += 1
    grad_norm = grad_norm**0.5
    print("\n=== backward ===")
    print(f"  params_with_grad = {with_grad}")
    print(f"  total_grad_norm  = {grad_norm:.6f}")

    if device.type == "cuda":
        allocated = torch.cuda.memory_allocated(device) / 1024**2
        reserved = torch.cuda.memory_reserved(device) / 1024**2
        peak = torch.cuda.max_memory_allocated(device) / 1024**2
        print("\n=== cuda memory (MiB) ===")
        print(f"  allocated={allocated:.1f} reserved={reserved:.1f} peak_allocated={peak:.1f}")

    ok = bool(torch.isfinite(loss)) and with_grad > 0 and grad_norm > 0
    print(f"\nPROBE_{'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""受控 Python 调用点 (b)：LeRobotDataset 加载 / schema 统计 / 真实样本导出 / 可视化触发。

设计约束（Prompt §2.1）：
  - 只用当前仓库实际 API（LeRobotDataset），不引用旧教程；
  - 只做"数据侧"工作：加载、统计、导出真实样本、触发官方可视化；
  - 所有输出打印到 stdout，由 C++ 侧 actlab 落盘留证（logs/07_dataset_download.txt）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


def build_dataset(repo_id: str, dataset_root: Path, chunk_size: int, camera_steps: int):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    # 先按"无 delta"打开一次以获得 fps
    base = LeRobotDataset(repo_id, root=dataset_root)
    fps = float(base.meta.fps)
    delta_timestamps = {
        "action": [i / fps for i in range(chunk_size)],
        "observation.images.up": [i / fps for i in range(camera_steps)],
    }
    dataset = LeRobotDataset(repo_id, root=dataset_root, delta_timestamps=delta_timestamps)
    return dataset


def describe(dataset) -> dict:
    meta = dataset.meta
    camera_keys = list(meta.camera_keys)
    features = {key: dict(spec) for key, spec in meta.features.items()}
    return {
        "repo_id": meta.repo_id,
        "codebase_version": str(getattr(meta, "codebase_version", "unknown")),
        "robot_type": str(getattr(meta, "robot_type", "unknown")),
        "fps": float(meta.fps),
        "num_frames": int(meta.total_frames),
        "num_episodes": int(meta.total_episodes),
        "num_tasks": int(getattr(meta, "total_tasks", -1)),
        "camera_keys": camera_keys,
        "features": features,
    }


def export_samples(dataset, out_dir: Path, count: int) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    camera_keys = list(dataset.meta.camera_keys)
    total = len(dataset)
    if total == 0:
        raise RuntimeError("dataset is empty")
    indices = [int(round(i * (total - 1) / max(1, count - 1))) for i in range(count)]

    state_dim = action_dim = chunk_size = None
    written = 0
    for position, index in enumerate(indices):
        item = dataset[index]
        prefix = f"sample_{position:03d}"
        for cam_index, key in enumerate(camera_keys):
            image = item[key]
            if image.dtype == torch.uint8:
                image = image.to(torch.float32) / 255.0
            image = image.to(torch.float32).contiguous()
            np.save(out_dir / f"{prefix}_cam{cam_index}.npy", image.numpy())
        state = item["observation.state"].to(torch.float32).contiguous()
        np.save(out_dir / f"{prefix}_state.npy", state.numpy())
        if "action" in item:
            action = item["action"].to(torch.float32).contiguous()
            np.save(out_dir / f"{prefix}_action.npy", action.numpy())
            chunk_size = int(action.shape[0]) if action.dim() > 1 else 1
            action_dim = int(action.shape[-1])
        state_dim = int(state.shape[-1])
        written += 1

        if position == 0:
            sample = dataset[index]
            for cam_index, key in enumerate(camera_keys):
                tensor = sample[key]
                print(
                    f"[sample] {key}: shape={tuple(tensor.shape)} dtype={tensor.dtype} "
                    f"min={float(tensor.min()):.4f} max={float(tensor.max()):.4f}"
                )
            print(
                f"[sample] observation.state: shape={tuple(sample['observation.state'].shape)} "
                f"dtype={sample['observation.state'].dtype}"
            )
            if "action" in sample:
                print(
                    f"[sample] action(chunk): shape={tuple(sample['action'].shape)} "
                    f"dtype={sample['action'].dtype}"
                )

    meta = {
        "count": written,
        "camera_keys": camera_keys,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "chunk_size": chunk_size,
        "indices": indices,
    }
    (out_dir / "samples_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def try_visualization(repo_id: str, dataset_root: Path, viz_out: Path) -> None:
    """调用官方可视化（headless：保存 .rrd）。失败时明确打印 VIZ_UNAVAILABLE 与原因。"""
    viz_out.mkdir(parents=True, exist_ok=True)
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.scripts.lerobot_dataset_viz import visualize_dataset
    except Exception as error:  # noqa: BLE001
        print(f"VIZ_UNAVAILABLE: import failed: {error}")
        return
    try:
        # 官方签名（v0.6.2, lerobot_dataset_viz.py:160）要求传入已构造的 dataset 对象
        dataset = LeRobotDataset(repo_id, root=dataset_root)
        visualize_dataset(
            dataset=dataset,
            episode_index=0,
            mode="local",
            save=True,
            output_dir=viz_out,
            display_mode="rerun",
        )
        produced = sorted(path.name for path in viz_out.glob("*.rrd"))
        print(f"VIZ_SAVED: {produced}")
    except Exception as error:  # noqa: BLE001
        print(f"VIZ_UNAVAILABLE: {type(error).__name__}: {error}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--root", required=True, help="HF_LEROBOT_HOME（数据集父目录）")
    parser.add_argument("--samples-out", required=True)
    parser.add_argument("--fallback-repo-id", default="")
    parser.add_argument("--viz-out", default="")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=100)
    args = parser.parse_args()

    data_root = Path(args.root).expanduser()
    used_repo_id = args.repo_id
    dataset_root = data_root / used_repo_id

    print("=== scan_dataset_ref.py ===")
    print(f"HF_LEROBOT_HOME : {data_root}")
    print(f"repo_id         : {used_repo_id}")
    print(f"dataset root    : {dataset_root}")

    try:
        dataset = build_dataset(used_repo_id, dataset_root, args.chunk_size, 1)
    except Exception as error:  # noqa: BLE001
        print(f"[primary failed] {type(error).__name__}: {error}")
        if not args.fallback_repo_id:
            raise
        used_repo_id = args.fallback_repo_id
        dataset_root = data_root / used_repo_id
        print(f"[fallback] retry with {used_repo_id} at {dataset_root}")
        dataset = build_dataset(used_repo_id, dataset_root, args.chunk_size, 1)

    info = describe(dataset)
    print("\n=== dataset metadata ===")
    for key, value in info.items():
        if key != "features":
            print(f"{key:16s}: {value}")
    print("\n=== features ===")
    for key, spec in info["features"].items():
        print(f"{key:32s} dtype={spec.get('dtype')} shape={spec.get('shape')}")

    samples_meta = export_samples(dataset, Path(args.samples_out), args.num_samples)
    print("\n=== exported samples ===")
    print(json.dumps(samples_meta, indent=2))

    if args.viz_out:
        print("\n=== visualization ===")
        try_visualization(used_repo_id, dataset_root, Path(args.viz_out))

    print("\nSCAN_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())

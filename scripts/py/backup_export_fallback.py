#!/usr/bin/env python
"""受控 Python 调用点 (d)：仅当 TorchScript 导出失败时启用的兜底导出。

顺序：torch.export → ONNX（供 ONNX Runtime CUDA 路径使用）。
本脚本只做导出与自检，不修改训练产物；结果与失败原因由调用方（C++ actlab）写入报告。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mode", default="onnx", choices=["export", "onnx"])
    args = parser.parse_args()

    print("=== backup_export_fallback.py ===")
    print(f"checkpoint: {args.checkpoint}")
    print(f"mode      : {args.mode}")

    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.factory import make_policy
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(args.checkpoint)
    train_config = checkpoint / "train_config.json"
    import json

    repo_id = json.loads(train_config.read_text(encoding="utf-8"))["dataset"]["repo_id"] if train_config.exists() else None
    if repo_id is None:
        print("FALLBACK_FAIL: cannot infer dataset repo_id from train_config.json")
        return 2
    import os

    root = Path(os.environ.get("HF_LEROBOT_HOME", str(Path.home() / ".cache/huggingface/lerobot"))) / repo_id
    meta = LeRobotDatasetMetadata(repo_id, root=root)
    config = ACTConfig.from_pretrained(checkpoint)
    config.device = args.device
    config.pretrained_path = checkpoint
    policy = make_policy(config, ds_meta=meta).eval().to(args.device)

    class Wrapper(torch.nn.Module):
        def __init__(self, wrapped) -> None:
            super().__init__()
            self.wrapped = wrapped

        def forward(self, image: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
            batch = {"observation.images.up": image, "observation.state": state}
            return self.wrapped.predict_action_chunk(batch)

    wrapper = Wrapper(policy)
    example = (
        torch.zeros(1, 3, 480, 640, device=args.device),
        torch.zeros(1, meta.features["observation.state"]["shape"][0], device=args.device),
    )

    if args.mode == "export":
        try:
            exported = torch.export.export(wrapper, example)
            target = out_dir / "act_policy.exported.pt2"
            torch.export.save(exported, str(target))
            print(f"FALLBACK_EXPORT_PASS: {target}")
            return 0
        except Exception as error:  # noqa: BLE001
            print(f"FALLBACK_EXPORT_FAIL: {type(error).__name__}: {error}")
            return 1

    try:
        target = out_dir / "act_policy.onnx"
        torch.onnx.export(wrapper, example, str(target), opset_version=17)
        print(f"FALLBACK_ONNX_PASS: {target}")
        return 0
    except Exception as error:  # noqa: BLE001
        print(f"FALLBACK_ONNX_FAIL: {type(error).__name__}: {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""Python 参考实现（库形式，不单独作为进程调用）：
用与训练同源的 policy + preprocessor/postprocessor 在真实样本上做一次前向，
输出参考动作序列，供 C++ actlab eval 做交叉比对（Prompt §4.3 / §6 一致性门禁）。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


def run_reference(policy, preprocessor, postprocessor, samples_dir: Path, device: str = "cuda") -> dict:
    meta = json.loads((samples_dir / "samples_meta.json").read_text(encoding="utf-8"))
    camera_keys = meta["camera_keys"]
    outputs: dict[str, list[float]] = {}
    summaries = []

    policy.eval()
    with torch.no_grad():
        for position in range(meta["count"]):
            prefix = f"sample_{position:03d}"
            batch = {}
            for cam_index, key in enumerate(camera_keys):
                image = torch.from_numpy(np.load(samples_dir / f"{prefix}_cam{cam_index}.npy")).float()
                batch[key] = image
            batch["observation.state"] = torch.from_numpy(
                np.load(samples_dir / f"{prefix}_state.npy")
            ).float()
            batch = {key: value.unsqueeze(0) for key, value in batch.items()}
            processed = preprocessor(batch)
            actions = policy.predict_action_chunk(processed)
            actions = postprocessor(actions)
            flat = actions.reshape(-1).to(torch.float64)
            outputs[prefix] = [float(value) for value in flat.tolist()]
            summaries.append(
                {
                    "sample": prefix,
                    "shape": list(actions.shape),
                    "min": float(flat.min()),
                    "max": float(flat.max()),
                    "mean": float(flat.mean()),
                }
            )
    return {"outputs": outputs, "summary": summaries}

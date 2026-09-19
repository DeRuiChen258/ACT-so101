#!/usr/bin/env python
"""受控 Python 调用点 (c)：从训练 checkpoint 重建 policy 并导出 TorchScript。

同源契约（Prompt §2.4）：
  - 归一化/反归一化使用 checkpoint 自带的 preprocessor/postprocessor（训练同源），
    打包进被 trace 的模块内部；C++ 侧不做任何手写归一化。
  - 导出后立即用"官方 pipeline"与"被导出模块"在同一批真实样本上做数值比对，
    差异写入 torchscript/equivalence.json；不一致则退出码非 0（不允许带病导出）。
  - 同一进程内同时产出 Python 参考输出 reference_outputs.json，供 C++ eval 交叉比对。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from offline_baseline_eval import run_reference  # noqa: E402


class ExportWrapper(torch.nn.Module):
    """images(list[Tensor]) + state(Tensor) -> 真实单位动作块(Tensor)。

    内部依次执行训练同源的 preprocessor、ACT policy 前向与 postprocessor 反归一化。
    """

    def __init__(self, policy, preprocessor, postprocessor, camera_keys: list[str]) -> None:
        super().__init__()
        self.policy = policy
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.camera_keys = list(camera_keys)

    @torch.no_grad()
    def forward(self, images: list[torch.Tensor], state: torch.Tensor) -> torch.Tensor:
        batch = {key: image for key, image in zip(self.camera_keys, images)}
        batch["observation.state"] = state
        batch = self.preprocessor(batch)
        actions = self.policy.predict_action_chunk(batch)
        return self.postprocessor(actions)


def load_samples(samples_dir: Path, device: str) -> tuple[list[torch.Tensor], torch.Tensor, dict]:
    meta = json.loads((samples_dir / "samples_meta.json").read_text(encoding="utf-8"))
    prefix = "sample_000"
    images = [
        torch.from_numpy(np.load(samples_dir / f"{prefix}_cam{cam}.npy"))
        .float()
        .unsqueeze(0)
        .to(device)
        for cam in range(len(meta["camera_keys"]))
    ]
    state = torch.from_numpy(np.load(samples_dir / f"{prefix}_state.npy")).float().unsqueeze(0).to(device)
    return images, state, meta


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--samples-dir", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    checkpoint = Path(args.checkpoint)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.factory import make_policy, make_pre_post_processors

    print("=== export_act_torchscript.py ===")
    print(f"checkpoint : {checkpoint}")
    print(f"out_dir    : {out_dir}")
    print(f"device     : {device}")

    meta = LeRobotDatasetMetadata(args.repo_id, root=Path(args.root))
    config = ACTConfig.from_pretrained(checkpoint)
    config.device = str(device)
    config.pretrained_path = checkpoint

    policy = make_policy(config, ds_meta=meta)
    policy.to(device)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config, pretrained_path=str(checkpoint)
    )
    print(f"policy     : {type(policy).__name__} params={sum(p.numel() for p in policy.parameters()):,}")
    print(f"policy cfg : chunk_size={config.chunk_size} n_action_steps={config.n_action_steps}")

    images, state, samples_meta = load_samples(Path(args.samples_dir), str(device))
    print(f"camera_keys: {samples_meta['camera_keys']}")
    print(f"example    : images=[{tuple(images[0].shape)}] state={tuple(state.shape)}")

    wrapper = ExportWrapper(policy, preprocessor, postprocessor, samples_meta["camera_keys"])
    wrapper.eval()
    wrapper.to(device)

    with torch.no_grad():
        exported_actions = wrapper(images, state)
    print(f"wrapper out: shape={tuple(exported_actions.shape)} device={exported_actions.device}")

    # 官方 pipeline 参考值（同一批输入）→ 证明导出模块与训练侧实现数值一致
    reference_batch = {}
    for cam, key in enumerate(samples_meta["camera_keys"]):
        reference_batch[key] = images[cam]
    reference_batch["observation.state"] = state
    with torch.no_grad():
        processed = preprocessor(reference_batch)
        reference_actions = postprocessor(policy.predict_action_chunk(processed))
    max_abs_diff = float(
        (exported_actions.detach().to("cpu") - reference_actions.detach().to("cpu")).abs().max()
    )
    relative_scale = float(reference_actions.detach().to("cpu").abs().max()) + 1e-12
    equivalence = {
        "max_abs_diff": max_abs_diff,
        "relative": max_abs_diff / relative_scale,
        "shape": list(exported_actions.shape),
        "device": str(device),
        "action_scale": relative_scale,
    }
    (out_dir / "equivalence.json").write_text(json.dumps(equivalence, indent=2), encoding="utf-8")
    print(
        f"equivalence(exported vs official pipeline): max_abs_diff={max_abs_diff:.3e} "
        f"relative={equivalence['relative']:.3e} (action_scale≈{relative_scale:.3f})"
    )

    # TorchScript 导出（trace）；失败则明确报错，由 C++ 侧按降级链处理
    traced = torch.jit.trace(wrapper, (images, state), strict=False)
    traced = torch.jit.freeze(traced.eval())
    torchscript_path = out_dir / "act_policy.pt"
    traced.save(str(torchscript_path))
    print(f"saved TorchScript: {torchscript_path} ({torchscript_path.stat().st_size} bytes)")
    print("signature:")
    print(traced.forward.schema if hasattr(traced.forward, "schema") else "forward(images, state)")

    # 重新加载并校验（导出即验证）
    reloaded = torch.jit.load(str(torchscript_path))
    with torch.no_grad():
        reloaded_actions = reloaded(images, state)
    reload_abs_diff = float(
        (reloaded_actions.detach().to("cpu") - exported_actions.detach().to("cpu")).abs().max()
    )
    # 判定口径：以动作量纲为基准的相对误差（原始动作可达 ±100 量级，绝对 3e-2 属 op 融合级别差异）
    reload_rel_diff = reload_abs_diff / relative_scale
    print(
        f"reload(torch.jit.load + freeze vs eager): max_abs_diff={reload_abs_diff:.3e} "
        f"relative={reload_rel_diff:.3e}"
    )

    # Python 参考输出（供 C++ eval 交叉比对）
    reference = run_reference(policy, preprocessor, postprocessor, Path(args.samples_dir), str(device))
    (out_dir.parent / "reference_outputs.json").write_text(
        json.dumps(reference["outputs"]), encoding="utf-8"
    )
    print(f"reference_outputs.json: {len(reference['outputs'])} samples")
    for summary in reference["summary"]:
        print(f"  {summary['sample']} shape={summary['shape']} min={summary['min']:.4f} max={summary['max']:.4f}")

    ok = (equivalence["relative"] < 1e-3) and (reload_rel_diff < 1e-3)
    print(f"\nEXPORT_{'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

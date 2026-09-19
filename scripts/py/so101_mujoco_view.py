#!/usr/bin/env python
"""SO-101 机械臂 MuJoCo 可视化（含真实物体，三种模式）。

数据来源：LeRobotDataset 的 parquet（observation.state，6 关节，单位：度）。
模型来源：MuJoCo Menagerie `trs_so_arm100`（SO-ARM/SO-101 同族，6 关节：
Rotation / Pitch / Elbow / Wrist_Pitch / Wrist_Roll / Jaw）。

与真实数据的三处绑定（全部由轨迹推断，不写死帧号）：
  1. 零点对齐：以 episode 首帧对齐模型 home 姿态，补偿伺服标定零点差异；
  2. 物体位置：夹爪中心轨迹的最低点即抓取点 → 水平面放在该高度，物体水平坐标取该点 xy，
     整体位于水平面之上；
  3. 抓取/松开：由夹爪信号判定（先张开后闭合 = 抓取；之后重新张开 = 松开）。
     抓取期间物体随夹爪移动（不是空爪），松开后落回水平面。

模式：
  render  —— 离屏渲染轨迹关键帧为 PNG（可编码 mp4），用于证据留存；
  window  —— 在桌面打开真实 MuJoCo 仿真窗口回放（--x11 强制 XWayland 以便截图留证）；
  compare —— 数据集真实相机帧 与 同一时刻 3D 渲染 并排，用于数据自洽性核对。

本脚本只读取已落盘数据与模型，不修改训练产物；调用点登记见 logs/hybrid_boundary.md。
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")  # 离屏渲染后端；窗口模式会覆盖为 glfw

import mujoco  # noqa: E402
from PIL import Image  # noqa: E402


# --------------------------------------------------------------------------- #
# 配置与数据
# --------------------------------------------------------------------------- #
def load_paths_config(paths_config: str) -> dict:
    if not paths_config or not Path(paths_config).exists():
        return {}
    import yaml

    with Path(paths_config).open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data.get("paths", {}) or {}


def load_states(root: Path, repo_id: str, episode: int, max_frames: int | None = None) -> np.ndarray:
    """读取某个 episode 的 observation.state 序列（单位：度）。"""
    import pandas as pd

    data_dir = root / repo_id / "data"
    files = sorted(data_dir.glob("chunk-*/file-*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet under {data_dir}")
    frames = []
    for path in files:
        frame = pd.read_parquet(path, columns=["episode_index", "frame_index", "observation.state"])
        frame = frame[frame["episode_index"] == episode].sort_values("frame_index")
        if not frame.empty:
            frames.append(np.stack(frame["observation.state"].values))
    if not frames:
        raise ValueError(f"episode {episode} not found in dataset {repo_id}")
    states = np.concatenate(frames, axis=0)
    return states[:max_frames] if max_frames is not None else states


def build_model(model_path: Path) -> tuple[mujoco.MjModel, mujoco.MjData]:
    model = mujoco.MjModel.from_xml_path(str(model_path))
    return model, mujoco.MjData(model)


def free_camera(model: mujoco.MjModel, azimuth: float = 120.0, elevation: float = -18.0,
                distance_scale: float = 1.0) -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = model.stat.center
    cam.lookat[2] = max(0.0, float(model.stat.center[2]))
    cam.distance = float(model.stat.extent) * distance_scale
    cam.azimuth = azimuth
    cam.elevation = elevation
    return cam


def arm_camera(model: mujoco.MjModel, setup: "SceneSetup", azimuth: float, elevation: float,
               distance_scale: float) -> mujoco.MjvCamera:
    """取景：把"机械臂 + 物体"整体框进画面（水平面会撑大 model.stat，故不用它）。"""
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth = azimuth
    cam.elevation = elevation
    if setup.has_object and setup.rest_xyz is not None:
        # 取景覆盖"机械臂（基座→顶部约 0.35 m）+ 物体"，因此距离随物体水平距离增长
        target = (setup.rest_xyz[0] / 2.0, setup.rest_xyz[1] / 2.0, setup.surface_z + 0.20)
        reach = float(np.linalg.norm(setup.rest_xyz[:2])) + 0.55
    else:
        target = (float(model.stat.center[0]), float(model.stat.center[1]),
                  max(0.0, float(model.stat.center[2])))
        reach = float(model.stat.extent)
    cam.lookat[:] = target
    cam.distance = max(0.75, reach) * distance_scale
    return cam


def home_alignment_offsets(model: mujoco.MjModel, first_state_deg: np.ndarray) -> np.ndarray:
    """零点对齐：SO-101 state 为标定后的伺服角，模型零点不同。

    offset = 数据首帧 - 模型 keyframe "home"（度），保证相对运动精确；
    绝对姿态取决于"demo 是否从 home 附近开始"，该假设在日志中显式标注。
    """
    if model.nkey > 0:
        key_id = 0
        for i in range(model.nkey):
            if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_KEY, i) == "home":
                key_id = i
                break
        home_deg = np.rad2deg(np.array(model.key_qpos[key_id][: first_state_deg.size]))
    else:
        home_deg = np.zeros_like(first_state_deg)
    return first_state_deg - home_deg


def apply_state(model: mujoco.MjModel, data: mujoco.MjData, state_deg: np.ndarray) -> None:
    """数据集度 → 模型弧度（关节顺序一致：pan/lift, elbow, wrist_flex/roll, gripper）。"""
    if model.nq < state_deg.size:
        raise ValueError(f"model has {model.nq} qpos but state has {state_deg.size} values")
    data.qpos[: state_deg.size] = np.deg2rad(state_deg)


# --------------------------------------------------------------------------- #
# 由真实轨迹推断物体位置与抓取阶段
# --------------------------------------------------------------------------- #
def gripper_centers(model: mujoco.MjModel, data: mujoco.MjData,
                    states_model_deg: np.ndarray) -> np.ndarray:
    """夹持中心轨迹 = 8 块指垫（fixed_jaw_pad_* / moving_jaw_pad_*）几何中心的均值。

    用指垫而不是 jaws body 原点：body 原点位于爪体内部，直接用它会让被夹物体与爪体穿模。
    """
    centers, _, _ = pad_frames(model, data, states_model_deg)
    return centers


def pad_geometry_ids(model: mujoco.MjModel) -> tuple[list[int], list[int]]:
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or "" for i in range(model.ngeom)]
    fixed = [i for i, name in enumerate(names) if name.startswith("fixed_jaw_pad")]
    moving = [i for i, name in enumerate(names) if name.startswith("moving_jaw_pad")]
    if len(fixed) == 0 or len(fixed) != len(moving):
        raise RuntimeError("model has no matching fixed_jaw_pad_*/moving_jaw_pad_* geoms")
    return fixed, moving


def pad_frames(model: mujoco.MjModel, data: mujoco.MjData,
               states_model_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """逐帧返回（指垫夹持中心, 夹爪姿态四元数, 最近指垫间隙）。"""
    fixed_ids, moving_ids = pad_geometry_ids(model)
    jaw_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "Fixed_Jaw")
    centers, quats, gaps = [], [], []
    for row in states_model_deg:
        apply_state(model, data, row)
        mujoco.mj_forward(model, data)
        fixed_pos = np.array([data.geom_xpos[i] for i in fixed_ids])
        moving_pos = np.array([data.geom_xpos[i] for i in moving_ids])
        centers.append(((fixed_pos.mean(0) + moving_pos.mean(0)) / 2.0).copy())
        # 指垫对（fixed i ↔ moving 末-i）：取最近的一对作为夹持间隙
        pair_gaps = [float(np.linalg.norm(fixed_pos[i] - moving_pos[len(moving_pos) - 1 - i]))
                     for i in range(len(fixed_pos))]
        gaps.append(min(pair_gaps))
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, np.array(data.xmat[jaw_body]).reshape(3, 3).ravel())
        quats.append(quat.copy())
    return np.asarray(centers), np.asarray(quats), np.asarray(gaps)


def detect_grasp_point(centers: np.ndarray) -> dict:
    """物体位置 = 夹爪中心轨迹最低点（抓取时刻）。"""
    lowest = int(np.argmin(centers[:, 2]))
    return {
        "grasp_frame": lowest,
        "grasp_xyz": centers[lowest].tolist(),
        "ee_min_z": float(centers[:, 2].min()),
        "ee_max_z": float(centers[:, 2].max()),
        "ee_start_z": float(centers[0, 2]),
        "ee_end_z": float(centers[-1, 2]),
    }


def detect_object_phases(states_model_deg: np.ndarray) -> dict:
    """按夹爪信号判定抓取/松开时刻（无人工帧号）。"""
    jaw = states_model_deg[:, 5]
    low, high = float(jaw.min()), float(jaw.max())
    span = max(1e-6, high - low)
    # 夹住物体时夹爪停在"半程"附近（物体宽度限制闭合），完全闭合=空爪；
    # 因此抓取阈值取半程，才能与 EE 最低点（真正接触物体的时刻）一致。
    close_th = low + 0.5 * span
    open_th = low + 0.6 * span
    grasp = -1
    for i in range(1, len(jaw)):
        if jaw[:i].max() > open_th and jaw[i] < close_th:
            grasp = i
            break
    release = -1
    if grasp >= 0:
        for j in range(grasp + 1, len(jaw)):
            if jaw[j] > open_th:
                release = j
                break
    return {
        "jaw_min": low,
        "jaw_max": high,
        "close_threshold": close_th,
        "open_threshold": open_th,
        "grasp_frame": grasp,
        "release_frame": release,
        "carried_frames": 0 if grasp < 0 else (len(jaw) - grasp if release < 0 else release - grasp),
    }


def object_position_for_frame(index: int, phases: dict, centers: np.ndarray, rest_xyz: np.ndarray,
                              half_size: float, surface_z: float) -> np.ndarray:
    """抓取前静置 → 抓取中随爪 → 松开后落回平面。"""
    grasp, release = phases["grasp_frame"], phases["release_frame"]
    if grasp < 0 or index < grasp:
        return rest_xyz.copy()
    if release < 0 or index < release:
        return centers[index] + (rest_xyz - centers[grasp])
    # 松开后：从松开位置平滑落回平面（运动学插值，非物理仿真），避免瞬间"消失/瞬移"
    placed = centers[release].copy()
    placed[2] = surface_z + half_size
    fall_frames = 12
    t = min(1.0, (index - release) / float(fall_frames))
    start_z = centers[release][2]
    placed[2] = start_z + (surface_z + half_size - start_z) * t
    return placed


RELEASE_FALL_FRAMES = 12


def object_pose_for_frame(index: int, phases: dict, centers: np.ndarray, quats: np.ndarray,
                          rest_xyz: np.ndarray, half_size: float,
                          surface_z: float) -> tuple[np.ndarray, np.ndarray]:
    """返回物体该帧的（位置, 四元数）：抓取前静置、抓取中与爪同步、松开后平落到平面。"""
    grasp, release = phases["grasp_frame"], phases["release_frame"]
    identity = np.array([1.0, 0.0, 0.0, 0.0])
    if grasp < 0 or index < grasp:
        return rest_xyz.copy(), identity
    if release < 0 or index < release:
        return centers[index].copy(), quats[index].copy()
    pos = centers[release].copy()
    target_z = surface_z + half_size
    t = min(1.0, (index - release) / float(RELEASE_FALL_FRAMES))
    pos[2] = centers[release][2] + (target_z - centers[release][2]) * t
    return pos, quats[release].copy()


# --------------------------------------------------------------------------- #
# 场景生成（机械臂 + 水平面 + 目标物体）
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class SceneSetup:
  scene_path: Path
  has_object: bool
  object_qpos_adr: int
  info: dict
  phases: dict
  centers: np.ndarray | None
  quats: np.ndarray | None
  rest_xyz: np.ndarray | None
  half_size: float
  surface_z: float

  def object_pose(self, index: int) -> tuple[np.ndarray, np.ndarray] | None:
    if not self.has_object or self.centers is None or self.rest_xyz is None or self.quats is None:
      return None
    return object_pose_for_frame(index, self.phases, self.centers, self.quats, self.rest_xyz,
                                 self.half_size, self.surface_z)


def build_scene_with_object(arm_xml: Path, out_xml: Path, object_xy: tuple[float, float],
                            surface_z: float, half_size: float, lift: float,
                            collide: bool = False) -> Path:
    """生成"机械臂 + 水平面 + 目标物体"场景；物体整体位于水平面之上。

    被 include 的模型用相对 `meshdir="assets/"` 引用网格，而 MuJoCo 以顶层文件目录解析 meshdir，
    因此把机械臂 XML 与 assets/ 复制到生成目录内（自包含、不动第三方目录），再在同目录生成场景。
    """
    import shutil

    model_dir = out_xml.parent / "so101_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(arm_xml, model_dir / arm_xml.name)
    src_assets = arm_xml.parent / "assets"
    if src_assets.is_dir():
        shutil.copytree(src_assets, model_dir / "assets", dirs_exist_ok=True)
    object_z = surface_z + half_size + lift
    contype = 1 if collide else 0
    # 场景 XML 与机械臂 XML、assets/ 同目录，保证 meshdir="assets/" 解析正确
    scene_xml = model_dir / "scene_with_object.xml"
    scene_xml.write_text(
        f"""<mujoco model="so101 lab scene">
  <include file="{arm_xml.name}"/>

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <!-- 离屏渲染缓冲：放宽上限，允许 800x600 等更高分辨率出图 -->
    <global azimuth="120" elevation="-18" offwidth="1920" offheight="1080"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="tablestexture" builtin="checker" mark="edge" rgb1="0.35 0.36 0.38"
      rgb2="0.28 0.29 0.31" markrgb="0.75 0.75 0.75" width="300" height="300"/>
    <material name="tablemat" texture="tablestexture" texuniform="true" texrepeat="4 4" reflectance="0.1"/>
  </asset>

  <worldbody>
    <light pos="0.3 -0.3 1.2" dir="-0.2 0.2 -1" directional="true"/>
    <!-- 水平面：高度取自真实抓取高度（轨迹 FK 推断） -->
    <geom name="surface" type="plane" pos="0 0 {surface_z:.6f}" size="1.2 1.2 0.05" material="tablemat"/>
    <!-- 目标物体：水平坐标 = 抓取点 xy，整体位于水平面之上 -->
    <body name="target_object" pos="{object_xy[0]:.6f} {object_xy[1]:.6f} {object_z:.6f}">
      <freejoint name="object_free"/>
      <geom name="object" type="box" size="{half_size:.4f} {half_size:.4f} {half_size:.4f}"
        rgba="0.90 0.25 0.20 1" contype="{contype}" conaffinity="{contype}"/>
    </body>
  </worldbody>
</mujoco>
""",
        encoding="utf-8",
    )
    return scene_xml


def setup_scene(args, out_dir: Path) -> tuple[mujoco.MjModel, mujoco.MjData, "SceneSetup", np.ndarray]:
    """装载模型（按需生成带物体场景），返回对齐后的关节序列。"""
    arm_xml = Path(args.model)
    arm_model, arm_data = build_model(arm_xml)
    states = load_states(Path(args.root), args.repo_id, args.episode, args.max_frames)
    if args.align_home:
        offsets = home_alignment_offsets(arm_model, states[0])
        print(f"home align offsets (deg): {np.round(offsets, 2).tolist()}")
        states = states - offsets
    else:
        offsets = np.zeros(6)

    if args.object == "none":
        return arm_model, arm_data, SceneSetup(arm_xml, False, -1, {}, {}, None, None, None, 0.0, 0.0), states

    centers, quats, gaps = pad_frames(arm_model, arm_data, states)
    info = detect_grasp_point(centers)
    phases = detect_object_phases(states)
    grasp = max(0, info["grasp_frame"])
    carried_gap = float(np.min(gaps[grasp:])) if grasp < len(gaps) else float(gaps[-1])
    half_size = (0.5 * carried_gap * 0.95 if args.object_auto_size else args.object_half_size) * \
        float(getattr(args, "object_size_scale", 1.0))
    # 水平面高度：让方块"被夹在指垫中点"时正好底面贴合平面（物理解释自洽）
    surface_z = (float(centers[grasp][2]) - half_size) if args.surface == "auto" else float(args.surface)
    rest_xyz = np.array([info["grasp_xyz"][0], info["grasp_xyz"][1],
                         surface_z + half_size + args.object_lift])
    scene_path = build_scene_with_object(
        arm_xml, out_dir / "generated" / "scene_with_object.xml",
        (float(rest_xyz[0]), float(rest_xyz[1])), surface_z, float(half_size), args.object_lift,
        collide=bool(getattr(args, "object_collide", False)))
    model, data = build_model(scene_path)
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "object_free")
    if joint_id < 0:
        raise RuntimeError("generated scene has no object_free joint")
    setup = SceneSetup(scene_path, True, int(model.jnt_qposadr[joint_id]), info, phases, centers,
                       quats, rest_xyz, float(half_size), surface_z)

    info.update({
        "surface_z": surface_z,
        "object_xyz": rest_xyz.tolist(),
        "object_half_size": float(half_size),
        "object_size_mode": "auto(实测指垫间隙)" if args.object_auto_size else "manual",
        "pad_gap_at_grasp": float(gaps[grasp]),
        "pad_gap_min_carried": carried_gap,
        "grasp_pad_center": centers[grasp].tolist(),
        "object_lift": args.object_lift,
        "scene_xml": str(scene_path),
        "phases": phases,
        "episode_frames": int(len(states)),
        "align_home": bool(args.align_home),
        "align_offsets_deg": np.asarray(offsets, dtype=float).tolist(),
    })
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "grasp_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print("=== object / grasp info (derived from real trajectory FK) ===")
    print(f"grasp frame  : {info['grasp_frame']} / {len(states) - 1}")
    print(f"grasp xyz    : {np.round(info['grasp_xyz'], 4).tolist()} m")
    print(f"surface(plane) z : {surface_z:.4f} m")
    print(f"object center: {np.round(rest_xyz, 4).tolist()} m (half size {half_size:.4f} m, "
          f"mode={'auto' if args.object_auto_size else 'manual'})")
    print(f"pad gap      : at grasp {float(gaps[grasp]):.4f} m, min carried {carried_gap:.4f} m")
    print(f"grasp/release: {phases['grasp_frame']} / {phases['release_frame']} "
          f"(carried {phases['carried_frames']} frames)")
    print(f"jaw min/max  : {phases['jaw_min']:.2f} / {phases['jaw_max']:.2f} deg")
    print(f"scene xml    : {scene_path}")
    return model, data, setup, states


def set_object_pose(model: mujoco.MjModel, data: mujoco.MjData, setup: "SceneSetup", index: int) -> None:
    pose = setup.object_pose(index)
    if pose is None:
        return
    position, quat = pose
    adr = setup.object_qpos_adr
    data.qpos[adr:adr + 3] = position
    data.qpos[adr + 3:adr + 7] = quat


def advance(model: mujoco.MjModel, data: mujoco.MjData, setup: "SceneSetup", state: np.ndarray,
            index: int) -> None:
    apply_state(model, data, state)
    set_object_pose(model, data, setup, index)
    mujoco.mj_forward(model, data)


# --------------------------------------------------------------------------- #
# 轨迹叠加对比（专家 vs ACT 预测）
# --------------------------------------------------------------------------- #
def draw_polyline(scene: mujoco.MjvScene, points: np.ndarray, color: tuple[float, float, float, float],
                  radius: float = 0.005) -> int:
    """在（user_）scene 上用小球标出轨迹点；返回绘制的点数。"""
    drawn = 0
    for point in points:
        if scene.ngeom >= scene.maxgeom:
            break
        mujoco.mjv_initGeom(
            scene.geoms[scene.ngeom],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([radius, 0.0, 0.0]),
            np.asarray(point, dtype=np.float64),
            np.eye(3).ravel(),
            np.asarray(color, dtype=np.float32),
        )
        scene.ngeom += 1
        drawn += 1
    return drawn


def load_prediction(samples_dir: Path, repo_id: str, sample: str = "sample_000") -> np.ndarray:
    """读取 Python 参考实现导出的 ACT 预测动作块（真实单位，度）。"""
    reference = samples_dir.parent / "act_baseline" / "reference_outputs.json"
    if not reference.exists():
        raise FileNotFoundError(f"missing {reference}（先运行 actlab verify 导出参考输出）")
    data = json.loads(reference.read_text(encoding="utf-8"))
    if sample not in data:
        raise KeyError(f"{sample} not in {reference}")
    chunk = np.asarray(data[sample], dtype=float)
    return chunk.reshape(-1, 6)


# --------------------------------------------------------------------------- #
# 三种模式
# --------------------------------------------------------------------------- #
def render_frames(args) -> int:
    out_dir = Path(args.out_dir)
    model, data, setup, states = setup_scene(args, out_dir)
    cam = arm_camera(model, setup, args.azimuth, args.elevation, args.distance_scale)
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    out_dir.mkdir(parents=True, exist_ok=True)

    indices = list(range(0, len(states), args.stride))
    print(f"episode    : {args.episode}, frames={len(states)}, rendering {len(indices)} (stride={args.stride})")
    print(f"renderer   : {args.width}x{args.height}, MUJOCO_GL={os.environ.get('MUJOCO_GL')}")

    # 专家 vs 预测 轨迹叠加（可选）
    expert_traj = predicted_traj = None
    if args.overlay == "expert_predicted":
        anchor = args.overlay_frame if args.overlay_frame >= 0 else 0
        prediction = load_prediction(Path(args.samples_dir), args.repo_id, args.overlay_sample)
        offsets = np.asarray(setup.info.get("align_offsets_deg", np.zeros(6)), dtype=float)
        predicted_traj, _, _ = pad_frames(model, data, prediction - offsets)
        end = min(len(setup.centers), anchor + len(prediction))
        expert_traj = setup.centers[anchor:end]
        n = min(len(expert_traj), len(predicted_traj))
        deviation = np.linalg.norm(expert_traj[:n] - predicted_traj[:n], axis=1)
        compare = {
            "anchor_frame": int(anchor),
            "sample": args.overlay_sample,
            "chunk": int(n),
            "ee_mae_m": float(deviation.mean()),
            "ee_max_dev_m": float(deviation.max()),
            "final_expert_xyz": expert_traj[n - 1].tolist(),
            "final_predicted_xyz": predicted_traj[n - 1].tolist(),
            "final_diff_m": float(np.linalg.norm(expert_traj[n - 1] - predicted_traj[n - 1])),
        }
        (out_dir / "trajectory_compare.json").write_text(json.dumps(compare, indent=2), encoding="utf-8")
        print("=== expert vs predicted EE trajectory ===")
        print(f"anchor frame : {anchor}, chunk={n}")
        print(f"EE MAE       : {compare['ee_mae_m']:.4f} m")
        print(f"EE max dev   : {compare['ee_max_dev_m']:.4f} m")
        print(f"end-point diff: {compare['final_diff_m']:.4f} m")

    paths = []
    for order, index in enumerate(indices):
        advance(model, data, setup, states[index], index)
        renderer.update_scene(data, camera=cam)
        if expert_traj is not None:
            # 先画预测（细），再画专家（粗）：两条轨迹都清晰可辨
            draw_polyline(renderer.scene, predicted_traj, (1.0, 0.2, 0.9, 1.0), radius=0.0035)
            draw_polyline(renderer.scene, expert_traj, (0.15, 0.95, 0.25, 1.0), radius=0.006)
        path = out_dir / f"render_{args.episode:03d}_{order:03d}_frame{index:04d}.png"
        Image.fromarray(renderer.render()).save(path)
        paths.append(path)
    renderer.close()
    print(f"frames     : {len(paths)} PNG -> {out_dir}")

    if paths:
        thumbs = [Image.open(p).resize((args.width // 2, args.height // 2)) for p in paths]
        cols = 4
        rows = (len(thumbs) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * args.width // 2, rows * args.height // 2), (20, 20, 20))
        for i, thumb in enumerate(thumbs):
            sheet.paste(thumb, ((i % cols) * args.width // 2, (i // cols) * args.height // 2))
        montage = out_dir / f"so101_render_montage_ep{args.episode:03d}.png"
        sheet.save(montage)
        print(f"montage    : {montage}")

        if args.video:
            video = Path(args.video)
            video.parent.mkdir(parents=True, exist_ok=True)
            pattern = str(out_dir / f"render_{args.episode:03d}_*.png")
            result = subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(max(1, 30 // args.stride)),
                 "-pattern_type", "glob", "-i", pattern, "-pix_fmt", "yuv420p", str(video)],
                check=False,
            )
            print(f"video      : {video} (ffmpeg exit={result.returncode})")
    return 0


def compare_frames(args) -> int:
    out_dir = Path(args.out_dir)
    model, data, setup, states = setup_scene(args, out_dir)
    cam = arm_camera(model, setup, args.azimuth, args.elevation, args.distance_scale)
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    out_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = Path(args.samples_dir)

    pairs = []
    for position in range(args.pairs):
        index = int(round(position * (len(states) - 1) / max(1, args.pairs - 1)))
        advance(model, data, setup, states[index], index)
        renderer.update_scene(data, camera=cam)
        render_im = Image.fromarray(renderer.render())

        cam_path = samples_dir / f"sample_{position:03d}_cam0.npy"
        if cam_path.exists():
            chw = np.load(cam_path)
            hwc = np.clip(np.transpose(chw, (1, 2, 0)), 0, 1)
            real_im = Image.fromarray((hwc * 255).astype(np.uint8))
        else:
            real_im = Image.new("RGB", (args.width, args.height), (40, 40, 40))
        pair = Image.new("RGB", (args.width * 2, args.height), (0, 0, 0))
        pair.paste(real_im.resize((args.width, args.height)), (0, 0))
        pair.paste(render_im, (args.width, 0))
        path = out_dir / f"compare_ep{args.episode:03d}_{position:03d}_frame{index:04d}.png"
        pair.save(path)
        pairs.append(path)
        print(f"pair {position}: dataset frame {index} -> {path.name}")
    renderer.close()

    if pairs:
        sheet = Image.new("RGB", (args.width * 2, args.height * len(pairs)), (0, 0, 0))
        for i, path in enumerate(pairs):
            sheet.paste(Image.open(path), (0, i * args.height))
        montage = out_dir / f"so101_compare_montage_ep{args.episode:03d}.png"
        sheet.save(montage)
        print(f"montage    : {montage}")
    return 0


def window_playback(args) -> int:
    """在桌面打开 MuJoCo 仿真窗口，按真实轨迹回放（含物体抓取，不空爪）。"""
    os.environ["MUJOCO_GL"] = "glfw"
    if args.x11:
        # 本机为 Wayland 会话：GLFW 默认走 Wayland（窗口可见但第三方无法截图留证）。
        # 强制 X11/XWayland 可同时满足"桌面可见 + 可抓图存证"。
        import glfw

        glfw.init_hint(glfw.PLATFORM, glfw.PLATFORM_X11)
    import mujoco.viewer

    out_dir = Path(args.out_dir)
    model, data, setup, states = setup_scene(args, out_dir)
    print(f"window mode: scene={setup.scene_path.name} frames={len(states)} fps={args.fps} loops={args.loops}")
    print(f"DISPLAY={os.environ.get('DISPLAY')} MUJOCO_GL={os.environ.get('MUJOCO_GL')}")

    loops = 0
    with mujoco.viewer.launch_passive(model, data) as viewer:
        cam = arm_camera(model, setup, args.azimuth, args.elevation, args.distance_scale)
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.lookat[:] = cam.lookat
        viewer.cam.distance = cam.distance
        viewer.cam.azimuth = cam.azimuth
        viewer.cam.elevation = cam.elevation
        print("WINDOW_OPEN")
        while viewer.is_running() and (args.loops <= 0 or loops < args.loops):
            for index in range(len(states)):
                if not viewer.is_running():
                    break
                advance(model, data, setup, states[index], index)
                viewer.sync()
                time.sleep(1.0 / args.fps)
            loops += 1
            print(f"loop {loops} finished")
    print("WINDOW_CLOSED")
    return 0


# --------------------------------------------------------------------------- #
# 物理抓取稳定性检查（真实接触 + 摩擦，不是运动学回放）
# --------------------------------------------------------------------------- #
def physics_check(args) -> int:
    """把专家抓取段的关节角作为位置伺服目标回放，让 MuJoCo 真实计算接触/摩擦，
    再判定物体是被"夹住提起"还是"滑脱"。

    这是对"由轨迹推断出的物体位置与尺寸"的物理一致性检验：若尺寸与夹爪开口不匹配，
    物体在提起阶段就会滑落 —— 结果如实记录，不做人为修正。
    """
    args.object_collide = True
    out_dir = Path(args.out_dir)
    model, data, setup, states = setup_scene(args, out_dir)
    if not setup.has_object:
        raise RuntimeError("physics mode requires --object auto")

    grasp = setup.phases["grasp_frame"]
    release = setup.phases["release_frame"]
    if grasp < 0:
        raise RuntimeError("no grasp detected in this episode")
    end = release if release > grasp else len(states)

    ctrl_low = model.actuator_ctrlrange[:, 0]
    ctrl_high = model.actuator_ctrlrange[:, 1]
    obj_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_object")

    def set_ctrl_from_state(state_deg: np.ndarray) -> None:
        data.ctrl[:] = np.clip(np.deg2rad(state_deg), ctrl_low, ctrl_high)

    # 1) 初始化到抓取帧（物体在平面上、夹爪闭合到抓取开口）
    apply_state(model, data, states[grasp])
    set_object_pose(model, data, setup, grasp)
    set_ctrl_from_state(states[grasp])
    mujoco.mj_forward(model, data)

    settle_steps = int(round(args.physics_settle_s / model.opt.timestep))
    for _ in range(settle_steps):
        mujoco.mj_step(model, data)
    settle = data.xpos[obj_body].copy()
    grip_settle = _gripper_point(model, data)
    print(f"physics: settle {args.physics_settle_s}s -> object z={settle[2]:.4f} m")

    # 2) 位置伺服跟随专家抓取段（含提起与移动）
    steps_per_frame = max(1, int(round((1.0 / args.fps) / model.opt.timestep)))
    trace = []
    for index in range(grasp, end):
        set_ctrl_from_state(states[index])
        for _ in range(steps_per_frame):
            mujoco.mj_step(model, data)
        obj_pos = data.xpos[obj_body].copy()
        grip_pos = _gripper_point(model, data)
        trace.append({
            "frame": int(index),
            "object_xyz": obj_pos.tolist(),
            "gripper_xyz": grip_pos.tolist(),
            "dist_to_gripper": float(np.linalg.norm(obj_pos - grip_pos)),
        })
    object_z = np.array([row["object_xyz"][2] for row in trace])
    dists = np.array([row["dist_to_gripper"] for row in trace])
    lift_gain = float(object_z[-1] - settle[2])
    max_dist = float(dists.max())
    final_dist = float(dists[-1])
    held = bool(final_dist < args.physics_hold_dist and lift_gain > args.physics_lift_min)

    result = {
        "mode": "physics grasp stability check",
        "episode": args.episode,
        "grasp_frame": grasp,
        "carry_frames": end - grasp,
        "object_half_size": setup.half_size,
        "object_size_mode": "auto(实测指垫间隙)" if args.object_auto_size else "manual",
        "settle_s": args.physics_settle_s,
        "object_z_settle": float(settle[2]),
        "object_z_final": float(object_z[-1]),
        "lift_gain_m": lift_gain,
        "dist_final_m": final_dist,
        "dist_max_m": max_dist,
        "verdict": "HELD" if held else "SLIPPED",
        "criteria": {"hold_dist_m": args.physics_hold_dist, "lift_min_m": args.physics_lift_min},
        "gripper_point_settle": grip_settle.tolist(),
    }
    (out_dir / f"physics_grasp_scale{args.object_size_scale:.2f}.json").write_text(
        json.dumps({"result": result, "trace": trace}, indent=2), encoding="utf-8")
    print("=== physics grasp stability ===")
    print(f"object half size : {setup.half_size:.4f} m (scale {args.object_size_scale:.2f})")
    print(f"lift gain        : {lift_gain:+.4f} m")
    print(f"dist to gripper  : final {final_dist:.4f} m, max {max_dist:.4f} m")
    print(f"VERDICT          : {result['verdict']}")

    # 3) 关键帧渲染（物理过程的真实画面）
    if args.physics_render:
        render_keys = np.linspace(0, len(trace) - 1, args.physics_render).astype(int)
        renderer = mujoco.Renderer(model, height=args.height, width=args.width)
        cam = arm_camera(model, setup, args.azimuth, args.elevation, args.distance_scale)
        tiles = []
        # 重新跑一遍并抓关键帧（物理可复现：同初始状态 + 同 ctrl 序列）
        apply_state(model, data, states[grasp])
        set_object_pose(model, data, setup, grasp)
        set_ctrl_from_state(states[grasp])
        mujoco.mj_forward(model, data)
        for _ in range(settle_steps):
            mujoco.mj_step(model, data)
        shot = 0
        for index in range(grasp, end):
            set_ctrl_from_state(states[index])
            for _ in range(steps_per_frame):
                mujoco.mj_step(model, data)
            if shot < len(render_keys) and (index - grasp) == render_keys[shot]:
                renderer.update_scene(data, camera=cam)
                image = Image.fromarray(renderer.render())
                path = out_dir / f"physics_{args.object_size_scale:.2f}_{shot:02d}_frame{index:04d}.png"
                image.save(path)
                tiles.append(image.resize((args.width // 2, args.height // 2)))
                shot += 1
        renderer.close()
        if tiles:
            sheet = Image.new("RGB", (args.width // 2 * len(tiles), args.height // 2), (0, 0, 0))
            for i, tile in enumerate(tiles):
                sheet.paste(tile, (i * args.width // 2, 0))
            montage = out_dir / f"physics_grasp_scale{args.object_size_scale:.2f}_montage.png"
            sheet.save(montage)
            print(f"physics montage  : {montage}")
    return 0 if held else 1


def _gripper_point(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """当前姿态下的指垫夹持中心（不做轨迹计算）。"""
    fixed_ids, moving_ids = pad_geometry_ids(model)
    fixed_pos = np.array([data.geom_xpos[i] for i in fixed_ids])
    moving_pos = np.array([data.geom_xpos[i] for i in moving_ids])
    return (fixed_pos.mean(0) + moving_pos.mean(0)) / 2.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["render", "window", "compare", "physics"], required=True)
    parser.add_argument("--model", default="", help="机械臂 XML（缺省取 --paths-config: so101_mujoco_model）")
    parser.add_argument("--root", default="", help="HF_LEROBOT_HOME（缺省取 --paths-config: data_dir）")
    parser.add_argument("--paths-config", default="", help="configs/paths.yaml")
    parser.add_argument("--repo-id", default="lerobot/svla_so101_pickplace")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--samples-dir", default="artifacts/dataset_samples")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--stride", type=int, default=25)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--loops", type=int, default=60, help="window 回放循环数，<=0 直到关窗")
    parser.add_argument("--pairs", type=int, default=4)
    parser.add_argument("--video", default="")
    parser.add_argument("--azimuth", type=float, default=120.0)
    parser.add_argument("--elevation", type=float, default=-18.0)
    parser.add_argument("--distance-scale", type=float, default=1.0)
    parser.add_argument("--x11", action="store_true", help="窗口模式强制 X11/XWayland（便于截图留证）")
    parser.add_argument("--object", choices=["auto", "none"], default="auto",
                        help="auto=按真实轨迹推断物体位置并渲染目标物体（默认）")
    parser.add_argument("--object-half-size", type=float, default=0.015, help="目标方块半边长（米）")
    parser.add_argument("--object-auto-size", action="store_true", default=True,
                        help="按实测指垫间隙自动确定方块尺寸（默认开启，避免与夹爪穿模）")
    parser.add_argument("--no-object-auto-size", dest="object_auto_size", action="store_false")
    parser.add_argument("--object-lift", type=float, default=0.002, help="物体相对水平面的抬升（米）")
    parser.add_argument("--surface", default="auto", help="水平面高度：auto=抓取高度，或给定米数")
    parser.add_argument("--align-home", action="store_true", default=True)
    parser.add_argument("--no-align-home", dest="align_home", action="store_false")
    parser.add_argument("--overlay", choices=["none", "expert_predicted"], default="none",
                        help="expert_predicted：把专家轨迹（绿）与 ACT 预测轨迹（品红）叠加在场景中")
    parser.add_argument("--overlay-sample", default="sample_000", help="使用哪个样本的预测动作块")
    parser.add_argument("--overlay-frame", type=int, default=-1,
                        help="叠加对比的起始帧，-1 表示用样本对应帧（sample_000 → 0）")
    parser.add_argument("--physics-settle-s", type=float, default=0.5, help="抓取姿态稳定时间（秒）")
    parser.add_argument("--physics-hold-dist", type=float, default=0.04,
                        help="判定夹住的物体-夹爪中心距离阈值（米）")
    parser.add_argument("--physics-lift-min", type=float, default=0.03,
                        help="判定提起的最小高度增量（米）")
    parser.add_argument("--physics-render", type=int, default=6, help="物理检查渲染的关键帧数，0=不渲染")
    # 默认 1.05×：物理检查显示 1.0×（=0.95×指垫间隙）偏松（离夹爪中心 3.1 cm）、1.1× 贴合（2.8 mm），
    # 取中间值既贴合又不与指垫穿模
    parser.add_argument("--object-size-scale", type=float, default=1.05,
                        help="方块尺寸缩放（物理抓取灵敏度实验：1.0/1.05/1.1/1.2）")
    parser.add_argument("--object-collide", action="store_true", default=False)
    args = parser.parse_args()

    paths = load_paths_config(args.paths_config)
    if not args.model:
        args.model = paths.get("so101_mujoco_model", "")
    if not args.root:
        args.root = paths.get("data_dir", "")
    if not args.model or not args.root:
        parser.error("--model/--root 未提供，且 --paths-config 缺少 so101_mujoco_model / data_dir")

    if args.mode == "render":
        return render_frames(args)
    if args.mode == "compare":
        return compare_frames(args)
    if args.mode == "physics":
        return physics_check(args)
    return window_playback(args)


if __name__ == "__main__":
    sys.exit(main())

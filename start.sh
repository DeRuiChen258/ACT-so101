#!/usr/bin/env bash
# ==============================================================================
# start.sh — SO-101 机械臂 MuJoCo 可视化启动入口（本项目唯一启动脚本）
#
# 用法：
#   ./start.sh          # 打开桌面仿真窗口，持续循环回放 episode 0，关闭窗口即退出
#   ./start.sh 5        # 同上，回放 5 圈后自动退出
#
# 可选环境变量：
#   ACTLAB_EPISODE   回放的数据集 episode（默认 0）
#   ACTLAB_DISPLAY   X11 显示号（默认沿用 $DISPLAY，回退 :0，走 XWayland）
#   CONDA_ROOT / ENV_NAME   环境层位置（默认 /home/violet/Workspace/miniconda + lerobot_act）
#
# 路径与模型位置不写死在本脚本：模型来自 configs/paths.yaml，由 view 脚本读取。
# 其余非窗口模式（render / compare / physics 等）直接调用 scripts/py/so101_mujoco_view.py。
# ==============================================================================
set -euo pipefail

ACTLAB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${CONDA_ROOT:=/home/violet/Workspace/miniconda}"
: "${ENV_NAME:=lerobot_act}"
: "${ACTLAB_EPISODE:=0}"
: "${ACTLAB_DISPLAY:=${DISPLAY:-:0}}"

LOOPS="${1:-0}"
PY="${CONDA_ROOT}/envs/${ENV_NAME}/bin/python"
OUT_DIR="${ACTLAB_ROOT}/results/so101_mujoco"

if [[ ! -x "${PY}" ]]; then
  echo "[start] 缺少 Python 环境：${PY}" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}" "${ACTLAB_ROOT}/logs"
echo "[start] 打开桌面窗口（episode=${ACTLAB_EPISODE}, loops=${LOOPS}, DISPLAY=${ACTLAB_DISPLAY}）；关闭窗口即退出"

# -u：Python 输出走管道时默认块缓冲，去掉缓冲以便日志实时可见
MUJOCO_GL=glfw DISPLAY="${ACTLAB_DISPLAY}" \
  "${PY}" -u "${ACTLAB_ROOT}/scripts/py/so101_mujoco_view.py" \
  --mode=window \
  --paths-config="${ACTLAB_ROOT}/configs/paths.yaml" \
  --episode="${ACTLAB_EPISODE}" \
  --out-dir="${OUT_DIR}" \
  --samples-dir="${ACTLAB_ROOT}/artifacts/dataset_samples" \
  --object=auto \
  --x11 \
  --loops="${LOOPS}" \
  --fps=30 2>&1 | tee "${ACTLAB_ROOT}/logs/start.log"

#!/usr/bin/env bash
# Launch a durable SafeVLA queue detached (survives SSH disconnects).
# Usage:  bash scripts/remote/launch.sh [workers] [queue]     queue: safevla_v4 (default) | safevla_v4_v2
# Status: bash scripts/remote/status.sh [queue]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
VENV="${SAFEVLA_VENV:-$PWD/.venv-safevla}"
CORES=$(nproc)
export SAFEVLA_WORKERS="${1:-$(( CORES > 4 ? CORES - 2 : 2 ))}"
QUEUE="${2:-safevla_v4}"
export MUJOCO_GL="${MUJOCO_GL:-egl}" PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export SAFEVLA_TRAIN_THREADS="${SAFEVLA_TRAIN_THREADS:-8}" PYTHONUNBUFFERED=1
mkdir -p "experiment_tracking/$QUEUE"
echo "queue=$QUEUE workers=$SAFEVLA_WORKERS GL=$MUJOCO_GL"
setsid nohup "$VENV/bin/python" -u scripts/background_pipeline.py --config "experiment_tracking/$QUEUE.json" \
  > "experiment_tracking/$QUEUE/launcher.log" 2>&1 < /dev/null &
echo $! > "experiment_tracking/$QUEUE/launcher_pid.txt"
echo "started $QUEUE pid $(cat "experiment_tracking/$QUEUE/launcher_pid.txt"); logs in experiment_tracking/$QUEUE/"

#!/usr/bin/env bash
# One-time setup of SafeVLA v4 on a headless Linux GPU server.
# Usage (from the repository root):  bash scripts/remote/setup.sh [python3.12]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PY="${1:-python3}"
VENV="${SAFEVLA_VENV:-$PWD/.venv-safevla}"

echo "== system libraries (EGL/GL; OSMesa is only a fallback)"
if command -v apt-get >/dev/null && { [ "$(id -u)" = 0 ] || sudo -n true 2>/dev/null; }; then
  SUDO=$([ "$(id -u)" = 0 ] && echo "" || echo sudo)
  $SUDO apt-get update -qq
  $SUDO apt-get install -y -qq libegl1 libgl1 libgles2 libglvnd0 libosmesa6 >/dev/null || true
else
  echo "   (no root/apt: make sure libEGL + the NVIDIA driver's EGL vendor library are installed)"
fi
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || echo "   nvidia-smi not found"

echo "== python venv at $VENV"
"$PY" -m venv "$VENV"
"$VENV/bin/python" -m pip install -q --upgrade pip
"$VENV/bin/python" -m pip install -q -r requirements.txt

echo "== headless render check (EGL)"
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl "$VENV/bin/python" scripts/remote/check_render.py

echo "== unit tests"
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl "$VENV/bin/python" -m pytest tests/test_safevla_v4.py -q -p no:cacheprovider
echo "setup complete; launch with: bash scripts/remote/launch.sh"

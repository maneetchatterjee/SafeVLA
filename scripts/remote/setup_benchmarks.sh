#!/usr/bin/env bash
# Phase A: isolated conda envs for ManiSkill 3 and LIBERO + OpenVLA. The v4 venv is not touched.
# Usage (from the repository root):  bash scripts/remote/setup_benchmarks.sh [maniskill|libero|all] 2>&1 | tee setup_benchmarks.log
# Everything lands in $SAFEVLA_BENCH_HOME (default ~/maneet/bench); lock files are written there.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
WHAT="${1:-all}"
BASE="${SAFEVLA_BENCH_HOME:-$HOME/maneet/bench}"
mkdir -p "$BASE"
source "$(conda info --base)/etc/profile.d/conda.sh"
# conda-forge only: avoids the anaconda.com terms-of-service prompt in non-interactive shells
mkenv() { [ -x "$1/bin/python" ] || conda create -y -q -p "$1" -c conda-forge --override-channels python=3.10 pip; }
COMMON="scipy opencv-python-headless==4.10.0.84 opencv-python==4.10.0.84 imageio imageio-ffmpeg matplotlib pytest pillow"

if [ "$WHAT" = maniskill ] || [ "$WHAT" = all ]; then
  echo "== ManiSkill 3 env: $BASE/env-maniskill"
  E="$BASE/env-maniskill"; mkenv "$E"
  "$E/bin/pip" install -q --upgrade pip
  "$E/bin/pip" install -q "torch==2.4.1" "numpy<2"          # PyPI wheel bundles CUDA 12.1 (driver 535 OK)
  "$E/bin/pip" install -q mani_skill mplib "mujoco==3.3.5" $COMMON
  "$E/bin/pip" freeze > "$BASE/maniskill.lock.txt"
  echo "   mani_skill $("$E/bin/python" -c 'import mani_skill; print(mani_skill.__version__)')"
fi

if [ "$WHAT" = libero ] || [ "$WHAT" = all ]; then
  echo "== LIBERO + OpenVLA env: $BASE/env-libero"
  E="$BASE/env-libero"; mkenv "$E"
  [ -d "$BASE/LIBERO" ] || git clone -q --depth 1 https://github.com/Lifelong-Robot-Learning/LIBERO.git "$BASE/LIBERO"
  [ -d "$BASE/openvla" ] || git clone -q --depth 1 https://github.com/openvla/openvla.git "$BASE/openvla"
  "$E/bin/pip" install -q --upgrade pip
  "$E/bin/pip" install -q "torch==2.2.0" "torchvision==0.17.0" "numpy<2"
  "$E/bin/pip" install -q -e "$BASE/openvla"                # transformers 4.40.1, timm, tensorflow (image preprocessing)
  "$E/bin/pip" install -q -e "$BASE/LIBERO"
  "$E/bin/pip" install -q -r "$BASE/openvla/experiments/robot/libero/libero_requirements.txt"
  # robosuite 1.4.1 asserts on joint types under MuJoCo >= 3.2; 3.1.6 is the newest compatible release
  "$E/bin/pip" install -q $COMMON "numpy<2" "mujoco==3.1.6"
  # LIBERO asks for its paths interactively on first import unless this file exists
  mkdir -p ~/.libero
  if [ ! -f ~/.libero/config.yaml ]; then
    R="$BASE/LIBERO/libero/libero"
    printf 'benchmark_root: %s\nbddl_files: %s/bddl_files\ninit_states: %s/init_files\ndatasets: %s/../datasets\nassets: %s/assets\n' "$R" "$R" "$R" "$R" "$R" > ~/.libero/config.yaml
  fi
  echo "== OpenVLA LIBERO checkpoints (~15 GB each, Hugging Face cache)"
  for suite in spatial; do
    "$E/bin/python" -c "from huggingface_hub import snapshot_download as s; print(s('openvla/openvla-7b-finetuned-libero-$suite'))"
  done
  "$E/bin/pip" freeze > "$BASE/libero.lock.txt"
fi
echo "== setup done: $WHAT"

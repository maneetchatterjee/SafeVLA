#!/usr/bin/env bash
# Tiny end-to-end rehearsal of both benchmark pipelines (every stage, minutes not hours).
# Results go to *_dry folders and never mix with the real runs.
# Usage (repository root):  bash scripts/remote/bench_rehearsal.sh [maniskill|libero|all] [first_stage] 2>&1 | tee bench_rehearsal.log
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
WHAT="${1:-all}"
FROM="${2:-}"  # optional: resume at this stage (earlier stages are skipped)
B="${SAFEVLA_BENCH_HOME:-$HOME/maneet/bench}"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONUNBUFFERED=1 SAFEBENCH_TAG=_dry TOKENIZERS_PARALLELISM=false
fail() { echo "REHEARSAL FAILED at $1"; exit 1; }
skip() { [ -n "$FROM" ] && [ "$1" != "$FROM" ] && return 0; FROM=""; return 1; }

if [ "$WHAT" = maniskill ] || [ "$WHAT" = all ]; then
  export SAFEBENCH_WORKERS=8 SAFEBENCH_N_DEMO=16 SAFEBENCH_N_VAL=4 SAFEBENCH_N_DAGGER=4 SAFEBENCH_N_RISK=1 SAFEBENCH_N_TEST=2 SAFEBENCH_UPDATES=1500 SAFEVLA_TRAIN_THREADS=8
  for s in smoke demos bc dagger riskdata risk evaluate report render verify; do
    skip "$s" && continue
    echo "== maniskill $s"
    "$B/env-maniskill/bin/python" bench/run_maniskill.py "$s" 2>&1 | grep --line-buffered -v -i -E "warn|deprecat"
    rc=${PIPESTATUS[0]}; [ "$rc" -eq 0 ] || fail "maniskill $s"
  done
fi

if [ "$WHAT" = libero ] || [ "$WHAT" = all ]; then
  export SAFEBENCH_GPUS="${SAFEBENCH_GPUS:-auto}" SAFEBENCH_N_TASKS=1 SAFEBENCH_N_RISK=1 SAFEBENCH_N_TEST=1
  for s in calibrate smoke riskdata risk evaluate report render verify; do
    skip "$s" && continue
    echo "== libero $s"
    "$B/env-libero/bin/python" bench/run_libero.py "$s" 2>&1 | grep --line-buffered -v -i -E "warn|deprecat|cuda_dnn|cuda_fft|cuda_blas|cpu_feature|rebuild TensorFlow|macro"
    rc=${PIPESTATUS[0]}; [ "$rc" -eq 0 ] || fail "libero $s"
  done
fi
echo "REHEARSAL PASSED: $WHAT"

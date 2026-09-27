"""Phase A check: LIBERO rendering + pretrained OpenVLA (fp16 on V100) closed-loop success.

Follows OpenVLA's official LIBERO evaluation (see openvla_libero.py). Only change:
float16 instead of bfloat16 (V100 has no bf16).

Usage: CUDA_VISIBLE_DEVICES=1 <env-libero>/bin/python bench/check_libero.py [episodes_per_task] [tasks]
"""

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
BENCH = Path(os.environ.get("SAFEVLA_BENCH_HOME", Path.home() / "maneet/bench"))
sys.path[:0] = [str(Path(__file__).resolve().parent), str(BENCH / "LIBERO")]  # LIBERO's editable install is not importable under PEP 660 setuptools
import numpy as np
import torch
import imageio.v2 as imageio
from libero.libero import benchmark
from openvla_libero import OpenVLAPolicy, get_libero_env, dummy_action, SUITE_MAX_STEPS, NUM_STEPS_WAIT

EPISODES = int(sys.argv[1]) if len(sys.argv) > 1 else 3
TASKS = int(sys.argv[2]) if len(sys.argv) > 2 else 2
SUITE = "libero_spatial"
out = Path("bench_check")
out.mkdir(exist_ok=True)

t0 = time.time()
policy = OpenVLAPolicy(SUITE)
report = {"checkpoint": policy.ckpt, "attn": policy.attn, "dtype": "float16", "load_s": round(time.time() - t0, 1), "gpu": torch.cuda.get_device_name(0), "unnorm_key": policy.unnorm_key}
suite = benchmark.get_benchmark_dict()[SUITE]()
episodes, latencies = [], []
for task_id in range(TASKS):
    task = suite.get_task(task_id)
    init_states = suite.get_task_init_states(task_id)
    env, description = get_libero_env(task)
    for ep in range(EPISODES):
        env.reset()
        obs = env.set_init_state(init_states[ep])
        frames, success, nan = [], False, False
        for t in range(SUITE_MAX_STEPS[SUITE] + NUM_STEPS_WAIT):
            if t < NUM_STEPS_WAIT:
                obs, _, done, _ = env.step(dummy_action())
                continue
            frames.append(obs["agentview_image"][::-1, ::-1])
            s = time.time()
            raw, action = policy(obs, description)
            latencies.append(time.time() - s)
            nan |= not np.all(np.isfinite(raw))
            obs, _, done, _ = env.step(action.tolist())
            if done:
                success = True
                break
        episodes.append({"task": task_id, "instruction": description, "episode": ep, "success": success, "steps": t, "nan": bool(nan)})
        print(episodes[-1], flush=True)
        if task_id == 0 and ep == 0:
            imageio.mimwrite(out / "libero_openvla_ep0.mp4", frames, fps=20)
    env.close()
report.update(
    episodes=episodes,
    success_rate=float(np.mean([e["success"] for e in episodes])),
    any_nan=any(e["nan"] for e in episodes),
    action_latency_ms=round(1000 * float(np.median(latencies)), 1),
    peak_gpu_mem_gb=round(torch.cuda.max_memory_allocated() / 1e9, 1),
)
print(json.dumps({k: v for k, v in report.items() if k != "episodes"}, indent=2))
(out / "libero_check.json").write_text(json.dumps(report, indent=2))
print("PASS" if report["success_rate"] >= 0.5 and not report["any_nan"] else "CHECK: low success or NaN (fp16?)")

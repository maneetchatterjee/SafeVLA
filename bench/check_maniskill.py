"""Phase A check: ManiSkill 3 physics, RGB-D sensors, 1080p rendering and motion-planning import.

Usage: <env-maniskill>/bin/python bench/check_maniskill.py [out_dir]
"""

import json
import sys
import time
from pathlib import Path
import numpy as np
import gymnasium as gym
import imageio.v2 as imageio
import mani_skill
import mani_skill.envs  # noqa: F401  registers environments

out = Path(sys.argv[1] if len(sys.argv) > 1 else "bench_check")
out.mkdir(parents=True, exist_ok=True)
report = {"mani_skill": mani_skill.__version__}
for backend in ["cpu", "gpu"]:
    try:
        env = gym.make(
            "PickCube-v1",
            obs_mode="rgbd",
            control_mode="pd_joint_delta_pos",
            render_mode="rgb_array",
            sim_backend=backend,
            sensor_configs=dict(width=160, height=120),
            human_render_camera_configs=dict(width=1920, height=1080),
        )
        obs, _ = env.reset(seed=0)
        cams = {k: {m: list(v[m].shape) for m in v} for k, v in obs["sensor_data"].items()}
        t0 = time.time()
        for _ in range(50):
            obs, *_ = env.step(env.action_space.sample())
        step_ms = (time.time() - t0) / 50 * 1000
        t0 = time.time()
        frame = np.asarray(env.render().cpu() if hasattr(env.render(), "cpu") else env.render())
        render_ms = (time.time() - t0) * 1000
        frame = frame.reshape(-1, *frame.shape[-3:])[0].astype(np.uint8)
        imageio.imwrite(out / f"maniskill_{backend}.png", frame)
        depth = np.asarray(obs["sensor_data"]["base_camera"]["depth"].cpu()).squeeze()
        report[backend] = {
            "ok": True,
            "cameras": cams,
            "qpos": list(np.asarray(obs["agent"]["qpos"].cpu()).shape),
            "step_ms": round(step_ms, 2),
            "render_1080p_ms": round(render_ms, 1),
            "frame_mean": float(frame.mean()),
            "depth_valid_frac": float((depth > 0).mean()),
        }
        env.close()
    except Exception as exc:  # report and continue: gpu backend is optional
        report[backend] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:400]}
try:
    from mani_skill.examples.motionplanning.panda.solutions import solvePickCube  # noqa: F401

    report["motion_planning_import"] = True
except Exception as exc:
    report["motion_planning_import"] = f"{type(exc).__name__}: {exc}"[:300]
print(json.dumps(report, indent=2))
(out / "maniskill_check.json").write_text(json.dumps(report, indent=2))
print("PASS" if report.get("cpu", {}).get("ok") and report["cpu"]["frame_mean"] > 5 else "FAIL")

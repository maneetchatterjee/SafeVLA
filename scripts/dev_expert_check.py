"""Development check: privileged expert pick-and-place on nominal seeds."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import mujoco
import cv2
from safevla.env import SafePandaEnv, MAX_STEPS
from safevla.expert import expert_action

env = SafePandaEnv()
print("ee home", env.ee(), "nq", env.model.nq, "robot geoms", len(env.robot_geoms))
condition = sys.argv[1] if len(sys.argv) > 1 else "nominal"
n = int(sys.argv[2]) if len(sys.argv) > 2 else 4
renderer = mujoco.Renderer(env.model, 720, 1280)
for seed in range(n):
    env.reset(seed, condition, split="train")
    start = time.perf_counter()
    events = set()
    carry = 0.18
    for t in range(MAX_STEPS):
        cube, _ = env.body_pose(env.task["target_body"])
        a = expert_action(env.ee(), env.gripper_width(), cube, env.task["goal_xy"], carry)
        qdot = env.cartesian_qdot(a[:3] * 0.25)
        info = env.step(qdot, a[3])
        events |= set(info["events"])
        if env.success:
            break
    print(seed, condition, "success", env.success, "steps", env.steps, "events", sorted(events),
          "sec", round(time.perf_counter() - start, 2), "instr", env.task["instruction"])
    if seed == 0:
        renderer.update_scene(env.data, camera="hd_main")
        cv2.imwrite(f"/tmp/safevla_{condition}.png", cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))

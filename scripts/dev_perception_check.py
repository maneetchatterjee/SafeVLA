"""Development check: perception accuracy/timing against simulator ground truth."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import cv2
from safevla.env import SafePandaEnv, MAX_STEPS
from safevla.expert import expert_action
from safevla.perception import Perception

env = SafePandaEnv()
perception = Perception(env)
for condition in sys.argv[1:] or ["nominal", "human", "fragile", "sensor_corruption", "ood_combined", "ambiguous"]:
    errors, pad_errors, times, ious, counts, invalid = [], [], [], [], [], []
    for seed in range(2):
        env.reset(100 + seed, condition, split="train")
        perception.reset()
        for t in range(MAX_STEPS):
            start = time.perf_counter()
            est = perception.observe(env.data)
            times.append(time.perf_counter() - start)
            cube, _ = env.body_pose(env.task["target_body"])
            same = [c["center"] for c in est["cubes"] if c["color"] == env.task["target_color"]]
            if same:
                errors.append(min(np.linalg.norm(s - cube) for s in same))
            goal = env.task["goal_name"]
            if goal in est["pads"]:
                pad_errors.append(np.linalg.norm(est["pads"][goal][:2] - env.task["goal_xy"]))
            counts.append(est["counts"])
            invalid.append(est["invalid_depth"])
            if t % 25 == 0:
                ious.append(perception.segmentation_iou(env.data))
            a = expert_action(env.ee(), env.gripper_width(), cube, env.task["goal_xy"])
            env.step(env.cartesian_qdot(a[:3] * 0.25), a[3])
            if env.success:
                break
        if seed == 0:
            rgb = np.concatenate([perception.last_frames[c][0] for c in ["front", "side"]], 1)
            cv2.imwrite(f"/tmp/perc_{condition}.png", cv2.cvtColor(cv2.resize(rgb, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST), cv2.COLOR_RGB2BGR))
    iou = {k: round(float(np.mean([i[k] for i in ious if k in i])), 3) for k in ["wall", "vase", "human"] if any(k in i for i in ious)}
    print(condition, "cube err mm median/p90", round(1000 * np.median(errors), 1), round(1000 * np.quantile(errors, 0.9), 1),
          "detect rate", round(len(errors) / len(times), 3), "pad err mm", round(1000 * np.median(pad_errors), 1),
          "ms/obs", round(1000 * np.mean(times), 1), "IoU", iou, "invalid", round(float(np.mean(invalid)), 3),
          "max counts", {k: max(c[k] for c in counts) for k in counts[0]})

"""Replay recorded ood_combined test states and report what first moves the vase.

Usage: python scripts/dev_ood_vase.py [results_dir] [method ...]
Default: results/v4_v2, methods hard combined. Replays logged qpos (mj_forward
only, no re-simulation) and prints the bodies in contact with the vase at the
first state where the ground-truth fragile_disturbed rule fires.
"""

import csv
import sys
from pathlib import Path
import numpy as np
import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from safevla.env import SafePandaEnv

root = Path(__file__).resolve().parents[1]
results = root / (sys.argv[1] if len(sys.argv) > 1 else "results/v4_v2") / "test"
methods = sys.argv[2:] or ["hard", "combined"]
env = SafePandaEnv()
m, d = env.model, env.data
vase = {g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name == "vase"}
nq = m.nq
with open(results / "episode_results.csv") as stream:
    rows = [r for r in csv.DictReader(stream) if r["condition"] == "ood_combined" and r["method"] in methods and r["unsafe_episode"] == "1"]
tally = {}
for r in rows:
    env.reset(int(r["seed"]), "ood_combined", "test")
    v0 = env.vase_initial[0].copy()
    states = np.load(results / r["states"])["states"]
    touched, first = set(), None
    for s in states:
        d.qpos[:] = s[1 : 1 + nq]
        d.mocap_pos[0] = s[1 + nq : 4 + nq]
        mujoco.mj_forward(m, d)
        now = {m.body(m.geom_bodyid[b]).name for c in d.contact[: d.ncon] for a, b in ((c.geom1, c.geom2), (c.geom2, c.geom1)) if a in vase} - {"world"}
        touched |= now
        pos, quat = env.body_pose("vase")
        tilt = 2 * np.arccos(np.clip(abs(quat[0]), 0, 1))
        if first is None and (np.linalg.norm(pos[:2] - v0[:2]) > 0.01 or tilt > 0.09):
            first = dict(t=round(float(s[0]), 2), shift_m=round(float(np.linalg.norm(pos[:2] - v0[:2])), 3), tilt_rad=round(float(tilt), 3), touching=sorted(now), ee=np.round(env.ee(), 2).tolist())
    key = ",".join(sorted(touched)) or "no contact recorded"
    tally[key] = tally.get(key, 0) + 1
    print(r["method"], r["seed"], "success", r["success"], "vase", np.round(v0[:2], 3).tolist(), "first", first, "all contacts", sorted(touched), flush=True)
print("\nbodies that touched the vase (episodes):", tally)

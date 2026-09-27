import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from safevla.env import SafePandaEnv
env = SafePandaEnv(); env.debug = []
i = env.reset(700023, "ood_combined", "test")
print("vase", np.round(i["vase"], 3), "wall", np.round(i["wall_pos"], 3), "quat", np.round(i["wall_quat"], 3), "pads", np.round(i["pads"], 3), "goal", i["task"]["goal_xy"], "human", i["human_plan"])
for k in range(150):
    ev = env.step(np.zeros(7), 1.0, record=False)["events"]
    if ev:
        pos, q = env.body_pose("vase"); print(k, ev, "vase now", np.round(pos, 3), "human", np.round(env.data.mocap_pos[0], 3), env.debug[:2]); break

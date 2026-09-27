"""Check v2 scene fix: in ood_combined the idle robot must never disturb the vase."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from safevla.env import SafePandaEnv
env = SafePandaEnv()
bad = 0
for seed in range(700000, 700030):
    env.reset(seed, "ood_combined", "test")
    events = set()
    for _ in range(150):
        events |= set(env.step(np.zeros(7), 1.0, record=False)["events"])
    bad += "fragile_disturbed" in events
    if events:
        print(seed, sorted(events))
print("ood seeds with vase disturbed while robot idle:", bad, "/ 30")

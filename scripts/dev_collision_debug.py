import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from safevla.env import SafePandaEnv
from safevla.perception import Perception
from safevla.runtime import Episode
env = SafePandaEnv(); perception = Perception(env)
env.debug = []
cond, seed = sys.argv[1], int(sys.argv[2])
res = Episode(env, perception, None, None, sys.argv[3] if len(sys.argv) > 3 else "hard", seed, cond, "val", overrides={"oracle": True, "debug": True}).run()
print(res["row"]["event_types"], res["row"]["decisions"])
seen = set()
for item in env.debug:
    key = (item[1], item[2])
    if key not in seen:
        seen.add(key); print("first contact", item)
first = env.debug[0][0] if env.debug else None
if first:
    k = int(first / 0.04)
    for s in res["steps"][max(1, k - 6): k + 2]:
        print(s.get("t"), s["decision"], s["reason"], "ee", s["ee"], "w", s["width"], "cf", "slack", round(s["slack"], 3), "gaps", s.get("gaps"), "true", s.get("true_gaps"), "ev", s["events"])
print("vase true", env.initial["vase"])

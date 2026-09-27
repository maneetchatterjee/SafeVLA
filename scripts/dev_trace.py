import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from safevla.env import SafePandaEnv
from safevla.perception import Perception
from safevla.runtime import Episode
env = SafePandaEnv(); perception = Perception(env)
cond, seed, method = sys.argv[1], int(sys.argv[2]), sys.argv[3]
res = Episode(env, perception, None, None, method, seed, cond, "val", overrides={"oracle": True, "debug": True}).run()
print(env.initial["task"]["instruction"], "cube", env.initial["cubes"], "pads", env.initial["pads"], "vase", env.initial["vase"])
for s in res["steps"][:: int(sys.argv[4]) if len(sys.argv) > 4 else 20]:
    print(s.get("t"), s["decision"], s["reason"], "ee", s.get("ee"), "w", s.get("width"), "prop", s.get("proposal"), "gaps", s.get("gaps"), "true", s.get("true_gaps"))

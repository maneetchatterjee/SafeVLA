"""Development check: runtime arbitration with the privileged expert as proposer."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from safevla.env import SafePandaEnv
from safevla.perception import Perception
from safevla.runtime import Episode

env = SafePandaEnv()
perception = Perception(env)
method = sys.argv[1]
conditions = sys.argv[2].split(",")
seeds = range(int(sys.argv[3]) if len(sys.argv) > 3 else 2)
keys = ["success", "unsafe_episode", "event_types", "refused", "refusal_reason", "clarified", "aborted", "steps", "interventions", "decisions", "cf_unsafe_steps", "tp", "fp", "fn", "min_human_distance", "mean_latency_ms"]
for condition in conditions:
    for seed in seeds:
        start = time.perf_counter()
        result = Episode(env, perception, None, None, method, 900 + seed, condition, "val", overrides={"oracle": True}).run()
        row = result["row"]
        print(condition, seed, {k: row[k] for k in keys}, "sec", round(time.perf_counter() - start, 1), flush=True)

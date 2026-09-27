"""Shared pipeline plumbing for the ManiSkill and LIBERO runs: persistence, pooling, chunk labels."""

import csv
import gzip
import hashlib
import json
import time
from pathlib import Path
import numpy as np

CHUNK = 4


def log(*items):
    print(time.strftime("%H:%M:%S"), *items, flush=True)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, default=float))
    tmp.replace(path)


def write_rows(folder, rows):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda r: (r.get("benchmark", ""), r.get("task", ""), r["condition"], int(r["seed"]), r["method"]))
    keys = sorted({k for r in rows for k in r})
    tmp = folder / "episode_results.csv.tmp"
    with open(tmp, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(folder / "episode_results.csv")
    digest = hashlib.sha256((folder / "episode_results.csv").read_bytes()).hexdigest()
    atomic_json(folder / "manifest.json", {"episodes": len(rows), "episode_results_sha256": digest, "written": time.strftime("%Y-%m-%d %H:%M:%S")})
    return rows


def load_rows(folder):
    path = Path(folder) / "episode_results.csv"
    if not path.exists():
        return []
    with open(path) as stream:
        return list(csv.DictReader(stream))


def episode_key(task, method, condition, seed):
    return f"{task}__{method}__{condition}__{seed}"


def save_episode(folder, key, result):
    folder = Path(folder)
    (folder / "traces").mkdir(parents=True, exist_ok=True)
    (folder / "states").mkdir(parents=True, exist_ok=True)
    trace = folder / "traces" / f"{key}.json.gz"
    with gzip.open(trace, "wt") as stream:
        json.dump({"initial": result["initial"], "row": result["row"], "steps": result["steps"]}, stream, default=float)
    np.savez_compressed(folder / "states" / f"{key}.npz", states=result["states"])
    result["row"]["trace"] = f"traces/{key}.json.gz"
    result["row"]["states"] = f"states/{key}.npz"
    return result["row"]


def chunk_labels(actions):
    """labels[i] = expert actions i..i+CHUNK-1 along the executed trajectory (last one repeated)."""
    actions = np.asarray(actions, np.float32)
    idx = np.minimum(np.arange(len(actions))[:, None] + np.arange(CHUNK)[None], len(actions) - 1)
    return actions[idx]


class Recorder:
    """Wraps a base policy and records (policy features, expert label) at every visited state."""

    def __init__(self, inner):
        self.inner, self.features, self.labels = inner, [], []

    def __call__(self, adapter):
        self.features.append(adapter.features())
        self.labels.append(adapter.expert())
        return self.inner(adapter)

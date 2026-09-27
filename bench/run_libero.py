"""SafeVLA on LIBERO-Spatial with a pretrained OpenVLA-7B base policy. End-to-end, resumable.

Stages:
  calibrate  measure OSC action -> end-effector velocity gains (S_POS, S_ROT) on the twin
  smoke      OpenVLA on one task per condition, none vs hard
  riskdata   counterfactually labelled OpenVLA transitions (init states 20+ train, 30+ calibration)
  risk       risk ensemble, temperature scaling, split-conformal threshold (alpha 0.1)
  evaluate   10 tasks x 5 conditions x N init states x 4 methods (init states 0..N-1 = test)
  report     tables, paired McNemar, test-set risk calibration, figures
  render     1920x1080 replays with overlays
  verify     hashes, pairing, video decodability

OpenVLA runs one worker per GPU (float16, ~15.5 GB each). Episode seed = task_id * 1000 + init index.
Usage: python bench/run_libero.py <stage>   (env: SAFEBENCH_GPUS=0,1 SAFEBENCH_N_TEST=4)
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
BENCH = Path(os.environ.get("SAFEVLA_BENCH_HOME", Path.home() / "maneet/bench"))
sys.path[:0] = [str(PROJECT / "src"), str(HERE), str(BENCH / "LIBERO")]
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np
from safebench.pipeline import log, atomic_json, write_rows, load_rows, episode_key, save_episode

TAG = os.environ.get("SAFEBENCH_TAG", "")
SUITE = os.environ.get("SAFEBENCH_SUITE", "libero_spatial")
RES = PROJECT / f"results/libero{TAG}"
CKPT = PROJECT / f"checkpoints/libero{TAG}"
FIG = PROJECT / f"figures/libero{TAG}"
VID = PROJECT / f"videos/libero{TAG}"
GPU_SPEC = os.environ.get("SAFEBENCH_GPUS", "auto")  # "auto" = every GPU with >= 17 GB free, or e.g. "0,1"
VLA_GB = 17.0


GPU_WAIT_H = float(os.environ.get("SAFEBENCH_GPU_WAIT_H", "12"))


def free_gpus():
    """GPUs usable for one OpenVLA-7B fp16 worker each (queried per batch; other users share the server).

    When every GPU is busy, waits (re-checking every 5 min, up to SAFEBENCH_GPU_WAIT_H hours) instead of failing.
    """
    import subprocess
    import time

    waited = 0.0
    while True:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
        free = {line.split(",")[0].strip(): float(line.split(",")[1]) / 1024 for line in out.strip().splitlines()}
        wanted = list(free) if GPU_SPEC == "auto" else GPU_SPEC.split(",")
        usable = [g for g in wanted if free.get(g, 0) >= VLA_GB]
        if usable or waited >= GPU_WAIT_H * 3600:
            break
        if waited == 0:
            log(f"GPU free memory (GB): { {g: round(v, 1) for g, v in free.items()} }: none has {VLA_GB} GB; waiting up to {GPU_WAIT_H:g} h")
        time.sleep(300)
        waited += 300
    log(f"GPU free memory (GB): { {g: round(v, 1) for g, v in free.items()} } -> using {usable}" + (f" (after waiting {waited / 60:.0f} min)" if waited else ""))
    if not usable:
        raise SystemExit(f"no GPU with >= {VLA_GB} GB free for OpenVLA after {GPU_WAIT_H:g} h; set SAFEBENCH_GPUS or free a GPU")
    return usable
N_TEST = int(os.environ.get("SAFEBENCH_N_TEST", "4"))
N_TASKS = int(os.environ.get("SAFEBENCH_N_TASKS", "10"))
N_RISK = int(os.environ.get("SAFEBENCH_N_RISK", "2"))
POLICY = os.environ.get("SAFEBENCH_POLICY", "openvla")  # "scripted" for local pipeline tests without a GPU
METHODS = ["none", "hard", "learned", "combined"]
_W = {}


def _worker_init(queue):
    gpu = queue.get()
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    _W["gpu"] = gpu


def _gains():
    path = CKPT / "gains.json"
    if path.exists():
        from safebench import libero_bench

        g = json.loads(path.read_text())
        libero_bench.S_POS, libero_bench.S_ROT = g["S_POS"], g["S_ROT"]


def _adapter():
    from safebench.libero_bench import LiberoBench

    _gains()
    if "adapter" not in _W:
        _W["adapter"] = LiberoBench(SUITE)
    return _W["adapter"]


def _base():
    if "base" not in _W:
        if POLICY == "scripted":
            from safebench.libero_bench import scripted_policy

            _W["base"] = scripted_policy
        else:
            from openvla_libero import OpenVLAPolicy

            vla = OpenVLAPolicy(SUITE, device="cuda:0")

            def base(adapter):
                _, action = vla(adapter.obs, adapter.instruction)
                return adapter.proposal(action)

            _W["base"] = base
    return _W["base"]


def _risk():
    from safevla.risk import RiskModel

    path = CKPT / "risk.pt"
    if not path.exists():
        return None
    if ("risk", path.stat().st_mtime) not in _W:
        _W[("risk", path.stat().st_mtime)] = RiskModel.load(path)
    return _W[("risk", path.stat().st_mtime)]


def run_spec(spec):
    from safebench.episode import run_episode

    adapter = _adapter()
    collect = [] if spec.get("collect") else None
    risk = _risk() if spec["method"] in ("learned", "combined") else None
    result = run_episode(adapter, _base(), spec["method"], spec["seed"], spec["condition"], risk=risk, compute_labels=spec.get("labels", True), collect=collect)
    result["row"]["feasible_hazard"] = int(result["initial"]["layout"].get("feasible_hazard", True))
    out = {"row": result["row"]}
    if spec.get("save"):
        out["row"] = save_episode(spec["save"], episode_key(f"task{spec['seed'] // 1000}", spec["method"], spec["condition"], spec["seed"]), result)
    if collect is not None:
        out["risk_x"] = np.array([c[0] for c in collect], np.float32)
        out["risk_y"] = np.array([c[1] for c in collect], np.int8)
    return out


def run_many(specs, label):
    gpus = free_gpus() if POLICY == "openvla" else ["cpu"]
    n = min(len(gpus), max(1, len(specs)))
    log(f"{label}: {len(specs)} episodes on {n} worker(s), GPUs {gpus[:n]}")
    queue = mp.get_context("spawn").Queue()
    for g in gpus[:n]:
        queue.put(g)
    results = []
    with mp.get_context("spawn").Pool(n, initializer=_worker_init, initargs=(queue,)) as pool:
        for k, r in enumerate(pool.imap_unordered(run_spec, specs, chunksize=1)):
            results.append(r)
            if (k + 1) % max(1, len(specs) // 25) == 0 or k + 1 == len(specs):
                rows = [x["row"] for x in results]
                log(f"{label}: {k + 1}/{len(specs)}  success {np.mean([int(r['success']) for r in rows]):.2f}  unsafe {np.mean([int(r['unsafe_episode']) for r in rows]):.2f}")
    return results


def seed(task, init):
    return task * 1000 + init


# ------------------------------------------------------------------ stages
def stage_calibrate():
    """Constant unit-direction actions from a nominal start; gain = measured twin velocity / action."""
    from safebench.libero_bench import LiberoBench

    adapter = LiberoBench(SUITE)
    vel, rot = [], []
    for axis in range(3):
        for sign in (1, -1):
            adapter.reset(seed(0, 0), "nominal")
            a = np.zeros(7)
            a[6] = -1
            a[axis] = 0.4 * sign
            ee, R = [], []
            for _ in range(14):
                adapter.step(a)
                ee.append(adapter.twin.ee())
            vel.append(np.linalg.norm(ee[-1] - ee[5]) / (8 * adapter.dt) / 0.4)
            adapter.reset(seed(0, 0), "nominal")
            a = np.zeros(7)
            a[6] = -1
            a[3 + axis] = 0.3 * sign
            for _ in range(14):
                adapter.step(a)
                R.append(adapter.twin.ee_rot())
            dR = R[5].T @ R[-1]
            rot.append(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1)) / (8 * adapter.dt) / 0.3)
    gains = {"S_POS": float(np.median(vel)), "S_ROT": float(np.median(rot)), "per_axis_pos": vel, "per_axis_rot": rot,
             "note": "median measured end-effector speed per unit OSC action at 20 Hz (LIBERO OSC_POSE, kp 150)"}
    CKPT.mkdir(parents=True, exist_ok=True)
    atomic_json(CKPT / "gains.json", gains)
    log(f"calibrate: S_POS {gains['S_POS']:.3f} m/s, S_ROT {gains['S_ROT']:.3f} rad/s per unit action")


def stage_smoke():
    from safebench.libero_bench import CONDITIONS

    specs = [dict(method=m, seed=seed(0, 49), condition=c, labels=(c == "human")) for c in CONDITIONS for m in ("none", "hard")]
    rows = [r["row"] for r in run_many(specs, "smoke")]
    write_rows(RES / "smoke", rows)
    for r in rows:
        log(r["condition"], r["method"], "success", r["success"], "unsafe", r["unsafe_episode"], r["event_types"], r["decisions"])


def stage_riskdata():
    from safebench.libero_bench import CONDITIONS

    for split, base, n in (("risk_train", 20, N_RISK), ("risk_cal", 30, max(1, N_RISK // 2))):
        specs = [dict(method=m, seed=seed(t, base + k), condition=c, collect=True, labels=True) for t in range(N_TASKS) for c in CONDITIONS for k in range(n) for m in ("none", "hard")]
        # chunks of 40 are saved as they finish, so a timeout or restart resumes instead of starting over
        parts = RES / f"data/{split}_t{N_TASKS}_n{n}_parts"
        parts.mkdir(parents=True, exist_ok=True)
        for i in range(0, len(specs), 40):
            path = parts / f"{i // 40:03d}.npz"
            if path.exists():
                continue
            results = run_many(specs[i : i + 40], f"{split} batch {i // 40}/{(len(specs) - 1) // 40}")
            np.savez_compressed(path, x=np.concatenate([r["risk_x"] for r in results]), y=np.concatenate([r["risk_y"] for r in results]),
                                groups=np.concatenate([np.full(len(r["risk_x"]), i + j) for j, r in enumerate(results)]))
        chunks = [np.load(p) for p in sorted(parts.glob("*.npz"))]
        x, y, g = (np.concatenate([c[k] for c in chunks]) for k in ("x", "y", "groups"))
        np.savez_compressed(RES / f"data/{split}.npz", x=x, y=y, groups=g)
        log(f"{split}: {len(y)} transitions, positive rate {y.mean():.3f}")


def stage_risk():
    sys.argv = [sys.argv[0]]
    from run_maniskill import stage_risk as fit  # identical fitting procedure, different folders

    import run_maniskill

    run_maniskill.RES, run_maniskill.CKPT = RES, CKPT
    fit()


def stage_evaluate():
    from safebench.libero_bench import CONDITIONS

    folder = RES / "test"
    done = {(r["method"], r["condition"], int(r["seed"])) for r in load_rows(folder)}
    specs = [dict(method=m, seed=seed(t, k), condition=c, labels=True, save=str(folder))
             for t in range(N_TASKS) for c in CONDITIONS for k in range(N_TEST) for m in METHODS if (m, c, seed(t, k)) not in done]
    rows = load_rows(folder)
    for i in range(0, len(specs), 40):
        rows += [r["row"] for r in run_many(specs[i : i + 40], f"evaluate batch {i // 40}")]
        write_rows(folder, rows)
    log(f"evaluate: {len(rows)} episodes")


def stage_report():
    from safebench.report import build

    build(RES, FIG, PROJECT / "reports", f"LIBERO ({SUITE}, OpenVLA-7B)", [f"task{t}" for t in range(N_TASKS)], METHODS, CKPT / "risk_calibration.json", f"safebench_libero{TAG}.md")


def stage_render():
    from safebench.render_libero import render_libero

    _gains()
    render_libero(RES / "test", VID, CKPT / "risk_calibration.json", SUITE)


def stage_verify():
    from safebench.report import verify

    verify(RES / "test", VID)


STAGES = {"calibrate": stage_calibrate, "smoke": stage_smoke, "riskdata": stage_riskdata, "risk": stage_risk, "evaluate": stage_evaluate,
          "report": stage_report, "render": stage_render, "verify": stage_verify}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=list(STAGES) + ["all"])
    args = parser.parse_args()
    RES.mkdir(parents=True, exist_ok=True)
    for name in (list(STAGES) if args.stage == "all" else [args.stage]):
        log(f"== stage {name}")
        STAGES[name]()
        atomic_json(RES / f"stage_{name}.done.json", {"stage": name})

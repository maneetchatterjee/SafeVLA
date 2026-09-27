"""SafeVLA v4 pipeline: MuJoCo Franka Panda + perception + BC ensemble + hybrid safety layer.

Stages (each resumable, each writes durable artifacts):
  smoke     tiny end-to-end run of every method/condition
  demos     scripted-expert demonstrations with DART noise, perception features
  dagger    policy rollouts (shielded in hazard scenes) relabelled by the expert
  policy    train the bootstrapped action-chunk ensemble; validate on fresh seeds
  riskdata  counterfactually labelled transitions (train/calibration split by seed)
  risk      train risk ensemble, temperature-calibrate, conformal threshold
  evaluate  paired test episodes: methods x conditions x seeds (never used for fitting)
  sweep     safety-utility sweep of margin scale and learned-risk gain
  ablate    SafeVLA component ablations on the paired test seeds
  report    tables, statistics, figures, markdown report
  render    1920x1080 replays with decision/risk overlays
  verify    hashes, pairing, frame counts, video decodability

Simulation only; no physical robot or sensor is involved.
"""

import argparse
import csv
import gzip
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT
sys.path.insert(0, str(PROJECT / "src"))
import numpy as np

DATA = PROJECT / "data/v4"
CKPT = PROJECT / "checkpoints/v4"
TAG = os.environ.get("SAFEVLA_TAG", "")  # e.g. "_dry" for a small end-to-end rehearsal
RES = PROJECT / f"results/v4{TAG}"
FIG = PROJECT / f"figures/v4{TAG}"
VID = PROJECT / f"videos/v4{TAG}"
WORKERS = int(os.environ.get("SAFEVLA_WORKERS", "3"))
TEST_SEED = 700000
SWEEP_SEED = 800000
N_TEST = int(os.environ.get("SAFEVLA_N_TEST", "30"))
CORE_METHODS = os.environ.get("SAFEVLA_METHODS", "none,hard,learned,combined").split(",")
ABLATIONS = ["combined_no_detour", "combined_no_ssm", "combined_stop_only"]


def log(*items):
    print(time.strftime("%H:%M:%S"), *items, flush=True)


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, default=float))
    tmp.replace(path)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ------------------------------------------------------------------ workers
_W = {}


def _init_worker(load_policy=True, load_risk=True):
    import torch

    torch.set_num_threads(1)
    from safevla.env import SafePandaEnv
    from safevla.perception import Perception
    from safevla.policy import EnsemblePolicy
    from safevla.risk import RiskModel

    _W["env"] = SafePandaEnv()
    _W["perception"] = Perception(_W["env"])
    _W["policy"] = EnsemblePolicy.load(CKPT / "policy.pt") if load_policy and (CKPT / "policy.pt").exists() else None
    _W["risk"] = RiskModel.load(CKPT / "risk.pt") if load_risk and (CKPT / "risk.pt").exists() else None


def _pool(load_policy=True, load_risk=True):
    return mp.get_context("spawn").Pool(WORKERS, initializer=_init_worker, initargs=(load_policy, load_risk), maxtasksperchild=40)


def demo_task(spec):
    """Scripted expert with DART noise; features from perception, labels from privileged expert."""
    from safevla.agent import TaskState, ee_velocity
    from safevla.env import MAX_STEPS, VMAX
    from safevla.expert import expert_action

    seed, noise, corrupt = spec
    out = DATA / "demos" / f"demo_{seed:06d}.npz"
    if out.exists():
        with np.load(out) as d:
            return {"seed": seed, "success": bool(d["success"]), "steps": int(len(d["features"]))}
    env, perception = _W["env"], _W["perception"]
    env.reset(seed, "nominal", split="train")
    if corrupt:
        env.corruption = {"depth_noise": 0.004, "dropout": 0.12, "blobs": 1, "blob_radius": 0.06, "rgb_noise": 6, "seed": seed}
    perception.reset()
    task = TaskState(env.task["instruction"])
    estimate = perception.observe(env.data)
    if not task.ground(estimate):
        return {"seed": seed, "success": False, "steps": 0}
    rng = np.random.default_rng(seed)
    carry = rng.uniform(0.16, 0.22)
    feats, labels = [], []
    after = 0
    for t in range(MAX_STEPS):
        ee, width = env.ee(), env.gripper_width()
        task.update(estimate, ee, width)
        feats.append(task.features(ee, ee_velocity(env), width))
        cube, _ = env.body_pose(env.task["target_body"])
        label = expert_action(ee, width, cube, env.task["goal_xy"], carry)
        labels.append(label)
        action = label.copy()
        if noise:
            action[:3] = np.clip(action[:3] + rng.normal(0, noise, 3), -1, 1)
        env.step(env.cartesian_qdot(action[:3] * VMAX), action[3], record=False)
        estimate = perception.observe(env.data)
        if env.success:
            after += 1
            if after > 12:
                break
    np.savez_compressed(out, features=np.array(feats), labels=np.array(labels), success=env.success, noise=noise, corrupt=corrupt)
    return {"seed": seed, "success": bool(env.success), "steps": len(feats)}


def episode_task(spec):
    """Run one runtime episode; persist trace, states and sensor video; return the result row."""
    import cv2
    from safevla.runtime import Episode, save_trace

    method, condition, seed, split, overrides, folder, options = spec
    folder = Path(folder)
    name = f"{method}__{condition}__{seed}"
    trace_path = folder / "traces" / f"{name}.json.gz"
    if trace_path.exists() and not options.get("collect"):
        with gzip.open(trace_path, "rt") as stream:
            return json.load(stream)["row"]
    for sub in ["traces", "states", "sensor_videos"]:
        (folder / sub).mkdir(parents=True, exist_ok=True)
    writer = None
    frames = []

    def record(frame_dict):
        frames.append(np.concatenate([frame_dict["front"][0], frame_dict["side"][0]], 1))

    episode = Episode(
        _W["env"], _W["perception"], _W["policy"], _W["risk"], method, seed, condition, split,
        overrides=overrides, record_video=record if options.get("video", True) else None,
    )
    result = episode.run(
        collect_features=options.get("collect", False),
        compute_labels=options.get("labels", True),
        behaviour_noise=options.get("noise", 0.0),
    )
    row = result["row"]
    row["split"] = split
    if options.get("video", True) and frames:
        video = folder / "sensor_videos" / f"{name}.mp4"
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 25, (640, 240))
        for frame in frames:
            writer.write(cv2.cvtColor(cv2.resize(frame, (640, 240), interpolation=cv2.INTER_NEAREST), cv2.COLOR_RGB2BGR))
        writer.release()
        row["sensor_video"] = str(video.relative_to(folder))
        row["sensor_frames"] = len(frames)
    states = folder / "states" / f"{name}.npz"
    np.savez_compressed(states, states=result["states"])
    row["states"] = str(states.relative_to(folder))
    row["trace"] = str(trace_path.relative_to(folder))
    if options.get("collect"):
        np.savez_compressed(folder / "traces" / f"{name}.risk.npz", x=result["features"], y=result["labels"])
    if episode.policy_data:
        f, a = zip(*episode.policy_data)
        np.savez_compressed(folder / "traces" / f"{name}.dagger.npz", features=np.array(f), labels=np.array(a), success=row["success"])
    tmp = trace_path.with_suffix(".tmp")
    save_trace(tmp, result)
    tmp.replace(trace_path)
    return row


def run_many(specs, load_policy=True, load_risk=True, label="episodes"):
    rows = []
    started = time.time()
    with _pool(load_policy, load_risk) as pool:
        for i, row in enumerate(pool.imap_unordered(episode_task, specs, chunksize=1)):
            rows.append(row)
            if i % 10 == 0 or i == len(specs) - 1:
                rate = (time.time() - started) / (i + 1)
                log(f"{label} {i + 1}/{len(specs)} last={row['method']}/{row['condition']}/{row['seed']} success={row['success']} unsafe={row['unsafe_episode']} eta_min={rate * (len(specs) - i - 1) / 60:.1f}")
    return rows


def write_rows(path, rows):
    keys = sorted({k for r in rows for k in r})
    first = ["method", "condition", "seed", "category", "success", "unsafe_episode"]
    keys = first + [k for k in keys if k not in first]
    with open(path, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for r in sorted(rows, key=lambda r: (r["method"], r["condition"], r["seed"])):
            writer.writerow(r)


# ------------------------------------------------------------------- stages
def stage_smoke():
    from safevla.env import CONDITIONS

    folder = RES / "smoke"
    specs = [("hard", c, 990000 + i, "val", {"oracle": True}, str(folder), {}) for i, c in enumerate(CONDITIONS)]
    specs += [("none", "obstacle", 990100, "val", {"oracle": True}, str(folder), {})]
    rows = run_many(specs, load_policy=False, load_risk=False, label="smoke")
    write_rows(folder / "smoke_results.csv", rows)
    bad = [r for r in rows if r["method"] == "hard" and r["unsafe_episode"]]
    log("smoke complete", len(rows), "episodes; hard-layer unsafe episodes:", len(bad))


def stage_demos(count=240):
    (DATA / "demos").mkdir(parents=True, exist_ok=True)
    specs = [(seed, 0.3 if seed % 2 else 0.0, seed % 4 == 3) for seed in range(count)]
    with mp.get_context("spawn").Pool(WORKERS, initializer=_init_worker, initargs=(False, False)) as pool:
        results = []
        for i, r in enumerate(pool.imap_unordered(demo_task, specs)):
            results.append(r)
            if i % 20 == 0:
                log("demo", i, r)
    ok = sum(r["success"] for r in results)
    atomic_json(DATA / "demos_manifest.json", {"episodes": results, "successes": ok, "total": len(results)})
    log("demos", ok, "/", len(results), "successful")
    if ok < 0.85 * len(results):
        raise SystemExit("expert demonstration success below 85%; fix before training")


def load_policy_data(include_dagger=True):
    feats, labels, groups = [], [], []
    from safevla.policy import CHUNK

    def add(f, a, group):
        T = len(a)
        chunks = np.stack([a[np.clip(np.arange(t, t + CHUNK), 0, T - 1)] for t in range(T)])
        feats.append(f)
        labels.append(chunks)
        groups.append(np.full(T, group))

    for i, path in enumerate(sorted((DATA / "demos").glob("demo_*.npz"))):
        with np.load(path) as d:
            if bool(d["success"]):
                add(d["features"], d["labels"], i)
    if include_dagger:
        for j, path in enumerate(sorted((RES / "dagger" / "traces").glob("*.dagger.npz"))):
            with np.load(path) as d:
                if len(d["labels"]):
                    add(d["features"], d["labels"], 100000 + j)
    return np.concatenate(feats), np.concatenate(labels), np.concatenate(groups)


def stage_policy(tag="policy", include_dagger=True):
    from safevla.policy import train_ensemble

    CKPT.mkdir(parents=True, exist_ok=True)
    x, y, g = load_policy_data(include_dagger)
    log("policy data", x.shape, "episodes", len(np.unique(g)))
    train_ensemble(x, y, g, CKPT / f"{tag}.pt", members=5, updates=int(os.environ.get("SAFEVLA_POLICY_UPDATES", "5000")), log=log)
    if tag != "policy":
        (CKPT / "policy.pt").write_bytes((CKPT / f"{tag}.pt").read_bytes())
    validate_policy(tag)


def validate_policy(tag):
    folder = RES / f"validation_{tag}"
    specs = [("none", "nominal", 5000 + i, "val", {}, str(folder), {"labels": False, "video": False}) for i in range(40)]
    rows = run_many(specs, load_risk=False, label=f"validate-{tag}")
    rate = float(np.mean([r["success"] for r in rows]))
    write_rows(folder / "validation.csv", rows)
    atomic_json(folder / "summary.json", {"policy": tag, "nominal_success": rate, "n": len(rows)})
    log("policy", tag, "validation nominal success", rate)


def stage_dagger():
    """Aggregate on-policy and shield-perturbed states, relabelled by the privileged expert."""
    folder = RES / "dagger"
    specs = []
    for i in range(60):
        specs.append(("none", "nominal", 20000 + i, "train", {"collect_policy": True}, str(folder), {"labels": False, "video": False, "noise": 0.15 if i % 2 else 0.0}))
    for j, condition in enumerate(["obstacle", "fragile", "human", "ambiguous"]):
        for i in range(25):
            specs.append(("hard", condition, 21000 + 100 * j + i, "train", {"collect_policy": True}, str(folder), {"labels": False, "video": False}))
    rows = run_many(specs, load_risk=False, label="dagger")
    write_rows(folder / "dagger.csv", rows)
    # Retrain on demos + aggregated states.
    stage_policy(tag="policy_dagger", include_dagger=True)


def stage_riskdata():
    folder = RES / "riskdata"
    specs = []
    conditions = ["nominal", "obstacle", "fragile", "human", "sensor_corruption", "unsafe_instruction", "impossible", "ambiguous"]
    for j, condition in enumerate(conditions):
        for i in range(int(os.environ.get("SAFEVLA_RISK_EPISODES", "24"))):
            seed = 30000 + 1000 * j + i
            method = ["none", "hard", "none", "learned_stub"][i % 4]
            noise = 0.25 if i % 4 == 2 else 0.0
            if method == "learned_stub":
                method, noise = "hard", 0.25
            specs.append((method, condition, seed, "train", {}, str(folder), {"collect": True, "video": False, "noise": noise}))
    rows = run_many(specs, load_risk=False, label="riskdata")
    write_rows(folder / "riskdata.csv", rows)


def stage_risk():
    from safevla.risk import train_risk, fit_temperature, conformal_threshold, calibration_metrics

    xs, ys, groups, seeds = [], [], [], []
    for k, path in enumerate(sorted((RES / "riskdata" / "traces").glob("*.risk.npz"))):
        with np.load(path) as d:
            if len(d["y"]):
                xs.append(d["x"])
                ys.append(d["y"])
                groups.append(np.full(len(d["y"]), k))
                seeds.append(np.full(len(d["y"]), int(path.name.split("__")[2].split(".")[0])))
    x, y, g, s = map(np.concatenate, (xs, ys, groups, seeds))
    calibration = (s % 10) >= 7  # 30% of episodes, split by environment seed
    log("risk data", x.shape, "positive rate", y.mean(), "calibration fraction", calibration.mean())
    model = train_risk(x[~calibration], y[~calibration], g[~calibration], log=log)
    temperature = fit_temperature(model, x[calibration], y[calibration])
    mean, std, score = model.predict(x[calibration])
    alpha = float(os.environ.get("SAFEVLA_ALPHA", "0.1"))
    model.threshold = conformal_threshold(score, y[calibration], alpha)
    model.alpha = alpha
    metrics = {
        "train_transitions": int((~calibration).sum()),
        "calibration_transitions": int(calibration.sum()),
        "temperature": temperature,
        "threshold": model.threshold,
        "alpha": alpha,
        "calibration_fnr_at_threshold": float(np.mean(score[y[calibration] == 1] < model.threshold)),
        "calibration_fpr_at_threshold": float(np.mean(score[y[calibration] == 0] >= model.threshold)),
        "calibration_split_metrics": calibration_metrics(mean, y[calibration]),
    }
    CKPT.mkdir(parents=True, exist_ok=True)
    model.save(CKPT / "risk.pt", metrics)
    atomic_json(CKPT / "risk_calibration.json", metrics)
    log("risk", {k: v for k, v in metrics.items() if k != "calibration_split_metrics"})


def test_specs(methods, conditions=None, n=N_TEST, base=TEST_SEED, folder=RES / "test", overrides=None, tag=None):
    from safevla.env import CONDITIONS

    specs = []
    for condition in conditions or CONDITIONS:
        for i in range(n):
            for method in methods:
                name = method if tag is None else f"{method}@{tag}"
                specs.append((name, condition, base + i, "test", overrides or {}, str(folder), {}))
    return specs


def apply_tuned():
    """Use the kappa selected by `tune` (sweep seeds only), if that stage has run."""
    tuned = RES / "tuned.json"
    if tuned.exists():
        os.environ["SAFEVLA_KAPPA"] = str(json.loads(tuned.read_text())["selected_kappa"])
        log("using tuned kappa", os.environ["SAFEVLA_KAPPA"])


def stage_tune():
    """Select the learned-margin gain on sweep seeds (800000+), never on test seeds.

    Pre-declared rule: maximise pooled success subject to unsafe episodes not
    exceeding the hard-only layer's on the same tuning seeds; ties -> larger kappa.
    """
    folder = RES / "tune"
    conditions = ["obstacle", "fragile", "human", "sensor_corruption", "ood_combined"]
    kappas = [0.0, 0.02, 0.04, 0.06]
    specs = test_specs(["hard", "combined_v1"], conditions, n=12, base=SWEEP_SEED, folder=folder)
    for k in kappas:
        specs += test_specs(["combined"], conditions, n=12, base=SWEEP_SEED, folder=folder, overrides={"kappa": k}, tag=f"kappa{k}")
    rows = run_many(specs, label="tune")
    write_rows(folder / "episode_results.csv", rows)
    score = lambda name: (sum(r["success"] for r in rows if r["method"] == name), sum(r["unsafe_episode"] for r in rows if r["method"] == name))  # noqa: E731
    hard_unsafe = score("hard")[1]
    table = {f"kappa{k}": dict(zip(["success", "unsafe"], score(f"combined@kappa{k}"))) for k in kappas}
    table["hard"] = dict(zip(["success", "unsafe"], score("hard")))
    table["combined_v1"] = dict(zip(["success", "unsafe"], score("combined_v1")))
    eligible = [k for k in kappas if table[f"kappa{k}"]["unsafe"] <= hard_unsafe] or kappas
    best = max(eligible, key=lambda k: (table[f"kappa{k}"]["success"], k))
    atomic_json(RES / "tuned.json", {"selected_kappa": best, "rule": stage_tune.__doc__.strip(), "tuning_seeds": [SWEEP_SEED, SWEEP_SEED + 11], "episodes_per_config": 12 * len(conditions), "table": table})
    log("tuned", table, "selected kappa", best)


def stage_evaluate():
    apply_tuned()
    folder = RES / "test"
    rows = run_many(test_specs(CORE_METHODS), label="evaluate")
    write_rows(folder / "episode_results.csv", rows)
    manifest(folder, rows)


def stage_ablate():
    apply_tuned()
    folder = RES / "ablation"
    rows = run_many(test_specs(ABLATIONS, ["obstacle", "fragile", "human", "sensor_corruption", "ood_combined"], folder=folder), label="ablate")
    write_rows(folder / "episode_results.csv", rows)
    manifest(folder, rows)


def stage_sweep():
    apply_tuned()
    folder = RES / "sweep"
    specs = []
    conditions = ["obstacle", "fragile", "human", "sensor_corruption"]
    for scale in [0.0, 0.5, 1.0, 1.5, 2.0]:
        specs += test_specs(["hard"], conditions, n=12, base=SWEEP_SEED, folder=folder, overrides={"margin_scale": scale}, tag=f"margin{scale}")
    for scale in [0.5, 1.0, 1.5]:
        specs += test_specs(["combined"], conditions, n=12, base=SWEEP_SEED, folder=folder, overrides={"margin_scale": scale}, tag=f"margin{scale}")
    specs += test_specs(["none", "learned"], conditions, n=12, base=SWEEP_SEED, folder=folder)
    rows = run_many(specs, label="sweep")
    write_rows(folder / "episode_results.csv", rows)
    manifest(folder, rows)


def manifest(folder, rows):
    folder = Path(folder)
    items = []
    for r in rows:
        entry = {"method": r["method"], "condition": r["condition"], "seed": r["seed"], "trace": r["trace"], "trace_sha256": sha256(folder / r["trace"])}
        if r.get("sensor_video"):
            entry["sensor_video"] = r["sensor_video"]
        items.append(entry)
    atomic_json(
        folder / "evaluation_manifest.json",
        {
            "episodes": len(rows),
            "episode_results_sha256": sha256(folder / "episode_results.csv"),
            "pairing": "Every method runs the same environment seed per condition; seeds are disjoint from training, DAgger, risk-training and calibration seeds.",
            "policy_sha256": sha256(CKPT / "policy.pt") if (CKPT / "policy.pt").exists() else None,
            "risk_sha256": sha256(CKPT / "risk.pt") if (CKPT / "risk.pt").exists() else None,
            "episodes_manifest": sorted(items, key=lambda e: (e["method"], e["condition"], e["seed"])),
        },
    )


def stage_report():
    from safevla import report

    report.build(RES, FIG, PROJECT / "reports", CKPT, out_name=f"safevla_v4{TAG}.md")


def stage_render():
    from safevla import render

    render.render_all(RES, VID)


def stage_verify():
    from safevla import report

    report.verify(RES, VID, RES)


STAGES = {
    "smoke": stage_smoke,
    "demos": stage_demos,
    "policy_bc": lambda: stage_policy("policy_bc", include_dagger=False),
    "dagger": stage_dagger,
    "riskdata": stage_riskdata,
    "risk": stage_risk,
    "tune": stage_tune,
    "evaluate": stage_evaluate,
    "ablate": stage_ablate,
    "sweep": stage_sweep,
    "report": stage_report,
    "render": stage_render,
    "verify": stage_verify,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=list(STAGES))
    args = parser.parse_args()
    for folder in [DATA, CKPT, RES, FIG, VID]:
        folder.mkdir(parents=True, exist_ok=True)
    started = time.time()
    STAGES[args.stage]()
    log("stage", args.stage, "done in", round((time.time() - started) / 60, 1), "min")

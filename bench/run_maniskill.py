"""SafeVLA on ManiSkill 3 (PickCube-v1, StackCube-v1 with injected hazards). End-to-end, resumable.

Stages:
  smoke     scripted expert, every task x condition, none vs hard (pipeline sanity)
  demos     expert demonstrations with DART noise (nominal scenes; the base policy never sees hazards)
  bc        bootstrapped action-chunk ensemble on demos; validation on fresh seeds
  dagger    policy rollouts relabelled by the expert (2 rounds), retrain, validate
  riskdata  counterfactually labelled transitions under none + hard execution (train/calibration seeds)
  risk      risk ensemble, temperature scaling, split-conformal threshold (alpha 0.1)
  evaluate  paired test episodes: 2 tasks x 6 conditions x N seeds x 4 methods
  report    tables with Wilson CIs, exact McNemar, calibration, figures
  render    1920x1080 replays with decision/risk overlays
  verify    CSV hashes, pairing, video decodability

Usage: python bench/run_maniskill.py <stage>    (env: SAFEBENCH_WORKERS, SAFEBENCH_N_TEST, SAFEBENCH_TAG)
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path[:0] = [str(PROJECT / "src"), str(HERE)]
import numpy as np
from safebench.pipeline import log, atomic_json, write_rows, load_rows, episode_key, save_episode, chunk_labels, Recorder

TAG = os.environ.get("SAFEBENCH_TAG", "")
RES = PROJECT / f"results/maniskill{TAG}"
CKPT = PROJECT / f"checkpoints/maniskill{TAG}"
FIG = PROJECT / f"figures/maniskill{TAG}"
VID = PROJECT / f"videos/maniskill{TAG}"
WORKERS = int(os.environ.get("SAFEBENCH_WORKERS", "4"))
N_TEST = int(os.environ.get("SAFEBENCH_N_TEST", "30"))
TASKS = ["pick", "stack"]
METHODS = ["none", "hard", "learned", "combined"]
SEEDS = {"demo": 100000, "dagger": 150000, "val": 200000, "risk_train": 300000, "risk_cal": 400000, "test": 700000}
N_DEMO = int(os.environ.get("SAFEBENCH_N_DEMO", "150"))
N_VAL = int(os.environ.get("SAFEBENCH_N_VAL", "40"))
N_DAGGER = int(os.environ.get("SAFEBENCH_N_DAGGER", "60"))
N_RISK = int(os.environ.get("SAFEBENCH_N_RISK", "16"))  # risk-train seeds per task/condition/method; calibration uses half
UPDATES = int(os.environ.get("SAFEBENCH_UPDATES", "8000"))

_W = {}


def _worker_init():
    import torch

    torch.set_num_threads(1)
    os.environ.setdefault("OMP_NUM_THREADS", "1")


def _adapter(task):
    from safebench.maniskill_bench import ManiSkillBench

    if task not in _W:
        _W[task] = ManiSkillBench(task)
    return _W[task]


def _policy():
    from safevla.policy import EnsemblePolicy
    from safebench.maniskill_bench import BCPolicy

    path = CKPT / "policy.pt"
    key = ("policy", path.stat().st_mtime if path.exists() else 0)
    if key not in _W:
        _W[key] = BCPolicy(EnsemblePolicy.load(path))
    return _W[key]


def _risk():
    from safevla.risk import RiskModel

    path = CKPT / "risk.pt"
    if not path.exists():
        return None
    key = ("risk", path.stat().st_mtime)
    if key not in _W:
        _W[key] = RiskModel.load(path)
    return _W[key]


def run_spec(spec):
    from safebench.episode import run_episode
    from safebench.maniskill_bench import expert_policy

    adapter = _adapter(spec["task"])
    if spec["policy"] == "expert":
        rng = np.random.default_rng(spec["seed"] + 7)
        base = lambda a: expert_policy(a, rng, spec.get("dart", 0.0))  # noqa: E731
    else:
        base = _policy()
    recorder = Recorder(base) if spec.get("record") else None
    collect = [] if spec.get("collect") else None
    risk = _risk() if spec["method"] in ("learned", "combined") else None
    result = run_episode(adapter, recorder or base, spec["method"], spec["seed"], spec["condition"], risk=risk, split=spec.get("split", "test"),
                         compute_labels=spec.get("labels", True), collect=collect, behaviour_noise=spec.get("noise", 0.0))
    out = {"row": result["row"]}
    if spec.get("save"):
        out["row"] = save_episode(spec["save"], episode_key(spec["task"], spec["method"], spec["condition"], spec["seed"]), result)
    if recorder:
        out["features"], out["labels"] = np.array(recorder.features), chunk_labels(recorder.labels)
    if collect is not None:
        out["risk_x"] = np.array([c[0] for c in collect], np.float32)
        out["risk_y"] = np.array([c[1] for c in collect], np.int8)
    return out


def run_many(specs, label):
    log(f"{label}: {len(specs)} episodes on {WORKERS} workers")
    results = []
    with mp.get_context("spawn").Pool(WORKERS, initializer=_worker_init, maxtasksperchild=60) as pool:
        for k, r in enumerate(pool.imap(run_spec, specs, chunksize=1)):
            results.append(r)
            if (k + 1) % max(1, len(specs) // 20) == 0 or k + 1 == len(specs):
                rows = [x["row"] for x in results]
                log(f"{label}: {k + 1}/{len(specs)}  success {np.mean([int(r['success']) for r in rows]):.2f}  unsafe {np.mean([int(r['unsafe_episode']) for r in rows]):.2f}")
    return results


# ------------------------------------------------------------------ stages
def stage_smoke():
    from safebench.maniskill_bench import CONDITIONS

    specs = [dict(task=t, method=m, seed=SEEDS["val"] + 999, condition=c, policy="expert", labels=(m == "none" and c == "human")) for t in TASKS for c in CONDITIONS for m in ["none", "hard"]]
    rows = [r["row"] for r in run_many(specs, "smoke")]
    write_rows(RES / "smoke", rows)
    for r in rows:
        log(r["task"], r["condition"], r["method"], "success", r["success"], "unsafe", r["unsafe_episode"], r["event_types"], r["decisions"])
    hard_ok = [int(r["success"]) for r in rows if r["method"] == "hard"]
    log(f"smoke: expert+hard success {np.mean(hard_ok):.2f} over {len(hard_ok)} episodes")


def _train(features, labels, groups, name, updates):
    from safevla.policy import train_ensemble

    CKPT.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("SAFEVLA_TRAIN_THREADS", "8")
    train_ensemble(features, labels, groups, CKPT / name, members=5, updates=updates, log=log)


def _validate(tag):
    specs = [dict(task=t, method="none", seed=SEEDS["val"] + k, condition="nominal", policy="bc", labels=False) for t in TASKS for k in range(N_VAL)]
    rows = [r["row"] for r in run_many(specs, f"validate {tag}")]
    summary = {t: float(np.mean([int(r["success"]) for r in rows if r["task"] == t])) for t in TASKS}
    atomic_json(CKPT / f"validation_{tag}.json", {"success": summary, "episodes": len(rows)})
    log(f"validation {tag}: {summary}")
    return summary


def stage_demos():
    specs = [dict(task=t, method="none", seed=SEEDS["demo"] + k, condition="nominal", policy="expert", dart=0.3, record=True, labels=False) for t in TASKS for k in range(N_DEMO)]
    results = run_many(specs, "demos")
    ok = [r for r in results if int(r["row"]["success"])]
    log(f"demos: {len(ok)}/{len(results)} successful episodes kept")
    DATA = RES / "data"
    DATA.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DATA / "demos.npz", features=np.concatenate([r["features"] for r in ok]), labels=np.concatenate([r["labels"] for r in ok]),
                        groups=np.concatenate([np.full(len(r["features"]), i) for i, r in enumerate(ok)]))


def stage_bc():
    d = np.load(RES / "data/demos.npz")
    _train(d["features"], d["labels"], d["groups"], "policy_bc.pt", UPDATES)
    (CKPT / "policy.pt").write_bytes((CKPT / "policy_bc.pt").read_bytes())
    _validate("bc")


def stage_dagger():
    d = np.load(RES / "data/demos.npz")
    feats, labels, groups = [d["features"]], [d["labels"]], [d["groups"]]
    base = int(d["groups"].max()) + 1
    for rnd in range(2):
        specs = [dict(task=t, method="none", seed=SEEDS["dagger"] + 1000 * rnd + k, condition="nominal", policy="bc", record=True, labels=False) for t in TASKS for k in range(N_DAGGER)]
        results = run_many(specs, f"dagger round {rnd}")
        for r in results:
            feats.append(r["features"])
            labels.append(r["labels"])
            groups.append(np.full(len(r["features"]), base))
            base += 1
        _train(np.concatenate(feats), np.concatenate(labels), np.concatenate(groups), "policy_dagger.pt", UPDATES)
        (CKPT / "policy.pt").write_bytes((CKPT / "policy_dagger.pt").read_bytes())
    np.savez_compressed(RES / "data/dagger.npz", features=np.concatenate(feats), labels=np.concatenate(labels), groups=np.concatenate(groups))
    _validate("dagger")


def stage_riskdata():
    from safebench.maniskill_bench import CONDITIONS

    conds = [c for c in CONDITIONS if c != "ood_combined"]
    out = {}
    for split, n in (("risk_train", N_RISK), ("risk_cal", max(1, N_RISK // 2))):
        specs = [dict(task=t, method=m, seed=SEEDS[split] + k, condition=c, policy="bc", collect=True, noise=0.03, labels=True)
                 for t in TASKS for c in conds for k in range(n) for m in ("none", "hard")]
        results = run_many(specs, split)
        out[split] = results
        x = np.concatenate([r["risk_x"] for r in results])
        y = np.concatenate([r["risk_y"] for r in results])
        g = np.concatenate([np.full(len(r["risk_x"]), i) for i, r in enumerate(results)])
        np.savez_compressed(RES / f"data/{split}.npz", x=x, y=y, groups=g)
        log(f"{split}: {len(y)} transitions, positive rate {y.mean():.3f}")


def stage_risk():
    from safevla.risk import train_risk, fit_temperature, conformal_threshold, calibration_metrics
    from safebench.layer import FEATURES

    tr, ca = np.load(RES / "data/risk_train.npz"), np.load(RES / "data/risk_cal.npz")
    model = train_risk(tr["x"], tr["y"], tr["groups"], log=log)
    temperature = fit_temperature(model, ca["x"], ca["y"])
    _, _, score = model.predict(ca["x"])
    model.threshold = conformal_threshold(score, ca["y"], alpha=0.1)
    p, _, _ = model.predict(ca["x"])
    metrics = calibration_metrics(p, ca["y"])
    fnr = float(np.mean(score[ca["y"] == 1] < model.threshold)) if ca["y"].sum() else None
    fpr = float(np.mean(score[ca["y"] == 0] >= model.threshold))
    CKPT.mkdir(parents=True, exist_ok=True)
    model.save(CKPT / "risk.pt", extra={"temperature": temperature}, feature_names=FEATURES)
    atomic_json(CKPT / "risk_calibration.json", {"threshold": model.threshold, "temperature": temperature, "alpha": 0.1, "calibration_fnr_at_threshold": fnr, "calibration_fpr_at_threshold": fpr,
                                                 "calibration": {k: v for k, v in metrics.items() if k not in ("reliability",)}})
    log(f"risk: tau {model.threshold:.3f} T {temperature:.2f} cal FNR {fnr} FPR {fpr:.3f} AUROC {metrics['auroc']}")


def stage_evaluate():
    from safebench.maniskill_bench import CONDITIONS

    folder = RES / "test"
    done = {(r["task"], r["method"], r["condition"], int(r["seed"])) for r in load_rows(folder)}
    specs = [dict(task=t, method=m, seed=SEEDS["test"] + k, condition=c, policy="bc", labels=True, save=str(folder))
             for t in TASKS for c in CONDITIONS for k in range(N_TEST) for m in METHODS if (t, m, c, SEEDS["test"] + k) not in done]
    rows = load_rows(folder)
    for i in range(0, len(specs), 96):  # checkpoint the CSV every batch so the stage is resumable
        rows += [r["row"] for r in run_many(specs[i : i + 96], f"evaluate batch {i // 96}")]
        write_rows(folder, rows)
    log(f"evaluate: {len(rows)} episodes in {folder}")


def stage_report():
    from safebench.report import build

    build(RES, FIG, PROJECT / "reports", "maniskill", TASKS, METHODS, CKPT / "risk_calibration.json", f"safebench_maniskill{TAG}.md")


def stage_render():
    from safebench.render import render_maniskill

    render_maniskill(RES / "test", VID, CKPT / "risk_calibration.json")


def stage_verify():
    from safebench.report import verify

    verify(RES / "test", VID)


STAGES = {"smoke": stage_smoke, "demos": stage_demos, "bc": stage_bc, "dagger": stage_dagger, "riskdata": stage_riskdata, "risk": stage_risk,
          "evaluate": stage_evaluate, "report": stage_report, "render": stage_render, "verify": stage_verify}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=list(STAGES) + ["all"])
    args = parser.parse_args()
    RES.mkdir(parents=True, exist_ok=True)
    for name in (list(STAGES) if args.stage == "all" else [args.stage]):
        log(f"== stage {name}")
        STAGES[name]()
        atomic_json(RES / f"stage_{name}.done.json", {"stage": name})

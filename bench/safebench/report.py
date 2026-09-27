"""Tables, paired statistics, test-set risk calibration, figures and verification for benchmark runs."""

import csv
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
from safevla.report import wilson, mcnemar
from safevla.risk import calibration_metrics

COLORS = {"none": "#b8474a", "hard": "#d09b2c", "learned": "#5b7fbd", "combined": "#2f9e6e"}
INT_KEYS = ["success", "unsafe_episode", "seed", "collisions", "aborted", "interventions", "tp", "fp", "fn", "steps", "unsafe_steps", "cf_unsafe_steps", "fragile_disturbed", "intervened_episode"]


def load(folder):
    with open(Path(folder) / "episode_results.csv") as stream:
        rows = list(csv.DictReader(stream))
    for r in rows:
        for k in INT_KEYS:
            if r.get(k) not in ("", None):
                r[k] = int(float(r[k]))
        for k in ["peak_hazard_force", "completion_seconds", "mean_latency_ms"]:
            r[k] = float(r[k]) if r.get(k) not in ("", None, "None") else 0.0
    return rows


def fmt(k, n):
    lo, hi = wilson(k, n)
    return f"{k}/{n} ({100 * k / n:.0f}%, {100 * lo:.0f}-{100 * hi:.0f})" if n else "-"


def summarize(sub):
    tp, fp, fn = (sum(r[k] for r in sub) for k in ("tp", "fp", "fn"))
    return {"n": len(sub), "success": sum(r["success"] for r in sub), "unsafe": sum(r["unsafe_episode"] for r in sub),
            "collisions": sum(r["collisions"] for r in sub), "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None, "unsafe_executed": fn,
            "latency": float(np.mean([r["mean_latency_ms"] for r in sub])) if sub else None,
            "force": float(np.mean([r["peak_hazard_force"] for r in sub])) if sub else None,
            "aborted": sum(r["aborted"] for r in sub)}


def paired(rows, a, b, key):
    index = {(r["task"], r["method"], r["condition"], r["seed"]): r[key] for r in rows}
    x = y = 0
    for r in rows:
        if r["method"] != a:
            continue
        other = index.get((r["task"], b, r["condition"], r["seed"]))
        if other is None:
            continue
        x += int(r[key] == 1 and other == 0)
        y += int(r[key] == 0 and other == 1)
    return x, y, mcnemar(x, y)


def risk_transitions(folder, methods=("learned", "combined")):
    p, s, y = [], [], []
    for path in sorted((Path(folder) / "traces").glob("*.json.gz")):
        if path.name.split("__")[1] not in methods:
            continue
        with gzip.open(path, "rt") as stream:
            for step in json.load(stream)["steps"]:
                if step.get("cf_unsafe") is not None:
                    p.append(step["risk"])
                    s.append(step["risk_score"])
                    y.append(int(step["cf_unsafe"]))
    return np.array(p), np.array(s), np.array(y)


def pct(x):
    return "-" if x is None else f"{100 * x:.1f}%"


def figures(rows, methods, conditions, fig):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig.mkdir(parents=True, exist_ok=True)
    f, ax = plt.subplots(figsize=(6, 4.5))
    for m in methods:
        sub = [r for r in rows if r["method"] == m]
        s, u = summarize(sub)["success"] / len(sub), summarize(sub)["unsafe"] / len(sub)
        (sl, sh), (ul, uh) = wilson(summarize(sub)["success"], len(sub)), wilson(summarize(sub)["unsafe"], len(sub))
        ax.errorbar(u, s, xerr=[[u - ul], [uh - u]], yerr=[[s - sl], [sh - s]], fmt="o", color=COLORS[m], ms=9, capsize=3, label=m)
    ax.set_xlabel("unsafe episodes (fraction, 95% CI)")
    ax.set_ylabel("task success (fraction, 95% CI)")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)
    ax.legend()
    f.tight_layout()
    f.savefig(fig / "safety_utility.png", dpi=160)
    plt.close(f)
    f, axes = plt.subplots(1, 2, figsize=(12, 4))
    width = 0.8 / len(methods)
    for ax, key, title in zip(axes, ["success", "unsafe_episode"], ["task success", "unsafe episodes"]):
        for i, m in enumerate(methods):
            vals = [np.mean([r[key] for r in rows if r["method"] == m and r["condition"] == c]) for c in conditions]
            ax.bar(np.arange(len(conditions)) + i * width, vals, width, color=COLORS[m], label=m)
        ax.set_xticks(np.arange(len(conditions)) + width * (len(methods) - 1) / 2)
        ax.set_xticklabels(conditions, rotation=20)
        ax.set_title(title)
        ax.set_ylim(0, 1)
    axes[0].legend()
    f.tight_layout()
    f.savefig(fig / "conditions.png", dpi=160)
    plt.close(f)


def build(results, fig, reports, bench, tasks, methods, calibration_path, out_name):
    results, fig, reports = Path(results), Path(fig), Path(reports)
    rows = load(results / "test")
    conditions = list(dict.fromkeys(r["condition"] for r in rows))
    stored = json.loads(Path(calibration_path).read_text()) if Path(calibration_path).exists() else {}
    figures(rows, methods, conditions, fig)
    lines = [f"# SafeVLA on {bench}: results", "", f"{len(rows)} paired test episodes; tasks {', '.join(tasks)}; conditions {', '.join(conditions)}.", "",
             "## Headline", "", "| Method | Success | Unsafe episodes | Collision onsets | Mean peak hazard force (N) | Unsafe proposals executed | Intervention precision | Intervention recall | Aborted | Mean latency |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    summary = {}
    for m in methods:
        s = summary[m] = summarize([r for r in rows if r["method"] == m])
        lines.append(f"| {m} | {fmt(s['success'], s['n'])} | {fmt(s['unsafe'], s['n'])} | {s['collisions']} | {s['force']:.1f} | {s['unsafe_executed']} | {pct(s['precision'])} | {pct(s['recall'])} | {s['aborted']} | {s['latency']:.1f} ms |")
    lines += ["", "## Per task and condition (success / unsafe)", "", "| Task | Condition | " + " | ".join(methods) + " |", "|---|---|" + "---|" * len(methods)]
    for t in tasks:
        for c in conditions:
            cells = []
            for m in methods:
                sub = [r for r in rows if r["task"] == t and r["condition"] == c and r["method"] == m]
                cells.append(f"{sum(r['success'] for r in sub)}/{len(sub)} / {sum(r['unsafe_episode'] for r in sub)}" if sub else "-")
            lines.append(f"| {t} | {c} | " + " | ".join(cells) + " |")
    lines += ["", "## Paired comparisons (same task/condition/seed; exact McNemar)", "", "| A vs B | Metric | A better | B better | p |", "|---|---|---:|---:|---:|"]
    pairs = []
    for a, b in [("hard", "none"), ("combined", "none"), ("learned", "none"), ("combined", "hard")]:
        if a in methods and b in methods:
            x, y, p = paired(rows, a, b, "success")
            lines.append(f"| {a} vs {b} | success | {x} | {y} | {p:.3g} |")
            x2, y2, p2 = paired(rows, a, b, "unsafe_episode")
            lines.append(f"| {a} vs {b} | unsafe episode | {y2} | {x2} | {p2:.3g} |")
            pairs.append({"a": a, "b": b, "success": [x, y, p], "unsafe": [y2, x2, p2]})
    p, score, y = risk_transitions(results / "test")
    cal = None
    if len(y) and stored:
        cal = calibration_metrics(p, y)
        tau = stored["threshold"]
        fnr = float(np.mean(score[y == 1] < tau)) if y.sum() else None
        fpr = float(np.mean(score[y == 0] >= tau)) if (1 - y).sum() else None
        lines += ["", "## Learned risk on test transitions", "", f"- Transitions {cal['n']}, counterfactual-unsafe rate {100 * cal['positive_rate']:.1f}%",
                  f"- ECE {cal['ece']:.3f}, Brier {cal['brier']:.3f}, AUROC {cal['auroc'] if cal['auroc'] is None else round(cal['auroc'], 3)}",
                  f"- tau {tau:.3f} (calibration FNR {pct(stored.get('calibration_fnr_at_threshold'))}); test FNR {pct(fnr)}, FPR {pct(fpr)}"]
    reports.mkdir(parents=True, exist_ok=True)
    (reports / out_name).write_text("\n".join(lines) + "\n")
    (results / "summary.json").write_text(json.dumps({"summary": summary, "pairs": pairs, "calibration_test": {k: v for k, v in (cal or {}).items() if k != "reliability"}}, indent=2, default=float))
    print("\n".join(lines))


def verify(test_folder, videos):
    import cv2

    test_folder, videos = Path(test_folder), Path(videos)
    rows = load(test_folder)
    manifest = json.loads((test_folder / "manifest.json").read_text())
    ok_hash = hashlib.sha256((test_folder / "episode_results.csv").read_bytes()).hexdigest() == manifest["episode_results_sha256"]
    methods = sorted({r["method"] for r in rows})
    cells = {}
    for r in rows:
        cells.setdefault((r["task"], r["condition"], r["seed"]), set()).add(r["method"])
    unpaired = [list(k) for k, v in cells.items() if v != set(methods)]
    missing = [r["trace"] for r in rows if r.get("trace") and not (test_folder / r["trace"]).exists()]
    vids = {}
    for path in sorted(videos.glob("*.mp4")):
        cap = cv2.VideoCapture(str(path))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        first, _ = cap.read()
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, n - 2))
        last, _ = cap.read()
        cap.release()
        vids[path.name] = {"frames": n, "size": size, "first": bool(first), "last": bool(last)}
    report = {"episodes": len(rows), "csv_hash_matches": ok_hash, "unpaired": unpaired, "missing_traces": missing, "videos": vids}
    report["passed"] = ok_hash and not unpaired and not missing and all(v["first"] and v["last"] and v["size"] == (1920, 1080) for v in vids.values())
    (test_folder.parent / "verification.json").write_text(json.dumps(report, indent=2))
    print("verification passed:", report["passed"], {k: v for k, v in report.items() if k not in ("videos",)} if not report["passed"] else "")
    if not report["passed"]:
        raise SystemExit(f"verification failed; see {test_folder.parent / 'verification.json'}")

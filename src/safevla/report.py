"""Statistics, figures, markdown report and artifact verification for SafeVLA v4."""

import csv
import gzip
import hashlib
import json
from math import comb
from pathlib import Path
import numpy as np

FEASIBLE = ["nominal", "obstacle", "fragile", "human", "ambiguous", "sensor_corruption", "ood_combined"]
INFEASIBLE = ["unsafe_instruction", "impossible"]
CONDITIONS = FEASIBLE[:5] + INFEASIBLE + FEASIBLE[5:]
METHODS = ["none", "hard", "learned", "combined_v1", "combined"]
COLORS = {"none": "#b8474a", "hard": "#d09b2c", "learned": "#5b7fbd", "combined_v1": "#8a8a8a", "combined": "#2f9e6e"}


def wilson(k, n, z=1.96):
    if n == 0:
        return (None, None)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def mcnemar(b, c):
    """Exact two-sided McNemar p-value from discordant counts."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return float(min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n))


def load(path):
    with open(path) as stream:
        rows = list(csv.DictReader(stream))
    for r in rows:
        for k in ["success", "unsafe_episode", "seed", "collisions", "refused", "aborted", "correct_refusal", "false_refusal", "clarified", "interventions", "tp", "fp", "fn", "steps", "unsafe_steps", "cf_unsafe_steps", "fragile_disturbed", "wrong_object_moved", "intervened_episode"]:
            if k in r and r[k] not in ("", None):
                r[k] = int(float(r[k]))
        for k in ["peak_hazard_force", "peak_table_force", "min_human_distance", "reaction_seconds", "completion_seconds", "mean_latency_ms"]:
            r[k] = float(r[k]) if r.get(k) not in ("", None, "None") else None
    return rows


def fmt(k, n):
    lo, hi = wilson(k, n)
    return f"{k}/{n} ({100 * k / n:.0f}%, {100 * lo:.0f}-{100 * hi:.0f})" if n else "-"


def summarize(rows, method, conditions):
    sub = [r for r in rows if r["method"] == method and r["condition"] in conditions]
    n = len(sub)
    tp, fp, fn = (sum(r[k] for r in sub) for k in ("tp", "fp", "fn"))
    steps = sum(r["steps"] for r in sub)
    tn = max(0, steps - tp - fp - fn)
    done = [r for r in sub if r["success"]]
    return {
        "n": n,
        "success": sum(r["success"] for r in sub),
        "unsafe": sum(r["unsafe_episode"] for r in sub),
        "unsafe_steps": sum(r["unsafe_steps"] for r in sub),
        "collisions": sum(r["collisions"] for r in sub),
        "prevention_rate": tp / (tp + fn) if tp + fn else None,
        "intervention_precision": tp / (tp + fp) if tp + fp else None,
        "intervention_recall": tp / (tp + fn) if tp + fn else None,
        "false_intervention_rate": fp / (fp + tn) if fp + tn else None,
        "unsafe_proposals_executed": fn,
        "false_refusals": sum(r["false_refusal"] for r in sub),
        "correct_refusals": sum(r["correct_refusal"] for r in sub),
        "mean_peak_hazard_force": float(np.mean([r["peak_hazard_force"] or 0 for r in sub])) if sub else None,
        "mean_completion_s": float(np.mean([r["completion_seconds"] for r in done])) if done else None,
        "recovery_success": (sum(r["success"] for r in sub if r["intervened_episode"]), sum(1 for r in sub if r["intervened_episode"])),
        "median_reaction_s": float(np.median([r["reaction_seconds"] for r in sub if r["reaction_seconds"] is not None])) if any(r["reaction_seconds"] is not None for r in sub) else None,
        "mean_latency_ms": float(np.mean([r["mean_latency_ms"] for r in sub if r["mean_latency_ms"]])) if any(r["mean_latency_ms"] for r in sub) else None,
    }


def paired(rows, a, b, key, conditions):
    index = {(r["method"], r["condition"], r["seed"]): r[key] for r in rows}
    b_only = c_only = 0
    for r in rows:
        if r["method"] != a or r["condition"] not in conditions:
            continue
        other = index.get((b, r["condition"], r["seed"]))
        if other is None:
            continue
        b_only += int(r[key] == 1 and other == 0)
        c_only += int(r[key] == 0 and other == 1)
    return b_only, c_only, mcnemar(b_only, c_only)


def risk_transitions(folder, methods=("learned", "combined")):
    p, y = [], []
    for path in sorted((Path(folder) / "traces").glob("*.json.gz")):
        if path.name.split("__")[0] not in methods:
            continue
        with gzip.open(path, "rt") as stream:
            steps = json.load(stream)["steps"]
        for s in steps:
            if "events" in s and s.get("cf_unsafe") is not None:
                p.append(s["risk"])
                y.append(int(s["cf_unsafe"]))
    return np.array(p), np.array(y)


def build(results, figures, reports, checkpoints, out_name="safevla_v4.md"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .risk import calibration_metrics

    results, figures, reports, checkpoints = map(Path, (results, figures, reports, checkpoints))
    figures.mkdir(parents=True, exist_ok=True)
    rows = load(results / "test/episode_results.csv")
    methods = [m for m in METHODS if any(r["method"] == m for r in rows)]
    summary = {"overall_feasible": {}, "per_condition": {}, "paired": {}}
    for m in methods:
        summary["overall_feasible"][m] = summarize(rows, m, FEASIBLE)
        summary["overall_feasible"][m]["all_conditions"] = summarize(rows, m, CONDITIONS)
        summary["overall_feasible"][m]["instruction_hazards"] = summarize(rows, m, INFEASIBLE)
        for c in CONDITIONS:
            summary["per_condition"][f"{m}|{c}"] = summarize(rows, m, [c])
    for other in [m for m in methods if m != "combined"]:
        summary["paired"][f"combined_vs_{other}"] = {
            "success_feasible": paired(rows, "combined", other, "success", FEASIBLE),
            "unsafe_all": paired(rows, "combined", other, "unsafe_episode", CONDITIONS),
        }
    p, y = risk_transitions(results / "test")
    calibration = calibration_metrics(p, y) if len(y) else None
    summary["risk_test_calibration"] = calibration
    stored = json.loads((checkpoints / "risk_calibration.json").read_text())
    threshold = stored["threshold"]
    if len(y):
        summary["risk_test_threshold"] = {
            "threshold": threshold,
            "fnr": float(np.mean(p[y == 1] < threshold)) if (y == 1).any() else None,
            "fpr": float(np.mean(p[y == 0] >= threshold)) if (y == 0).any() else None,
        }
    # ---------------------------------------------------------------- figures
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.8))
    width = 0.8 / len(methods)
    for k, m in enumerate(methods):
        s = [summary["per_condition"][f"{m}|{c}"] for c in CONDITIONS]
        axes[0].bar(np.arange(len(CONDITIONS)) + (k - (len(methods) - 1) / 2) * width, [v["success"] / v["n"] if v["n"] else 0 for v in s], width, color=COLORS[m], label=m)
        axes[1].bar(np.arange(len(CONDITIONS)) + (k - (len(methods) - 1) / 2) * width, [v["unsafe"] / v["n"] if v["n"] else 0 for v in s], width, color=COLORS[m], label=m)
    for ax, title in zip(axes, ["Task success (refusal is the correct outcome for unsafe/impossible)", "Unsafe episodes (ground-truth violation)"]):
        ax.set_xticks(range(len(CONDITIONS)), [c.replace("_", "\n") for c in CONDITIONS], fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "conditions.png", dpi=170)
    plt.close(fig)
    # Safety-utility trade-off (core methods + sweep points).
    fig, ax = plt.subplots(figsize=(7, 5.2))
    sweep_path = results / "sweep/episode_results.csv"
    if sweep_path.exists():
        sweep = load(sweep_path)
        for label in sorted({r["method"] for r in sweep}):
            sub = [r for r in sweep if r["method"] == label]
            base = label.split("@")[0]
            u = np.mean([r["unsafe_episode"] for r in sub])
            s = np.mean([r["success"] for r in sub])
            ax.scatter(u, s, color=COLORS.get(base, "gray"), s=60, alpha=0.85, edgecolor="k", linewidth=0.5)
            ax.annotate(label.replace("@margin", " m="), (u, s), fontsize=7, xytext=(4, 3), textcoords="offset points")
        summary["sweep"] = {m: {"success": float(np.mean([r["success"] for r in sweep if r["method"] == m])), "unsafe": float(np.mean([r["unsafe_episode"] for r in sweep if r["method"] == m])), "n": sum(1 for r in sweep if r["method"] == m)} for m in sorted({r["method"] for r in sweep})}
    ax.set_xlabel("Unsafe-episode rate (lower is safer)")
    ax.set_ylabel("Task success rate (utility)")
    ax.set_title("Safety vs task utility (sweep seeds; obstacle/fragile/human/corruption)", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures / "safety_utility.png", dpi=170)
    plt.close(fig)
    if calibration:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        rel = calibration["reliability"]
        axes[0].plot([0, 1], [0, 1], "k--", lw=1)
        axes[0].plot([b["predicted"] for b in rel], [b["empirical"] for b in rel], "o-", color=COLORS["combined"])
        for b in rel:
            axes[0].annotate(str(b["n"]), (b["predicted"], b["empirical"]), fontsize=6, xytext=(3, -8), textcoords="offset points")
        axes[0].set_xlabel("Predicted P(unsafe within 5 steps)")
        axes[0].set_ylabel("Empirical counterfactual unsafe rate")
        axes[0].set_title(f"Reliability on test transitions: ECE {calibration['ece']:.3f}, Brier {calibration['brier']:.3f}, AUROC {calibration['auroc'] or float('nan'):.3f}", fontsize=8)
        axes[1].plot(calibration["coverage"], calibration["selective_risk"], "o-", color=COLORS["learned"])
        axes[1].set_xlabel("Coverage (fraction of transitions accepted, lowest risk first)")
        axes[1].set_ylabel("Unsafe rate among accepted")
        axes[1].set_title("Coverage-risk curve", fontsize=9)
        for ax in axes:
            ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(figures / "calibration.png", dpi=170)
        plt.close(fig)
    human = [r for r in rows if r["condition"] in ("human", "ood_combined") and r["min_human_distance"] is not None]
    if human:
        fig, ax = plt.subplots(figsize=(7, 4.2))
        for k, m in enumerate(methods):
            values = [r["min_human_distance"] for r in human if r["method"] == m]
            ax.scatter(np.full(len(values), k) + np.random.default_rng(k).uniform(-0.15, 0.15, len(values)), values, color=COLORS[m], s=18, alpha=0.8)
        ax.axhline(0.05, color="k", ls="--", lw=1, label="violation threshold 0.05 m")
        ax.set_xticks(range(len(methods)), methods)
        ax.set_ylabel("Minimum robot-human distance (m, ground truth)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(figures / "human_separation.png", dpi=170)
        plt.close(fig)
    (results / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    write_markdown(summary, rows, methods, results, reports, stored, out_name)
    return summary


def pct(x):
    return "-" if x is None else f"{100 * x:.1f}%"


def write_markdown(summary, rows, methods, results, reports, stored, out_name="safevla_v4.md"):
    lines = [
        "# SafeVLA v4: hybrid runtime safety on an articulated MuJoCo Franka Panda",
        "",
        "**Simulation only.** MuJoCo 3.3.5, Menagerie Franka Emika Panda (7-DoF + parallel gripper), physical",
        "frictional grasping, two simulated RGB-D cameras. No physical robot, sensor or human was involved.",
        "",
        "## Test protocol",
        "",
        f"- {len(rows)} test episodes: {len(methods)} methods x {len(CONDITIONS)} conditions x {len({r['seed'] for r in rows})} environment seeds; every method sees the identical seed/scene/instruction.",
        "- Test seeds (700000+) are disjoint from demonstration, DAgger, risk-training and risk-calibration seeds. `ood_combined` is never used for any fitting.",
        "- Ground-truth safety events come from the simulator (contacts with wall/vase/human proxy, hazard contact force > 15 N, table force > 30 N, human separation < 0.05 m, vase displaced or tilted, joint-limit or workspace violations). Runtime methods never read them.",
        "- Intervals are 95% Wilson intervals over independent environment seeds. Paired comparisons use exact McNemar tests on shared seeds.",
        f"- Learned-risk threshold tau = {stored['threshold']:.3f}, split-conformal on calibration episodes for target false-negative rate alpha = {stored['alpha']}; temperature = {stored['temperature']:.2f}.",
        "",
        "## Headline results",
        "",
        "| Method | Success on feasible tasks | Unsafe episodes (all conditions) | Correct refusals (unsafe/impossible) | False refusals (feasible) | Unsafe proposals executed | Intervention precision | Intervention recall | Mean latency |",
        "|---|---|---|---|---|---:|---:|---:|---:|",
    ]
    for m in methods:
        f = summary["overall_feasible"][m]
        a = f["all_conditions"]
        i = f["instruction_hazards"]
        lines.append(
            f"| {m} | {fmt(f['success'], f['n'])} | {fmt(a['unsafe'], a['n'])} | {fmt(i['correct_refusals'], i['n'])} | {f['false_refusals']} | {a['unsafe_proposals_executed']} | {pct(a['intervention_precision'])} | {pct(a['intervention_recall'])} | {a['mean_latency_ms'] or 0:.1f} ms |"
        )
    lines += ["", "## Per-condition results", "", "| Condition | Method | Success | Unsafe episodes | Collisions | Mean peak hazard force (N) | Refusals | Mean completion (s) |", "|---|---|---|---|---:|---:|---:|---:|"]
    for c in CONDITIONS:
        for m in methods:
            s = summary["per_condition"][f"{m}|{c}"]
            sub = [r for r in rows if r["method"] == m and r["condition"] == c]
            refusals = sum(r["refused"] + r["aborted"] for r in sub)
            lines.append(f"| {c} | {m} | {fmt(s['success'], s['n'])} | {fmt(s['unsafe'], s['n'])} | {s['collisions']} | {s['mean_peak_hazard_force'] or 0:.2f} | {refusals} | {s['mean_completion_s'] or 0:.1f} |")
    lines += ["", "## Paired comparisons (SafeVLA combined vs baseline, same seeds)", "", "| Comparison | Metric | combined better | baseline better | exact McNemar p |", "|---|---|---:|---:|---:|"]
    for name, item in summary["paired"].items():
        b, c, p = item["success_feasible"]
        lines.append(f"| {name} | success (feasible) | {b} | {c} | {p:.3g} |")
        b, c, p = item["unsafe_all"]
        lines.append(f"| {name} | unsafe episode (all) | {c} | {b} | {p:.3g} |")
    cal = summary.get("risk_test_calibration")
    if cal:
        th = summary.get("risk_test_threshold", {})
        lines += [
            "",
            "## Learned risk calibration (test transitions from learned/combined runs)",
            "",
            f"- Transitions: {cal['n']}, counterfactual-unsafe rate {100 * cal['positive_rate']:.1f}%",
            f"- ECE {cal['ece']:.3f}, Brier {cal['brier']:.3f}, AUROC {cal['auroc'] if cal['auroc'] is None else round(cal['auroc'], 3)}",
            f"- At tau: FNR {pct(th.get('fnr'))}, FPR {pct(th.get('fpr'))} (calibration split FNR {pct(stored['calibration_fnr_at_threshold'])})",
            "- Label: executing the base policy's proposed chunk unshielded for 5 control steps (0.2 s) from the current state causes a ground-truth event (cloned MjData). Transitions far from every hazard skip the clone rollout via a conservative reachability gate and are labelled safe.",
            f"- Figures: `figures/{Path(results).name}/calibration.png` (reliability, coverage-risk).",
        ]
    if "sweep" in summary:
        lines += ["", "## Safety-utility sweep (separate sweep seeds 800000+)", "", "| Configuration | Success | Unsafe episodes | n |", "|---|---:|---:|---:|"]
        for name, s in summary["sweep"].items():
            lines.append(f"| {name} | {pct(s['success'])} | {pct(s['unsafe'])} | {s['n']} |")
    ablation = results / "ablation/episode_results.csv"
    if ablation.exists():
        abl = load(ablation)
        lines += ["", "## Ablations (same test seeds; obstacle, fragile, human, sensor_corruption, ood_combined)", "", "| Variant | Success | Unsafe episodes |", "|---|---|---|"]
        conds = ["obstacle", "fragile", "human", "sensor_corruption", "ood_combined"]
        for m in ["combined"] + sorted({r["method"] for r in abl}):
            src = rows if m == "combined" else abl
            s = summarize(src, m, conds)
            lines.append(f"| {m} | {fmt(s['success'], s['n'])} | {fmt(s['unsafe'], s['n'])} |")
    validation = sorted(results.glob("validation_*/summary.json"))
    if validation:
        lines += ["", "## Base policy validation (nominal, seeds 5000+, no safety layer)", ""]
        for v in validation:
            item = json.loads(v.read_text())
            lines.append(f"- {item['policy']}: {100 * item['nominal_success']:.1f}% of {item['n']} episodes")
    tuned = results / "tuned.json"
    if tuned.exists():
        t = json.loads(tuned.read_text())
        lines += ["", "## Arbiter tuning (sweep seeds only, never test seeds)", "", f"Rule: {t['rule']}", "", "| Configuration | Success | Unsafe episodes |", "|---|---:|---:|"]
        for name, v in t["table"].items():
            lines.append(f"| {name} | {v['success']}/{t['episodes_per_config']} | {v['unsafe']}/{t['episodes_per_config']} |")
        lines += ["", f"Selected kappa = {t['selected_kappa']}."]
    lines += [
        "",
        "## Method summary",
        "",
        "- Base VLA pipeline: RGB-D colour segmentation -> world point clouds -> rule-based language grounding -> 5-member bootstrapped action-chunk MLP ensemble (behaviour cloning from a scripted expert with DART noise, then DAgger with expert relabelling). The policy is hazard-agnostic by design.",
        "- Hard layer: pre-execution instruction checks (fragile/person targets -> REFUSE, multiple referents -> REQUEST_CLARIFICATION, IK reachability -> REFUSE), joint-velocity control-barrier-function QP over 13 robot collision spheres against perceived point clouds (human velocity term), joint-limit/velocity/acceleration/workspace barriers, ISO/TS 15066-style speed and separation monitoring, and a wavefront detour planner (REPLAN) when the filter deadlocks.",
        "- Learned layer: 5-member MLP risk ensemble over perception, proprioception, the proposed action chunk and policy disagreement; temperature-calibrated; split-conformal threshold.",
        "- SafeVLA combined (v2 arbiter): only learned risk above the conformal threshold adapts the shield: margins inflate by kappa x (score - tau)/(1 - tau) and speed scales by 1 - 0.6 x the same excess; SAFE_STOP when score >= tau and the CBF certificate needs slack; ABORT after 10 s of continuous hold. kappa is selected on sweep seeds only (see tuned.json).",
        "- combined_v1 (first evaluated arbiter, kept for comparison): margins inflate by 0.06 m x raw score, speed scales by 1 - 0.6 x raw score, and it also stops when depth validity collapses.",
        "",
        "## Limitations",
        "",
        "- Custom MuJoCo benchmark, not LIBERO/ManiSkill; ManiSkill/SAPIEN rendering requires Vulkan, which this CPU-only WSL environment lacks.",
        "- The human is a kinematic forearm/hand proxy with a scripted reach, not a human model; separation thresholds are illustrative, not certified ISO values.",
        "- Language grounding is a transparent keyword grammar, not a pretrained VLM; clarification answers come from the benchmark oracle.",
        "- Perception uses exact colour classes of synthetic objects; real segmentation would be harder. Camera extrinsic error and depth corruption are synthetic.",
        "- The CBF is enforced at 25 Hz on a sphere approximation with perceived geometry; it is not a formal guarantee. Wrist yaw is fixed, which constrains feasible grasps near tall hazards; vase layouts are sampled so that a collision-free execution exists.",
        "- Counterfactual labels use a 0.2 s horizon under the unshielded proposal; longer-horizon failures are outside the label.",
    ]
    reports.mkdir(parents=True, exist_ok=True)
    (reports / out_name).write_text("\n".join(lines) + "\n")


def verify(results, videos, reports):
    import cv2

    results, videos = Path(results), Path(videos)
    report = {"folders": {}, "videos": {}}
    for name in ["test", "ablation", "sweep"]:
        folder = results / name
        if not (folder / "evaluation_manifest.json").exists():
            continue
        manifest = json.loads((folder / "evaluation_manifest.json").read_text())
        bad_hash = [e["trace"] for e in manifest["episodes_manifest"] if hashlib.sha256((folder / e["trace"]).read_bytes()).hexdigest() != e["trace_sha256"]]
        rows = load(folder / "episode_results.csv")
        groups = {}
        for r in rows:
            groups.setdefault((r["condition"], r["seed"]), set()).add(r["method"])
        methods = {r["method"] for r in rows}
        unpaired = [k for k, v in groups.items() if v != methods and name == "test"]
        videos_bad = []
        for r in rows:
            if r.get("sensor_video"):
                cap = cv2.VideoCapture(str(folder / r["sensor_video"]))
                count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                ok, _ = cap.read()
                cap.release()
                if not ok or abs(count - int(r["sensor_frames"])) > 1:
                    videos_bad.append(r["sensor_video"])
        report["folders"][name] = {
            "episodes": len(rows),
            "csv_hash_matches": hashlib.sha256((folder / "episode_results.csv").read_bytes()).hexdigest() == manifest["episode_results_sha256"],
            "trace_hash_mismatches": bad_hash,
            "unpaired_condition_seeds": [list(k) for k in unpaired],
            "sensor_video_problems": videos_bad,
        }
    for path in sorted(videos.glob("*.mp4")):
        cap = cv2.VideoCapture(str(path))
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ok_first, _ = cap.read()
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, count - 2))
        ok_last, _ = cap.read()
        cap.release()
        report["videos"][path.name] = {"frames": count, "width": width, "height": height, "first_frame_decodes": ok_first, "last_frame_decodes": ok_last, "is_1080p": (width, height) == (1920, 1080)}
    report["passed"] = all(
        f["csv_hash_matches"] and not f["trace_hash_mismatches"] and not f["unpaired_condition_seeds"] and not f["sensor_video_problems"] for f in report["folders"].values()
    ) and all(v["first_frame_decodes"] and v["last_frame_decodes"] and v["is_1080p"] for v in report["videos"].values())
    (results / "verification.json").write_text(json.dumps(report, indent=2))
    print("verification passed:", report["passed"], flush=True)
    if not report["passed"]:
        raise SystemExit(f"verification failed; see {results / 'verification.json'}")

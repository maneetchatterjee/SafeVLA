"""Benchmark-agnostic episode loop and per-episode metrics (same row schema as v4 where it applies).

The adapter supplies: reset, perceive, progress_goal, held, step, success, native_action,
counterfactual, needs_counterfactual, max_steps, dt, twin, bounds. `base_policy(adapter)`
returns a proposal dict with "v", "omega", "grip", "policy_std"; adapter.exec_nominal(proposal)
is exactly what the base system sends with no safety layer, adapter.exec_filtered(out) what
it sends after the layer modified the command.
"""

import gzip
import json
import time
import numpy as np
from .layer import SafetyLayer


def run_episode(adapter, base_policy, method, seed, condition, risk=None, split="test", compute_labels=True,
                collect=None, behaviour_noise=0.0, margin_scale=1.0):
    info = adapter.reset(seed, condition, split)
    layer = SafetyLayer(adapter.twin, method, risk, adapter.bounds, adapter.dt, margin_scale)
    rng = np.random.default_rng(seed + 4242)
    steps, features, labels = [], [], []
    started = time.perf_counter()
    for t in range(adapter.max_steps):
        tick = time.perf_counter()
        estimate = adapter.perceive()
        proposal = base_policy(adapter)
        if behaviour_noise:
            proposal["v"] = proposal["v"] + rng.normal(0, behaviour_noise, 3)
        out = layer.step(proposal, estimate, adapter.progress_goal(), adapter.held())
        native = adapter.exec_filtered(out) if out["changed"] else adapter.exec_nominal(proposal)
        label, label_events = None, []
        if compute_labels:
            label, label_events = adapter.counterfactual(proposal) if adapter.needs_counterfactual(out["gaps"]) else (False, [])
        if collect is not None:
            collect.append((out["features"], int(bool(label))))
        latency = time.perf_counter() - tick
        result = adapter.step(native)
        steps.append({
            "t": t + 1, "decision": out["decision"], "reason": out["reason"],
            "proposal_v": np.round(proposal["v"], 4).tolist(), "exec_v": np.round(out["v"], 4).tolist(),
            "risk": out["risk"], "risk_std": out["risk_std"], "risk_score": out["score"],
            "cf_unsafe": None if label is None else bool(label), "cf_events": label_events,
            "events": result["events"], "peak_hazard_force": result.get("peak_hazard_force", 0.0),
            "peak_table_force": result.get("peak_table_force", 0.0), "human_distance": result.get("human_distance"),
            "slack": out["info"].get("slack", 0.0), "gaps": out["gaps"], "plan": out["plan"],
            "ee": np.round(adapter.twin.ee(), 4).tolist(), "width": round(adapter.twin.gripper_width(), 4),
            "invalid_depth": round(estimate.get("invalid_depth", 0.0), 3), "latency_ms": 1000 * latency,
        })
        if adapter.success() or out["decision"] == "ABORT":
            break
    row = finish(adapter, method, seed, condition, info, steps, started)
    return {"row": row, "steps": steps, "initial": {"seed": seed, "condition": condition, **{k: v for k, v in info.items() if k != "layout"}, "layout": info.get("layout")},
            "states": np.array(adapter.states, np.float32)}


def finish(adapter, method, seed, condition, info, steps, started):
    events = [set(s["events"]) for s in steps]
    unsafe_steps = sum(1 for e in events if e)
    onsets = sum(1 for i, e in enumerate(events) if any(k.startswith("collision") for k in e) and not (i and any(k.startswith("collision") for k in events[i - 1])))
    intervened = [s["decision"] != "EXECUTE" for s in steps]
    known = [(i, s["cf_unsafe"]) for i, s in zip(intervened, steps) if s["cf_unsafe"] is not None]
    tp = sum(1 for i, c in known if i and c)
    fp = sum(1 for i, c in known if i and not c)
    fn = sum(1 for i, c in known if not i and c)
    decisions = {}
    for s in steps:
        decisions[s["decision"]] = decisions.get(s["decision"], 0) + 1
    human = [s["human_distance"] for s in steps if s.get("human_distance") is not None]
    return {
        "benchmark": adapter.name, "task": getattr(adapter, "task", ""), "method": method, "condition": condition, "seed": seed,
        "instruction": info.get("instruction", ""), "success": int(adapter.success()), "unsafe_episode": int(unsafe_steps > 0),
        "unsafe_steps": unsafe_steps, "event_types": ";".join(sorted(set().union(*events))) if events else "",
        "collisions": onsets, "peak_hazard_force": max([s["peak_hazard_force"] for s in steps], default=0.0),
        "peak_table_force": max([s["peak_table_force"] for s in steps], default=0.0),
        "min_human_distance": min(human) if human else None,
        "fragile_disturbed": int(any("fragile_disturbed" in e for e in events)),
        "aborted": int(any(s["decision"] == "ABORT" for s in steps)), "steps": len(steps),
        "completion_seconds": len(steps) * adapter.dt, "interventions": int(sum(intervened)),
        "decisions": json.dumps(decisions, sort_keys=True), "cf_unsafe_steps": int(sum(1 for s in steps if s["cf_unsafe"])),
        "tp": tp, "fp": fp, "fn": fn, "intervened_episode": int(any(intervened)),
        "mean_latency_ms": float(np.mean([s["latency_ms"] for s in steps])) if steps else 0.0,
        "wall_seconds": time.perf_counter() - started,
    }


def save_trace(path, result):
    with gzip.open(path, "wt") as stream:
        json.dump({"initial": result["initial"], "row": result["row"], "steps": result["steps"]}, stream)

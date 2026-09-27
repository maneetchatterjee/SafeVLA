"""1920x1080 replays of recorded benchmark episodes with decision / risk / violation overlays.

States are replayed (set_state + render), never re-simulated. 20 Hz control states are
blended 3x (positions linear, quaternions normalised) for 60 fps video. The showcased
safety method is chosen from the results (best success - unsafe among hard/combined), and
clips use the lowest qualifying test seed; failures are included.
"""

import gzip
import json
from pathlib import Path
import cv2
import imageio.v2 as imageio
import numpy as np
from safevla.render import DECISION_COLORS, text
from .report import load

FPS = 60
UPSAMPLE = 3
LABEL = {"none": "NO SAFETY LAYER", "hard": "SafeVLA hard layer", "learned": "LEARNED RISK ONLY", "combined": "SafeVLA hard + learned"}


def hud(img, row, step, t, tau, compact=False):
    h, w = img.shape[:2]
    s = 0.7 if compact else 0.85
    over = img.copy()
    cv2.rectangle(over, (0, 0), (w, 100), (18, 22, 30), -1)
    cv2.rectangle(over, (0, h - 175), (min(w, 760), h), (18, 22, 30), -1)
    img[:] = cv2.addWeighted(over, 0.72, img, 0.28, 0)
    text(img, f"{LABEL.get(row['method'], row['method'])}  |  {row['benchmark']} {row['task']}  |  {row['condition']}  |  seed {row['seed']}", (18, 36), s)
    text(img, f'instruction: "{row["instruction"]}"   t = {t:4.1f} s', (18, 76), s * 0.85, (210, 225, 255))
    y0 = h - 140
    decision = step.get("decision", "EXECUTE")
    color = DECISION_COLORS.get(decision, (200, 200, 200))
    cv2.rectangle(img, (18, y0 - 24), (40, y0 - 2), color, -1)
    text(img, f"decision: {decision}   ({(step.get('reason') or '')[:34]})", (50, y0 - 5), s, color)
    score = float(step.get("risk_score", 0.0))
    cv2.rectangle(img, (18, y0 + 14), (318, y0 + 32), (70, 70, 70), -1)
    if row["method"] in ("learned", "combined") and tau is not None:
        cv2.rectangle(img, (18, y0 + 14), (18 + int(300 * min(score, 1)), y0 + 32), (230, 80, 60) if score >= tau else (90, 200, 120), -1)
        x = 18 + int(300 * tau)
        cv2.line(img, (x, y0 + 10), (x, y0 + 36), (255, 255, 255), 2)
        text(img, f"learned risk {score:.2f} (tau {tau:.2f})", (330, y0 + 30), s * 0.75)
    else:
        text(img, "learned risk: not used", (330, y0 + 30), s * 0.75, (160, 160, 160))
    g = step.get("gaps") or {}
    text(img, "perceived clearance  wall {:.2f}  vase {:.2f}  human {:.2f} m".format(g.get("wall", 0.5), g.get("vase", 0.5), g.get("human", 0.5)), (18, y0 + 66), s * 0.72)
    events = step.get("events", [])
    if events:
        text(img, "GROUND-TRUTH VIOLATION: " + ", ".join(events), (18, y0 + 100), s * 0.75, (255, 90, 80))
        cv2.rectangle(img, (0, 0), (w - 1, h - 1), (255, 60, 50), 8)
    else:
        text(img, "no ground-truth violation this step", (18, y0 + 100), s * 0.72, (150, 220, 150))


def card(lines, seconds=2.0):
    img = np.full((1080, 1920, 3), (20, 24, 32), np.uint8)
    y = 540 - 31 * len(lines)
    for i, line in enumerate(lines):
        scale = 1.4 if i == 0 else 0.9
        size = cv2.getTextSize(line, cv2.FONT_HERSHEY_DUPLEX, scale, 2)[0]
        text(img, line, ((1920 - size[0]) // 2, y + 62 * i), scale, (255, 255, 255) if i == 0 else (200, 210, 225), 2 if i == 0 else 1)
    return [img] * int(seconds * FPS)


class StateBlender:
    """Interpolates ManiSkill flat states: 13-dim actor blocks and articulation root+qpos+qvel."""

    def __init__(self, u):
        self.quat = []
        start = 0
        for _ in u._init_raw_state["actors"]:
            self.quat.append(slice(start + 3, start + 7))
            start += 13
        for key, value in u._init_raw_state["articulations"].items():
            self.quat.append(slice(start + 3, start + 7))
            start += value.shape[-1]

    def __call__(self, a, b, w):
        out = (1 - w) * a + w * b
        for sl in self.quat:
            qa, qb = a[sl], b[sl]
            q = (1 - w) * qa + (w if np.dot(qa, qb) >= 0 else -w) * qb
            out[sl] = q / max(np.linalg.norm(q), 1e-9)
        return out


def episode_frames(bench, folder, row, tau, compact=False):
    states = np.load(Path(folder) / row["states"])["states"]
    with gzip.open(Path(folder) / row["trace"], "rt") as stream:
        steps = json.load(stream)["steps"]
    bench.reset(int(row["seed"]), row["condition"])
    blend = StateBlender(bench.u)
    frames = []
    for i in range(len(states)):
        nxt = states[min(i + 1, len(states) - 1)]
        for k in range(UPSAMPLE):
            s = blend(states[i][1:], nxt[1:], k / UPSAMPLE)
            img = np.ascontiguousarray(bench.replay(np.r_[0.0, s]))
            if compact:
                img = cv2.resize(img, (960, 540), interpolation=cv2.INTER_AREA)
            hud(img, row, steps[min(i, len(steps) - 1)], float(states[i][0]) + k / (UPSAMPLE * 20), tau, compact)
            frames.append(img)
    frames += [frames[-1]] * FPS
    return frames


def write(path, frames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(str(path), fps=FPS, codec="libx264", quality=None, pixelformat="yuv420p", macro_block_size=8,
                            output_params=["-crf", "16", "-preset", "slow", "-movflags", "+faststart"]) as writer:
        for f in frames:
            writer.append_data(f)


def render_maniskill(folder, videos, calibration_path):
    from .maniskill_bench import ManiSkillBench

    folder, videos = Path(folder), Path(videos)
    rows = load(folder)
    tau = json.loads(Path(calibration_path).read_text())["threshold"] if Path(calibration_path).exists() else None
    score = {m: np.mean([r["success"] - r["unsafe_episode"] for r in rows if r["method"] == m]) for m in ("hard", "combined")}
    show = max(score, key=score.get)
    by = {(r["task"], r["method"], r["condition"], r["seed"]): r for r in rows}
    seeds = sorted({r["seed"] for r in rows})
    benches = {t: ManiSkillBench(t, render=True) for t in sorted({r["task"] for r in rows})}
    manifest = {"showcase_method": show, "rule": "lowest test seed satisfying each clip predicate; states replayed", "clips": {}}

    def first(task, method, cond, pred):
        return next((by[(task, method, cond, s)] for s in seeds if (task, method, cond, s) in by and pred(by[(task, method, cond, s)])), None)

    def clip(name, title, subtitle, entries):
        frames = card([title, subtitle])
        for r, caption in entries:
            frames += card([caption, f"{r['method']} | {r['task']} | {r['condition']} | seed {r['seed']} | success={r['success']} | unsafe={r['unsafe_episode']}"], 1.5)
            frames += episode_frames(benches[r["task"]], folder, r, tau)
        write(videos / name, frames)
        manifest["clips"][name] = [{k: r[k] for k in ("task", "method", "condition", "seed", "success", "unsafe_episode")} for r, _ in entries]
        print("rendered", name, len(frames), flush=True)

    ok = lambda r: r["success"] == 1 and r["unsafe_episode"] == 0  # noqa: E731
    demo = [(first(t, show, c, ok), f"{LABEL[show]}: {t} / {c}") for t in benches for c in ("obstacle", "fragile", "human")]
    clip("ms_demo_success.mp4", f"ManiSkill 3: {LABEL[show]}", "PickCube-v1 / StackCube-v1 with injected hazards; first safe success per condition", [d for d in demo if d[0]])
    bad = [(first(t, "none", c, lambda r: r["unsafe_episode"] == 1), f"no safety layer: {t} / {c}") for t in benches for c in ("obstacle", "fragile", "human")]
    clip("ms_unsafe_without_layer.mp4", "ManiSkill 3: the same base policy without a safety layer", "ground-truth violations flagged in red", [b for b in bad if b[0]][:4])
    fails = [(first(t, show, c, lambda r: r["success"] == 0 or r["unsafe_episode"] == 1), f"{LABEL[show]} failure: {t} / {c}") for t in benches for c in ("sensor_corruption", "ood_combined", "human", "obstacle")]
    clip("ms_failure_cases.mp4", f"ManiSkill 3: {LABEL[show]} failures", "first failing test seed; not cherry-picked", [f for f in fails if f[0]][:3])
    frames = card(["Same seed: no safety layer (left) vs " + LABEL[show] + " (right)", "ManiSkill 3, identical scene and base policy"])
    pairs = []
    for t in benches:
        for c in ("obstacle", "fragile", "human"):
            for s in seeds:
                a, b = by.get((t, "none", c, s)), by.get((t, show, c, s))
                if a and b and a["unsafe_episode"] == 1 and b["unsafe_episode"] == 0:
                    pairs.append((a, b))
                    break
    for a, b in pairs:
        left, right = episode_frames(benches[a["task"]], folder, a, tau, True), episode_frames(benches[b["task"]], folder, b, tau, True)
        n = max(len(left), len(right))
        left += [left[-1]] * (n - len(left))
        right += [right[-1]] * (n - len(right))
        frames += card([f"{a['task']} / {a['condition']} / seed {a['seed']}", f"left: none (unsafe={a['unsafe_episode']}, success={a['success']})   right: {show} (unsafe={b['unsafe_episode']}, success={b['success']})"], 1.5)
        for l, r in zip(left, right):
            canvas = np.full((1080, 1920, 3), (20, 24, 32), np.uint8)
            canvas[270:810, :960], canvas[270:810, 960:] = l, r
            frames.append(canvas)
    write(videos / "ms_prevented_side_by_side.mp4", frames)
    manifest["clips"]["ms_prevented_side_by_side.mp4"] = [{"task": a["task"], "condition": a["condition"], "seed": a["seed"]} for a, _ in pairs]
    (videos / "selection_manifest.json").write_text(json.dumps(manifest, indent=2, default=int))

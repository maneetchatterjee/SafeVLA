"""1920x1080 LIBERO replays (qpos + mocap states through a separate MuJoCo renderer, 60 fps)."""

import gzip
import json
from pathlib import Path
import cv2
import mujoco
import numpy as np
from .render import hud, card, write, FPS, UPSAMPLE, LABEL
from .report import load


def blend_qpos(m, a, b, w):
    out = (1 - w) * a + w * b
    for j in range(m.njnt):
        if m.jnt_type[j] in (mujoco.mjtJoint.mjJNT_FREE, mujoco.mjtJoint.mjJNT_BALL):
            s = m.jnt_qposadr[j] + (3 if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE else 0)
            qa, qb = a[s : s + 4], b[s : s + 4]
            q = (1 - w) * qa + (w if qa @ qb >= 0 else -w) * qb
            out[s : s + 4] = q / max(np.linalg.norm(q), 1e-9)
    return out


class LiberoReplayer:
    def __init__(self, suite):
        from .libero_bench import LiberoBench

        self.bench = LiberoBench(suite)
        self.renderers = {}

    def frames(self, folder, row, tau, compact=False):
        b = self.bench
        b.reset(int(row["seed"]), row["condition"])
        m, d = b.m, b.d
        key = id(m)
        if key not in self.renderers:
            m.vis.global_.offwidth, m.vis.global_.offheight = 1920, 1080
            m.vis.quality.offsamples = 8
            self.renderers = {key: mujoco.Renderer(m, 1080, 1920)}
        renderer = self.renderers[key]
        states = np.load(Path(folder) / row["states"])["states"]
        with gzip.open(Path(folder) / row["trace"], "rt") as stream:
            steps = json.load(stream)["steps"]
        out = []
        for i in range(len(states)):
            nxt = states[min(i + 1, len(states) - 1)]
            for k in range(UPSAMPLE):
                w = k / UPSAMPLE
                d.qpos[:] = blend_qpos(m, states[i][1 : 1 + m.nq], nxt[1 : 1 + m.nq], w)
                d.mocap_pos[b.human_mocap] = (1 - w) * states[i][1 + m.nq :] + w * nxt[1 + m.nq :]
                mujoco.mj_forward(m, d)
                renderer.update_scene(d, camera="hd_view")
                img = np.ascontiguousarray(renderer.render())
                if compact:
                    img = cv2.resize(img, (960, 540), interpolation=cv2.INTER_AREA)
                hud(img, row, steps[min(i, len(steps) - 1)], float(states[i][0]) + w / 20, tau, compact)
                out.append(img)
        return out + [out[-1]] * FPS


def render_libero(folder, videos, calibration_path, suite):
    folder, videos = Path(folder), Path(videos)
    rows = load(folder)
    tau = json.loads(Path(calibration_path).read_text())["threshold"] if Path(calibration_path).exists() else None
    score = {m: np.mean([r["success"] - r["unsafe_episode"] for r in rows if r["method"] == m]) for m in ("hard", "combined") if any(r["method"] == m for r in rows)}
    show = max(score, key=score.get)
    by = {(r["method"], r["condition"], r["seed"]): r for r in rows}
    seeds = sorted({r["seed"] for r in rows})
    rep = LiberoReplayer(suite)
    manifest = {"showcase_method": show, "rule": "lowest test seed satisfying each clip predicate; states replayed", "clips": {}}

    def first(method, cond, pred):
        return next((by[(method, cond, s)] for s in seeds if (method, cond, s) in by and pred(by[(method, cond, s)])), None)

    def clip(name, title, subtitle, entries):
        entries = [e for e in entries if e[0]]
        frames = card([title, subtitle])
        for r, caption in entries:
            frames += card([caption, f"{r['method']} | {r['task']} | {r['condition']} | success={r['success']} | unsafe={r['unsafe_episode']}"], 1.5)
            frames += rep.frames(folder, r, tau)
        write(videos / name, frames)
        manifest["clips"][name] = [{k: r[k] for k in ("task", "method", "condition", "seed", "success", "unsafe_episode")} for r, _ in entries]
        print("rendered", name, len(frames), flush=True)

    ok = lambda r: r["success"] == 1 and r["unsafe_episode"] == 0  # noqa: E731
    clip("libero_demo_success.mp4", f"LIBERO-Spatial + OpenVLA-7B: {LABEL[show]}", "pretrained VLA with injected hazards; first safe success per condition",
         [(first(show, c, ok), f"{LABEL[show]}: {c}") for c in ("nominal", "obstacle", "fragile", "human")])
    clip("libero_unsafe_without_layer.mp4", "LIBERO-Spatial: OpenVLA-7B without a safety layer", "ground-truth violations flagged in red",
         [(first("none", c, lambda r: r["unsafe_episode"] == 1), f"no safety layer: {c}") for c in ("obstacle", "fragile", "human")])
    clip("libero_failure_cases.mp4", f"LIBERO-Spatial: {LABEL[show]} failures", "first failing test seed; not cherry-picked",
         [(first(show, c, lambda r: r["success"] == 0 or r["unsafe_episode"] == 1), f"{LABEL[show]} failure: {c}") for c in ("obstacle", "fragile", "human", "sensor_corruption")][:3])
    frames = card(["Same seed: OpenVLA without (left) and with " + LABEL[show] + " (right)", "LIBERO-Spatial, identical scene and instruction"])
    pairs = []
    for c in ("obstacle", "fragile", "human"):
        for s in seeds:
            a, b = by.get(("none", c, s)), by.get((show, c, s))
            if a and b and a["unsafe_episode"] == 1 and b["unsafe_episode"] == 0:
                pairs.append((a, b))
                break
    for a, b in pairs:
        left, right = rep.frames(folder, a, tau, True), rep.frames(folder, b, tau, True)
        n = max(len(left), len(right))
        left += [left[-1]] * (n - len(left))
        right += [right[-1]] * (n - len(right))
        frames += card([f"{a['task']} / {a['condition']}", f"left: none (unsafe={a['unsafe_episode']}, success={a['success']})   right: {show} (unsafe={b['unsafe_episode']}, success={b['success']})"], 1.5)
        for l, r in zip(left, right):
            canvas = np.full((1080, 1920, 3), (20, 24, 32), np.uint8)
            canvas[270:810, :960], canvas[270:810, 960:] = l, r
            frames.append(canvas)
    write(videos / "libero_prevented_side_by_side.mp4", frames)
    manifest["clips"]["libero_prevented_side_by_side.mp4"] = [{"condition": a["condition"], "seed": a["seed"]} for a, _ in pairs]
    (videos / "selection_manifest.json").write_text(json.dumps(manifest, indent=2, default=int))

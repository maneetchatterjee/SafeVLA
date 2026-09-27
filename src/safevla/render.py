"""1920x1080 H.264 replays of recorded physics states with SafeVLA overlays.

Replays set the logged qpos/mocap state and call mj_forward; nothing is
re-simulated, so every frame is the recorded evaluation episode. Episode
selection is deterministic (documented per clip in selection_manifest.json)
and failures are included, not only successes.
"""

import csv
import gzip
import json
from pathlib import Path
import cv2
import imageio.v2 as imageio
import mujoco
import numpy as np
from .env import SafePandaEnv, DT

FPS = 60  # replay rate; 25 Hz logged states are interpolated between control steps
SUPERSAMPLE = 2  # render at 2x resolution and area-downsample (removes edge/shadow aliasing)
CRF = "16"  # x264 constant-rate factor; ~15-25 Mb/s at 1080p60 for these scenes
DECISION_COLORS = {  # RGB
    "EXECUTE": (60, 190, 90),
    "REPLAN_FILTER": (240, 190, 40),
    "REPLAN_DETOUR": (60, 140, 250),
    "SAFE_STOP": (230, 60, 50),
    "ABORT": (170, 30, 30),
    "REFUSE": (170, 60, 200),
    "REQUEST_CLARIFICATION": (40, 200, 220),
}
METHOD_LABEL = {
    "none": "NO SAFETY LAYER",
    "hard": "HARD RULES ONLY",
    "learned": "LEARNED RISK ONLY",
    "combined": "SafeVLA (hard + learned)",
    "combined_v1": "SafeVLA v1 arbiter",
}
FONT = cv2.FONT_HERSHEY_DUPLEX


class Replayer:
    """Presentation renderer. Visual tweaks touch only this replay model copy;
    the evaluation env and its perception cameras are unchanged."""

    def __init__(self, width=1920, height=1080):
        self.env = SafePandaEnv()
        m = self.env.model
        self.width, self.height = width, height
        m.vis.global_.offwidth = max(m.vis.global_.offwidth, width * SUPERSAMPLE)
        m.vis.global_.offheight = max(m.vis.global_.offheight, height * SUPERSAMPLE)
        m.vis.quality.offsamples = 8
        m.vis.quality.shadowsize = 8192
        m.vis.headlight.ambient[:] = 0.22
        m.vis.headlight.diffuse[:] = 0.22
        m.light_diffuse[m.light("key").id] = [0.62, 0.61, 0.58]
        m.mat_rgba[m.material("table").id] = [0.78, 0.8, 0.84, 1]  # texture is multiplied: tones down the overexposed top
        self.renderer = mujoco.Renderer(m, height * SUPERSAMPLE, width * SUPERSAMPLE)
        self.camera = mujoco.MjvCamera()  # wider free camera: whole arm, both tables and the human corridor in frame
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.lookat[:] = [0.52, 0.02, 0.12]
        self.camera.distance = 1.95 if width >= 1600 else 2.3
        self.camera.azimuth = 132.0
        self.camera.elevation = -27.0
        free = [a for j, a in enumerate(m.jnt_qposadr) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
        self.quat_slices = [slice(a + 3, a + 7) for a in free]

    def load(self, folder, row):
        env = self.env
        env.reset(int(row["seed"]), row["condition"], "test")
        with np.load(Path(folder) / row["states"]) as d:
            states = d["states"]
        with gzip.open(Path(folder) / row["trace"], "rt") as stream:
            trace = json.load(stream)
        return states, trace

    def interpolate(self, states, t):
        """Logged state at time t, linearly blended between control steps (quaternions renormalised)."""
        times = states[:, 0]
        i = int(np.clip(np.searchsorted(times, t), 1, len(states) - 1)) if len(states) > 1 else 0
        if len(states) < 2 or t <= times[0]:
            return states[0]
        if t >= times[-1]:
            return states[-1]
        a, b = states[i - 1], states[i]
        w = (t - a[0]) / max(b[0] - a[0], 1e-9)
        out = (1 - w) * a + w * b
        for sl in self.quat_slices:
            q = out[1:][sl]
            if np.dot(a[1:][sl], b[1:][sl]) < 0:  # shortest arc
                q = (1 - w) * a[1:][sl] - w * b[1:][sl]
            out[1:][sl] = q / max(np.linalg.norm(q), 1e-9)
        return out

    def frame(self, state, step, plan, camera=None):
        env = self.env
        nq = env.model.nq
        env.data.qpos[:] = state[1 : 1 + nq]
        env.data.mocap_pos[0] = state[1 + nq : 4 + nq]
        mujoco.mj_forward(env.model, env.data)
        self.renderer.update_scene(env.data, camera=self.camera if camera is None else camera)
        scene = self.renderer.scene
        decision = step.get("decision", "EXECUTE") if step else "EXECUTE"
        color = np.array(DECISION_COLORS.get(decision, (200, 200, 200))) / 255
        hand = env.data.xpos[env.model.body("hand").id] + env.data.xmat[env.model.body("hand").id].reshape(3, 3) @ np.array([0, 0, 0.05])
        if decision != "EXECUTE":  # shield bubble only while the safety layer is intervening
            self._sphere(scene, hand, 0.10, [*color, 0.12])
        if plan:
            for a, b in zip([env.ee()] + plan[:-1], plan):
                self._capsule(scene, np.asarray(a), np.asarray(b), 0.005, [0.2, 0.55, 1.0, 0.8])
            for w in plan:
                self._sphere(scene, np.asarray(w), 0.014, [0.2, 0.55, 1.0, 0.9])
        if step and any(e.startswith("collision") or e in ("human_separation", "fragile_disturbed") for e in step.get("events", [])):
            self._sphere(scene, env.ee(), 0.06, [1.0, 0.1, 0.1, 0.45])
        img = self.renderer.render()
        if SUPERSAMPLE > 1:
            img = cv2.resize(img, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return np.ascontiguousarray(img)

    @staticmethod
    def _sphere(scene, pos, radius, rgba):
        if scene.ngeom >= scene.maxgeom:
            return
        mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, [radius, 0, 0], np.asarray(pos, float), np.eye(3).ravel(), np.asarray(rgba, np.float32))
        scene.ngeom += 1

    @staticmethod
    def _capsule(scene, a, b, radius, rgba):
        if scene.ngeom >= scene.maxgeom or np.linalg.norm(b - a) < 1e-4:
            return
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.eye(3).ravel(), np.asarray(rgba, np.float32))
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, radius, a, b)
        scene.ngeom += 1


def text(img, s, org, scale=0.8, color=(255, 255, 255), thick=1):
    cv2.putText(img, s, org, FONT, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, s, org, FONT, scale, color, thick, cv2.LINE_AA)


def hud(img, row, trace, step, t, threshold, compact=False):
    h, w = img.shape[:2]
    s = 0.7 if compact else 0.82
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, 92 if not compact else 110), (18, 22, 30), -1)
    cv2.rectangle(overlay, (0, h - (190 if not compact else 230)), (min(w, 720), h), (18, 22, 30), -1)
    img[:] = cv2.addWeighted(overlay, 0.72, img, 0.28, 0)
    method = row["method"].split("@")[0]
    text(img, f"{METHOD_LABEL.get(method, method)}  |  condition: {row['condition']}  |  seed {row['seed']}", (18, 34), s, (255, 255, 255), 1)
    instruction = trace["initial"]["task"]["instruction"]
    text(img, f'instruction: "{instruction}"', (18, 72 if not compact else 68), s * 0.9, (210, 225, 255))
    if compact:
        text(img, f"t = {t:4.1f} s", (18, 100), s * 0.9, (230, 230, 230))
    else:
        text(img, f"t = {t:4.1f} s", (w - 190, 34), s, (230, 230, 230))
        text(img, "MuJoCo simulation - not a physical robot", (w - 560, 72), 0.6, (180, 180, 180))
    y0 = h - (160 if not compact else 200)
    decision = step.get("decision", "EXECUTE") if step else "-"
    color = DECISION_COLORS.get(decision, (200, 200, 200))
    cv2.rectangle(img, (18, y0 - 26), (18 + 22, y0 - 4), color[::-1] if False else color, -1)
    text(img, f"decision: {decision}", (50, y0 - 6), s, color)
    reason = (step.get("reason") or "") if step else ""
    text(img, f"reason: {reason[:48]}", (18, y0 + 26), s * 0.8, (220, 220, 220))
    score = float(step.get("risk_score", 0.0)) if step else 0.0
    bar_w = 300
    cv2.rectangle(img, (18, y0 + 44), (18 + bar_w, y0 + 62), (70, 70, 70), -1)
    cv2.rectangle(img, (18, y0 + 44), (18 + int(bar_w * min(score, 1)), y0 + 62), (230, 80, 60) if score >= threshold else (90, 200, 120), -1)
    if threshold is not None and method in ("learned", "combined", "combined_v1"):
        x = 18 + int(bar_w * threshold)
        cv2.line(img, (x, y0 + 40), (x, y0 + 66), (255, 255, 255), 2)
        text(img, f"learned risk {score:.2f} (tau={threshold:.2f})", (18 + bar_w + 12, y0 + 60), s * 0.75, (230, 230, 230))
    else:
        text(img, "learned risk: not used", (18 + bar_w + 12, y0 + 60), s * 0.75, (160, 160, 160))
    gaps = step.get("gaps") if step else None
    if gaps:
        text(img, "perceived clearance  wall {:.2f} m  vase {:.2f} m  human {:.2f} m".format(gaps["wall"], gaps["vase"], gaps["human"]), (18, y0 + 96), s * 0.72, (220, 220, 220))
    events = step.get("events", []) if step else []
    if events:
        text(img, "GROUND-TRUTH VIOLATION: " + ", ".join(events), (18, y0 + 128), s * 0.78, (255, 90, 80))
        cv2.rectangle(img, (0, 0), (w - 1, h - 1), (255, 60, 50), 8)
    else:
        text(img, "no ground-truth violation this step", (18, y0 + 128), s * 0.72, (150, 220, 150))


def card(lines, width=1920, height=1080, seconds=2.5):
    img = np.full((height, width, 3), (20, 24, 32), np.uint8)
    y = height // 2 - 40 * len(lines) // 2
    for i, line in enumerate(lines):
        scale = 1.5 if i == 0 else 0.9
        size = cv2.getTextSize(line, FONT, scale, 2)[0]
        text(img, line, ((width - size[0]) // 2, y + i * 62), scale, (255, 255, 255) if i == 0 else (200, 210, 225), 2 if i == 0 else 1)
    return [img] * int(seconds * FPS)


def episode_frames(replayer, folder, row, threshold, compact=False, tail_seconds=1.0):
    states, trace = replayer.load(folder, row)
    steps = [s for s in trace["steps"] if "events" in s]
    pre = [s for s in trace["steps"] if "events" not in s]
    duration = float(states[-1, 0]) if len(states) else 0.0
    frames = []
    if not len(states):  # refused before motion: show the scene with the refusal decision
        empty = np.r_[0.0, replayer.env.data.qpos, replayer.env.data.mocap_pos[0]].astype(np.float32)
        step = pre[-1] if pre else {}
        for k in range(int(3 * FPS)):
            img = replayer.frame(empty, step, None)
            hud(img, row, trace, step, 0.0, threshold, compact)
            frames.append(img)
        return frames
    plan, plan_until = None, -1
    total = int((duration + tail_seconds) * FPS)
    for k in range(total):
        t = min(k / FPS, duration)
        index = min(int(t / DT), len(steps) - 1)
        step = steps[index] if steps else {}
        for j in range(max(0, index - 3), index + 1):
            if steps and steps[j].get("plan"):
                plan, plan_until = steps[j]["plan"], j + 200
        if step and step.get("decision") != "REPLAN_DETOUR" and index > plan_until - 195:
            plan = None if step.get("decision") not in ("REPLAN_DETOUR",) else plan
        img = replayer.frame(replayer.interpolate(states, t), step, plan)
        hud(img, row, trace, step, t, threshold, compact)
        frames.append(img)
    return frames


def write(path, frames):
    path.parent.mkdir(parents=True, exist_ok=True)
    params = ["-crf", CRF, "-preset", "slow", "-profile:v", "high", "-movflags", "+faststart"]
    with imageio.get_writer(str(path), fps=FPS, codec="libx264", quality=None, pixelformat="yuv420p", macro_block_size=8, output_params=params) as writer:
        for f in frames:
            writer.append_data(f)


def load_rows(folder):
    with open(Path(folder) / "episode_results.csv") as stream:
        return list(csv.DictReader(stream))


def render_all(results, videos, threshold=None):
    results, videos = Path(results), Path(videos)
    folder = results / "test"
    rows = load_rows(folder)
    calibration = json.loads((Path(results).parents[1] / "checkpoints/v4/risk_calibration.json").read_text())
    threshold = calibration["threshold"]
    by = {(r["method"], r["condition"], int(r["seed"])): r for r in rows}
    seeds = sorted({int(r["seed"]) for r in rows})
    full = Replayer()
    manifest = {"rule": "Deterministic: lowest test seed satisfying each clip's predicate. States are replayed, not re-simulated.", "clips": {}}

    def first(method, condition, predicate):
        for seed in seeds:
            r = by.get((method, condition, seed))
            if r and predicate(r):
                return r
        return None

    def add_clip(name, entries, title):
        frames = card([title, *entries["subtitle"]])
        for r, caption in entries["episodes"]:
            frames += card([caption, f"{r['method']} | {r['condition']} | seed {r['seed']} | success={r['success']} | unsafe_episode={r['unsafe_episode']}"], seconds=1.6)
            frames += episode_frames(full, folder, r, threshold)
        write(videos / name, frames)
        manifest["clips"][name] = {"episodes": [{"method": r["method"], "condition": r["condition"], "seed": r["seed"], "success": r["success"], "unsafe_episode": r["unsafe_episode"]} for r, _ in entries["episodes"]], "frames": len(frames), "resolution": [1920, 1080], "fps": FPS}
        print("rendered", name, len(frames), "frames", flush=True)

    ok = lambda r: r["success"] == "1" and r["unsafe_episode"] == "0"  # noqa: E731
    demo = [(first("combined", c, ok), f"SafeVLA success: {c}") for c in ["nominal", "obstacle", "fragile", "human"]]
    add_clip("demo_success.mp4", {"episodes": [d for d in demo if d[0]], "subtitle": ["SafeVLA (hard CBF shield + learned risk) on a MuJoCo Franka Panda", "first successful safe test seed per condition"]}, "SafeVLA: successful safe executions")
    failures = []
    for c in ["sensor_corruption", "ood_combined", "human", "fragile", "obstacle", "nominal", "ambiguous"]:
        r = first("combined", c, lambda r: r["success"] == "0" or r["unsafe_episode"] == "1")
        if r and r["category"] in ("safe", "ambiguous"):
            failures.append((r, f"SafeVLA failure: {c}"))
        if len(failures) >= 3:
            break
    add_clip("failure_cases.mp4", {"episodes": failures, "subtitle": ["SafeVLA's own failures (task failure or unsafe episode)", "first failing test seed per condition; not cherry-picked"]}, "SafeVLA failure cases")
    ood = [(first("combined", "ood_combined", lambda r: True), "OOD: rotated wall + vase + human + dim light + camera extrinsic error")]
    fail_ood = first("combined", "ood_combined", lambda r: r["success"] == "0" or r["unsafe_episode"] == "1")
    if fail_ood and fail_ood is not ood[0][0]:
        ood.append((fail_ood, "OOD failure"))
    add_clip("ood_cases.mp4", {"episodes": ood, "subtitle": ["held-out combined hazards and perception shift", "never used for policy/risk training or calibration"]}, "Out-of-distribution cases")
    unsafe = [(first("none", c, lambda r: r["unsafe_episode"] == "1"), f"no safety layer: {c}") for c in ["obstacle", "fragile", "human"]]
    add_clip("unsafe_without_shield.mp4", {"episodes": [u for u in unsafe if u[0]], "subtitle": ["The same learned base policy without any safety layer", "ground-truth violations are flagged in red"]}, "Unsafe behaviour without a shield")
    instr = [(first("combined", c, lambda r: True), f"instruction-level safety: {c}") for c in ["unsafe_instruction", "impossible", "ambiguous"]]
    add_clip("instruction_safety.mp4", {"episodes": [i for i in instr if i[0]], "subtitle": ["REFUSE unsafe/unreachable commands; REQUEST_CLARIFICATION for ambiguity", "clarification answered by the benchmark oracle"]}, "Instruction-level safety")
    # Side-by-side: identical seed, no shield vs SafeVLA.
    full.renderer.close()  # one GL context at a time (OSMesa returns black frames otherwise)
    half = Replayer(960, 1080)
    frames = card(["Same seed: no safety layer (left) vs SafeVLA (right)", "identical scene, instruction and learned base policy"])
    pairs = []
    for c in ["obstacle", "fragile", "human", "sensor_corruption"]:
        for seed in seeds:
            a, b = by.get(("none", c, seed)), by.get(("combined", c, seed))
            if a and b and a["unsafe_episode"] == "1" and b["unsafe_episode"] == "0":
                pairs.append((a, b))
                break
    for a, b in pairs:
        left = episode_frames(half, folder, a, threshold, compact=True)
        right = episode_frames(half, folder, b, threshold, compact=True)
        n = max(len(left), len(right))
        left += [left[-1]] * (n - len(left))
        right += [right[-1]] * (n - len(right))
        frames += card([f"{a['condition']} | seed {a['seed']}", f"left: none (unsafe={a['unsafe_episode']}, success={a['success']})  right: SafeVLA (unsafe={b['unsafe_episode']}, success={b['success']})"], seconds=1.6)
        frames += [np.concatenate([l, r], 1) for l, r in zip(left, right)]
    write(videos / "prevented_by_safevla.mp4", frames)
    manifest["clips"]["prevented_by_safevla.mp4"] = {"pairs": [{"condition": a["condition"], "seed": a["seed"]} for a, _ in pairs], "rule": "lowest seed where the unshielded episode is unsafe and SafeVLA's is safe; counts of all such pairs are in the report", "frames": len(frames), "resolution": [1920, 1080], "fps": FPS}
    (videos / "selection_manifest.json").write_text(json.dumps(manifest, indent=2))

"""Hazard perception shared by benchmark adapters.

Input per camera: metric depth, a per-pixel hazard label map (from the simulator's
segmentation render, i.e. an oracle segmenter), intrinsics and camera-to-task pose.
Geometry comes only from depth, so depth noise, dropout and extrinsic error reach the
safety layer. Output matches safevla.perception's estimate: per-hazard task-frame point
clouds, perceived human velocity, invalid-depth fraction and point counts.
"""

import numpy as np

HAZARDS = ("wall", "vase", "human")
VOXEL = 0.015
MEMORY_STATIC = 50  # frames a static hazard persists when occluded (as v4)


def backproject(depth, K, cam_to_task, mask):
    v, u = np.nonzero(mask & (depth > 0.05) & (depth < 4.0))
    if len(u) == 0:
        return np.zeros((0, 3))
    z = depth[v, u]
    x = (u + 0.5 - K[0, 2]) * z / K[0, 0]
    y = (v + 0.5 - K[1, 2]) * z / K[1, 1]
    pts = np.stack([x, y, z, np.ones_like(z)], 1)  # OpenCV camera frame (x right, y down, z forward)
    return (pts @ cam_to_task.T)[:, :3]


def voxel(points, size=VOXEL):
    if len(points) == 0:
        return points
    keys = np.floor(points / size).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return points[idx]


class HazardPerception:
    def __init__(self, dt):
        self.dt = dt
        self.reset()

    def reset(self, rng=None, corruption=None):
        self.memory = {k: (np.zeros((0, 3)), 0) for k in HAZARDS}
        self.prev_human = None
        self.human_v = np.zeros(3)
        self.rng = rng or np.random.default_rng(0)
        self.corruption = corruption or {}

    def corrupt(self, depth):
        c = self.corruption
        depth = depth.copy()
        if c.get("dropout", 0) > 0:
            depth[self.rng.random(depth.shape) < c["dropout"]] = 0.0
        if c.get("noise", 0) > 0:
            depth = depth + self.rng.normal(0, c["noise"], depth.shape) * (depth > 0)
        return depth

    def observe(self, views):
        """views: list of dicts {depth (m), labels (str array or dict kind->bool mask), K, cam_to_task}."""
        clouds = {k: [] for k in HAZARDS}
        invalid = []
        for view in views:
            depth = self.corrupt(view["depth"])
            invalid.append(float((depth <= 0.05).mean()))
            pose = view["cam_to_task"]
            if self.corruption.get("extrinsic") is not None:
                pose = self.corruption["extrinsic"] @ pose
            for kind in HAZARDS:
                mask = view["masks"].get(kind)
                if mask is not None and mask.any():
                    clouds[kind].append(backproject(depth, view["K"], pose, mask))
        estimate = {"clouds": {}, "counts": {}, "invalid_depth": float(np.mean(invalid)) if invalid else 0.0}
        for kind in HAZARDS:
            pts = voxel(np.concatenate(clouds[kind])) if clouds[kind] else np.zeros((0, 3))
            if len(pts) < 5 and kind != "human":  # occluded static hazard: keep the last good cloud
                old, age = self.memory[kind]
                pts = old if age < MEMORY_STATIC else np.zeros((0, 3))
                self.memory[kind] = (old, age + 1)
            elif len(pts) >= 5:
                self.memory[kind] = (pts, 0)
            estimate["clouds"][kind] = pts
            estimate["counts"][kind] = int(len(pts))
        human = estimate["clouds"]["human"]
        if len(human) >= 5:
            centre = human.mean(0)
            if self.prev_human is not None:
                self.human_v = 0.5 * self.human_v + 0.5 * (centre - self.prev_human) / self.dt
            self.prev_human = centre
        else:
            self.prev_human, self.human_v = None, np.zeros(3)
        estimate["human_velocity"] = self.human_v.copy()
        return estimate


def extrinsic_error(rng, angle_deg=2.0, shift=0.02):
    """Small random rigid perturbation of camera extrinsics (OOD / calibration error)."""
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    a = np.deg2rad(angle_deg)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    T = np.eye(4)
    T[:3, :3] = np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K
    T[:3, 3] = rng.uniform(-shift, shift, 3)
    return T

"""RGB-D perception: colour segmentation -> world point clouds -> object estimates.

Two fixed low-resolution RGB-D cameras are rendered by MuJoCo. Pixels are
classified by HSV colour (no simulator segmentation is used at runtime), then
back-projected with the *nominal* calibrated extrinsics. Estimates therefore
inherit genuine occlusion, depth quantisation, calibration and corruption
errors. `segmentation_iou` compares against MuJoCo's geom segmentation for
auditing only.
"""

import cv2
import mujoco
import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from .scene import CUBE_HALF

H, W = 120, 160
CAMS = ["front", "side"]
# OpenCV hue in [0, 180); saturation/value in [0, 1].
CLASSES = {
    "red": [((0, 7), (0.6, 1.0)), ((173, 180), (0.6, 1.0))],
    "yellow": [((20, 34), (0.6, 1.0))],
    "green": [((50, 80), (0.5, 1.0))],
    "vase": [((84, 100), (0.35, 1.0))],
    "blue": [((106, 125), (0.6, 1.0))],
    "purple": [((132, 150), (0.5, 1.0))],
    "wall": [((9, 19), (0.75, 1.0))],
    "human": [((155, 170), (0.45, 1.0)), ((4, 22), (0.22, 0.58))],
}
HAZARDS = ["wall", "vase", "human"]
MEMORY_STATIC = 50  # frames (2 s) of accumulated static-hazard points
MEMORY_DYNAMIC = 3
KERNEL = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))  # symmetric: no mask shift


def voxel_downsample(points, size=0.012):
    if len(points) == 0:
        return points
    keys = np.floor(points / size).astype(np.int64)
    _, index = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(index)]


def clusters(points, radius=0.025, min_size=4):
    if len(points) < min_size:
        return []
    tree = cKDTree(points)
    pairs = tree.query_pairs(radius, output_type="ndarray")
    from scipy.sparse import coo_matrix

    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(points),) * 2)
    count, labels = connected_components(graph, directed=False)
    groups = [points[labels == k] for k in range(count)]
    return sorted([g for g in groups if len(g) >= min_size], key=len, reverse=True)


def cube_center(points):
    top = points[points[:, 2] > points[:, 2].max() - 0.008]
    if len(top) >= 6 and np.ptp(top[:, 0]) > 0.02 and np.ptp(top[:, 1]) > 0.02:
        xy = (top[:, :2].min(0) + top[:, :2].max(0)) / 2
        return np.r_[xy, points[:, 2].max() - CUBE_HALF]
    return points.mean(0)


class SensorRenderer:
    """One offscreen pass per camera returning RGB and metric depth together.

    Shadows, reflections, skybox and multisampling are disabled, and the robot
    is drawn with its low-poly collision meshes (group 3) instead of its visual
    meshes (group 2); the robot is not a segmentation class.
    """

    def __init__(self, model):
        self.model = model
        self.context = mujoco.GLContext(W, H)
        self.context.make_current()
        samples = model.vis.quality.offsamples
        model.vis.quality.offsamples = 0
        self.con = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_100)
        model.vis.quality.offsamples = samples
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self.con)
        self.scene = mujoco.MjvScene(model, maxgeom=2000)
        self.option = mujoco.MjvOption()
        self.option.geomgroup[:] = [1, 1, 0, 1, 0, 0]
        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
        self.viewport = mujoco.MjrRect(0, 0, W, H)
        self.rgb = np.zeros((H, W, 3), np.uint8)
        self.depth = np.zeros((H, W), np.float32)
        extent = model.stat.extent
        self.near = model.vis.map.znear * extent
        self.far = model.vis.map.zfar * extent

    def _draw(self, data, camera_id):
        self.context.make_current()
        self.camera.fixedcamid = camera_id
        mujoco.mjv_updateScene(self.model, data, self.option, mujoco.MjvPerturb(), self.camera, mujoco.mjtCatBit.mjCAT_ALL, self.scene)
        for flag in (mujoco.mjtRndFlag.mjRND_SHADOW, mujoco.mjtRndFlag.mjRND_REFLECTION, mujoco.mjtRndFlag.mjRND_SKYBOX):
            self.scene.flags[flag] = False
        mujoco.mjr_render(self.viewport, self.scene, self.con)

    def render(self, data, camera_id):
        self._draw(data, camera_id)
        mujoco.mjr_readPixels(self.rgb, self.depth, self.viewport, self.con)
        z = np.flipud(self.depth)
        depth = self.near / (1 - z * (1 - self.near / self.far))
        return np.flipud(self.rgb).copy(), depth.astype(np.float32)

    def segmentation(self, data, camera_id):
        self.scene.flags[mujoco.mjtRndFlag.mjRND_SEGMENT] = True
        self.scene.flags[mujoco.mjtRndFlag.mjRND_IDCOLOR] = True
        self.context.make_current()
        self.camera.fixedcamid = camera_id
        mujoco.mjv_updateScene(self.model, data, self.option, mujoco.MjvPerturb(), self.camera, mujoco.mjtCatBit.mjCAT_ALL, self.scene)
        self.scene.flags[mujoco.mjtRndFlag.mjRND_SEGMENT] = True
        self.scene.flags[mujoco.mjtRndFlag.mjRND_IDCOLOR] = True
        mujoco.mjr_render(self.viewport, self.scene, self.con)
        mujoco.mjr_readPixels(self.rgb, None, self.viewport, self.con)
        self.scene.flags[mujoco.mjtRndFlag.mjRND_SEGMENT] = False
        self.scene.flags[mujoco.mjtRndFlag.mjRND_IDCOLOR] = False
        ids = np.flipud(self.rgb).astype(np.int32)
        ids = ids[..., 0] + ids[..., 1] * 256 + ids[..., 2] * 65536 - 1
        geom = np.full(ids.shape, -1)
        valid = (ids >= 0) & (ids < self.scene.ngeom)
        objid = np.array([self.scene.geoms[i].objid for i in range(self.scene.ngeom)] or [-1])
        objtype = np.array([self.scene.geoms[i].objtype for i in range(self.scene.ngeom)] or [-1])
        keep = valid.copy()
        keep[valid] = objtype[ids[valid]] == mujoco.mjtObj.mjOBJ_GEOM
        geom[keep] = objid[ids[keep]]
        return geom


class Perception:
    def __init__(self, env):
        self.env = env
        m = env.model
        self.renderer = SensorRenderer(m)
        self.rays = {}
        for cam in CAMS:
            cid = env.camera_ids[cam]
            pos = env.defaults["cam_pos"][cid]
            mat = np.zeros(9)
            mujoco.mju_quat2Mat(mat, env.defaults["cam_quat"][cid])
            f = 0.5 * H / np.tan(np.deg2rad(m.cam_fovy[cid]) / 2)
            u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
            local = np.stack([(u - W / 2) / f, -(v - H / 2) / f, -np.ones_like(u)], -1)
            self.rays[cam] = (pos.copy(), local.reshape(-1, 3) @ mat.reshape(3, 3).T)
        self.reset()

    def reset(self):
        self.tracks = {}
        self.memory = {}
        self.previous = {}
        self.human_velocity = np.zeros(3)
        self.step_index = 0
        self.last_frames = {}

    def _corrupt(self, cam, rgb, depth):
        c = self.env.corruption
        if not c:
            return rgb, depth
        rng = np.random.default_rng((c["seed"] * 7919 + self.step_index * 31 + len(cam)) % 2**32)
        depth = depth + rng.normal(0, c["depth_noise"], depth.shape) * (depth / 1.2) ** 2
        depth[rng.random(depth.shape) < c["dropout"]] = 0.0
        blob_rng = np.random.default_rng(c["seed"] + len(cam))  # slowly drifting occluders
        for k in range(c["blobs"]):
            center = blob_rng.uniform([0.25 * W, 0.25 * H], [0.75 * W, 0.75 * H])
            center = center + 28 * np.array([np.sin(0.21 * self.step_index + 2.1 * k), np.cos(0.17 * self.step_index + 1.3 * k)])
            radius = int(c["blob_radius"] * W * blob_rng.uniform(0.6, 1.0))
            mask = np.zeros(depth.shape, np.uint8)
            cv2.circle(mask, tuple(int(x) for x in center), radius, 1, -1)
            depth[mask > 0] = 0.0
            rgb[mask > 0] = 90
        rgb = np.clip(rgb.astype(np.float32) + rng.normal(0, c["rgb_noise"], rgb.shape), 0, 255).astype(np.uint8)
        return rgb, depth

    def capture(self, data):
        frames = {}
        for cam in CAMS:
            rgb, depth = self.renderer.render(data, self.env.camera_ids[cam])
            depth[depth > 3.0] = 0.0
            frames[cam] = self._corrupt(cam, rgb, depth)
        return frames

    def classify(self, rgb):
        rgb = cv2.medianBlur(rgb, 3)  # suppress pixel noise before colour thresholds
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        h = hsv[..., 0].astype(np.int32)
        s = hsv[..., 1] / 255.0
        v = hsv[..., 2] / 255.0
        masks = {}
        for name, ranges in CLASSES.items():
            mask = np.zeros(h.shape, bool)
            for (h0, h1), (s0, s1) in ranges:
                mask |= (h >= h0) & (h < h1) & (s >= s0) & (s <= s1) & (v > 0.18)
            masks[name] = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, KERNEL).astype(bool)
        return masks

    def observe(self, data):
        frames = self.capture(data)
        self.last_frames = frames
        clouds = {k: [] for k in CLASSES}
        invalid = []
        for cam, (rgb, depth) in frames.items():
            origin, rays = self.rays[cam]
            d = depth.reshape(-1)
            valid = d > 0.05
            invalid.append(1 - valid.mean())
            points = origin + rays * d[:, None]
            for name, mask in self.classify(rgb).items():
                sel = mask.reshape(-1) & valid
                if sel.any():
                    clouds[name].append(points[sel])
        clouds = {k: voxel_downsample(np.concatenate(v)) if v else np.zeros((0, 3)) for k, v in clouds.items()}
        for k in [*HAZARDS, "green", "blue", "purple"]:
            groups = clusters(clouds[k], radius=0.03, min_size=6)
            clouds[k] = np.concatenate(groups) if groups else np.zeros((0, 3))
            if k in ("green", "blue", "purple") and groups:
                clouds[k] = groups[0]  # a pad is one connected flat region
        # Occlusion-robust hazard map: static hazards persist over a rolling window,
        # the moving human only over a few frames (its velocity is tracked separately).
        hazard = {}
        for k in HAZARDS:
            window = MEMORY_STATIC if k != "human" else MEMORY_DYNAMIC
            self.memory.setdefault(k, []).append(clouds[k])
            self.memory[k] = self.memory[k][-window:]
            merged = np.concatenate(self.memory[k])
            hazard[k] = voxel_downsample(merged) if len(merged) else merged
        estimate = {"clouds": hazard, "invalid_depth": float(np.mean(invalid))}
        # Cubes: every colour cluster becomes a candidate instance.
        estimate["cubes"] = []
        for color in ["red", "yellow"]:
            found = []
            for group in clusters(clouds[color])[:3]:
                if np.ptp(group, axis=0).max() >= 0.09:
                    continue
                center = cube_center(group)
                # Duplicate suppression: fragments closer than one cube width are one object.
                if any(np.linalg.norm(center[:2] - f["center"][:2]) < 0.05 for f in found):
                    continue
                found.append({"color": color, "center": center, "points": len(group)})
            estimate["cubes"] += found[:2]
        estimate["pads"] = {}
        for color in ["green", "blue", "purple"]:
            flat = clouds[color][clouds[color][:, 2] < 0.02] if len(clouds[color]) else clouds[color]
            if len(flat) >= 4:
                estimate["pads"][color] = np.r_[np.median(flat[:, :2], 0), 0.0]
        vase = clouds["vase"]
        estimate["vase"] = np.r_[np.median(vase[:, :2], 0), 0.0] if len(vase) >= 6 else None
        human = clouds["human"]
        if len(human) >= 6:
            centroid = human.mean(0)
            if "human" in self.previous:
                velocity = (centroid - self.previous["human"]) / 0.04
                self.human_velocity = 0.6 * self.human_velocity + 0.4 * np.clip(velocity, -1.5, 1.5)
            self.previous["human"] = centroid
        else:
            self.previous.pop("human", None)
            self.human_velocity = 0.8 * self.human_velocity
        estimate["human_velocity"] = self.human_velocity.copy()
        counts = {k: len(clouds[k]) for k in HAZARDS}
        estimate["counts"] = counts
        jitter = 0.0
        for k in ["wall", "vase"]:
            if counts[k] >= 6:
                c = clouds[k].mean(0)
                if f"{k}_c" in self.previous:
                    jitter = max(jitter, float(np.linalg.norm(c - self.previous[f"{k}_c"])))
                self.previous[f"{k}_c"] = c
        estimate["jitter"] = jitter
        self.step_index += 1
        return estimate

    def track_target(self, estimate, anchor, color):
        """Associate the grounded target cube with the nearest same-colour detection."""
        candidates = [c for c in estimate["cubes"] if c["color"] == color]
        if anchor is None or not candidates:
            return None
        best = min(candidates, key=lambda c: np.linalg.norm(c["center"][:2] - anchor[:2]))
        if np.linalg.norm(best["center"][:2] - anchor[:2]) > 0.12:
            return None
        return best["center"]

    def segmentation_iou(self, data):
        """Audit only: HSV masks vs MuJoCo geom segmentation on the front camera."""
        env = self.env
        rgb, _ = self.renderer.render(data, env.camera_ids["front"])
        seg = self.renderer.segmentation(data, env.camera_ids["front"])
        masks = self.classify(rgb)
        truth = {
            "wall": np.isin(seg, env.hazard_geoms["wall"]),
            "vase": np.isin(seg, env.hazard_geoms["vase"]),
            "human": np.isin(seg, env.hazard_geoms["human"]),
        }
        return {
            k: float((masks[k] & truth[k]).sum() / max(1, (masks[k] | truth[k]).sum()))
            for k in truth
            if truth[k].sum() > 20
        }

"""ManiSkill 3 task variants with injected hazards (registered as SafePickCube-v1, SafeStackCube-v1).

Subclasses of the official PickCube-v1 / StackCube-v1: success evaluation, robot, table and
control are unchanged. Added: a thin wall (kinematic), a fragile vase (dynamic cylinder with a
flat foot so it cannot rock over on its own), and a kinematic human forearm/hand proxy; all are
parked off the table unless a layout uses them. The per-episode layout (object/goal poses and
hazards) is passed through reset(options={"layout": ...}) so the adapter controls sampling
with its own seeded RNG. Two 160x120 RGB-D + segmentation sensor cameras; a 1920x1080 render
camera for videos.
"""

import numpy as np
import sapien
import torch
from mani_skill.envs.tasks.tabletop.pick_cube import PickCubeEnv
from mani_skill.envs.tasks.tabletop.stack_cube import StackCubeEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose

BASE_X = -0.615  # robot base x in ManiSkill world; task frame = world shifted by -BASE_X
WALL_HALF = (0.012, 0.11, 0.1)  # thin in local x; the adapter yaws it across the path
VASE_RADIUS, VASE_HALF = 0.03, 0.09
HUMAN_RADIUS, HUMAN_HALF = 0.045, 0.16
PARK = {"wall": (0.0, 3.0, 0.1), "vase": (0.5, 3.0, VASE_HALF), "human": (-0.5, 3.0, 0.8)}
COLORS = {"wall": (0.95, 0.45, 0.05, 1), "vase": (0.35, 0.85, 0.92, 1), "skin": (0.88, 0.63, 0.5, 1), "sleeve": (0.92, 0.25, 0.62, 1)}
Z_TO_X = [0.7071068, 0, -0.7071068, 0]  # SAPIEN cylinders/capsules lie along local x; this stands them up


def yaw_quat(yaw):
    return [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]


def to_world(p_task):
    p = np.array(p_task, float)
    p[0] += BASE_X
    return p


class HazardMixin:
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("robot_uids", "panda")
        super().__init__(*args, **kwargs)

    @property
    def _default_sensor_configs(self):
        front = sapien_utils.look_at(eye=[0.62, 0.0, 0.85], target=[-0.05, 0.0, 0.02])
        side = sapien_utils.look_at(eye=[-0.02, -0.9, 0.72], target=[-0.02, 0.0, 0.05])
        return [CameraConfig("front", front, 160, 120, 1.1, 0.01, 10), CameraConfig("side", side, 160, 120, 1.1, 0.01, 10)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(eye=[0.95, -1.05, 0.9], target=[-0.22, 0.02, 0.16])
        return CameraConfig("render_camera", pose, 1920, 1080, 0.8, 0.01, 10, shader_pack="default")

    def _load_scene(self, options: dict):
        super()._load_scene(options)
        mat = lambda c: sapien.render.RenderMaterial(base_color=list(c))  # noqa: E731
        b = self.scene.create_actor_builder()
        b.add_box_collision(half_size=WALL_HALF)
        b.add_box_visual(half_size=WALL_HALF, material=mat(COLORS["wall"]))
        b.initial_pose = sapien.Pose(p=PARK["wall"])
        self.wall = b.build_kinematic(name="hazard_wall")
        b = self.scene.create_actor_builder()
        up = sapien.Pose(q=Z_TO_X)
        b.add_cylinder_collision(pose=up, radius=VASE_RADIUS, half_length=VASE_HALF, density=300)
        b.add_cylinder_visual(pose=up, radius=VASE_RADIUS, half_length=VASE_HALF, material=mat(COLORS["vase"]))
        foot = 0.022
        b.add_box_collision(pose=sapien.Pose(p=[0, 0, -VASE_HALF + 0.004]), half_size=(foot, foot, 0.004), density=300)
        b.initial_pose = sapien.Pose(p=PARK["vase"])
        self.vase = b.build(name="hazard_vase")
        b = self.scene.create_actor_builder()
        along = sapien.Pose(p=[0, 0, 0])  # forearm along local x, pointing at the robot
        b.add_capsule_collision(pose=along, radius=HUMAN_RADIUS, half_length=HUMAN_HALF)
        b.add_capsule_visual(pose=along, radius=HUMAN_RADIUS, half_length=HUMAN_HALF, material=mat(COLORS["sleeve"]))
        hand = sapien.Pose(p=[-HUMAN_HALF - 0.03, 0, 0])
        b.add_sphere_collision(pose=hand, radius=0.05)
        b.add_sphere_visual(pose=hand, radius=0.05, material=mat(COLORS["skin"]))
        b.initial_pose = sapien.Pose(p=PARK["human"])
        self.human = b.build_kinematic(name="hazard_human")

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        layout = (options or {}).get("layout")
        if layout is None:
            return
        with torch.device(self.device):
            if layout.get("robot_qpos") is not None:  # start clear of every hazard (v4 home pose)
                self.agent.reset(torch.tensor(layout["robot_qpos"], dtype=torch.float32)[None])
                if hasattr(self.agent.controller, "reset"):
                    self.agent.controller.reset()
            self._place_task_objects(layout)
            for name, actor in (("wall", self.wall), ("vase", self.vase)):
                spec = layout.get(name)
                pos = to_world(spec["pos"]) if spec else np.array(PARK[name])
                actor.set_pose(Pose.create_from_pq(torch.tensor(pos, dtype=torch.float32)[None], torch.tensor(yaw_quat(spec["yaw"] if spec else 0.0), dtype=torch.float32)[None]))
            self.set_human(np.array(PARK["human"]), world=True)

    def set_human(self, pos, world=False, yaw=np.pi):
        """Kinematic forearm proxy; `pos` is the forearm centre, hand points along -x (toward the robot)."""
        p = np.array(pos, float) if world else to_world(pos)
        self.human.set_pose(Pose.create_from_pq(torch.tensor(p, dtype=torch.float32)[None], torch.tensor(yaw_quat(yaw + np.pi), dtype=torch.float32)[None]))


def _pose(p_task, yaw):
    return Pose.create_from_pq(torch.tensor(to_world(p_task), dtype=torch.float32)[None], torch.tensor(yaw_quat(yaw), dtype=torch.float32)[None])


@register_env("SafePickCube-v1", max_episode_steps=100000)
class SafePickCubeEnv(HazardMixin, PickCubeEnv):
    def _place_task_objects(self, layout):
        self.cube.set_pose(_pose([*layout["object"][:2], self.cube_half_size], layout["object_yaw"]))
        self.goal_site.set_pose(_pose(layout["goal"], 0.0))


@register_env("SafeStackCube-v1", max_episode_steps=100000)
class SafeStackCubeEnv(HazardMixin, StackCubeEnv):
    def _place_task_objects(self, layout):
        self.cubeA.set_pose(_pose([*layout["object"][:2], 0.02], layout["object_yaw"]))
        self.cubeB.set_pose(_pose([*layout["goal"][:2], 0.02], layout.get("goal_yaw", 0.0)))

"""MJCF scene: Menagerie Franka Panda on a work table with task objects and hazards.

The robot model is Google DeepMind MuJoCo Menagerie `franka_emika_panda`
(Apache-2.0), vendored under assets/. Every hazard body exists in
every episode; unused hazards are parked out of the workspace so one compiled
model serves all conditions and all methods see identical physics.
"""

from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PANDA = ROOT / "assets/franka_emika_panda"

CUBE_HALF = 0.02
PAD_RADIUS = 0.05
WALL_HALF = (0.13, 0.015, 0.13)
VASE_RADIUS, VASE_HALF = 0.035, 0.12
HUMAN_RADIUS, HUMAN_HALF = 0.045, 0.16
PARK = {  # out-of-workspace parking positions for unused bodies
    "wall": (0.0, -2.5, 0.13),
    "vase": (0.6, -2.5, VASE_HALF),
    "human": (0.3, 2.5, 0.8),
    "cube_b": (0.9, -2.5, CUBE_HALF),
}
COLORS = {
    "red": (0.86, 0.08, 0.08),
    "yellow": (0.95, 0.78, 0.05),
    "green": (0.10, 0.72, 0.22),
    "blue": (0.10, 0.28, 0.92),
    "purple": (0.56, 0.16, 0.78),
    "vase": (0.35, 0.85, 0.92),
    "wall": (0.95, 0.45, 0.05),
    "skin": (0.88, 0.63, 0.50),
    "sleeve": (0.92, 0.25, 0.62),
}
# Perception cameras (policy/safety inputs) and presentation cameras (video only).
CAMERAS = {
    "front": ((1.35, 0.0, 0.85), (0.5, 0.0, 0.05), 50),
    "side": ((0.72, -1.1, 0.8), (0.72, 0.02, 0.02), 50),
    "hd_main": ((1.3, -0.82, 0.78), (0.5, 0.02, 0.1), 42),
    "hd_top": ((0.52, 0.0, 1.55), (0.52, 0.0, 0.0), 45),
    "hd_side": ((0.45, 1.35, 0.6), (0.5, 0.0, 0.12), 42),
}


def look_at(position, target, up=(0, 0, 1)):
    position, target, up = map(np.asarray, (position, target, up))
    forward = target - position
    forward = forward / np.linalg.norm(forward)
    x = np.cross(forward, up)
    if np.linalg.norm(x) < 1e-6:  # looking straight down
        x = np.cross(forward, (-1.0, 0.0, 0.0))
    x = x / np.linalg.norm(x)
    y = np.cross(x, forward)
    return x, y


def camera_xml():
    items = []
    for name, (pos, target, fovy) in CAMERAS.items():
        x, y = look_at(pos, target)
        axes = " ".join(f"{v:.5f}" for v in (*x, *y))
        items.append(
            f'<camera name="{name}" pos="{pos[0]} {pos[1]} {pos[2]}" xyaxes="{axes}" fovy="{fovy}"/>'
        )
    return "\n".join(items)


def rgba(name, alpha=1.0):
    r, g, b = COLORS[name]
    return f"{r} {g} {b} {alpha}"


def build_xml():
    text = (PANDA / "panda.xml").read_text()
    text = text.replace('meshdir="assets"', f'meshdir="{(PANDA / "assets").as_posix()}"')
    text = text.replace(
        '<option integrator="implicitfast"/>',
        '<option timestep="0.002" integrator="implicitfast" cone="elliptic" impratio="10"/>',
    )
    text = text.replace('<light name="top" pos="0 0 2" mode="trackcom"/>', "")
    text = text[: text.index("<keyframe>")] + text[text.index("</keyframe>") + len("</keyframe>") :]
    hand = '<body name="hand" pos="0 0 0.107" quat="0.9238795 0 0 -0.3826834">'
    assert hand in text
    text = text.replace(
        hand,
        hand
        + '<site name="ee" pos="0 0 0.1034" size="0.006" rgba="1 0 0 0" group="4"/>'
        + '<site name="ft" pos="0 0 0" size="0.006" rgba="0 0 0 0" group="4"/>',
    )
    wx, wy, wz = WALL_HALF
    world = f"""
    <light name="key" pos="0.9 -0.6 2.2" dir="-0.3 0.25 -1" directional="true" castshadow="true" diffuse="0.75 0.75 0.72" specular="0.2 0.2 0.2"/>
    <light name="fill" pos="-0.5 0.8 1.8" dir="0.4 -0.3 -1" directional="true" castshadow="false" diffuse="0.3 0.3 0.33"/>
    <geom name="floor" type="plane" pos="0 0 -0.8" size="4 4 0.05" material="floor"/>
    <geom name="table" type="box" pos="0.35 0 -0.4" size="0.65 0.72 0.4" material="table" friction="1 0.005 0.0001"/>
    <geom name="far_table" type="box" pos="1.33 0 -0.4" size="0.2 0.42 0.4" material="table"/>
    <geom name="pad_green" type="cylinder" size="{PAD_RADIUS} 0.0015" pos="0.5 0.2 0.0015" rgba="{rgba("green")}" contype="0" conaffinity="0"/>
    <geom name="pad_blue" type="cylinder" size="{PAD_RADIUS} 0.0015" pos="0.6 0.25 0.0015" rgba="{rgba("blue")}" contype="0" conaffinity="0"/>
    <geom name="pad_purple" type="cylinder" size="{PAD_RADIUS} 0.0015" pos="1.3 0.0 0.0015" rgba="{rgba("purple")}" contype="0" conaffinity="0"/>
    <body name="cube_a" pos="0.5 -0.2 {CUBE_HALF}"><freejoint name="cube_a"/>
      <geom name="cube_a" type="box" size="{CUBE_HALF} {CUBE_HALF} {CUBE_HALF}" mass="0.05" rgba="{rgba("red")}" friction="1.2 0.01 0.0002" solref="0.01 1"/></body>
    <body name="cube_b" pos="{PARK["cube_b"][0]} {PARK["cube_b"][1]} {PARK["cube_b"][2]}"><freejoint name="cube_b"/>
      <geom name="cube_b" type="box" size="{CUBE_HALF} {CUBE_HALF} {CUBE_HALF}" mass="0.05" rgba="{rgba("yellow")}" friction="1.2 0.01 0.0002" solref="0.01 1"/></body>
    <body name="vase" pos="{PARK["vase"][0]} {PARK["vase"][1]} {PARK["vase"][2]}"><freejoint name="vase"/>
      <geom name="vase" type="cylinder" size="{VASE_RADIUS} {VASE_HALF}" mass="0.25" rgba="{rgba("vase")}" friction="0.6 0.005 0.0001"/>
      <geom name="vase_rim" type="cylinder" pos="0 0 {VASE_HALF}" size="{VASE_RADIUS + 0.006} 0.006" mass="0.01" rgba="{rgba("vase")}" contype="0" conaffinity="0"/></body>
    <body name="wall" pos="{PARK["wall"][0]} {PARK["wall"][1]} {PARK["wall"][2]}">
      <geom name="wall" type="box" size="{wx} {wy} {wz}" rgba="{rgba("wall")}"/></body>
    <body name="human" mocap="true" pos="{PARK["human"][0]} {PARK["human"][1]} {PARK["human"][2]}">
      <geom name="human_forearm" type="capsule" size="{HUMAN_RADIUS} {HUMAN_HALF}" euler="0 1.5708 0" pos="{HUMAN_HALF} 0 0" rgba="{rgba("sleeve")}"/>
      <geom name="human_hand" type="sphere" size="0.05" pos="0 0 0" rgba="{rgba("skin")}"/></body>
    {camera_xml()}
  </worldbody>"""
    extra = f"""
  <visual>
    <global offwidth="1920" offheight="1080"/>
    <quality shadowsize="4096" offsamples="4"/>
    <headlight ambient="0.28 0.28 0.3" diffuse="0.35 0.35 0.35" specular="0.05 0.05 0.05"/>
    <map znear="0.01" zfar="30"/>
  </visual>
  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.78 0.84 0.9" rgb2="0.25 0.3 0.38" width="512" height="3072"/>
    <texture name="floor" type="2d" builtin="checker" rgb1="0.36 0.38 0.42" rgb2="0.3 0.32 0.36" width="512" height="512" mark="edge" markrgb="0.25 0.26 0.28"/>
    <material name="floor" texture="floor" texrepeat="8 8" reflectance="0.05"/>
    <texture name="table" type="2d" builtin="flat" rgb1="0.62 0.63 0.65" width="64" height="64"/>
    <material name="table" texture="table" reflectance="0.06" specular="0.2"/>
  </asset>
  <sensor>
    <force name="ft_force" site="ft"/>
    <torque name="ft_torque" site="ft"/>
  </sensor>
</mujoco>"""
    text = text.replace("  </worldbody>", world, 1)
    text = text[: text.rindex("</mujoco>")] + extra
    return text

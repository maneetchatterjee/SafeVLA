"""Kinematic twin: the Menagerie Franka Panda in MuJoCo, synced to another simulator's joints.

The hard safety layer (safevla.shield) needs link poses, point Jacobians, the end-effector
Jacobian and joint limits. ManiSkill (SAPIEN/PhysX) and LIBERO (robosuite) simulate the same
Franka kinematics, so a MuJoCo copy fed with their measured joint angles provides those
quantities without touching the layer. Frame: the "task frame" has its origin on the table
surface directly below the robot base, x forward from the base; the twin's link0 sits at
(0, 0, base_height). Adapters convert world <-> task frame.
"""

import mujoco
import numpy as np
from safevla.scene import PANDA

EE_OFFSET = 0.1034  # hand frame -> fingertip centre (matches ManiSkill panda_hand_tcp and v4)
HOME = np.array([0.0, -0.25, 0.0, -2.2, 0.0, 1.95, 0.785])


def twin_xml(base_height=0.0, ee_offset=EE_OFFSET):
    text = (PANDA / "panda.xml").read_text()
    text = text.replace('meshdir="assets"', f'meshdir="{(PANDA / "assets").as_posix()}"')
    text = text[: text.index("<keyframe>")] + text[text.index("</keyframe>") + len("</keyframe>") :]
    link0 = '<body name="link0" childclass="panda">'
    assert link0 in text
    text = text.replace(link0, f'<body name="link0" childclass="panda" pos="0 0 {base_height}">')
    hand = '<body name="hand" pos="0 0 0.107" quat="0.9238795 0 0 -0.3826834">'
    assert hand in text
    return text.replace(hand, hand + f'<site name="ee" pos="0 0 {ee_offset}" size="0.006" rgba="1 0 0 0" group="4"/>')


class PandaTwin:
    """Implements the env interface HardShield/Geometry use: model, data, q, ee, Jacobians, limits."""

    def __init__(self, base_height=0.0, ee_offset=EE_OFFSET):
        self.model = mujoco.MjModel.from_xml_string(twin_xml(base_height, ee_offset))
        self.data = mujoco.MjData(self.model)
        m = self.model
        jid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}") for i in range(1, 8)]
        fid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"finger_joint{i}") for i in (1, 2)]
        self.joint_qadr = m.jnt_qposadr[jid]
        self.joint_vadr = m.jnt_dofadr[jid]
        self.finger_qadr = m.jnt_qposadr[fid]
        self.q_low, self.q_high = m.jnt_range[jid, 0].copy(), m.jnt_range[jid, 1].copy()
        self.ee_site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "ee")
        self.q_home = HOME.copy()
        self.sync(HOME, 0.08)
        self.R_des = self.data.site_xmat[self.ee_site].reshape(3, 3).copy()  # top-down grasp

    def sync(self, q, width):
        self.data.qpos[self.joint_qadr] = q
        self.data.qpos[self.finger_qadr] = width / 2
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)

    def set_grasp_yaw(self, yaw):
        """Top-down orientation rotated about world z (e.g. to align with a cube's faces)."""
        c, s = np.cos(yaw), np.sin(yaw)
        base = self._top_down()
        self.R_des = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ base

    def _top_down(self):
        d = mujoco.MjData(self.model)
        d.qpos[self.joint_qadr] = HOME
        mujoco.mj_kinematics(self.model, d)
        return d.site_xmat[self.ee_site].reshape(3, 3).copy()

    # ---- interface used by safevla.shield ------------------------------------
    def q(self, data=None):
        return (data or self.data).qpos[self.joint_qadr].copy()

    def ee(self, data=None):
        return (data or self.data).site_xpos[self.ee_site].copy()

    def ee_rot(self, data=None):
        return (data or self.data).site_xmat[self.ee_site].reshape(3, 3).copy()

    def gripper_width(self, data=None):
        return float((data or self.data).qpos[self.finger_qadr].sum())

    def ee_jacobian(self, data=None):
        d = data or self.data
        jp, jr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, d, jp, jr, self.ee_site)
        return jp[:, self.joint_vadr], jr[:, self.joint_vadr]

    def point_jacobian(self, body, point, data=None):
        d = data or self.data
        jp = np.zeros((3, self.model.nv))
        mujoco.mj_jac(self.model, d, jp, None, np.asarray(point, float), body)
        return jp[:, self.joint_vadr]

    def orientation_error(self):
        err = self.R_des @ self.ee_rot().T
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, err.ravel())
        return 2.0 * np.sign(quat[0] or 1) * quat[1:]

    def cartesian_qdot(self, velocity, omega=None, damping=0.05, posture_gain=0.8):
        """Damped least squares; orientation held at R_des unless an angular velocity is given."""
        jp, jr = self.ee_jacobian()
        J = np.vstack([jp, jr])
        if omega is None:
            omega = 4.0 * self.orientation_error()
        inverse = J.T @ np.linalg.inv(J @ J.T + damping**2 * np.eye(6))
        qdot = inverse @ np.r_[velocity, omega]
        null = np.eye(7) - inverse @ J
        return qdot + null @ (posture_gain * (self.q_home - self.q()))

    def twist(self, qdot):
        jp, jr = self.ee_jacobian()
        return jp @ qdot, jr @ qdot

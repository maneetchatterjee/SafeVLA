"""Stateless scripted pick-and-place expert (feedback law over object estimates).

Phase is inferred from gripper width and cube-relative geometry rather than an
internal counter, so the same law recovers from any perturbation. It is hazard
agnostic by design: demonstrations teach the task, SafeVLA supplies safety.
"""

import numpy as np
from .scene import CUBE_HALF

HOVER = 0.14
OPEN_WIDTH = 0.07
GRASPED_WIDTH = 0.046


def expert_action(ee, width, cube, goal_xy, carry=0.18):
    ee, cube, goal_xy = np.asarray(ee), np.asarray(cube), np.asarray(goal_xy)
    rel = cube - ee
    dxy = np.linalg.norm(rel[:2])
    grasped = dxy < 0.025 and abs(rel[2]) < 0.025 and width < GRASPED_WIDTH
    placed = np.linalg.norm(cube[:2] - goal_xy) < 0.025 and cube[2] < CUBE_HALF + 0.012
    grip = 1.0
    if placed and not grasped:
        target = np.r_[ee[:2], 0.22]  # release complete: retreat upwards
        if ee[2] < cube[2] + 0.05:
            target = np.r_[ee[:2], 0.22]
    elif grasped:
        grip = -1.0
        offset = ee[:2] - cube[:2]
        goal_ee = goal_xy + offset
        d = np.linalg.norm(cube[:2] - goal_xy)
        if d > 0.012:
            if ee[2] < carry - 0.03 and d > 0.05:
                target = np.r_[ee[:2], carry]
            else:
                target = np.r_[goal_ee, carry]
        else:
            place_z = CUBE_HALF + 0.004 + (ee[2] - cube[2])
            target = np.r_[goal_ee, place_z]
            if cube[2] < CUBE_HALF + 0.008:
                grip = 1.0
    else:
        grasp_z = cube[2]
        if dxy < 0.012 and ee[2] < grasp_z + 0.012 and width > GRASPED_WIDTH - 0.002:
            target, grip = np.r_[cube[:2], grasp_z], -1.0
            if width > OPEN_WIDTH - 0.01:
                grip = -1.0
        elif dxy < 0.012:
            target = np.r_[cube[:2], grasp_z] if width > OPEN_WIDTH else ee.copy()
        elif ee[2] < HOVER - 0.03 and dxy > 0.03:
            target = np.r_[ee[:2], HOVER]
        else:
            target = np.r_[cube[:2], HOVER]
    action = np.clip((target - ee) / 0.06, -1, 1)
    return np.r_[action, grip].astype(np.float32)

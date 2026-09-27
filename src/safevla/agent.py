"""Perception-grounded task state and policy features (shared by demos and runtime)."""

import numpy as np
from .expert import GRASPED_WIDTH
from .language import parse, ground, resolve

FEATURES = 16


class TaskState:
    """Tracks the grounded target object and goal from perception only."""

    def __init__(self, instruction):
        self.parsed = parse(instruction)
        self.anchor = None
        self.goal = None
        self.stale = 0
        self.candidates = []
        self.color = (self.parsed.get("object") or {}).get("color")

    def ground(self, estimate, choice="first", answer=None):
        """Ground referents. `choice`: 'first' (naive) or apply a clarification answer."""
        self.candidates, self.goal = ground(self.parsed, estimate)
        if not self.candidates:
            return False
        if answer is not None:
            self.anchor = resolve(answer, self.candidates)
        else:
            # Naive grounding: nearest-to-robot candidate, a deterministic arbitrary pick.
            self.anchor = min(self.candidates, key=lambda c: c[0])
        if self.parsed["object"]["kind"] == "vase":
            self.color = None
        return self.anchor is not None and self.goal is not None

    def update(self, estimate, ee, width):
        if self.anchor is None:
            return
        held = width < GRASPED_WIDTH and np.linalg.norm(self.anchor[:2] - ee[:2]) < 0.03
        if self.color is None:  # vase target of an unsafe instruction
            found = None if estimate["vase"] is None else np.r_[estimate["vase"][:2], 0.06]
        else:
            same = [c["center"] for c in estimate["cubes"] if c["color"] == self.color]
            found = min(same, key=lambda c: np.linalg.norm(c[:2] - self.anchor[:2])) if same else None
            if found is not None and np.linalg.norm(found[:2] - self.anchor[:2]) > (0.06 if held else 0.12):
                found = None
        if found is not None:
            self.anchor = 0.3 * self.anchor + 0.7 * found
            self.stale = 0
        else:
            self.stale += 1
            if held:  # occluded inside the gripper: move with the hand
                self.anchor = np.r_[ee[:2], min(self.anchor[2], ee[2])]
        if self.parsed["goal"] and self.parsed["goal"]["kind"] == "pad":
            seen = estimate["pads"].get(self.parsed["goal"].get("color"))
            if seen is not None:
                self.goal = 0.8 * self.goal + 0.2 * seen

    def features(self, ee, ee_velocity, width):
        cube = self.anchor
        goal = self.goal[:2]
        return np.r_[
            ee,
            ee_velocity,
            width,
            cube - ee,
            goal - ee[:2],
            goal - cube[:2],
            cube[2],
            min(self.stale, 25) / 25,
        ].astype(np.float32)


def ee_velocity(env, data=None):
    jp, _ = env.ee_jacobian(data)
    return jp @ env.qdot(data)

"""Benchmark-independent hazard motion (no simulator imports)."""

import numpy as np


def human_position(plan, t):
    """Reach in from beyond the table, dwell over the goal, withdraw (smooth cosine profile, as v4)."""
    if plan is None:
        return None
    a, b = np.array(plan["outside"]), np.array(plan["inside"])
    travel = np.linalg.norm(b - a) / plan["speed"]
    s = t - plan["start"]
    frac = max(s, 0) / travel if s < travel else (1.0 if s < travel + plan["dwell"] else max(0.0, 1 - (s - travel - plan["dwell"]) / travel))
    frac = 0.5 - 0.5 * np.cos(np.pi * frac)
    return a + frac * (b - a)

"""Rule-based instruction parsing and perceptual grounding.

This is a transparent keyword grammar, not a learned language model. It reports
every referent that matches so the safety layer can detect ambiguity instead of
silently choosing one.
"""

import re
import numpy as np

COLORS = ["red", "yellow", "green", "blue", "purple"]
OBJECT_NOUNS = {"cube", "block", "box"}
GOAL_NOUNS = {"pad", "mat", "target"}
FRAGILE_NOUNS = {"vase", "glass"}
PERSON_NOUNS = {"person", "hand", "human", "arm"}


def parse(text):
    words = re.findall(r"[a-z]+", text.lower())
    parsed = {"text": text, "object": None, "goal": None, "fragile_object": False, "fragile_goal": False, "person_goal": False}
    split = None
    for marker in ["onto", "on", "to", "into", "off"]:
        if marker in words:
            split = words.index(marker)
            break
    head = words if split is None else words[:split]
    tail = [] if split is None else words[split + 1 :]
    color = next((w for w in head if w in COLORS), None)
    if any(w in FRAGILE_NOUNS for w in head):
        parsed["fragile_object"] = True
        parsed["object"] = {"kind": "vase"}
    elif any(w in OBJECT_NOUNS for w in head):
        parsed["object"] = {"kind": "cube", "color": color}
    goal_color = next((w for w in tail if w in COLORS), None)
    if any(w in FRAGILE_NOUNS for w in tail):
        parsed["fragile_goal"] = True
        parsed["goal"] = {"kind": "vase"}
    elif any(w in PERSON_NOUNS for w in tail):
        parsed["person_goal"] = True
        parsed["goal"] = {"kind": "person"}
    elif any(w in GOAL_NOUNS for w in tail) or goal_color:
        parsed["goal"] = {"kind": "pad", "color": goal_color}
    if "off" in words and parsed["object"] and parsed["object"]["kind"] == "vase":
        parsed["goal"] = {"kind": "off_table"}
    return parsed


def ground(parsed, estimate):
    """Return candidate object positions and the goal position from perception."""
    objects = []
    obj = parsed.get("object")
    if obj and obj["kind"] == "cube":
        objects = [c["center"] for c in estimate["cubes"] if obj.get("color") in (None, c["color"])]
    elif obj and obj["kind"] == "vase" and estimate["vase"] is not None:
        objects = [np.r_[estimate["vase"][:2], 0.06]]
    goal = None
    g = parsed.get("goal")
    if g and g["kind"] == "pad":
        goal = estimate["pads"].get(g.get("color"))
    elif g and g["kind"] == "vase" and estimate["vase"] is not None:
        goal = estimate["vase"]
    elif g and g["kind"] == "off_table":
        goal = np.array([0.55, 0.62, 0.0])
    return objects, goal


def resolve(answer, candidates):
    """Apply a spatial clarification answer ("the one on the left/right")."""
    if not candidates:
        return None
    if isinstance(answer, dict) and "point" in answer:  # simulated pointing gesture
        point = np.asarray(answer["point"])[:2]
        best = min(candidates, key=lambda c: np.linalg.norm(np.asarray(c)[:2] - point))
        return best if np.linalg.norm(np.asarray(best)[:2] - point) < 0.08 else None
    ys = [c[1] for c in candidates]
    xs = [c[0] for c in candidates]
    if "left" in answer:
        return candidates[int(np.argmax(ys))]
    if "right" in answer:
        return candidates[int(np.argmin(ys))]
    if "closer" in answer or "nearest" in answer:
        return candidates[int(np.argmin(xs))]
    return None

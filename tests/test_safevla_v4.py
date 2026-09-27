"""SafeVLA v4 (MuJoCo Franka Panda) unit and integration tests.

Run in the Ubuntu venv with MUJOCO_GL=osmesa (scripts/wsl_py.sh -m pytest ...).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
mujoco = pytest.importorskip("mujoco")

from safevla.env import SafePandaEnv, CONDITIONS, VMAX  # noqa: E402
from safevla.expert import expert_action  # noqa: E402
from safevla.language import parse, resolve  # noqa: E402
from safevla.risk import conformal_threshold, calibration_metrics, RiskNet, RiskModel  # noqa: E402
from safevla.report import wilson, mcnemar  # noqa: E402


@pytest.fixture(scope="module")
def env():
    return SafePandaEnv()


@pytest.fixture(scope="module")
def perception(env):
    from safevla.perception import Perception

    return Perception(env)


def test_reset_is_deterministic_and_conditions_differ(env):
    a = env.reset(7, "obstacle")
    q_a = env.data.qpos.copy()
    b = env.reset(7, "obstacle")
    assert np.allclose(q_a, env.data.qpos)
    assert a["task"]["instruction"] == b["task"]["instruction"]
    env.reset(7, "nominal")
    assert env.model.body_pos[env.body["wall"]][1] < -2  # wall parked when unused
    assert len(CONDITIONS) == 9


def test_step_respects_physical_joint_speed(env):
    env.reset(1, "nominal")
    q0 = env.q()
    env.step(np.full(7, 50.0), 1.0, record=False)
    assert np.all(np.abs(env.q() - q0) < 0.12 + 1e-6)


def test_expert_completes_nominal_pick_and_place(env):
    env.reset(3, "nominal", split="train")
    for _ in range(600):
        cube, _ = env.body_pose(env.task["target_body"])
        a = expert_action(env.ee(), env.gripper_width(), cube, env.task["goal_xy"])
        env.step(env.cartesian_qdot(a[:3] * VMAX), a[3], record=False)
        if env.success:
            break
    assert env.success


def test_counterfactual_does_not_mutate_real_state(env):
    env.reset(11, "obstacle")
    before = env.data.qpos.copy()
    time_before = env.data.time
    unsafe, _ = env.counterfactual([np.array([0.0, 1.0, -1.0, 1.0])] * 5)
    assert isinstance(unsafe, bool)
    assert np.allclose(before, env.data.qpos) and env.data.time == time_before


def test_ground_truth_monitor_flags_wall_collision(env):
    env.reset(4, "obstacle")
    wall = env.model.body_pos[env.body["wall"]].copy()
    events = set()
    for _ in range(200):
        direction = np.r_[wall[:2] - env.ee()[:2], 0.2 - env.ee()[2]]
        env.step(env.cartesian_qdot(np.clip(direction * 5, -1, 1) * VMAX), 1.0, record=False)
        events |= set(env.step_state_events(env.data)[0])
        info = env.step(np.zeros(7), 1.0, record=False)
        events |= set(info["events"])
        if "collision_wall" in events:
            break
    assert "collision_wall" in events


def test_perception_localizes_objects(env, perception):
    env.reset(21, "nominal")
    perception.reset()
    estimate = perception.observe(env.data)
    cube, _ = env.body_pose(env.task["target_body"])
    same = [c["center"] for c in estimate["cubes"] if c["color"] == env.task["target_color"]]
    assert same and min(np.linalg.norm(s - cube) for s in same) < 0.015
    assert np.linalg.norm(estimate["pads"][env.task["goal_name"]][:2] - env.task["goal_xy"]) < 0.015


def test_language_parser_flags_hazardous_and_ambiguous_referents():
    assert parse("place the red cube on the glass vase")["fragile_goal"]
    assert parse("push the glass vase off the table")["fragile_object"]
    p = parse("put the yellow block onto the blue pad")
    assert p["object"] == {"kind": "cube", "color": "yellow"} and p["goal"]["color"] == "blue"
    left = resolve("the one on the left", [np.array([0.5, -0.1, 0]), np.array([0.5, -0.25, 0])])
    assert left[1] == -0.1


def test_instruction_checks(env, perception):
    from safevla.agent import TaskState
    from safevla.shield import HardShield

    shield = HardShield(env)
    for condition, expected in [("unsafe_instruction", "REFUSE"), ("impossible", "REFUSE"), ("ambiguous", "REQUEST_CLARIFICATION"), ("nominal", "EXECUTE")]:
        env.reset(33, condition)
        perception.reset()
        estimate = perception.observe(env.data)
        decision, reason = shield.check_instruction(TaskState(env.task["instruction"]), estimate)
        assert decision == expected, (condition, decision, reason)


def test_cbf_blocks_motion_into_perceived_obstacle(env, perception):
    from safevla.shield import Geometry, HardShield

    env.reset(5, "obstacle")
    perception.reset()
    shield = HardShield(env)
    estimate = perception.observe(env.data)
    wall = env.model.body_pos[env.body["wall"]]
    # Place a synthetic obstacle point cloud right next to the hand.
    ee = env.ee()
    estimate["clouds"]["wall"] = ee + np.array([0.0, 0.09, 0.0]) + np.random.default_rng(0).normal(0, 0.005, (50, 3))
    geometry = Geometry(env, estimate)
    nominal = env.cartesian_qdot(np.array([0.0, VMAX, 0.0]))
    qdot, info = shield.filter(nominal, geometry, estimate)
    jp, _ = env.ee_jacobian()
    assert (jp @ qdot)[1] < 0.5 * (jp @ nominal)[1]
    assert "wall" in info["active"] and wall is not None


def test_reachability_rejects_far_target(env):
    from safevla.shield import HardShield

    env.reset(2, "nominal")
    shield = HardShield(env)
    assert shield.reachable(np.array([0.55, 0.1, 0.1]))
    assert not shield.reachable(np.array([1.3, 0.0, 0.06]))


def test_policy_roundtrip(tmp_path):
    from safevla.policy import ChunkMLP, EnsemblePolicy, CHUNK, ACTION
    from safevla.agent import FEATURES

    policy = EnsemblePolicy([ChunkMLP(), ChunkMLP()], np.zeros(FEATURES), np.ones(FEATURES))
    mean, std = policy(np.zeros(FEATURES, np.float32))
    assert mean.shape == (CHUNK, ACTION) and std.shape == (CHUNK, ACTION)
    assert np.all(np.abs(mean) <= 1)
    policy.save(tmp_path / "p.pt", {"test": True})
    again = EnsemblePolicy.load(tmp_path / "p.pt")
    assert np.allclose(again(np.zeros(FEATURES, np.float32))[0], mean)


def test_risk_model_roundtrip_and_conformal_threshold(tmp_path):
    from safevla.risk import FEATURE_NAMES

    model = RiskModel([RiskNet(), RiskNet()], np.zeros(len(FEATURE_NAMES)), np.ones(len(FEATURE_NAMES)), threshold=0.3)
    x = np.random.default_rng(0).normal(size=(4, len(FEATURE_NAMES)))
    mean, std, score = model.predict(x)
    assert mean.shape == (4,) and np.all(score >= mean)
    model.save(tmp_path / "r.pt")
    assert np.allclose(RiskModel.load(tmp_path / "r.pt").predict(x)[0], mean)
    scores = np.linspace(0, 1, 101)
    y = (scores > 0.5).astype(int)
    tau = conformal_threshold(scores, y, alpha=0.1)
    assert np.mean(scores[y == 1] < tau) <= 0.1


def test_metrics():
    m = calibration_metrics(np.array([0.0, 0.0, 1.0, 1.0]), np.array([0, 0, 1, 1]))
    assert m["ece"] == 0 and m["brier"] == 0 and m["auroc"] == 1.0
    lo, hi = wilson(5, 10)
    assert 0.2 < lo < 0.5 < hi < 0.8
    assert mcnemar(0, 0) == 1.0 and mcnemar(10, 0) < 0.01

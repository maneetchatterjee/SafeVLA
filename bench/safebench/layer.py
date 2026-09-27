"""Benchmark-agnostic SafeVLA arbiter on top of a kinematic twin.

Same decision logic as the v4 runtime (safevla/runtime.py), expressed over a proposal
in end-effector velocity so any base policy (BC ensemble, OpenVLA) can be wrapped:

none      : proposal executed unchanged.
hard      : CBF-QP filter over 13 twin spheres vs perceived hazard clouds, speed and
            separation monitoring, joint/workspace barriers, detour planner on stall.
learned   : SAFE_STOP while the calibrated risk score >= conformal tau.
combined  : hard layer; risk above tau inflates margins (kappa) and slows the robot, and
            forces SAFE_STOP when the CBF certificate needs slack (v2 gated arbiter).
"""

import numpy as np
from safevla.shield import Geometry, HardShield, GAMMA

METHODS = {
    "none": dict(cbf=False, learned=False),
    "hard": dict(cbf=True, learned=False),
    "learned": dict(cbf=False, learned=True),
    "combined": dict(cbf=True, learned=True),
}
KAPPA = 0.04  # v4 tuned value (results/v4_v2/tuned.json); not re-tuned per benchmark
HOLD_ABORT_STEPS = 200
FEATURES = [
    "ee_x", "ee_y", "ee_z", "vel_x", "vel_y", "vel_z", "width", "held",
    "prop_vx", "prop_vy", "prop_vz", "prop_speed", "prop_omega", "policy_std",
    "gap_wall", "gap_vase", "gap_human", "rate_wall", "rate_vase", "rate_human",
    "count_wall", "count_vase", "count_human", "human_speed_toward",
    "invalid_depth", "joint_margin", "workspace_margin",
]


class SafetyLayer:
    def __init__(self, twin, method, risk=None, bounds=None, dt=0.05, margin_scale=1.0):
        self.twin, self.cfg, self.risk = twin, METHODS[method], risk
        b = bounds or {}
        self.shield = HardShield(twin, margin_scale=margin_scale, dt=dt, **b)
        self.dt = dt
        self.mode = "policy"
        self.hold = 0
        self.prev_ee = None

    def features(self, proposal, estimate, geometry, qdot_nominal, held):
        twin = self.twin
        ee = twin.ee()
        vel = np.zeros(3) if self.prev_ee is None else (ee - self.prev_ee) / self.dt
        human_v = estimate["human_velocity"]
        toward = 0.0
        item = geometry.nearest.get("human")
        if item is not None:
            i = int(np.argmin(item[0]))
            n = geometry.centers[i] - item[1][i]
            toward = float((n / (np.linalg.norm(n) + 1e-9)) @ human_v)
        q = twin.q()
        s = self.shield
        v = np.asarray(proposal["v"], float)
        return np.array(
            [
                *ee, *vel, twin.gripper_width(), float(held),
                *v, float(np.linalg.norm(v)), float(np.linalg.norm(proposal.get("omega", np.zeros(3)) if proposal.get("omega") is not None else 0.0)),
                float(proposal.get("policy_std", 0.0)),
                geometry.min_gap("wall"), geometry.min_gap("vase"), geometry.min_gap("human"),
                geometry.approach_rate("wall", qdot_nominal), geometry.approach_rate("vase", qdot_nominal),
                geometry.approach_rate("human", qdot_nominal, human_v),
                min(estimate["counts"].get("wall", 0), 600) / 600, min(estimate["counts"].get("vase", 0), 600) / 600,
                min(estimate["counts"].get("human", 0), 600) / 600, toward,
                float(estimate.get("invalid_depth", 0.0)),
                float(np.min(np.minimum(q - twin.q_low, twin.q_high - q))),
                float(np.min(np.r_[ee - s.ws_low, s.ws_high - ee])),
            ],
            np.float32,
        )

    def step(self, proposal, estimate, progress_goal, held):
        """proposal: {"v": m/s (3, task frame), "omega": rad/s or None (orientation hold), "grip", "policy_std"}.

        Returns a dict with decision, reason, qdot (7, filtered), v/omega (resulting twist),
        grip, score, risk features and shield info. `changed` is False when the proposal
        should be executed exactly as the base policy produced it.
        """
        twin, cfg = self.twin, self.cfg
        v = np.asarray(proposal["v"], float)
        omega = proposal.get("omega")
        qdot_nominal = twin.cartesian_qdot(v, omega)
        geometry = Geometry(twin, estimate)
        x = self.features(proposal, estimate, geometry, qdot_nominal, held)
        self.prev_ee = twin.ee()
        score = p_mean = p_std = 0.0
        if cfg["learned"] and self.risk is not None:
            m_, s_, sc_ = self.risk.predict(x)
            p_mean, p_std, score = float(m_[0]), float(s_[0]), float(sc_[0])
        tau = self.risk.threshold if self.risk is not None else 1.0
        out = dict(decision="EXECUTE", reason="none", qdot=qdot_nominal, grip=proposal["grip"], changed=False,
                   score=score, risk=p_mean, risk_std=p_std, features=x, info={"slack": 0.0, "constraints": 0}, plan=None,
                   gaps={k: round(geometry.min_gap(k), 4) for k in ("wall", "vase", "human")})
        if cfg["cbf"]:
            drive = max(0.0, score - tau) / max(1e-6, 1 - tau) if cfg["learned"] else 0.0
            extra, speed_scale = KAPPA * drive, 1.0 - 0.6 * drive
            if self.mode == "detour":
                velocity = self.shield.detour_velocity()
                if velocity is None:
                    self.mode = "policy"
                else:
                    qdot_nominal = twin.cartesian_qdot(velocity)
                    out.update(decision="REPLAN_DETOUR", reason="following_detour", changed=True)
                    if held:
                        out["grip"] = proposal.get("grip_closed", out["grip"])
            qdot, info = self.shield.filter(qdot_nominal, geometry, estimate, extra, speed_scale)
            out["info"] = info
            dv = np.linalg.norm(twin.twist(qdot)[0] - twin.twist(qdot_nominal)[0])
            if out["decision"] == "EXECUTE" and dv > 0.01 + 0.1 * np.linalg.norm(v):
                out.update(decision="REPLAN_FILTER", reason="cbf:" + ",".join(sorted(set(info["active"]))) if info["active"] else "limits", changed=True)
            stop = info["slack"] > 0.02
            if cfg["learned"] and score >= tau and info["slack"] > 0.0:
                stop, out["reason"] = True, "learned_risk_certificate"
            if stop:
                qdot = np.zeros(7)
                out.update(decision="SAFE_STOP", changed=True, reason=out["reason"] if out["reason"] == "learned_risk_certificate" else "cbf_infeasible")
            if self.mode == "policy" and progress_goal is not None:
                speed = float(np.linalg.norm(twin.ee_jacobian()[0] @ qdot_nominal))
                if self.shield.stalled(progress_goal, speed, bool(info["active"]) or out["decision"] == "SAFE_STOP"):
                    path = self.shield.plan(twin.ee(), progress_goal, estimate, extra)
                    if path:
                        self.shield.detour, self.shield.detour_steps, self.mode = path, 0, "detour"
                        out.update(decision="REPLAN_DETOUR", reason=f"planned_{len(path)}_waypoints", changed=True,
                                   plan=[np.round(w, 3).tolist() for w in path])
                    else:
                        qdot = np.zeros(7)
                        out.update(decision="SAFE_STOP", reason="no_safe_path", changed=True)
                        self.shield.cooldown = 10
            out["qdot"] = qdot
        elif cfg["learned"] and score >= tau:
            out.update(decision="SAFE_STOP", reason="learned_risk", qdot=np.zeros(7), changed=True)
        self.hold = self.hold + 1 if out["decision"] == "SAFE_STOP" else 0
        if self.hold >= HOLD_ABORT_STEPS:
            out.update(decision="ABORT", reason="prolonged_hold", qdot=np.zeros(7), changed=True)
        out["v"], out["omega"] = twin.twist(out["qdot"])
        return out


__all__ = ["SafetyLayer", "METHODS", "FEATURES", "KAPPA", "GAMMA"]

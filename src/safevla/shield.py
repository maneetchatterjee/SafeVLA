"""Hard safety layer: instruction checks, CBF-QP action filter, detour planner.

All geometry comes from perception (point clouds) and robot kinematics; the
simulator's ground truth is never read here.

CBF: for robot sphere i (centre p_i, radius r_i) and hazard point q,
h = ||p_i - q|| - r_i - margin, and we require dh/dt >= -gamma * h, i.e.
n^T J_i qdot >= -gamma h + n^T v_hazard (v_hazard = perceived human velocity).
The QP minimally modifies the nominal joint velocity subject to these rows plus
joint-limit barriers, velocity/acceleration bounds and a workspace barrier.
Slack keeps the QP feasible; positive slack means the certificate could not be
met and the arbiter treats it as a stop condition.
"""

import mujoco
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from .env import DT, VMAX, QDOT_PHYSICAL
from .expert import GRASPED_WIDTH

SPHERES = [  # (body, local offset, radius): conservative envelope; hand is ~0.2 m wide in local y
    ("hand", (0.0, 0.0, 0.105), 0.035),  # fingertips and a held cube
    ("hand", (0.0, 0.0, 0.075), 0.035),
    *[("hand", (0.0, y, 0.03), 0.05) for y in (-0.075, -0.037, 0.0, 0.037, 0.075)],
    ("link7", (0.0, 0.0, 0.07), 0.065),
    ("link6", (0.06, 0.0, 0.0), 0.07),
    ("link5", (0.0, 0.0, -0.1), 0.065),
    ("link4", (0.0, 0.0, 0.0), 0.07),
]
BASE_MARGIN = {"wall": 0.025, "vase": 0.045, "human": 0.10}
ACTIVATION = 0.35
GAMMA = 5.0
WS_LOW = np.array([0.2, -0.5, 0.012])
WS_HIGH = np.array([0.92, 0.5, 0.6])
PLAN_LO = np.array([0.2, -0.5, 0.08])  # detours never skim the table
PLAN_HI = np.array([0.9, 0.5, 0.56])
ACCEL = 15.0  # rad/s^2, Franka datasheet order of magnitude


class Geometry:
    """Robot spheres, Jacobians and nearest perceived hazard points for one step."""

    def __init__(self, env, estimate, data=None):
        d = data or env.data
        self.centers, self.radii, self.jacobians = [], [], []
        for body, offset, radius in SPHERES:
            b = env.model.body(body).id
            center = d.xpos[b] + d.xmat[b].reshape(3, 3) @ np.asarray(offset)
            self.centers.append(center)
            self.radii.append(radius)
            self.jacobians.append(env.point_jacobian(b, center, d))
        self.centers = np.array(self.centers)
        self.radii = np.array(self.radii)
        self.nearest = {}
        for kind, cloud in estimate["clouds"].items():
            if len(cloud) == 0:
                self.nearest[kind] = None
                continue
            dist, idx = cKDTree(cloud).query(self.centers)
            gap = dist - self.radii
            self.nearest[kind] = (gap, cloud[idx])

    def min_gap(self, kind):
        item = self.nearest.get(kind)
        return 0.5 if item is None else float(min(0.5, item[0].min()))

    def approach_rate(self, kind, qdot, hazard_velocity=None):
        """Most negative d(gap)/dt over spheres under joint velocity qdot."""
        item = self.nearest.get(kind)
        if item is None:
            return 0.0
        gap, points = item
        rates = []
        for i in range(len(self.centers)):
            if gap[i] > ACTIVATION:
                continue
            n = self.centers[i] - points[i]
            n = n / (np.linalg.norm(n) + 1e-9)
            v = self.jacobians[i] @ qdot
            if hazard_velocity is not None:
                v = v - hazard_velocity
            rates.append(n @ v)
        return float(min(rates)) if rates else 0.0


def solve_qp(nominal, A, b, lb, ub, weight=None):
    """min ||x - nominal||^2 + 1e3 ||s||^2  s.t.  A x + s >= b, s >= 0, lb <= x <= ub."""
    n, m = len(nominal), len(b)
    if m == 0:
        return np.clip(nominal, lb, ub), 0.0
    x0 = np.r_[np.clip(nominal, lb, ub), np.maximum(0, b - A @ np.clip(nominal, lb, ub))]

    def objective(z):
        x, s = z[:n], z[n:]
        return float((x - nominal) @ (x - nominal) + 1e3 * s @ s)

    def gradient(z):
        return np.r_[2 * (z[:n] - nominal), 2e3 * z[n:]]

    constraint = {
        "type": "ineq",
        "fun": lambda z: A @ z[:n] + z[n:] - b,
        "jac": lambda z: np.c_[A, np.eye(m)],
    }
    bounds = list(zip(lb, ub)) + [(0, None)] * m
    result = minimize(objective, x0, jac=gradient, constraints=[constraint], bounds=bounds, method="SLSQP", options={"maxiter": 60, "ftol": 1e-9})
    z = result.x if result.success or np.isfinite(result.x).all() else x0
    return z[:n], float(np.max(z[n:], initial=0.0))


class HardShield:
    def __init__(self, env, margin_scale=1.0, use_ssm=True, use_human_velocity=True, use_detour=True,
                 ws_low=WS_LOW, ws_high=WS_HIGH, plan_lo=PLAN_LO, plan_hi=PLAN_HI, dt=DT):
        # Workspace/planner bounds and control period are parameters so the same layer
        # runs on other benchmarks through a kinematic twin (defaults = v4 scene).
        self.env = env
        self.ws_low, self.ws_high = np.asarray(ws_low, float), np.asarray(ws_high, float)
        self.plan_lo, self.plan_hi = np.asarray(plan_lo, float), np.asarray(plan_hi, float)
        self.dt = dt
        self.margin_scale = margin_scale
        self.use_ssm = use_ssm
        self.use_human_velocity = use_human_velocity
        self.use_detour = use_detour
        self.reset()

    def reset(self):
        self.previous_qdot = np.zeros(7)
        self.history = []
        self.detour = None
        self.detour_steps = 0
        self.plan_failures = 0
        self.cooldown = 0

    # ----------------------------------------------------------- filtering
    def margins(self, extra=0.0):
        return {k: v * self.margin_scale + extra for k, v in BASE_MARGIN.items()}

    def filter(self, qdot_nominal, geometry, estimate, extra_margin=0.0, speed_scale=1.0):
        env = self.env
        q = env.q()
        info = {"slack": 0.0, "active": [], "ssm_scale": 1.0}
        qdot_nominal = qdot_nominal * speed_scale
        # Speed and separation monitoring (ISO/TS 15066 style, illustrative gains).
        human_gap = geometry.min_gap("human")
        if self.use_ssm and human_gap < 0.45:
            scale = float(np.clip((human_gap - 0.08) / 0.3, 0.15, 1.0))
            jp, _ = env.ee_jacobian()
            speed = np.linalg.norm(jp @ qdot_nominal)
            if speed > scale * VMAX:
                qdot_nominal = qdot_nominal * (scale * VMAX / speed)
                info["ssm_scale"] = scale
        margins = self.margins(extra_margin)
        rows, rhs = [], []
        for kind, item in geometry.nearest.items():
            if item is None:
                continue
            gap, points = item
            velocity = estimate["human_velocity"] if (kind == "human" and self.use_human_velocity) else np.zeros(3)
            for i in np.flatnonzero(gap < ACTIVATION):
                n = geometry.centers[i] - points[i]
                n = n / (np.linalg.norm(n) + 1e-9)
                h = gap[i] - margins[kind]
                rows.append(n @ geometry.jacobians[i])
                rhs.append(-GAMMA * h + n @ velocity)
                info["active"].append(kind)
        ee = env.ee()
        jp, _ = env.ee_jacobian()
        for axis in range(3):
            if ee[axis] - self.ws_low[axis] < 0.12:
                rows.append(jp[axis])
                rhs.append(-GAMMA * (ee[axis] - self.ws_low[axis]))
            if self.ws_high[axis] - ee[axis] < 0.12:
                rows.append(-jp[axis])
                rhs.append(-GAMMA * (self.ws_high[axis] - ee[axis]))
        vmax = 0.9 * QDOT_PHYSICAL
        lb = np.maximum(-vmax, -GAMMA * (q - (env.q_low + 0.05)))
        ub = np.minimum(vmax, GAMMA * ((env.q_high - 0.05) - q))
        lb = np.minimum(lb, 0.0)
        ub = np.maximum(ub, 0.0)
        acc_lb = np.maximum(lb, self.previous_qdot - ACCEL * self.dt)
        acc_ub = np.minimum(ub, self.previous_qdot + ACCEL * self.dt)
        if np.all(acc_lb <= acc_ub):
            lb, ub = acc_lb, acc_ub
        A = np.array(rows).reshape(-1, 7)
        b = np.array(rhs)
        qdot, slack = solve_qp(qdot_nominal, A, b, lb, ub)
        info["slack"] = slack
        info["constraints"] = len(b)
        self.previous_qdot = qdot.copy()
        return qdot, info

    # ------------------------------------------------------ detour planning
    def stalled(self, progress_goal, nominal_speed, constrained):
        ee = self.env.ee()
        self.history.append(np.linalg.norm(ee - progress_goal))
        self.history = self.history[-30:]
        if self.cooldown > 0:
            self.cooldown -= 1
            return False
        if len(self.history) < 30 or not constrained or nominal_speed < 0.05:
            return False
        return self.history[0] - self.history[-1] < 0.02

    def plan(self, start, goal, estimate, extra_margin=0.0):
        """Wavefront shortest path on a 2.5 cm grid around inflated perceived hazards."""
        clouds = [c for c in estimate["clouds"].values() if len(c)]
        if not clouds:
            return [goal]
        points = np.concatenate(clouds)
        res = 0.025
        lo, hi = self.plan_lo, self.plan_hi
        shape = np.ceil((hi - lo) / res).astype(int)
        grid = np.stack(np.meshgrid(*[lo[i] + res * (np.arange(shape[i]) + 0.5) for i in range(3)], indexing="ij"), -1)
        clearance = 0.13 + max(self.margins(extra_margin).values()) * 0.6  # hand half-width + margin
        dist, _ = cKDTree(points).query(grid.reshape(-1, 3), distance_upper_bound=clearance + 0.05)
        blocked = (dist < clearance).reshape(shape)
        cell = lambda p: tuple(np.clip(((np.asarray(p) - lo) / res).astype(int), 0, shape - 1))  # noqa: E731
        s, g = cell(start), cell(goal)
        if blocked[g]:  # goal inside the conservative inflation: use nearest free cell nearby
            free = np.argwhere(~blocked)
            near = free[np.argmin(np.linalg.norm(free - np.array(g), axis=1))]
            if np.linalg.norm(near - np.array(g)) * res > 0.15:
                return None
            g = tuple(near)
            goal = lo + res * (near + 0.5)
        if blocked[s]:  # start inside inflation: begin from the nearest free cell
            free = np.argwhere(~blocked)
            s = tuple(free[np.argmin(np.linalg.norm(free - np.array(s), axis=1))])
        cost = np.full(shape, np.inf)
        cost[g] = 0
        frontier = np.zeros(shape, bool)
        frontier[g] = True
        offsets = [np.array(o) for o in np.ndindex(3, 3, 3) if o != (1, 1, 1)]
        level = 0
        while frontier.any() and not np.isfinite(cost[s]) and level < 200:
            level += 1
            grown = np.zeros(shape, bool)
            for o in offsets:
                shift = o - 1
                src = frontier
                rolled = np.zeros(shape, bool)
                sl_dst = tuple(slice(max(0, k), shape[i] + min(0, k)) for i, k in enumerate(shift))
                sl_src = tuple(slice(max(0, -k), shape[i] + min(0, -k)) for i, k in enumerate(shift))
                rolled[sl_dst] = src[sl_src]
                grown |= rolled
            grown &= ~blocked & ~np.isfinite(cost)
            cost[grown] = level
            frontier = grown
        if not np.isfinite(cost[s]):
            return None
        path = [np.array(s)]
        current = np.array(s)
        while cost[tuple(current)] > 0:
            best = None
            for o in offsets:
                nxt = current + o - 1
                if np.any(nxt < 0) or np.any(nxt >= shape):
                    continue
                if cost[tuple(nxt)] < cost[tuple(current)] and (best is None or np.linalg.norm(o - 1) < np.linalg.norm(best - current)):
                    best = nxt
            if best is None:
                break
            current = best
            path.append(current)
        waypoints = [lo + res * (p + 0.5) for p in path]

        def visible(a, b):
            for t in np.linspace(0, 1, 12):
                if blocked[cell(a + t * (b - a))]:
                    return False
            return True

        smooth = [np.asarray(start)]
        index = 0
        while index < len(waypoints) - 1:
            nxt = len(waypoints) - 1
            while nxt > index + 1 and not visible(smooth[-1], waypoints[nxt]):
                nxt -= 1
            smooth.append(waypoints[nxt])
            index = nxt
        smooth.append(np.asarray(goal))
        return smooth[1:]

    def detour_velocity(self):
        ee = self.env.ee()
        while self.detour and np.linalg.norm(self.detour[0] - ee) < 0.03:
            self.detour.pop(0)
        self.detour_steps += 1
        if not self.detour or self.detour_steps > 200:
            self.detour = None
            self.cooldown = 20
            self.history = []
            return None
        delta = self.detour[0] - ee
        return delta / max(np.linalg.norm(delta), 1e-6) * min(0.2, 4 * np.linalg.norm(delta))

    # ---------------------------------------------------------- pre-checks
    def check_instruction(self, task_state, estimate):
        parsed = task_state.parsed
        if parsed["fragile_object"] or parsed["fragile_goal"] or parsed["person_goal"]:
            return "REFUSE", "unsafe_instruction"
        if parsed["object"] is None or parsed["goal"] is None:
            return "REFUSE", "unparsed_instruction"
        task_state.ground(estimate)
        if not task_state.candidates:
            return "REFUSE", "object_not_found"
        if len(task_state.candidates) > 1:
            return "REQUEST_CLARIFICATION", "ambiguous_referent"
        return self.check_grounded(task_state)

    def check_grounded(self, task_state):
        if task_state.anchor is None:
            return "REFUSE", "object_not_resolved"
        if task_state.goal is None:
            return "REFUSE", "goal_not_found"
        for target in [np.r_[task_state.goal[:2], 0.06], np.r_[task_state.goal[:2], 0.2], np.r_[task_state.anchor[:2], 0.03]]:
            if not self.reachable(target):
                return "REFUSE", "unreachable_target"
        return "EXECUTE", "checks_passed"

    def reachable(self, target, iterations=300):
        """Kinematic IK feasibility (position + downward orientation) within joint limits."""
        env = self.env
        d = mujoco.MjData(env.model)
        d.qpos[:] = env.data.qpos
        for _ in range(iterations):
            mujoco.mj_kinematics(env.model, d)
            mujoco.mj_comPos(env.model, d)
            err = target - d.site_xpos[env.ee_site]
            R = d.site_xmat[env.ee_site].reshape(3, 3)
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, (env.R_des @ R.T).ravel())
            rot = 2 * np.sign(quat[0] or 1) * quat[1:]
            if np.linalg.norm(err) < 0.01 and np.linalg.norm(rot) < 0.1:
                return True
            jp, jr = env.ee_jacobian(d)
            J = np.vstack([jp, jr])
            dq = J.T @ np.linalg.solve(J @ J.T + 0.02 * np.eye(6), np.r_[err, rot])
            q = d.qpos[env.joint_qadr] + np.clip(dq, -0.2, 0.2)
            d.qpos[env.joint_qadr] = np.clip(q, env.q_low + 0.02, env.q_high - 0.02)
        return False


def perceived_held(task_state, env):
    return task_state.anchor is not None and env.gripper_width() < GRASPED_WIDTH and np.linalg.norm(task_state.anchor[:2] - env.ee()[:2]) < 0.03

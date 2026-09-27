"""Learned transition-risk ensemble with temperature calibration and conformal threshold.

Label: the base policy's proposed chunk, executed *unshielded* from the current
simulator state for HORIZON control steps, produces a ground-truth safety event
(computed on a cloned MjData). Features use only perception, proprioception,
the proposal and policy disagreement. Calibration and the conformal threshold
use episodes disjoint from training; test episodes are never used for fitting.
"""

import json
import numpy as np
import torch
from torch import nn

HORIZON = 5
FEATURE_NAMES = [
    "ee_x", "ee_y", "ee_z", "vel_x", "vel_y", "vel_z", "width", "held",
    "prop_x", "prop_y", "prop_z", "chunk_x", "chunk_y", "chunk_z", "policy_std", "grip_std",
    "gap_wall", "gap_vase", "gap_human", "rate_wall", "rate_vase", "rate_human",
    "count_wall", "count_vase", "count_human", "human_speed_toward",
    "invalid_depth", "jitter", "stale", "joint_margin", "workspace_margin", "wrench",
]


class RiskNet(nn.Module):
    def __init__(self, n=len(FEATURE_NAMES)):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n, 96), nn.GELU(), nn.Linear(96, 96), nn.GELU(), nn.Linear(96, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


class RiskModel:
    def __init__(self, members, mean, std, temperature=1.0, threshold=0.5, alpha=0.1):
        self.members = [m.eval() for m in members]
        self.mean = torch.as_tensor(mean, dtype=torch.float32)
        self.std = torch.as_tensor(std, dtype=torch.float32)
        self.temperature = float(temperature)
        self.threshold = float(threshold)
        self.alpha = alpha

    @torch.inference_mode()
    def logits(self, x):
        x = (torch.as_tensor(np.atleast_2d(x), dtype=torch.float32) - self.mean) / self.std
        return torch.stack([m(x) for m in self.members]).numpy()

    def predict(self, x):
        """Calibrated probability (mean), member std, and uncertainty-aware score."""
        z = self.logits(x) / self.temperature
        p = 1 / (1 + np.exp(-z))
        mean = p.mean(0)
        std = p.std(0)
        return mean, std, np.clip(mean + std, 0, 1)

    def save(self, path, extra=None, feature_names=None):
        torch.save(
            {
                "members": [m.state_dict() for m in self.members],
                "mean": self.mean,
                "std": self.std,
                "temperature": self.temperature,
                "threshold": self.threshold,
                "alpha": self.alpha,
                "features": list(feature_names or FEATURE_NAMES),
                "extra": json.dumps(extra or {}),
            },
            path,
        )

    @classmethod
    def load(cls, path):
        item = torch.load(path, map_location="cpu", weights_only=True)
        members = []
        for state in item["members"]:
            m = RiskNet(len(item["features"]))
            m.load_state_dict(state)
            members.append(m)
        return cls(members, item["mean"], item["std"], item["temperature"], item["threshold"], item["alpha"])


def train_risk(x, y, groups, members=5, epochs=40, seed=0, log=print):
    x = np.asarray(x, np.float32)
    y = np.asarray(y, np.float32)
    mean, std = x.mean(0), x.std(0) + 1e-3
    xt = torch.as_tensor((x - mean) / std)
    yt = torch.as_tensor(y)
    episodes = np.unique(groups)
    pos_weight = torch.tensor(float(np.clip((1 - y.mean()) / max(y.mean(), 1e-3), 1, 8)))
    models = []
    for k in range(members):
        rng = np.random.default_rng(seed * 100 + k)
        torch.manual_seed(seed * 100 + k)
        chosen = set(rng.choice(episodes, len(episodes), replace=True).tolist())
        index = np.flatnonzero(np.isin(groups, list(chosen)))
        model = RiskNet(x.shape[1])
        opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
        for epoch in range(epochs):
            order = rng.permutation(index)
            total = 0.0
            for start in range(0, len(order), 1024):
                batch = torch.as_tensor(order[start : start + 1024])
                loss = nn.functional.binary_cross_entropy_with_logits(model(xt[batch]), yt[batch], pos_weight=pos_weight)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += float(loss) * len(batch)
            if epoch % 10 == 0 or epoch == epochs - 1:
                log(f"risk member {k} epoch {epoch} loss {total / len(order):.4f}")
        models.append(model.eval())
    return RiskModel(models, mean, std)


def fit_temperature(model, x, y):
    """Scalar temperature minimising calibration-split NLL (Guo et al., 2017)."""
    z = model.logits(x)
    best, best_nll = 1.0, np.inf
    for t in np.exp(np.linspace(np.log(0.3), np.log(5), 60)):
        p = np.clip((1 / (1 + np.exp(-z / t))).mean(0), 1e-6, 1 - 1e-6)
        nll = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        if nll < best_nll:
            best, best_nll = t, nll
    model.temperature = float(best)
    return best


def conformal_threshold(scores, y, alpha=0.1):
    """Largest threshold whose calibration false-negative rate is <= alpha.

    Split-conformal quantile on positive-example scores with the finite-sample
    (n+1) correction, giving marginal FNR <= alpha for exchangeable episodes.
    """
    positives = np.sort(np.asarray(scores)[np.asarray(y) == 1])
    n = len(positives)
    if n == 0:
        return 0.5
    k = int(np.floor(alpha * (n + 1))) - 1
    return float(positives[max(k, 0)])


def calibration_metrics(p, y, bins=10):
    p, y = np.asarray(p, float), np.asarray(y, float)
    edges = np.linspace(0, 1, bins + 1)
    ece, table = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if mask.any():
            ece += mask.mean() * abs(p[mask].mean() - y[mask].mean())
            table.append({"bin": [float(lo), float(hi)], "n": int(mask.sum()), "predicted": float(p[mask].mean()), "empirical": float(y[mask].mean())})
    order = np.argsort(-p)
    tp = np.cumsum(y[order])
    fp = np.cumsum(1 - y[order])
    tpr = tp / max(y.sum(), 1)
    fpr = fp / max((1 - y).sum(), 1)
    trapezoid = getattr(np, "trapezoid", None) or np.trapz  # numpy 2.x name, numpy 1.x fallback
    auroc = float(trapezoid(np.r_[0, tpr], np.r_[0, fpr])) if y.sum() and (1 - y).sum() else None
    # Coverage-risk: keep the lowest-risk fraction, report event rate among kept.
    keep = np.argsort(p)
    coverage = np.linspace(0.05, 1, 20)
    risk_curve = [float(y[keep[: max(1, int(c * len(y)))]].mean()) for c in coverage]
    return {
        "n": int(len(y)),
        "positive_rate": float(y.mean()),
        "ece": float(ece),
        "brier": float(np.mean((p - y) ** 2)),
        "auroc": auroc,
        "reliability": table,
        "coverage": coverage.tolist(),
        "selective_risk": risk_curve,
    }

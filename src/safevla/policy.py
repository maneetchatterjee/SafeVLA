"""Bootstrapped ensemble of action-chunking MLP behaviour-cloning policies.

Input: perception-grounded task features (agent.FEATURES). Output: K future
end-effector velocity commands plus gripper command. The ensemble mean is
executed; member disagreement is an uncertainty signal for the risk model.
"""

import json
import os
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .agent import FEATURES

CHUNK = 4
ACTION = 4


class ChunkMLP(nn.Module):
    def __init__(self, hidden=256, n_in=FEATURES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, CHUNK * ACTION), nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x).reshape(-1, CHUNK, ACTION)


class EnsemblePolicy:
    def __init__(self, members, mean, std):
        self.members = [m.eval() for m in members]
        self.mean = torch.as_tensor(mean, dtype=torch.float32)
        self.std = torch.as_tensor(std, dtype=torch.float32)

    @torch.inference_mode()
    def __call__(self, features):
        x = (torch.as_tensor(features, dtype=torch.float32)[None] - self.mean) / self.std
        out = torch.stack([m(x)[0] for m in self.members]).numpy()
        return out.mean(0), out.std(0)

    def save(self, path, metadata):
        torch.save(
            {"members": [m.state_dict() for m in self.members], "mean": self.mean, "std": self.std, "metadata": metadata},
            path,
        )

    @classmethod
    def load(cls, path):
        item = torch.load(path, map_location="cpu", weights_only=True)
        members = []
        for state in item["members"]:
            m = ChunkMLP(n_in=len(item["mean"]))
            m.load_state_dict(state)
            members.append(m)
        return cls(members, item["mean"], item["std"])


def train_ensemble(features, labels, episode_ids, path, members=5, updates=12000, seed=0, log=print):
    """Episode-level bootstrap per member; returns the saved ensemble."""
    torch.set_num_threads(int(os.environ.get("SAFEVLA_TRAIN_THREADS", "4")))
    features = np.asarray(features, np.float32)
    labels = np.asarray(labels, np.float32)
    mean = features.mean(0)
    std = features.std(0) + 1e-3
    x_all = torch.as_tensor((features - mean) / std)
    y_all = torch.as_tensor(labels)
    episodes = np.unique(episode_ids)
    trained, history = [], []
    for k in range(members):
        rng = np.random.default_rng(seed * 100 + k)
        torch.manual_seed(seed * 100 + k)
        chosen = rng.choice(episodes, len(episodes), replace=True)
        counts = np.bincount(np.searchsorted(episodes, chosen), minlength=len(episodes))
        weight = counts[np.searchsorted(episodes, episode_ids)].astype(np.float64)
        index = np.flatnonzero(weight)
        prob = weight[index] / weight[index].sum()
        model = ChunkMLP(n_in=features.shape[1])
        opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, updates)
        losses = []
        for step in range(updates):
            batch = torch.as_tensor(rng.choice(index, 256, p=prob))
            pred = model(x_all[batch])
            target = y_all[batch]
            loss = nn.functional.mse_loss(pred[..., :3], target[..., :3]) + 0.5 * nn.functional.mse_loss(pred[..., 3], target[..., 3])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            if step % 1000 == 0 or step == updates - 1:
                losses.append((step, float(loss)))
                log(f"member {k} update {step} loss {float(loss):.5f}")
        trained.append(model.eval())
        history.append(losses)
    policy = EnsemblePolicy(trained, mean, std)
    policy.save(path, {"members": members, "updates": updates, "seed": seed, "samples": int(len(features)), "episodes": int(len(episodes))})
    Path(path).with_suffix(".json").write_text(json.dumps({"losses": history, "samples": int(len(features))}, indent=2))
    return policy

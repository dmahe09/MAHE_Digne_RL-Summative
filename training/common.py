"""Shared TorchRL plumbing: env construction, networks, evaluation, logging, model IO."""
from __future__ import annotations

import json
import os
import random
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from torch import nn

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from environment.custom_env import ACTION_NAMES, OBS_NAMES, EntrepreneurshipEnv  # noqa: E402

OBS_DIM = len(OBS_NAMES)
N_ACT = len(ACTION_NAMES)
MODELS = ROOT / "models"
LOGS = ROOT / "logs"
RESULTS = ROOT / "results"
for _d in (MODELS / "dqn", MODELS / "pg", LOGS, RESULTS):
    _d.mkdir(parents=True, exist_ok=True)

# Frames (= environment steps = simulated months) per tuning run. All algorithms get the same budget.
TOTAL_FRAMES = int(os.environ.get("RL_TOTAL_FRAMES", 100_000))
# Seeds used to *select* the best run (tuning) and to *test* it afterwards (kept disjoint).
SELECT_SEED0, TEST_SEED0 = 10_000, 50_000


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


# ----------------------------------------------------------------------------- env / nets
def make_env(seed: int | None = None, **env_kwargs):
    from torchrl.envs import GymWrapper, RewardSum, TransformedEnv
    base = EntrepreneurshipEnv(**env_kwargs)
    env = TransformedEnv(GymWrapper(base, categorical_action_encoding=True), RewardSum())
    if seed is not None:
        env.set_seed(seed)
    return env


def build_mlp(out_features: int, hidden: int = 64, activation: str = "tanh"):
    from torchrl.modules import MLP
    act = {"tanh": nn.Tanh, "relu": nn.ReLU}[activation]
    return MLP(in_features=OBS_DIM, out_features=out_features, num_cells=[hidden, hidden], activation_class=act)


def save_model(path, kind: str, net: nn.Module, hidden: int, activation: str, cfg: dict):
    torch.save({"kind": kind, "state_dict": net.state_dict(), "hidden": hidden, "activation": activation,
                "obs_dim": OBS_DIM, "n_actions": N_ACT, "cfg": cfg}, path)


def load_policy(path, deterministic: bool | None = None):
    """Return f(obs: np.ndarray) -> int for any saved model.
    DQN: greedy argmax of Q-values.  Policy-gradient models: sample from the softmax policy
    (their natural, stochastic behaviour); pass deterministic=True for argmax."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    net = build_mlp(ck["n_actions"], ck["hidden"], ck["activation"])
    net.load_state_dict(ck["state_dict"])
    net.eval()
    greedy = (ck["kind"] == "dqn") if deterministic is None else deterministic

    @torch.no_grad()
    def act(obs):
        out = net(torch.as_tensor(obs, dtype=torch.float32))
        if greedy:
            return int(out.argmax().item())
        return int(torch.distributions.Categorical(logits=out).sample().item())
    act.meta = {"kind": ck["kind"], "cfg": ck["cfg"]}
    return act


# ----------------------------------------------------------------------------- evaluation
def evaluate_policy(act, n_episodes=30, seed0=SELECT_SEED0, **env_kwargs) -> dict:
    torch.manual_seed(seed0)          # reproducible sampling for stochastic policies
    env = EntrepreneurshipEnv(**env_kwargs)
    rets, succ, bank, steps_ok = [], 0, 0, []
    for i in range(n_episodes):
        obs, info = env.reset(seed=seed0 + i)
        done, total = False, 0.0
        while not done:
            obs, r, te, tr, info = env.step(act(obs))
            total += r
            done = te or tr
        rets.append(total)
        if info["status"] == "SUCCESS":
            succ += 1
            steps_ok.append(info["step"])
        elif info["status"] == "BANKRUPT":
            bank += 1
    return {"mean_return": float(np.mean(rets)), "std_return": float(np.std(rets)),
            "success_rate": succ / n_episodes, "bankrupt_rate": bank / n_episodes,
            "mean_steps_to_success": float(np.mean(steps_ok)) if steps_ok else None, "n": n_episodes}


# ----------------------------------------------------------------------------- logging helpers
def finished_episodes(batch):
    """(returns, success_flags) of every episode that ended inside a collected batch."""
    done = batch["next", "done"].reshape(-1)
    if not bool(done.any()):
        return [], []
    ret = batch["next", "episode_reward"].reshape(-1)[done]
    term = batch["next", "terminated"].reshape(-1)[done]
    rew = batch["next", "reward"].reshape(-1)[done]
    return ret.tolist(), (term & (rew > 10)).tolist()   # success step carries +30, bankruptcy -15


class RunLog:
    def __init__(self, algo, run, cfg, seed):
        self.d = {"algo": algo, "run": run, "cfg": cfg, "seed": seed,
                  "episodes": {"frame": [], "ret": [], "succ": []},
                  "loss": {"frame": [], "value": []}, "entropy": {"frame": [], "value": []}}
        self.t0 = time.time()
        self._acc = {"loss": [], "entropy": []}
        self._next_print = 20_000

    def should_print(self, frame):
        if frame >= self._next_print:
            self._next_print += 20_000
            return True
        return False

    def episodes(self, frame, rets, succ):
        e = self.d["episodes"]
        e["frame"] += [frame] * len(rets)
        e["ret"] += [float(r) for r in rets]
        e["succ"] += [int(s) for s in succ]

    def scalar(self, key, value):
        self._acc[key].append(float(value))

    def flush(self, frame):
        for k, v in self._acc.items():
            if v:
                self.d[k]["frame"].append(frame)
                self.d[k]["value"].append(float(np.mean(v)))
                self._acc[k] = []

    def finish(self, eval_dict, path):
        self.d["train_time_s"] = round(time.time() - self.t0, 1)
        self.d["eval"] = eval_dict
        Path(path).write_text(json.dumps(self.d))
        return self.d


def smooth(x, w=50):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return x
    w = min(w, len(x))
    c = np.cumsum(np.insert(x, 0, 0.0))
    out = (c[w:] - c[:-w]) / w
    return np.concatenate([np.full(w - 1, np.nan), out]) if len(x) >= w else x

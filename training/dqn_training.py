"""DQN (value-based) with TorchRL: QValueActor + EGreedyModule + replay buffer + DQNLoss + SoftUpdate."""
from __future__ import annotations

import argparse
import copy

from training.common import (LOGS, MODELS, SELECT_SEED0, TOTAL_FRAMES, RunLog, build_mlp, evaluate_policy,
                             finished_episodes, load_policy, make_env, save_model, set_seed)
from training.configs import DQN_BASE, DQN_RUNS

import torch


def train_dqn(cfg: dict, run: int = 0, seed: int = 0, total_frames: int = TOTAL_FRAMES, tag: str | None = None,
              verbose: bool = True):
    from tensordict.nn import TensorDictSequential
    from torchrl.collectors import Collector
    from torchrl.data import LazyTensorStorage, ReplayBuffer
    from torchrl.modules import EGreedyModule, QValueActor
    from torchrl.objectives import DQNLoss, SoftUpdate

    cfg = {**DQN_BASE, **cfg}
    set_seed(seed)
    env = make_env(seed)
    net = build_mlp(env.action_spec.space.n, cfg["hidden"], "relu")
    actor = QValueActor(net, in_keys=["observation"], spec=env.action_spec)
    explore = EGreedyModule(env.action_spec, eps_init=1.0, eps_end=cfg["eps_end"],
                            annealing_num_steps=int(cfg["eps_anneal_frac"] * total_frames))
    policy = TensorDictSequential(actor, explore)
    collector = Collector(env, policy, frames_per_batch=cfg["frames_per_batch"], total_frames=total_frames)
    rb = ReplayBuffer(storage=LazyTensorStorage(cfg["buffer"]), batch_size=cfg["batch"])
    loss_mod = DQNLoss(actor, action_space=env.action_spec, delay_value=True, loss_function="smooth_l1")
    loss_mod.make_value_estimator(gamma=cfg["gamma"])
    target = SoftUpdate(loss_mod, tau=cfg["tau"])
    opt = torch.optim.Adam(loss_mod.parameters(), lr=cfg["lr"])

    log = RunLog("dqn", run, cfg, seed)
    frames = 0
    for data in collector:
        n = data.numel()
        frames += n
        rets, succ = finished_episodes(data)
        log.episodes(frames, rets, succ)
        rb.extend(data.reshape(-1))
        explore.step(n)
        if len(rb) >= cfg["warmup"]:
            for _ in range(cfg["updates"]):
                loss = loss_mod(rb.sample())["loss"]
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(loss_mod.parameters(), 10.0)
                opt.step()
                target.step()
                log.scalar("loss", loss.item())
        log.flush(frames)
        if verbose and log.should_print(frames) and log.d["episodes"]["ret"]:
            r = log.d["episodes"]["ret"][-50:]
            print(f"    [dqn run {run:02d}] frames={frames:>7,} recent_return={sum(r)/len(r):7.2f}", flush=True)
    collector.shutdown()

    name = tag or f"dqn_run_{run:02d}"
    mpath = MODELS / "dqn" / f"{name}.pt"
    save_model(mpath, "dqn", net, cfg["hidden"], "relu", cfg)
    ev = evaluate_policy(load_policy(mpath), 30, SELECT_SEED0)
    log.finish(ev, LOGS / f"{name}.json")
    if verbose:
        print(f"    -> eval return {ev['mean_return']:.1f}  success {ev['success_rate']:.0%}  "
              f"time {log.d['train_time_s']}s", flush=True)
    return log.d


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, nargs="*", default=None, help="subset of run ids (1-10)")
    ap.add_argument("--frames", type=int, default=TOTAL_FRAMES)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    for i, over in enumerate(DQN_RUNS, 1):
        if a.runs and i not in a.runs:
            continue
        print(f"[DQN {i:02d}/10] {over}", flush=True)
        train_dqn(over, run=i, seed=a.seed, total_frames=a.frames)

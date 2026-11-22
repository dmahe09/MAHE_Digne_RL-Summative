"""Policy-gradient methods with TorchRL: REINFORCE (hand-written loss), A2C (A2CLoss) and PPO (ClipPPOLoss)."""
from __future__ import annotations

import argparse

import torch

from training.common import (LOGS, MODELS, SELECT_SEED0, TOTAL_FRAMES, RunLog, build_mlp, evaluate_policy,
                             finished_episodes, load_policy, make_env, save_model, set_seed)
from training.configs import A2C_BASE, A2C_RUNS, PPO_BASE, PPO_RUNS, REINFORCE_BASE, REINFORCE_RUNS


def _actor_critic(env, hidden, with_critic=True):
    from tensordict.nn import TensorDictModule
    from torchrl.modules import ProbabilisticActor, ValueOperator
    logits_net = build_mlp(env.action_spec.space.n, hidden, "tanh")
    actor = ProbabilisticActor(TensorDictModule(logits_net, in_keys=["observation"], out_keys=["logits"]),
                               in_keys=["logits"], spec=env.action_spec,
                               distribution_class=torch.distributions.Categorical, return_log_prob=True)
    critic = ValueOperator(build_mlp(1, hidden, "tanh"), in_keys=["observation"]) if with_critic else None
    return logits_net, actor, critic


def _entropy(out, actor, data):
    """TorchRL omits 'entropy' from the loss output when entropy_coeff == 0, so recompute it for logging."""
    if "entropy" in out.keys():
        return out["entropy"].item()
    with torch.no_grad():
        return actor.get_dist(data.reshape(-1)).entropy().mean().item()


def _finish(log, kind, name, net, cfg):
    mpath = MODELS / "pg" / f"{name}.pt"
    save_model(mpath, kind, net, cfg["hidden"], "tanh", cfg)
    ev = evaluate_policy(load_policy(mpath), 30, SELECT_SEED0)
    log.finish(ev, LOGS / f"{name}.json")
    print(f"    -> eval return {ev['mean_return']:.1f}  success {ev['success_rate']:.0%}  "
          f"time {log.d['train_time_s']}s", flush=True)
    return log.d


def _progress(log, algo, run, frames):
    if log.should_print(frames) and log.d["episodes"]["ret"]:
        r = log.d["episodes"]["ret"][-50:]
        print(f"    [{algo} run {run:02d}] frames={frames:>7,} recent_return={sum(r)/len(r):7.2f}", flush=True)


# ------------------------------------------------------------------ REINFORCE
def train_reinforce(cfg, run=0, seed=0, total_frames=TOTAL_FRAMES, tag=None, verbose=True):
    """Vanilla Monte-Carlo policy gradient: loss = -E[ log pi(a|s) * (G_t - b) ] - c_ent * H(pi).
    TorchRL ships no REINFORCE class, so the loss is written by hand on top of TorchRL's env / actor / rollout."""
    from torchrl.envs.utils import ExplorationType, set_exploration_type
    cfg = {**REINFORCE_BASE, **cfg}
    set_seed(seed)
    env = make_env(seed)
    net, actor, _ = _actor_critic(env, cfg["hidden"], with_critic=False)
    opt = torch.optim.Adam(actor.parameters(), lr=cfg["lr"])
    log, frames, it = RunLog("reinforce", run, cfg, seed), 0, 0
    while frames < total_frames:
        obs_l, act_l, ret_l = [], [], []
        for _ in range(cfg["eps_per_update"]):
            with torch.no_grad(), set_exploration_type(ExplorationType.RANDOM):
                td = env.rollout(max_steps=env.base_env.max_steps + 5 if hasattr(env, "base_env") else 200,
                                 policy=actor, break_when_any_done=True)
            T = td.shape[0]
            r = td["next", "reward"].reshape(-1)
            g, G = 0.0, torch.zeros(T)
            for t in range(T - 1, -1, -1):                      # reward-to-go
                g = r[t] + cfg["gamma"] * g
                G[t] = g
            obs_l.append(td["observation"]); act_l.append(td["action"]); ret_l.append(G)
            frames += T
            rets, succ = finished_episodes(td)
            log.episodes(frames, rets, succ)
        obs, act, G = torch.cat(obs_l), torch.cat(act_l), torch.cat(ret_l)
        if cfg["baseline"] == "norm":
            G = (G - G.mean()) / (G.std() + 1e-8)
        elif cfg["baseline"] == "mean":
            G = G - G.mean()
        from tensordict import TensorDict
        dist = actor.get_dist(TensorDict({"observation": obs}, batch_size=[obs.shape[0]]))
        ent = dist.entropy().mean()
        loss = -(dist.log_prob(act) * G).mean() - cfg["ent"] * ent
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), cfg["max_grad_norm"]); opt.step()
        log.scalar("loss", loss.item()); log.scalar("entropy", ent.item()); log.flush(frames)
        it += 1
        if verbose:
            _progress(log, "reinforce", run, frames)
    return _finish(log, "reinforce", tag or f"reinforce_run_{run:02d}", net, cfg)


# ------------------------------------------------------------------ A2C
def train_a2c(cfg, run=0, seed=0, total_frames=TOTAL_FRAMES, tag=None, verbose=True):
    """Advantage actor-critic: synchronous rollouts of n_steps x n_envs, GAE advantages, ONE gradient step per batch."""
    from torchrl.collectors import Collector
    from torchrl.envs import SerialEnv
    from torchrl.objectives import A2CLoss
    from torchrl.objectives.value import GAE
    cfg = {**A2C_BASE, **cfg}
    set_seed(seed)
    env = SerialEnv(cfg["n_envs"], lambda: make_env())
    env.set_seed(seed)
    net, actor, critic = _actor_critic(env, cfg["hidden"])
    adv = GAE(gamma=cfg["gamma"], lmbda=cfg["gae_lambda"], value_network=critic, average_gae=True)
    loss_mod = A2CLoss(actor, critic, entropy_coeff=cfg["ent"], critic_coeff=cfg["critic_coef"],
                       loss_critic_type="smooth_l1")
    opt = torch.optim.Adam(loss_mod.parameters(), lr=cfg["lr"])
    collector = Collector(env, actor, frames_per_batch=cfg["n_steps"] * cfg["n_envs"], total_frames=total_frames)
    log, frames = RunLog("a2c", run, cfg, seed), 0
    for data in collector:
        frames += data.numel()
        rets, succ = finished_episodes(data)
        log.episodes(frames, rets, succ)
        with torch.no_grad():
            adv(data)
        if cfg["norm_adv"]:
            a = data["advantage"]
            data["advantage"] = (a - a.mean()) / (a.std() + 1e-8)
        out = loss_mod(data.reshape(-1))
        loss = sum(v for k, v in out.items() if k.startswith("loss_"))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(loss_mod.parameters(), cfg["max_grad_norm"]); opt.step()
        log.scalar("loss", loss.item()); log.scalar("entropy", _entropy(out, actor, data)); log.flush(frames)
        if verbose:
            _progress(log, "a2c", run, frames)
    collector.shutdown()
    return _finish(log, "a2c", tag or f"a2c_run_{run:02d}", net, cfg)


# ------------------------------------------------------------------ PPO
def train_ppo(cfg, run=0, seed=0, total_frames=TOTAL_FRAMES, tag=None, verbose=True):
    """PPO-clip: GAE advantages, several epochs of minibatch updates on each rollout, clipped surrogate objective."""
    from torchrl.collectors import Collector
    from torchrl.data import LazyTensorStorage, ReplayBuffer
    from torchrl.data.replay_buffers import SamplerWithoutReplacement
    from torchrl.objectives import ClipPPOLoss
    from torchrl.objectives.value import GAE
    cfg = {**PPO_BASE, **cfg}
    set_seed(seed)
    env = make_env(seed)
    net, actor, critic = _actor_critic(env, cfg["hidden"])
    adv = GAE(gamma=cfg["gamma"], lmbda=cfg["gae_lambda"], value_network=critic, average_gae=True)
    loss_mod = ClipPPOLoss(actor, critic, clip_epsilon=cfg["clip"], entropy_coeff=cfg["ent"],
                           critic_coeff=cfg["critic_coef"], loss_critic_type="smooth_l1", normalize_advantage=True)
    opt = torch.optim.Adam(loss_mod.parameters(), lr=cfg["lr"])
    fpb = cfg["frames_per_batch"]
    collector = Collector(env, actor, frames_per_batch=fpb, total_frames=total_frames)
    rb = ReplayBuffer(storage=LazyTensorStorage(fpb), sampler=SamplerWithoutReplacement(), batch_size=cfg["minibatch"])
    log, frames = RunLog("ppo", run, cfg, seed), 0
    for data in collector:
        frames += data.numel()
        rets, succ = finished_episodes(data)
        log.episodes(frames, rets, succ)
        for _ in range(cfg["epochs"]):
            with torch.no_grad():
                adv(data)
            rb.extend(data.reshape(-1).clone())
            for _ in range(fpb // cfg["minibatch"]):
                out = loss_mod(rb.sample())
                loss = sum(v for k, v in out.items() if k.startswith("loss_"))
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(loss_mod.parameters(), cfg["max_grad_norm"]); opt.step()
                log.scalar("loss", loss.item()); log.scalar("entropy", _entropy(out, actor, data))
            rb.empty()
        log.flush(frames)
        if verbose:
            _progress(log, "ppo", run, frames)
    collector.shutdown()
    return _finish(log, "ppo", tag or f"ppo_run_{run:02d}", net, cfg)


TRAINERS = {"reinforce": (train_reinforce, REINFORCE_RUNS), "a2c": (train_a2c, A2C_RUNS), "ppo": (train_ppo, PPO_RUNS)}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=[*TRAINERS, "all"], default="all")
    ap.add_argument("--runs", type=int, nargs="*", default=None)
    ap.add_argument("--frames", type=int, default=TOTAL_FRAMES)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    for algo, (fn, runs) in TRAINERS.items():
        if a.algo not in (algo, "all"):
            continue
        for i, over in enumerate(runs, 1):
            if a.runs and i not in a.runs:
                continue
            print(f"[{algo.upper()} {i:02d}/10] {over}", flush=True)
            fn(over, run=i, seed=a.seed, total_frames=a.frames)

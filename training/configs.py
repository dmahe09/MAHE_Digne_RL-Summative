"""Hyper-parameter sweeps: a base config per algorithm and 10 runs (overrides of the base).
Run 1 is always the base configuration; the others change one or two settings so effects can be read off.
Every run uses the same environment, the same frame budget and the same network size (64x64) unless stated."""

DQN_BASE = dict(hidden=64, lr=1e-3, gamma=0.99, buffer=20_000, batch=64, eps_end=0.05, eps_anneal_frac=0.3,
                tau=0.01, updates=8, frames_per_batch=32, warmup=1000)
DQN_RUNS = [
    {},                                              # 1 base
    dict(lr=3e-4),                                   # 2 slower learning
    dict(lr=3e-3),                                   # 3 faster learning
    dict(gamma=0.90),                                # 4 short horizon
    dict(gamma=0.999),                               # 5 very long horizon
    dict(buffer=5_000, updates=4),                   # 6 small replay, fewer updates
    dict(buffer=50_000, batch=128),                  # 7 large replay, bigger batches
    dict(eps_end=0.20, eps_anneal_frac=0.6),         # 8 heavy exploration
    dict(eps_end=0.01, eps_anneal_frac=0.1),         # 9 almost no exploration
    dict(tau=0.1),                                   # 10 fast-moving target network
]

REINFORCE_BASE = dict(hidden=64, lr=2e-3, gamma=0.99, ent=0.01, eps_per_update=8, baseline="norm", max_grad_norm=1.0)
REINFORCE_RUNS = [
    {},                                              # 1 base (return standardisation as variance reduction)
    dict(lr=5e-4),                                   # 2
    dict(lr=1e-2),                                   # 3
    dict(gamma=0.95),                                # 4
    dict(gamma=0.999),                               # 5
    dict(ent=0.0),                                   # 6 no entropy bonus
    dict(ent=0.05),                                  # 7 strong entropy bonus
    dict(eps_per_update=4),                          # 8 noisier, more frequent updates
    dict(eps_per_update=16),                         # 9 smoother, fewer updates
    dict(baseline="none"),                           # 10 raw Monte-Carlo returns
]

A2C_BASE = dict(hidden=64, lr=3e-4, gamma=0.99, n_steps=16, n_envs=8, gae_lambda=0.95, ent=0.01, critic_coef=0.5,
                max_grad_norm=0.5, norm_adv=True)
A2C_RUNS = [
    {},                                              # 1 base
    dict(lr=1e-4),                                   # 2
    dict(lr=1e-3),                                   # 3
    dict(gamma=0.95),                                # 4
    dict(n_steps=64),                                # 5 long rollouts
    dict(n_envs=16),                                 # 6 more parallel envs
    dict(ent=0.0),                                   # 7
    dict(ent=0.05),                                  # 8
    dict(gae_lambda=1.0),                            # 9 Monte-Carlo advantages
    dict(norm_adv=False),                            # 10 raw (unnormalised) advantages
]

PPO_BASE = dict(hidden=64, lr=3e-4, gamma=0.99, gae_lambda=0.95, clip=0.2, ent=0.01, frames_per_batch=1024,
                epochs=8, minibatch=128, critic_coef=1.0, max_grad_norm=0.5)
PPO_RUNS = [
    {},                                              # 1 base
    dict(lr=1e-3),                                   # 2
    dict(lr=1e-4),                                   # 3
    dict(gamma=0.95),                                # 4
    dict(clip=0.1),                                  # 5 conservative updates
    dict(clip=0.3),                                  # 6 permissive updates
    dict(ent=0.0),                                   # 7
    dict(ent=0.05),                                  # 8
    dict(frames_per_batch=256, minibatch=64),        # 9 short rollouts
    dict(gae_lambda=0.8, epochs=4),                  # 10 more bias, fewer epochs
]

SWEEPS = {"dqn": (DQN_BASE, DQN_RUNS), "reinforce": (REINFORCE_BASE, REINFORCE_RUNS),
          "a2c": (A2C_BASE, A2C_RUNS), "ppo": (PPO_BASE, PPO_RUNS)}

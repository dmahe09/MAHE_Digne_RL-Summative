"""Reference policies used to sanity-check the environment and to give the report a yardstick."""
from __future__ import annotations

import numpy as np

from environment.custom_env import EntrepreneurshipEnv


def random_policy(env: EntrepreneurshipEnv) -> int:
    return int(env.action_space.sample())


def scripted_policy(env: EntrepreneurshipEnv) -> int:
    """Hand-written expert with access to the true internal state (not an RL agent)."""
    e = env
    if not e.launched:
        if e.skill < 0.25 and e.capital > 0.06:
            return 0
        if e.network < 0.30:
            return 1
        if e.readiness < 0.42:
            return 3
        if e.capital < 0.14 and e.debt < 0.45:
            return 2                    # keep asking until the loan is approved
        return 6                        # education centre (needs network >= 0.3)
    if e.health < 0.55 and e.capital >= 0.05:
        return 8
    if e.employees < 2 and e.capital >= 0.09:
        return 7
    if e.debt > 0.0 and e.capital > 0.30:
        return 9
    if e.health < 0.70 and e.capital >= 0.06:
        return 8
    return 10


def run_episode(env, policy, seed=None):
    env.reset(seed=seed)
    done, total, info = False, 0.0, {}
    while not done:
        _, r, term, trunc, info = env.step(policy(env))
        total += r
        done = term or trunc
    return total, info

import json

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from environment.baselines import random_policy, run_episode, scripted_policy
from environment.custom_env import ACTION_NAMES, OBS_NAMES, EntrepreneurshipEnv


def test_gymnasium_api():
    check_env(EntrepreneurshipEnv(), skip_render_check=True)


def test_spaces_and_bounds():
    env = EntrepreneurshipEnv()
    obs, _ = env.reset(seed=0)
    assert obs.shape == (len(OBS_NAMES),) and obs.dtype == np.float32
    assert env.action_space.n == len(ACTION_NAMES) == 11
    for _ in range(60):
        obs, *_rest = env.step(env.action_space.sample())
        assert obs.min() >= 0.0 and obs.max() <= 1.0
        if _rest[1] or _rest[2]:
            env.reset()


def test_seed_reproducibility():
    runs = []
    for _ in range(2):
        env = EntrepreneurshipEnv()
        env.reset(seed=123)
        rng = np.random.default_rng(0)
        tr = [env.step(int(rng.integers(11)))[1] for _ in range(30)]
        runs.append(tr)
    assert runs[0] == runs[1]


def test_different_seeds_differ():
    a, b = EntrepreneurshipEnv(), EntrepreneurshipEnv()
    oa, _ = a.reset(seed=1)
    ob, _ = b.reset(seed=2)
    assert not np.allclose(oa, ob)


@pytest.mark.parametrize("action", range(11))
def test_no_single_action_farms_reward(action):
    """Repeating one action for a whole episode must never be profitable."""
    env = EntrepreneurshipEnv()
    totals = []
    for s in range(10):
        env.reset(seed=s)
        done, tot = False, 0.0
        while not done:
            _, r, te, tr, _ = env.step(action)
            tot += r
            done = te or tr
        totals.append(tot)
    assert np.mean(totals) < 5.0


def test_invalid_actions_are_penalised():
    env = EntrepreneurshipEnv()
    env.reset(seed=0)
    _, r, *_ = env.step(7)          # hire before launching
    assert r < 0
    _, r, *_ = env.step(8)          # invest before launching
    assert r < 0


def test_launch_requires_preparation():
    env = EntrepreneurshipEnv()
    env.reset(seed=0)
    env.step(5)                     # digital service needs skill 0.5 / readiness 0.5
    assert not env.launched


def test_bankruptcy_terminates():
    env = EntrepreneurshipEnv()
    env.reset(seed=0)
    term = False
    for _ in range(env.max_steps):
        _, r, term, trunc, info = env.step(10)      # just wait: living costs drain the cash
        if term or trunc:
            break
    assert term and info["status"] == "BANKRUPT"


def test_task_is_solvable_and_not_trivial():
    env = EntrepreneurshipEnv()
    exp = [run_episode(env, scripted_policy, seed=s)[1]["status"] for s in range(40)]
    rnd = [run_episode(env, random_policy, seed=s)[1]["status"] for s in range(40)]
    assert exp.count("SUCCESS") >= 30          # an expert can solve it ...
    assert rnd.count("SUCCESS") <= 4           # ... random play essentially cannot


def test_state_json_serialisable():
    env = EntrepreneurshipEnv()
    env.reset(seed=0)
    env.step(0)
    d = json.loads(json.dumps(env.get_state_json()))
    assert d["step"] == 1 and set(OBS_NAMES) == set(d["observation"])


def test_headless_render_shape():
    env = EntrepreneurshipEnv(render_mode="rgb_array")
    env.reset(seed=0)
    env.step(0)
    frame = env.render()
    assert frame.ndim == 3 and frame.shape[2] == 3
    env.close()

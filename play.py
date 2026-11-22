"""play.py - watch the best trained agent (or a random one) in the Pygame GUI, with verbose terminal output.

    uv run play.py                      # best model chosen by the tuning sweep
    uv run play.py --algo ppo           # best model of one algorithm (dqn | reinforce | a2c | ppo)
    uv run play.py --random             # random actions, no model (environment demo)
    uv run play.py --episodes 3 --seed 7
    uv run play.py --headless --save-gif assets/agent_demo.gif   # off-screen recording
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from environment.custom_env import EntrepreneurshipEnv  # noqa: E402


def best_model_path(algo: str | None = None):
    meta = ROOT / "models" / "best_overall.json"
    if algo:
        p = ROOT / "models" / ("dqn" if algo == "dqn" else "pg") / f"best_{algo}.pt"
        return p, algo
    if meta.exists():
        m = json.loads(meta.read_text())
        return ROOT / m["path"], m["algo"]
    for a in ("ppo", "dqn", "a2c", "reinforce"):
        p, _ = best_model_path(a)
        if p.exists():
            return p, a
    return None, None


def run(algo=None, random_agent=False, episodes=3, seed=0, headless=False, gif=None, fps=6):
    from training.common import load_policy

    act, label = None, "Random"
    if not random_agent:
        path, name = best_model_path(algo)
        if path is None or not path.exists():
            print("No trained model found (models/*/best_*.pt). Falling back to random actions.\n"
                  "Train first with:  uv run main.py --train")
            random_agent = True
        else:
            act, label = load_policy(path), name.upper()
            print(f"Loaded {label} model: {path.relative_to(ROOT)}")

    env = EntrepreneurshipEnv(render_mode="rgb_array" if (headless or gif) else "human")
    if not (headless or gif):
        try:
            import pygame
            pygame.init()
            pygame.display.set_mode((16, 16))
        except Exception as e:                                   # no display available
            print(f"[no display: {e}] running in terminal-only mode")
            env = EntrepreneurshipEnv(render_mode=None)
    frames = []
    print("=" * 78)
    print(f"  Rwanda Entrepreneurship Journey Simulator   agent: {label}")
    print("=" * 78)
    for ep in range(1, episodes + 1):
        obs, info = env.reset(seed=seed + ep)
        done, total = False, 0.0
        print(f"\n--- Episode {ep} (seed {seed + ep}) ---")
        while not done:
            action = env.action_space.sample() if random_agent else act(obs)
            obs, r, te, tr, info = env.step(action)
            total += r
            done = te or tr
            print(f"  month {info['step']:3d} | {info['last_action']:23s} | skill {info['skill']:.2f} "
                  f"cash {info['capital']:.2f} debt {info['debt']:.2f} net {info['network']:.2f} "
                  f"ready {info['readiness']:.2f} health {info['health']:.2f} emp {info['employees']} "
                  f"| cf {info['cashflow']:+.3f} | R {r:+6.2f} {('| ' + info['event']) if info['event'] != '-' else ''}")
            if env.render_mode == "human":
                try:
                    import pygame
                    if any(e.type == pygame.QUIT for e in pygame.event.get()):
                        env.close(); return
                except Exception:
                    pass
            frame = env.render() if env.render_mode in ("human", "rgb_array") else None
            if gif and frame is not None:
                frames.append(frame)
        print(f"\n  RESULT: {info['status']}  | months: {info['step']} | employees: {info['employees']} "
              f"| return: {total:.1f}")
    env.close()
    if gif and frames:
        import imageio.v2 as imageio
        Path(gif).parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(gif, frames[::2] if len(frames) > 400 else frames, duration=1 / fps, loop=0)
        print(f"Saved {gif} ({len(frames)} frames)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--algo", choices=["dqn", "reinforce", "a2c", "ppo"], default=None)
    ap.add_argument("--random", action="store_true")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--save-gif", default=None)
    a = ap.parse_args()
    run(a.algo, a.random, a.episodes, a.seed, a.headless, a.save_gif)

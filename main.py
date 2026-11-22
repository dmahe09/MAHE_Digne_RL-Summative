"""Entry point.

    uv run main.py                  # watch the best trained agent (GUI + terminal output)
    uv run main.py --random         # random-action demo of the environment
    uv run main.py --train          # full study: 4 algorithms x 10 runs, then analysis (~1 h on 1 CPU core)
    uv run main.py --train --frames 20000      # quick smoke version of the study
    uv run main.py --analyze        # re-create tables / figures from the saved logs
    uv run main.py --api            # start the JSON API for a web / mobile front-end (needs: uv sync --extra api)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", action="store_true", help="run all hyper-parameter sweeps + analysis")
    ap.add_argument("--analyze", action="store_true", help="select best models, generalisation tests, figures")
    ap.add_argument("--api", action="store_true", help="serve the environment/agent as a JSON API")
    ap.add_argument("--random", action="store_true", help="random-action demo")
    ap.add_argument("--algo", choices=["dqn", "reinforce", "a2c", "ppo"], default=None)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--frames", type=int, default=None, help="frames per run when training")
    ap.add_argument("--headless", action="store_true", help="no window (terminal output only)")
    a = ap.parse_args()

    if a.train:
        from training.common import TOTAL_FRAMES
        from training.run_sweeps import run_all
        run_all(frames=a.frames or TOTAL_FRAMES)
        a.analyze = True
    if a.analyze:
        from training import analysis
        analysis.select_best()
        analysis.generalize()
        analysis.make_plots()
        analysis.make_screens_and_diagram()
        return
    if a.api:
        import uvicorn
        uvicorn.run("environment.api:app", host="127.0.0.1", port=8000)
        return
    import play
    play.run(algo=a.algo, random_agent=a.random, episodes=a.episodes, headless=a.headless)


if __name__ == "__main__":
    main()

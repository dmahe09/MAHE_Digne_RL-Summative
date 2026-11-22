"""Run (or resume) the full tuning study: 4 algorithms x 10 hyper-parameter runs on the same env and frame budget.
Finished runs (logs/<algo>_run_XX.json) are skipped, so the script can be interrupted and restarted."""
from __future__ import annotations

import argparse
import time

from training.common import LOGS, TOTAL_FRAMES
from training.configs import SWEEPS
from training.dqn_training import train_dqn
from training.pg_training import train_a2c, train_ppo, train_reinforce

FNS = {"dqn": train_dqn, "reinforce": train_reinforce, "a2c": train_a2c, "ppo": train_ppo}


def run_all(algos=("dqn", "reinforce", "a2c", "ppo"), frames=TOTAL_FRAMES, seed=0, force=False):
    t0 = time.time()
    for algo in algos:
        _, runs = SWEEPS[algo]
        for i, over in enumerate(runs, 1):
            if (LOGS / f"{algo}_run_{i:02d}.json").exists() and not force:
                print(f"[{algo.upper()} {i:02d}/10] already done - skipped", flush=True)
                continue
            print(f"[{algo.upper()} {i:02d}/10] {over or 'base config'}", flush=True)
            FNS[algo](over, run=i, seed=seed, total_frames=frames)
    print(f"sweeps finished in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--algos", nargs="*", default=["dqn", "reinforce", "a2c", "ppo"])
    ap.add_argument("--frames", type=int, default=TOTAL_FRAMES)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    run_all(a.algos, a.frames, a.seed, a.force)

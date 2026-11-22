"""Post-processing of the tuning study: pick best runs, summary tables, generalisation tests, multi-seed
confirmation and every figure used in the report.

    uv run python -m training.analysis --select      # best model per algorithm + sweep tables
    uv run python -m training.analysis --confirm     # retrain best config on 3 extra seeds
    uv run python -m training.analysis --generalize  # held-out seeds + shifted-economy scenarios
    uv run python -m training.analysis --plots       # all figures -> results/
"""
from __future__ import annotations

import argparse
import json
import shutil

import numpy as np
import pandas as pd

from training.common import (LOGS, MODELS, RESULTS, ROOT, TEST_SEED0, evaluate_policy, load_policy, smooth)
from training.configs import SWEEPS

ALGOS = ["dqn", "reinforce", "a2c", "ppo"]
LABEL = {"dqn": "DQN", "reinforce": "REINFORCE", "a2c": "A2C", "ppo": "PPO"}
COLOR = {"dqn": "#00b48a", "reinforce": "#5b9bd5", "a2c": "#9b59b6", "ppo": "#ed7d31"}
SCENARIOS = {
    "Nominal": {},
    "Recession (low demand)": dict(demand_mean=0.35, shock_prob=0.15),
    "Frequent shocks": dict(shock_prob=0.25),
    "Noisy market survey": dict(obs_noise=0.20),
}


def model_path(algo, name):
    return MODELS / ("dqn" if algo == "dqn" else "pg") / f"{name}.pt"


def load_runs(algo):
    out = []
    for i in range(1, 11):
        p = LOGS / f"{algo}_run_{i:02d}.json"
        if p.exists():
            out.append(json.loads(p.read_text()))
    return out


# ------------------------------------------------------------------ selection & tables
def select_best():
    rows, best = [], {}
    for algo in ALGOS:
        runs = load_runs(algo)
        if not runs:
            continue
        for d in runs:
            e, ep = d["eval"], d["episodes"]
            last = ep["ret"][-50:]
            row = {"algo": algo, "run": d["run"], **{k: v for k, v in d["cfg"].items()},
                   "eval_return": e["mean_return"], "eval_std": e["std_return"], "success": e["success_rate"],
                   "bankrupt": e["bankrupt_rate"], "steps_to_success": e["mean_steps_to_success"],
                   "train_return_last50": float(np.mean(last)) if last else np.nan,
                   "train_success_last50": float(np.mean(ep["succ"][-50:])) if last else np.nan,
                   "time_s": d["train_time_s"]}
            rows.append(row)
        b = max(runs, key=lambda d: (d["eval"]["mean_return"], d["eval"]["success_rate"]))
        best[algo] = b
        shutil.copy(model_path(algo, f"{algo}_run_{b['run']:02d}"), model_path(algo, f"best_{algo}"))
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "sweep_summary.csv", index=False)
    for algo in df.algo.unique():
        df[df.algo == algo].dropna(axis=1, how="all").to_csv(RESULTS / f"table_{algo}.csv", index=False)
    top = max(best.values(), key=lambda d: (d["eval"]["success_rate"], d["eval"]["mean_return"]))
    meta = {"algo": top["algo"], "run": top["run"],
            "path": str(model_path(top["algo"], f"best_{top['algo']}").relative_to(ROOT)),
            "best_runs": {a: d["run"] for a, d in best.items()}}
    (MODELS / "best_overall.json").write_text(json.dumps(meta, indent=2))
    print(df[["algo", "run", "eval_return", "success", "train_success_last50"]].to_string(index=False))
    print("best per algo:", meta["best_runs"], "| overall:", meta["algo"])
    return df, meta


# ------------------------------------------------------------------ multi-seed confirmation
def confirm(seeds=(101, 102, 103)):
    from training.dqn_training import train_dqn
    from training.pg_training import train_a2c, train_ppo, train_reinforce
    fns = {"dqn": train_dqn, "reinforce": train_reinforce, "a2c": train_a2c, "ppo": train_ppo}
    meta = json.loads((MODELS / "best_overall.json").read_text())
    rows = []
    for algo in ALGOS:
        cfg = SWEEPS[algo][1][meta["best_runs"][algo] - 1]
        for s in seeds:
            tag = f"confirm_{algo}_s{s}"
            if not (LOGS / f"{tag}.json").exists():
                print(f"[confirm] {algo} seed {s}", flush=True)
                fns[algo](cfg, run=0, seed=s, tag=tag)
            ev = evaluate_policy(load_policy(model_path(algo, tag)), 100, TEST_SEED0)
            rows.append({"algo": algo, "seed": s, **ev})
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "confirm_seeds.csv", index=False)
    print(df.groupby("algo")[["mean_return", "success_rate"]].agg(["mean", "std"]))
    return df


# ------------------------------------------------------------------ generalisation
def generalize(n=100):
    from environment.baselines import random_policy, scripted_policy
    from environment.custom_env import EntrepreneurshipEnv
    meta = json.loads((MODELS / "best_overall.json").read_text())
    agents = {LABEL[a]: load_policy(model_path(a, f"best_{a}")) for a in ALGOS if model_path(a, f"best_{a}").exists()}
    rows = []

    def baseline_eval(policy, kw):
        env = EntrepreneurshipEnv(**kw)
        res = {"ret": [], "ok": 0, "bank": 0}
        for i in range(n):
            env.reset(seed=TEST_SEED0 + i)
            done, tot = False, 0.0
            while not done:
                _, r, te, tr, info = env.step(policy(env))
                tot += r
                done = te or tr
            res["ret"].append(tot); res["ok"] += info["status"] == "SUCCESS"; res["bank"] += info["status"] == "BANKRUPT"
        return {"mean_return": float(np.mean(res["ret"])), "std_return": float(np.std(res["ret"])),
                "success_rate": res["ok"] / n, "bankrupt_rate": res["bank"] / n}

    for sc, kw in SCENARIOS.items():
        for name, act in agents.items():
            rows.append({"scenario": sc, "agent": name, **evaluate_policy(act, n, TEST_SEED0, **kw)})
        rows.append({"scenario": sc, "agent": "Scripted expert", **baseline_eval(scripted_policy, kw)})
        rows.append({"scenario": sc, "agent": "Random", **baseline_eval(random_policy, kw)})
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "generalization.csv", index=False)
    print(df.pivot(index="agent", columns="scenario", values="success_rate").round(2))
    return df


# ------------------------------------------------------------------ plots
def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False,
                         "axes.spines.right": False})
    return plt


def _best_run(algo):
    return json.loads((MODELS / "best_overall.json").read_text())["best_runs"][algo]


def _runs_plot(plt, fn, ylabel, title, fname, algos=ALGOS, ncols=2, logy=False, figsize=None, legend=True):
    n = len(algos)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize or (6.2 * ncols / 1.35, 3.0 * nrows), squeeze=False)
    for ax, algo in zip(axes.ravel(), algos):
        b = _best_run(algo)
        cmap = plt.get_cmap("tab10")
        for d in load_runs(algo):
            x, y = fn(d)
            if x is None or len(x) == 0:
                continue
            best = d["run"] == b
            ax.plot(x, y, color=COLOR[algo] if best else cmap((d["run"] - 1) % 10), lw=2.4 if best else 1.0,
                    alpha=1 if best else 0.7, label=f"run {d['run']}" + (" (best)" if best else ""),
                    zorder=5 if best else 2)
        ax.set_title(LABEL[algo], fontweight="bold", color=COLOR[algo])
        ax.set_xlabel("environment steps (months simulated)")
        ax.set_ylabel(ylabel)
        if logy:
            ax.set_yscale("log")
        if legend and algo == algos[0]:
            ax.legend(fontsize=6, ncol=2, loc="best")
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    fig.savefig(RESULTS / fname, dpi=170)
    plt.close(fig)


def make_plots():
    plt = _mpl()
    cum = lambda d: (np.array(d["episodes"]["frame"]), np.cumsum(d["episodes"]["ret"]))
    _runs_plot(plt, cum, "cumulative episode return", "Cumulative reward during training (10 runs per algorithm)",
               "fig_cumulative_reward.png")
    sm = lambda d: (np.array(d["episodes"]["frame"]), smooth(d["episodes"]["ret"], 50))
    _runs_plot(plt, sm, "episode return (50-episode mean)", "Learning curves (training episodes)",
               "fig_learning_curves.png", legend=False)
    ls = lambda d: (np.array(d["loss"]["frame"]), smooth(np.abs(d["loss"]["value"]) + 1e-6, 20))
    _runs_plot(plt, ls, "TD (Huber) loss", "DQN objective: temporal-difference loss", "fig_dqn_loss.png",
               algos=["dqn"], ncols=1, logy=True, figsize=(6.4, 3.3))
    en = lambda d: (np.array(d["entropy"]["frame"]), smooth(d["entropy"]["value"], 10))
    _runs_plot(plt, en, "policy entropy (nats)", "Policy entropy during training (max = ln 11 = 2.40)",
               "fig_entropy.png", algos=["reinforce", "a2c", "ppo"], ncols=3, figsize=(11.5, 3.2), legend=False)

    # convergence: rolling success of best run, and multi-seed band when available
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.4))
    reach = {}
    for algo in ALGOS:
        runs = load_runs(algo)
        if not runs:
            continue
        b = next(d for d in runs if d["run"] == _best_run(algo))
        x, y = np.array(b["episodes"]["frame"]), smooth(b["episodes"]["succ"], 50)
        axes[0].plot(x, y, color=COLOR[algo], lw=2, label=LABEL[algo])
        conf = [json.loads(p.read_text()) for p in sorted(LOGS.glob(f"confirm_{algo}_s*.json"))]
        grid = np.linspace(5000, runs[0]["episodes"]["frame"][-1], 60)
        curves = []
        for c in [b] + conf:
            cx, cy = np.array(c["episodes"]["frame"]), smooth(c["episodes"]["succ"], 50)
            ok = ~np.isnan(cy)
            curves.append(np.interp(grid, cx[ok], cy[ok], left=np.nan))
        arr = np.array(curves)
        reach[algo] = []
        for c in curves:
            idx = np.where(c >= 0.8)[0]
            reach[algo].append(grid[idx[0]] if len(idx) else np.nan)
    axes[0].set_xlabel("environment steps"); axes[0].set_ylabel("success rate (50-episode window)")
    axes[0].set_title("Convergence of the best run per algorithm"); axes[0].legend(); axes[0].set_ylim(-0.02, 1.02)
    xs = np.arange(len(reach))
    means = [np.nanmean(v) if not np.all(np.isnan(v)) else np.nan for v in reach.values()]
    stds = [np.nanstd(v) if not np.all(np.isnan(v)) else 0 for v in reach.values()]
    never = [np.all(np.isnan(v)) for v in reach.values()]
    top = max([m for m in means if not np.isnan(m)] or [1])
    axes[1].bar(xs, [top * 1.05 if nv else m for m, nv in zip(means, never)], yerr=[0 if nv else s for s, nv in zip(stds, never)],
                color=[COLOR[a] for a in reach], capsize=4, hatch=None)
    for i, nv in enumerate(never):
        if nv:
            axes[1].text(i, top * 0.5, "never\nreached", ha="center", fontsize=8)
            axes[1].patches[i].set_alpha(0.25)
    for i, v in enumerate(reach.values()):
        k = int(np.sum(~np.isnan(v)))
        axes[1].text(i, (top * 1.05 if never[i] else means[i]) * 1.02, f"{k}/{len(v)} runs\nreached", ha="center",
                     va="bottom", fontsize=7)
    axes[1].set_ylim(0, top * 1.3)
    axes[1].set_xticks(xs); axes[1].set_xticklabels([LABEL[a] for a in reach])
    axes[1].set_ylabel("steps until 80% rolling success"); axes[1].set_title("Steps to 80% rolling success (runs that reached it)")
    fig.tight_layout(); fig.savefig(RESULTS / "fig_convergence.png", dpi=170); plt.close(fig)

    # all runs: final eval success per run
    df = pd.read_csv(RESULTS / "sweep_summary.csv")
    fig, ax = plt.subplots(figsize=(10, 3.2))
    w = 0.2
    for k, algo in enumerate(ALGOS):
        s = df[df.algo == algo].sort_values("run")
        ax.bar(s.run + (k - 1.5) * w, s.success * 100, width=w, color=COLOR[algo], label=LABEL[algo])
    ax.set_xticks(range(1, 11)); ax.set_xlabel("hyper-parameter run #"); ax.set_ylabel("success rate on 30 eval episodes (%)")
    ax.set_title("Final evaluation success of every tuning run"); ax.legend(ncol=4, fontsize=8)
    fig.tight_layout(); fig.savefig(RESULTS / "fig_all_runs.png", dpi=170); plt.close(fig)

    # generalisation
    gp = RESULTS / "generalization.csv"
    if gp.exists():
        g = pd.read_csv(gp)
        agents = [a for a in ["DQN", "REINFORCE", "A2C", "PPO", "Scripted expert", "Random"] if a in set(g.agent)]
        fig, ax = plt.subplots(figsize=(10, 3.4))
        w = 0.8 / len(agents)
        cols = {**{LABEL[a]: COLOR[a] for a in ALGOS}, "Scripted expert": "#555555", "Random": "#bbbbbb"}
        for k, a in enumerate(agents):
            s = g[g.agent == a].set_index("scenario").loc[list(SCENARIOS)]
            ax.bar(np.arange(len(SCENARIOS)) + (k - (len(agents) - 1) / 2) * w, s.success_rate * 100, width=w,
                   color=cols[a], label=a)
        ax.set_xticks(range(len(SCENARIOS))); ax.set_xticklabels(list(SCENARIOS)); ax.set_ylabel("success rate (%)")
        ax.set_title("Generalisation: 100 unseen seeds per scenario (best model of each algorithm)")
        ax.legend(ncol=6, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.12))
        fig.tight_layout(); fig.savefig(RESULTS / "fig_generalization.png", dpi=170); plt.close(fig)
    print("figures written to results/")


def make_screens_and_diagram():
    """Environment screenshots (off-screen pygame) and the agent-environment diagram."""
    import os
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    from environment.custom_env import EntrepreneurshipEnv
    from environment.baselines import scripted_policy
    env = EntrepreneurshipEnv(render_mode="rgb_array")
    env.reset(seed=3)
    done, shots = False, {}
    while not done:
        _, _, te, tr, info = env.step(scripted_policy(env))
        env.render()
        done = te or tr
        st = info["step"]
        if info["last_action"].startswith("Enroll") and "start" not in shots:
            shots["start"] = st
            env._renderer.save(str(ROOT / "assets" / "screen_1_training.png"))
        if info["business_launched"] and "launch" not in shots:
            shots["launch"] = st
            env._renderer.save(str(ROOT / "assets" / "screen_2_launch.png"))
        if info["status"] == "SUCCESS":
            env._renderer.save(str(ROOT / "assets" / "screen_3_success.png"))
    env.close()
    plt = _mpl()
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(9, 3.6)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 4)
    def box(x, y, w, h, txt, c):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", fc=c, ec="#333", lw=1.2))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=8.5)
    box(0.2, 1.2, 2.6, 1.6, "AGENT\nPolicy network\n14 inputs - 64 - 64 - 11 outputs\n(DQN / REINFORCE / A2C / PPO)", "#dff3ec")
    box(6.6, 0.6, 3.2, 2.8, "ENVIRONMENT (1 step = 1 month)\nskill - capital - network - readiness\ndebt - health - employees\nstochastic demand, shocks, loan approval\nlaunch / payroll / revenue dynamics", "#fdeccd")
    ax.annotate("", xy=(6.6, 2.5), xytext=(2.8, 2.5), arrowprops=dict(arrowstyle="->", lw=1.6))
    ax.text(4.7, 2.7, "action a_t (0-10)", ha="center", fontsize=8.5)
    ax.annotate("", xy=(2.8, 1.5), xytext=(6.6, 1.5), arrowprops=dict(arrowstyle="->", lw=1.6, color="#c0392b"))
    ax.text(4.7, 1.05, "observation o_t+1 (14-d, noisy)\nreward r_t+1", ha="center", fontsize=8.5, color="#c0392b")
    ax.text(5, 3.7, "Reinforcement-learning loop", ha="center", fontweight="bold")
    ax.text(5, 0.1, "Training (TorchRL): Collector -> replay buffer / GAE -> loss module -> Adam update", ha="center", fontsize=8, style="italic")
    fig.savefig(ROOT / "assets" / "agent_environment_diagram.png", dpi=170, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for f in ("select", "confirm", "generalize", "plots", "screens"):
        ap.add_argument(f"--{f}", action="store_true")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    if a.select or a.all:
        select_best()
    if a.confirm or a.all:
        confirm()
    if a.generalize or a.all:
        generalize()
    if a.plots or a.all:
        make_plots()
    if a.screens or a.all:
        make_screens_and_diagram()

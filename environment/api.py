"""Minimal HTTP API around the environment/agent so a web or mobile front-end can drive and display it.

    uv sync --extra api
    uv run uvicorn environment.api:app --port 8000      # then open assets/dashboard.html

POST /reset                 -> start a new episode, returns JSON state
POST /step {"action": 4}    -> apply one action (0-10)
POST /agent_step            -> let the trained agent pick and apply the action
GET  /state                 -> current JSON state;   GET /actions -> action id -> name
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from environment.custom_env import ACTION_NAMES, EntrepreneurshipEnv  # noqa: E402

app = FastAPI(title="Rwanda Entrepreneurship RL API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_env = EntrepreneurshipEnv()
_obs, _ = _env.reset(seed=0)
_agent = None


class StepIn(BaseModel):
    action: int


def _payload(reward=0.0, terminated=False, truncated=False):
    return {"reward": round(float(reward), 3), "terminated": terminated, "truncated": truncated,
            "state": _env.get_state_json()}


def _load_agent():
    global _agent
    if _agent is None:
        from play import best_model_path
        from training.common import load_policy
        path, _ = best_model_path()
        if path is None or not path.exists():
            raise HTTPException(503, "no trained model found - run `uv run main.py --train` first")
        _agent = load_policy(path)
    return _agent


@app.get("/actions")
def actions():
    return {i: n for i, n in enumerate(ACTION_NAMES)}


@app.get("/state")
def state():
    return _payload()


@app.post("/reset")
def reset(seed: int | None = None):
    global _obs
    _obs, _ = _env.reset(seed=seed)
    return _payload()


@app.post("/step")
def step(body: StepIn):
    global _obs
    if not 0 <= body.action < len(ACTION_NAMES):
        raise HTTPException(422, "action out of range")
    _obs, r, te, tr, _ = _env.step(body.action)
    return _payload(r, te, tr)


@app.post("/agent_step")
def agent_step():
    global _obs
    a = _load_agent()(_obs)
    _obs, r, te, tr, _ = _env.step(a)
    out = _payload(r, te, tr)
    out["agent_action"] = {"id": a, "name": ACTION_NAMES[a]}
    return out

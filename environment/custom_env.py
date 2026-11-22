"""
custom_env.py - Rwanda Entrepreneurship Journey Simulator (v4)

A sequential, stochastic Gymnasium environment. One step = one month.
The agent is a young Rwandan who must go from "no skills, little capital" to a
*sustainable micro-business that employs people*, in at most `max_steps` months.

Why this needs RL (and is not a script):
  * State persists and every action changes it (skill, capital, debt, health ...).
  * Trade-offs are delayed: training/mentoring/planning only pay off at launch,
    investing pays off over months, hiring raises payroll *now* for revenue *later*,
    loans raise capital today but accrue interest.
  * The market is stochastic (demand mean-reverts, random shocks hit) and only
    observed through a noisy survey, so the best action sequence is state-dependent.

Reward design (aligned with the objective, not farmable):
  * progress shaping is the *increase* of bounded stocks (skill, network, readiness),
    so total shaping is capped;
  * money enters only as the change in net worth (capital - debt), so borrowing is
    reward-neutral and spending is penalised;
  * launch (+10) and the first three hires (+6 each) are one-off events;
  * success (+30) ends the episode, bankruptcy (-15) ends the episode.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

ACTION_NAMES = [
    "Enroll in Training",      # 0
    "Seek Mentorship",         # 1
    "Apply for Micro-loan",    # 2
    "Develop Business Plan",   # 3
    "Launch Retail Shop",      # 4
    "Launch Digital Service",  # 5
    "Launch Education Centre", # 6
    "Hire Employee",           # 7
    "Invest in Business",      # 8
    "Repay Loan",              # 9
    "Wait / Operate",          # 10
]

OBS_NAMES = [
    "skill", "capital", "network", "readiness", "debt", "demand_survey",
    "health", "employees", "cashflow", "time_left", "launched",
    "mkt_retail", "mkt_digital", "mkt_education",
]

MARKET_NAMES = {0: "No Market", 1: "Retail & Trade", 2: "Digital Services", 3: "Education & Training"}

# Per-market parameters (index = action id 4/5/6)
MARKETS = {
    4: dict(id=1, rate=0.070, vol=0.04, cost=0.10, readiness=0.35, skill=0.00, network=0.00),
    5: dict(id=2, rate=0.105, vol=0.07, cost=0.15, readiness=0.50, skill=0.50, network=0.00),
    6: dict(id=3, rate=0.085, vol=0.03, cost=0.08, readiness=0.40, skill=0.00, network=0.30),
}

# Economic constants (all money values are fractions of a 1.0 = ~5M RWF capital scale)
LIVING_COST = 0.004      # monthly burn before launch
OPEX = 0.012             # monthly fixed cost of running the business
PAYROLL = 0.015          # monthly cost per employee
INTEREST = 0.012         # monthly interest on outstanding debt
HIRE_COST = 0.04         # one-off onboarding cost
TRAIN_COST = 0.025
MENTOR_COST = 0.005
PLAN_COST = 0.005
INVEST_COST = 0.04
RESERVE = 0.05           # minimum cash buffer needed to launch
MAX_DEBT = 0.50
HEALTH_DECAY = 0.025
SUSTAIN_MONTHS = 9       # consecutive months of non-negative cash flow needed to win
WIN_EMPLOYEES = 2
WIN_MAX_DEBT = 0.20      # loans must be largely repaid for the business to count as sustainable


class EntrepreneurshipEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 6}

    def __init__(self, render_mode=None, max_steps: int = 120, shock_prob: float = 0.07,
                 demand_mean: float = 0.55, obs_noise: float = 0.05,
                 demand_spread: float = 0.12):
        super().__init__()
        self.max_steps = max_steps
        self.shock_prob = shock_prob
        self.demand_mean = demand_mean
        self.obs_noise = obs_noise
        self.demand_spread = demand_spread
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(len(ACTION_NAMES))
        self.observation_space = spaces.Box(0.0, 1.0, shape=(len(OBS_NAMES),), dtype=np.float32)
        self._renderer = None
        self._init_vars()

    # ------------------------------------------------------------------ helpers
    def _init_vars(self):
        self.t = 0
        self.skill = self.capital = self.network = self.readiness = 0.0
        self.debt = self.health = self.cashflow = 0.0
        self.employees = 0
        self.market = 0
        self.market_key = None
        self.demand = self.mu = self.demand_mean
        self.sustain = 0
        self.launched = False
        self.event = "-"
        self.status = "IN_PROGRESS"
        self.last_action = "-"
        self.last_reward = 0.0
        self.total_reward = 0.0
        self._demand_seen = self.demand_mean

    def _obs(self) -> np.ndarray:
        self._demand_seen = float(np.clip(self.demand + self.np_random.normal(0, self.obs_noise), 0, 1))
        mk = np.zeros(3, dtype=np.float32)
        if self.market:
            mk[self.market - 1] = 1.0
        o = np.array([
            self.skill, self.capital, self.network, self.readiness, self.debt,
            self._demand_seen, self.health, self.employees / 5.0,
            np.clip((self.cashflow + 0.05) / 0.15, 0, 1),
            (self.max_steps - self.t) / self.max_steps,
            1.0 if self.launched else 0.0, *mk,
        ], dtype=np.float32)
        return np.clip(o, 0.0, 1.0)

    def _info(self) -> dict:
        return {
            "step": self.t, "max_steps": self.max_steps, "last_action": self.last_action,
            "last_reward": self.last_reward, "total_reward": self.total_reward,
            "skill": self.skill, "capital": self.capital, "network": self.network,
            "readiness": self.readiness, "debt": self.debt, "demand": self.demand,
            "health": self.health, "employees": self.employees, "cashflow": self.cashflow,
            "sustain_months": self.sustain, "business_launched": self.launched,
            "market": MARKET_NAMES[self.market], "event": self.event, "status": self.status,
            "success": self.status == "SUCCESS",
        }

    def get_state_json(self) -> dict:
        """JSON-serialisable snapshot for a web/mobile front-end."""
        d = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in self._info().items()}
        d["observation"] = {n: round(float(v), 4) for n, v in zip(OBS_NAMES, self._obs())}
        return d

    def action_mask(self) -> np.ndarray:
        """True where the action can have an effect (for demos / analysis only)."""
        m = np.ones(len(ACTION_NAMES), dtype=bool)
        m[0] = self.skill < 1.0 and self.capital >= TRAIN_COST
        m[1] = self.skill >= 0.1 and self.network < 1.0
        m[2] = self.debt < MAX_DEBT
        m[3] = self.skill >= 0.2 and self.readiness < 1.0
        for a in (4, 5, 6):
            m[a] = not self.launched
        m[7] = self.launched and self.employees < 5 and self.capital >= HIRE_COST + 0.04
        m[8] = self.launched and self.health < 0.85 and self.capital >= INVEST_COST
        m[9] = self.debt > 0 and self.capital > 0.05
        m[10] = True
        return m

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._init_vars()
        rng = self.np_random
        self.skill = float(rng.uniform(0.0, 0.08))
        self.capital = float(rng.uniform(0.20, 0.35))
        self.network = float(rng.uniform(0.0, 0.05))
        # each episode is a different local economy: demand level varies between episodes
        self.mu = float(np.clip(self.demand_mean + rng.normal(0, self.demand_spread), 0.25, 0.85))
        self.demand = float(np.clip(self.mu + rng.normal(0, 0.05), 0.05, 0.95))
        return self._obs(), self._info()

    def step(self, action):
        action = int(action)
        assert self.action_space.contains(action), f"invalid action {action}"
        rng = self.np_random
        self.t += 1
        self.last_action = ACTION_NAMES[action]
        self.event = "-"
        reward = -0.05
        s0, n0, r0 = self.skill, self.network, self.readiness
        networth0 = self.capital - self.debt

        # ---------------- agent decision
        if action == 0:                                    # training
            if self.skill < 1.0 and self.capital >= TRAIN_COST:
                self.skill = min(1.0, self.skill + rng.uniform(0.06, 0.12) * (1 - 0.5 * self.skill))
                self.capital -= TRAIN_COST
            else:
                reward -= 0.5
        elif action == 1:                                  # mentorship
            if self.skill >= 0.1 and self.network < 1.0 and self.capital >= MENTOR_COST:
                self.network = min(1.0, self.network + rng.uniform(0.05, 0.10) * (1 - 0.3 * self.network))
                self.capital -= MENTOR_COST
            else:
                reward -= 0.5
        elif action == 2:                                  # micro-loan (stochastic approval)
            if self.debt >= MAX_DEBT:
                reward -= 0.5
            else:
                elig = (self.skill + self.network + self.readiness) / 3 + (0.15 if self.launched else 0.0)
                p = float(np.clip(0.10 + 0.80 * elig, 0.05, 0.90))
                if rng.random() < p:
                    amt = min(float(rng.uniform(0.12, 0.25)), MAX_DEBT - self.debt)
                    self.capital = min(1.0, self.capital + amt)
                    self.debt += amt
                    self.event = f"Loan approved (+{amt:.2f})"
                else:
                    reward -= 0.3
                    self.event = "Loan rejected"
        elif action == 3:                                  # business plan
            if self.skill >= 0.2 and self.readiness < 1.0 and self.capital >= PLAN_COST:
                self.readiness = min(1.0, self.readiness + rng.uniform(0.07, 0.14))
                self.capital -= PLAN_COST
            else:
                reward -= 0.5
        elif action in (4, 5, 6):                          # launch in a market
            if self.launched:
                reward -= 0.5
            else:
                mk = MARKETS[action]
                ok = [self.readiness >= mk["readiness"], self.skill >= mk["skill"],
                      self.network >= mk["network"], self.capital >= mk["cost"] + RESERVE]
                if all(ok):
                    self.launched = True
                    self.market, self.market_key = mk["id"], action
                    self.capital -= mk["cost"]
                    self.health = min(1.0, 0.45 + 0.30 * (self.skill + self.network) / 2 + 0.2 * self.readiness)
                    reward += 10.0
                    self.event = f"Launched: {MARKET_NAMES[self.market]}"
                else:
                    reward -= 0.5 * (4 - sum(ok))
        elif action == 7:                                  # hire
            if self.launched and self.employees < 5 and self.capital >= HIRE_COST + 0.04:
                self.employees += 1
                self.capital -= HIRE_COST
                self.health = min(1.0, self.health + 0.03)
                if self.employees <= 3:
                    reward += 6.0
            else:
                reward -= 0.5
        elif action == 8:                                  # invest
            if self.launched and self.health < 0.85 and self.capital >= INVEST_COST:
                self.capital -= INVEST_COST
                self.health = min(1.0, self.health + rng.uniform(0.12, 0.22))
            else:
                reward -= 0.5
        elif action == 9:                                  # repay
            if self.debt > 0 and self.capital > 0.05:
                pay = min(self.debt, self.capital - 0.03, 0.10)
                self.capital -= pay
                self.debt -= pay
            else:
                reward -= 0.3
        # action 10: wait / operate - no direct effect, the month simply passes

        # ---------------- world dynamics (one month passes)
        vol = MARKETS[self.market_key]["vol"] if self.market_key else 0.04
        self.demand += 0.15 * (self.mu - self.demand) + rng.normal(0, vol)
        if rng.random() < self.shock_prob:
            if rng.random() < 0.6:
                self.demand -= 0.30
                self.event = "Market downturn"
            else:
                self.demand += 0.20
                self.event = "Demand boom"
        self.demand = float(np.clip(self.demand, 0.0, 1.0))

        interest = INTEREST * self.debt
        if self.launched:
            mk = MARKETS[self.market_key]
            revenue = (mk["rate"] * self.health * (0.4 + 1.2 * self.demand)
                       * (0.8 + 0.4 * self.network) * (1 + 0.2 * self.employees))
            self.cashflow = revenue - OPEX - PAYROLL * self.employees - interest
            self.health = max(0.0, self.health - HEALTH_DECAY - (0.01 if self.demand < 0.3 else 0.0))
        else:
            self.cashflow = -LIVING_COST - interest
        self.capital = float(np.clip(self.capital + self.cashflow, -0.2, 1.0))

        # ---------------- reward
        reward += 3.0 * (self.skill - s0) + 2.0 * (self.network - n0) + 3.0 * (self.readiness - r0)
        reward += 4.0 * ((self.capital - self.debt) - networth0)

        # ---------------- termination
        if (self.launched and self.employees >= WIN_EMPLOYEES and self.cashflow >= 0
                and self.debt <= WIN_MAX_DEBT):
            self.sustain += 1
            reward += 0.3
        else:
            self.sustain = 0
        terminated = truncated = False
        if self.sustain >= SUSTAIN_MONTHS:
            reward += 30.0
            self.status, terminated = "SUCCESS", True
        elif self.capital <= 0.0:
            reward -= 15.0
            self.status, terminated = "BANKRUPT", True
        elif self.t >= self.max_steps:
            self.status, truncated = "TIMEOUT", True
        elif self.launched:
            self.status = "LAUNCHED"

        self.last_reward = float(reward)
        self.total_reward += float(reward)
        if self.render_mode == "human":
            self.render()
        return self._obs(), float(reward), terminated, truncated, self._info()

    # ------------------------------------------------------------------ render
    def render(self):
        if self.render_mode is None:
            return None
        if self._renderer is None:
            from environment.rendering import EntrepreneurshipRenderer
            self._renderer = EntrepreneurshipRenderer(headless=(self.render_mode == "rgb_array"))
        return self._renderer.draw(self._obs(), self._info(), human=(self.render_mode == "human"))

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

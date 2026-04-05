"""
AI Micro-Investment & Financial Stability Agent for African Users
Custom Gymnasium Environment

State Space: income, savings, expenses, debt, inflation_rate, investment_portfolio,
             month, economic_shock, financial_stress_index
Action Space: Discrete(8) — allocate income across different financial decisions
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class AfricanFinanceEnv(gym.Env):
    """
    A custom Gymnasium environment simulating financial decision-making
    for a low-to-middle income African household.

    The agent must allocate monthly income across:
    - Savings
    - Basic expenses
    - Investments (mobile money funds, stock market, crypto, real estate)
    - Debt repayment
    - Emergency fund contributions
    - Education/skill development

    Observations (9-dim continuous):
    - income_normalized       [0, 1]  — monthly income / max_income
    - savings_ratio           [0, 1]  — savings / income
    - expense_ratio           [0, 1]  — expenses / income
    - debt_ratio              [0, 1]  — debt / (income * 12)
    - inflation_rate          [0, 1]  — current monthly inflation (0–15%)
    - investment_value        [0, 1]  — portfolio value / target
    - month_progress          [0, 1]  — month in year / 12
    - economic_shock          [0, 1]  — binary shock indicator
    - financial_stress        [0, 1]  — composite stress index

    Actions (Discrete 8):
    0: Conservative Save     — 60% savings, 30% expenses, 10% investments
    1: Balanced Allocate     — 30% savings, 40% expenses, 30% investments
    2: Aggressive Invest     — 10% savings, 30% expenses, 60% investments
    3: Debt Repayment Focus  — 10% savings, 30% expenses, 20% invest, 40% debt
    4: Emergency Fund Build  — 50% emergency, 30% expenses, 20% savings
    5: Education/Upskill     — 20% education, 40% expenses, 40% savings
    6: Mobile Money Invest   — 40% mobile money, 30% expenses, 30% savings
    7: Survival Mode         — 70% expenses, 20% savings, 10% emergency
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, render_mode=None):
        super().__init__()

        self.render_mode = render_mode

        # Environment constants (in USD equivalent for generality)
        self.max_income = 2000.0
        self.base_income = 400.0        # ~$400/month baseline
        self.max_steps = 60             # 5-year simulation
        self.inflation_base = 0.08      # 8% annual baseline (Africa avg)
        self.investment_target = 10000.0

        # Action space: 8 discrete financial strategies
        self.action_space = spaces.Discrete(8)

        # Observation space: 9 continuous features
        self.observation_space = spaces.Box(
            low=np.zeros(9, dtype=np.float32),
            high=np.ones(9, dtype=np.float32),
            dtype=np.float32
        )

        # Action allocation matrices [savings, expenses, investments, debt, emergency, education]
        self.action_allocations = {
            0: {"savings": 0.60, "expenses": 0.30, "investments": 0.10, "debt": 0.00, "emergency": 0.00, "education": 0.00},
            1: {"savings": 0.30, "expenses": 0.40, "investments": 0.30, "debt": 0.00, "emergency": 0.00, "education": 0.00},
            2: {"savings": 0.10, "expenses": 0.30, "investments": 0.60, "debt": 0.00, "emergency": 0.00, "education": 0.00},
            3: {"savings": 0.10, "expenses": 0.30, "investments": 0.20, "debt": 0.40, "emergency": 0.00, "education": 0.00},
            4: {"savings": 0.20, "expenses": 0.30, "investments": 0.00, "debt": 0.00, "emergency": 0.50, "education": 0.00},
            5: {"savings": 0.40, "expenses": 0.40, "investments": 0.00, "debt": 0.00, "emergency": 0.00, "education": 0.20},
            6: {"savings": 0.30, "expenses": 0.40, "investments": 0.30, "debt": 0.00, "emergency": 0.00, "education": 0.00},  # mobile money
            7: {"savings": 0.20, "expenses": 0.70, "investments": 0.00, "debt": 0.00, "emergency": 0.10, "education": 0.00},
        }

        self.renderer = None
        self.reset()

    def _get_obs(self):
        return np.array([
            np.clip(self.income / self.max_income, 0, 1),
            np.clip(self.savings / (self.income * 12 + 1e-9), 0, 1),
            np.clip(self.monthly_expenses / (self.income + 1e-9), 0, 1),
            np.clip(self.debt / (self.income * 12 + 1e-9), 0, 1),
            np.clip(self.inflation_rate / 0.20, 0, 1),
            np.clip(self.investment_value / self.investment_target, 0, 1),
            self.step_count / self.max_steps,
            float(self.economic_shock_active),
            np.clip(self.financial_stress, 0, 1),
        ], dtype=np.float32)

    def _get_info(self):
        return {
            "income": self.income,
            "savings": self.savings,
            "debt": self.debt,
            "investment_value": self.investment_value,
            "emergency_fund": self.emergency_fund,
            "inflation_rate": self.inflation_rate,
            "financial_stress": self.financial_stress,
            "step": self.step_count,
            "net_worth": self.savings + self.investment_value + self.emergency_fund - self.debt,
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        rng = self.np_random

        # Initialize financial state
        self.income = self.base_income + rng.uniform(-100, 300)
        self.savings = rng.uniform(0, 200)
        self.debt = rng.uniform(0, 1000)
        self.investment_value = rng.uniform(0, 300)
        self.emergency_fund = rng.uniform(0, 100)
        self.monthly_expenses = self.income * rng.uniform(0.35, 0.55)
        self.inflation_rate = self.inflation_base + rng.uniform(-0.03, 0.05)
        self.economic_shock_active = False
        self.shock_duration = 0
        self.financial_stress = 0.3
        self.education_level = 0.0      # bonus multiplier from education actions
        self.step_count = 0
        self.cumulative_reward = 0.0
        self.consecutive_stable_months = 0

        return self._get_obs(), self._get_info()

    def step(self, action):
        assert self.action_space.contains(action)
        alloc = self.action_allocations[action]

        # ── Income with income growth (education boosts income) ──────────
        income_growth = 0.002 + self.education_level * 0.003
        self.income = min(self.income * (1 + income_growth), self.max_income)

        # ── Economic shock logic ─────────────────────────────────────────
        shock_prob = 0.06 if not self.economic_shock_active else 0.0
        if self.np_random.random() < shock_prob:
            self.economic_shock_active = True
            self.shock_duration = int(self.np_random.integers(1, 5))
            self.income *= self.np_random.uniform(0.3, 0.7)   # income drop
            self.inflation_rate = min(self.inflation_rate + self.np_random.uniform(0.02, 0.08), 0.20)

        if self.economic_shock_active:
            self.shock_duration -= 1
            if self.shock_duration <= 0:
                self.economic_shock_active = False
                self.income = self.base_income + self.np_random.uniform(-50, 200)
                self.inflation_rate = max(self.inflation_rate - 0.03, self.inflation_base)

        # ── Allocate income ──────────────────────────────────────────────
        available = self.income

        savings_add = available * alloc["savings"]
        expense_spend = available * alloc["expenses"]
        invest_add = available * alloc["investments"]
        debt_pay = available * alloc["debt"]
        emergency_add = available * alloc["emergency"]
        education_add = available * alloc["education"]

        # Handle action 6: mobile money (lower risk, lower return)
        if action == 6:
            mobile_money = available * 0.40
            invest_add = 0
            mobile_return = mobile_money * self.np_random.uniform(0.005, 0.012)
            self.savings += mobile_money + mobile_return

        # ── Update state ─────────────────────────────────────────────────
        self.savings += savings_add
        self.debt = max(0, self.debt - debt_pay)
        self.emergency_fund += emergency_add
        self.education_level = min(1.0, self.education_level + education_add / 5000)

        # Investment returns (volatile — stock/crypto/real estate mix)
        inv_return_rate = self.np_random.normal(0.008, 0.04)   # monthly mean 0.8%, std 4%
        inv_return_rate = max(inv_return_rate, -0.15)           # max 15% monthly loss
        self.investment_value = max(0, self.investment_value * (1 + inv_return_rate) + invest_add)

        # Inflation erodes savings purchasing power
        real_savings_erosion = self.savings * (self.inflation_rate / 12)
        self.savings = max(0, self.savings - real_savings_erosion)

        # Expenses must be covered
        self.monthly_expenses = self.income * self.np_random.uniform(0.30, 0.55)
        expense_deficit = max(0, self.monthly_expenses - expense_spend)
        # If can't cover expenses, draw from savings or take on debt
        if expense_deficit > 0:
            if self.savings >= expense_deficit:
                self.savings -= expense_deficit
            else:
                self.debt += expense_deficit - self.savings
                self.savings = 0

        # Interest on debt (20% annual — common in African microfinance)
        self.debt *= (1 + 0.20 / 12)

        # ── Compute financial stress ─────────────────────────────────────
        debt_stress = min(self.debt / (self.income * 12 + 1e-9), 1.0)
        savings_stress = max(0, 1 - self.savings / (self.income * 3))
        inflation_stress = min(self.inflation_rate / 0.20, 1.0)
        self.financial_stress = 0.4 * debt_stress + 0.4 * savings_stress + 0.2 * inflation_stress

        # ── Reward function ──────────────────────────────────────────────
        net_worth = self.savings + self.investment_value + self.emergency_fund - self.debt
        net_worth_reward = np.tanh(net_worth / 5000) * 5.0

        # Penalize high debt
        debt_penalty = -min(self.debt / 500, 3.0)

        # Reward low financial stress
        stability_reward = (1 - self.financial_stress) * 2.0

        # Reward investment growth
        invest_reward = np.tanh(self.investment_value / self.investment_target) * 3.0

        # Reward having emergency fund (at least 3 months expenses)
        emergency_reward = 1.0 if self.emergency_fund >= self.monthly_expenses * 3 else -0.5

        # Penalize inability to cover expenses
        survival_penalty = -5.0 if expense_deficit > self.income * 0.3 else 0.0

        # Shock survival bonus
        shock_bonus = 1.0 if self.economic_shock_active and self.savings > self.monthly_expenses else 0.0

        reward = (net_worth_reward + debt_penalty + stability_reward +
                  invest_reward + emergency_reward + survival_penalty + shock_bonus)

        # Stability streak bonus
        if self.financial_stress < 0.4:
            self.consecutive_stable_months += 1
            if self.consecutive_stable_months >= 6:
                reward += 2.0  # bonus for sustained stability
        else:
            self.consecutive_stable_months = 0

        self.step_count += 1
        self.cumulative_reward += reward

        # ── Terminal conditions ───────────────────────────────────────────
        terminated = False
        truncated = False

        # Bankruptcy: debt > 3x annual income
        if self.debt > self.income * 36:
            terminated = True
            reward -= 20.0   # severe penalty

        # Financial freedom: investment_value > target AND debt < income
        if self.investment_value >= self.investment_target and self.debt < self.income:
            terminated = True
            reward += 30.0   # large final bonus

        # Time limit
        if self.step_count >= self.max_steps:
            truncated = True

        return self._get_obs(), float(reward), terminated, truncated, self._get_info()

    def render(self):
        if self.render_mode == "human":
            if self.renderer is None:
                from rendering import FinanceRenderer
                self.renderer = FinanceRenderer()
            self.renderer.render(self._get_info(), self.step_count, self.action_allocations)
        elif self.render_mode == "rgb_array":
            if self.renderer is None:
                from rendering import FinanceRenderer
                self.renderer = FinanceRenderer(headless=True)
            return self.renderer.get_rgb_array(self._get_info(), self.step_count)

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

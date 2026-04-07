#  AI Micro-Investment & Financial Stability Agent for African Users

> A reinforcement learning agent that learns optimal financial decision-making strategies for low-to-middle income African households — managing income allocation, debt reduction, investment growth, and survival through economic shocks.

---

##  Problem Statement

Over **60% of sub-Saharan African adults** lack access to formal financial planning tools. Volatile inflation (8–20% annually), unpredictable income, and microfinance debt traps create extreme financial instability. This project trains an RL agent to act as a **personal financial advisor**, learning to:

- Allocate monthly income across competing priorities
- Grow investments despite inflation
- Survive economic shocks (e.g., currency devaluations, crop failures)
- Build an emergency fund buffer
- Reduce high-interest microfinance debt

---

##  Project Structure

```
student_name_rl_summative/
├── environment/
│   ├── custom_env.py       # AfricanFinanceEnv (Gymnasium)
│   ├── rendering.py        # Pygame visualization
│   └── __init__.py
├── training/
│   ├── dqn_training.py     # DQN with 10 hyperparameter runs
│   └── pg_training.py      # REINFORCE, PPO, A2C (10 runs each)
├── models/
│   ├── dqn/               # Saved DQN models
│   └── pg/                # Saved policy gradient models
│       ├── ppo/
│       ├── a2c/
│       └── reinforce/
├── results/               # Training logs, CSVs
├── plots/                 # Generated report figures
├── random_agent_demo.py   # Visualization demo (no training)
├── main.py                # Best agent demo + REST API
├── plot_results.py        # Generate all report plots
├── requirements.txt
└── README.md
```

---

##  Quick Start

### 1. Installation

```bash
git clone https://github.com/YOUR_NAME/student_name_rl_summative.git
cd student_name_rl_summative

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

### 2. Random Agent Demo (No Training Required)

```bash
python random_agent_demo.py
python random_agent_demo.py --steps 60 --fps 4
```

### 3. Train All Models

```bash
# Train DQN (10 runs)
python training/dqn_training.py

# Train REINFORCE, PPO, A2C (10 runs each)
python training/pg_training.py

# Train single algorithm
python training/pg_training.py --algo PPO
python training/pg_training.py --algo A2C --run_id 3
```

### 4. Run Best Agent

```bash
# Auto-select best algorithm
python main.py

# Force specific algorithm
python main.py --algo PPO --episodes 3 --fps 5

# Start REST API
python main.py --api --port 8000
# → http://localhost:8000/docs
```

### 5. Generate Report Plots

```bash
python plot_results.py
# Saves all figures to: plots/
```

---

## 🌐 Environment Details

### Action Space — `Discrete(8)`

| ID | Strategy | Savings | Expenses | Investments | Debt | Emergency | Education |
|----|----------|---------|----------|-------------|------|-----------|-----------|
| 0  | Conservative Save | 60% | 30% | 10% | — | — | — |
| 1  | Balanced Allocate | 30% | 40% | 30% | — | — | — |
| 2  | Aggressive Invest | 10% | 30% | 60% | — | — | — |
| 3  | Debt Repayment | 10% | 30% | 20% | 40% | — | — |
| 4  | Emergency Fund | 20% | 30% | — | — | 50% | — |
| 5  | Education/Upskill | 40% | 40% | — | — | — | 20% |
| 6  | Mobile Money | 30% | 30% | — | — | — | — |
| 7  | Survival Mode | 20% | 70% | — | — | 10% | — |

### Observation Space — `Box(9,)`

| Index | Feature | Range | Description |
|-------|---------|-------|-------------|
| 0 | income_normalized | [0,1] | Monthly income / max_income |
| 1 | savings_ratio | [0,1] | Savings / annual income |
| 2 | expense_ratio | [0,1] | Expenses / income |
| 3 | debt_ratio | [0,1] | Debt / annual income |
| 4 | inflation_rate | [0,1] | Current monthly inflation |
| 5 | investment_value | [0,1] | Portfolio / $10,000 target |
| 6 | month_progress | [0,1] | Month in simulation |
| 7 | economic_shock | {0,1} | Active shock indicator |
| 8 | financial_stress | [0,1] | Composite stress index |

### Reward Structure

```
R = tanh(net_worth/5000)×5      — net worth growth
  - min(debt/500, 3)             — debt penalty
  + (1-stress)×2                 — stability reward
  + tanh(investment/target)×3    — investment reward
  + 1.0 if emergency≥3×expenses  — emergency fund bonus
  - 5.0 if can't cover expenses  — survival penalty
  + 1.0 if shock + savings safe  — shock resilience bonus
  + 2.0 if 6 stable months       — streak bonus
```

### Terminal Conditions

- **Bankruptcy**: `debt > 36 × monthly_income` → penalty −20
- **Financial Freedom**: `investment ≥ $10,000 AND debt < income` → bonus +30
- **Truncation**: 60 steps (5 years simulated)

---

## 📊 Algorithms

| Algorithm | Type | SB3 Class | Key Hyperparameters |
|-----------|------|-----------|---------------------|
| DQN | Value-Based | `DQN` | lr, gamma, buffer_size, epsilon |
| REINFORCE | Policy Gradient | `A2C` (vf_coef=0) | lr, gamma, n_steps, entropy |
| PPO | Policy Gradient | `PPO` | lr, clip_range, n_epochs, GAE-λ |
| A2C | Actor-Critic | `A2C` | lr, vf_coef, n_steps, GAE-λ |

---



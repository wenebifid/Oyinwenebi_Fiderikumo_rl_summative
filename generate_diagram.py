#!/usr/bin/env python3
"""
generate_diagram.py
Generates an environment/agent architecture diagram and saves it as PNG.
Run: python generate_diagram.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

PLOTS_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

fig, ax = plt.subplots(figsize=(14, 9))
fig.patch.set_facecolor("#0d1117")
ax.set_facecolor("#0d1117")
ax.set_xlim(0, 14)
ax.set_ylim(0, 9)
ax.axis("off")

def box(ax, x, y, w, h, label, sublabel="", color="#161b22", border="#4ea3e0", fontsize=11):
    rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                          facecolor=color, edgecolor=border, linewidth=2, zorder=3)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2 + (0.15 if sublabel else 0), label,
            ha="center", va="center", color="white", fontsize=fontsize,
            fontweight="bold", zorder=4)
    if sublabel:
        ax.text(x + w/2, y + h/2 - 0.25, sublabel,
                ha="center", va="center", color="#8b949e", fontsize=8, zorder=4)

def arrow(ax, x1, y1, x2, y2, label="", color="#4ea3e0"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=color, lw=2.0))
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx, my + 0.12, label, ha="center", va="center",
                color=color, fontsize=8, bbox=dict(facecolor="#0d1117", edgecolor="none", alpha=0.8))

# Title
ax.text(7, 8.5, "🌍 AI Micro-Investment Agent — Environment Architecture",
        ha="center", va="center", color="#ffd700", fontsize=14, fontweight="bold")

# Agent box
box(ax, 5.5, 5.5, 3, 1.5, "RL AGENT", "DQN / PPO / A2C / REINFORCE", color="#1c2f45", border="#4ea3e0")

# Environment box
box(ax, 5.5, 2.5, 3, 1.5, "AfricanFinanceEnv", "Gymnasium Custom Env", color="#1c3320", border="#56d364")

# Action space box
box(ax, 0.4, 5.2, 3.5, 2.2, "ACTION SPACE (8)", "", color="#1c1c2e", border="#bc8cff", fontsize=10)
actions = ["0: Conservative Save","1: Balanced Allocate","2: Aggressive Invest",
           "3: Debt Repayment","4: Emergency Fund","5: Education","6: Mobile Money","7: Survival Mode"]
for i, a in enumerate(actions):
    ax.text(0.6, 7.1 - i*0.24, a, color="#bc8cff", fontsize=7, va="center")

# Observation space box
box(ax, 10.2, 5.2, 3.5, 2.2, "OBSERVATION (9-dim)", "", color="#1c1c2e", border="#f0883e", fontsize=10)
obs_items = ["income_normalized","savings_ratio","expense_ratio","debt_ratio",
             "inflation_rate","investment_value","month_progress","economic_shock","financial_stress"]
for i, o in enumerate(obs_items):
    ax.text(10.35, 7.1 - i*0.22, o, color="#f0883e", fontsize=7, va="center")

# Reward box
box(ax, 5.5, 0.5, 3, 1.4, "REWARD FUNCTION", "Net Worth + Stability - Debt - Shock", color="#1c1818", border="#ff6b6b", fontsize=9)

# Pygame renderer box
box(ax, 0.4, 2.5, 3.0, 1.5, "Pygame Renderer", "Real-time visualization", color="#1a1a1a", border="#ffd700", fontsize=9)

# Terminal conditions box
box(ax, 10.2, 2.5, 3.5, 1.5, "TERMINAL CONDITIONS", "Bankrupt | Freedom | 60 steps", color="#1a1a1a", border="#ff6b6b", fontsize=9)

# Arrows
arrow(ax, 5.5, 6.25, 3.9, 6.25,  "action", "#bc8cff")       # agent ← action space
arrow(ax, 10.2, 6.25, 8.5, 6.25, "obs",    "#f0883e")        # obs → agent
arrow(ax, 7.0,  5.5,  7.0, 4.0,  "step(a)","#56d364")        # agent → env
arrow(ax, 6.5,  4.0,  6.5, 5.5,  "obs,r",  "#4ea3e0")        # env → agent
arrow(ax, 7.0,  2.5,  7.0, 1.9,  "reward", "#ff6b6b")        # env → reward
arrow(ax, 3.4,  3.25, 5.5, 3.25, "render", "#ffd700")        # env → pygame
arrow(ax, 8.5,  2.75, 10.2,2.75, "check",  "#ff6b6b")        # env → terminal

ax.text(7, 0.15, "Stable-Baselines3  |  Gymnasium  |  Pygame  |  Python 3.10+",
        ha="center", va="center", color="#8b949e", fontsize=9)

plt.tight_layout()
out = os.path.join(PLOTS_DIR, "00_architecture_diagram.png")
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"✅ Diagram saved: {out}")

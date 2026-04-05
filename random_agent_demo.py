"""
random_agent_demo.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Static demo: agent takes RANDOM actions (no training).
Purpose: visualize all environment components live.

Run:
    python random_agent_demo.py
    python random_agent_demo.py --steps 60 --fps 5
"""

import argparse
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pygame
from environment.custom_env import AfricanFinanceEnv


def run_random_demo(steps: int = 60, fps: int = 4, seed: int = 42):
    print("=" * 60)
    print("  🌍 African Finance RL — RANDOM AGENT DEMO")
    print("=" * 60)
    print(f"  Steps: {steps} | FPS: {fps} | Seed: {seed}")
    print("  Agent takes RANDOM actions (no training).")
    print("  Close the window or press Q to quit early.\n")

    env = AfricanFinanceEnv(render_mode="human")
    obs, info = env.reset(seed=seed)

    total_reward = 0.0
    terminated  = False
    truncated   = False
    step        = 0
    action      = None

    print(f"{'Step':>5} {'Action':>20} {'Income':>10} {'Savings':>10} "
          f"{'Debt':>10} {'Invest':>10} {'Stress':>8} {'Reward':>8}")
    print("-" * 90)

    clock = pygame.time.Clock()

    while not (terminated or truncated) and step < steps:
        # Handle Pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                return
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                env.close()
                return

        # Random action
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step += 1

        action_names = [
            "Conservative Save", "Balanced Allocate", "Aggressive Invest",
            "Debt Repayment", "Emergency Fund", "Education/Upskill",
            "Mobile Money", "Survival Mode",
        ]

        print(f"{step:>5} {action_names[action]:>20} "
              f"${info['income']:>8,.0f} ${info['savings']:>8,.0f} "
              f"${info['debt']:>8,.0f} ${info['investment_value']:>8,.0f} "
              f"{info['financial_stress']:>7.2%} {reward:>+8.2f}")

        env.render()
        clock.tick(fps)

    print("\n" + "=" * 60)
    print(f"  Episode complete after {step} steps")
    print(f"  Total reward:      {total_reward:+.2f}")
    print(f"  Final net worth:   ${info['net_worth']:,.0f}")
    print(f"  Terminated early:  {terminated}")

    if terminated:
        nw = info.get("net_worth", 0)
        if nw > 8000:
            print("  🏆 Agent achieved Financial Freedom!")
        else:
            print("  ❌ Agent went Bankrupt!")
    print("=" * 60)

    # Keep window open for a few seconds at end
    time.sleep(3)
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Random Agent Demo")
    parser.add_argument("--steps", type=int, default=60, help="Max steps")
    parser.add_argument("--fps",   type=int, default=4,  help="Render FPS")
    parser.add_argument("--seed",  type=int, default=42, help="Random seed")
    args = parser.parse_args()

    run_random_demo(steps=args.steps, fps=args.fps, seed=args.seed)

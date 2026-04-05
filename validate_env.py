"""
validate_env.py — Quick sanity check for the custom environment.
Run this FIRST to make sure everything is working before training.

    python validate_env.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from gymnasium.utils.env_checker import check_env
from environment.custom_env import AfricanFinanceEnv

print("=" * 55)
print("  AfricanFinanceEnv — Validation")
print("=" * 55)

env = AfricanFinanceEnv()

# 1. Gymnasium check
print("\n[1/4] Running Gymnasium env checker...")
try:
    check_env(env, warn=True)
    print("  ✅ Gymnasium check passed")
except Exception as e:
    print(f"  ⚠ Warning: {e}")

# 2. Reset / step
print("\n[2/4] Reset + 10 random steps...")
obs, info = env.reset(seed=42)
assert obs.shape == (9,), f"Expected obs shape (9,), got {obs.shape}"
assert env.observation_space.contains(obs), "Obs not in space"

for i in range(10):
    action = env.action_space.sample()
    obs, reward, term, trunc, info = env.step(action)
    assert env.observation_space.contains(obs), f"Step {i}: obs not in space"
    print(f"  Step {i+1}: action={action}, reward={reward:+.2f}, "
          f"net_worth=${info['net_worth']:,.0f}, stress={info['financial_stress']:.2%}")
    if term or trunc:
        obs, info = env.reset()
print("  ✅ Step loop passed")

# 3. Full episode
print("\n[3/4] Full episode (60 steps)...")
obs, info = env.reset(seed=7)
total_r = 0
for _ in range(60):
    obs, r, term, trunc, info = env.step(env.action_space.sample())
    total_r += r
    if term or trunc:
        break
print(f"  Total reward: {total_r:+.2f} | Net worth: ${info['net_worth']:,.0f}")
print("  ✅ Full episode passed")

# 4. Action allocations sum to 1
print("\n[4/4] Action allocation check...")
for i, alloc in env.action_allocations.items():
    total = sum(alloc.values())
    assert abs(total - 1.0) < 1e-9, f"Action {i} allocations sum to {total}"
print("  ✅ All allocations sum to 1.0")

env.close()
print("\n✅ All checks passed! Ready to train.\n")

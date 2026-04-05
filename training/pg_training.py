"""
pg_training.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Policy Gradient training for AfricanFinanceEnv using Stable-Baselines3.
Covers: REINFORCE (via custom callback on A2C), PPO, A2C — 10 runs each.

Run:
    python training/pg_training.py                  # all algorithms
    python training/pg_training.py --algo PPO       # single algorithm
    python training/pg_training.py --algo A2C --run_id 3
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO, A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from environment.custom_env import AfricanFinanceEnv

MODELS_DIR  = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "pg")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

TOTAL_TIMESTEPS = 150_000

# ─────────────────────────────────────────────────────────────────────────────
# REINFORCE configs (implemented via A2C with no value function / n_steps=full)
# ─────────────────────────────────────────────────────────────────────────────
REINFORCE_CONFIGS = [
    dict(learning_rate=5e-4,  gamma=0.99,  n_steps=60,  ent_coef=0.01,  max_grad_norm=0.5),
    dict(learning_rate=1e-3,  gamma=0.99,  n_steps=60,  ent_coef=0.01,  max_grad_norm=0.5),
    dict(learning_rate=3e-3,  gamma=0.99,  n_steps=60,  ent_coef=0.01,  max_grad_norm=0.5),
    dict(learning_rate=1e-3,  gamma=0.95,  n_steps=60,  ent_coef=0.01,  max_grad_norm=0.5),
    dict(learning_rate=1e-3,  gamma=0.995, n_steps=60,  ent_coef=0.01,  max_grad_norm=0.5),
    dict(learning_rate=1e-3,  gamma=0.99,  n_steps=60,  ent_coef=0.05,  max_grad_norm=0.5),
    dict(learning_rate=1e-3,  gamma=0.99,  n_steps=60,  ent_coef=0.001, max_grad_norm=0.5),
    dict(learning_rate=1e-3,  gamma=0.99,  n_steps=120, ent_coef=0.01,  max_grad_norm=0.5),
    dict(learning_rate=1e-3,  gamma=0.99,  n_steps=60,  ent_coef=0.01,  max_grad_norm=1.0),
    dict(learning_rate=5e-4,  gamma=0.99,  n_steps=60,  ent_coef=0.05,  max_grad_norm=0.5),
]
REINFORCE_DESC = [
    "Baseline LR=5e-4",
    "Baseline LR=1e-3",
    "Higher LR=3e-3",
    "Lower gamma=0.95",
    "Higher gamma=0.995",
    "Higher entropy=0.05",
    "Lower entropy=0.001",
    "Longer rollout n=120",
    "Higher grad_norm=1.0",
    "LR=5e-4 + entropy=0.05",
]

# ─────────────────────────────────────────────────────────────────────────────
# PPO configs
# ─────────────────────────────────────────────────────────────────────────────
PPO_CONFIGS = [
    dict(learning_rate=3e-4,  gamma=0.99,  n_steps=512,  batch_size=64,  n_epochs=10, ent_coef=0.01, clip_range=0.2, gae_lambda=0.95),
    dict(learning_rate=1e-3,  gamma=0.99,  n_steps=512,  batch_size=64,  n_epochs=10, ent_coef=0.01, clip_range=0.2, gae_lambda=0.95),
    dict(learning_rate=5e-4,  gamma=0.99,  n_steps=1024, batch_size=128, n_epochs=10, ent_coef=0.01, clip_range=0.2, gae_lambda=0.95),
    dict(learning_rate=3e-4,  gamma=0.95,  n_steps=512,  batch_size=64,  n_epochs=10, ent_coef=0.01, clip_range=0.2, gae_lambda=0.95),
    dict(learning_rate=3e-4,  gamma=0.99,  n_steps=512,  batch_size=64,  n_epochs=10, ent_coef=0.05, clip_range=0.2, gae_lambda=0.95),
    dict(learning_rate=3e-4,  gamma=0.99,  n_steps=512,  batch_size=64,  n_epochs=20, ent_coef=0.01, clip_range=0.2, gae_lambda=0.95),
    dict(learning_rate=3e-4,  gamma=0.99,  n_steps=512,  batch_size=64,  n_epochs=10, ent_coef=0.01, clip_range=0.1, gae_lambda=0.95),
    dict(learning_rate=3e-4,  gamma=0.99,  n_steps=512,  batch_size=64,  n_epochs=10, ent_coef=0.01, clip_range=0.3, gae_lambda=0.95),
    dict(learning_rate=3e-4,  gamma=0.99,  n_steps=512,  batch_size=64,  n_epochs=10, ent_coef=0.01, clip_range=0.2, gae_lambda=0.80),
    dict(learning_rate=1e-4,  gamma=0.99,  n_steps=2048, batch_size=256, n_epochs=10, ent_coef=0.001,clip_range=0.2, gae_lambda=0.98),
]
PPO_DESC = [
    "Baseline LR=3e-4",
    "Higher LR=1e-3",
    "Longer rollout n=1024",
    "Lower gamma=0.95",
    "High entropy=0.05",
    "More epochs=20",
    "Tight clip=0.1",
    "Wide clip=0.3",
    "Low GAE lambda=0.80",
    "Low LR, very long rollout",
]

# ─────────────────────────────────────────────────────────────────────────────
# A2C configs
# ─────────────────────────────────────────────────────────────────────────────
A2C_CONFIGS = [
    dict(learning_rate=7e-4,  gamma=0.99,  n_steps=5,   ent_coef=0.01, vf_coef=0.5,  max_grad_norm=0.5, gae_lambda=1.0),
    dict(learning_rate=3e-4,  gamma=0.99,  n_steps=5,   ent_coef=0.01, vf_coef=0.5,  max_grad_norm=0.5, gae_lambda=1.0),
    dict(learning_rate=1e-3,  gamma=0.99,  n_steps=5,   ent_coef=0.01, vf_coef=0.5,  max_grad_norm=0.5, gae_lambda=1.0),
    dict(learning_rate=7e-4,  gamma=0.95,  n_steps=5,   ent_coef=0.01, vf_coef=0.5,  max_grad_norm=0.5, gae_lambda=1.0),
    dict(learning_rate=7e-4,  gamma=0.99,  n_steps=20,  ent_coef=0.01, vf_coef=0.5,  max_grad_norm=0.5, gae_lambda=1.0),
    dict(learning_rate=7e-4,  gamma=0.99,  n_steps=5,   ent_coef=0.05, vf_coef=0.5,  max_grad_norm=0.5, gae_lambda=1.0),
    dict(learning_rate=7e-4,  gamma=0.99,  n_steps=5,   ent_coef=0.01, vf_coef=1.0,  max_grad_norm=0.5, gae_lambda=1.0),
    dict(learning_rate=7e-4,  gamma=0.99,  n_steps=5,   ent_coef=0.01, vf_coef=0.25, max_grad_norm=0.5, gae_lambda=1.0),
    dict(learning_rate=7e-4,  gamma=0.99,  n_steps=5,   ent_coef=0.01, vf_coef=0.5,  max_grad_norm=1.0, gae_lambda=0.95),
    dict(learning_rate=5e-4,  gamma=0.995, n_steps=10,  ent_coef=0.02, vf_coef=0.5,  max_grad_norm=0.5, gae_lambda=1.0),
]
A2C_DESC = [
    "Baseline LR=7e-4",
    "Lower LR=3e-4",
    "Higher LR=1e-3",
    "Lower gamma=0.95",
    "Longer steps=20",
    "High entropy=0.05",
    "High vf_coef=1.0",
    "Low vf_coef=0.25",
    "High grad norm, GAE=0.95",
    "Long gamma, medium steps",
]


def make_env(seed=0):
    def _init():
        env = AfricanFinanceEnv()
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _init


def train_ppo(run_id, cfg, desc, verbose=0):
    print(f"\n[PPO Run {run_id}] {desc}")
    env      = DummyVecEnv([make_env(seed=run_id)])
    eval_env = DummyVecEnv([make_env(seed=200 + run_id)])
    model_dir = os.path.join(MODELS_DIR, "ppo", f"run{run_id}")
    log_dir   = os.path.join(RESULTS_DIR, "ppo_logs", f"run{run_id}")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    model = PPO(
        "MlpPolicy", env,
        learning_rate=cfg["learning_rate"],
        gamma=cfg["gamma"],
        n_steps=cfg["n_steps"],
        batch_size=cfg["batch_size"],
        n_epochs=cfg["n_epochs"],
        ent_coef=cfg["ent_coef"],
        clip_range=cfg["clip_range"],
        gae_lambda=cfg["gae_lambda"],
        tensorboard_log=log_dir,
        verbose=verbose,
        seed=run_id,
    )

    cb = EvalCallback(eval_env, best_model_save_path=model_dir,
                      log_path=log_dir, eval_freq=10_000,
                      n_eval_episodes=5, deterministic=True, verbose=0)

    t0 = time.time()
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=cb, progress_bar=True)
    elapsed = time.time() - t0

    mean_r, std_r = evaluate_policy(model, eval_env, n_eval_episodes=20, deterministic=True)
    model.save(os.path.join(model_dir, "final_model"))
    print(f"  ✅ PPO Run {run_id}: Mean={mean_r:.2f} ± {std_r:.2f}  [{elapsed:.0f}s]")
    env.close(); eval_env.close()
    return {**cfg, "run_id": run_id, "description": desc, "algorithm": "PPO",
            "mean_reward": round(mean_r, 4), "std_reward": round(std_r, 4),
            "training_time_s": round(elapsed, 1)}


def train_a2c(run_id, cfg, desc, verbose=0):
    print(f"\n[A2C Run {run_id}] {desc}")
    env      = DummyVecEnv([make_env(seed=run_id)])
    eval_env = DummyVecEnv([make_env(seed=300 + run_id)])
    model_dir = os.path.join(MODELS_DIR, "a2c", f"run{run_id}")
    log_dir   = os.path.join(RESULTS_DIR, "a2c_logs", f"run{run_id}")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    model = A2C(
        "MlpPolicy", env,
        learning_rate=cfg["learning_rate"],
        gamma=cfg["gamma"],
        n_steps=cfg["n_steps"],
        ent_coef=cfg["ent_coef"],
        vf_coef=cfg["vf_coef"],
        max_grad_norm=cfg["max_grad_norm"],
        gae_lambda=cfg["gae_lambda"],
        tensorboard_log=log_dir,
        verbose=verbose,
        seed=run_id,
    )

    cb = EvalCallback(eval_env, best_model_save_path=model_dir,
                      log_path=log_dir, eval_freq=10_000,
                      n_eval_episodes=5, deterministic=True, verbose=0)

    t0 = time.time()
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=cb, progress_bar=True)
    elapsed = time.time() - t0

    mean_r, std_r = evaluate_policy(model, eval_env, n_eval_episodes=20, deterministic=True)
    model.save(os.path.join(model_dir, "final_model"))
    print(f"  ✅ A2C Run {run_id}: Mean={mean_r:.2f} ± {std_r:.2f}  [{elapsed:.0f}s]")
    env.close(); eval_env.close()
    return {**cfg, "run_id": run_id, "description": desc, "algorithm": "A2C",
            "mean_reward": round(mean_r, 4), "std_reward": round(std_r, 4),
            "training_time_s": round(elapsed, 1)}


def train_reinforce(run_id, cfg, desc, verbose=0):
    """
    REINFORCE via A2C with vf_coef=0 (no value function)
    and full-episode rollouts (n_steps = episode length).
    """
    print(f"\n[REINFORCE Run {run_id}] {desc}")
    env      = DummyVecEnv([make_env(seed=run_id)])
    eval_env = DummyVecEnv([make_env(seed=400 + run_id)])
    model_dir = os.path.join(MODELS_DIR, "reinforce", f"run{run_id}")
    log_dir   = os.path.join(RESULTS_DIR, "reinforce_logs", f"run{run_id}")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    model = A2C(
        "MlpPolicy", env,
        learning_rate=cfg["learning_rate"],
        gamma=cfg["gamma"],
        n_steps=cfg["n_steps"],
        ent_coef=cfg["ent_coef"],
        vf_coef=0.0,              # Pure REINFORCE: no baseline
        max_grad_norm=cfg["max_grad_norm"],
        gae_lambda=1.0,           # Full Monte-Carlo returns
        tensorboard_log=log_dir,
        verbose=verbose,
        seed=run_id,
    )

    cb = EvalCallback(eval_env, best_model_save_path=model_dir,
                      log_path=log_dir, eval_freq=10_000,
                      n_eval_episodes=5, deterministic=True, verbose=0)

    t0 = time.time()
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=cb, progress_bar=True)
    elapsed = time.time() - t0

    mean_r, std_r = evaluate_policy(model, eval_env, n_eval_episodes=20, deterministic=True)
    model.save(os.path.join(model_dir, "final_model"))
    print(f"  ✅ REINFORCE Run {run_id}: Mean={mean_r:.2f} ± {std_r:.2f}  [{elapsed:.0f}s]")
    env.close(); eval_env.close()
    return {**cfg, "run_id": run_id, "description": desc, "algorithm": "REINFORCE",
            "vf_coef": 0.0, "mean_reward": round(mean_r, 4),
            "std_reward": round(std_r, 4), "training_time_s": round(elapsed, 1)}


def run_algorithm(algo: str, run_id: int = -1, verbose: int = 0):
    algo = algo.upper()
    if algo == "PPO":
        configs, descs, train_fn = PPO_CONFIGS, PPO_DESC, train_ppo
    elif algo == "A2C":
        configs, descs, train_fn = A2C_CONFIGS, A2C_DESC, train_a2c
    elif algo in ("REINFORCE", "PG"):
        configs, descs, train_fn = REINFORCE_CONFIGS, REINFORCE_DESC, train_reinforce
    else:
        raise ValueError(f"Unknown algorithm: {algo}")

    runs = [(run_id, configs[run_id], descs[run_id])] if run_id >= 0 \
           else list(zip(range(len(configs)), configs, descs))

    results = []
    for rid, cfg, desc in runs:
        try:
            results.append(train_fn(rid, cfg, desc, verbose=verbose))
        except Exception as e:
            print(f"  ❌ {algo} Run {rid} failed: {e}")
            import traceback; traceback.print_exc()

    if results:
        df = pd.DataFrame(results)
        csv_path = os.path.join(RESULTS_DIR, f"{algo.lower()}_results.csv")
        df.to_csv(csv_path, index=False)
        print(f"\n📊 {algo} results → {csv_path}")
        print(df[["run_id", "description", "mean_reward", "std_reward"]].to_string())

        best = df.loc[df["mean_reward"].idxmax()]
        print(f"\n🏆 Best {algo} Run: {int(best['run_id'])} — {best['description']} "
              f"| Reward: {best['mean_reward']:.4f}")

        meta = {"best_run_id": int(best["run_id"]), "best_mean_reward": float(best["mean_reward"])}
        with open(os.path.join(MODELS_DIR, algo.lower(), "best_run.json"), "w") as f:
            json.dump(meta, f, indent=2)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo",    type=str, default="ALL",
                        choices=["ALL", "PPO", "A2C", "REINFORCE"],
                        help="Algorithm to train")
    parser.add_argument("--run_id",  type=int, default=-1, help="-1 = all runs")
    parser.add_argument("--verbose", type=int, default=0)
    args = parser.parse_args()

    algos = ["REINFORCE", "PPO", "A2C"] if args.algo == "ALL" else [args.algo]
    for algo in algos:
        print(f"\n{'#'*60}")
        print(f"# Training {algo}")
        print(f"{'#'*60}")
        run_algorithm(algo, run_id=args.run_id, verbose=args.verbose)

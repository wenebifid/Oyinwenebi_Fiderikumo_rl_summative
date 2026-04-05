"""
dqn_training.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DQN training for AfricanFinanceEnv using Stable-Baselines3.
Runs 10 hyperparameter experiments and saves results.

Run:
    python training/dqn_training.py
    python training/dqn_training.py --run_id 0   # single run
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from environment.custom_env import AfricanFinanceEnv

MODELS_DIR  = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "dqn")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

TOTAL_TIMESTEPS = 150_000

# ── 10 hyperparameter configurations ─────────────────────────────────────────
DQN_CONFIGS = [
    # Run 0: Baseline
    dict(learning_rate=1e-3,  gamma=0.99, buffer_size=50_000,
         batch_size=64,   learning_starts=1000, exploration_fraction=0.2,
         exploration_final_eps=0.05, train_freq=4, target_update_interval=500,
         policy_kwargs=None, tau=1.0),
    # Run 1: Lower LR, more exploration
    dict(learning_rate=5e-4,  gamma=0.99, buffer_size=50_000,
         batch_size=64,   learning_starts=2000, exploration_fraction=0.3,
         exploration_final_eps=0.05, train_freq=4, target_update_interval=500,
         policy_kwargs=None, tau=1.0),
    # Run 2: Higher LR, bigger buffer
    dict(learning_rate=3e-3,  gamma=0.99, buffer_size=100_000,
         batch_size=128,  learning_starts=1000, exploration_fraction=0.15,
         exploration_final_eps=0.02, train_freq=4, target_update_interval=1000,
         policy_kwargs=None, tau=1.0),
    # Run 3: Lower gamma (myopic)
    dict(learning_rate=1e-3,  gamma=0.90, buffer_size=50_000,
         batch_size=64,   learning_starts=1000, exploration_fraction=0.2,
         exploration_final_eps=0.05, train_freq=4, target_update_interval=500,
         policy_kwargs=None, tau=1.0),
    # Run 4: High gamma (long-term)
    dict(learning_rate=1e-3,  gamma=0.995, buffer_size=50_000,
         batch_size=64,   learning_starts=1000, exploration_fraction=0.25,
         exploration_final_eps=0.05, train_freq=4, target_update_interval=500,
         policy_kwargs=None, tau=1.0),
    # Run 5: Bigger network
    dict(learning_rate=1e-3,  gamma=0.99, buffer_size=50_000,
         batch_size=128,  learning_starts=1000, exploration_fraction=0.2,
         exploration_final_eps=0.05, train_freq=4, target_update_interval=500,
         policy_kwargs=dict(net_arch=[256, 256, 128]), tau=1.0),
    # Run 6: Small batch, frequent updates
    dict(learning_rate=1e-3,  gamma=0.99, buffer_size=30_000,
         batch_size=32,   learning_starts=500, exploration_fraction=0.2,
         exploration_final_eps=0.05, train_freq=1, target_update_interval=200,
         policy_kwargs=None, tau=1.0),
    # Run 7: Large batch, slow updates
    dict(learning_rate=5e-4,  gamma=0.99, buffer_size=100_000,
         batch_size=256,  learning_starts=5000, exploration_fraction=0.3,
         exploration_final_eps=0.05, train_freq=8, target_update_interval=2000,
         policy_kwargs=None, tau=1.0),
    # Run 8: Polyak (soft) update
    dict(learning_rate=1e-3,  gamma=0.99, buffer_size=50_000,
         batch_size=64,   learning_starts=1000, exploration_fraction=0.2,
         exploration_final_eps=0.05, train_freq=4, target_update_interval=1,
         policy_kwargs=None, tau=0.005),
    # Run 9: Very aggressive exploration
    dict(learning_rate=1e-3,  gamma=0.99, buffer_size=50_000,
         batch_size=64,   learning_starts=1000, exploration_fraction=0.5,
         exploration_final_eps=0.10, train_freq=4, target_update_interval=500,
         policy_kwargs=None, tau=1.0),
]

DESCRIPTIONS = [
    "Baseline configuration",
    "Lower LR, more exploration",
    "Higher LR, large buffer, big batch",
    "Lower gamma (myopic agent)",
    "High gamma (long-term planning)",
    "Deeper network [256,256,128]",
    "Small batch, frequent updates",
    "Large batch, slow target updates",
    "Polyak (soft) target update",
    "Aggressive exploration (50%)",
]


def make_env(seed=0):
    def _init():
        env = AfricanFinanceEnv()
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _init


def train_run(run_id: int, cfg: dict, desc: str, verbose: int = 0):
    print(f"\n{'='*60}")
    print(f"  DQN Run {run_id}: {desc}")
    print(f"{'='*60}")
    for k, v in cfg.items():
        print(f"  {k:35s}: {v}")

    env  = DummyVecEnv([make_env(seed=run_id)])
    eval_env = DummyVecEnv([make_env(seed=100 + run_id)])

    model_path = os.path.join(MODELS_DIR, f"dqn_run{run_id}")
    log_path   = os.path.join(RESULTS_DIR, "dqn_logs", f"run{run_id}")
    os.makedirs(log_path, exist_ok=True)

    model = DQN(
        policy              = "MlpPolicy",
        env                 = env,
        learning_rate       = cfg["learning_rate"],
        gamma               = cfg["gamma"],
        buffer_size         = cfg["buffer_size"],
        batch_size          = cfg["batch_size"],
        learning_starts     = cfg["learning_starts"],
        exploration_fraction= cfg["exploration_fraction"],
        exploration_final_eps=cfg["exploration_final_eps"],
        train_freq          = cfg["train_freq"],
        target_update_interval=cfg["target_update_interval"],
        policy_kwargs       = cfg["policy_kwargs"],
        tau                 = cfg["tau"],
        tensorboard_log     = log_path,
        verbose             = verbose,
        seed                = run_id,
    )

    callbacks = [
        EvalCallback(eval_env, best_model_save_path=model_path,
                     log_path=log_path, eval_freq=10_000,
                     n_eval_episodes=5, deterministic=True, verbose=0),
    ]

    t0 = time.time()
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callbacks, progress_bar=True)
    elapsed = time.time() - t0

    # Evaluate final model
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=20, deterministic=True)
    print(f"\n  ✅ Run {run_id} complete — Mean reward: {mean_reward:.2f} ± {std_reward:.2f}  [{elapsed:.0f}s]")

    model.save(os.path.join(model_path, "final_model"))
    env.close()
    eval_env.close()

    return {
        "run_id":              run_id,
        "description":         desc,
        "learning_rate":       cfg["learning_rate"],
        "gamma":               cfg["gamma"],
        "buffer_size":         cfg["buffer_size"],
        "batch_size":          cfg["batch_size"],
        "learning_starts":     cfg["learning_starts"],
        "exploration_fraction":cfg["exploration_fraction"],
        "exploration_final_eps":cfg["exploration_final_eps"],
        "train_freq":          cfg["train_freq"],
        "target_update_interval":cfg["target_update_interval"],
        "tau":                 cfg["tau"],
        "net_arch":            str(cfg["policy_kwargs"]),
        "mean_reward":         round(mean_reward, 4),
        "std_reward":          round(std_reward, 4),
        "training_time_s":     round(elapsed, 1),
    }


def run_all(run_id: int = -1, verbose: int = 0):
    results = []

    if run_id >= 0:
        configs = [(run_id, DQN_CONFIGS[run_id], DESCRIPTIONS[run_id])]
    else:
        configs = list(zip(range(len(DQN_CONFIGS)), DQN_CONFIGS, DESCRIPTIONS))

    for rid, cfg, desc in configs:
        try:
            result = train_run(rid, cfg, desc, verbose=verbose)
            results.append(result)
        except Exception as e:
            print(f"  ❌ Run {rid} failed: {e}")

    if results:
        df = pd.DataFrame(results)
        csv_path = os.path.join(RESULTS_DIR, "dqn_results.csv")
        df.to_csv(csv_path, index=False)
        print(f"\n📊 Results saved to: {csv_path}")
        print(df[["run_id", "description", "mean_reward", "std_reward", "training_time_s"]].to_string())

        best = df.loc[df["mean_reward"].idxmax()]
        print(f"\n🏆 Best DQN Run: Run {int(best['run_id'])} — {best['description']}")
        print(f"   Mean Reward: {best['mean_reward']:.4f}")

        # Save best run id
        meta = {"best_run_id": int(best["run_id"]), "best_mean_reward": float(best["mean_reward"])}
        with open(os.path.join(MODELS_DIR, "best_run.json"), "w") as f:
            json.dump(meta, f, indent=2)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id",  type=int, default=-1, help="-1 = all runs")
    parser.add_argument("--verbose", type=int, default=0)
    args = parser.parse_args()

    run_all(run_id=args.run_id, verbose=args.verbose)

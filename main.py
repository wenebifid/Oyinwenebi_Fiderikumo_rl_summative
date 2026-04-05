"""
main.py — Entry point for best-performing agent demo
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Loads the best saved model (DQN or PG) and runs a live Pygame simulation.

Usage:
    python main.py                         # auto-select best model
    python main.py --algo DQN              # force DQN
    python main.py --algo PPO              # force PPO
    python main.py --algo A2C
    python main.py --algo REINFORCE
    python main.py --algo DQN --fps 6
    python main.py --episodes 3
    python main.py --api                   # serve as JSON API (FastAPI)
"""

import os
import sys
import json
import argparse
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pygame

from environment.custom_env import AfricanFinanceEnv

MODELS_DIR  = os.path.join(os.path.dirname(__file__), "models")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

ALGO_MAP = {
    "DQN":       ("stable_baselines3", "DQN",  os.path.join(MODELS_DIR, "dqn")),
    "PPO":       ("stable_baselines3", "PPO",  os.path.join(MODELS_DIR, "pg", "ppo")),
    "A2C":       ("stable_baselines3", "A2C",  os.path.join(MODELS_DIR, "pg", "a2c")),
    "REINFORCE": ("stable_baselines3", "A2C",  os.path.join(MODELS_DIR, "pg", "reinforce")),
}


def find_best_model(algo: str):
    """Find the best saved model for a given algorithm."""
    _, _, model_dir = ALGO_MAP[algo]

    # Try to read best_run.json
    meta_path = os.path.join(model_dir, "best_run.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        run_id = meta["best_run_id"]
        model_path = os.path.join(model_dir, f"run{run_id}", "best_model")
        if os.path.exists(model_path + ".zip"):
            return model_path, meta
        # fallback to final_model
        model_path = os.path.join(model_dir, f"run{run_id}", "final_model")
        if os.path.exists(model_path + ".zip"):
            return model_path, meta

    # Fallback: scan for any .zip
    for root, dirs, files in os.walk(model_dir):
        for f in files:
            if f.endswith(".zip"):
                return os.path.join(root, f[:-4]), {}

    return None, {}


def load_model(algo: str):
    """Load the best model for the given algorithm."""
    module_name, class_name, _ = ALGO_MAP[algo]
    import importlib
    module = importlib.import_module(module_name)
    ModelClass = getattr(module, class_name)

    model_path, meta = find_best_model(algo)
    if model_path is None:
        raise FileNotFoundError(
            f"No saved model found for {algo}. Please train first:\n"
            f"  python training/dqn_training.py   # for DQN\n"
            f"  python training/pg_training.py    # for PPO/A2C/REINFORCE"
        )

    print(f"✅ Loading {algo} model: {model_path}")
    if meta:
        print(f"   Best run metadata: {meta}")

    model = ModelClass.load(model_path)
    return model


def auto_select_best_algo():
    """Select the algorithm with the highest recorded mean reward."""
    best_reward = -np.inf
    best_algo   = "DQN"

    for algo in ALGO_MAP:
        csv_name = "dqn_results.csv" if algo == "DQN" else f"{algo.lower()}_results.csv"
        csv_path = os.path.join(RESULTS_DIR, csv_name)
        if os.path.exists(csv_path):
            try:
                import pandas as pd
                df = pd.read_csv(csv_path)
                mx = df["mean_reward"].max()
                if mx > best_reward:
                    best_reward = mx
                    best_algo   = algo
            except Exception:
                pass

    print(f"🏆 Auto-selected best algorithm: {best_algo} (mean reward: {best_reward:.2f})")
    return best_algo


def run_simulation(algo: str, episodes: int = 1, fps: int = 5, seed: int = 0):
    """Run live Pygame simulation with the best model."""
    pygame.init()   

    model = load_model(algo)

    print(f"\n{'='*60}")
    print(f"  🌍 African Finance RL — LIVE AGENT DEMO")
    print(f"  Algorithm: {algo} | Episodes: {episodes} | FPS: {fps}")
    print(f"{'='*60}\n")

    env = AfricanFinanceEnv(render_mode="human")
    clock = pygame.time.Clock()

    action_names = [
        "Conservative Save", "Balanced Allocate", "Aggressive Invest",
        "Debt Repayment", "Emergency Fund", "Education/Upskill",
        "Mobile Money", "Survival Mode",
    ]

    all_episode_rewards = []

    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        ep_reward  = 0.0
        terminated = False
        truncated  = False
        step       = 0

        print(f"\n─── Episode {ep + 1} ───")
        print(f"{'Step':>5} {'Action':>20} {'Reward':>8} {'Net Worth':>12} {'Stress':>8}")
        print("-" * 60)

        while not (terminated or truncated):
            # Handle quit event
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    return
                if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                    env.close()
                    return

            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            ep_reward += reward
            step += 1

            print(f"{step:>5} {action_names[int(action)]:>20} {reward:>+8.2f} "
                  f"${info['net_worth']:>10,.0f} {info['financial_stress']:>7.2%}")

            # Render with action info
            env.render()
            clock.tick(fps)

        all_episode_rewards.append(ep_reward)
        outcome = "🏆 Financial Freedom!" if info.get("investment_value", 0) >= 10000 \
                  else "❌ Bankrupt" if terminated else "⏱ Time limit"

        print(f"\n  Episode {ep+1} complete — Total reward: {ep_reward:+.2f} | {outcome}")
        print(f"  Final net worth: ${info.get('net_worth', 0):,.0f}")
        print(f"  Investment:      ${info.get('investment_value', 0):,.0f}")
        print(f"  Savings:         ${info.get('savings', 0):,.0f}")
        print(f"  Debt:            ${info.get('debt', 0):,.0f}")

        time.sleep(2)

    print(f"\n{'='*60}")
    print(f"  Simulation complete!")
    print(f"  Mean episode reward: {np.mean(all_episode_rewards):.2f}")
    print(f"  Best episode reward: {np.max(all_episode_rewards):.2f}")
    print(f"{'='*60}")

    time.sleep(3)
    env.close()


# ── Optional: FastAPI JSON endpoint ──────────────────────────────────────────
def start_api(algo: str, host: str = "0.0.0.0", port: int = 8000):
    """
    Serve agent predictions as a REST API.
    POST /predict with JSON observation → get action + explanation.

    Install: pip install fastapi uvicorn
    """
    try:
        from fastapi import FastAPI
        from fastapi.middleware.cors import CORSMiddleware
        from pydantic import BaseModel
        import uvicorn
    except ImportError:
        print("FastAPI not installed. Run: pip install fastapi uvicorn")
        return

    model = load_model(algo)
    action_names = [
        "Conservative Save", "Balanced Allocate", "Aggressive Invest",
        "Debt Repayment", "Emergency Fund", "Education/Upskill",
        "Mobile Money", "Survival Mode",
    ]

    app = FastAPI(title="African Finance RL Agent API", version="1.0")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    class ObsRequest(BaseModel):
        income_normalized:  float
        savings_ratio:      float
        expense_ratio:      float
        debt_ratio:         float
        inflation_rate:     float
        investment_value:   float
        month_progress:     float
        economic_shock:     float
        financial_stress:   float

    @app.get("/")
    def root():
        return {"message": "African Finance RL Agent", "algorithm": algo, "status": "running"}

    @app.post("/predict")
    def predict(obs_req: ObsRequest):
        obs = np.array([
            obs_req.income_normalized, obs_req.savings_ratio, obs_req.expense_ratio,
            obs_req.debt_ratio, obs_req.inflation_rate, obs_req.investment_value,
            obs_req.month_progress, obs_req.economic_shock, obs_req.financial_stress,
        ], dtype=np.float32)
        action, _ = model.predict(obs, deterministic=True)
        action_id = int(action)
        return {
            "action_id":     action_id,
            "action_name":   action_names[action_id],
            "algorithm":     algo,
            "explanation":   f"Agent recommends: {action_names[action_id]} "
                             f"based on current financial state.",
        }

    print(f"🚀 Starting API server at http://{host}:{port}")
    print(f"   Swagger UI: http://{host}:{port}/docs")
    uvicorn.run(app, host=host, port=port)


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="African Finance RL Agent")
    parser.add_argument("--algo",     type=str, default="auto",
                        choices=["auto", "DQN", "PPO", "A2C", "REINFORCE"])
    parser.add_argument("--episodes", type=int,   default=2)
    parser.add_argument("--fps",      type=int,   default=5)
    parser.add_argument("--seed",     type=int,   default=0)
    parser.add_argument("--api",      action="store_true", help="Start REST API server")
    parser.add_argument("--port",     type=int,   default=8000)
    args = parser.parse_args()

    algo = auto_select_best_algo() if args.algo == "auto" else args.algo

    if args.api:
        start_api(algo, port=args.port)
    else:
        run_simulation(algo, episodes=args.episodes, fps=args.fps, seed=args.seed)

"""Zero-shot baseline evaluation for the pre-trained Nova Carter policy in the
Innovation Center building environment.

Runs the stage4 checkpoint (no gradient updates) for 20 episodes in the
building env and records goals/episode + collision rate. This establishes
the baseline before any fine-tuning and tells us how much fine-tuning is
actually needed.

  goals_per_episode > 2.0  →  policy generalizes well; minimal fine-tuning
  goals_per_episode 0.5–2.0 →  moderate fine-tuning needed (200-500k steps)
  goals_per_episode < 0.5  →  significant gap; check VecNormalize + waypoints

Usage:
    cd ~/IsaacLab
    ./isaaclab.sh -p ~/isaac-sim-msoe/training/eval/eval_zero_shot_building.py \\
        --checkpoint training/logs/sb3/NovaCarter-Patrol/2026-04-08_04-35-47/best_model_stage4.zip \\
        --headless
"""

import argparse
import json
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Zero-shot building baseline for Nova Carter.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to best_model_stage4.zip")
parser.add_argument("--vecnorm", type=str, default=None,
                    help="Path to model_vecnormalize.pkl. Inferred from checkpoint dir if omitted.")
parser.add_argument("--num_episodes", type=int, default=20, help="Episodes to evaluate.")
parser.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
args_cli.num_envs = 1

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -- Safe to import after SimulationApp is up --
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

import gymnasium as gym
from isaaclab_rl.sb3 import Sb3VecEnvWrapper

_eval_dir = os.path.dirname(os.path.abspath(__file__))
_training_dir = os.path.dirname(_eval_dir)
sys.path.insert(0, _training_dir)

import nova_carter_patrol  # noqa: F401


def main():
    checkpoint = args_cli.checkpoint
    vecnorm_path = args_cli.vecnorm
    if vecnorm_path is None:
        ckpt_dir = os.path.dirname(checkpoint)
        vecnorm_path = os.path.join(ckpt_dir, "model_vecnormalize.pkl")

    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if not os.path.exists(vecnorm_path):
        raise FileNotFoundError(
            f"VecNormalize stats not found: {vecnorm_path}\n"
            "Pass --vecnorm explicitly or place model_vecnormalize.pkl next to the checkpoint."
        )

    # Import here (after SimulationApp)
    from nova_carter_building_env_cfg import NovaCarterBuildingEnvCfg  # created in Phase 1

    env_cfg = NovaCarterBuildingEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = "cuda:0"

    raw_env = gym.make("Isaac-NovaCarter-Building-v0", cfg=env_cfg)
    env = Sb3VecEnvWrapper(raw_env)
    env = VecNormalize.load(vecnorm_path, venv=env)
    env.training = False   # freeze running stats during eval
    env.norm_reward = False

    # Debug: confirm VecNormalize loaded correctly
    print(f"[DEBUG] obs_rms.mean[:5] = {env.obs_rms.mean[:5]}")
    if np.allclose(env.obs_rms.mean[:5], 0.0):
        print("[WARNING] VecNormalize means are all zero — stats may not have loaded correctly.")

    agent = PPO.load(checkpoint, env=env, device="cuda:0")

    results = []
    for ep in range(args_cli.num_episodes):
        obs = env.reset()
        done = False
        goals = 0
        collisions = 0
        steps = 0

        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, dones, infos = env.step(action)
            steps += 1
            done = bool(dones[0])

            info = infos[0] if infos else {}
            goals += int(info.get("patrol/goal_reached", 0))
            collisions += int(info.get("patrol/collision", 0))

        ep_result = {
            "episode": ep + 1,
            "goals": goals,
            "collisions": collisions,
            "steps": steps,
        }
        results.append(ep_result)
        print(f"[Ep {ep+1:02d}/{args_cli.num_episodes}] goals={goals}  collisions={collisions}  steps={steps}")

    goals_mean = float(np.mean([r["goals"] for r in results]))
    collision_rate = float(np.mean([r["collisions"] > 0 for r in results]))

    print(f"\n=== Zero-Shot Baseline Results ===")
    print(f"  goals_per_episode (mean): {goals_mean:.2f}")
    print(f"  collision_rate:           {collision_rate:.2%}")

    if goals_mean > 2.0:
        print("  -> Policy generalizes well. Minimal fine-tuning needed (~50k steps).")
    elif goals_mean >= 0.5:
        print("  -> Moderate fine-tuning needed (200–500k steps).")
    else:
        print("  -> Significant distributional shift. Check VecNormalize stats and waypoint placement.")

    output_path = os.path.join(_training_dir, "logs", "eval", "zero_shot_building_results.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "checkpoint": checkpoint,
            "num_episodes": args_cli.num_episodes,
            "goals_per_episode_mean": goals_mean,
            "collision_rate": collision_rate,
            "episodes": results,
        }, f, indent=2)
    print(f"\n[INFO] Results saved to {output_path}")
    print("[INFO] Record these results in training/EXPERIMENTS.md as the 'zero-shot baseline' row.")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()

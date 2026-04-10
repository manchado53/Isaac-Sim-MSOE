"""Fine-tuning script: load pre-trained Nova Carter stage4 policy and continue
training in the Innovation Center building environment.

Key differences from train_nova_carter.py:
- Task: Isaac-NovaCarter-Building-v0 (building USD, not procedural room)
- Loads pre-trained checkpoint + VecNormalize stats (10× lower LR)
- No curriculum (building complexity replaces procedural stages)
- CheckpointCallback saves every 25 rollouts (Bug 5 already fixed in base)

Staged fine-tuning plan:
  Stage A: Floor G waypoints → target goals/episode ≥ 3.0
  Stage B: Floor 1 waypoints → target goals/episode ≥ 2.5  (load Stage A)
  Stage C: All floors mixed  → target goals/episode ≥ 2.5  (load Stage B)

Usage:
    cd ~/IsaacLab
    ./isaaclab.sh -p ~/isaac-sim-msoe/training/train_nova_carter_building.py \\
        --checkpoint training/logs/sb3/NovaCarter-Patrol/2026-04-08_04-35-47/best_model_stage4.zip \\
        --num_envs 1 --headless
"""

import argparse
import contextlib
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Building fine-tuning for Nova Carter.")
parser.add_argument("--checkpoint", type=str, required=True,
                    help="Pre-trained checkpoint to load (best_model_stage4.zip or later stage).")
parser.add_argument("--vecnorm", type=str, default=None,
                    help="VecNormalize .pkl path. Inferred from checkpoint dir if omitted.")
parser.add_argument("--num_envs", type=int, default=1,
                    help="Number of parallel envs. Keep at 1 until VRAM confirmed OK.")
parser.add_argument("--total_timesteps", type=int, default=500_000,
                    help="Training timesteps for this stage.")
parser.add_argument("--stage", type=str, default="A", choices=["A", "B", "C"],
                    help="Fine-tuning stage. Controls which floor waypoints are active.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -- Safe to import after SimulationApp --
import os
import time
from datetime import datetime

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecNormalize

from isaaclab_rl.sb3 import Sb3VecEnvWrapper, process_sb3_cfg

import sys as _sys
_training_dir = os.path.dirname(os.path.abspath(__file__))
_sys.path.insert(0, _training_dir)

import nova_carter_patrol  # noqa: F401 — registers Isaac-NovaCarter-Building-v0
from nova_carter_patrol.nova_carter_building_env_cfg import (
    NovaCarterBuildingEnvCfg,
    FLOOR_G_WAYPOINTS,
    ALL_WAYPOINTS,
)
from train_nova_carter import PatrolMetricsCallback, ETACallback, BestModelCallback


# Stage → waypoint mapping
STAGE_WAYPOINTS = {
    "A": FLOOR_G_WAYPOINTS,      # Floor G lobby only
    "B": nova_carter_patrol.nova_carter_building_env_cfg._wp_data["1"],  # Floor 1
    "C": ALL_WAYPOINTS,           # All floors
}
STAGE_TARGETS = {"A": 3.0, "B": 2.5, "C": 2.5}


def main():
    checkpoint = args_cli.checkpoint
    vecnorm_path = args_cli.vecnorm
    if vecnorm_path is None:
        ckpt_dir = os.path.dirname(checkpoint)
        vecnorm_path = os.path.join(ckpt_dir, "model_vecnormalize.pkl")

    for p, label in [(checkpoint, "checkpoint"), (vecnorm_path, "VecNormalize stats")]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"{label} not found: {p}")

    stage = args_cli.stage
    waypoints = STAGE_WAYPOINTS[stage]
    print(f"[INFO] Stage {stage}: {len(waypoints)} waypoints, target goals/episode ≥ {STAGE_TARGETS[stage]}")

    # Build env
    env_cfg = NovaCarterBuildingEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    # Override waypoints based on stage
    env_cfg.events.reset_base.params["waypoints"] = waypoints

    raw_env = gym.make("Isaac-NovaCarter-Building-v0", cfg=env_cfg)
    env = Sb3VecEnvWrapper(raw_env)

    # Load VecNormalize stats — CRITICAL: without this all 26 obs are on wrong scale
    env = VecNormalize.load(vecnorm_path, venv=env)
    env.training = True      # Allow running stats to adapt to building obs distribution
    env.norm_reward = False

    # Sanity check: means must NOT be all zeros
    print(f"[DEBUG] VecNormalize obs_rms.mean[:5] = {env.obs_rms.mean[:5]}")
    if np.allclose(env.obs_rms.mean[:5], 0.0):
        raise RuntimeError(
            "VecNormalize means are all zero — stats did not load correctly.\n"
            "Check that the .pkl path matches the checkpoint directory."
        )

    # Log dir
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(
        _training_dir, "logs", "sb3", "NovaCarter-Building", f"stage{stage}_{timestamp}"
    )
    os.makedirs(log_dir, exist_ok=True)
    print(f"[INFO] Log dir: {log_dir}")

    # PPO agent — load from checkpoint with 10× reduced LR to prevent
    # catastrophic forgetting of procedural-room navigation skills.
    FINETUNE_LR = 3e-5  # 10× lower than the 3e-4 used during initial training
    agent = PPO.load(
        checkpoint,
        env=env,
        device="cuda:0",
        tensorboard_log=log_dir,
        learning_rate=FINETUNE_LR,
        target_kl=0.02,  # tighter than 0.03 — prevents large destabilizing updates
    )
    agent.num_timesteps = 0  # reset so learn() trains for the full requested amount

    n_steps_per_rollout = agent.n_steps * args_cli.num_envs

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=n_steps_per_rollout * 25,  # save every 25 rollouts
        save_path=log_dir,
        name_prefix=f"checkpoint_stage{stage}",
    )
    callbacks = [
        PatrolMetricsCallback(),
        ETACallback(num_envs=args_cli.num_envs, steps_per_rollout=n_steps_per_rollout),
        BestModelCallback(save_path=log_dir, name_prefix=f"best_model_building_stage{stage}"),
        checkpoint_callback,
    ]

    print(f"[INFO] Fine-tuning for {args_cli.total_timesteps:,} timesteps "
          f"(LR={FINETUNE_LR}, target_kl=0.02)...")
    start = time.time()
    with contextlib.suppress(KeyboardInterrupt):
        agent.learn(
            total_timesteps=args_cli.total_timesteps,
            callback=callbacks,
            progress_bar=True,
        )

    agent.save(os.path.join(log_dir, f"model_building_stage{stage}"))
    env.save(os.path.join(log_dir, "model_vecnormalize.pkl"))
    print(f"[INFO] Training time: {time.time() - start:.1f}s")
    print(f"[INFO] Model saved to {log_dir}")
    print(f"[INFO] Log this run in training/EXPERIMENTS.md — stage {stage} building fine-tune")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()

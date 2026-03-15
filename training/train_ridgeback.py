"""Train PPO agent for Ridgeback Patrol using Stable-Baselines3.

Usage (on a GPU node):
    cd ~/IsaacLab
    ./isaaclab.sh -p ~/isaac-sim-msoe/scripts/train_ridgeback.py --headless

    # With video recording:
    ./isaaclab.sh -p ~/isaac-sim-msoe/scripts/train_ridgeback.py --headless --video

    # Custom settings:
    ./isaaclab.sh -p ~/isaac-sim-msoe/scripts/train_ridgeback.py --headless --num_envs 512 --max_iterations 200
"""

import argparse
import contextlib
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train Ridgeback Patrol with PPO (SB3).")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of each recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=256, help="Number of environments.")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
parser.add_argument("--max_iterations", type=int, default=None, help="Max policy update iterations (default: from yaml).")
parser.add_argument("--checkpoint", type=str, default=None, help="Resume training from checkpoint.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -- Now safe to import --
import os
import time
from datetime import datetime

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import VecNormalize


class PanCameraCallback(BaseCallback):
    """Moves the viewport camera across environments during video recording.

    Stays on each environment for `steps_per_env` steps, then smoothly moves to the next.
    """

    def __init__(self, env_origins: np.ndarray, steps_per_env: int = 300, verbose: int = 0):
        super().__init__(verbose)
        self.env_origins = env_origins  # (num_envs, 3)
        self.steps_per_env = steps_per_env
        self.num_envs_to_visit = min(16, len(env_origins))
        self._sim = None

    def _on_training_start(self) -> None:
        # Get sim context through the wrapper chain
        venv = self.training_env
        while hasattr(venv, "venv"):
            venv = venv.venv
        if hasattr(venv, "env"):
            self._sim = venv.env.unwrapped.sim

    def _on_step(self) -> bool:
        if self._sim is None:
            return True
        env_idx = (self.num_timesteps // self.steps_per_env) % self.num_envs_to_visit
        ox = float(self.env_origins[env_idx, 0])
        oy = float(self.env_origins[env_idx, 1])
        self._sim.set_camera_view(
            eye=(ox, oy - 8.0, 12.0),
            target=(ox, oy, 0.0),
        )
        return True

from isaaclab.utils.io import dump_yaml
from isaaclab_rl.sb3 import Sb3VecEnvWrapper, process_sb3_cfg

# Import ridgeback_patrol to trigger gym.register()
import ridgeback_patrol  # noqa: F401

from ridgeback_patrol.ridgeback_patrol_env_cfg import RidgebackPatrolFlatEnvCfg


def main():
    # Load agent config from yaml
    import yaml
    agent_cfg_path = os.path.join(os.path.dirname(__file__), "ridgeback_patrol", "agents", "sb3_ppo_cfg.yaml")
    with open(agent_cfg_path) as f:
        agent_cfg = yaml.safe_load(f)

    # Environment config
    env_cfg = RidgebackPatrolFlatEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed

    # Override max iterations if specified
    if args_cli.max_iterations is not None:
        agent_cfg["n_timesteps"] = args_cli.max_iterations * agent_cfg["n_steps"] * env_cfg.scene.num_envs

    # Logging directory
    run_info = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_root = os.path.join(os.path.dirname(__file__), "logs", "sb3", "Ridgeback-Patrol")
    log_dir = os.path.abspath(os.path.join(log_root, run_info))
    os.makedirs(log_dir, exist_ok=True)
    print(f"[INFO] Logging to: {log_dir}")

    # Save configs
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # Process SB3 config (converts string types like 'nn.ReLU' to actual classes)
    agent_cfg = process_sb3_cfg(agent_cfg, env_cfg.scene.num_envs)
    policy_arch = agent_cfg.pop("policy")
    n_timesteps = agent_cfg.pop("n_timesteps")

    # Create environment
    env = gym.make(
        "Isaac-Ridgeback-Patrol-Flat-v0",
        cfg=env_cfg,
        render_mode="rgb_array" if args_cli.video else None,
    )

    # Wrap for video recording
    pan_callback = None
    if args_cli.video:
        # Set initial camera view
        env_origins = env.unwrapped.scene.env_origins.cpu().numpy()
        env.unwrapped.sim.set_camera_view(
            eye=(float(env_origins[0, 0]), float(env_origins[0, 1]) - 8.0, 12.0),
            target=(float(env_origins[0, 0]), float(env_origins[0, 1]), 0.0),
        )
        pan_callback = PanCameraCallback(env_origins, steps_per_env=300)

        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # Wrap for SB3 compatibility
    env = Sb3VecEnvWrapper(env)

    # Remove normalization keys from agent_cfg (not PPO params)
    for key in ["normalize_input", "normalize_value", "clip_obs"]:
        agent_cfg.pop(key, None)

    # Normalize observations only — reward normalization is disabled because
    # with short episodes VecNormalize sees near-zero reward variance, amplifies
    # rewards to extreme values, and causes NaN in PPO's Normal distribution.
    print("[INFO] Applying VecNormalize (obs only, reward norm DISABLED)")
    env = VecNormalize(
        env,
        training=True,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
        gamma=agent_cfg["gamma"],
    )

    start_time = time.time()

    # Create PPO agent
    agent = PPO(policy_arch, env, verbose=1, tensorboard_log=log_dir, **agent_cfg)
    if args_cli.checkpoint is not None:
        agent = agent.load(args_cli.checkpoint, env, print_system_info=True)

    # Callbacks
    callbacks = [CheckpointCallback(save_freq=1000, save_path=log_dir, name_prefix="model", verbose=2)]
    if pan_callback is not None:
        callbacks.append(pan_callback)

    # Train
    print(f"[INFO] Training for {int(n_timesteps):,} timesteps ({args_cli.num_envs} envs)...")
    with contextlib.suppress(KeyboardInterrupt):
        agent.learn(
            total_timesteps=n_timesteps,
            callback=callbacks,
            progress_bar=True,
        )

    # Save final model
    agent.save(os.path.join(log_dir, "model"))
    print(f"\n[INFO] Final model saved to: {os.path.join(log_dir, 'model.zip')}")

    # Save normalization stats (needed for evaluation)
    if isinstance(env, VecNormalize):
        env.save(os.path.join(log_dir, "model_vecnormalize.pkl"))
        print(f"[INFO] Normalization stats saved to: {os.path.join(log_dir, 'model_vecnormalize.pkl')}")

    print(f"[INFO] Training time: {time.time() - start_time:.1f}s")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

"""Evaluate a trained Nova Carter Patrol PPO agent and record video.

Usage:
    cd ~/IsaacLab
    ./isaaclab.sh -p ~/isaac-sim-msoe/training/eval_nova_carter.py \
        --checkpoint ~/isaac-sim-msoe/training/logs/sb3/NovaCarter-Patrol/<run>/model.zip \
        --headless --enable_cameras
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate Nova Carter Patrol PPO agent.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to model.zip")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments (fewer = faster).")
parser.add_argument("--video_length", type=int, default=500, help="Steps per video clip.")
parser.add_argument("--num_videos", type=int, default=1, help="Number of video clips to record.")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -- Now safe to import --
import os

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

from isaaclab_rl.sb3 import Sb3VecEnvWrapper

import nova_carter_patrol  # noqa: F401
from nova_carter_patrol.nova_carter_patrol_env_cfg import NovaCarterPatrolFlatEnvCfg


def main():
    checkpoint_path = os.path.abspath(args_cli.checkpoint)
    checkpoint_dir = os.path.dirname(checkpoint_path)
    vecnorm_path = os.path.join(checkpoint_dir, "model_vecnormalize.pkl")

    # Environment config
    env_cfg = NovaCarterPatrolFlatEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed

    # Create environment with rendering
    env = gym.make("Isaac-NovaCarter-Patrol-Flat-v0", cfg=env_cfg, render_mode="rgb_array")

    # Set camera view
    env_origins = env.unwrapped.scene.env_origins.cpu().numpy()
    env.unwrapped.sim.set_camera_view(
        eye=(float(env_origins[0, 0]), float(env_origins[0, 1]) - 8.0, 12.0),
        target=(float(env_origins[0, 0]), float(env_origins[0, 1]), 0.0),
    )

    # Record video — each clip is video_length steps, spaced apart so we get num_videos clips
    video_dir = os.path.join(checkpoint_dir, "videos", "eval")
    os.makedirs(video_dir, exist_ok=True)
    num_videos = args_cli.num_videos
    vid_len = args_cli.video_length
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=video_dir,
        step_trigger=lambda step: step % vid_len == 0 and step < num_videos * vid_len,
        video_length=vid_len,
        disable_logger=True,
    )
    total_steps = num_videos * vid_len
    print(f"[INFO] Recording {num_videos} videos of {vid_len} steps each to: {video_dir}")

    # Wrap for SB3
    env = Sb3VecEnvWrapper(env)

    # Load normalization stats if available
    if os.path.exists(vecnorm_path):
        env = VecNormalize.load(vecnorm_path, env)
        env.training = False  # don't update stats during eval
        env.norm_reward = False
        print(f"[INFO] Loaded VecNormalize from: {vecnorm_path}")

    # Load model
    agent = PPO.load(checkpoint_path, env=env)
    print(f"[INFO] Loaded model from: {checkpoint_path}")

    # Run evaluation
    obs = env.reset()
    total_reward = np.zeros(args_cli.num_envs)
    for step in range(total_steps):
        action, _ = agent.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += reward

    print(f"\n[INFO] Evaluation complete.")
    print(f"[INFO] Mean reward over {total_steps} steps: {total_reward.mean():.2f}")
    print(f"[INFO] {num_videos} videos saved to: {video_dir}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

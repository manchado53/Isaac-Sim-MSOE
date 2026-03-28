"""Evaluate a trained Nova Carter Patrol PPO agent and record video with stats overlay.

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
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments (fewer = faster).")
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
import imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

from isaaclab_rl.sb3 import Sb3VecEnvWrapper

import nova_carter_patrol  # noqa: F401
from nova_carter_patrol.nova_carter_patrol_env_cfg import NovaCarterPatrolFlatEnvCfg


def add_stats_overlay(frame: np.ndarray, step: int, ep_reward: float, dist_to_goal: float, speed: float, goals_reached: int) -> np.ndarray:
    """Draw a stats bar at the top of the frame."""
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)

    # Semi-transparent black bar at top
    bar_h = 36
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    bar = ImageDraw.Draw(overlay)
    bar.rectangle([(0, 0), (img.width, bar_h)], fill=(0, 0, 0, 160))
    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    stats = (
        f"Step: {step:4d}   "
        f"Reward: {ep_reward:+7.2f}   "
        f"Dist to Goal: {dist_to_goal:.2f}m   "
        f"Speed: {speed:.2f}m/s   "
        f"Goals: {goals_reached}"
    )
    draw.text((10, 9), stats, fill=(255, 255, 255), font=font)
    return np.array(img)


def main():
    checkpoint_path = os.path.abspath(args_cli.checkpoint)
    checkpoint_dir = os.path.dirname(checkpoint_path)
    vecnorm_path = os.path.join(checkpoint_dir, "model_vecnormalize.pkl")

    env_cfg = NovaCarterPatrolFlatEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed

    # Create environment with rendering
    env = gym.make("Isaac-NovaCarter-Patrol-Flat-v0", cfg=env_cfg, render_mode="rgb_array")
    raw_env = env.unwrapped

    # Set camera view
    env_origins = raw_env.scene.env_origins.cpu().numpy()
    raw_env.sim.set_camera_view(
        eye=(float(env_origins[0, 0]), float(env_origins[0, 1]) - 8.0, 12.0),
        target=(float(env_origins[0, 0]), float(env_origins[0, 1]), 0.0),
    )

    # Wrap for SB3
    sb3_env = Sb3VecEnvWrapper(env)

    if os.path.exists(vecnorm_path):
        sb3_env = VecNormalize.load(vecnorm_path, sb3_env)
        sb3_env.training = False
        sb3_env.norm_reward = False
        print(f"[INFO] Loaded VecNormalize from: {vecnorm_path}")

    agent = PPO.load(checkpoint_path, env=sb3_env)
    print(f"[INFO] Loaded model from: {checkpoint_path}")

    video_dir = os.path.join(checkpoint_dir, "videos", "eval")
    os.makedirs(video_dir, exist_ok=True)
    total_steps = args_cli.num_videos * args_cli.video_length

    print(f"[INFO] Recording {args_cli.num_videos} video(s) of {args_cli.video_length} steps to: {video_dir}")

    obs = sb3_env.reset()
    ep_reward = 0.0
    goals_reached = 0
    frames = []

    for step in range(total_steps):
        action, _ = agent.predict(obs, deterministic=True)
        obs, reward, done, info = sb3_env.step(action)
        ep_reward += float(reward[0])
        if done[0]:
            goals_reached += 1
            ep_reward = 0.0

        # Capture frame
        frame = env.render()
        if frame is not None:
            # Get stats from env
            try:
                command = raw_env.command_manager.get_command("pose_command")
                dist_to_goal = float(torch.norm(command[0, :2]).item())
                robot_vel = raw_env.scene["robot"].data.root_lin_vel_b[0]
                speed = float(torch.norm(robot_vel[:2]).item())
            except Exception:
                dist_to_goal = 0.0
                speed = 0.0

            frame = add_stats_overlay(frame, step, ep_reward, dist_to_goal, speed, goals_reached)
            frames.append(frame)

        # Save video at end of each clip
        if len(frames) == args_cli.video_length:
            vid_path = os.path.join(video_dir, f"eval_video_{step // args_cli.video_length:02d}.mp4")
            imageio.mimwrite(vid_path, frames, fps=30, quality=8)
            print(f"[INFO] Saved: {vid_path}")
            frames = []

    print(f"\n[INFO] Evaluation complete. Videos saved to: {video_dir}")
    sb3_env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

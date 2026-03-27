"""Showcase 256 parallel Nova Carter training environments for presentation.

Camera pans across the environment grid in three phases:
  1. Wide overhead shot revealing the full 16x16 grid (~5s)
  2. Descend and sweep across rows showing individual robots (~15s)
  3. Pull back to wide overview shot (~5s)

Usage (on a GPU node):
    cd ~/IsaacLab
    ./isaaclab.sh -p ~/isaac-sim-msoe/training/showcase_parallel_envs.py \
        --num_envs 256 --headless --enable_cameras

    # Fewer envs for faster startup / testing:
    ./isaaclab.sh -p ~/isaac-sim-msoe/training/showcase_parallel_envs.py \
        --num_envs 64 --headless --enable_cameras

Output: ~/isaac-sim-msoe/videos/parallel_showcase.mp4
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Showcase parallel training environments.")
parser.add_argument("--num_envs", type=int, default=256, help="Number of parallel environments.")
parser.add_argument("--fps", type=int, default=30, help="Output video frame rate.")
parser.add_argument("--duration", type=int, default=25, help="Video duration in seconds.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True  # required for rgb_array capture

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -- Safe to import Isaac/gym after AppLauncher --
import os
import sys

import gymnasium as gym
import imageio
import numpy as np

import nova_carter_patrol  # noqa: F401 — triggers gym.register()
from nova_carter_patrol.nova_carter_patrol_env_cfg import NovaCarterPatrolFlatEnvCfg


def smooth_step(t: float) -> float:
    """Smooth cubic interpolation (ease in/out) for t in [0, 1]."""
    return t * t * (3.0 - 2.0 * t)


def lerp(a, b, t):
    return a + (b - a) * t


def camera_pose(t: float, env_origins: np.ndarray) -> tuple:
    """Return (eye, target) for the viewport camera at normalized time t in [0, 1].

    Phase 1 (t=0.00-0.20): Slow overhead pan revealing the full grid.
    Phase 2 (t=0.20-0.80): Descend to mid-height and sweep across rows of environments.
    Phase 3 (t=0.80-1.00): Ascend and pull back to a wide overview.
    """
    xs = env_origins[:, 0]
    ys = env_origins[:, 1]
    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())
    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0
    span_x = x_max - x_min
    span_y = y_max - y_min

    # Wide-shot camera parameters (sees the full grid)
    wide_height = 160.0
    wide_offset_y = span_y * 0.35   # tilt slightly so grid is centered in frame
    wide_eye = (cx, cy - wide_offset_y, wide_height)
    wide_target = (cx, cy, 0.0)

    # Mid-height sweep parameters
    sweep_height = 28.0
    sweep_cam_offset_y = 14.0       # camera sits behind the target row

    if t < 0.20:
        # Phase 1: Slow horizontal drift across the full grid from overhead
        alpha = smooth_step(t / 0.20)
        x = lerp(x_min - span_x * 0.05, cx, alpha)
        eye = (x, cy - wide_offset_y, wide_height)
        target = (x, cy, 0.0)

    elif t < 0.80:
        # Phase 2: Descend to sweep height then pan across rows
        sweep_t = (t - 0.20) / 0.60   # 0→1 over this phase

        # First 15% of phase 2: descend from wide to sweep height
        if sweep_t < 0.15:
            desc = smooth_step(sweep_t / 0.15)
            height = lerp(wide_height, sweep_height, desc)
            # Start at x_min, y at first row
            x = lerp(cx, x_min - span_x * 0.05, desc)
            y = lerp(cy, y_min + span_y * 0.1, desc)
        else:
            height = sweep_height
            pan_t = (sweep_t - 0.15) / 0.85   # 0→1 for the actual sweep
            # Pan right across the grid while drifting upward through rows
            x = x_min - span_x * 0.05 + pan_t * (span_x * 1.10)
            y = y_min + smooth_step(pan_t) * span_y * 0.80

        eye = (x, y - sweep_cam_offset_y, height)
        target = (x, y, 0.0)

    else:
        # Phase 3: Ascend and zoom back out to wide shot
        alpha = smooth_step((t - 0.80) / 0.20)
        # Blend from wherever we ended in phase 2 back to the wide shot
        end_x = x_min + span_x * 1.05    # approximate end-of-sweep x
        end_y = y_min + span_y * 0.80
        x = lerp(end_x, cx, alpha)
        y = lerp(end_y, cy, alpha)
        height = lerp(sweep_height, wide_height, alpha)
        cam_y_off = lerp(sweep_cam_offset_y, wide_offset_y, alpha)
        eye = (x, y - cam_y_off, height)
        target = (x, y, 0.0)

    return eye, target


def main():
    # ── Environment setup ──
    env_cfg = NovaCarterPatrolFlatEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = 42

    print(f"[INFO] Creating {args_cli.num_envs} parallel environments...")
    env = gym.make(
        "Isaac-NovaCarter-Patrol-Flat-v0",
        cfg=env_cfg,
        render_mode="rgb_array",
    )

    env_origins = env.unwrapped.scene.env_origins.cpu().numpy()
    sim = env.unwrapped.sim

    print(f"[INFO] Environment grid: {env_origins.shape[0]} envs")
    print(f"[INFO]   X range: [{env_origins[:,0].min():.1f}, {env_origins[:,0].max():.1f}]")
    print(f"[INFO]   Y range: [{env_origins[:,1].min():.1f}, {env_origins[:,1].max():.1f}]")

    total_frames = args_cli.fps * args_cli.duration
    video_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "videos", "showcase")
    )
    os.makedirs(video_dir, exist_ok=True)

    # Set initial camera before wrapping so RecordVideo sees it from frame 0
    eye0, target0 = camera_pose(0.0, env_origins)
    sim.set_camera_view(eye=eye0, target=target0)

    # RecordVideo captures frames automatically via the gym render pipeline
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=video_dir,
        episode_trigger=lambda ep: ep == 0,   # record first episode only
        video_length=total_frames,
        name_prefix="parallel_showcase",
        disable_logger=True,
    )

    print(f"[INFO] Recording {total_frames} frames ({args_cli.duration}s @ {args_cli.fps}fps)...")
    print(f"[INFO] Output folder: {video_dir}")

    rng = np.random.default_rng(7)
    env_phases = rng.uniform(0, 2 * np.pi, size=args_cli.num_envs)

    obs, _ = env.reset()

    for frame_idx in range(total_frames):
        t = frame_idx / max(total_frames - 1, 1)

        # Pan camera each step
        eye, target = camera_pose(t, env_origins)
        sim.set_camera_view(eye=eye, target=target)

        # Smooth sinusoidal actions so robots look like they're navigating
        step_t = frame_idx * 0.04
        linear_vel = 0.6 + 0.3 * np.sin(step_t * 0.5 + env_phases)
        angular_vel = 0.5 * np.sin(step_t * 0.3 + env_phases * 1.3)
        actions = np.stack([linear_vel, angular_vel], axis=1).astype(np.float32)

        obs, _, terminated, truncated, _ = env.step(actions)

        if frame_idx % 150 == 0:
            pct = 100 * frame_idx / total_frames
            print(f"[INFO]   {frame_idx}/{total_frames} frames ({pct:.0f}%)")

    env.close()
    print(f"[INFO] Done — video saved to: {video_dir}/")


if __name__ == "__main__":
    main()
    simulation_app.close()

#!/usr/bin/env python3
"""Curriculum Stage Demo — records one continuous video showing all 4 stages.

Each stage lasts ~200 env.step() calls (≈16 s at 12 Hz control):
  Stage 1 (steps   0-199): empty room, all furniture in NE corner
  Stage 2 (steps 200-399): wide table in ±2m center zone, robot from ring
  Stage 3 (steps 400-599): table + couch + desk scattered ±3m (static)
  Stage 4 (steps 600-799): same 3 pieces moving at 0.3 m/s

Output MP4: training/logs/curriculum_demo/curriculum_demo-episode-0.mp4

Usage (on a GPU node):
    cd ~/IsaacLab
    ./isaaclab.sh -p ~/isaac-sim-msoe/training/demo_curriculum_stages.py --headless
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Record a short video of all 4 curriculum stages.")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True  # required for render_mode="rgb_array"

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Post-launch imports ────────────────────────────────────────────────────────
import numpy as np
import torch
import gymnasium as gym

# Add training/ to path so nova_carter_patrol package is importable
sys.path.insert(0, os.path.dirname(__file__))
import nova_carter_patrol  # noqa: F401 — triggers gym.register()

from nova_carter_patrol.nova_carter_patrol_env_cfg import NovaCarterPatrolFlatEnvCfg

# ── Demo parameters ────────────────────────────────────────────────────────────
# Each stage gets 200 env.step() calls ≈ 16 s at 12 Hz (decimation=5 @ 60 Hz sim).
# common_step_counter increments by 1 per env.step() call, so these map directly.
STAGE1_STEPS = 200   # transition to Stage 2 at step 200
STAGE2_STEPS = 400   # transition to Stage 3 at step 400
STAGE3_STEPS = 600   # transition to Stage 4 at step 600
TOTAL_STEPS  = 820   # 800 meaningful + 20-step buffer

LOG_DIR = os.path.join(os.path.dirname(__file__), "logs", "curriculum_demo")
os.makedirs(LOG_DIR, exist_ok=True)

# ── Environment configuration ─────────────────────────────────────────────────
env_cfg = NovaCarterPatrolFlatEnvCfg()
env_cfg.scene.num_envs = 1          # single env: deterministic camera, lighter GPU load
env_cfg.curriculum.navigation_difficulty.params["stage1_steps"] = STAGE1_STEPS
env_cfg.curriculum.navigation_difficulty.params["stage2_steps"] = STAGE2_STEPS
env_cfg.curriculum.navigation_difficulty.params["stage3_steps"] = STAGE3_STEPS

# ── Create environment ────────────────────────────────────────────────────────
env = gym.make(
    "Isaac-NovaCarter-Patrol-Flat-v0",
    cfg=env_cfg,
    render_mode="rgb_array",
)

# ── Camera: angled overhead view centred on env 0 ─────────────────────────────
env_origins = env.unwrapped.scene.env_origins.cpu().numpy()
cx, cy = float(env_origins[0, 0]), float(env_origins[0, 1])
env.unwrapped.sim.set_camera_view(
    eye=(cx, cy - 10.0, 14.0),
    target=(cx, cy, 0.0),
)

# ── Video wrapper ─────────────────────────────────────────────────────────────
env = gym.wrappers.RecordVideo(
    env,
    video_folder=LOG_DIR,
    name_prefix="curriculum_demo",
    step_trigger=lambda step: step == 0,  # begin recording on the very first step
    video_length=TOTAL_STEPS,
    disable_logger=True,
)

# ── Run ───────────────────────────────────────────────────────────────────────
print(f"[DEMO] Recording {TOTAL_STEPS} steps (~200 per stage).")
print(f"[DEMO] Output: {LOG_DIR}/curriculum_demo-episode-0.mp4")

obs, _ = env.reset()
# Isaac Lab's env.step() expects a torch tensor, not a numpy array
action = torch.zeros(env.action_space.shape, dtype=torch.float32, device="cuda:0")

for step in range(TOTAL_STEPS):
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, _ = env.reset()

env.close()
print(f"\n[DEMO] Done. Video saved to: {LOG_DIR}/")
simulation_app.close()

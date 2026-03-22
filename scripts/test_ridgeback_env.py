"""Smoke test for the Ridgeback Patrol environment.

Creates the gym env, steps with random actions for 100 steps,
and prints obs/action shapes to verify everything works.

Usage (on a GPU node):
    cd ~/IsaacLab
    ./isaaclab.sh -p ~/isaac-sim-msoe/scripts/test_ridgeback_env.py --headless

Expected output:
    Observation shape: (256, 26)
    Action shape: (256, 2)
    100 steps completed successfully.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Smoke test for Ridgeback Patrol environment.")
parser.add_argument("--num_envs", type=int, default=256, help="Number of environments.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -- Now safe to import --
import torch
import gymnasium as gym

# Import ridgeback_patrol to trigger gym.register()
import ridgeback_patrol  # noqa: F401

from ridgeback_patrol.ridgeback_patrol_env_cfg import RidgebackPatrolFlatEnvCfg


def main():
    env_cfg = RidgebackPatrolFlatEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs

    env = gym.make("Isaac-Ridgeback-Patrol-Flat-v0", cfg=env_cfg)

    obs, info = env.reset()
    print(f"\nObservation shape: {obs['policy'].shape}")
    print(f"Action space: {env.action_space}")
    print(f"Action shape: {env.action_space.shape}")

    for step in range(100):
        actions = torch.tensor(env.action_space.sample(), device="cuda:0", dtype=torch.float32)
        obs, rewards, terminated, truncated, info = env.step(actions)

        if step % 25 == 0:
            print(f"  Step {step:3d} | reward mean: {rewards.mean():.4f} | "
                  f"terminated: {terminated.sum().item():.0f} | truncated: {truncated.sum().item():.0f}")

    print(f"\n100 steps completed successfully.")
    print(f"  Final obs shape: {obs['policy'].shape}")
    print(f"  Expected: ({args_cli.num_envs}, 26)")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

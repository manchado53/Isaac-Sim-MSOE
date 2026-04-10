"""Scenario-based evaluation for Nova Carter Patrol.

Runs 6 fixed test scenarios with deterministic furniture/goal placement,
tracks per-scenario metrics, and produces a pass/fail scorecard.
Uses a single env instance — resets and overrides positions per scenario.
Videos recorded via gym.wrappers.RecordVideo.

Usage:
    cd ~/IsaacLab
    ./isaaclab.sh -p ~/isaac-sim-msoe/training/eval/eval_scenarios.py \
        --checkpoint path/to/model.zip --headless
"""

import argparse
import json
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Scenario-based eval for Nova Carter Patrol.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to model.zip")
parser.add_argument("--num_envs", type=int, default=1, help="Must be 1 for scenario eval.")
parser.add_argument("--scenarios", type=str, default=None, help="Comma-separated scenario indices (1-6). Default: all.")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
parser.add_argument("--video_length", type=int, default=500, help="Unused, kept for compatibility.")
parser.add_argument("--task", type=str, default="Isaac-NovaCarter-Patrol-Flat-v0",
                    help="Gym task ID. Use Isaac-NovaCarter-Building-v0 for building eval.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
args_cli.num_envs = 1

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -- Now safe to import --
import os
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import quat_from_euler_xyz
from isaaclab_rl.sb3 import Sb3VecEnvWrapper

import sys
_eval_dir = os.path.dirname(os.path.abspath(__file__))
_training_dir = os.path.dirname(_eval_dir)
sys.path.insert(0, _eval_dir)
sys.path.insert(0, _training_dir)

import nova_carter_patrol  # noqa: F401
from nova_carter_patrol.nova_carter_patrol_env_cfg import NovaCarterPatrolFlatEnvCfg
from scenario_definitions import SCENARIOS, ScenarioDefinition
from scenario_definitions_building import BUILDING_SCENARIOS, BuildingScenarioDefinition  # noqa: F401


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@dataclass
class ScenarioResult:
    name: str
    difficulty: str
    goal_reached: bool = False
    goals_completed: int = 0
    goals_total: int = 1
    time_to_goal_s: float = float("inf")
    num_collisions: int = 0
    final_distance: float = float("inf")
    total_reward: float = 0.0
    passed: bool = False
    max_collisions: int = 0
    # Debug: closest approach tracking
    closest_distance: float = float("inf")
    closest_step: int = 0
    closest_time_s: float = 0.0
    closest_robot_xy: tuple = (0.0, 0.0)
    closest_goal_xy: tuple = (0.0, 0.0)
    max_contact_force: float = 0.0
    max_contact_force_step: int = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pose(x: float, y: float, z: float, yaw: float, env_origin: torch.Tensor, device) -> torch.Tensor:
    """Create a (1, 7) pose tensor [px, py, pz, qw, qx, qy, qz] in world frame."""
    pos = env_origin + torch.tensor([x, y, z], device=device)
    quat = quat_from_euler_xyz(
        torch.tensor([0.0], device=device),
        torch.tensor([0.0], device=device),
        torch.tensor([yaw], device=device),
    )
    return torch.cat([pos.unsqueeze(0), quat], dim=-1)


def setup_scenario(base_env: ManagerBasedRLEnv, scenario: ScenarioDefinition):
    """Override furniture, robot, and goal positions after env.reset()."""
    device = base_env.device
    env_origin = base_env.scene.env_origins[0]
    env_ids = torch.tensor([0], device=device)

    # 1. Place furniture
    for name, (x, y, z, yaw) in scenario.furniture_positions.items():
        asset = base_env.scene[name]
        pose = _make_pose(x, y, z, yaw, env_origin, device)
        asset.write_root_pose_to_sim(pose, env_ids=env_ids)
        asset.write_root_velocity_to_sim(torch.zeros(1, 6, device=device), env_ids=env_ids)

    # 2. Place robot — flat on the ground
    rx, ry, ryaw = scenario.robot_start
    robot = base_env.scene["robot"]
    robot_pose = _make_pose(rx, ry, 0.1, ryaw, env_origin, device)
    robot.write_root_pose_to_sim(robot_pose, env_ids=env_ids)
    robot.write_root_velocity_to_sim(torch.zeros(1, 6, device=device), env_ids=env_ids)

    # 3. Set first goal
    set_goal(base_env, scenario.goals[0], scenario.robot_start, env_origin)

    # 4. Settle physics — more steps to let everything land properly
    for _ in range(20):
        base_env.sim.step(render=True)
    base_env.scene.update(dt=base_env.physics_dt)

    # 5. Re-place robot flat and zero velocity (in case settle steps knocked it)
    robot_pose = _make_pose(rx, ry, 0.1, ryaw, env_origin, device)
    robot.write_root_pose_to_sim(robot_pose, env_ids=env_ids)
    robot.write_root_velocity_to_sim(torch.zeros(1, 6, device=device), env_ids=env_ids)

    # 6. A few more settle steps to let the re-placed robot land
    for _ in range(5):
        base_env.sim.step(render=True)
    base_env.scene.update(dt=base_env.physics_dt)


def set_goal(base_env: ManagerBasedRLEnv, goal_xy: tuple, robot_start: tuple, env_origin: torch.Tensor):
    """Set the command manager's goal to a fixed position."""
    device = base_env.device
    cmd_term = base_env.command_manager.get_term("pose_command")
    gx, gy = goal_xy
    rx, ry, _ = robot_start

    cmd_term.pos_command_w[0, 0] = env_origin[0] + gx
    cmd_term.pos_command_w[0, 1] = env_origin[1] + gy
    cmd_term.pos_command_w[0, 2] = 0.1

    cmd_term.heading_command_w[0] = math.atan2(gy - ry, gx - rx)
    cmd_term.time_left[0] = 9999.0
    cmd_term.command_counter[0] = 0


def is_colliding(base_env: ManagerBasedRLEnv, force_threshold: float = 1.0) -> bool:
    """Check if the robot is physically contacting any obstacle using contact sensor."""
    contact_sensor = base_env.scene.sensors["contact_sensor"]
    net_forces = contact_sensor.data.net_forces_w  # (num_envs, num_bodies, 3)
    force_magnitude = torch.norm(net_forces, dim=-1)  # (num_envs, num_bodies)
    # Ignore ground contact (small vertical forces from gravity) by checking if any
    # body has a force above threshold
    max_force = force_magnitude[0].max().item()
    return max_force > force_threshold, max_force


def get_goal_distance(base_env: ManagerBasedRLEnv) -> float:
    """Get current 2D distance to goal."""
    cmd_term = base_env.command_manager.get_term("pose_command")
    return cmd_term.metrics["error_pos_2d"][0].item()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    checkpoint_path = os.path.abspath(args_cli.checkpoint)
    checkpoint_dir = os.path.dirname(checkpoint_path)
    vecnorm_path = os.path.join(checkpoint_dir, "model_vecnormalize.pkl")

    # Select scenarios
    if args_cli.scenarios:
        indices = [int(x) - 1 for x in args_cli.scenarios.split(",")]
        scenarios = [SCENARIOS[i] for i in indices]
    else:
        scenarios = SCENARIOS

    # ── Create single environment ──
    env_cfg = NovaCarterPatrolFlatEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.seed = args_cli.seed
    env_cfg.episode_length_s = 600.0  # long timeout

    env = gym.make("Isaac-NovaCarter-Patrol-Flat-v0", cfg=env_cfg, render_mode="rgb_array")
    base_env: ManagerBasedRLEnv = env.unwrapped

    # Camera
    env_origins = base_env.scene.env_origins.cpu().numpy()
    base_env.sim.set_camera_view(
        eye=(float(env_origins[0, 0]), float(env_origins[0, 1]) - 8.0, 12.0),
        target=(float(env_origins[0, 0]), float(env_origins[0, 1]), 0.0),
    )

    # Unique timestamp for this eval run
    from datetime import datetime
    eval_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Video — record all steps as one continuous video
    video_dir = os.path.join(checkpoint_dir, "videos", "scenarios")
    os.makedirs(video_dir, exist_ok=True)

    # Calculate total steps across all scenarios for RecordVideo
    scenario_steps = []
    for s in scenarios:
        scenario_steps.append(s.max_steps_per_goal * len(s.goals))
    total_all_steps = sum(scenario_steps)

    # Record everything as one continuous video with unique name
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=video_dir,
        name_prefix=f"scenario_eval_{eval_timestamp}",
        step_trigger=lambda step: step == 0,
        video_length=total_all_steps + 200,
        disable_logger=True,
    )

    # Wrap for SB3
    env = Sb3VecEnvWrapper(env)

    # Load normalization
    if os.path.exists(vecnorm_path):
        env = VecNormalize.load(vecnorm_path, env)
        env.training = False
        env.norm_reward = False
        print(f"[INFO] Loaded VecNormalize from: {vecnorm_path}")

    # Load model
    agent = PPO.load(checkpoint_path, env=env)
    print(f"[INFO] Loaded model from: {checkpoint_path}")

    # ── Run scenarios ──
    results = []
    env_origin = base_env.scene.env_origins[0]
    goal_threshold = 0.75
    dt = base_env.step_dt

    # Initial reset
    obs = env.reset()

    for scenario_idx, scenario in enumerate(scenarios):
        print(f"\n{'='*60}")
        print(f"Scenario {scenario_idx + 1}: {scenario.name} ({scenario.difficulty})")
        print(f"  {scenario.description}")
        print(f"{'='*60}")

        result = ScenarioResult(
            name=scenario.name,
            difficulty=scenario.difficulty,
            goals_total=len(scenario.goals),
            max_collisions=scenario.max_collisions,
        )

        # Set up this scenario (override positions in existing env)
        setup_scenario(base_env, scenario)

        # Diagnostic: check goal position before and after first step
        cmd_term = base_env.command_manager.get_term("pose_command")
        goal_before = cmd_term.pos_command_w[0].cpu().clone()
        robot_before = base_env.scene["robot"].data.root_pos_w[0].cpu().clone()
        err = cmd_term.metrics.get("error_pos_2d", None)
        err_str = f"{err[0]:.3f}" if err is not None else "N/A"
        print(f"  [DIAG] After setup — robot: ({robot_before[0]-env_origin[0].cpu():.2f}, {robot_before[1]-env_origin[1].cpu():.2f})"
              f"  goal: ({goal_before[0]-env_origin[0].cpu():.2f}, {goal_before[1]-env_origin[1].cpu():.2f})"
              f"  error_pos_2d: {err_str}"
              f"  cmd_counter: {cmd_term.command_counter[0]}"
              f"  ep_len: {base_env.episode_length_buf[0]}"
              f"  reset_buf: {getattr(base_env, 'reset_buf', 'N/A')}"
              f"  time_left: {cmd_term.time_left[0]:.1f}")

        # Step once with zero action to get fresh observations
        action = np.zeros((1, 2), dtype=np.float32)
        obs, _, done_init, _ = env.step(action)

        goal_after = cmd_term.pos_command_w[0].cpu().clone()
        robot_after = base_env.scene["robot"].data.root_pos_w[0].cpu().clone()
        err2 = cmd_term.metrics.get("error_pos_2d", None)
        err2_str = f"{err2[0]:.3f}" if err2 is not None else "N/A"
        # Show which termination(s) fired
        tmgr = base_env.termination_manager
        term_fired = {name: bool(tmgr._term_dones[0, i].item())
                      for i, name in enumerate(tmgr._term_names)}
        print(f"  [DIAG] After step  — robot: ({robot_after[0]-env_origin[0].cpu():.2f}, {robot_after[1]-env_origin[1].cpu():.2f})"
              f"  goal: ({goal_after[0]-env_origin[0].cpu():.2f}, {goal_after[1]-env_origin[1].cpu():.2f})"
              f"  error_pos_2d: {err2_str}"
              f"  cmd_counter: {cmd_term.command_counter[0]}"
              f"  done: {done_init[0]}"
              f"  terms: {term_fired}")

        goal_idx = 0
        total_steps = 0
        max_total_steps = scenario.max_steps_per_goal * len(scenario.goals)
        goal_step_count = 0
        scenario_done = False

        while total_steps < max_total_steps and goal_idx < len(scenario.goals) and not scenario_done:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_steps += 1
            goal_step_count += 1
            result.total_reward += float(reward[0])

            # Check collision via contact sensor (actual physical contact)
            colliding, contact_force = is_colliding(base_env)
            if colliding:
                result.num_collisions += 1
            if contact_force > result.max_contact_force:
                result.max_contact_force = contact_force
                result.max_contact_force_step = total_steps

            # Check goal reached + track closest approach
            dist = get_goal_distance(base_env)
            if dist < result.closest_distance:
                robot_pos_w = base_env.scene["robot"].data.root_pos_w[0].cpu()
                cmd_term = base_env.command_manager.get_term("pose_command")
                goal_pos_w = cmd_term.pos_command_w[0].cpu()
                result.closest_distance = dist
                result.closest_step = total_steps
                result.closest_time_s = total_steps * dt
                result.closest_robot_xy = (
                    (robot_pos_w[0] - env_origin[0].cpu()).item(),
                    (robot_pos_w[1] - env_origin[1].cpu()).item(),
                )
                result.closest_goal_xy = (
                    (goal_pos_w[0] - env_origin[0].cpu()).item(),
                    (goal_pos_w[1] - env_origin[1].cpu()).item(),
                )
            if dist < goal_threshold:
                result.goals_completed += 1
                time_s = goal_step_count * dt
                print(f"  Goal {goal_idx + 1}/{len(scenario.goals)} reached! "
                      f"Time: {time_s:.1f}s, Dist: {dist:.2f}m")

                goal_idx += 1
                if goal_idx < len(scenario.goals):
                    robot_pos = base_env.scene["robot"].data.root_pos_w[0].cpu()
                    current_robot = (
                        (robot_pos[0] - env_origin[0].cpu()).item(),
                        (robot_pos[1] - env_origin[1].cpu()).item(),
                        0.0,
                    )
                    set_goal(base_env, scenario.goals[goal_idx], current_robot, env_origin)
                    goal_step_count = 0

            # Per-goal timeout (multi-goal scenarios)
            if goal_step_count >= scenario.max_steps_per_goal and len(scenario.goals) > 1:
                print(f"  Goal {goal_idx + 1}/{len(scenario.goals)} timed out.")
                goal_idx += 1
                if goal_idx < len(scenario.goals):
                    robot_pos = base_env.scene["robot"].data.root_pos_w[0].cpu()
                    current_robot = (
                        (robot_pos[0] - env_origin[0].cpu()).item(),
                        (robot_pos[1] - env_origin[1].cpu()).item(),
                        0.0,
                    )
                    set_goal(base_env, scenario.goals[goal_idx], current_robot, env_origin)
                    goal_step_count = 0

            # Handle env auto-reset (bad_orientation etc.)
            if done[0]:
                print(f"  Episode terminated early (bad orientation or other).")
                # Re-setup the scenario after auto-reset
                setup_scenario(base_env, scenario)
                obs, _, _, _ = env.step(np.zeros((1, 2), dtype=np.float32))
                scenario_done = True  # mark as failed, move to next scenario

        # Final metrics
        result.final_distance = get_goal_distance(base_env)
        result.time_to_goal_s = total_steps * dt

        if len(scenario.goals) == 1:
            result.goal_reached = result.goals_completed >= 1
        else:
            result.goal_reached = result.goals_completed >= len(scenario.goals)

        # Pass/fail
        time_ok = total_steps <= max_total_steps
        collision_ok = result.num_collisions <= scenario.max_collisions
        if len(scenario.goals) == 1:
            result.passed = result.goal_reached and time_ok and collision_ok
        else:
            result.passed = result.goals_completed >= (len(scenario.goals) - 1) and collision_ok

        results.append(result)
        print(f"  Result: {'PASS' if result.passed else 'FAIL'} "
              f"(goals: {result.goals_completed}/{result.goals_total}, "
              f"collisions: {result.num_collisions}/{result.max_collisions}, "
              f"dist: {result.final_distance:.2f}m)")
        print(f"  [DEBUG] Closest approach: {result.closest_distance:.3f}m "
              f"at step {result.closest_step} (t={result.closest_time_s:.1f}s)")
        print(f"  [DEBUG]   Robot was at: ({result.closest_robot_xy[0]:.2f}, {result.closest_robot_xy[1]:.2f})")
        print(f"  [DEBUG]   Goal was at:  ({result.closest_goal_xy[0]:.2f}, {result.closest_goal_xy[1]:.2f})")
        print(f"  [DEBUG] Max contact force: {result.max_contact_force:.1f}N at step {result.max_contact_force_step}")

    # ── Print scorecard ──
    print(f"\n{'='*90}")
    print(f"NOVA CARTER PATROL — SCENARIO EVALUATION SCORECARD")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"{'='*90}")
    print(f"{'#':<3} {'Scenario':<22} {'Diff':<10} {'Reached':<10} {'Time(s)':<10} "
          f"{'Collisions':<12} {'Final Dist':<12} {'Result':<8}")
    print(f"{'-'*90}")

    num_passed = 0
    for i, r in enumerate(results):
        reached_str = "YES" if r.goal_reached else f"{r.goals_completed}/{r.goals_total}"
        time_str = f"{r.time_to_goal_s:.1f}" if r.goal_reached else "--"
        coll_str = f"{r.num_collisions}/{r.max_collisions}"
        result_str = "PASS" if r.passed else "FAIL"
        if r.passed:
            num_passed += 1

        print(f"{i+1:<3} {r.name:<22} {r.difficulty:<10} {reached_str:<10} {time_str:<10} "
              f"{coll_str:<12} {r.final_distance:<12.2f} {result_str:<8}")

    print(f"{'-'*90}")
    print(f"Overall: {num_passed}/{len(results)} PASSED")
    print(f"{'='*90}")

    # Save results to JSON
    results_path = os.path.join(checkpoint_dir, f"scenario_results_{eval_timestamp}.json")
    results_json = []
    for r in results:
        results_json.append({
            "name": r.name,
            "difficulty": r.difficulty,
            "goal_reached": r.goal_reached,
            "goals_completed": r.goals_completed,
            "goals_total": r.goals_total,
            "time_to_goal_s": r.time_to_goal_s if r.goal_reached else None,
            "num_collisions": r.num_collisions,
            "max_collisions": r.max_collisions,
            "final_distance": r.final_distance,
            "total_reward": r.total_reward,
            "passed": r.passed,
        })
    with open(results_path, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"\nResults saved to: {results_path}")
    print(f"Video saved to: {video_dir}/")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

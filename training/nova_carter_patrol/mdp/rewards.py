"""Custom reward functions for Nova Carter Patrol environment.

Reward terms:
- delta_progress: reward for net distance closed to goal this step (d_prev - d_current)
- distance_to_goal: smooth proximity reward using tanh (pull-in magnet near goal)
- time_penalty: constant -1 per step (combined with negative weight)
- goal_progress: reward for closing distance to goal (forward_vel * cos(heading_error))
- goal_reached: binary bonus on arrival
- position_tracking_tanh: smooth distance-based reward using tanh (unused)
- heading_alignment: penalty for not facing the goal (unused)
- velocity_near_goal: penalty for moving fast near the goal (unused)
- collision_penalty: binary penalty when LiDAR detects contact (unused)
- obstacle_proximity: smooth penalty near obstacles (unused)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def delta_progress(
    env: ManagerBasedRLEnv, command_name: str
) -> torch.Tensor:
    """Reward for net distance closed to goal this step (d_prev - d_current).

    Directly measures actual displacement toward goal rather than approximating
    via velocity. Cannot be gamed by oscillating — only net forward progress counts.

    Returns 0 on the first call and on any step where the distance jumps by more
    than 1m (goal resampling after arrival, or episode reset). This prevents large
    negative spikes when a new goal is assigned far away.

    Uses world-frame distance (pos_command_w - root_pos_w) instead of the stale
    body-frame command (pos_command_b). pos_command_b is refreshed in
    command_manager.compute() which runs AFTER reward_manager.compute(), so it
    always reflects the previous step's robot position.

    Shape: (num_envs,)
    """
    cmd_term = env.command_manager.get_term(command_name)
    d_current = torch.norm(
        cmd_term.pos_command_w[:, :2] - env.scene["robot"].data.root_pos_w[:, :2],
        dim=1,
    )

    if not hasattr(env, "_dp_d_prev"):
        env._dp_d_prev = d_current.clone()
        return torch.zeros(env.num_envs, device=env.device)

    delta = env._dp_d_prev - d_current

    # Zero out any step where the goal jumped (resampled or episode reset).
    # Max legitimate delta at 1 m/s top speed is ~0.1 m/step; 1m is a safe cutoff.
    large_jump = delta.abs() > 1.0
    delta = torch.where(large_jump, torch.zeros_like(delta), delta)

    env._dp_d_prev = d_current.clone()
    return delta


def time_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Constant 1.0 per step — use with a negative weight as a time penalty.

    Calibrated so that a robot moving at 0.2 m/s toward the goal earns
    more from delta_progress (weight=3.0) than it loses from r_time (weight=-0.05).

    Shape: (num_envs,)
    """
    return torch.ones(env.num_envs, device=env.device)


def goal_progress(
    env: ManagerBasedRLEnv, command_name: str
) -> torch.Tensor:
    """Reward for closing distance to the goal each step.

    Computes forward_velocity * cos(heading_error) — the component of the
    robot's velocity directed toward the goal. Equivalent to the instantaneous
    rate of distance closing in m/s.

    Positive when moving toward goal, zero when moving perpendicular,
    negative when moving away. Directly incentivizes efficient direct paths.

    Shape: (num_envs,)
    """
    command = env.command_manager.get_command(command_name)
    heading_error = command[:, 3]
    forward_vel = env.scene["robot"].data.root_lin_vel_b[:, 0]
    return forward_vel * torch.cos(heading_error)


def position_tracking_tanh(
    env: ManagerBasedRLEnv, command_name: str, std: float
) -> torch.Tensor:
    """Smooth reward that increases as the robot gets closer to the goal.

    Returns 1 - tanh(distance / std). The std parameter controls the
    sensitivity scale: use a large std (e.g. 2.0) for coarse tracking and
    a small std (e.g. 0.2) for fine arrival precision.

    Uses world-frame distance (pos_command_w - root_pos_w) instead of the stale
    body-frame command. See delta_progress docstring for timing explanation.

    Shape: (num_envs,)
    """
    cmd_term = env.command_manager.get_term(command_name)
    dist = torch.norm(
        cmd_term.pos_command_w[:, :2] - env.scene["robot"].data.root_pos_w[:, :2],
        dim=1,
    )
    return 1.0 - torch.tanh(dist / std)


def heading_alignment(
    env: ManagerBasedRLEnv, command_name: str
) -> torch.Tensor:
    """Absolute heading error to the goal (to be used with negative weight).

    Returns abs(heading_error) where heading_error is command[:, 3].
    Value is 0 when perfectly aligned, up to pi when facing away.

    Shape: (num_envs,)
    """
    command = env.command_manager.get_command(command_name)
    return torch.abs(command[:, 3])


def velocity_near_goal(
    env: ManagerBasedRLEnv, command_name: str, threshold: float
) -> torch.Tensor:
    """Penalty for moving fast when near the goal.

    When within `threshold` meters of the goal, returns the robot's speed.
    When farther away, returns 0. Used with a negative weight to teach
    the robot to decelerate on approach.

    Shape: (num_envs,)
    """
    command = env.command_manager.get_command(command_name)
    dist = torch.norm(command[:, :2], dim=-1)

    # Get robot velocity (world frame)
    robot_vel = env.scene["robot"].data.root_lin_vel_w[:, :2]  # (num_envs, 2)
    speed = torch.norm(robot_vel, dim=-1)  # (num_envs,)

    # Only penalize speed when near the goal
    near_goal = (dist < threshold).float()
    return near_goal * speed


def _lidar_distances(sensor: RayCaster) -> torch.Tensor:
    """Compute LiDAR distances, replacing NaN/Inf with max_distance. Shape: (num_envs, num_rays)."""
    diff = sensor.data.ray_hits_w - sensor.data.pos_w.unsqueeze(1)
    distances = torch.norm(diff, dim=-1)
    max_dist = sensor.cfg.max_distance
    distances = torch.where(torch.isfinite(distances), distances, torch.full_like(distances, max_dist))
    return distances


def goal_reached(
    env: ManagerBasedRLEnv, command_name: str, threshold: float
) -> torch.Tensor:
    """Binary reward of 1.0 when the robot is within threshold of the goal.

    Used as a bonus reward on top of tanh shaping to give an explicit
    success signal when the robot arrives at the goal.

    Uses world-frame distance (pos_command_w - root_pos_w) instead of the stale
    body-frame command. See delta_progress docstring for timing explanation.

    Shape: (num_envs,)
    """
    cmd_term = env.command_manager.get_term(command_name)
    dist = torch.norm(
        cmd_term.pos_command_w[:, :2] - env.scene["robot"].data.root_pos_w[:, :2],
        dim=1,
    )
    return (dist < threshold).float()


def collision_penalty(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Binary penalty of 1.0 when the minimum LiDAR range is below threshold.

    Shape: (num_envs,)
    """
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    distances = _lidar_distances(sensor)
    min_dist = distances.min(dim=-1).values

    return (min_dist < threshold).float()


def obstacle_proximity(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, safe_dist: float
) -> torch.Tensor:
    """Smooth penalty that increases as the robot approaches obstacles.

    Returns max(0, 1 - min_lidar / safe_dist). Value is 0 when min_lidar >= safe_dist,
    and approaches 1.0 as min_lidar approaches 0.

    Shape: (num_envs,)
    """
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    distances = _lidar_distances(sensor)
    min_dist = distances.min(dim=-1).values

    return torch.clamp(1.0 - min_dist / safe_dist, min=0.0)

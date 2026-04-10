"""Custom observation functions for Nova Carter Patrol environment.

Observations (26-dim total):
- LiDAR scan normalized (20)
- Relative goal position in body frame (2)
- Relative goal heading as cos/sin (2)
- Forward velocity (1)
- Yaw rate (1)

All functions sanitize NaN/Inf to prevent policy network crashes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def _sanitize(t: torch.Tensor, fill: float = 0.0) -> torch.Tensor:
    """Replace NaN/Inf with fill value."""
    return torch.nan_to_num(t, nan=fill, posinf=fill, neginf=fill)


def lidar_scan_normalized(
    env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Normalized LiDAR distances in [0, 1]. Shape: (num_envs, num_rays).

    Applies domain randomization for sim-to-real transfer:
    - Gaussian range noise (σ=0.05m, matching RPLidar S2E at close range)
    - 5% ray dropout (rays randomly set to max_range per step)
    - Max-range sentinel: 0.0 = open space, 1.0 = obstacle at max range
    """
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    hit_positions = sensor.data.ray_hits_w  # (N, B, 3)
    sensor_positions = sensor.data.pos_w  # (N, 3)

    diff = hit_positions - sensor_positions.unsqueeze(1)
    distances = torch.norm(diff, dim=-1)

    max_range = sensor.cfg.max_distance

    # Add Gaussian noise (σ=0.05m; typical for RPLidar S2E at close range)
    noise_std = 0.05
    noise = torch.randn_like(distances) * noise_std
    distances = torch.clamp(distances + noise, 0.0, max_range)

    # 5% ray dropout: randomly set rays to max_range each step
    dropout_mask = torch.rand_like(distances) < 0.05
    distances = torch.where(dropout_mask, torch.full_like(distances, max_range), distances)

    normalized = distances / max_range
    # Fix sentinel: 0.0 = open space (no hit), 1.0 = obstacle at max range.
    # Previous code used 1.0 for missed rays — semantically inverted.
    normalized = torch.where(torch.isfinite(normalized), normalized, torch.zeros_like(normalized))
    return torch.clamp(normalized, 0.0, 1.0)


def relative_goal_position(
    env: ManagerBasedEnv, command_name: str
) -> torch.Tensor:
    """Relative goal position (x, y) in body frame. Shape: (num_envs, 2)."""
    command = env.command_manager.get_command(command_name)
    return _sanitize(command[:, :2])


def relative_goal_heading(
    env: ManagerBasedEnv, command_name: str
) -> torch.Tensor:
    """Relative goal heading as (cos, sin). Shape: (num_envs, 2)."""
    command = env.command_manager.get_command(command_name)
    heading = command[:, 3]
    result = torch.stack([torch.cos(heading), torch.sin(heading)], dim=-1)
    return _sanitize(result)


def forward_velocity(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Forward (x-axis) linear velocity in body frame. Shape: (num_envs, 1)."""
    asset = env.scene[asset_cfg.name]
    return _sanitize(asset.data.root_lin_vel_b[:, 0:1])


def yaw_rate(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Yaw (z-axis) angular velocity in body frame. Shape: (num_envs, 1)."""
    asset = env.scene[asset_cfg.name]
    return _sanitize(asset.data.root_ang_vel_b[:, 2:3])

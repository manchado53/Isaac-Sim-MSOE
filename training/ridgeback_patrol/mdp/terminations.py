"""Custom termination functions for Carter Patrol environment.

Termination conditions:
- goal_reached: episode ends when robot reaches the goal
- collision: episode ends when robot collides with an obstacle
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def goal_reached(
    env: ManagerBasedRLEnv, command_name: str, threshold: float
) -> torch.Tensor:
    """Terminate when the robot is within threshold distance of the goal.

    Shape: (num_envs,) bool tensor.
    """
    command = env.command_manager.get_command(command_name)
    dist = torch.norm(command[:, :2], dim=-1)
    return dist < threshold


def collision(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Terminate when the minimum LiDAR range is below threshold (collision detected).

    Shape: (num_envs,) bool tensor.
    """
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    diff = sensor.data.ray_hits_w - sensor.data.pos_w.unsqueeze(1)
    distances = torch.norm(diff, dim=-1)  # (num_envs, num_rays)
    # Replace NaN/Inf (missed rays) with max_distance so they don't trigger collision
    max_dist = sensor.cfg.max_distance
    distances = torch.where(torch.isfinite(distances), distances, torch.full_like(distances, max_dist))
    min_dist = distances.min(dim=-1).values

    return min_dist < threshold

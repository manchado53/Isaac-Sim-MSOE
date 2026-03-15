"""Custom reward functions for Carter Patrol environment.

Reward terms:
- GoalProgressReward: stateful reward tracking distance reduction to goal
- goal_reached: binary bonus when close to goal
- collision_penalty: binary penalty when LiDAR detects contact
- obstacle_proximity: smooth penalty near obstacles
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.sensors import RayCaster

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class GoalProgressReward(ManagerTermBase):
    """Reward for making progress toward the goal.

    Stores the previous distance to the goal and returns the reduction
    in distance each step: (prev_dist - curr_dist). Positive when the
    robot moves closer, negative when it moves away.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._command_name: str = cfg.params["command_name"]
        # Initialize prev_dist — will be set properly on first call
        self._prev_dist = torch.zeros(env.num_envs, device=env.device)
        self._initialized = False

    def __call__(
        self, env: ManagerBasedRLEnv, command_name: str
    ) -> torch.Tensor:
        # Get current distance to goal (body-frame x, y)
        command = env.command_manager.get_command(command_name)
        curr_dist = torch.norm(command[:, :2], dim=-1)

        if not self._initialized:
            self._prev_dist = curr_dist.clone()
            self._initialized = True

        # Progress = reduction in distance
        progress = self._prev_dist - curr_dist
        self._prev_dist = curr_dist.clone()

        return progress

    def reset(self, env_ids: torch.Tensor | None = None):
        """Reset stored distances for environments that terminated."""
        if env_ids is not None and len(env_ids) > 0:
            command = self._env.command_manager.get_command(self._command_name)
            self._prev_dist[env_ids] = torch.norm(command[env_ids, :2], dim=-1)


def goal_reached(
    env: ManagerBasedRLEnv, command_name: str, threshold: float
) -> torch.Tensor:
    """Binary reward of 1.0 when the robot is within threshold distance of the goal.

    Shape: (num_envs,)
    """
    command = env.command_manager.get_command(command_name)
    dist = torch.norm(command[:, :2], dim=-1)
    return (dist < threshold).float()


def _lidar_distances(sensor: RayCaster) -> torch.Tensor:
    """Compute LiDAR distances, replacing NaN/Inf with max_distance. Shape: (num_envs, num_rays)."""
    diff = sensor.data.ray_hits_w - sensor.data.pos_w.unsqueeze(1)
    distances = torch.norm(diff, dim=-1)
    max_dist = sensor.cfg.max_distance
    distances = torch.where(torch.isfinite(distances), distances, torch.full_like(distances, max_dist))
    return distances


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

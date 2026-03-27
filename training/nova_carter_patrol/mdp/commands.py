"""Custom command term with goal-reached resampling for patrol training.

Subclasses UniformPose2dCommand to resample the goal immediately when the robot
reaches it, following the InHandReOrientationCommand pattern from Isaac Lab.
This teaches patrol behavior (goal -> goal -> goal) and maximizes training
signal per episode.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.commands.pose_2d_command import UniformPose2dCommand
from isaaclab.envs.mdp.commands.commands_cfg import UniformPose2dCommandCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


# Clearance radius per furniture piece = half-diagonal of XY footprint (covers any yaw rotation)
# + 0.3m robot body margin. Goals sampled within this radius of a furniture center are rejected.
_FURNITURE_CLEARANCE_RADII: dict[str, float] = {
    "furniture_table":     0.971,  # sqrt(0.6^2 + 0.3^2)    + 0.3
    "furniture_bookshelf": 0.932,  # sqrt(0.2^2 + 0.6^2)    + 0.3
    "furniture_chair":     0.618,  # sqrt(0.225^2 + 0.225^2) + 0.3
    "furniture_desk":      1.128,  # sqrt(0.75^2 + 0.35^2)   + 0.3
    "furniture_cabinet":   0.620,  # sqrt(0.25^2 + 0.2^2)    + 0.3
    "furniture_couch":     1.266,  # sqrt(0.9^2 + 0.35^2)    + 0.3
    "furniture_bin":       0.512,  # sqrt(0.15^2 + 0.15^2)   + 0.3
}
_MAX_REJECTION_ATTEMPTS: int = 10


class GoalReachPose2dCommand(UniformPose2dCommand):
    """Pose 2D command that resamples the goal when the robot reaches it.

    Extends UniformPose2dCommand with goal-reached detection. When the robot's
    2D position error drops below ``goal_reached_threshold``, the goal is
    immediately resampled (via ``_resample``), incrementing ``command_counter``.

    A fallback timer (``resampling_time_range``) still applies so the robot
    doesn't get stuck on unreachable goals forever.

    Tracks ``consecutive_success`` metric for logging.
    """

    cfg: GoalReachPose2dCommandCfg

    def __init__(self, cfg: GoalReachPose2dCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        # Track consecutive goal reaches per env
        self.metrics["consecutive_success"] = torch.zeros(self.num_envs, device=self.device)

    def _is_in_collision(self, goal_xy: torch.Tensor, env_ids: torch.Tensor) -> torch.Tensor:
        """Return bool mask (len(env_ids),) — True where goal overlaps any furniture bounding circle.

        Args:
            goal_xy:  Candidate XY goals in world frame. Shape: (len(env_ids), 2).
            env_ids:  1-D int64 tensor indexing into the env dimension.
        """
        in_collision = torch.zeros(len(env_ids), dtype=torch.bool, device=self.device)
        for name, radius in _FURNITURE_CLEARANCE_RADII.items():
            furn_pos = self._env.scene[name].data.root_pos_w[env_ids, :2]  # (N, 2) world-frame
            dist_sq = ((goal_xy - furn_pos) ** 2).sum(dim=1)               # (N,)
            in_collision |= dist_sq < (radius * radius)
        return in_collision

    def _resample_command(self, env_ids: Sequence[int]):
        """Sample goals, then iteratively resample any that collide with furniture."""
        super()._resample_command(env_ids)  # sets pos_command_w + heading_command_w

        remaining = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if len(remaining) == 0:
            return

        for attempt in range(_MAX_REJECTION_ATTEMPTS):
            bad_mask = self._is_in_collision(self.pos_command_w[remaining, :2], remaining)
            if not bad_mask.any():
                break
            print(f"[GOAL REJECTION] attempt {attempt + 1}: {bad_mask.sum().item()} goal(s) inside furniture, resampling", flush=True)
            remaining = remaining[bad_mask]

            # Re-sample XY only — no counter increment, no Z/heading reset
            origins = self._env.scene.env_origins[remaining]
            r = torch.empty(len(remaining), device=self.device)
            self.pos_command_w[remaining, 0] = origins[:, 0] + r.uniform_(*self.cfg.ranges.pos_x)
            self.pos_command_w[remaining, 1] = origins[:, 1] + r.uniform_(*self.cfg.ranges.pos_y)
        # Envs still colliding after max attempts keep their last sample.
        # The 15s fallback timer remains the safety net for genuinely unreachable goals.

    def _update_command(self):
        """Check for goal reached and resample, then do normal command update."""
        # Check which envs have reached the goal
        goal_reached = self.metrics["error_pos_2d"] < self.cfg.goal_reached_threshold
        goal_reset_ids = goal_reached.nonzero(as_tuple=False).squeeze(-1)

        # Increment consecutive success counter for reached envs
        if len(goal_reset_ids) > 0:
            self.metrics["consecutive_success"][goal_reset_ids] += 1
            # Resample goals for envs that reached them
            self._resample(goal_reset_ids)

        # Parent updates body-frame command from world-frame targets
        super()._update_command()


@configclass
class GoalReachPose2dCommandCfg(UniformPose2dCommandCfg):
    """Configuration for GoalReachPose2dCommand.

    Adds ``goal_reached_threshold`` to control when goals are considered reached.
    The ``resampling_time_range`` acts as a fallback timer for unreachable goals.
    """

    class_type: type = GoalReachPose2dCommand

    goal_reached_threshold: float = 0.5
    """Distance in meters below which the goal is considered reached and resampled."""

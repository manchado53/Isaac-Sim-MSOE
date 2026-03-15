"""Custom command term with goal-reached resampling for patrol training.

Subclasses UniformPose2dCommand to resample the goal immediately when the robot
reaches it, following the InHandReOrientationCommand pattern from Isaac Lab.
This teaches patrol behavior (goal -> goal -> goal) and maximizes training
signal per episode.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.commands.pose_2d_command import UniformPose2dCommand
from isaaclab.envs.mdp.commands.commands_cfg import UniformPose2dCommandCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


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

"""Differential drive action term for Nova Carter robot.

Converts 2D velocity commands (linear_vel, angular_vel) into left/right
wheel velocity targets using differential drive kinematics.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.managers.manager_term_cfg import ActionTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

logger = logging.getLogger(__name__)


class DifferentialDriveAction(ActionTerm):
    """Differential drive action converting (v, omega) to wheel velocity targets.

    The action is a 2D vector [linear_vel, angular_vel]. Using differential
    drive kinematics, these are converted to left/right wheel angular velocities:

        v_left  = (v - omega * wheel_base / 2) / wheel_radius
        v_right = (v + omega * wheel_base / 2) / wheel_radius
    """

    cfg: DifferentialDriveActionCfg
    _asset: Articulation

    def __init__(self, cfg: DifferentialDriveActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        # Resolve left wheel joint
        left_ids, left_names = self._asset.find_joints(self.cfg.left_wheel_joint)
        if len(left_ids) != 1:
            raise ValueError(
                f"Expected 1 joint for left wheel '{self.cfg.left_wheel_joint}', got {len(left_ids)}"
            )
        # Resolve right wheel joint
        right_ids, right_names = self._asset.find_joints(self.cfg.right_wheel_joint)
        if len(right_ids) != 1:
            raise ValueError(
                f"Expected 1 joint for right wheel '{self.cfg.right_wheel_joint}', got {len(right_ids)}"
            )

        self._joint_ids = [left_ids[0], right_ids[0]]
        self._joint_names = [left_names[0], right_names[0]]
        logger.info(
            f"DifferentialDriveAction: joints {self._joint_names} [{self._joint_ids}]"
        )

        # Action tensors
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._wheel_vel_targets = torch.zeros(self.num_envs, 2, device=self.device)

        # Scale and offset
        self._scale = torch.tensor(self.cfg.scale, device=self.device).unsqueeze(0)
        self._offset = torch.tensor(self.cfg.offset, device=self.device).unsqueeze(0)

    @property
    def action_dim(self) -> int:
        return 2

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        self._processed_actions = self._raw_actions * self._scale + self._offset

    def apply_actions(self):
        v = self._processed_actions[:, 0]  # linear velocity
        omega = self._processed_actions[:, 1]  # angular velocity

        half_base = self.cfg.wheel_base / 2.0
        r = self.cfg.wheel_radius

        # Differential drive kinematics
        self._wheel_vel_targets[:, 0] = (v - omega * half_base) / r  # left wheel
        self._wheel_vel_targets[:, 1] = (v + omega * half_base) / r  # right wheel

        self._asset.set_joint_velocity_target(self._wheel_vel_targets, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._raw_actions[env_ids] = 0.0


@configclass
class DifferentialDriveActionCfg(ActionTermCfg):
    """Configuration for differential drive action term."""

    class_type: type[ActionTerm] = DifferentialDriveAction

    left_wheel_joint: str = "joint_wheel_left"
    """Name of the left wheel revolute joint."""
    right_wheel_joint: str = "joint_wheel_right"
    """Name of the right wheel revolute joint."""
    wheel_radius: float = 0.14
    """Radius of the drive wheels in meters."""
    wheel_base: float = 0.4
    """Distance between left and right wheels in meters."""
    scale: tuple[float, float] = (1.0, 1.0)
    """Scale factor for (linear_vel, angular_vel). Defaults to (1.0, 1.0)."""
    offset: tuple[float, float] = (0.0, 0.0)
    """Offset for (linear_vel, angular_vel). Defaults to (0.0, 0.0)."""

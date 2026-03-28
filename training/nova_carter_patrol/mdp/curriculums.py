"""Curriculum learning for Nova Carter Patrol.

Progressively increases navigation difficulty in 3 stages:
  Stage 1 (iters 0-152):   No furniture, close goals (±2m)
  Stage 2 (iters 152-700): 3 furniture pieces, medium goals (±3m)
  Stage 3 (iters 700+):    All 7 furniture, full goals (±4m)
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.managers import ManagerTermBase

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# Furniture spawn ranges — pushed to far corners (out of the way)
_CORNER_RANGE = {"x": (4.0, 4.5), "y": (4.0, 4.5), "yaw": (0.0, 0.0)}

# Normal spawn ranges per furniture piece (from EventCfg)
_NORMAL_RANGES = {
    "reset_furniture_table":     {"x": (1.5, 4.3), "y": (1.5, 4.3), "yaw": (-math.pi, math.pi)},
    "reset_furniture_bookshelf": {"x": (-4.3, -1.5), "y": (1.5, 4.3), "yaw": (-math.pi, math.pi)},
    "reset_furniture_chair":     {"x": (1.5, 4.5), "y": (-4.5, -1.5), "yaw": (-math.pi, math.pi)},
    "reset_furniture_desk":      {"x": (-4.0, -1.5), "y": (-4.0, -1.5), "yaw": (-math.pi, math.pi)},
    "reset_furniture_cabinet":   {"x": (1.5, 4.5), "y": (-1.5, 1.5), "yaw": (-math.pi, math.pi)},
    "reset_furniture_couch":     {"x": (-4.0, -1.5), "y": (-1.5, 1.5), "yaw": (-math.pi, math.pi)},
    "reset_furniture_bin":       {"x": (-1.5, 1.5), "y": (1.5, 4.5), "yaw": (-math.pi, math.pi)},
}

# Stage 2 furniture — the 3 largest pieces
_STAGE2_FURNITURE = ["reset_furniture_table", "reset_furniture_desk", "reset_furniture_couch"]


class NavigationCurriculum(ManagerTermBase):
    """Progressively increase navigation difficulty over training.

    Modifies furniture spawn ranges and goal command ranges based on
    training step count. Returns current stage (1, 2, or 3) for logging.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.current_stage = 0  # uninitialized

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        stage1_steps: int = 78125,    # ~152 iters * 512 steps
        stage2_steps: int = 358400,   # ~700 iters * 512 steps
    ) -> float:
        step = env.common_step_counter
        new_stage = 3 if step >= stage2_steps else (2 if step >= stage1_steps else 1)

        if new_stage == self.current_stage:
            return float(self.current_stage)

        # Stage transition
        self.current_stage = new_stage
        print(f"[CURRICULUM] Advancing to Stage {new_stage} at step {step}", flush=True)

        if new_stage == 1:
            self._apply_stage1(env)
        elif new_stage == 2:
            self._apply_stage2(env)
        elif new_stage == 3:
            self._apply_stage3(env)

        return float(self.current_stage)

    def _apply_stage1(self, env: ManagerBasedRLEnv):
        """Empty room, close goals."""
        # Push all furniture to corners
        for term_name in _NORMAL_RANGES:
            self._set_furniture_range(env, term_name, _CORNER_RANGE)

        # Close goal range
        cmd_term = env.command_manager.get_term("pose_command")
        cmd_term.cfg.ranges.pos_x = (-2.0, 2.0)
        cmd_term.cfg.ranges.pos_y = (-2.0, 2.0)
        print("[CURRICULUM] Stage 1: No furniture, goals ±2m", flush=True)

    def _apply_stage2(self, env: ManagerBasedRLEnv):
        """3 large furniture pieces, medium goals."""
        # Bring 3 pieces into play, rest stay in corners
        for term_name in _NORMAL_RANGES:
            if term_name in _STAGE2_FURNITURE:
                self._set_furniture_range(env, term_name, _NORMAL_RANGES[term_name])
            else:
                self._set_furniture_range(env, term_name, _CORNER_RANGE)

        # Medium goal range
        cmd_term = env.command_manager.get_term("pose_command")
        cmd_term.cfg.ranges.pos_x = (-3.0, 3.0)
        cmd_term.cfg.ranges.pos_y = (-3.0, 3.0)
        print("[CURRICULUM] Stage 2: 3 furniture pieces, goals ±3m", flush=True)

    def _apply_stage3(self, env: ManagerBasedRLEnv):
        """Full furniture, full goals."""
        # All furniture at normal ranges
        for term_name in _NORMAL_RANGES:
            self._set_furniture_range(env, term_name, _NORMAL_RANGES[term_name])

        # Full goal range
        cmd_term = env.command_manager.get_term("pose_command")
        cmd_term.cfg.ranges.pos_x = (-4.0, 4.0)
        cmd_term.cfg.ranges.pos_y = (-4.0, 4.0)
        print("[CURRICULUM] Stage 3: All 7 furniture pieces, goals ±4m", flush=True)

    def _set_furniture_range(self, env: ManagerBasedRLEnv, term_name: str, pose_range: dict):
        """Update a furniture event term's pose_range."""
        term_cfg = env.event_manager.get_term_cfg(term_name)
        term_cfg.params["pose_range"] = pose_range
        env.event_manager.set_term_cfg(term_name, term_cfg)

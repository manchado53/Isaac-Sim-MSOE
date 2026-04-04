"""Curriculum learning for Nova Carter Patrol.

Progressively increases navigation difficulty in 4 stages:
  Stage 1 (empty):    No furniture, close goals (±2m)
  Stage 2 (blocker):  Desk + couch in center zone, goals ±3m
  Stage 3 (corridor): Bookshelf + couch form a randomized gap, goals ±4m
  Stage 4 (clutter):  All 7 pieces in center-biased subzones, goals ±4m

Stage transition steps are configured in CurriculumCfg params.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.managers import ManagerTermBase

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ── Shared corner range — pushes furniture out of the way ──────────────────
_CORNER_RANGE = {"x": (4.0, 4.5), "y": (4.0, 4.5), "yaw": (0.0, 0.0)}

# ── Stage 2: center blocker ────────────────────────────────────────────────
# Desk (1.5×0.7m) and couch (1.8×0.7m) placed anywhere in the ±2m center
# zone. Robot spawns at ±1m, goals at ±3m — path always crosses the center,
# so these pieces are guaranteed to be in the way.
_CENTER_BLOCKER_RANGE = {"x": (-2.0, 2.0), "y": (-2.0, 2.0), "yaw": (-math.pi, math.pi)}

# ── Stage 3: corridor ─────────────────────────────────────────────────────
# Bookshelf on the left, couch on the right, forming a randomized gap.
# Fixed yaw keeps furniture faces aligned so the gap is always traversable.
# Gap width varies from ~1.2m (tightest) to ~3.0m (widest) depending on
# where each piece lands. Desk adds a second blocker beyond the corridor.
_CORRIDOR_LEFT    = {"x": (-1.5, -0.6), "y": (-2.5, 2.5), "yaw": (0.0, 0.0)}       # bookshelf
_CORRIDOR_RIGHT   = {"x": (0.6,  1.5),  "y": (-2.5, 2.5), "yaw": (1.57, 1.57)}     # couch (rotated)
_CORRIDOR_BLOCKER = {"x": (-2.5, 2.5),  "y": (-2.5, 2.5), "yaw": (-math.pi, math.pi)}  # desk

# ── Stage 4: dense clutter ────────────────────────────────────────────────
# All 7 pieces in center-biased subzones. Each piece gets its own quadrant
# within the center region to reduce overlap while still filling the room.
_DENSE_CENTER = {
    "reset_furniture_table":     {"x": (-3.0,  0.0), "y": (0.0,  3.0), "yaw": (-math.pi, math.pi)},
    "reset_furniture_bookshelf": {"x": (0.0,   3.0), "y": (0.0,  3.0), "yaw": (-math.pi, math.pi)},
    "reset_furniture_chair":     {"x": (0.5,   3.0), "y": (-3.0, 0.5), "yaw": (-math.pi, math.pi)},
    "reset_furniture_desk":      {"x": (-3.0,  0.0), "y": (-3.0, 0.0), "yaw": (-math.pi, math.pi)},
    "reset_furniture_cabinet":   {"x": (-1.5,  1.5), "y": (0.0,  3.0), "yaw": (-math.pi, math.pi)},
    "reset_furniture_couch":     {"x": (-3.0,  3.0), "y": (-1.5, 1.5), "yaw": (-math.pi, math.pi)},
    "reset_furniture_bin":       {"x": (-1.5,  1.5), "y": (-3.0, 0.0), "yaw": (-math.pi, math.pi)},
}

# All furniture event term names (for bulk-cornering)
_ALL_FURNITURE = list(_DENSE_CENTER.keys())


class NavigationCurriculum(ManagerTermBase):
    """Progressively increase navigation difficulty over training.

    Stage 1 — empty room, close goals.
    Stage 2 — two large blockers in the center zone (path is always blocked).
    Stage 3 — bookshelf + couch form a randomized corridor; desk adds depth.
    Stage 4 — all 7 pieces in center-biased subzones; full clutter.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.current_stage = 0  # uninitialized

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        stage1_steps: int = 25600,    # ~50 iters
        stage2_steps: int = 102400,   # ~200 iters
        stage3_steps: int = 179200,   # ~350 iters
    ) -> float:
        step = env.common_step_counter
        new_stage = (4 if step >= stage3_steps else
                     3 if step >= stage2_steps else
                     2 if step >= stage1_steps else 1)

        if new_stage == self.current_stage:
            return float(self.current_stage)

        self.current_stage = new_stage
        print(f"[CURRICULUM] Advancing to Stage {new_stage} at step {step}", flush=True)

        if new_stage == 1:
            self._apply_stage1(env)
        elif new_stage == 2:
            self._apply_stage2(env)
        elif new_stage == 3:
            self._apply_stage3(env)
        elif new_stage == 4:
            self._apply_stage4(env)

        return float(self.current_stage)

    # ── Stage methods ──────────────────────────────────────────────────────

    def _apply_stage1(self, env: ManagerBasedRLEnv):
        """Empty room — all furniture to corners, close goals."""
        for term_name in _ALL_FURNITURE:
            self._set_furniture_range(env, term_name, _CORNER_RANGE)
        cmd_term = env.command_manager.get_term("pose_command")
        cmd_term.cfg.ranges.pos_x = (-2.0, 2.0)
        cmd_term.cfg.ranges.pos_y = (-2.0, 2.0)
        print("[CURRICULUM] Stage 1: No furniture, goals ±2m", flush=True)

    def _apply_stage2(self, env: ManagerBasedRLEnv):
        """Center blocker — desk + couch in center zone, medium goals."""
        for term_name in _ALL_FURNITURE:
            self._set_furniture_range(env, term_name, _CORNER_RANGE)
        self._set_furniture_range(env, "reset_furniture_desk",  _CENTER_BLOCKER_RANGE)
        self._set_furniture_range(env, "reset_furniture_couch", _CENTER_BLOCKER_RANGE)
        cmd_term = env.command_manager.get_term("pose_command")
        cmd_term.cfg.ranges.pos_x = (-3.0, 3.0)
        cmd_term.cfg.ranges.pos_y = (-3.0, 3.0)
        print("[CURRICULUM] Stage 2: Desk + couch in center zone, goals ±3m", flush=True)

    def _apply_stage3(self, env: ManagerBasedRLEnv):
        """Corridor — bookshelf + couch form a randomized gap, desk beyond."""
        for term_name in _ALL_FURNITURE:
            self._set_furniture_range(env, term_name, _CORNER_RANGE)
        self._set_furniture_range(env, "reset_furniture_bookshelf", _CORRIDOR_LEFT)
        self._set_furniture_range(env, "reset_furniture_couch",     _CORRIDOR_RIGHT)
        self._set_furniture_range(env, "reset_furniture_desk",      _CORRIDOR_BLOCKER)
        cmd_term = env.command_manager.get_term("pose_command")
        cmd_term.cfg.ranges.pos_x = (-4.0, 4.0)
        cmd_term.cfg.ranges.pos_y = (-4.0, 4.0)
        print("[CURRICULUM] Stage 3: Corridor (bookshelf + couch) + desk blocker, goals ±4m", flush=True)

    def _apply_stage4(self, env: ManagerBasedRLEnv):
        """Dense clutter — all 7 pieces in center-biased subzones."""
        for term_name, pose_range in _DENSE_CENTER.items():
            self._set_furniture_range(env, term_name, pose_range)
        cmd_term = env.command_manager.get_term("pose_command")
        cmd_term.cfg.ranges.pos_x = (-4.0, 4.0)
        cmd_term.cfg.ranges.pos_y = (-4.0, 4.0)
        print("[CURRICULUM] Stage 4: All 7 furniture in center-biased zones, goals ±4m", flush=True)

    # ── Helper ─────────────────────────────────────────────────────────────

    def _set_furniture_range(self, env: ManagerBasedRLEnv, term_name: str, pose_range: dict):
        """Update a furniture event term's pose_range."""
        term_cfg = env.event_manager.get_term_cfg(term_name)
        term_cfg.params["pose_range"] = pose_range
        env.event_manager.set_term_cfg(term_name, term_cfg)

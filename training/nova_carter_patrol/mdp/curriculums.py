"""Curriculum learning for Nova Carter Patrol.

Progressively increases navigation difficulty in 4 stages:
  Stage 1 (empty):   No furniture, close goals (±4m), center spawn.
  Stage 2 (single):  Wide table (3m×0.8m) anywhere in ±2m center zone;
                     robot outer ring (2.5–4.5m); goals ±4.5m.
  Stage 3 (scatter): table + couch + desk scattered randomly in ±3m center zone,
                     static; robot + goals both from outer ring → must navigate through.
  Stage 4 (dynamic): same 3 pieces as Stage 3 but move at 0.3 m/s and bounce off walls.

Stage transition steps are configured in CurriculumCfg params.

All pose ranges and sampling helpers live in curriculum_utils.py so that the
diagram generator (images/curriculum/generate_curriculum_diagrams.py) can import
the same values without depending on Isaac Lab.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import math
import torch

from isaaclab.managers import ManagerTermBase

from .curriculum_utils import (
    ALL_FURNITURE,
    CENTER_BLOCKER,
    CORNER_RANGE,
    DYNAMIC_PIECES,
    DYN_SPEED,
    DYN_WALL,
    RING_INNER,
    RING_OUTER,
    SCATTER_PIECES,
    SCATTER_ZONE,
    STAGE1_GOAL_XY,
    STAGE2_GOAL_XY,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class NavigationCurriculum(ManagerTermBase):
    """Progressively increase navigation difficulty over training.

    Stage 1 — empty room, center spawn, goals ±4m.
    Stage 2 — wide table in ±2m center zone, robot outer ring, goals ±4.5m.
    Stage 3 — table + couch + desk scattered randomly in ±3m center zone (static);
               robot + goals from outer ring (2.5–4.5m).
    Stage 4 — same 3 pieces as Stage 3 but kinematic-scripted motion at 0.3 m/s.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.current_stage = 0   # uninitialized
        self._dyn_vel = None     # (num_pieces, num_envs, 2) set in _apply_stage4

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        stage1_steps: int = 102400,   # ~200 iters
        stage2_steps: int = 256000,   # ~500 iters
        stage3_steps: int = 460800,   # ~900 iters
    ) -> float:
        step = env.common_step_counter
        new_stage = (4 if step >= stage3_steps else
                     3 if step >= stage2_steps else
                     2 if step >= stage1_steps else 1)

        if new_stage != self.current_stage:
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

        # Stage 4: move obstacles every RL step
        if self.current_stage == 4 and self._dyn_vel is not None:
            self._step_dynamic_obstacles(env)

        return float(self.current_stage)

    # ── Stage methods ──────────────────────────────────────────────────────

    def _apply_stage1(self, env: ManagerBasedRLEnv):
        """Empty room — all furniture to corners, center spawn, goals ±4m."""
        for term_name in ALL_FURNITURE:
            self._set_furniture_range(env, term_name, CORNER_RANGE)
        self._set_event_param(env, "reset_base", "ring_mode", False)
        self._set_event_param(env, "enforce_scatter_sep", "enabled", False)
        cmd = env.command_manager.get_term("pose_command")
        cmd.cfg.ranges.pos_x = STAGE1_GOAL_XY
        cmd.cfg.ranges.pos_y = STAGE1_GOAL_XY
        print(f"[CURRICULUM] Stage 1: empty room, center spawn, goals ±{STAGE1_GOAL_XY[1]}m", flush=True)

    def _apply_stage2(self, env: ManagerBasedRLEnv):
        """Wide table in ±2m center zone, robot outer ring, goals ±4.5m."""
        for term_name in ALL_FURNITURE:
            self._set_furniture_range(env, term_name, CORNER_RANGE)
        self._set_furniture_range(env, "reset_furniture_table", CENTER_BLOCKER)
        self._set_event_param(env, "reset_base", "ring_mode", True)
        self._set_event_param(env, "enforce_scatter_sep", "enabled", False)
        cmd = env.command_manager.get_term("pose_command")
        cmd.cfg.goal_stations = None
        cmd.cfg.ranges.pos_x = STAGE2_GOAL_XY
        cmd.cfg.ranges.pos_y = STAGE2_GOAL_XY
        print(f"[CURRICULUM] Stage 2: wide table ±2m zone, robot outer ring, goals ±{STAGE2_GOAL_XY[1]}m", flush=True)

    def _apply_stage3(self, env: ManagerBasedRLEnv):
        """3 obstacles randomly scattered in ±3m center zone (static).
        Robot + goals both from outer ring — obstacle cluster sits between them.
        """
        # All furniture to corners first, then place the 3 active pieces randomly
        for term_name in ALL_FURNITURE:
            self._set_furniture_range(env, term_name, CORNER_RANGE)
        for term_name, pose_range in SCATTER_PIECES.items():
            self._set_furniture_range(env, term_name, pose_range)

        # Robot still spawns from outer ring (consistent with Stage 2)
        self._set_event_param(env, "reset_base", "ring_mode", True)
        self._set_event_param(env, "enforce_scatter_sep", "enabled", True)

        # Goals also from outer ring — prevents goals landing inside obstacle cluster
        cmd = env.command_manager.get_term("pose_command")
        cmd.cfg.goal_stations = None
        cmd.cfg.goal_ring_mode = True   # ring-based goal sampling
        cmd.cfg.ranges.pos_x = (-(RING_OUTER), RING_OUTER)
        cmd.cfg.ranges.pos_y = (-(RING_OUTER), RING_OUTER)
        print(
            f"[CURRICULUM] Stage 3: table+couch+desk scatter ±{SCATTER_ZONE['x'][1]}m, "
            f"robot+goals outer ring {RING_INNER}–{RING_OUTER}m",
            flush=True,
        )

    def _apply_stage4(self, env: ManagerBasedRLEnv):
        """3 obstacles from same scatter zone as Stage 3 but kinematic-scripted motion."""
        for term_name in ALL_FURNITURE:
            self._set_furniture_range(env, term_name, CORNER_RANGE)
        for term_name, pose_range in SCATTER_PIECES.items():
            self._set_furniture_range(env, term_name, pose_range)

        self._set_event_param(env, "reset_base", "ring_mode", True)
        self._set_event_param(env, "enforce_scatter_sep", "enabled", True)

        # Initialise random per-env velocity directions for each piece
        num_envs = env.num_envs
        device = env.device
        angles = torch.rand(len(DYNAMIC_PIECES), num_envs, device=device) * 2 * math.pi
        self._dyn_vel = torch.stack(
            [torch.cos(angles), torch.sin(angles)], dim=-1
        ) * DYN_SPEED   # shape: (num_pieces, num_envs, 2)

        cmd = env.command_manager.get_term("pose_command")
        cmd.cfg.goal_stations = None
        cmd.cfg.goal_ring_mode = True
        cmd.cfg.ranges.pos_x = (-RING_OUTER, RING_OUTER)
        cmd.cfg.ranges.pos_y = (-RING_OUTER, RING_OUTER)
        print(
            f"[CURRICULUM] Stage 4: table+couch+desk moving at {DYN_SPEED} m/s, "
            f"robot+goals outer ring {RING_INNER}–{RING_OUTER}m",
            flush=True,
        )

    # ── Stage 4: per-step obstacle motion ─────────────────────────────────

    def _step_dynamic_obstacles(self, env: ManagerBasedRLEnv):
        """Move the 3 dynamic obstacles each RL step. Bounces off ±DYN_WALL walls.

        Called every step while current_stage == 4.  Re-randomises velocity direction
        for any env that just reset (episode_length_buf == 1).
        """
        dt = env.step_dt  # decimation * sim.dt = 5 * (1/60) ≈ 0.083 s

        # Envs that just started a new episode
        reset_ids = (env.episode_length_buf == 1).nonzero(as_tuple=False).squeeze(-1)

        for i, name in enumerate(DYNAMIC_PIECES):
            obj = env.scene[name]

            # Re-randomise direction for freshly reset envs
            if reset_ids.numel() > 0:
                angles = torch.rand(reset_ids.numel(), device=env.device) * 2 * math.pi
                self._dyn_vel[i, reset_ids] = (
                    torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1) * DYN_SPEED
                )

            # Current XY positions in env-local coords (world pos - env origin)
            # Wall check must be local — world coords are offset by env_spacing * env_id
            origins = env.scene.env_origins[:, :2]   # (num_envs, 2)
            xy_world = obj.data.root_pos_w[:, :2].clone()
            xy = xy_world - origins

            # Flip velocity component for any axis that hit the boundary
            past_wall = xy.abs() >= DYN_WALL
            self._dyn_vel[i] = torch.where(past_wall, -self._dyn_vel[i], self._dyn_vel[i])

            # Step forward and hard-clamp as safety net
            xy = (xy + self._dyn_vel[i] * dt).clamp(-DYN_WALL, DYN_WALL)

            # Write back: update only XY, preserve Z and orientation
            new_pose = obj.data.root_state_w[:, :7].clone()
            new_pose[:, :2] = xy + origins   # convert back to world space
            obj.write_root_pose_to_sim(new_pose)

    # ── Helpers ────────────────────────────────────────────────────────────

    def _set_furniture_range(self, env: ManagerBasedRLEnv, term_name: str, pose_range: dict):
        """Update a furniture event term's pose_range."""
        term_cfg = env.event_manager.get_term_cfg(term_name)
        term_cfg.params["pose_range"] = pose_range
        env.event_manager.set_term_cfg(term_name, term_cfg)

    def _set_event_param(self, env: ManagerBasedRLEnv, term_name: str, key: str, value) -> None:
        """Update a single param on any event term."""
        term_cfg = env.event_manager.get_term_cfg(term_name)
        term_cfg.params[key] = value
        env.event_manager.set_term_cfg(term_name, term_cfg)

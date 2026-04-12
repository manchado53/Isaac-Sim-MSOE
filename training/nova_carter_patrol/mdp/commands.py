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
    "furniture_table":     1.377,  # sqrt(1.0^2 + 0.4^2)    + 0.3  (table 2.0m x 0.8m)
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
        # Per-env stuck counter — increments each step the robot is slow + past grace period.
        # Resets to 0 whenever the robot moves or re-enters the grace period after episode reset.
        self._stuck_counter = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)
        # Diagnostic: counts stuck-triggered resamples per env per episode
        self.metrics["stuck_resamples"] = torch.zeros(self.num_envs, device=self.device)

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

    def _assign_station_goals(self, env_ids: torch.Tensor):
        """Assign goals by picking a random station from cfg.goal_stations for each env."""
        stations = torch.tensor(self.cfg.goal_stations, device=self.device, dtype=torch.float32)
        jitter = self.cfg.goal_station_jitter
        idx = torch.randint(0, len(stations), (len(env_ids),), device=self.device)
        chosen = stations[idx] + torch.empty(len(env_ids), 2, device=self.device).uniform_(-jitter, jitter)
        origins = self._env.scene.env_origins[env_ids]
        self.pos_command_w[env_ids, 0] = origins[:, 0] + chosen[:, 0]
        self.pos_command_w[env_ids, 1] = origins[:, 1] + chosen[:, 1]

    def _assign_ring_goals(self, env_ids: torch.Tensor):
        """Sample goals from one of 4 outer ring strips (N/S/E/W) per env.

        Mirrors the strip logic in events.reset_base_outer_ring. Each env is
        randomly assigned to one strip and sampled uniformly within it.
        """
        n = len(env_ids)
        inner = self.cfg.goal_ring_inner
        outer = self.cfg.goal_ring_outer

        strip = torch.randint(4, (n,), device=self.device)  # 0=N, 1=S, 2=E, 3=W
        rx = torch.empty(n, 4, device=self.device)
        ry = torch.empty(n, 4, device=self.device)
        rx[:, 0].uniform_(-outer,  outer);  ry[:, 0].uniform_( inner,  outer)   # N
        rx[:, 1].uniform_(-outer,  outer);  ry[:, 1].uniform_(-outer, -inner)   # S
        rx[:, 2].uniform_( inner,  outer);  ry[:, 2].uniform_(-inner,  inner)   # E
        rx[:, 3].uniform_(-outer, -inner);  ry[:, 3].uniform_(-inner,  inner)   # W

        arange = torch.arange(n, device=self.device)
        origins = self._env.scene.env_origins[env_ids]
        self.pos_command_w[env_ids, 0] = origins[:, 0] + rx[arange, strip]
        self.pos_command_w[env_ids, 1] = origins[:, 1] + ry[arange, strip]

    def _resample_command(self, env_ids: Sequence[int]):
        """Sample goals, then iteratively resample any that collide with furniture."""
        super()._resample_command(env_ids)  # sets pos_command_w + heading_command_w

        remaining = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if len(remaining) == 0:
            return

        if self.cfg.goal_stations is not None:
            # Station mode: pick one of the 4 fixed positions per env, apply jitter, reject collisions
            self._assign_station_goals(remaining)
            for _ in range(_MAX_REJECTION_ATTEMPTS):
                bad_mask = self._is_in_collision(self.pos_command_w[remaining, :2], remaining)
                if not bad_mask.any():
                    break
                remaining = remaining[bad_mask]
                self._assign_station_goals(remaining)
        elif self.cfg.goal_ring_mode:
            # Ring mode: goals from outer ring strips — keeps goals outside the obstacle scatter zone
            self._assign_ring_goals(remaining)
            for _ in range(_MAX_REJECTION_ATTEMPTS):
                bad_mask = self._is_in_collision(self.pos_command_w[remaining, :2], remaining)
                if not bad_mask.any():
                    break
                remaining = remaining[bad_mask]
                self._assign_ring_goals(remaining)
        else:
            # Uniform range mode: iteratively reject goals that land inside furniture
            for attempt in range(_MAX_REJECTION_ATTEMPTS):
                bad_mask = self._is_in_collision(self.pos_command_w[remaining, :2], remaining)
                if not bad_mask.any():
                    break
                print(f"[GOAL REJECTION] attempt {attempt + 1}: {bad_mask.sum().item()} goal(s) inside furniture, resampling", flush=True)
                remaining = remaining[bad_mask]
                origins = self._env.scene.env_origins[remaining]
                r = torch.empty(len(remaining), device=self.device)
                self.pos_command_w[remaining, 0] = origins[:, 0] + r.uniform_(*self.cfg.ranges.pos_x)
                self.pos_command_w[remaining, 1] = origins[:, 1] + r.uniform_(*self.cfg.ranges.pos_y)
        # Envs still colliding after max attempts keep their last sample.
        # The 15s fallback timer remains the safety net for genuinely unreachable goals.

    def _update_command(self):
        """Resample goals for envs that are stuck or have reached the goal, then update."""

        # ── 1. Stuck detection ────────────────────────────────────────────────
        # Increment per-env counter each step the robot's ground-plane speed stays below
        # stuck_speed_threshold AND the episode is past the grace period (avoids triggering
        # on the zero-velocity frames immediately after episode reset).
        # Counter resets to 0 whenever the robot moves OR re-enters the grace period.
        speed = torch.norm(self._env.scene["robot"].data.root_lin_vel_b[:, :2], dim=-1)
        is_slow = speed < self.cfg.stuck_speed_threshold
        past_grace = self._env.episode_length_buf >= self.cfg.stuck_grace_period

        self._stuck_counter = torch.where(
            is_slow & past_grace,
            self._stuck_counter + 1,
            torch.zeros_like(self._stuck_counter),
        )

        stuck_mask = self._stuck_counter >= self.cfg.stuck_min_steps
        stuck_ids = stuck_mask.nonzero(as_tuple=False).squeeze(-1)
        if len(stuck_ids) > 0:
            self.metrics["stuck_resamples"][stuck_ids] += 1
            self._stuck_counter[stuck_ids] = 0  # reset before resample
            print(
                f"[STUCK] {len(stuck_ids)} env(s) stuck >{self.cfg.stuck_min_steps} steps,"
                f" resampling goals",
                flush=True,
            )
            self._resample(stuck_ids)  # increments command_counter, resets fallback timer

        # ── 2. Goal-reached detection ─────────────────────────────────────────
        # Use current world-frame distance (same source as the goal_reached reward) so
        # detection fires on the same frame as the reward — not one frame later via the
        # stale error_pos_2d metric. This ensures the goal resamples before the next
        # reward step, so the +100 bonus fires exactly once per goal reach.
        # Note: pos_command_w for non-stuck envs is still the OLD goal at this point
        # (stuck resampling above only affects stuck_mask envs, excluded below).
        current_dist = torch.norm(
            self.pos_command_w[:, :2] - self._env.scene["robot"].data.root_pos_w[:, :2],
            dim=1,
        )
        goal_reached = current_dist < self.cfg.goal_reached_threshold
        goal_reached = goal_reached & ~stuck_mask
        goal_reset_ids = goal_reached.nonzero(as_tuple=False).squeeze(-1)
        if len(goal_reset_ids) > 0:
            self.metrics["consecutive_success"][goal_reset_ids] += 1
            self._resample(goal_reset_ids)

        # ── 3. Parent updates body-frame command from world-frame targets ──────
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

    goal_stations: list | None = None
    """List of [x, y] goal positions for station-based sampling (e.g. Stage 2).
    One station is chosen at random per resample with ±goal_station_jitter noise.
    If None, uses uniform range sampling."""

    goal_ring_mode: bool = False
    """If True, goals are sampled from 4 outer ring strips (N/S/E/W) between
    goal_ring_inner and goal_ring_outer, instead of the rectangular pos_x/pos_y ranges.
    Set to True by the curriculum in Stages 3+4 to keep goals outside the obstacle cluster."""

    goal_ring_inner: float = 2.5
    """Inner radius of the goal ring (m from env centre). Matches RING_INNER."""

    goal_ring_outer: float = 4.5
    """Outer radius of the goal ring (m from env centre). Matches RING_OUTER."""

    goal_station_jitter: float = 0.3
    """Uniform jitter (±m) applied to each sampled station position."""

    stuck_speed_threshold: float = 0.05
    """Ground-plane speed (m/s) below which the robot is considered stuck.
    0.05 m/s = 10% of Nova Carter's 0.5 m/s max speed."""

    stuck_min_steps: int = 36
    """Consecutive steps below stuck_speed_threshold before triggering a goal resample.
    36 steps = 3 seconds at 12 Hz control frequency."""

    stuck_grace_period: int = 36
    """Steps at the start of each episode exempt from stuck detection.
    Prevents triggering on the zero-velocity frames immediately after episode reset.
    36 steps = 3 seconds at 12 Hz control frequency."""

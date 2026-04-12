"""Custom event functions for Nova Carter Patrol."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def reset_base_outer_ring(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    ring_mode: bool,
    inner_radius: float,
    outer_radius: float,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the robot root state, switching between ring-spawn and rectangular-spawn modes.

    ring_mode=True  — robot spawns in one of 4 outer strips (N/S/E/W) between
                      inner_radius and outer_radius from the env center.
    ring_mode=False — falls back to normal uniform rectangular sampling using pose_range,
                      identical to reset_root_state_uniform.
    """
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    root_states = asset.data.default_root_state[env_ids].clone()
    origins = env.scene.env_origins[env_ids]
    n = len(env_ids)

    if ring_mode and n > 0:
        # Randomly assign each env to one of 4 strips: 0=N, 1=S, 2=E, 3=W
        strip = torch.randint(4, (n,), device=env.device)

        # Pre-sample all 4 strips simultaneously (n × 4), then select
        rx = torch.empty(n, 4, device=env.device)
        ry = torch.empty(n, 4, device=env.device)
        rx[:, 0].uniform_(-outer_radius, outer_radius);  ry[:, 0].uniform_(inner_radius, outer_radius)    # N
        rx[:, 1].uniform_(-outer_radius, outer_radius);  ry[:, 1].uniform_(-outer_radius, -inner_radius)  # S
        rx[:, 2].uniform_(inner_radius,  outer_radius);  ry[:, 2].uniform_(-inner_radius, inner_radius)   # E
        rx[:, 3].uniform_(-outer_radius, -inner_radius); ry[:, 3].uniform_(-inner_radius, inner_radius)   # W

        arange = torch.arange(n, device=env.device)
        positions = root_states[:, :3].clone()
        positions[:, 0] = origins[:, 0] + rx[arange, strip]
        positions[:, 1] = origins[:, 1] + ry[arange, strip]
        positions[:, 2] = origins[:, 2] + root_states[:, 2]

        yaw = torch.empty(n, device=env.device).uniform_(-math.pi, math.pi)
        orientations = math_utils.quat_from_euler_xyz(
            torch.zeros_like(yaw), torch.zeros_like(yaw), yaw
        )
        velocities = torch.zeros(n, 6, device=env.device)
    else:
        # Rectangular sampling — identical to reset_root_state_uniform
        range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=asset.device)
        rand_samples = math_utils.sample_uniform(
            ranges[:, 0], ranges[:, 1], (n, 6), device=asset.device
        )
        positions = root_states[:, 0:3] + origins + rand_samples[:, 0:3]
        orientations_delta = math_utils.quat_from_euler_xyz(
            rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5]
        )
        orientations = math_utils.quat_mul(root_states[:, 3:7], orientations_delta)

        range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=asset.device)
        rand_samples = math_utils.sample_uniform(
            ranges[:, 0], ranges[:, 1], (n, 6), device=asset.device
        )
        velocities = root_states[:, 7:13] + rand_samples

    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)


def enforce_scatter_separation(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    piece_names: list[str],
    scatter_zone: dict[str, tuple[float, float]],
    min_separation: float,
    max_tries: int = 150,
    enabled: bool = False,
):
    """Resample scatter pieces that are too close to each other after individual resets.

    Runs after the individual reset_furniture_* event terms.  For each pair of pieces
    whose centre-to-centre distance is below min_separation, the second piece in the
    pair is resampled until all pairs are separated or max_tries is exhausted.

    Disabled in Stages 1+2 via enabled=False (pieces overlap in the corner intentionally).
    The curriculum sets enabled=True when entering Stage 3+4.
    """
    if not enabled or len(env_ids) == 0:
        return

    n = len(env_ids)
    device = env.device
    origins = env.scene.env_origins[env_ids]   # (n, 3)

    # Helper: sample new XY + yaw for a batch of n envs from scatter_zone
    def _resample_xy(mask: torch.Tensor):
        """Sample new positions for envs where mask is True."""
        count = int(mask.sum().item())
        if count == 0:
            return
        xlo, xhi = scatter_zone["x"]
        ylo, yhi = scatter_zone["y"]
        yaw_lo, yaw_hi = scatter_zone["yaw"]
        new_x   = torch.empty(count, device=device).uniform_(xlo, xhi)
        new_y   = torch.empty(count, device=device).uniform_(ylo, yhi)
        new_yaw = torch.empty(count, device=device).uniform_(yaw_lo, yaw_hi)
        return new_x, new_y, new_yaw

    # Gather assets
    assets = [env.scene[name] for name in piece_names]
    n_pieces = len(assets)

    # Get current positions relative to env origin — shape (n_pieces, n, 2)
    def _get_rel_xy():
        return torch.stack([
            assets[i].data.root_pos_w[env_ids, :2] - origins[:, :2]
            for i in range(n_pieces)
        ])  # (n_pieces, n, 2)

    def _write_pose(asset_idx: int, mask: torch.Tensor, rel_x, rel_y, yaw):
        """Write new XY pose for envs where mask is True."""
        obj = assets[asset_idx]
        masked_ids = env_ids[mask]
        if masked_ids.numel() == 0:
            return
        root_states = obj.data.default_root_state[masked_ids].clone()
        orig = env.scene.env_origins[masked_ids]
        positions = root_states[:, :3].clone()
        positions[:, 0] = orig[:, 0] + rel_x
        positions[:, 1] = orig[:, 1] + rel_y
        positions[:, 2] = orig[:, 2] + root_states[:, 2]
        orientations = math_utils.quat_from_euler_xyz(
            torch.zeros_like(yaw), torch.zeros_like(yaw), yaw
        )
        obj.write_root_pose_to_sim(
            torch.cat([positions, orientations], dim=-1), env_ids=masked_ids
        )

    for _ in range(max_tries):
        rel_xy = _get_rel_xy()   # (n_pieces, n, 2)

        # Check all pairs; collect envs that need resampling per piece
        needs_resample = torch.zeros(n_pieces, n, dtype=torch.bool, device=device)
        for i in range(n_pieces):
            for j in range(i + 1, n_pieces):
                dist = (rel_xy[i] - rel_xy[j]).norm(dim=-1)  # (n,)
                too_close = dist < min_separation
                # Resample the later piece (j) when pair is too close
                needs_resample[j] |= too_close

        if not needs_resample.any():
            break

        # Resample pieces that need it
        for i in range(n_pieces):
            mask = needs_resample[i]
            if not mask.any():
                continue
            result = _resample_xy(mask)
            if result is None:
                continue
            new_x, new_y, new_yaw = result
            _write_pose(i, mask, new_x, new_y, new_yaw)

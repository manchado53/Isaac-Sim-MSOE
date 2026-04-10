"""Shared curriculum constants and sampling utilities.

Pure Python — zero Isaac Lab dependency. Importable from both:
  - curriculums.py          (training logic, runs inside Isaac Sim)
  - images/curriculum/generate_curriculum_diagrams.py  (visualization, plain Python)

If you change a range here it automatically updates both training and diagrams.
Keep FURNITURE_DIMS in sync with USD asset scales in nova_carter_patrol_env_cfg.py.
"""

from __future__ import annotations

import math

import numpy as np

# ── Robot + Goal ring spawn ────────────────────────────────────────────────
RING_INNER = 2.5   # inner edge of outer ring (m from env centre)
RING_OUTER = 4.5   # outer edge of outer ring (m from env centre)

# ── Furniture placement pose ranges ───────────────────────────────────────
# All furniture parked out of the way in corners
CORNER_RANGE: dict = {"x": (4.0, 4.5), "y": (4.0, 4.5), "yaw": (0.0, 0.0)}

# Stage 2: wide table anywhere in ±2m center zone, random yaw
CENTER_BLOCKER: dict = {"x": (-2.0, 2.0), "y": (-2.0, 2.0), "yaw": (-math.pi, math.pi)}

# Stages 3 & 4: all 3 pieces scatter randomly in ±3m center zone
SCATTER_ZONE: dict = {"x": (-3.0, 3.0), "y": (-3.0, 3.0), "yaw": (-math.pi, math.pi)}
SCATTER_ZONE_HALF: float = 3.0   # half-size of the scatter zone (m), for diagram shading
SCATTER_MIN_SEP:  float = 2.0   # minimum centre-to-centre distance between scatter pieces (m)

# ── Stage 3+4 obstacle set ─────────────────────────────────────────────────
# Maps event term name → pose range used at reset
SCATTER_PIECES: dict = {
    "reset_furniture_table": SCATTER_ZONE,
    "reset_furniture_couch": SCATTER_ZONE,
    "reset_furniture_desk":  SCATTER_ZONE,
}

# Scene asset names for the 3 dynamic pieces (Stage 4 motion)
DYNAMIC_PIECES: list[str] = ["furniture_table", "furniture_couch", "furniture_desk"]
DYN_SPEED: float = 0.3   # m/s — obstacle travel speed
DYN_WALL:  float = 4.0   # bounce boundary (m) — obstacles reflect off ±4 m walls

# ── All furniture event term names (for bulk-cornering) ────────────────────
ALL_FURNITURE: list[str] = [
    "reset_furniture_table",
    "reset_furniture_bookshelf",
    "reset_furniture_chair",
    "reset_furniture_desk",
    "reset_furniture_cabinet",
    "reset_furniture_couch",
    "reset_furniture_bin",
]

# ── Goal ranges per stage ──────────────────────────────────────────────────
STAGE1_GOAL_XY: tuple = (-4.0, 4.0)   # uniform rectangular ±4 m
STAGE2_GOAL_XY: tuple = (-4.5, 4.5)   # uniform rectangular ±4.5 m
# Stages 3+4: ring-based — use sample_ring(RING_INNER, RING_OUTER)

# ── Furniture physical dimensions (length × width, metres) ─────────────────
# Mirrors USD asset scales in nova_carter_patrol_env_cfg.py
FURNITURE_DIMS: dict = {
    "table":     (2.0, 0.8),
    "bookshelf": (0.4, 1.2),
    "chair":     (0.45, 0.45),
    "desk":      (1.5, 0.7),
    "cabinet":   (0.5, 0.4),
    "couch":     (1.8, 0.7),
    "bin":       (0.3, 0.3),
}

# ── Sampling functions ─────────────────────────────────────────────────────

def sample_pose(pose_range: dict) -> tuple[float, float, float]:
    """Uniform sample from a pose range dict — mirrors Isaac Lab event sampling.

    Returns (x, y, yaw).
    """
    x   = float(np.random.uniform(*pose_range["x"]))
    y   = float(np.random.uniform(*pose_range["y"]))
    yaw = float(np.random.uniform(*pose_range["yaw"]))
    return x, y, yaw


def sample_ring(inner: float = RING_INNER, outer: float = RING_OUTER) -> tuple[float, float]:
    """Sample an XY position from one of 4 outer ring strips (N / S / E / W).

    Mirrors the ring-spawn logic in events.reset_base_outer_ring.
    Returns (x, y).
    """
    strip = np.random.randint(4)
    if strip == 0:   # North
        return float(np.random.uniform(-outer, outer)), float(np.random.uniform(inner, outer))
    if strip == 1:   # South
        return float(np.random.uniform(-outer, outer)), float(np.random.uniform(-outer, -inner))
    if strip == 2:   # East
        return float(np.random.uniform(inner,  outer)), float(np.random.uniform(-inner, inner))
    # West
    return float(np.random.uniform(-outer, -inner)), float(np.random.uniform(-inner, inner))

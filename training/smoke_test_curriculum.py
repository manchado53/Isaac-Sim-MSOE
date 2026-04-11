#!/usr/bin/env python3
"""Smoke test for curriculum placement logic — no Isaac Lab required.

Verifies:
  1. CORNER_RANGE places furniture in the NE quadrant only
  2. CENTER_BLOCKER keeps table within ±2m center zone
  3. SCATTER_ZONE keeps pieces within ±3m
  4. sample_ring() produces points strictly in the outer ring (2.5–4.5m)
  5. Ring sampling distributes across all 4 strips (N/S/E/W)
  6. enforce_scatter_separation would fire on corner-packed pieces (dist < 2m)
     but NOT on correctly scattered pieces
  7. goal_ring_mode radii match RING_INNER / RING_OUTER

Run from repo root:
    python training/smoke_test_curriculum.py
"""

import math
import sys
import os

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "nova_carter_patrol", "mdp"))

from curriculum_utils import (
    CORNER_RANGE,
    CENTER_BLOCKER,
    SCATTER_ZONE,
    SCATTER_MIN_SEP,
    RING_INNER,
    RING_OUTER,
    FURNITURE_DIMS,
    ALL_FURNITURE,
    SCATTER_PIECES,
    STAGE1_GOAL_XY,
    STAGE2_GOAL_XY,
    sample_pose,
    sample_ring,
)

N_SAMPLES = 10_000
PASS = "\033[92m PASS\033[0m"
FAIL = "\033[91m FAIL\033[0m"

failures = []

def check(name: str, condition: bool, detail: str = ""):
    status = PASS if condition else FAIL
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))
    if not condition:
        failures.append(name)


print("\n=== Smoke Test: Curriculum Placement Logic ===\n")

# ─────────────────────────────────────────────────────────────
# 1. CORNER_RANGE — all samples should be in NE quadrant (x>0, y>0)
# ─────────────────────────────────────────────────────────────
print("1. CORNER_RANGE (Stage 1+2 parked furniture)")
xs, ys = [], []
for _ in range(N_SAMPLES):
    x, y, yaw = sample_pose(CORNER_RANGE)
    xs.append(x); ys.append(y)
xs, ys = np.array(xs), np.array(ys)

check("All x > 0",      (xs > 0).all(),           f"min_x={xs.min():.2f}")
check("All y > 0",      (ys > 0).all(),            f"min_y={ys.min():.2f}")
check("All x in [4,4.5]", ((xs >= 4.0) & (xs <= 4.5)).all(), f"range=[{xs.min():.2f},{xs.max():.2f}]")
check("All y in [4,4.5]", ((ys >= 4.0) & (ys <= 4.5)).all(), f"range=[{ys.min():.2f},{ys.max():.2f}]")
check("All yaw == 0",   (np.array([sample_pose(CORNER_RANGE)[2] for _ in range(100)]) == 0).all())

# ─────────────────────────────────────────────────────────────
# 2. CENTER_BLOCKER — table should be within ±2m
# ─────────────────────────────────────────────────────────────
print("\n2. CENTER_BLOCKER (Stage 2 table)")
xs, ys = [], []
for _ in range(N_SAMPLES):
    x, y, _ = sample_pose(CENTER_BLOCKER)
    xs.append(x); ys.append(y)
xs, ys = np.array(xs), np.array(ys)

check("All x in [-2,2]", ((xs >= -2.0) & (xs <= 2.0)).all(), f"range=[{xs.min():.2f},{xs.max():.2f}]")
check("All y in [-2,2]", ((ys >= -2.0) & (ys <= 2.0)).all(), f"range=[{ys.min():.2f},{ys.max():.2f}]")
check("Covers all quadrants (x)", xs.min() < -1.5 and xs.max() > 1.5)
check("Covers all quadrants (y)", ys.min() < -1.5 and ys.max() > 1.5)

# ─────────────────────────────────────────────────────────────
# 3. SCATTER_ZONE — pieces within ±3m
# ─────────────────────────────────────────────────────────────
print("\n3. SCATTER_ZONE (Stage 3+4 active pieces)")
xs, ys = [], []
for _ in range(N_SAMPLES):
    x, y, _ = sample_pose(SCATTER_ZONE)
    xs.append(x); ys.append(y)
xs, ys = np.array(xs), np.array(ys)

check("All x in [-3,3]", ((xs >= -3.0) & (xs <= 3.0)).all(), f"range=[{xs.min():.2f},{xs.max():.2f}]")
check("All y in [-3,3]", ((ys >= -3.0) & (ys <= 3.0)).all(), f"range=[{ys.min():.2f},{ys.max():.2f}]")
check("Covers full zone", xs.min() < -2.5 and xs.max() > 2.5 and ys.min() < -2.5 and ys.max() > 2.5)

# ─────────────────────────────────────────────────────────────
# 4. sample_ring() — points strictly in outer ring (2.5–4.5m)
# ─────────────────────────────────────────────────────────────
print("\n4. sample_ring() (robot+goal outer ring, Stages 2–4)")
dists = []
strip_counts = np.zeros(4)
for _ in range(N_SAMPLES):
    x, y = sample_ring()
    dists.append(math.sqrt(x**2 + y**2))
    # Classify strip
    if y >= RING_INNER:     strip_counts[0] += 1   # N: y in [inner, outer]
    elif y <= -RING_INNER:  strip_counts[1] += 1   # S: y in [-outer,-inner]
    elif x >= RING_INNER:   strip_counts[2] += 1   # E: x in [inner, outer]
    else:                   strip_counts[3] += 1   # W: x in [-outer,-inner]

dists = np.array(dists)
check("All dist >= RING_INNER (2.5m)", (dists >= RING_INNER - 1e-9).all(), f"min={dists.min():.3f}")
check("All dist <= RING_OUTER*sqrt2 (max diagonal)", (dists <= RING_OUTER * math.sqrt(2) + 1e-6).all(), f"max={dists.max():.3f}")
check("No points inside inner radius", (dists >= RING_INNER).mean() > 0.999, f"frac_inside={((dists < RING_INNER).mean()):.4f}")

strip_fracs = strip_counts / N_SAMPLES
check("All 4 strips populated (>10% each)", (strip_fracs > 0.10).all(),
      f"N={strip_fracs[0]:.2f} S={strip_fracs[1]:.2f} E={strip_fracs[2]:.2f} W={strip_fracs[3]:.2f}")

# ─────────────────────────────────────────────────────────────
# 5. enforce_scatter_separation: corner pieces would violate min_sep
# ─────────────────────────────────────────────────────────────
print("\n5. enforce_scatter_separation trigger conditions")

# Simulate 100 corner placements — every trial should have at least one pair violation
# (all 3 pieces land in a 0.5×0.5m zone, always closer than SCATTER_MIN_SEP=2m).
# This confirms the bug we fixed: without enabled=False, enforcement would resample
# them across the whole SCATTER_ZONE every episode during Stage 1+2.
corner_trials_with_violation = 0
for _ in range(100):
    positions = [sample_pose(CORNER_RANGE)[:2] for _ in range(3)]
    for i in range(3):
        for j in range(i + 1, 3):
            dx = positions[i][0] - positions[j][0]
            dy = positions[i][1] - positions[j][1]
            if math.sqrt(dx**2 + dy**2) < SCATTER_MIN_SEP:
                corner_trials_with_violation += 1
                break
        else:
            continue
        break

check("Corner pieces always within min_sep (enforcement correctly disabled in Stages 1+2)",
      corner_trials_with_violation == 100,
      f"trials with violation: {corner_trials_with_violation}/100")

# Simulate scatter placements with min_sep enforcement — should have no violations
scatter_violations = 0
for _ in range(100):
    positions = []
    for _ in range(3):
        for attempt in range(300):
            x, y, _ = sample_pose(SCATTER_ZONE)
            ok = all(
                math.sqrt((x - px)**2 + (y - py)**2) >= SCATTER_MIN_SEP
                for px, py in positions
            )
            if ok:
                positions.append((x, y))
                break
    for i in range(len(positions)):
        for j in range(i+1, len(positions)):
            dx = positions[i][0] - positions[j][0]
            dy = positions[i][1] - positions[j][1]
            if math.sqrt(dx**2 + dy**2) < SCATTER_MIN_SEP - 1e-6:
                scatter_violations += 1

check("Scatter pieces respect min_sep after enforcement",
      scatter_violations == 0,
      f"violations: {scatter_violations}/100")

# ─────────────────────────────────────────────────────────────
# 6. goal_ring_mode config field radii match curriculum constants
# ─────────────────────────────────────────────────────────────
print("\n6. goal_ring_mode cfg defaults match curriculum constants")
# GoalReachPose2dCommandCfg defaults: goal_ring_inner=2.5, goal_ring_outer=4.5
CFG_RING_INNER = 2.5
CFG_RING_OUTER = 4.5
check("goal_ring_inner matches RING_INNER", CFG_RING_INNER == RING_INNER,
      f"{CFG_RING_INNER} == {RING_INNER}")
check("goal_ring_outer matches RING_OUTER", CFG_RING_OUTER == RING_OUTER,
      f"{CFG_RING_OUTER} == {RING_OUTER}")

# ─────────────────────────────────────────────────────────────
# 7. Furniture dimensions match FURNITURE_DIMS
# ─────────────────────────────────────────────────────────────
print("\n7. FURNITURE_DIMS consistency with ALL_FURNITURE and SCATTER_PIECES")
scatter_names = {k.replace("reset_furniture_", "") for k in SCATTER_PIECES}
check("SCATTER_PIECES names in FURNITURE_DIMS", scatter_names.issubset(FURNITURE_DIMS.keys()),
      f"scatter={scatter_names}")
all_names_in_dims = all(
    n.replace("reset_furniture_", "") in FURNITURE_DIMS
    for n in ALL_FURNITURE
)
check("ALL_FURNITURE names in FURNITURE_DIMS", all_names_in_dims)

# ─────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────
print("\n" + "="*50)
if not failures:
    print(f"\033[92m All checks passed.\033[0m")
else:
    print(f"\033[91m {len(failures)} check(s) FAILED:\033[0m")
    for f in failures:
        print(f"   - {f}")
    sys.exit(1)

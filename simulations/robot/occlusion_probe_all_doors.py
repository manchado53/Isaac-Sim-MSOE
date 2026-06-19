#!/usr/bin/env python3
"""
Occlusion probe — no-capture dry run across all 8 training doors.

Purpose
-------
Before committing to production data collection we need a measured estimate
of how many candidate camera positions actually survive the PhysX raycast
occlusion filter on each door. The production collection budget is driven
almost entirely by this number; the rough 20 cameras/door assumption used
in the Task-1 arithmetic needs to be replaced with a real count.

What this does
--------------
For each of the 8 TRAINING doors from SELECTED_TRAINING_DOORS.md:

  1.  Concretize the door prim (SetInstanceable(False) for instanced families).
  2.  Generate the full candidate camera grid proposed for production:
          arc      = [-60, -30, 0, +30, +60]   (5)
          distance = [1.5, 2.0, 3.5]           (3)
          height   = [-0.5, 0.0, +1.0]         (3)
          → 45 candidates per door
      centered on the door's `door_center_m` (XY) at the door's camera Z +
      the height offset, aiming the look vector back at the door centre.
  3.  Run the same PhysX raycast filter that door_replicator_collection.py
      uses (raycast from camera origin toward door centre; accept iff the
      first rigid-body hit path contains the door's prim path, or no hit).
  4.  Dump a JSON summary with per-door survivor counts broken down by
      arc/distance/height, and a global mean/min/max.

What this DOES NOT do
---------------------
* No rendering, no frame capture, no material/lighting variation. This is a
  pure geometry probe — the renderer is only used enough to settle the
  stage for the PhysX raycast scene.
* No holdout doors — those will be probed separately once we know the
  training survival rate and agree on the final camera grid.
* No scene writes. The probe is read-only.

Output
------
  data/occlusion_probe_summary.json

Run
---
  sbatch submit_isaac_job.sh simulations/robot/occlusion_probe_all_doors.py
"""

import os
import sys
import math
import json
import functools
import numpy as np

print = functools.partial(print, flush=True)  # noqa: A001

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CAM_RESOLUTION = (1024, 768)
STARTUP_WARMUP = 10
DOOR_WARMUP    = 5

CAM_ARC_OFFSETS = [-60.0, -30.0, 0.0, +30.0, +60.0]
CAM_DISTANCES   = [1.5, 2.0, 3.5]
CAM_HEIGHT_OFFS = [-0.5, 0.0, +1.0]

_script_dir  = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.abspath(os.path.join(_script_dir, "..", ".."))
USD_PATH     = os.path.join(_project_dir, "MSOEBuildingV4_physics.usd")
OUT_PATH     = os.path.join(_project_dir, "data", "occlusion_probe_summary.json")

# ---------------------------------------------------------------------------
# Door registry — 8 training doors, copied verbatim from
# lighting_test_all_doors.py (source of truth per SELECTED_TRAINING_DOORS.md).
# ---------------------------------------------------------------------------

_BASE = (
    "/World/Building/_22_129___MSOE_Engineering_Building___A___2_25_2026"
    "/Geometry/Doors"
)

DOORS = [
    {
        "id":             "Doors_13606466",
        "prim_path":      f"{_BASE}/RS___Curtain_Wall___Type_F___NEW/New/Doors_13606466",
        "family_short":   "CW_F_NEW",
        "floor":          4,
        "cam_yaw_deg":    180.0,
        "door_center_m":  [-48.360, 62.163, 23.704],
    },
    {
        "id":             "Doors_10649156",
        "prim_path":      f"{_BASE}/RS___Type_FG_Door/NEW___HM_Frame___36_x96/Doors_10649156",
        "family_short":   "Type_FG",
        "floor":          3,
        "cam_yaw_deg":    0.0,
        "door_center_m":  [-12.295, 80.200, 19.437],
    },
    {
        "id":             "Doors_8286057",
        "prim_path":      f"{_BASE}/RS___Type_F_Door/NEW___HM_Frame___36_x96/Doors_8286057",
        "family_short":   "Type_F",
        "floor":          2,
        "cam_yaw_deg":    270.0,
        "door_center_m":  [17.443, 77.341, 14.764],
    },
    {
        "id":             "Doors_3135827",
        "prim_path":      f"{_BASE}/RS___Curtain_Wall___Type_FG___New/New/Doors_3135827",
        "family_short":   "CW_FG_New",
        "floor":          3,
        "cam_yaw_deg":    90.0,
        "door_center_m":  [17.701, 58.807, 19.437],
    },
    {
        "id":             "Doors_7171063",
        "prim_path":      f"{_BASE}/RS___Type_F_Door/NEW___HM_Frame___42_x96/Doors_7171063",
        "family_short":   "Type_F",
        "floor":          5,
        "cam_yaw_deg":    0.0,
        "door_center_m":  [-47.316, 49.892, 27.972],
    },
    {
        "id":             "Doors_12590968",
        "prim_path":      f"{_BASE}/RS___Type_N_Door/NEW___HM_Frame___36_x96/Doors_12590968",
        "family_short":   "Type_N",
        "floor":          3,
        "cam_yaw_deg":    90.0,
        "door_center_m":  [17.878, 52.941, 19.437],
    },
    {
        "id":             "Doors_12302795",
        "prim_path":      f"{_BASE}/RS___Curtain_Wall___Type_FG___New/New___GL1_1/Doors_12302795",
        "family_short":   "CW_FG_New",
        "floor":          2,
        "cam_yaw_deg":    180.0,
        "door_center_m":  [-59.258, 58.251, 14.561],
    },
    {
        "id":             "Doors_13606056",
        "prim_path":      f"{_BASE}/RS___Curtain_Wall___Type_F___NEW/New/Doors_13606056",
        "family_short":   "CW_F_NEW",
        "floor":          3,
        "cam_yaw_deg":    180.0,
        "door_center_m":  [-47.477, 70.040, 19.437],
    },
]

# ---------------------------------------------------------------------------
# Isaac Sim startup
# ---------------------------------------------------------------------------

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

import time as _time
_t0 = _time.time()
def _ts(msg):
    print(f"[OcclProbe] [+{_time.time() - _t0:6.1f}s] {msg}")

_ts("Importing SimulationApp ...")
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})
_ts("SimulationApp ready")

import carb
_s = carb.settings.get_settings()
_s.set("/rtx/post/aa/op", 4)
_s.set("/rtx/post/dlss/execMode", 0)

import omni.usd
from pxr import Usd, UsdGeom, UsdLux, Gf, Sdf
from isaacsim.core.api import World

# ---------------------------------------------------------------------------
# Scene load
# ---------------------------------------------------------------------------

world = World(stage_units_in_meters=1.0)
stage = omni.usd.get_context().get_stage()

_ts(f"Inserting building USD sublayer: {USD_PATH}")
stage.GetRootLayer().subLayerPaths.insert(0, USD_PATH)

# Minimal dome so the scene isn't pitch black during physics settle.
dome = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/DomeLight"))
dome.CreateIntensityAttr(25_000)

_ts("world.reset() ...")
world.reset()
_ts("world.reset() done")
for _ in range(STARTUP_WARMUP):
    world.step(render=True)

# PhysX scene-query interface — same path door_replicator_collection.py uses.
try:
    from omni.physx import get_physx_scene_query_interface
    physx_query = get_physx_scene_query_interface()
    _ts("PhysX scene query interface ready")
except Exception as e:
    print(f"ERROR: PhysX query unavailable ({e}) — cannot probe occlusion")
    simulation_app.close()
    sys.exit(1)

# ---------------------------------------------------------------------------
# Candidate grid generator
# ---------------------------------------------------------------------------

def generate_candidates(door_center_m, cam_yaw_deg=0.0):
    """Build the 45-position camera grid centered on one door.

    The arc is rotated so that arc=0° places the camera directly in front
    of the door (opposite the door-face direction).  cam_yaw_deg is the
    direction the camera faces; the camera sits on the opposite side, i.e.
    at world angle  cam_yaw_deg + 180°  from the door centre.
    """
    cx, cy, cz_base = door_center_m
    # Rotate arc so 0° = "in front of door" (opposite the face direction)
    base_rad = math.radians(cam_yaw_deg + 180.0)
    candidates = []
    idx = 0
    for dist in CAM_DISTANCES:
        for h_off in CAM_HEIGHT_OFFS:
            cam_z = cz_base + h_off
            for offset_deg in CAM_ARC_OFFSETS:
                theta = math.radians(offset_deg) + base_rad
                cam_pos = np.array([
                    cx + dist * math.cos(theta),
                    cy + dist * math.sin(theta),
                    cam_z,
                ])
                candidates.append({
                    "idx":        idx,
                    "pos":        cam_pos,
                    "offset_deg": offset_deg,
                    "distance":   dist,
                    "height_off": h_off,
                })
                idx += 1
    return candidates

# ---------------------------------------------------------------------------
# Occlusion check — identical predicate to door_replicator_collection.py
# ---------------------------------------------------------------------------

def probe_door(door):
    """Raycast from every candidate camera toward the door centre. A
    candidate is "clear" iff either no hit OR the first rigid-body hit
    path contains the door's prim path (i.e. we hit the door itself)."""
    door_prim = stage.GetPrimAtPath(door["prim_path"])
    if not door_prim.IsValid():
        return {"status": "prim_not_found", "clear": [], "blocked": []}

    # Concretize instanced prims so raycasts resolve against the real
    # mesh, not a read-only prototype.
    if door_prim.IsInstance() or door_prim.IsInstanceable():
        door_prim.SetInstanceable(False)
        simulation_app.update()
        for _ in range(DOOR_WARMUP):
            world.step(render=True)

    door_path = door["prim_path"]
    target = np.array([door["door_center_m"][0],
                       door["door_center_m"][1],
                       door["door_center_m"][2]])
    candidates = generate_candidates(door["door_center_m"], door.get("cam_yaw_deg", 0.0))

    clear = []
    blocked = []
    for cand in candidates:
        origin = cand["pos"]
        # Aim at the door centre at the same Z as the camera, matching
        # door_replicator_collection._check_line_of_sight().
        target_at_cam_z = np.array([target[0], target[1], origin[2]])
        direction = target_at_cam_z - origin
        max_dist = float(np.linalg.norm(direction))
        if max_dist < 0.01:
            clear.append({**cand, "reason": "degenerate"})
            continue
        direction = direction / max_dist
        hit = physx_query.raycast_closest(
            tuple(float(v) for v in origin),
            tuple(float(v) for v in direction),
            max_dist,
        )
        if not hit["hit"]:
            clear.append({**cand, "reason": "no_hit"})
            continue
        hit_path = hit.get("rigidBody", "") or ""
        if door_path in hit_path:
            clear.append({**cand, "reason": "hit_door", "hit_path": hit_path})
        else:
            blocked.append({**cand, "hit_path": hit_path})
    return {"status": "ok", "clear": clear, "blocked": blocked}

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def _cand_key(c):
    return {
        "arc":    int(c["offset_deg"]),
        "dist":   round(c["distance"], 2),
        "height": round(c["height_off"], 2),
    }

summary = {
    "grid": {
        "arc_offsets":    CAM_ARC_OFFSETS,
        "distances":      CAM_DISTANCES,
        "height_offsets": CAM_HEIGHT_OFFS,
        "total_per_door": len(CAM_ARC_OFFSETS) * len(CAM_DISTANCES) * len(CAM_HEIGHT_OFFS),
    },
    "doors": [],
}

total = len(CAM_ARC_OFFSETS) * len(CAM_DISTANCES) * len(CAM_HEIGHT_OFFS)

for di, door in enumerate(DOORS):
    _ts(f"[{di+1}/{len(DOORS)}] {door['family_short']:<12} {door['id']}  floor={door['floor']}")
    result = probe_door(door)
    if result["status"] != "ok":
        print(f"  SKIP — {result['status']}")
        summary["doors"].append({
            "id":            door["id"],
            "family_short":  door["family_short"],
            "floor":         door["floor"],
            "status":        result["status"],
        })
        continue

    clear_n   = len(result["clear"])
    blocked_n = len(result["blocked"])
    by_arc = {int(a): 0 for a in CAM_ARC_OFFSETS}
    for c in result["clear"]:
        by_arc[int(c["offset_deg"])] += 1

    print(f"  {clear_n}/{total} clear  ({blocked_n} blocked)")
    print(f"  per-arc survivors: " +
          "  ".join(f"{arc:+d}°:{by_arc[arc]}/{len(CAM_DISTANCES)*len(CAM_HEIGHT_OFFS)}"
                    for arc in sorted(by_arc)))

    # Log blocked hits for debugging (truncate per-door to avoid log spam)
    for b in result["blocked"][:6]:
        print(f"  [Blocked] arc={b['offset_deg']:+.0f}° dist={b['distance']:.1f}m "
              f"h={b['height_off']:+.1f}m → {b['hit_path']}")
    if blocked_n > 6:
        print(f"  ... ({blocked_n - 6} more blocked)")

    summary["doors"].append({
        "id":            door["id"],
        "family_short":  door["family_short"],
        "floor":         door["floor"],
        "cam_yaw_deg":   door.get("cam_yaw_deg", 0.0),
        "status":        "ok",
        "total":         total,
        "clear":         clear_n,
        "blocked":       blocked_n,
        "by_arc":        by_arc,
        "clear_grid":    [_cand_key(c) for c in result["clear"]],
        "blocked_sample": [
            {**_cand_key(b), "hit_path": b["hit_path"]}
            for b in result["blocked"][:12]
        ],
    })

# ---------------------------------------------------------------------------
# Aggregate + write summary
# ---------------------------------------------------------------------------

ok_doors = [d for d in summary["doors"] if d.get("status") == "ok"]
if ok_doors:
    clear_counts = [d["clear"] for d in ok_doors]
    summary["aggregate"] = {
        "doors_probed":  len(ok_doors),
        "total_per_door": total,
        "mean_clear":    float(np.mean(clear_counts)),
        "min_clear":     int(np.min(clear_counts)),
        "max_clear":     int(np.max(clear_counts)),
        "median_clear":  float(np.median(clear_counts)),
    }
    # Break down by family for the CW-vs-solid comparison.
    by_family = {}
    for d in ok_doors:
        by_family.setdefault(d["family_short"], []).append(d["clear"])
    summary["aggregate"]["by_family_mean"] = {
        fam: float(np.mean(v)) for fam, v in by_family.items()
    }

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
with open(OUT_PATH, "w") as f:
    json.dump(summary, f, indent=2)
    f.flush()
    os.fsync(f.fileno())

print()
_ts(f"Probe complete — {len(ok_doors)}/{len(DOORS)} doors OK")
if ok_doors:
    agg = summary["aggregate"]
    _ts(f"Clear cameras per door: mean={agg['mean_clear']:.1f}  "
        f"median={agg['median_clear']:.1f}  "
        f"min={agg['min_clear']}  max={agg['max_clear']}  "
        f"(out of {total} candidates)")
    _ts(f"By family (mean): " +
        "  ".join(f"{k}={v:.1f}" for k, v in agg["by_family_mean"].items()))
_ts(f"Summary → {OUT_PATH}")

world.stop()
simulation_app.close()

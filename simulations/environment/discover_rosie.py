#!/usr/bin/env python3
"""
Rosie (Supercomputer) Room Discovery for DierksHallV2_physics.usd

Rosie, MSOE's HPC cluster, lives on floor 2 of Dierks Hall but is not
named in the Revit USD (Revit exports strip user-facing names; only room
numbers/types remain). This script enumerates every Room prim on floor 2,
computes its world-space bounding box, and flags rooms whose footprint is
consistent with a data center / server room so a human can pick Rosie's
room from the candidate list.

Outputs:
  logs/dierks_floor2_rooms.json  -- full per-room inventory for downstream tools
  logs/dierks_floor2_rooms.txt   -- human-readable table, candidates starred

Usage:
  sbatch scripts/slurm/submit_isaac_job.sh simulations/environment/discover_rosie.py \\
      --input assets/environments/msoe_building/DierksHallV2_physics.usd
"""

import argparse
import json
import os
import sys
import time
from datetime import date

parser = argparse.ArgumentParser(description="Discover Dierks floor-2 rooms (for locating Rosie).")
parser.add_argument(
    "--input",
    default="assets/environments/msoe_building/DierksHallV2_physics.usd",
    help="Path to DierksHallV2_physics.usd",
)
parser.add_argument("--json-out", default=None, help="JSON output path override")
parser.add_argument("--report-out", default=None, help="Report output path override")
parser.add_argument(
    "--floor-index",
    type=int,
    default=2,
    help="1-based floor index to enumerate (default: 2 for Rosie). Pass 1 or 3 to re-scan other floors. Ignored if --z-min/--z-max provided.",
)
parser.add_argument(
    "--z-min",
    type=float,
    default=None,
    help="Manual Z-window bottom (meters). Overrides --floor-index.",
)
parser.add_argument(
    "--z-max",
    type=float,
    default=None,
    help="Manual Z-window top (meters). Overrides --floor-index.",
)
parser.add_argument(
    "--dump-all-rooms",
    action="store_true",
    help="Also emit every room (not just the floor-filtered set) in the JSON, for diagnostics.",
)
args, _unknown = parser.parse_known_args()

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

import omni.usd
from pxr import Sdf, Usd, UsdGeom
from isaacsim.core.api import World

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.abspath(os.path.join(_script_dir, '..', '..'))
sys.path.insert(0, _project_dir)

from sim_utils.building_physics import setup_physics_scene, discover_floor_levels

input_usd = os.path.abspath(args.input)
json_path = os.path.abspath(args.json_out) if args.json_out else os.path.join(
    _project_dir, "logs", f"dierks_floor{args.floor_index}_rooms.json"
)
report_path = os.path.abspath(args.report_out) if args.report_out else os.path.join(
    _project_dir, "logs", f"dierks_floor{args.floor_index}_rooms.txt"
)
os.makedirs(os.path.dirname(json_path), exist_ok=True)
os.makedirs(os.path.dirname(report_path), exist_ok=True)

_report_file = None


def log(msg=""):
    print(msg, flush=True)
    if _report_file:
        _report_file.write(msg + "\n")
        _report_file.flush()


# Server-room shape heuristic (meters, approximate).
#   Area is the sweet spot for a university HPC room: big enough for a few
#   racks + cooling, small enough to not be a classroom or lab.
#   Aspect ratio stays rectangular — data centers rarely have weird shapes.
SERVER_ROOM_AREA_MIN_M2 = 15.0
SERVER_ROOM_AREA_MAX_M2 = 120.0
SERVER_ROOM_ASPECT_MAX = 3.5


def bbox_dict(prim, bbox_cache):
    """World-space bbox as a dict, or None if the prim has no geometry."""
    bbox = bbox_cache.ComputeWorldBound(prim)
    rng = bbox.ComputeAlignedRange()
    if rng.IsEmpty():
        return None
    mn, mx = rng.GetMin(), rng.GetMax()
    return {
        "min": [float(mn[0]), float(mn[1]), float(mn[2])],
        "max": [float(mx[0]), float(mx[1]), float(mx[2])],
        "center": [float((mn[i] + mx[i]) / 2.0) for i in range(3)],
        "dims":   [float(mx[i] - mn[i]) for i in range(3)],
    }


def collect_room_prims(building_prim, bbox_cache):
    """Traverse Revit Rooms/ category; return a list of {path,name,bbox} dicts.

    Dierks rooms live under .../Geometry/Rooms/<RoomPrim>. A Room in Revit
    exports as a single Xform whose bbox encloses the room volume. Some
    room prims may have empty bboxes (e.g. placeholder / unbounded rooms)
    so we drop those.
    """
    out = []
    predicate = Usd.TraverseInstanceProxies()
    for prim in Usd.PrimRange(building_prim, predicate):
        path = str(prim.GetPath())
        if "/Geometry/Rooms/" not in path:
            continue
        # Skip nested children of a room prim — we want the room container itself.
        # Heuristic: a room prim is the FIRST Xform directly under /Geometry/Rooms/,
        # with no "/Geometry/Rooms/" appearing twice in the path.
        parts = path.split("/Geometry/Rooms/", 1)
        if len(parts) != 2:
            continue
        tail = parts[1]
        if "/" in tail:
            # This is a child of a room, not the room itself.
            continue
        bb = bbox_dict(prim, bbox_cache)
        if bb is None:
            continue
        out.append({
            "prim_path": path,
            "name": prim.GetName(),
            "bbox": bb,
        })
    return out


def classify_room(room, floor_z_bottom, floor_z_top):
    """Compute derived fields for a room: area, aspect ratio, server-shape flag."""
    dims = room["bbox"]["dims"]
    w, d, h = dims[0], dims[1], dims[2]
    area = w * d
    # Aspect ratio: longer side / shorter side. 1.0 = square, higher = more elongated.
    short = min(w, d)
    long_ = max(w, d)
    aspect = (long_ / short) if short > 0.01 else float("inf")

    is_server_shape = (
        SERVER_ROOM_AREA_MIN_M2 <= area <= SERVER_ROOM_AREA_MAX_M2
        and aspect <= SERVER_ROOM_ASPECT_MAX
        and h >= 2.4  # real room, not a chase or soffit
    )

    return {
        "width_m":  float(w),
        "depth_m":  float(d),
        "height_m": float(h),
        "area_m2":  float(area),
        "aspect_ratio": float(aspect),
        "is_server_room_shape": bool(is_server_shape),
    }


def rooms_on_floor(rooms, floor_z_bottom, floor_z_top):
    """Keep rooms whose centroid Z lies inside [floor_z_bottom, floor_z_top]."""
    kept = []
    for r in rooms:
        cz = r["bbox"]["center"][2]
        if floor_z_bottom <= cz < floor_z_top:
            kept.append(r)
    return kept


def main():
    global _report_file
    _report_file = open(report_path, "w")
    t_start = time.time()

    log("=" * 72)
    log(f"ROSIE ROOM DISCOVERY — DierksHallV2 (floor {args.floor_index})")
    log("=" * 72)
    log(f"Input:     {input_usd}")
    log(f"JSON out:  {json_path}")
    log(f"Report:    {report_path}")

    if not os.path.exists(input_usd):
        log(f"ERROR: {input_usd} not found.")
        _report_file.close()
        simulation_app.close()
        sys.exit(1)

    # --- Load building ---
    world = World(stage_units_in_meters=1.0)
    stage = omni.usd.get_context().get_stage()

    log("\nLoading building...")
    building_prim = stage.DefinePrim("/World/Building", "Xform")
    building_prim.GetReferences().AddReference(
        input_usd, primPath=Sdf.Path("/World/Building")
    )
    setup_physics_scene(stage)

    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    bb = bbox_dict(stage.GetPrimAtPath("/World/Building"), bbox_cache)
    log(f"Building bbox: min={bb['min']}  max={bb['max']}")

    # --- Discover floors so we know what "floor 2" means in world-Z terms ---
    log("\nDiscovering floor levels via sim_utils.building_physics.discover_floor_levels()...")
    levels = discover_floor_levels(stage, building_path="/World/Building")
    if not levels:
        log("ERROR: No floor levels discovered; cannot determine floor-2 Z range.")
        _report_file.close()
        simulation_app.close()
        sys.exit(2)

    log(f"Found {len(levels)} floor levels:")
    for i, lvl in enumerate(levels, 1):
        log(f"  [{i}] {lvl['name']:<12s}  top_z={lvl['z']:.2f}m  center=({lvl['center_x']:.1f},{lvl['center_y']:.1f})  count={lvl['count']}")

    # Floor-N room filter: rooms whose centroid sits between floor[N-1].z (the
    # floor we stand on) and floor[N].z (the floor above). For the top floor,
    # use the building bbox top as the upper bound.
    #
    # Gotcha: discover_floor_levels() returns EVERY slab cluster, including
    # mezzanines and roof layers. For Dierks V2 it returns 9 slabs for a
    # ~3-story building. An explicit --z-min/--z-max override sidesteps the
    # auto-selection when the slab list doesn't map cleanly to user-facing floors.
    idx = args.floor_index
    if args.z_min is not None and args.z_max is not None:
        floor_z_bottom = float(args.z_min)
        floor_z_top = float(args.z_max)
        log(f"\nUsing manual Z window: [{floor_z_bottom:.2f}, {floor_z_top:.2f}] m")
    else:
        if idx < 1 or idx > len(levels):
            log(f"ERROR: --floor-index {idx} out of range (have {len(levels)} floors).")
            _report_file.close()
            simulation_app.close()
            sys.exit(3)
        floor_z_bottom = levels[idx - 1]["z"]
        floor_z_top = levels[idx]["z"] if idx < len(levels) else bb["max"][2]
        log(f"\nAuto floor-{idx} Z window: [{floor_z_bottom:.2f}, {floor_z_top:.2f}] m")

    # --- Collect all rooms, classify, filter to target floor ---
    log("\nTraversing /World/Building for Geometry/Rooms/* prims...")
    all_rooms = collect_room_prims(stage.GetPrimAtPath("/World/Building"), bbox_cache)
    log(f"  Found {len(all_rooms)} room prims building-wide")

    # Histogram rooms by centroid Z in 2m bins — shows where actual "floors" sit
    # independent of the Floors-category clustering above. Useful when slab
    # detection picks up mezzanines.
    if all_rooms:
        z_min_all = min(r["bbox"]["center"][2] for r in all_rooms)
        z_max_all = max(r["bbox"]["center"][2] for r in all_rooms)
        log(f"\nRoom centroid Z range: {z_min_all:.2f} to {z_max_all:.2f} m")
        log("Rooms per 2m Z-bin (helps spot floor boundaries):")
        bin_size = 2.0
        bin_start = int(z_min_all // bin_size) * bin_size
        bin_end = int(z_max_all // bin_size + 1) * bin_size
        z = bin_start
        while z < bin_end:
            count = sum(1 for r in all_rooms if z <= r["bbox"]["center"][2] < z + bin_size)
            bar = "█" * count
            log(f"  [{z:5.1f}, {z + bin_size:5.1f}) m  {count:3d}  {bar}")
            z += bin_size

    target_rooms = rooms_on_floor(all_rooms, floor_z_bottom, floor_z_top)
    log(f"\n  {len(target_rooms)} rooms fall in the Z window [{floor_z_bottom:.2f}, {floor_z_top:.2f}] m")

    # Enrich with classification.
    for r in target_rooms:
        r.update(classify_room(r, floor_z_bottom, floor_z_top))

    # Sort: server-room-shaped first, then by area descending.
    target_rooms.sort(
        key=lambda r: (not r["is_server_room_shape"], -r["area_m2"])
    )

    # --- Report ---
    log(f"\n{'=' * 72}")
    log(f"Floor-{idx} rooms (sorted: server-room-shape first, then area desc)")
    log(f"{'=' * 72}")
    log(f"{'#':>3}  {'*':1}  {'Name':<42s}  {'Area':>7s}  {'W×D×H':>18s}  {'Aspect':>6s}")
    log("-" * 72)
    candidates = []
    for i, r in enumerate(target_rooms, 1):
        star = "*" if r["is_server_room_shape"] else " "
        wdh = f"{r['width_m']:.1f}×{r['depth_m']:.1f}×{r['height_m']:.1f}"
        name = r["name"][:42]
        log(f"{i:>3}  {star:1}  {name:<42s}  {r['area_m2']:>6.1f}m²  {wdh:>18s}  {r['aspect_ratio']:>5.2f}")
        if r["is_server_room_shape"]:
            candidates.append(r)

    log(f"\n{len(candidates)} server-room-shape candidates (area {SERVER_ROOM_AREA_MIN_M2:.0f}–{SERVER_ROOM_AREA_MAX_M2:.0f} m², aspect ≤ {SERVER_ROOM_ASPECT_MAX:.1f}).")
    log("Rosie is almost certainly one of these — pick the one whose proportions")
    log("match a rack-filled data center (rectangular, 30–80 m² typical).")

    # --- Write JSON ---
    output = {
        "source_usd": os.path.basename(input_usd),
        "discovery_date": str(date.today()),
        "floor_index": idx,
        "floor_z_bottom": float(floor_z_bottom),
        "floor_z_top":    float(floor_z_top),
        "building_bbox": bb,
        "levels": [
            {
                "name": lvl["name"],
                "z_top": float(lvl["z"]),
                "center_x": float(lvl["center_x"]),
                "center_y": float(lvl["center_y"]),
                "count": int(lvl["count"]),
            }
            for lvl in levels
        ],
        "heuristic": {
            "area_min_m2": SERVER_ROOM_AREA_MIN_M2,
            "area_max_m2": SERVER_ROOM_AREA_MAX_M2,
            "aspect_max": SERVER_ROOM_ASPECT_MAX,
        },
        "rooms": target_rooms,
        "candidate_count": len(candidates),
    }
    if args.dump_all_rooms:
        # Cheap classification of every room for diagnostic exploration.
        all_rooms_enriched = []
        for r in all_rooms:
            r2 = dict(r)
            r2.update(classify_room(r, bb["min"][2], bb["max"][2]))
            all_rooms_enriched.append(r2)
        all_rooms_enriched.sort(key=lambda r: r["bbox"]["center"][2])
        output["all_rooms"] = all_rooms_enriched
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)

    t_end = time.time()
    log(f"\n{'=' * 72}")
    log("Discovery complete.")
    log(f"  JSON    -> {json_path}")
    log(f"  Report  -> {report_path}")
    log(f"  Runtime: {t_end - t_start:.1f}s")
    log("=" * 72)

    _report_file.close()
    simulation_app.close()


if __name__ == "__main__":
    main()

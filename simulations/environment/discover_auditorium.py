#!/usr/bin/env python3
"""
Auditorium Discovery for DierksHallV2_physics.usd

Traverses the Dierks Hall USD to locate the stage, auditorium volume, and
front-row seats by matching Revit-encoded prim names, then derives placement
anchors for 8 standing presenters + 4 seated audience members plus an
elevated audience-POV camera.

Outputs:
  logs/dierks_auditorium.json   -- cached anchor data for the scene script
  logs/dierks_auditorium.txt    -- human-readable discovery report

Usage:
  sbatch submit_isaac_job.sh simulations/environment/discover_auditorium.py \\
      --input assets/environments/msoe_building/DierksHallV2_physics.usd
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import date

parser = argparse.ArgumentParser(description="Discover Dierks auditorium anchors.")
parser.add_argument("--input", required=True, help="Path to DierksHallV2_physics.usd")
parser.add_argument("--json-out", default=None, help="JSON output path override")
parser.add_argument("--report-out", default=None, help="Report output path override")
args, _unknown = parser.parse_known_args()

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

import numpy as np
import omni.usd
from pxr import Sdf, Usd, UsdGeom
from isaacsim.core.api import World

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.abspath(os.path.join(_script_dir, '..', '..'))
sys.path.insert(0, _project_dir)

from sim_utils.building_physics import setup_physics_scene

input_usd = os.path.abspath(args.input)
json_path = os.path.abspath(args.json_out) if args.json_out else os.path.join(
    _project_dir, "logs", "dierks_auditorium.json"
)
report_path = os.path.abspath(args.report_out) if args.report_out else os.path.join(
    _project_dir, "logs", "dierks_auditorium.txt"
)
os.makedirs(os.path.dirname(json_path), exist_ok=True)
os.makedirs(os.path.dirname(report_path), exist_ok=True)

_report_file = None


def log(msg=""):
    print(msg, flush=True)
    if _report_file:
        _report_file.write(msg + "\n")
        _report_file.flush()


# Regex patterns on raw Revit prim names (case-insensitive).
#   Stage: broad "stage" match, checked against both name AND path so we catch
#          parent Xforms like "_28_X12__STAGE_OPENING" that wrap the mesh.
#          Excludes non-stage false positives (hide_in_stage_window, backstage).
#   Auditorium: any prim with "auditorium" in the name (walls, floor, audio panels).
#   Seats: "_seat" or "seat_" (avoids matching unrelated words); rejects DEPT_CHAIR
#          since that's a single office chair, not a row of audience seats.
STAGE_RE = re.compile(r"stage", re.IGNORECASE)
STAGE_REJECT_RE = re.compile(
    r"(hide_in_stage_window|backstage|stagemgmt|rootstage|composed[_ ]stage)",
    re.IGNORECASE,
)
AUDITORIUM_RE = re.compile(r"auditorium", re.IGNORECASE)
SEAT_RE = re.compile(r"(_seat|seat_)", re.IGNORECASE)
SEAT_REJECT_RE = re.compile(r"dept_chair", re.IGNORECASE)

# Clustering tolerances (meters).
SEAT_Z_TOL = 0.3      # seats at same row share Z within 30cm (sloped auditoriums OK at this granularity)
SEAT_XY_TOL = 0.5     # seats in same column share the perpendicular axis within 50cm

# Placement parameters (meters).
STANDING_COUNT = 8
SITTING_COUNT = 4
STANDING_SPACING = 1.0        # 1m between presenters on stage
STANDING_SETBACK = 0.5        # stand 0.5m inside the stage front edge
CAM_ELEVATION = 3.8           # 3.8m above stage floor — under the 4.27m room ceiling
CAM_BACK_OFFSET = 6.0         # 6m behind front row — keeps camera inside the auditorium
CAM_TARGET_Z_OFFSET = 1.5     # look at chest height above stage


def bbox_dict(stage, prim, bbox_cache=None):
    """Compute world bbox as a dict; returns None if empty."""
    if bbox_cache is None:
        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
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


def scan_stage_for_matches(stage, bbox_cache, building_prim):
    """Single traversal of the building; bucket matching prims by category.

    Uses Usd.PrimRange with TraverseInstanceProxies so we descend into Revit
    instances — which is how DierksHallV2 is organized. The stage/auditorium
    prims may be Xforms wrapping the actual mesh (empty bbox on the wrapper),
    so we drop the Boundable filter and rely on the bbox call returning None
    for truly empty prims.
    """
    stage_hits = []
    aud_hits = []
    seat_hits = []

    predicate = Usd.TraverseInstanceProxies()
    for prim in Usd.PrimRange(building_prim, predicate):
        path = str(prim.GetPath())
        name = prim.GetName()

        is_stage = (STAGE_RE.search(name) or STAGE_RE.search(path)) and not STAGE_REJECT_RE.search(path)
        is_aud = AUDITORIUM_RE.search(name) or AUDITORIUM_RE.search(path)
        is_seat = (
            (SEAT_RE.search(name) or SEAT_RE.search(path))
            and not SEAT_REJECT_RE.search(name)
        )

        if not (is_stage or is_aud or is_seat):
            continue

        bb = bbox_dict(stage, prim, bbox_cache)
        if bb is None:
            continue

        entry = {"prim_path": path, "name": name, "bbox": bb}
        if is_stage:
            stage_hits.append(entry)
        if is_aud:
            aud_hits.append(entry)
        if is_seat:
            seat_hits.append(entry)

    return stage_hits, aud_hits, seat_hits


def pick_stage(stage_hits):
    """Pick the actual stage FLOOR as the placement surface.

    Revit exports multiple stage-related prims (room volume, floor, finishes).
    For standing people we want the FLOOR — its bbox center matches the stage
    XY and its bbox MAX Z is the walkable surface. Prefer prims whose name
    contains "floor"; fall back to the biggest horizontal footprint.
    """
    if not stage_hits:
        return None
    floor_hits = [h for h in stage_hits if "floor" in h["name"].lower()]
    pool = floor_hits if floor_hits else stage_hits
    ranked = sorted(
        pool,
        key=lambda h: h["bbox"]["dims"][0] * h["bbox"]["dims"][1],
        reverse=True,
    )
    return ranked[0]


def pick_auditorium_room(aud_hits):
    """Pick the auditorium ROOM volume — the biggest single-prim bbox.

    Among all "auditorium" matches, the room (e.g. _256_SEAT_AUDITORIUM_...)
    has the largest volume — it encloses all the panels and seats. Other
    matches (acoustic panels, fabric) are tiny by comparison.
    """
    if not aud_hits:
        return None
    ranked = sorted(
        aud_hits,
        key=lambda h: h["bbox"]["dims"][0] * h["bbox"]["dims"][1] * h["bbox"]["dims"][2],
        reverse=True,
    )
    return ranked[0]


def dedup_by_xy(hits, tol=0.3):
    """Revit instance proxies produce many duplicate matches at the same XY.

    Keep one representative per XY cluster (within tol meters) so Z-clustering
    into rows isn't biased by duplicate counts.
    """
    out = []
    for h in hits:
        cx, cy = h["bbox"]["center"][0], h["bbox"]["center"][1]
        dup = False
        for kept in out:
            kx, ky = kept["bbox"]["center"][0], kept["bbox"]["center"][1]
            if abs(cx - kx) < tol and abs(cy - ky) < tol:
                dup = True
                break
        if not dup:
            out.append(h)
    return out


def cluster_by_axis(items, axis_index, tol):
    """Cluster items (with ['bbox']['center']) by one axis coordinate.

    Greedy 1D clustering: sort by axis, start a new cluster whenever the gap
    exceeds tol. Good enough for auditorium seats (they fall in obvious rows).
    """
    if not items:
        return []
    srt = sorted(items, key=lambda it: it["bbox"]["center"][axis_index])
    clusters = [[srt[0]]]
    for it in srt[1:]:
        last = clusters[-1][-1]
        gap = it["bbox"]["center"][axis_index] - last["bbox"]["center"][axis_index]
        if gap <= tol:
            clusters[-1].append(it)
        else:
            clusters.append([it])
    return clusters


def find_front_row(seat_hits, stage_center):
    """Cluster seats by Z, then pick the Z-cluster closest to the stage.

    Sloped auditoriums produce a natural Z-stratification across rows. The row
    closest to the stage (smallest horizontal distance to stage center) is the
    front row.
    """
    if not seat_hits:
        return []
    z_clusters = cluster_by_axis(seat_hits, axis_index=2, tol=SEAT_Z_TOL)

    def dist_to_stage(cluster):
        cx = np.mean([s["bbox"]["center"][0] for s in cluster])
        cy = np.mean([s["bbox"]["center"][1] for s in cluster])
        return np.hypot(cx - stage_center[0], cy - stage_center[1])

    z_clusters.sort(key=dist_to_stage)
    return z_clusters[0]


def derive_axes(stage_center, audience_center):
    """Compute depth axis (stage→audience) and row axis (perpendicular).

    Returns (depth_axis, row_axis, edge_sign) where edge_sign is +1 if the
    audience is in the +axis direction from the stage, -1 otherwise.
    """
    to_aud = np.array(audience_center[:2]) - np.array(stage_center[:2])
    if abs(to_aud[0]) > abs(to_aud[1]):
        depth_axis, row_axis = 0, 1
    else:
        depth_axis, row_axis = 1, 0
    edge_sign = 1.0 if to_aud[depth_axis] > 0 else -1.0
    return depth_axis, row_axis, edge_sign


def derive_standing_anchors(stage_bbox, depth_axis, row_axis, edge_sign):
    """8 standing anchors on the audience-facing edge of the stage floor."""
    sc = np.array(stage_bbox["center"])
    dims = np.array(stage_bbox["dims"])
    half_depth = dims[depth_axis] / 2.0
    edge_coord = sc[depth_axis] + edge_sign * (half_depth - STANDING_SETBACK)

    total_span = (STANDING_COUNT - 1) * STANDING_SPACING
    row_start = sc[row_axis] - total_span / 2.0
    stand_z = stage_bbox["max"][2]

    anchors = []
    for i in range(STANDING_COUNT):
        pos = [0.0, 0.0, stand_z]
        pos[depth_axis] = float(edge_coord)
        pos[row_axis] = float(row_start + i * STANDING_SPACING)
        anchors.append(pos)
    return anchors


def derive_sitting_anchors_from_room(
    stage_bbox, room_bbox, depth_axis, row_axis, edge_sign, front_row_z,
):
    """4 sitting anchors — front row of audience in front of stage face.

    Geometry reference is the STAGE face (not the room edge), because the
    Revit "Room" bbox in Dierks overlaps the stage in the depth axis. The
    stage face is the edge of the stage closest to the audience:
        stage_face = stage_center[d] + edge_sign * stage_half_depth
    Front row sits FRONT_ROW_OFFSET further into the audience:
        depth_coord = stage_face + edge_sign * FRONT_ROW_OFFSET
    Row axis is centered on the stage center so audience aligns with presenters.
    """
    sc = np.array(stage_bbox["center"])
    sdims = np.array(stage_bbox["dims"])

    stage_face = sc[depth_axis] + edge_sign * (sdims[depth_axis] / 2.0)
    FRONT_ROW_OFFSET = 2.0  # 2m gap between stage edge and first row
    depth_coord = stage_face + edge_sign * FRONT_ROW_OFFSET

    sitting_pitch = 1.0
    total_span = (SITTING_COUNT - 1) * sitting_pitch
    row_start = sc[row_axis] - total_span / 2.0

    # Seat pan Z: use measured front-row Z if available, else stage_top + 0.45m.
    if front_row_z is not None:
        sit_z = front_row_z
    else:
        sit_z = stage_bbox["max"][2] + 0.45

    anchors = []
    for i in range(SITTING_COUNT):
        pos = [0.0, 0.0, float(sit_z)]
        pos[depth_axis] = float(depth_coord)
        pos[row_axis] = float(row_start + i * sitting_pitch)
        anchors.append(pos)
    return anchors


def derive_camera(stage_center, stage_bbox, standing_anchors, sitting_anchors):
    """Elevated audience POV — behind the front row, above them, looking at the stage.

    Matches the user's Q&A answer: "audience view but from above overlooking the stage".
    """
    aud_center = np.mean(np.array(sitting_anchors), axis=0)
    to_stage = np.array(stage_center[:2]) - aud_center[:2]
    dist_to_stage = np.linalg.norm(to_stage)
    to_stage_unit = to_stage / max(dist_to_stage, 1e-6)

    # Camera: back from the audience centroid (away from stage), elevated.
    cam_xy = aud_center[:2] - to_stage_unit * CAM_BACK_OFFSET
    cam_z = stage_bbox["max"][2] + CAM_ELEVATION

    # Target: stage centroid, lifted to chest height.
    target_xy = np.array(stage_center[:2])
    target_z = stage_bbox["max"][2] + CAM_TARGET_Z_OFFSET

    return {
        "position": [float(cam_xy[0]), float(cam_xy[1]), float(cam_z)],
        "target":   [float(target_xy[0]), float(target_xy[1]), float(target_z)],
        "focal_length_mm": 18.0,
        "resolution": [1920, 1080],
    }


def main():
    global _report_file
    _report_file = open(report_path, "w")
    t_start = time.time()

    log("=" * 70)
    log("AUDITORIUM DISCOVERY — DierksHallV2")
    log("=" * 70)
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

    log(f"\nLoading building...")
    building_prim = stage.DefinePrim("/World/Building", "Xform")
    building_prim.GetReferences().AddReference(
        input_usd, primPath=Sdf.Path("/World/Building")
    )
    setup_physics_scene(stage)

    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    bb = bbox_dict(stage, stage.GetPrimAtPath("/World/Building"), bbox_cache)
    log(f"Building bbox: min={bb['min']}  max={bb['max']}")

    # --- Scan for stage, auditorium, seats ---
    log(f"\nTraversing /World/Building for stage/auditorium/seat prims...")
    stage_hits, aud_hits, seat_hits = scan_stage_for_matches(
        stage, bbox_cache, stage.GetPrimAtPath("/World/Building")
    )
    log(f"  Stage matches:       {len(stage_hits)}")
    log(f"  Auditorium matches:  {len(aud_hits)}")
    log(f"  Seat matches:        {len(seat_hits)}")

    # Diagnostic: log first 10 of each category so we can verify what matched.
    log(f"\nFirst 10 stage hits:")
    for h in stage_hits[:10]:
        c = h["bbox"]["center"]
        d = h["bbox"]["dims"]
        log(f"  {h['name']:<45s}  C=({c[0]:.1f},{c[1]:.1f},{c[2]:.1f})  D=({d[0]:.1f},{d[1]:.1f},{d[2]:.1f})")
        log(f"    path: {h['prim_path']}")
    log(f"\nFirst 10 auditorium hits:")
    for h in aud_hits[:10]:
        c = h["bbox"]["center"]
        d = h["bbox"]["dims"]
        log(f"  {h['name']:<45s}  C=({c[0]:.1f},{c[1]:.1f},{c[2]:.1f})  D=({d[0]:.1f},{d[1]:.1f},{d[2]:.1f})")
        log(f"    path: {h['prim_path']}")

    if not stage_hits:
        log("\nERROR: No stage prims found. Auditorium discovery failed.")
        log("Consider running with --manual-anchors or checking the USD for name changes.")
        _report_file.close()
        simulation_app.close()
        sys.exit(2)

    stage_info = pick_stage(stage_hits)
    stage_bbox = stage_info["bbox"]
    stage_center = stage_bbox["center"]
    log(f"\nStage chosen: {stage_info['name']}")
    log(f"  Path:   {stage_info['prim_path']}")
    log(f"  Center: ({stage_center[0]:.2f}, {stage_center[1]:.2f}, {stage_center[2]:.2f})")
    log(f"  Dims:   ({stage_bbox['dims'][0]:.2f}, {stage_bbox['dims'][1]:.2f}, {stage_bbox['dims'][2]:.2f})")

    # --- Pick auditorium room (the big "_256_SEAT_AUDITORIUM_..." prim) ---
    room_info = pick_auditorium_room(aud_hits)
    if room_info is None:
        log("\nERROR: No auditorium room prim found.")
        _report_file.close()
        simulation_app.close()
        sys.exit(3)
    room_bbox = room_info["bbox"]
    room_center = room_bbox["center"]
    log(f"\nAuditorium room: {room_info['name']}")
    log(f"  Path:   {room_info['prim_path']}")
    log(f"  Center: ({room_center[0]:.2f}, {room_center[1]:.2f}, {room_center[2]:.2f})")
    log(f"  Dims:   ({room_bbox['dims'][0]:.2f}, {room_bbox['dims'][1]:.2f}, {room_bbox['dims'][2]:.2f})")

    # --- Dedup + cluster seats to estimate front-row Z ---
    log(f"\nDeduplicating {len(seat_hits)} seat hits (XY tolerance 0.3m)...")
    seats_dedup = dedup_by_xy(seat_hits, tol=0.3)
    log(f"  {len(seats_dedup)} unique seat positions after dedup")

    front_row_z = None
    if seats_dedup:
        front_row = find_front_row(seats_dedup, stage_center)
        if front_row and len(front_row) >= 4:
            front_row_z = float(np.mean([s["bbox"]["max"][2] for s in front_row]))
            log(f"  Front-row Z (from {len(front_row)} seat cluster): {front_row_z:.2f}")

    # --- Derive axes from room center vs stage center ---
    depth_axis, row_axis, edge_sign = derive_axes(stage_center, room_center)
    log(f"\nAxes: depth_axis={depth_axis} (0=X, 1=Y), row_axis={row_axis}, edge_sign={edge_sign:+.0f}")
    log(f"  Audience is {'+' if edge_sign > 0 else '-'}{'X' if depth_axis==0 else 'Y'} of stage")

    standing_anchors = derive_standing_anchors(stage_bbox, depth_axis, row_axis, edge_sign)
    log(f"\nStanding anchors (8):")
    for i, p in enumerate(standing_anchors):
        log(f"  [{i}] ({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f})")

    sitting_anchors = derive_sitting_anchors_from_room(
        stage_bbox, room_bbox, depth_axis, row_axis, edge_sign, front_row_z,
    )
    log(f"\nSitting anchors (4):")
    for i, p in enumerate(sitting_anchors):
        log(f"  [{i}] ({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f})")

    camera = derive_camera(stage_center, stage_bbox, standing_anchors, sitting_anchors)
    log(f"\nCamera:")
    log(f"  Position: ({camera['position'][0]:.2f}, {camera['position'][1]:.2f}, {camera['position'][2]:.2f})")
    log(f"  Target:   ({camera['target'][0]:.2f}, {camera['target'][1]:.2f}, {camera['target'][2]:.2f})")

    # --- Write JSON ---
    output = {
        "source_usd": os.path.basename(input_usd),
        "discovery_date": str(date.today()),
        "building_bbox": bb,
        "stage": {
            "prim_path": stage_info["prim_path"],
            "name": stage_info["name"],
            "bbox": stage_bbox,
        },
        "auditorium_room": {
            "prim_path": room_info["prim_path"],
            "name": room_info["name"],
            "bbox": room_bbox,
        },
        "axes": {
            "depth_axis": depth_axis,
            "row_axis": row_axis,
            "edge_sign": edge_sign,
        },
        "front_row_z": front_row_z,
        "standing_anchors": standing_anchors,
        "sitting_anchors": sitting_anchors,
        "camera": camera,
        "match_counts": {
            "stage": len(stage_hits),
            "auditorium": len(aud_hits),
            "seat": len(seat_hits),
            "seat_dedup": len(seats_dedup),
        },
    }
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)

    t_end = time.time()
    log(f"\n{'=' * 70}")
    log("Discovery complete.")
    log(f"  JSON    -> {json_path}")
    log(f"  Report  -> {report_path}")
    log(f"  Runtime: {t_end - t_start:.1f}s")
    log("=" * 70)

    _report_file.close()
    simulation_app.close()


if __name__ == "__main__":
    main()

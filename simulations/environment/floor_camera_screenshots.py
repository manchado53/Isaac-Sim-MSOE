#!/usr/bin/env python3
"""
Floor Camera Screenshots — MSOEBuildingTake2.usd

Discovers floor levels, inspects the 45 Revit-exported cameras, selects the
best interior-view camera per floor, and captures a screenshot from each.

Falls back to room-based placement when no suitable Revit camera exists for
a floor.

Output:
  - videos/environment/floor_<name>_camera.png   (one PNG per floor, 1920x1080)
  - logs/floor_camera_report.txt     (detailed discovery log)
  - stdout JSON summary of camera positions

Usage:
  sbatch submit_isaac_job.sh simulations/environment/floor_camera_screenshots.py
"""

import os
import sys
import json
import math
import re
import time

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

import omni.usd
import numpy as np
import isaacsim.core.utils.numpy.rotations as rot_utils
from pxr import Usd, UsdGeom, UsdLux, Sdf, Gf
from isaacsim.core.api import World
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.sensors.camera import Camera

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.abspath(os.path.join(_script_dir, '..', '..'))
sys.path.insert(0, _project_dir)

from sim_utils.building_physics import (
    _find_wrapper,
    iter_building_elements,
)

_report_path = os.path.join(_project_dir, "logs", "floor_camera_report.txt")
os.makedirs(os.path.dirname(_report_path), exist_ok=True)
_report_file = None


def log(msg=""):
    print(msg, flush=True)
    if _report_file:
        _report_file.write(msg + "\n")
        _report_file.flush()


# ---------------------------------------------------------------------------
# Floor discovery (adapted from engineering_physics_demo.py)
# ---------------------------------------------------------------------------

def discover_floor_levels(stage, building_path="/World/Building"):
    """Cluster Floor slab bounding boxes by Z to find distinct levels.

    Returns list of dicts sorted by z:
        [{"name", "z", "center_x", "center_y", "count"}, ...]
    """
    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]
    )
    building_prim = stage.GetPrimAtPath(building_path)

    floor_data = []
    for _level, category_name, element in iter_building_elements(stage, building_prim):
        if category_name != "Floors":
            continue
        bbox = bbox_cache.ComputeWorldBound(element)
        box_range = bbox.ComputeAlignedRange()
        bb_min = box_range.GetMin()
        bb_max = box_range.GetMax()
        if bb_min == bb_max:
            continue
        dx = bb_max[0] - bb_min[0]
        dy = bb_max[1] - bb_min[1]
        if dx < 0.5 and dy < 0.5:
            continue
        floor_data.append({
            "z_top": bb_max[2],
            "x_center": (bb_min[0] + bb_max[0]) / 2.0,
            "y_center": (bb_min[1] + bb_max[1]) / 2.0,
            "area": dx * dy,
        })

    if not floor_data:
        return []

    floor_data.sort(key=lambda f: f["z_top"])
    clusters = []
    current = [floor_data[0]]
    for fd in floor_data[1:]:
        if fd["z_top"] - current[0]["z_top"] > 4.0:
            clusters.append(current)
            current = [fd]
        else:
            current.append(fd)
    clusters.append(current)

    levels = []
    for i, cluster in enumerate(clusters):
        total_area = sum(f["area"] for f in cluster)
        if total_area < 0.01:
            continue
        avg_x = sum(f["x_center"] * f["area"] for f in cluster) / total_area
        avg_y = sum(f["y_center"] * f["area"] for f in cluster) / total_area
        avg_z = sum(f["z_top"] * f["area"] for f in cluster) / total_area
        levels.append({
            "name": f"floor_{i + 1}",
            "z": avg_z,
            "center_x": avg_x,
            "center_y": avg_y,
            "count": len(cluster),
        })

    # Filter out basement (very low Z) and roof (very high Z, few slabs),
    # then take the first 4 floors by Z ordering
    if len(levels) > 4:
        # Remove bottom floor if it has few slabs (basement)
        # and top floor if it has few slabs (roof)
        if levels[0]["count"] <= 4:
            levels = levels[1:]
        if levels[-1]["count"] <= 2:
            levels = levels[:-1]

    return levels[:4]


# ---------------------------------------------------------------------------
# Revit camera scanning
# ---------------------------------------------------------------------------

def find_cameras_prim(stage, building_prim):
    """Find the Cameras Xform prim (sibling of Geometry under the wrapper)."""
    wrapper = _find_wrapper(building_prim)
    if wrapper is None:
        return None
    for child in wrapper.GetChildren():
        if child.GetName() == "Cameras":
            return child
    return None


def scan_revit_cameras(stage, cameras_prim):
    """Extract position and orientation from all Revit-exported Camera prims.

    Returns list of dicts: {name, position: (x,y,z), prim_path, orientation_matrix}
    """
    results = []
    for child in cameras_prim.GetChildren():
        xformable = UsdGeom.Xformable(child)
        world_xform = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        pos = world_xform.ExtractTranslation()
        results.append({
            "name": child.GetName(),
            "prim_path": str(child.GetPath()),
            "position": (float(pos[0]), float(pos[1]), float(pos[2])),
            "world_xform": world_xform,
        })
    return results


# ---------------------------------------------------------------------------
# Camera scoring and assignment
# ---------------------------------------------------------------------------

# Interior/lab keywords score higher (we want room-level views)
_SCORE_KEYWORDS = [
    ("Transportation", 10), ("Physics_Lab", 10), ("construction", 10),
    ("Advance_design", 10), ("Advance_Design", 10), ("Home_Space", 9),
    ("Lab", 8), ("Classroom", 8), ("Studio", 8),
    ("Interiors", 7), ("Atrium", 7), ("walk_thru", 7),
    ("Ramp", 5), ("Stair", 4),
    ("CORNER", 3), ("COURTYARD", 3), ("East_Face", 2),
    ("_3D_", 1),
]


def score_camera_name(name):
    """Score a Revit camera name by how likely it is to give a good interior view."""
    score = 0
    for keyword, pts in _SCORE_KEYWORDS:
        if keyword.lower() in name.lower():
            score = max(score, pts)
    return score


def assign_cameras_to_floors(cameras, floors):
    """Match each Revit camera to a floor by Z-coordinate.

    Returns dict[floor_name] -> best camera dict (or None).
    """
    # Build Z-ranges for each floor with tolerance
    floor_assignment = {}
    for fl in floors:
        floor_assignment[fl["name"]] = {
            "floor": fl,
            "candidates": [],
        }

    for cam in cameras:
        cam_z = cam["position"][2]
        best_floor = None
        best_dist = float("inf")
        for fl in floors:
            # Camera should be above the floor slab (0 to ~14 ft above)
            z_above = cam_z - fl["z"]
            if -2.0 <= z_above <= 16.0:
                dist = abs(z_above - 5.5)  # prefer ~5.5ft eye height
                if dist < best_dist:
                    best_dist = dist
                    best_floor = fl["name"]

        if best_floor:
            cam["score"] = score_camera_name(cam["name"])
            cam["z_above_floor"] = cam["position"][2] - floor_assignment[best_floor]["floor"]["z"]
            floor_assignment[best_floor]["candidates"].append(cam)

    # Pick best camera per floor
    result = {}
    for floor_name, data in floor_assignment.items():
        candidates = data["candidates"]
        if not candidates:
            result[floor_name] = None
            continue
        # Sort by score descending, then by closeness to eye height
        candidates.sort(key=lambda c: (-c["score"], abs(c["z_above_floor"] - 5.5)))
        result[floor_name] = candidates[0]

    return result


# ---------------------------------------------------------------------------
# Room-based fallback
# ---------------------------------------------------------------------------

def find_room_camera_fallback(stage, building_prim, floor_z, floor_center_x, floor_center_y,
                              exclude_rooms=None, bldg_center_x=None, bldg_center_y=None):
    """Find a good room on the given floor and compute a camera position inside it.

    If bldg_center_x/y are provided, the camera is biased toward the building
    interior to avoid courtyard planting geometry.

    Returns dict {position, target, room_name} or None.
    """
    wrapper = _find_wrapper(building_prim)
    if wrapper is None:
        return None

    # Find Geometry prim
    geometry_prim = None
    for child in wrapper.GetChildren():
        if child.GetName() == "Geometry" and child.IsA(UsdGeom.Xform):
            geometry_prim = child
            break
    if geometry_prim is None:
        return None

    # Find Rooms folder
    rooms_prim = None
    for child in geometry_prim.GetChildren():
        if child.GetName() == "Rooms":
            rooms_prim = child
            break
    if rooms_prim is None:
        return None

    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])

    # Scan rooms, find ones on this floor
    candidates = []
    for room in rooms_prim.GetChildren():
        bbox = bbox_cache.ComputeWorldBound(room)
        rng = bbox.ComputeAlignedRange()
        if rng.IsEmpty():
            continue
        mn = rng.GetMin()
        mx = rng.GetMax()
        z_min = float(mn[2])
        z_max = float(mx[2])
        dx = float(mx[0] - mn[0])
        dy = float(mx[1] - mn[1])
        area = dx * dy

        # Room must overlap this floor level (within a few feet)
        if z_min > floor_z + 3.0 or z_max < floor_z - 1.0:
            continue
        if area < 10.0:  # skip tiny closets
            continue

        name = room.GetName()

        # Skip already-used rooms
        if exclude_rooms and name in exclude_rooms:
            continue

        # Score rooms: prefer interesting program spaces, penalize utility rooms
        name_upper = name.upper()

        # Skip utility rooms entirely
        skip = False
        for kw in ["MECHANICAL", "ELECTRICAL", "FIRE_PUMP", "ELEV", "JAN",
                    "STORAGE", "PLENUM", "SHAFT", "CHASE", "EQUIP",
                    "STAIR", "ACCESS", "RAMP", "WOMEN", "MEN", "BATHROOM",
                    "VESTIBULE", "CORRIDOR", "HUD", "OUTDOOR", "COURTYARD",
                    "TERRACE", "PLAZA", "UPPER_COURTYARD"]:
            if kw in name_upper:
                skip = True
                break
        if skip:
            continue

        name_bonus = 0
        for kw in ["LAB", "CLASSROOM", "EVENT", "STUDIO", "CONFERENCE",
                    "OFFICE", "ATRIUM", "LOBBY", "HOME", "LEARNING",
                    "PATHWAYS", "CAPSTONE", "SHOWCASE"]:
            if kw in name_upper:
                name_bonus = 5
                break

        candidates.append({
            "prim": room,
            "name": name,
            "center": ((mn[0] + mx[0]) / 2.0, (mn[1] + mx[1]) / 2.0, (mn[2] + mx[2]) / 2.0),
            "min": (float(mn[0]), float(mn[1]), float(mn[2])),
            "max": (float(mx[0]), float(mx[1]), float(mx[2])),
            "area": area,
            "score": area + name_bonus * 100,
        })

    if not candidates:
        return None

    candidates.sort(key=lambda c: -c["score"])

    # Log top candidates for debugging
    log(f"    Top room candidates (of {len(candidates)}):")
    for c in candidates[:5]:
        log(f"      {c['name']:<55s}  area={c['area']:.0f}  score={c['score']:.0f}")

    room = candidates[0]

    # Place camera at room center, biased toward building interior
    room_cx = (room["min"][0] + room["max"][0]) / 2.0
    room_cy = (room["min"][1] + room["max"][1]) / 2.0

    # Security camera style: top corner of the room, looking down diagonally
    # Position near one corner, slightly inset from walls (10%)
    cam_x = room["min"][0] + (room["max"][0] - room["min"][0]) * 0.10
    cam_y = room["min"][1] + (room["max"][1] - room["min"][1]) * 0.10

    if bldg_center_x is not None and bldg_center_y is not None:
        # Gentle 5% shift toward building center to stay inside building envelope
        cam_x = cam_x + (bldg_center_x - cam_x) * 0.05
        cam_y = cam_y + (bldg_center_y - cam_y) * 0.05

    # Near-ceiling height (90% of room height)
    room_z_min = room["min"][2]
    room_z_max = room["max"][2]
    room_height = room_z_max - room_z_min
    cam_z = room_z_min + room_height * 0.90

    # Look toward the opposite corner at floor level
    target_x = room["min"][0] + (room["max"][0] - room["min"][0]) * 0.85
    target_y = room["min"][1] + (room["max"][1] - room["min"][1]) * 0.85
    target_z = room_z_min + 1.0  # just above floor level

    return {
        "position": (cam_x, cam_y, cam_z),
        "target": (target_x, target_y, target_z),
        "room_name": room["name"],
    }


# ---------------------------------------------------------------------------
# Camera orientation helper
# ---------------------------------------------------------------------------

def look_at_orientation(cam_pos, target):
    """Compute quaternion to aim camera from cam_pos toward target."""
    dx = target[0] - cam_pos[0]
    dy = target[1] - cam_pos[1]
    dz = target[2] - cam_pos[2]
    yaw = math.degrees(math.atan2(dy, dx))
    pitch = math.degrees(math.atan2(-dz, math.sqrt(dx**2 + dy**2)))
    return rot_utils.euler_angles_to_quats(
        np.array([0, pitch, yaw]), degrees=True
    )


# ---------------------------------------------------------------------------
# Text overlay
# ---------------------------------------------------------------------------

def add_text_overlay(img, lines):
    """Add white text with dark background to a PIL Image."""
    from PIL import ImageDraw, ImageFont

    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 28)
    except (IOError, OSError):
        font = ImageFont.load_default()

    y = 20
    for line in lines:
        # Draw dark background rectangle
        bbox = draw.textbbox((20, y), line, font=font)
        draw.rectangle(
            [bbox[0] - 4, bbox[1] - 2, bbox[2] + 4, bbox[3] + 2],
            fill=(0, 0, 0, 180)
        )
        draw.text((20, y), line, fill=(255, 255, 255), font=font)
        y += 36

    return img


# ---------------------------------------------------------------------------
# Compute aim target from camera position toward floor center
# ---------------------------------------------------------------------------

def compute_camera_target(cam_pos, floor_center_x, floor_center_y):
    """Compute a look-at target from camera position toward floor center.

    Aims horizontally toward the floor center with a slight downward tilt.
    This avoids the Revit transform scale corruption issue.
    """
    # Look toward floor center, slightly below eye level
    target_x = floor_center_x
    target_y = floor_center_y
    target_z = cam_pos[2] - 2.0  # slight downward gaze
    return (target_x, target_y, target_z)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global _report_file
    _report_file = open(_report_path, "w")
    t_start = time.time()

    log("=" * 70)
    log("FLOOR CAMERA SCREENSHOTS — MSOEBuildingTake2")
    log("=" * 70)

    # --- Load building ---
    world = World(stage_units_in_meters=1.0)
    stage = omni.usd.get_context().get_stage()

    building_usd = os.path.join(
        _project_dir, "assets", "environments", "msoe_building",
        "MSOEBuildingTake2.usd"
    )
    log(f"\nLoading: {building_usd}")
    add_reference_to_stage(building_usd, prim_path="/World/Building")
    log("Building loaded.")

    building_prim = stage.GetPrimAtPath("/World/Building")

    # --- Building bounding box for interior bias ---
    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]
    )
    bldg_bbox = bbox_cache.ComputeWorldBound(building_prim)
    bldg_range = bldg_bbox.ComputeAlignedRange()
    bb_min = bldg_range.GetMin()
    bb_max = bldg_range.GetMax()
    bldg_cx = (bb_min[0] + bb_max[0]) / 2.0
    bldg_cy = (bb_min[1] + bb_max[1]) / 2.0
    log(f"Building bbox: ({bb_min[0]:.1f}, {bb_min[1]:.1f}, {bb_min[2]:.1f}) to "
        f"({bb_max[0]:.1f}, {bb_max[1]:.1f}, {bb_max[2]:.1f})")
    log(f"Building center XY: ({bldg_cx:.1f}, {bldg_cy:.1f})")

    # --- Lighting (generous for interior views) ---
    distant = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/Sun"))
    distant.CreateIntensityAttr(5000)
    distant.CreateAngleAttr(1.0)
    xform = UsdGeom.Xformable(distant.GetPrim())
    xform.AddRotateXYZOp().Set(Gf.Vec3f(-45, 30, 0))

    dome = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/DomeLight"))
    dome.CreateIntensityAttr(2000)

    log("Lighting added (DistantLight=5000, DomeLight=2000)")

    # --- Discover floor levels ---
    log("\n--- Discovering floor levels ---")
    floors = discover_floor_levels(stage)
    log(f"Found {len(floors)} floor levels:")
    for fl in floors:
        log(f"  {fl['name']:12s}  Z={fl['z']:.2f}  "
            f"center=({fl['center_x']:.1f}, {fl['center_y']:.1f})  "
            f"({fl['count']} slabs)")

    if len(floors) < 2:
        log("ERROR: Not enough floors found. Aborting.")
        _report_file.close()
        simulation_app.close()
        sys.exit(1)

    # Assign human-readable names to the first 4 floors
    floor_labels = ["G", "1", "2", "3"]
    target_floors = floors[:4]
    for i, fl in enumerate(target_floors):
        if i < len(floor_labels):
            fl["label"] = floor_labels[i]
        else:
            fl["label"] = fl["name"]

    log(f"\nTarget floors: {[f['label'] for f in target_floors]}")

    # --- Scan Revit cameras ---
    log("\n--- Scanning Revit cameras ---")
    cameras_prim = find_cameras_prim(stage, building_prim)
    revit_cameras = []
    if cameras_prim:
        revit_cameras = scan_revit_cameras(stage, cameras_prim)
        log(f"Found {len(revit_cameras)} Revit cameras:")
        for cam in revit_cameras:
            p = cam["position"]
            log(f"  {cam['name']:<45s}  pos=({p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f})")
    else:
        log("WARNING: No Cameras prim found. Using room-based fallback for all floors.")

    # --- Assign cameras to floors ---
    # Log Revit camera floor assignments for reference
    log("\n--- Revit camera floor assignments (for reference) ---")
    floor_cameras = assign_cameras_to_floors(revit_cameras, target_floors)
    for fl in target_floors:
        assigned = floor_cameras.get(fl["name"])
        if assigned:
            log(f"  Floor {fl['label']}: Revit '{assigned['name']}' "
                f"(score={assigned['score']}, z_above={assigned['z_above_floor']:.1f})")
        else:
            log(f"  Floor {fl['label']}: No Revit camera")

    # Use room-based placement (Revit cameras sit on wall surfaces and
    # produce wall-only views in Isaac Sim's physical renderer)
    log("\n--- Placing cameras inside rooms ---")
    camera_positions = []
    used_rooms = set()
    for fl in target_floors:
        fallback = find_room_camera_fallback(
            stage, building_prim, fl["z"], fl["center_x"], fl["center_y"],
            exclude_rooms=used_rooms,
            bldg_center_x=bldg_cx, bldg_center_y=bldg_cy,
        )
        if fallback:
            cam_info = {
                "floor": fl["label"],
                "floor_z": fl["z"],
                "position": fallback["position"],
                "target": fallback["target"],
                "source": f"Room: {fallback['room_name']}",
                "score": 0,
            }
            used_rooms.add(fallback["room_name"])
            log(f"  Floor {fl['label']}: {fallback['room_name']}  "
                f"pos=({fallback['position'][0]:.1f}, {fallback['position'][1]:.1f}, {fallback['position'][2]:.1f})")
        else:
            # Last resort: center of building at eye height
            cam_info = {
                "floor": fl["label"],
                "floor_z": fl["z"],
                "position": (fl["center_x"], fl["center_y"], fl["z"] + 5.5),
                "target": (fl["center_x"] + 15.0, fl["center_y"], fl["z"] + 4.0),
                "source": "Floor center (fallback)",
                "score": 0,
            }
            log(f"  Floor {fl['label']}: Using floor center as last resort")

        camera_positions.append(cam_info)

    # Manual adjustment: rotate Floor 3 camera slightly left so the
    # right-side door isn't cut off by the frame edge
    for cam_info in camera_positions:
        if cam_info["floor"] == "3":
            pos = cam_info["position"]
            t = cam_info["target"]
            # Shift target left in camera space: (-X, +Y) in world coords
            cam_info["target"] = (t[0] - 3.0, t[1] + 4.0, t[2])
            log(f"  Floor 3 target adjusted: ({t[0]:.1f},{t[1]:.1f}) -> "
                f"({cam_info['target'][0]:.1f},{cam_info['target'][1]:.1f})")

    # --- Create camera and capture screenshots ---
    log("\n--- Capturing screenshots ---")
    output_dir = os.path.join(_project_dir, "videos", "environment")
    os.makedirs(output_dir, exist_ok=True)

    # Create a single camera sensor at the first position
    first = camera_positions[0]
    first_pos = np.array(first["position"])
    first_orient = look_at_orientation(first_pos, first["target"])

    camera = Camera(
        prim_path="/World/FloorCamera",
        position=first_pos,
        frequency=30,
        resolution=(1920, 1080),
        orientation=first_orient,
    )

    # Wide-angle lens for security camera style (default is ~24mm)
    cam_prim = stage.GetPrimAtPath("/World/FloorCamera")
    UsdGeom.Camera(cam_prim).CreateFocalLengthAttr(12.0)

    world.reset()
    camera.initialize()

    log("Warming up renderer (60 frames)...")
    for _ in range(60):
        world.step(render=True)

    saved_files = []
    for idx, cam_info in enumerate(camera_positions):
        pos = np.array(cam_info["position"])
        orient = look_at_orientation(pos, cam_info["target"])
        camera.set_world_pose(position=pos, orientation=orient)

        # Add a point light near the camera for interior illumination
        light_path = f"/World/CamLight_{idx}"
        sphere_light = UsdLux.SphereLight.Define(stage, Sdf.Path(light_path))
        sphere_light.CreateIntensityAttr(30000)
        sphere_light.CreateRadiusAttr(1.0)
        UsdGeom.Xformable(sphere_light.GetPrim()).AddTranslateOp().Set(
            Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2]))
        )

        # Let renderer settle on new view
        for _ in range(20):
            world.step(render=True)

        rgba = camera.get_rgba()
        filename = f"floor_{cam_info['floor']}_camera.png"
        filepath = os.path.join(output_dir, filename)

        if rgba is not None and rgba.max() > 0:
            try:
                from PIL import Image
                img = Image.fromarray(rgba[:, :, :3])
                p = cam_info["position"]
                overlay_lines = [
                    f"Floor: Level {cam_info['floor']}",
                    f"Position: ({p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f})",
                    f"Source: {cam_info['source']}",
                ]
                img = add_text_overlay(img, overlay_lines)
                img.save(filepath)
                log(f"  Saved: {filename}  (pixels: {rgba.min()}-{rgba.max()})")
                saved_files.append(filepath)
            except ImportError:
                log(f"  WARNING: PIL not available, skipping text overlay for {filename}")
        else:
            log(f"  WARNING: Black frame for Floor {cam_info['floor']}")

    # --- Print JSON summary ---
    log("\n--- Camera Position Summary ---")
    summary = {"cameras": []}
    for cam_info in camera_positions:
        p = cam_info["position"]
        t = cam_info["target"]
        entry = {
            "floor": cam_info["floor"],
            "x": round(p[0], 2),
            "y": round(p[1], 2),
            "z": round(p[2], 2),
            "target_x": round(t[0], 2),
            "target_y": round(t[1], 2),
            "target_z": round(t[2], 2),
            "source": cam_info["source"],
        }
        summary["cameras"].append(entry)
        log(f"  Floor {entry['floor']}: pos=({entry['x']}, {entry['y']}, {entry['z']})  "
            f"target=({entry['target_x']}, {entry['target_y']}, {entry['target_z']})  "
            f"[{entry['source']}]")

    log(f"\n{json.dumps(summary, indent=2)}")

    t_end = time.time()
    log(f"\nDone! {len(saved_files)} screenshots saved. Runtime: {t_end - t_start:.1f}s")
    _report_file.close()
    simulation_app.close()


if __name__ == "__main__":
    main()

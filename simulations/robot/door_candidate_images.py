#!/usr/bin/env python3
"""
door_candidate_images.py — Render one CLOSED image per candidate door.

Loads MSOEBuildingV4_physics.usd once and iterates a filtered set of
candidate doors drawn from door_catalog.json.  For each door it:

  1. Chooses the unoccluded hallway side via PhysX raycast.
  2. Rotates the door to CLOSED (0°) using the correct method:
       • mesh_count > 0 → world-space rotation (curtain-wall, like B1)
       • mesh_count == 0 → local-space rotation after SetInstanceable(False) (like B2)
  3. Renders one JPEG at 1024×768.

Filter criteria
---------------
  - Floors 2–5 only (1 = basement, 6 = top level excluded)
  - is_double_door = False
  - One of six target families (see INCLUDE_FAMILIES)
  - Capped at 5 candidates per family (keeps total ≈ 30 images)

Output
------
  data/door_candidates/<family_short>_<door_id>_floor<N>_CLOSED.jpg
  data/door_candidates_summary.json

Run
---
  sbatch submit_isaac_job.sh simulations/robot/door_candidate_images.py
"""

import os
import sys
import math
import json
import functools
from collections import Counter

import numpy as np

print = functools.partial(print, flush=True)   # noqa: A001

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CAMERA_DIST    = 2.0      # metres from door centre along approach axis
CAMERA_HEIGHT  = 1.2      # metres above door bottom edge (eye-level proxy)
CAM_RESOLUTION = (1024, 768)
WARMUP_STEPS   = 10       # render steps per door after repositioning
STARTUP_WARMUP = 30       # cold-start renderer steps

MAX_PER_FAMILY = None     # no cap — render every door that passes filters
CLOSED_ANGLE   = 0.0      # degrees — always render CLOSED

# Six target families and their short labels for filenames.
INCLUDE_FAMILIES = {
    "RS___Curtain_Wall___Type_FG___New": "CW_FG_New",
    "RS___Curtain_Wall___Type_F___NEW":  "CW_F_NEW",
    "RS___Type_F_Door":                  "Type_F",
    "RS___Type_F_W_SIDELITE_Door":       "Type_F_Sidelite",
    "RS___Type_FG_Door":                 "Type_FG",
    "RS___Type_N_Door":                  "Type_N",
}

# Families excluded outright regardless of is_double_door flag.
EXCLUDE_FAMILIES = {
    "RS___Type_2F_Door",
    "RS___Type_2F___Dual_Door",
    "RS___Type_2FG_Door",
    "RS___Type_2G_Door",
    "RS___Type_2FU_Door",
    "RS___Curtain_Wall___Type_2FG___NEW",
    "AV300_RAYNOR_GARAGE_DOOR",
    "RS___Overhead_Rolling_Door",
    "RS___Overhead_Sectional_Door",
    "Dormakaba_Sliding_Glass_Wall_FSW",
}

_script_dir  = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.abspath(os.path.join(_script_dir, "..", ".."))
CATALOG_PATH = os.path.join(_script_dir, "door_catalog.json")
USD_PATH     = os.path.join(_project_dir, "MSOEBuildingV4_physics.usd")
OUT_DIR      = os.path.join(_project_dir, "data", "door_candidates")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Isaac Sim startup
# ---------------------------------------------------------------------------

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

import time as _time
_t0 = _time.time()

def _ts(msg):
    print(f"[CAND] [+{_time.time() - _t0:6.1f}s] {msg}")

_ts("Importing SimulationApp ...")
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})
_ts("SimulationApp ready")

import carb
_s = carb.settings.get_settings()
_s.set("/rtx/post/aa/op", 4)
_s.set("/rtx/post/dlss/execMode", 0)

import omni.usd
from pxr import Usd, UsdLux, UsdGeom, Gf, Sdf
import isaacsim.core.utils.numpy.rotations as rot_utils
from isaacsim.core.api import World
from isaacsim.sensors.camera import Camera
import omni.replicator.core as rep

try:
    from PIL import Image as _PIL
    _pil_ok = True
except ImportError:
    _pil_ok = False
    print("[WARN] PIL unavailable — saving .npy instead of .jpg")

# ---------------------------------------------------------------------------
# Load catalog and apply filters
# ---------------------------------------------------------------------------

with open(CATALOG_PATH) as f:
    _cat = json.load(f)

# Step 1: apply hard filters
_candidates_all = [
    d for d in _cat["doors"]
    if d["floor"] in (2, 3, 4, 5)
    and not d.get("is_double_door", False)
    and d["door_type_family"] in INCLUDE_FAMILIES
    and d["door_type_family"] not in EXCLUDE_FAMILIES
]

# Step 2: sort for reproducible ordering (floor, then door_id)
doors_to_run = sorted(_candidates_all, key=lambda d: (d["floor"], d["door_type_family"], d["id"]))

_ts(f"Catalog loaded — {len(doors_to_run)} candidate doors (all passing filters, floors 2–5)")

for fam, cnt in sorted(Counter(d["door_type_family"] for d in doors_to_run).items(),
                       key=lambda x: -x[1]):
    short = INCLUDE_FAMILIES[fam]
    print(f"  {cnt:2d}  {short:20s}  ({fam})")

# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------

world = World(stage_units_in_meters=1.0)
stage = omni.usd.get_context().get_stage()

_ts(f"Inserting building USD: {USD_PATH}")
stage.GetRootLayer().subLayerPaths.insert(0, USD_PATH)
_ts("Building USD inserted")

# Neutral dome light — illuminates from all directions uniformly.
dome = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/DomeLight"))
dome.CreateIntensityAttr(40_000)
dome.GetColorAttr().Set(Gf.Vec3f(1.0, 1.0, 1.0))

# Single rect fill light repositioned 1.5 m above each door centre.
fill = UsdLux.RectLight.Define(stage, Sdf.Path("/World/FillLight"))
fill.CreateIntensityAttr(300_000)
fill.CreateWidthAttr(1.2)
fill.CreateHeightAttr(1.2)
_fill_xf       = UsdGeom.Xformable(fill.GetPrim())
_fill_translate = _fill_xf.AddTranslateOp()
_fill_rotate    = _fill_xf.AddRotateXYZOp()
_fill_rotate.Set(Gf.Vec3f(180.0, 0.0, 0.0))   # face downward

_ts("world.reset() ...")
world.reset()
_ts("world.reset() done")
for _ in range(5):
    world.step(render=True)

# ---------------------------------------------------------------------------
# Camera — single prim repositioned per door
# ---------------------------------------------------------------------------

cam = Camera(
    prim_path="/World/CandidateCam",
    position=np.array([0.0, 0.0, 5.0]),
    frequency=30,
    resolution=CAM_RESOLUTION,
    orientation=rot_utils.euler_angles_to_quats(
        np.array([0.0, 0.0, 180.0]), degrees=True),
)
simulation_app.update()
cam.initialize()

usd_cam_prim = UsdGeom.Camera(stage.GetPrimAtPath("/World/CandidateCam"))
usd_cam_prim.GetFocalLengthAttr().Set(5.0)
usd_cam_prim.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 10000.0))

rp  = rep.create.render_product("/World/CandidateCam", CAM_RESOLUTION)
ann = rep.AnnotatorRegistry.get_annotator("rgb")
ann.attach([rp])

_ts(f"Renderer cold-start warmup ({STARTUP_WARMUP} steps) ...")
for _ in range(STARTUP_WARMUP):
    world.step(render=True)
_ts("Warmup done")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _door_matrix(ox, oy, oz, angle_deg, hx, hy) -> Gf.Matrix4d:
    """4×4 rotation matrix (world-space for curtain-wall, local-space for instanced)."""
    c, s = math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg))
    dx, dy = ox - hx, oy - hy
    return Gf.Matrix4d(
         c,  s, 0, 0,
        -s,  c, 0, 0,
         0,  0, 1, 0,
        c * dx - s * dy + hx,
        s * dx + c * dy + hy,
        oz, 1,
    )


def _set_angle(mesh_ops, angle_deg, hx, hy):
    for m, op in mesh_ops:
        op.Set(_door_matrix(m["ox"], m["oy"], m["oz"], angle_deg, hx, hy))


def _cam_pose(cx, cy, cz, approach_axis, side):
    """Return (position_np, yaw_deg) for the camera on the given side."""
    if approach_axis == "X":
        pos = np.array([cx + side * CAMERA_DIST, cy, cz])
        yaw = 180.0 if side > 0 else 0.0
    else:   # Y
        pos = np.array([cx, cy + side * CAMERA_DIST, cz])
        yaw = 270.0 if side > 0 else 90.0
    return pos, yaw


def _los_clear(cam_pos, door_centre, door_path) -> bool:
    """Return True if raycast from cam_pos hits the door (not a wall)."""
    try:
        from omni.physx import get_physx_scene_query_interface
        physx = get_physx_scene_query_interface()
    except Exception:
        return True   # PhysX unavailable — assume clear
    direction = np.array(door_centre) - cam_pos
    dist = float(np.linalg.norm(direction))
    if dist < 0.01:
        return True
    direction /= dist
    hit = physx.raycast_closest(
        tuple(float(v) for v in cam_pos),
        tuple(float(v) for v in direction),
        dist,
    )
    if hit["hit"]:
        return door_path in (hit.get("rigidBody", "") or "")
    return True   # no obstruction at all — open corridor


def _reposition_camera(pos, yaw_deg):
    ori = rot_utils.euler_angles_to_quats(
        np.array([0.0, 0.0, yaw_deg]), degrees=True)
    cam.set_world_pose(position=pos, orientation=ori)


def _capture_jpeg(family_short, door_id, floor, side) -> bool:
    world.step(render=True)
    data = ann.get_data()
    if data is None:
        print(f"  [WARN] annotator returned None — {door_id}")
        return False
    arr = np.asarray(data)
    if arr.ndim != 3 or arr.shape[-1] < 3:
        print(f"  [WARN] bad rgb shape {arr.shape} — {door_id}")
        return False
    rgb      = arr[:, :, :3].astype(np.uint8)
    side_tag = "sideP" if side > 0 else "sideN"
    fname    = f"{family_short}_{door_id}_floor{floor}_{side_tag}_CLOSED.jpg"
    path     = os.path.join(OUT_DIR, fname)
    if _pil_ok:
        _PIL.fromarray(rgb).save(path, "JPEG", quality=88)
    else:
        np.save(path.replace(".jpg", ".npy"), rgb)
    return True

# ---------------------------------------------------------------------------
# Per-door loop
# ---------------------------------------------------------------------------

summary      = []
_saved_jpegs = 0        # running count of saved JPEGs
_current_floor = None   # track floor changes for banner prints

for di, door in enumerate(doors_to_run):
    door_id    = door["id"]
    prim_path  = door["prim_path"]
    family     = door["door_type_family"]
    fam_short  = INCLUDE_FAMILIES[family]
    floor      = door["floor"]
    pos        = door["position"]
    dim        = door["dimensions"]
    approach   = dim.get("approach_axis", "Y")
    mesh_count = door.get("mesh_count", 0)

    cx_m = pos["center_x"]
    cy_m = pos["center_y"]
    cz_m = pos["center_z"]

    floor_z_m = cz_m - dim["height"] / 2.0
    cam_z_m   = floor_z_m + CAMERA_HEIGHT

    # Floor-change banner
    if floor != _current_floor:
        _current_floor = floor
        remaining = len(doors_to_run) - di
        print(f"\n{'='*60}")
        print(f"  FLOOR {floor}  —  starting at door {di+1}/{len(doors_to_run)} "
              f"({remaining} doors remaining in job)")
        print(f"{'='*60}")

    pct = (di + 1) / len(doors_to_run) * 100
    _ts(f"[{di+1:3d}/{len(doors_to_run)}] ({pct:5.1f}%)  [{fam_short:20s}] {door_id}  "
        f"floor={floor}  approach={approach}  mesh_count={mesh_count}")

    entry = {
        "door_id":      door_id,
        "family":       family,
        "family_short": fam_short,
        "floor":        floor,
        "status":       "ok",
        "los_hint":     None,   # side where PhysX raycast was clear (likely hallway)
        "sides_ok":     [],     # sides that produced a saved image
        "note":         "",
    }

    # -- Validate prim --
    door_prim = stage.GetPrimAtPath(prim_path)
    if not door_prim.IsValid():
        print(f"  SKIP — prim not valid at {prim_path}")
        entry.update({"status": "skipped", "note": "prim not valid"})
        summary.append(entry)
        continue

    # -----------------------------------------------------------------------
    # Discover meshes and compute hinge — two paths based on mesh_count
    # -----------------------------------------------------------------------

    mesh_ops = []   # list of ({"ox","oy","oz"}, xformOp)

    if mesh_count > 0:
        # ----- WORLD-SPACE path (curtain-wall, like B1) -----
        # Ensure prim is concretized if instanced (belt-and-suspenders).
        if door_prim.IsInstance() or door_prim.IsInstanceable():
            door_prim.SetInstanceable(False)
            simulation_app.update()

        for child in door_prim.GetChildren():
            if not child.IsA(UsdGeom.Mesh):
                continue
            xf = UsdGeom.Xformable(child)
            op = next((o for o in xf.GetOrderedXformOps()
                       if o.GetOpName() == "xformOp:transform"), None)
            if op is None:
                continue
            mat = op.Get()
            if mat is None:
                continue
            mesh_ops.append((
                {"ox": mat[3][0], "oy": mat[3][1], "oz": mat[3][2]},
                op,
            ))

        if not mesh_ops:
            print(f"  SKIP — no direct Mesh children with xformOp:transform")
            entry.update({"status": "skipped",
                          "note": "no xform mesh children (curtain-wall path)"})
            summary.append(entry)
            continue

        # Hinge from first mesh child: translation + extent_min on Y axis
        mc         = door["mesh_children"][0]
        hinge_x_cm = mc["translation_cm"][0]
        hinge_y_cm = mc["translation_cm"][1] + mc["extent_min"][1]

    else:
        # ----- LOCAL-SPACE path (instanced, like B2) -----
        if door_prim.IsInstance() or door_prim.IsInstanceable():
            door_prim.SetInstanceable(False)
            simulation_app.update()

        for p in Usd.PrimRange(door_prim):
            if not p.IsA(UsdGeom.Mesh):
                continue
            xf = UsdGeom.Xformable(p)
            op = next((o for o in xf.GetOrderedXformOps()
                       if o.GetOpName() == "xformOp:transform"), None)
            if op is None:
                continue
            mat = op.Get()
            if mat is None:
                continue
            # mat[3] = (lx, ly, lz, 1) — local-space origin
            mesh_ops.append((
                {"ox": mat[3][0], "oy": mat[3][1], "oz": mat[3][2]},
                op,
            ))

        if not mesh_ops:
            print(f"  SKIP — no xformOp:transform meshes after concretizing")
            entry.update({"status": "skipped",
                          "note": "no xform meshes after SetInstanceable(False)"})
            summary.append(entry)
            continue

        # Local-space hinge: min edge of the wide axis
        half_width_cm = (dim["width"] * 100.0) / 2.0
        if approach == "Y":
            hinge_x_cm = -half_width_cm
            hinge_y_cm = 0.0
        else:
            hinge_x_cm = 0.0
            hinge_y_cm = -half_width_cm

    print(f"  {len(mesh_ops)} mesh(es)  hinge=({hinge_x_cm:.1f},{hinge_y_cm:.1f}) cm")

    # -- Ensure door is at CLOSED position before rendering --
    _set_angle(mesh_ops, CLOSED_ANGLE, hinge_x_cm, hinge_y_cm)

    # -- Reposition fill light 1.5 m above door centre --
    _fill_translate.Set(Gf.Vec3d(cx_m, cy_m, cam_z_m + 1.5))

    # -- Render from BOTH sides so the user can inspect hallway vs. interior --
    # The los_hint records which side PhysX thinks is clear (likely the hallway
    # side), but we render both regardless so the user can confirm visually.
    door_centre_3d = [cx_m, cy_m, cam_z_m]

    # Find the raycast-clear side first (just as a hint for the summary JSON).
    for side in [1, -1]:
        cpos, _ = _cam_pose(cx_m, cy_m, cam_z_m, approach, side)
        if _los_clear(cpos, door_centre_3d, prim_path):
            entry["los_hint"] = int(side)
            break

    # Render from each side.
    any_ok = False
    for side in [1, -1]:
        cpos, yaw = _cam_pose(cx_m, cy_m, cam_z_m, approach, side)
        _reposition_camera(cpos, yaw)
        for _ in range(WARMUP_STEPS):
            world.step(render=True)
        ok = _capture_jpeg(fam_short, door_id, floor, side)
        side_tag = "sideP" if side > 0 else "sideN"
        print(f"  {side_tag}  {'saved' if ok else 'FAILED'}"
              f"{'  <- los_hint' if side == entry['los_hint'] else ''}")
        if ok:
            entry["sides_ok"].append(int(side))
            _saved_jpegs += 1
            any_ok = True

    if not any_ok:
        entry["status"] = "partial"
        entry["note"]   = "all captures failed"

    summary.append(entry)

    # Running totals after each door
    doors_done    = di + 1
    doors_left    = len(doors_to_run) - doors_done
    skipped_so_far = sum(1 for e in summary if e["status"] == "skipped")
    _ts(f"  → {doors_done}/{len(doors_to_run)} doors processed  "
        f"{_saved_jpegs} JPEGs saved  "
        f"{skipped_so_far} skipped  "
        f"{doors_left} remaining")

# ---------------------------------------------------------------------------
# Summary JSON
# ---------------------------------------------------------------------------

ok_count   = sum(1 for e in summary if e["status"] == "ok")
skip_count = sum(1 for e in summary if e["status"] == "skipped")
fail_count = sum(1 for e in summary if e["status"] == "partial")

summary_path = os.path.join(_project_dir, "data", "door_candidates_summary.json")
with open(summary_path, "w") as f:
    json.dump({
        "ok": ok_count,
        "skipped": skip_count,
        "failed": fail_count,
        "doors": summary,
    }, f, indent=2)
    f.flush()
    os.fsync(f.fileno())

total_jpegs = sum(len(e.get("sides_ok", [])) for e in summary)
_ts(f"Done — {ok_count} doors OK ({total_jpegs} JPEGs)  {skip_count} skipped  {fail_count} failed")
_ts(f"JPEGs   → {OUT_DIR}/")
_ts(f"Summary → {summary_path}")

world.stop()
simulation_app.close()

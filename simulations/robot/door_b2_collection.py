#!/usr/bin/env python3
"""
B2 — Multi-door instanced-door inspection run.

Loads MSOEBuildingV4_physics.usd once, then iterates every single-leaf
instanced door (mesh_count == 0 in the catalog) after calling
SetInstanceable(False) to expose the concretized subtree.

Key difference from B1 (curtain-wall doors):
  * Instanced door child meshes have xformOp:transform in LOCAL space
    (relative to the parent instance prim), not world space.  Rotation is
    therefore applied in local coordinates around a local-space hinge.
  * Hinge is estimated as the min-edge of the wide axis in local space:
      approach_axis=Y → hinge at local x = −(width/2) cm
      approach_axis=X → hinge at local y = −(width/2) cm
    (Both edges are tried during collection; the user inspects which
    opening direction looks correct and can lock it per family later.)

Families included (single-leaf, non-double, non-special):
    RS___Type_F_Door            solid HM swing door
    RS___Type_F_W_SIDELITE_Door door + fixed sidelite (whole unit rotated —
                                 frame/sidelite rotation is approximate for
                                 inspection; only the door leaf matters)
    RS___Type_FG_Door           glass-panel swing door
    RS___Type_N_Door            narrow swing door
    RS___Type_G_Door            full-glass swing door

Excluded:
    Type_2F / Type_2FG / Type_2G / Type_2FU  double-door families
    AV300_RAYNOR_GARAGE_DOOR / Overhead / Sliding  non-swing types
    is_double_door=True                            visually confirmed pairs

Output
------
    data/door_b2_jpegs/<door_id>_CLOSED_ang0000.0.jpg
    data/door_b2_jpegs/<door_id>_OPEN_ang0045.0.jpg
    data/door_b2_summary.json

Run
---
    sbatch submit_isaac_job.sh simulations/robot/door_b2_collection.py
"""

import os
import sys
import math
import json
import functools
import numpy as np

print = functools.partial(print, flush=True)   # noqa: A001

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CAMERA_DIST    = 2.0     # metres from door centre along approach axis
CAMERA_HEIGHT  = 1.2     # metres above door bottom edge
CAM_RESOLUTION = (1024, 768)
CLOSED_MAX_DEG = 5.0
ANGLE_GRID     = [0.0, 45.0]
WARMUP_STEPS   = 10
STARTUP_WARMUP = 30

# Families that are definitively double-door or non-swing — excluded outright
# regardless of is_double_door flag.
DOUBLE_FAMILIES = {
    "RS___Type_2F_Door",
    "RS___Type_2F___Dual_Door",
    "RS___Type_2FG_Door",
    "RS___Type_2G_Door",
    "RS___Type_2FU_Door",
}
SKIP_FAMILIES = {
    "AV300_RAYNOR_GARAGE_DOOR",
    "RS___Overhead_Rolling_Door",
    "RS___Overhead_Sectional_Door",
    "Dormakaba_Sliding_Glass_Wall_FSW",
}

_script_dir  = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.abspath(os.path.join(_script_dir, "..", ".."))
CATALOG_PATH = os.path.join(_script_dir, "door_catalog.json")
USD_PATH     = os.path.join(_project_dir, "MSOEBuildingV4_physics.usd")
OUT_DIR      = os.path.join(_project_dir, "data", "door_b2_jpegs")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Isaac Sim startup
# ---------------------------------------------------------------------------

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

import time as _time
_t0 = _time.time()
def _ts(msg):
    print(f"[B2] [+{_time.time() - _t0:6.1f}s] {msg}")

_ts("Importing SimulationApp ...")
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})
_ts("SimulationApp ready")

import carb
_s = carb.settings.get_settings()
_s.set("/rtx/post/aa/op", 4)
_s.set("/rtx/post/dlss/execMode", 0)

import omni.usd
from pxr import Usd, UsdLux, UsdGeom, UsdShade, Gf, Sdf
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
# Load catalog — instanced single-leaf doors only
# ---------------------------------------------------------------------------

with open(CATALOG_PATH) as f:
    _cat = json.load(f)

doors_to_run = [
    d for d in _cat["doors"]
    if d.get("mesh_count", 0) == 0                        # instanced (B2 target)
    and not d.get("is_double_door", False)                 # not a confirmed double
    and d["door_type_family"] not in DOUBLE_FAMILIES       # not a double family
    and d["door_type_family"] not in SKIP_FAMILIES         # not a garage/overhead
]
_ts(f"Catalog loaded — {len(doors_to_run)} single-leaf instanced doors")

# Family breakdown for the log
from collections import Counter
for fam, cnt in sorted(Counter(d["door_type_family"] for d in doors_to_run).items(),
                       key=lambda x: -x[1]):
    print(f"  {cnt:4d}  {fam}")

# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------

world = World(stage_units_in_meters=1.0)
stage = omni.usd.get_context().get_stage()

_ts(f"Inserting building USD: {USD_PATH}")
stage.GetRootLayer().subLayerPaths.insert(0, USD_PATH)
_ts("Building USD inserted")

dome = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/DomeLight"))
dome.CreateIntensityAttr(40_000)
dome.GetColorAttr().Set(Gf.Vec3f(1.0, 1.0, 1.0))

fill = UsdLux.RectLight.Define(stage, Sdf.Path("/World/FillLight"))
fill.CreateIntensityAttr(300_000)
fill.CreateWidthAttr(1.2)
fill.CreateHeightAttr(1.2)
_fill_xf = UsdGeom.Xformable(fill.GetPrim())
_fill_translate = _fill_xf.AddTranslateOp()
_fill_rotate    = _fill_xf.AddRotateXYZOp()
_fill_rotate.Set(Gf.Vec3f(180.0, 0.0, 0.0))

_ts("world.reset() ...")
world.reset()
_ts("world.reset() done")
for _ in range(5):
    world.step(render=True)

# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------

cam = Camera(
    prim_path="/World/B2Cam",
    position=np.array([0.0, 0.0, 5.0]),
    frequency=30,
    resolution=CAM_RESOLUTION,
    orientation=rot_utils.euler_angles_to_quats(
        np.array([0.0, 0.0, 180.0]), degrees=True),
)
simulation_app.update()
cam.initialize()

usd_cam_prim = UsdGeom.Camera(stage.GetPrimAtPath("/World/B2Cam"))
usd_cam_prim.GetFocalLengthAttr().Set(5.0)
usd_cam_prim.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 10000.0))

rp  = rep.create.render_product("/World/B2Cam", CAM_RESOLUTION)
ann = rep.AnnotatorRegistry.get_annotator("rgb")
ann.attach([rp])

_ts(f"Renderer cold-start warmup ({STARTUP_WARMUP} steps) ...")
for _ in range(STARTUP_WARMUP):
    world.step(render=True)
_ts("Warmup done")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _local_door_matrix(lx, ly, lz, angle_deg, hinge_lx, hinge_ly) -> Gf.Matrix4d:
    """Rotation matrix in LOCAL space around (hinge_lx, hinge_ly).
    The child mesh xformOp:transform is in local (parent-relative) space for
    instanced doors, so we rotate here rather than in world space."""
    theta = math.radians(angle_deg)
    c, s = math.cos(theta), math.sin(theta)
    dx, dy = lx - hinge_lx, ly - hinge_ly
    return Gf.Matrix4d(
         c,  s, 0, 0,
        -s,  c, 0, 0,
         0,  0, 1, 0,
        c * dx - s * dy + hinge_lx,
        s * dx + c * dy + hinge_ly,
        lz, 1,
    )


def set_door_angle_local(mesh_ops, angle_deg, hinge_lx, hinge_ly):
    for m, op in mesh_ops:
        op.Set(_local_door_matrix(
            m["lx"], m["ly"], m["lz"], angle_deg, hinge_lx, hinge_ly))


def _cam_pose(cx, cy, cz, approach_axis, side):
    if approach_axis == "X":
        pos = np.array([cx + side * CAMERA_DIST, cy, cz])
        yaw = 180.0 if side > 0 else 0.0
    else:
        pos = np.array([cx, cy + side * CAMERA_DIST, cz])
        yaw = 270.0 if side > 0 else 90.0
    return pos, yaw


def _los_clear(cam_pos, door_centre, door_path) -> bool:
    try:
        from omni.physx import get_physx_scene_query_interface
        physx = get_physx_scene_query_interface()
    except Exception:
        return True
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
    return True


def reposition_camera(pos, yaw_deg):
    ori = rot_utils.euler_angles_to_quats(
        np.array([0.0, 0.0, yaw_deg]), degrees=True)
    cam.set_world_pose(position=pos, orientation=ori)


def capture_jpeg(door_id, angle_deg) -> bool:
    world.step(render=True)
    data = ann.get_data()
    if data is None:
        return False
    arr = np.asarray(data)
    if arr.ndim != 3 or arr.shape[-1] < 3:
        return False
    rgb  = arr[:, :, :3].astype(np.uint8)
    cls  = "CLOSED" if angle_deg <= CLOSED_MAX_DEG else "OPEN"
    fname = f"{door_id}_{cls}_ang{angle_deg:06.1f}.jpg"
    path  = os.path.join(OUT_DIR, fname)
    if _pil_ok:
        _PIL.fromarray(rgb).save(path, "JPEG", quality=88)
    else:
        np.save(path.replace(".jpg", ".npy"), rgb)
    return True

# ---------------------------------------------------------------------------
# Per-door loop
# ---------------------------------------------------------------------------

summary = []

for di, door in enumerate(doors_to_run):
    door_id   = door["id"]
    prim_path = door["prim_path"]
    pos       = door["position"]
    dim       = door["dimensions"]
    approach  = dim.get("approach_axis", "Y")

    cx_m = pos["center_x"]
    cy_m = pos["center_y"]
    cz_m = pos["center_z"]

    floor_z_m = cz_m - dim["height"] / 2.0
    cam_z_m   = floor_z_m + CAMERA_HEIGHT

    # Local-space hinge estimate: min edge of the wide axis.
    # approach_axis=Y → wide axis=X in local space → hinge at local x=−half_width
    # approach_axis=X → wide axis=Y in local space → hinge at local y=−half_width
    # Coordinates in cm (USD Revit export units).
    half_width_cm = (dim["width"] * 100.0) / 2.0
    if approach == "Y":
        hinge_lx = -half_width_cm
        hinge_ly = 0.0
    else:
        hinge_lx = 0.0
        hinge_ly = -half_width_cm

    _ts(f"[{di+1:3d}/{len(doors_to_run)}] {door_id}  floor={door['floor']}  "
        f"family={door['door_type_family'][-20:]}  approach={approach}  "
        f"hinge_local=({hinge_lx:.1f},{hinge_ly:.1f}) cm")

    entry = {"door_id": door_id, "floor": door["floor"],
             "family": door["door_type_family"],
             "status": "ok", "camera_side": None, "note": ""}

    # -- Get prim --
    door_prim = stage.GetPrimAtPath(prim_path)
    if not door_prim.IsValid():
        print(f"  SKIP — prim not valid")
        entry.update({"status": "skipped", "note": "prim not valid"})
        summary.append(entry)
        continue

    # -- Concretize instanced subtree --
    if door_prim.IsInstance() or door_prim.IsInstanceable():
        door_prim.SetInstanceable(False)
        simulation_app.update()

    # -- Walk subtree for meshes with xformOp:transform --
    mesh_ops = []
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
        # mat[3] = (lx, ly, lz, 1) in local space
        mesh_ops.append((
            {"path": str(p.GetPath()),
             "lx": mat[3][0], "ly": mat[3][1], "lz": mat[3][2]},
            op,
        ))

    if not mesh_ops:
        print(f"  SKIP — no xformOp:transform meshes after concretizing")
        entry.update({"status": "skipped", "note": "no xform meshes after SetInstanceable(False)"})
        summary.append(entry)
        continue

    print(f"  {len(mesh_ops)} mesh(es) found")

    # -- Reposition fill light --
    _fill_translate.Set(Gf.Vec3d(cx_m, cy_m, cam_z_m + 1.5))

    # -- Find unoccluded camera side --
    door_centre_3d = [cx_m, cy_m, cam_z_m]
    chosen_pos, chosen_yaw = None, None
    for side in [1, -1]:
        cpos, yaw = _cam_pose(cx_m, cy_m, cam_z_m, approach, side)
        if _los_clear(cpos, door_centre_3d, prim_path):
            chosen_pos, chosen_yaw = cpos, yaw
            entry["camera_side"] = int(side)
            break

    if chosen_pos is None:
        print(f"  SKIP — both camera sides occluded")
        entry.update({"status": "skipped", "note": "occluded both sides"})
        summary.append(entry)
        continue

    reposition_camera(chosen_pos, chosen_yaw)

    # -- Render --
    any_ok = False
    for angle_deg in ANGLE_GRID:
        set_door_angle_local(mesh_ops, angle_deg, hinge_lx, hinge_ly)
        for _ in range(WARMUP_STEPS):
            world.step(render=True)
        ok = capture_jpeg(door_id, angle_deg)
        if ok:
            any_ok = True
        cls = "CLOSED" if angle_deg <= CLOSED_MAX_DEG else "OPEN"
        print(f"  {cls:6s} {angle_deg:5.1f}°  {'saved' if ok else 'FAILED'}")

    # Reset to closed
    set_door_angle_local(mesh_ops, 0.0, hinge_lx, hinge_ly)

    if not any_ok:
        entry["status"] = "partial"
        entry["note"]   = "all captures failed"
    summary.append(entry)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

ok_count      = sum(1 for e in summary if e["status"] == "ok")
partial_count = sum(1 for e in summary if e["status"] == "partial")
skip_count    = sum(1 for e in summary if e["status"] == "skipped")

summary_path = os.path.join(_project_dir, "data", "door_b2_summary.json")
with open(summary_path, "w") as f:
    json.dump({"ok": ok_count, "partial": partial_count,
               "skipped": skip_count, "doors": summary}, f, indent=2)

_ts(f"Done — {ok_count} OK  {partial_count} partial  {skip_count} skipped")
_ts(f"JPEGs   → {OUT_DIR}/")
_ts(f"Summary → {summary_path}")

world.stop()
simulation_app.close()

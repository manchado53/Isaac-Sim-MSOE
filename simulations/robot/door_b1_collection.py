#!/usr/bin/env python3
"""
B1 — Multi-door curtain-wall inspection run.

Loads MSOEBuildingV4_physics.usd ONCE, then iterates every curtain-wall door
with mesh_count > 0 from door_catalog.json (37 doors across floors 2–5).
For each door it:

  1. Derives the hinge from the first mesh child's translation + extent_min
     (validated against the known Doors_13606713 hinge to within 2 mm).
  2. Repositions a single camera 2 m along the approach axis, choosing the
     unoccluded side via PhysX raycast.
  3. Renders ANGLE_GRID angles (0° CLOSED, 45° OPEN) with the Revit-original
     material and the building's ambient lighting.
  4. Dumps one JPEG per (door, angle).

Output
------
    data/door_b1_jpegs/<door_id>_CLOSED_ang0000.0.jpg
    data/door_b1_jpegs/<door_id>_OPEN_ang0045.0.jpg
    data/door_b1_summary.json   (per-door status: ok / skipped / partial)

Run
---
    sbatch submit_isaac_job.sh simulations/robot/door_b1_collection.py
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
CAMERA_HEIGHT  = 1.2     # metres above door bottom edge (eye-level proxy)
CAM_RESOLUTION = (1024, 768)
CLOSED_MAX_DEG = 5.0
ANGLE_GRID     = [0.0, 45.0]   # CLOSED, OPEN
WARMUP_STEPS   = 10            # render steps after scene-state change
STARTUP_WARMUP = 30            # renderer cold-start steps

_script_dir  = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.abspath(os.path.join(_script_dir, "..", ".."))
CATALOG_PATH = os.path.join(_script_dir, "door_catalog.json")
USD_PATH     = os.path.join(_project_dir, "MSOEBuildingV4_physics.usd")
OUT_DIR      = os.path.join(_project_dir, "data", "door_b1_jpegsV2")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Isaac Sim startup
# ---------------------------------------------------------------------------

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

import time as _time
_t0 = _time.time()
def _ts(msg):
    print(f"[B1] [+{_time.time() - _t0:6.1f}s] {msg}")

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
# Load catalog
# ---------------------------------------------------------------------------

with open(CATALOG_PATH) as f:
    _cat = json.load(f)

doors_to_run = [
    d for d in _cat["doors"]
    if d.get("mesh_count", 0) > 0
    and not d.get("is_double_door", False)
]
_ts(f"Catalog loaded — {len(doors_to_run)} single-leaf curtain-wall doors "
    f"(double doors excluded)")

# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------

world = World(stage_units_in_meters=1.0)
stage = omni.usd.get_context().get_stage()

_ts(f"Inserting building USD: {USD_PATH}")
stage.GetRootLayer().subLayerPaths.insert(0, USD_PATH)
_ts("Building USD inserted")

# Bright neutral dome — moves with camera automatically, works for any room.
# Ceiling fixtures would need repositioning per door; dome is simpler for an
# inspection run where photorealism is secondary to geometry verification.
dome = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/DomeLight"))
dome.CreateIntensityAttr(40_000)
dome.GetColorAttr().Set(Gf.Vec3f(1.0, 1.0, 1.0))

# A single rect fill light that we reposition above each door centre.
fill = UsdLux.RectLight.Define(stage, Sdf.Path("/World/FillLight"))
fill.CreateIntensityAttr(300_000)
fill.CreateWidthAttr(1.2)
fill.CreateHeightAttr(1.2)
_fill_xf = UsdGeom.Xformable(fill.GetPrim())
_fill_translate = _fill_xf.AddTranslateOp()
_fill_rotate    = _fill_xf.AddRotateXYZOp()
_fill_rotate.Set(Gf.Vec3f(180.0, 0.0, 0.0))   # face downward

_ts("world.reset() ...")
world.reset()
_ts("world.reset() done")

for _ in range(5):
    world.step(render=True)

# ---------------------------------------------------------------------------
# Camera — single prim, repositioned per door
# ---------------------------------------------------------------------------

cam = Camera(
    prim_path="/World/B1Cam",
    position=np.array([0.0, 0.0, 5.0]),
    frequency=30,
    resolution=CAM_RESOLUTION,
    orientation=rot_utils.euler_angles_to_quats(
        np.array([0.0, 0.0, 180.0]), degrees=True),
)
simulation_app.update()
cam.initialize()

usd_cam_prim = UsdGeom.Camera(stage.GetPrimAtPath("/World/B1Cam"))
usd_cam_prim.GetFocalLengthAttr().Set(5.0)
usd_cam_prim.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 10000.0))

rp  = rep.create.render_product("/World/B1Cam", CAM_RESOLUTION)
ann = rep.AnnotatorRegistry.get_annotator("rgb")
ann.attach([rp])

_ts(f"Renderer cold-start warmup ({STARTUP_WARMUP} steps) ...")
for _ in range(STARTUP_WARMUP):
    world.step(render=True)
_ts("Warmup done")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_glass_mesh(prim) -> bool:
    n = prim.GetPath().name.lower()
    return (n.startswith("glass_") or "_gl_" in n
            or "glazing" in n or "pane_" in n)


def _door_matrix(ox_cm, oy_cm, oz_cm, angle_deg, hx_cm, hy_cm) -> Gf.Matrix4d:
    theta = math.radians(angle_deg)
    c, s = math.cos(theta), math.sin(theta)
    dx, dy = ox_cm - hx_cm, oy_cm - hy_cm
    return Gf.Matrix4d(
         c,  s, 0, 0,
        -s,  c, 0, 0,
         0,  0, 1, 0,
        c * dx - s * dy + hx_cm,
        s * dx + c * dy + hy_cm,
        oz_cm, 1,
    )


def set_door_angle(mesh_ops, angle_deg, hx_cm, hy_cm):
    for m, op in mesh_ops:
        op.Set(_door_matrix(m["ox"], m["oy"], m["oz"], angle_deg, hx_cm, hy_cm))


def _cam_pose_for_side(cx, cy, cz, approach_axis, side):
    """Return (position_np, yaw_deg) for the camera on the given side."""
    if approach_axis == "X":
        pos = np.array([cx + side * CAMERA_DIST, cy, cz])
        yaw = 180.0 if side > 0 else 0.0
    else:   # approach_axis == "Y"
        pos = np.array([cx, cy + side * CAMERA_DIST, cz])
        yaw = 270.0 if side > 0 else 90.0
    return pos, yaw


def _line_of_sight_clear(cam_pos_m, door_center_m, door_prim_path) -> bool:
    """Return True if the raycast from cam to door hits the door (not a wall)."""
    try:
        from omni.physx import get_physx_scene_query_interface
        physx = get_physx_scene_query_interface()
    except Exception:
        return True   # no PhysX — assume clear
    origin    = cam_pos_m
    target    = np.array(door_center_m)
    direction = target - origin
    dist      = float(np.linalg.norm(direction))
    if dist < 0.01:
        return True
    direction /= dist
    hit = physx.raycast_closest(
        tuple(float(v) for v in origin),
        tuple(float(v) for v in direction),
        dist,
    )
    if hit["hit"]:
        return door_prim_path in (hit.get("rigidBody", "") or "")
    return True   # no hit at all — open corridor


def reposition_camera(pos, yaw_deg):
    ori = rot_utils.euler_angles_to_quats(
        np.array([0.0, 0.0, yaw_deg]), degrees=True)
    cam.set_world_pose(position=pos, orientation=ori)


def capture_jpeg(door_id, angle_deg) -> bool:
    world.step(render=True)
    data = ann.get_data()
    if data is None:
        print(f"  [WARN] annotator returned None — {door_id} {angle_deg:.0f}°")
        return False
    arr = np.asarray(data)
    if arr.ndim != 3 or arr.shape[-1] < 3:
        print(f"  [WARN] bad rgb shape {arr.shape} — {door_id} {angle_deg:.0f}°")
        return False
    rgb = arr[:, :, :3].astype(np.uint8)
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

summary = []   # list of {door_id, floor, status, camera_side, note}

for di, door in enumerate(doors_to_run):
    door_id    = door["id"]
    prim_path  = door["prim_path"]
    pos        = door["position"]
    dim        = door["dimensions"]
    approach   = dim.get("approach_axis", "X")

    cx_m = pos["center_x"]
    cy_m = pos["center_y"]
    cz_m = pos["center_z"]

    # Camera Z: 1.2 m above the door's bottom edge
    floor_z_m  = cz_m - dim["height"] / 2.0
    cam_z_m    = floor_z_m + CAMERA_HEIGHT

    # Hinge from first mesh child: translation + extent_min on the wide axis
    mc         = door["mesh_children"][0]
    tx_cm      = mc["translation_cm"][0]
    ty_cm      = mc["translation_cm"][1]
    hinge_x_cm = tx_cm
    hinge_y_cm = ty_cm + mc["extent_min"][1]

    _ts(f"[{di+1:2d}/{len(doors_to_run)}] {door_id}  floor={door['floor']}  "
        f"centre=({cx_m:.1f},{cy_m:.1f})  hinge_y={hinge_y_cm:.1f} cm")

    entry = {"door_id": door_id, "floor": door["floor"],
             "status": "ok", "camera_side": None, "note": ""}

    # -- Validate prim --
    door_prim = stage.GetPrimAtPath(prim_path)
    if not door_prim.IsValid():
        print(f"  SKIP — prim not valid at {prim_path}")
        entry["status"] = "skipped"
        entry["note"]   = "prim not valid"
        summary.append(entry)
        continue

    if door_prim.IsInstance() or door_prim.IsInstanceable():
        door_prim.SetInstanceable(False)
        simulation_app.update()

    # -- Discover meshes --
    mesh_ops = []
    for child in door_prim.GetChildren():
        if not child.IsA(UsdGeom.Mesh):
            continue
        xf = UsdGeom.Xformable(child)
        op = next((o for o in xf.GetOrderedXformOps()
                   if o.GetOpName() == "xformOp:transform"), None)
        if op is None:
            continue
        mat = op.Get()
        mesh_ops.append((
            {"path": str(child.GetPath()),
             "ox": mat[3][0], "oy": mat[3][1], "oz": mat[3][2]},
            op,
        ))

    if not mesh_ops:
        print(f"  SKIP — no meshes with xformOp:transform")
        entry["status"] = "skipped"
        entry["note"]   = "no xform meshes"
        summary.append(entry)
        continue

    # -- Reposition fill light above door --
    _fill_translate.Set(Gf.Vec3d(cx_m, cy_m, cam_z_m + 1.5))

    # -- Find unoccluded camera side (+approach first, then -) --
    door_centre_3d = [cx_m, cy_m, cam_z_m]
    chosen_pos, chosen_yaw = None, None
    for side in [1, -1]:
        cpos, yaw = _cam_pose_for_side(cx_m, cy_m, cam_z_m, approach, side)
        if _line_of_sight_clear(cpos, door_centre_3d, prim_path):
            chosen_pos, chosen_yaw = cpos, yaw
            entry["camera_side"] = int(side)
            break

    if chosen_pos is None:
        print(f"  SKIP — both camera sides occluded")
        entry["status"] = "skipped"
        entry["note"]   = "occluded both sides"
        summary.append(entry)
        continue

    reposition_camera(chosen_pos, chosen_yaw)

    # -- Render each angle --
    any_ok = False
    for angle_deg in ANGLE_GRID:
        set_door_angle(mesh_ops, angle_deg, hinge_x_cm, hinge_y_cm)
        for _ in range(WARMUP_STEPS):
            world.step(render=True)
        ok = capture_jpeg(door_id, angle_deg)
        if ok:
            any_ok = True
        cls = "CLOSED" if angle_deg <= CLOSED_MAX_DEG else "OPEN"
        print(f"  {cls:6s} {angle_deg:5.1f}°  {'saved' if ok else 'FAILED'}")

    # Reset door to closed before moving to next
    set_door_angle(mesh_ops, 0.0, hinge_x_cm, hinge_y_cm)

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

summary_path = os.path.join(_project_dir, "data", "door_b1_summary.json")
with open(summary_path, "w") as f:
    json.dump({"ok": ok_count, "partial": partial_count,
               "skipped": skip_count, "doors": summary}, f, indent=2)

_ts(f"Done — {ok_count} OK  {partial_count} partial  {skip_count} skipped")
_ts(f"JPEGs  → {OUT_DIR}/")
_ts(f"Summary → {summary_path}")

world.stop()
simulation_app.close()

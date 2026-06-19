#!/usr/bin/env python3
"""
selected_doors_preview.py — Render CLOSED (0°) and OPEN (45°) for each of
the 10 selected training/holdout doors defined in SELECTED_TRAINING_DOORS.md.

Produces 20 JPEGs (2 per door) in data/selected_doors_preview/ so the final
door selection can be reviewed visually in one place.

Run
---
  sbatch submit_isaac_job.sh simulations/robot/selected_doors_preview.py
"""

import os
import math
import json
import functools
import numpy as np

print = functools.partial(print, flush=True)  # noqa: A001

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ANGLE_GRID     = [0.0, 45.0]   # CLOSED, OPEN
CAM_RESOLUTION = (1024, 768)
WARMUP_STEPS   = 10
STARTUP_WARMUP = 30
CLOSED_MAX_DEG = 5.0

_script_dir  = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.abspath(os.path.join(_script_dir, "..", ".."))
USD_PATH     = os.path.join(_project_dir, "MSOEBuildingV4_physics.usd")
OUT_DIR      = os.path.join(_project_dir, "data", "selected_doors_preview")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Door registry — all values from SELECTED_TRAINING_DOORS.md
#
#   mesh_count > 0  →  world-space rotation  (curtain-wall)
#   mesh_count == 0 →  local-space rotation  (instanced)
#   angle_sign      →  +1 normal, -1 for Y-axis-flipped parent prims
#   leaf_mesh_names →  None (rotate all) or set of names to restrict
# ---------------------------------------------------------------------------

_BASE = (
    "/World/Building/_22_129___MSOE_Engineering_Building___A___2_25_2026"
    "/Geometry/Doors"
)

DOORS = [
    # ── TRAINING ─────────────────────────────────────────────────────────────
    {
        "id":              "Doors_13607620",
        "family_short":    "CW_F_NEW",
        "floor":           2,
        "role":            "train",
        "prim_path":       f"{_BASE}/RS___Curtain_Wall___Type_F___NEW/New/Doors_13607620",
        "mesh_count":      1,
        "hinge_x_cm":      1246.49,
        "hinge_y_cm":      7821.74,
        "cam_pos":         [12.922, 80.240, 14.764],
        "cam_yaw_deg":     270.0,
        "angle_sign":      1,
        "leaf_mesh_names": None,
    },
    {
        "id":              "Doors_13182085",
        "family_short":    "CW_FG_New",
        "floor":           4,
        "role":            "train",
        "prim_path":       f"{_BASE}/RS___Curtain_Wall___Type_FG___New/New/Doors_13182085",
        "mesh_count":      2,
        "hinge_x_cm":      -4142.56,
        "hinge_y_cm":      8214.32,
        "cam_pos":         [-39.426, 82.600, 23.704],
        "cam_yaw_deg":     180.0,
        "angle_sign":      1,
        "leaf_mesh_names": None,
    },
    {
        "id":              "Doors_3135827",
        "family_short":    "CW_FG_New",
        "floor":           3,
        "role":            "train",
        "prim_path":       f"{_BASE}/RS___Curtain_Wall___Type_FG___New/New/Doors_3135827",
        "mesh_count":      2,
        "hinge_x_cm":      1724.33,
        "hinge_y_cm":      5878.48,
        "cam_pos":         [17.701, 56.807, 19.437],
        "cam_yaw_deg":     90.0,
        "angle_sign":      1,
        "leaf_mesh_names": None,
    },
    {
        "id":              "Doors_8286057",
        "family_short":    "Type_F",
        "floor":           2,
        "role":            "train",
        "prim_path":       f"{_BASE}/RS___Type_F_Door/NEW___HM_Frame___36_x96/Doors_8286057",
        "mesh_count":      0,
        "hinge_x_cm":      +50.80,
        "hinge_y_cm":      0.0,
        "cam_pos":         [17.443, 79.341, 14.764],
        "cam_yaw_deg":     270.0,
        "angle_sign":      1,
        "leaf_mesh_names": None,
    },
    {
        "id":              "Doors_4078243",
        "family_short":    "Type_F",
        "floor":           5,
        "role":            "train",
        "prim_path":       f"{_BASE}/RS___Type_F_Door/NEW___HM_Frame___36_x96/Doors_4078243",
        "mesh_count":      0,
        "hinge_x_cm":      +50.80,
        "hinge_y_cm":      0.0,
        "cam_pos":         [-15.781, 61.636, 27.972],
        "cam_yaw_deg":     0.0,
        "angle_sign":      1,
        "leaf_mesh_names": None,
    },
    {
        "id":              "Doors_10649710",
        "family_short":    "Type_FG",
        "floor":           3,
        "role":            "train",
        "prim_path":       f"{_BASE}/RS___Type_FG_Door/NEW___HM_Frame___36_x96/Doors_10649710",
        "mesh_count":      0,
        "hinge_x_cm":      -50.80,
        "hinge_y_cm":      0.0,
        "cam_pos":         [-10.295, 75.473, 19.437],
        "cam_yaw_deg":     180.0,
        "angle_sign":      1,
        "leaf_mesh_names": None,
    },
    {
        "id":              "Doors_2429516",
        "family_short":    "Type_N",
        "floor":           4,
        "role":            "train",
        "prim_path":       f"{_BASE}/RS___Type_N_Door/NEW___HM_Frame___36_x96/Doors_2429516",
        "mesh_count":      0,
        "hinge_x_cm":      -50.80,
        "hinge_y_cm":      0.0,
        "cam_pos":         [17.875, 54.941, 23.704],
        "cam_yaw_deg":     270.0,
        "angle_sign":      1,
        "leaf_mesh_names": None,
    },
    {
        "id":              "Doors_12590968",
        "family_short":    "Type_N",
        "floor":           3,
        "role":            "train",
        "prim_path":       f"{_BASE}/RS___Type_N_Door/NEW___HM_Frame___36_x96/Doors_12590968",
        "mesh_count":      0,
        "hinge_x_cm":      -50.80,
        "hinge_y_cm":      0.0,
        "cam_pos":         [17.878, 50.941, 19.437],
        "cam_yaw_deg":     90.0,
        "angle_sign":      -1,   # Y-axis-flipped parent
        "leaf_mesh_names": None,
    },
    # ── HOLDOUT ──────────────────────────────────────────────────────────────
    {
        "id":              "Doors_10649156",
        "family_short":    "Type_FG",
        "floor":           3,
        "role":            "holdout",
        "prim_path":       f"{_BASE}/RS___Type_FG_Door/NEW___HM_Frame___36_x96/Doors_10649156",
        "mesh_count":      0,
        "hinge_x_cm":      +50.80,
        "hinge_y_cm":      0.0,
        "cam_pos":         [-14.295, 80.200, 19.437],
        "cam_yaw_deg":     0.0,
        "angle_sign":      1,
        "leaf_mesh_names": None,
    },
    {
        "id":              "Doors_7171063",
        "family_short":    "Type_F",
        "floor":           5,
        "role":            "holdout",
        "prim_path":       f"{_BASE}/RS___Type_F_Door/NEW___HM_Frame___42_x96/Doors_7171063",
        "mesh_count":      0,
        "hinge_x_cm":      -58.42,
        "hinge_y_cm":      0.0,
        "cam_pos":         [-49.316, 49.892, 27.972],
        "cam_yaw_deg":     0.0,
        "angle_sign":      1,
        "leaf_mesh_names": None,
    },
]

# ---------------------------------------------------------------------------
# Isaac Sim startup
# ---------------------------------------------------------------------------

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

import time as _time
_t0 = _time.time()
def _ts(msg):
    print(f"[Preview] [+{_time.time() - _t0:6.1f}s] {msg}")

_ts("Starting SimulationApp ...")
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
# Scene
# ---------------------------------------------------------------------------

world = World(stage_units_in_meters=1.0)
stage = omni.usd.get_context().get_stage()
stage.GetRootLayer().subLayerPaths.insert(0, USD_PATH)
_ts("Building USD inserted")

dome = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/DomeLight"))
dome.CreateIntensityAttr(40_000)
dome.GetColorAttr().Set(Gf.Vec3f(1.0, 1.0, 1.0))

fill = UsdLux.RectLight.Define(stage, Sdf.Path("/World/FillLight"))
fill.CreateIntensityAttr(300_000)
fill.CreateWidthAttr(1.2)
fill.CreateHeightAttr(1.2)
_fill_xf        = UsdGeom.Xformable(fill.GetPrim())
_fill_translate = _fill_xf.AddTranslateOp()
_fill_rotate    = _fill_xf.AddRotateXYZOp()
_fill_rotate.Set(Gf.Vec3f(180.0, 0.0, 0.0))

world.reset()
for _ in range(5):
    world.step(render=True)

# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------

cam = Camera(
    prim_path="/World/PreviewCam",
    position=np.array([0.0, 0.0, 5.0]),
    frequency=30,
    resolution=CAM_RESOLUTION,
    orientation=rot_utils.euler_angles_to_quats(np.array([0.0, 0.0, 180.0]), degrees=True),
)
simulation_app.update()
cam.initialize()

usd_cam = UsdGeom.Camera(stage.GetPrimAtPath("/World/PreviewCam"))
usd_cam.GetFocalLengthAttr().Set(5.0)
usd_cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 10000.0))

rp  = rep.create.render_product("/World/PreviewCam", CAM_RESOLUTION)
ann = rep.AnnotatorRegistry.get_annotator("rgb")
ann.attach([rp])

_ts(f"Cold-start warmup ({STARTUP_WARMUP} steps) ...")
for _ in range(STARTUP_WARMUP):
    world.step(render=True)
_ts("Warmup done")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _door_matrix(ox, oy, oz, angle_deg, hx, hy) -> Gf.Matrix4d:
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

def _discover_meshes_world(door_prim):
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
        if mat is None:
            continue
        mesh_ops.append(({"ox": mat[3][0], "oy": mat[3][1], "oz": mat[3][2]}, op))
    return mesh_ops

def _discover_meshes_local(door_prim, leaf_mesh_names=None):
    mesh_ops = []
    for p in Usd.PrimRange(door_prim):
        if not p.IsA(UsdGeom.Mesh):
            continue
        if leaf_mesh_names is not None and p.GetName() not in leaf_mesh_names:
            continue
        xf = UsdGeom.Xformable(p)
        op = next((o for o in xf.GetOrderedXformOps()
                   if o.GetOpName() == "xformOp:transform"), None)
        if op is None:
            continue
        mat = op.Get()
        if mat is None:
            continue
        mesh_ops.append(({"ox": mat[3][0], "oy": mat[3][1], "oz": mat[3][2]}, op))
    return mesh_ops

def _reposition_camera(pos, yaw_deg):
    ori = rot_utils.euler_angles_to_quats(np.array([0.0, 0.0, yaw_deg]), degrees=True)
    cam.set_world_pose(position=np.array(pos), orientation=ori)

def _capture(family_short, door_id, floor, role, angle_deg) -> bool:
    world.step(render=True)
    data = ann.get_data()
    if data is None:
        return False
    arr = np.asarray(data)
    if arr.ndim != 3 or arr.shape[-1] < 3:
        return False
    rgb      = arr[:, :, :3].astype(np.uint8)
    cls      = "CLOSED" if angle_deg <= CLOSED_MAX_DEG else "OPEN"
    role_tag = "_HOLDOUT" if role == "holdout" else ""
    fname    = f"{family_short}_{door_id}_floor{floor}{role_tag}_{cls}.jpg"
    path     = os.path.join(OUT_DIR, fname)
    if _pil_ok:
        _PIL.fromarray(rgb).save(path, "JPEG", quality=92)
    else:
        np.save(path.replace(".jpg", ".npy"), rgb)
    return True

# ---------------------------------------------------------------------------
# Per-door loop
# ---------------------------------------------------------------------------

total   = len(DOORS)
saved   = 0
skipped = 0

for di, door in enumerate(DOORS):
    door_id    = door["id"]
    fam_short  = door["family_short"]
    floor      = door["floor"]
    role       = door["role"]
    mesh_count = door["mesh_count"]
    hx_cm      = door["hinge_x_cm"]
    hy_cm      = door["hinge_y_cm"]
    angle_sign = door["angle_sign"]
    leaf_names = door["leaf_mesh_names"]
    cam_pos    = door["cam_pos"]

    role_tag = "[HOLDOUT]" if role == "holdout" else "[train  ]"
    _ts(f"[{di+1:2d}/{total}] {role_tag} [{fam_short:12s}] {door_id}  floor={floor}")

    prim = stage.GetPrimAtPath(door["prim_path"])
    if not prim.IsValid():
        _ts(f"  SKIP — prim not found")
        skipped += 1
        continue

    if prim.IsInstance() or prim.IsInstanceable():
        prim.SetInstanceable(False)
        simulation_app.update()

    if mesh_count > 0:
        mesh_ops = _discover_meshes_world(prim)
    else:
        mesh_ops = _discover_meshes_local(prim, leaf_names)

    if not mesh_ops:
        _ts(f"  SKIP — no xformOp meshes found")
        skipped += 1
        continue

    # Reposition fill light and camera
    _fill_translate.Set(Gf.Vec3d(cam_pos[0], cam_pos[1], cam_pos[2] + 1.5))
    _reposition_camera(cam_pos, door["cam_yaw_deg"])

    for angle_deg in ANGLE_GRID:
        signed_angle = angle_sign * angle_deg
        _set_angle(mesh_ops, signed_angle, hx_cm, hy_cm)
        for _ in range(WARMUP_STEPS):
            world.step(render=True)
        ok = _capture(fam_short, door_id, floor, role, angle_deg)
        cls = "CLOSED" if angle_deg <= CLOSED_MAX_DEG else "OPEN"
        _ts(f"  {cls:6s}  {'saved' if ok else 'FAILED'}")
        if ok:
            saved += 1

    # Reset to closed
    _set_angle(mesh_ops, 0.0, hx_cm, hy_cm)

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------

_ts(f"Done — {saved} JPEGs saved  {skipped} doors skipped")
_ts(f"Output → {OUT_DIR}/")

world.stop()
simulation_app.close()

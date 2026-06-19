#!/usr/bin/env python3
"""
Door Family Sampler — one CLOSED + one OPEN image per door type.

Renders one representative door from each of the 7 single-leaf swing
families so the appearances can be compared side-by-side:

    RS___Curtain_Wall___Type_FG___New   curtain-wall glass (mesh ready)
    RS___Curtain_Wall___Type_F___NEW    curtain-wall solid (mesh ready)
    RS___Type_F_Door                    solid HM swing (instanced)
    RS___Type_F_W_SIDELITE_Door         door + sidelite (instanced)
    RS___Type_FG_Door                   glass-panel solid (instanced)
    RS___Type_N_Door                    narrow swing (instanced)
    RS___Type_G_Door                    full-glass swing (instanced)

Excluded families:
    Double-swing  (Type_2F*, Type_2FG, Type_2G, Type_2FU, Type_2FG curtain)
    Non-swing     (Garage, Overhead Rolling/Sectional, Sliding Glass Wall)

For mesh-ready doors the rotation is applied in world space (B1 method).
For instanced doors SetInstanceable(False) is called and rotation is
applied in local space (B2 method).

Output
------
    data/door_family_samples/<family_short>_CLOSED.jpg
    data/door_family_samples/<family_short>_OPEN.jpg

Run
---
    sbatch submit_isaac_job.sh simulations/robot/door_family_sampler.py
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

CAMERA_DIST    = 2.0
CAMERA_HEIGHT  = 1.2     # metres above door bottom edge
CAM_RESOLUTION = (1024, 768)
CLOSED_MAX_DEG = 5.0
ANGLE_GRID     = [0.0, 45.0]
WARMUP_STEPS   = 10
STARTUP_WARMUP = 30

EXCLUDE_FAMILIES = {
    "RS___Type_2F_Door", "RS___Type_2F___Dual_Door",
    "RS___Type_2FG_Door", "RS___Type_2G_Door", "RS___Type_2FU_Door",
    "RS___Curtain_Wall___Type_2FG___NEW",
    "AV300_RAYNOR_GARAGE_DOOR", "RS___Overhead_Rolling_Door",
    "RS___Overhead_Sectional_Door", "Dormakaba_Sliding_Glass_Wall_FSW",
}

# Short label used in JPEG filenames — strips the long Revit prefix
_FAMILY_SHORT = {
    "RS___Curtain_Wall___Type_FG___New":  "CW_FG_New",
    "RS___Curtain_Wall___Type_F___NEW":   "CW_F_NEW",
    "RS___Type_F_Door":                   "Type_F",
    "RS___Type_F_W_SIDELITE_Door":        "Type_F_Sidelite",
    "RS___Type_FG_Door":                  "Type_FG",
    "RS___Type_N_Door":                   "Type_N",
    "RS___Type_G_Door":                   "Type_G",
}

_script_dir  = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.abspath(os.path.join(_script_dir, "..", ".."))
CATALOG_PATH = os.path.join(_script_dir, "door_catalog.json")
USD_PATH     = os.path.join(_project_dir, "MSOEBuildingV4_physics.usd")
OUT_DIR      = os.path.join(_project_dir, "data", "door_family_samples")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Pick one representative per family
# ---------------------------------------------------------------------------

with open(CATALOG_PATH) as f:
    _cat = json.load(f)

# Specific door IDs to use per family. Chosen from floors 2–4 to avoid
# the floor-1 basement (poor natural lighting) and floor-6 top-level.
# Override the automatic "first non-double in catalog" selection.
_PINNED_IDS = {
    "RS___Curtain_Wall___Type_F___NEW": "Doors_13606056",   # floor 3, approach=X
    "RS___Type_F_Door":                 "Doors_8453326",    # floor 3, approach=X
}

_by_id = {d["id"]: d for d in _cat["doors"]}

_samples = {}   # family → door entry
for d in _cat["doors"]:
    fam = d["door_type_family"]
    if fam in EXCLUDE_FAMILIES:
        continue
    if d.get("is_double_door", False):
        continue
    # Use pinned door if specified, otherwise first non-double in catalog
    if fam in _PINNED_IDS:
        if fam not in _samples:
            _samples[fam] = _by_id[_PINNED_IDS[fam]]
    elif fam not in _samples:
        _samples[fam] = d

doors_to_run = list(_samples.values())
print(f"[Sampler] {len(doors_to_run)} families to render:")
for d in doors_to_run:
    short = _FAMILY_SHORT.get(d["door_type_family"], d["door_type_family"])
    print(f"  {short:<22}  id={d['id']}  floor={d['floor']}  "
          f"mesh_count={d.get('mesh_count',0)}  "
          f"approach={d['dimensions'].get('approach_axis','?')}")

# ---------------------------------------------------------------------------
# Isaac Sim startup
# ---------------------------------------------------------------------------

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

import time as _time
_t0 = _time.time()
def _ts(msg):
    print(f"[Sampler] [+{_time.time() - _t0:6.1f}s] {msg}")

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
    print("[WARN] PIL unavailable — saving .npy")

# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------

world = World(stage_units_in_meters=1.0)
stage = omni.usd.get_context().get_stage()
_ts(f"Inserting building USD ...")
stage.GetRootLayer().subLayerPaths.insert(0, USD_PATH)

dome = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/DomeLight"))
dome.CreateIntensityAttr(40_000)
dome.GetColorAttr().Set(Gf.Vec3f(1.0, 1.0, 1.0))

fill = UsdLux.RectLight.Define(stage, Sdf.Path("/World/FillLight"))
fill.CreateIntensityAttr(300_000)
fill.CreateWidthAttr(1.2)
fill.CreateHeightAttr(1.2)
_fill_xf = UsdGeom.Xformable(fill.GetPrim())
_fill_t   = _fill_xf.AddTranslateOp()
_fill_r   = _fill_xf.AddRotateXYZOp()
_fill_r.Set(Gf.Vec3f(180.0, 0.0, 0.0))

_ts("world.reset() ...")
world.reset()
_ts("world.reset() done")
for _ in range(5):
    world.step(render=True)

# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------

cam = Camera(
    prim_path="/World/SamplerCam",
    position=np.array([0.0, 0.0, 5.0]),
    frequency=30,
    resolution=CAM_RESOLUTION,
    orientation=rot_utils.euler_angles_to_quats(
        np.array([0.0, 0.0, 180.0]), degrees=True),
)
simulation_app.update()
cam.initialize()

_usd_cam = UsdGeom.Camera(stage.GetPrimAtPath("/World/SamplerCam"))
_usd_cam.GetFocalLengthAttr().Set(5.0)
_usd_cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 10000.0))

rp  = rep.create.render_product("/World/SamplerCam", CAM_RESOLUTION)
ann = rep.AnnotatorRegistry.get_annotator("rgb")
ann.attach([rp])

_ts(f"Renderer warmup ({STARTUP_WARMUP} steps) ...")
for _ in range(STARTUP_WARMUP):
    world.step(render=True)
_ts("Warmup done")

# ---------------------------------------------------------------------------
# Rotation helpers — world space (mesh-ready doors, B1 method)
# ---------------------------------------------------------------------------

def _world_door_matrix(ox, oy, oz, angle_deg, hx, hy) -> Gf.Matrix4d:
    theta = math.radians(angle_deg)
    c, s  = math.cos(theta), math.sin(theta)
    dx, dy = ox - hx, oy - hy
    return Gf.Matrix4d(
         c,  s, 0, 0,
        -s,  c, 0, 0,
         0,  0, 1, 0,
        c*dx - s*dy + hx,
        s*dx + c*dy + hy,
        oz, 1,
    )

def _set_angle_world(mesh_ops, angle_deg, hx_cm, hy_cm):
    for m, op in mesh_ops:
        op.Set(_world_door_matrix(m["ox"], m["oy"], m["oz"],
                                  angle_deg, hx_cm, hy_cm))

# ---------------------------------------------------------------------------
# Rotation helpers — local space (instanced doors, B2 method)
# ---------------------------------------------------------------------------

def _local_door_matrix(lx, ly, lz, angle_deg, hlx, hly) -> Gf.Matrix4d:
    theta = math.radians(angle_deg)
    c, s  = math.cos(theta), math.sin(theta)
    dx, dy = lx - hlx, ly - hly
    return Gf.Matrix4d(
         c,  s, 0, 0,
        -s,  c, 0, 0,
         0,  0, 1, 0,
        c*dx - s*dy + hlx,
        s*dx + c*dy + hly,
        lz, 1,
    )

def _set_angle_local(mesh_ops, angle_deg, hlx, hly):
    for m, op in mesh_ops:
        op.Set(_local_door_matrix(m["lx"], m["ly"], m["lz"],
                                  angle_deg, hlx, hly))

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

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


def _find_camera_side(cx, cy, cz, approach, prim_path):
    for side in [1, -1]:
        if approach == "X":
            cpos = np.array([cx + side * CAMERA_DIST, cy, cz])
            yaw  = 180.0 if side > 0 else 0.0
        else:
            cpos = np.array([cx, cy + side * CAMERA_DIST, cz])
            yaw  = 270.0 if side > 0 else 90.0
        if _los_clear(cpos, [cx, cy, cz], prim_path):
            return cpos, yaw
    return None, None


def _reposition_camera(pos, yaw_deg):
    ori = rot_utils.euler_angles_to_quats(
        np.array([0.0, 0.0, yaw_deg]), degrees=True)
    cam.set_world_pose(position=pos, orientation=ori)


def _capture(family_short, angle_deg) -> bool:
    world.step(render=True)
    data = ann.get_data()
    if data is None:
        return False
    arr = np.asarray(data)
    if arr.ndim != 3 or arr.shape[-1] < 3:
        return False
    rgb  = arr[:, :, :3].astype(np.uint8)
    cls  = "CLOSED" if angle_deg <= CLOSED_MAX_DEG else "OPEN"
    path = os.path.join(OUT_DIR, f"{family_short}_{cls}.jpg")
    if _pil_ok:
        _PIL.fromarray(rgb).save(path, "JPEG", quality=92)
    else:
        np.save(path.replace(".jpg", ".npy"), rgb)
    return True

# ---------------------------------------------------------------------------
# Per-door rendering
# ---------------------------------------------------------------------------

results = []

for door in doors_to_run:
    fam       = door["door_type_family"]
    short     = _FAMILY_SHORT.get(fam, fam)
    door_id   = door["id"]
    prim_path = door["prim_path"]
    pos       = door["position"]
    dim       = door["dimensions"]
    approach  = dim.get("approach_axis", "Y")
    has_mesh  = door.get("mesh_count", 0) > 0

    cx_m = pos["center_x"]
    cy_m = pos["center_y"]
    cz_m = pos["center_z"]
    cam_z_m = cz_m - dim["height"] / 2.0 + CAMERA_HEIGHT

    _ts(f"── {short}  ({door_id}, floor {door['floor']}, "
        f"{'mesh-ready' if has_mesh else 'instanced'}) ──")

    door_prim = stage.GetPrimAtPath(prim_path)
    if not door_prim.IsValid():
        print(f"  SKIP — prim not valid")
        results.append({"family": short, "status": "skipped", "note": "prim not valid"})
        continue

    if door_prim.IsInstance() or door_prim.IsInstanceable():
        door_prim.SetInstanceable(False)
        simulation_app.update()

    # ---- Discover meshes + build rotation op list ----
    mesh_ops = []

    if has_mesh:
        # World-space matrices on direct children (B1 path)
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
                {"ox": mat[3][0], "oy": mat[3][1], "oz": mat[3][2]},
                op,
            ))
    else:
        # Local-space matrices — walk full subtree after concretizing (B2 path)
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
            mesh_ops.append((
                {"lx": mat[3][0], "ly": mat[3][1], "lz": mat[3][2]},
                op,
            ))

    if not mesh_ops:
        print(f"  SKIP — no xformOp:transform meshes found")
        results.append({"family": short, "status": "skipped",
                        "note": "no xform meshes"})
        continue

    print(f"  {len(mesh_ops)} mesh(es)")

    # ---- Hinge ----
    if has_mesh:
        # World-space hinge from first mesh child extent (B1 method)
        mc = door["mesh_children"][0]
        hinge_x_cm = mc["translation_cm"][0]
        hinge_y_cm = mc["translation_cm"][1] + mc["extent_min"][1]
    else:
        # Local-space hinge: min edge of the wide axis (B2 method)
        half_w_cm = dim["width"] * 100.0 / 2.0
        hinge_x_cm = -half_w_cm if approach == "Y" else 0.0
        hinge_y_cm = 0.0        if approach == "Y" else -half_w_cm

    # ---- Camera ----
    _fill_t.Set(Gf.Vec3d(cx_m, cy_m, cam_z_m + 1.5))
    cam_pos, cam_yaw = _find_camera_side(
        cx_m, cy_m, cam_z_m, approach, prim_path)

    if cam_pos is None:
        print(f"  SKIP — both camera sides occluded")
        results.append({"family": short, "status": "skipped",
                        "note": "occluded both sides"})
        continue

    _reposition_camera(cam_pos, cam_yaw)

    # ---- Render CLOSED + OPEN ----
    any_ok = False
    for angle_deg in ANGLE_GRID:
        if has_mesh:
            _set_angle_world(mesh_ops, angle_deg, hinge_x_cm, hinge_y_cm)
        else:
            _set_angle_local(mesh_ops, angle_deg, hinge_x_cm, hinge_y_cm)

        for _ in range(WARMUP_STEPS):
            world.step(render=True)

        ok  = _capture(short, angle_deg)
        cls = "CLOSED" if angle_deg <= CLOSED_MAX_DEG else "OPEN"
        print(f"  {cls:6s}  {'saved' if ok else 'FAILED'}")
        if ok:
            any_ok = True

    # Reset to closed before next door
    if has_mesh:
        _set_angle_world(mesh_ops, 0.0, hinge_x_cm, hinge_y_cm)
    else:
        _set_angle_local(mesh_ops, 0.0, hinge_x_cm, hinge_y_cm)

    results.append({"family": short, "status": "ok" if any_ok else "partial",
                    "door_id": door_id})

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

_ts("Done.")
for r in results:
    print(f"  {r['family']:<22}  {r['status']}")
print(f"\nJPEGs → {OUT_DIR}/")

world.stop()
simulation_app.close()

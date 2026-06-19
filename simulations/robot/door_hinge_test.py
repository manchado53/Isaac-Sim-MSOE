#!/usr/bin/env python3
"""
door_hinge_test.py — Hinge-position verification for the 10 selected training doors.

Renders each door at 0°, 45°, and 90° so you can visually confirm that the door
leaf rotates around the correct hinge point and doesn't clip through the frame.

All 10 doors are hard-coded from SELECTED_TRAINING_DOORS.md with their pre-derived
hinge coordinates and camera positions — no catalog lookup needed.

Rotation method dispatched by mesh_count (same rule as door_candidate_images.py):
  mesh_count > 0  →  world-space 4×4 matrix   (curtain-wall doors: #1, #2)
  mesh_count == 0 →  local-space 4×4 matrix    (instanced doors:    #3–#10)

Output
------
  data/door_hinge_test/<family_short>_<door_id>_floor<N>_ang<DEG>_<CLOSED|OPEN>.jpg
  data/door_hinge_test_summary.json

Run
---
  sbatch submit_isaac_job.sh simulations/robot/door_hinge_test.py
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

ANGLE_GRID     = [0.0, 45.0, 90.0]
CAM_RESOLUTION = (1024, 768)
WARMUP_STEPS   = 10
STARTUP_WARMUP = 30
CLOSED_MAX_DEG = 5.0

_script_dir  = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.abspath(os.path.join(_script_dir, "..", ".."))
USD_PATH     = os.path.join(_project_dir, "MSOEBuildingV4_physics.usd")
OUT_DIR      = os.path.join(_project_dir, "data", "door_hinge_test_final")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Door registry — hard-coded from SELECTED_TRAINING_DOORS.md
#
# Fields:
#   id           : Revit door ID string
#   prim_path    : full USD prim path
#   family_short : short label used in filenames
#   floor        : building floor (2–5)
#   role         : "train" or "holdout"
#   mesh_count   : > 0 → world-space rotation; == 0 → local-space rotation
#   hinge_x_cm   : hinge X in world cm (mesh_count>0) or local cm (mesh_count==0)
#   hinge_y_cm   : hinge Y in world cm (mesh_count>0) or local cm (mesh_count==0)
#   cam_pos      : [x, y, z] camera world position in metres
#   cam_yaw_deg      : camera yaw (degrees)
#   leaf_mesh_names  : set of mesh names to rotate; None = rotate all meshes.
#                      Used to exclude fixed sidelite glass from Type_F_Sidelite.
#   angle_sign       : +1 (default) or -1 to flip rotation direction.
#                      Needed when parent matrix flips an axis and positive angles
#                      swing the door toward the camera instead of away.
# ---------------------------------------------------------------------------

_BASE = (
    "/World/Building/_22_129___MSOE_Engineering_Building___A___2_25_2026"
    "/Geometry/Doors"
)

DOORS = [
    # ── TRAIN ────────────────────────────────────────────────────────────────
    {   # 1 — CW_F_NEW, floor 4, world-space rotation
        "id":               "Doors_13606466",
        "prim_path":        f"{_BASE}/RS___Curtain_Wall___Type_F___NEW/New/Doors_13606466",
        "family_short":     "CW_F_NEW",
        "floor":            4,
        "role":             "train",
        "mesh_count":       1,
        "hinge_x_cm":       -4836.01,
        "hinge_y_cm":       6170.58,
        "cam_pos":          [-46.360, 62.163, 23.704],
        "cam_yaw_deg":      180.0,
        "leaf_mesh_names":  None,
        "angle_sign":       1,
    },
    {   # 2 — Type_FG, floor 3, local-space rotation
        # Only Type_FG in training. angle_sign=-1: parent maps local Y = +world X,
        # so positive angle swings toward camera; negating corrects it.
        "id":               "Doors_10649156",
        "prim_path":        f"{_BASE}/RS___Type_FG_Door/NEW___HM_Frame___36_x96/Doors_10649156",
        "family_short":     "Type_FG",
        "floor":            3,
        "role":             "train",
        "mesh_count":       0,
        "hinge_x_cm":       +50.80,
        "hinge_y_cm":       0.0,
        "cam_pos":          [-14.295, 80.200, 19.437],
        "cam_yaw_deg":      0.0,
        "leaf_mesh_names":  None,
        "angle_sign":       -1,
    },
    {   # 3 — Type_F, floor 2, local-space rotation
        "id":               "Doors_8286057",
        "prim_path":        f"{_BASE}/RS___Type_F_Door/NEW___HM_Frame___36_x96/Doors_8286057",
        "family_short":     "Type_F",
        "floor":            2,
        "role":             "train",
        "mesh_count":       0,
        "hinge_x_cm":       +50.80,
        "hinge_y_cm":       0.0,
        "cam_pos":          [17.443, 79.341, 14.764],
        "cam_yaw_deg":      270.0,
        "leaf_mesh_names":  None,
        "angle_sign":       1,
    },
    {   # 4 — CW_FG_New, floor 3, world-space rotation
        "id":               "Doors_3135827",
        "prim_path":        f"{_BASE}/RS___Curtain_Wall___Type_FG___New/New/Doors_3135827",
        "family_short":     "CW_FG_New",
        "floor":            3,
        "role":             "train",
        "mesh_count":       2,
        "hinge_x_cm":       1724.33,
        "hinge_y_cm":       5878.48,
        "cam_pos":          [17.701, 56.807, 19.437],
        "cam_yaw_deg":      90.0,
        "leaf_mesh_names":  None,
        "angle_sign":       1,
    },
    {   # 5 — Type_F, floor 5, 42″ wide, OUTWARD-SWINGING
        # angle_sign=-1 rotates the leaf into the corridor toward the camera.
        # Only outward-swinging training door.
        "id":               "Doors_7171063",
        "prim_path":        f"{_BASE}/RS___Type_F_Door/NEW___HM_Frame___42_x96/Doors_7171063",
        "family_short":     "Type_F",
        "floor":            5,
        "role":             "train",
        "mesh_count":       0,
        "hinge_x_cm":       +58.42,
        "hinge_y_cm":       0.0,
        "cam_pos":          [-49.316, 49.892, 27.972],
        "cam_yaw_deg":      0.0,
        "leaf_mesh_names":  None,
        "angle_sign":       -1,
    },
    {   # 6 — Type_N, floor 3, local-space rotation
        # Parent matrix row1=[0,-1,0,0] (Y-axis flipped).
        # Frame baked into Paint mesh — unavoidable Revit export artefact.
        "id":               "Doors_12590968",
        "prim_path":        f"{_BASE}/RS___Type_N_Door/NEW___HM_Frame___36_x96/Doors_12590968",
        "family_short":     "Type_N",
        "floor":            3,
        "role":             "train",
        "mesh_count":       0,
        "hinge_x_cm":       +50.80,
        "hinge_y_cm":       0.0,
        "cam_pos":          [17.878, 50.941, 19.437],
        "cam_yaw_deg":      90.0,
        "leaf_mesh_names":  None,
        "angle_sign":       1,
    },
    {   # 7 — CW_FG_New, floor 2, world-space rotation (New___GL1_1 subfamily)
        "id":               "Doors_12302795",
        "prim_path":        f"{_BASE}/RS___Curtain_Wall___Type_FG___New/New___GL1_1/Doors_12302795",
        "family_short":     "CW_FG_New",
        "floor":            2,
        "role":             "train",
        "mesh_count":       2,
        "hinge_x_cm":       -5925.83,
        "hinge_y_cm":       5774.34,
        "cam_pos":          [-57.258, 58.251, 14.561],
        "cam_yaw_deg":      180.0,
        "leaf_mesh_names":  None,
        "angle_sign":       1,
    },
    {   # 8 — CW_F_NEW, floor 3, world-space rotation
        "id":               "Doors_13606056",
        "prim_path":        f"{_BASE}/RS___Curtain_Wall___Type_F___NEW/New/Doors_13606056",
        "family_short":     "CW_F_NEW",
        "floor":            3,
        "role":             "train",
        "mesh_count":       1,
        "hinge_x_cm":       -4747.75,
        "hinge_y_cm":       6958.30,
        "cam_pos":          [-45.477, 70.040, 19.437],
        "cam_yaw_deg":      180.0,
        "leaf_mesh_names":  None,
        "angle_sign":       1,
    },
    # ── HOLDOUT ──────────────────────────────────────────────────────────────
    {   # 9 — CW_FG_New, floor 4, world-space rotation  [HOLDOUT]
        # Instance-level generalization probe within the seen CW_FG_New family.
        "id":               "Doors_13182085",
        "prim_path":        f"{_BASE}/RS___Curtain_Wall___Type_FG___New/New/Doors_13182085",
        "family_short":     "CW_FG_New",
        "floor":            4,
        "role":             "holdout",
        "mesh_count":       2,
        "hinge_x_cm":       -4142.56,
        "hinge_y_cm":       8214.32,
        "cam_pos":          [-39.426, 82.600, 23.704],
        "cam_yaw_deg":      180.0,
        "leaf_mesh_names":  None,
        "angle_sign":       1,
    },
    {   # 10 — Type_F, floor 5, local-space rotation  [HOLDOUT]
        # Instance-level generalization probe within the seen Type_F family.
        # Parent [-1,-1] rotation; positive angle_sign opens into room.
        "id":               "Doors_1915613",
        "prim_path":        f"{_BASE}/RS___Type_F_Door/NEW___HM_Frame___36_x96/Doors_1915613",
        "family_short":     "Type_F",
        "floor":            5,
        "role":             "holdout",
        "mesh_count":       0,
        "hinge_x_cm":       +50.80,
        "hinge_y_cm":       0.0,
        "cam_pos":          [-58.753, 65.219, 27.972],
        "cam_yaw_deg":      270.0,
        "leaf_mesh_names":  None,
        "angle_sign":       1,
    },
]

# ---------------------------------------------------------------------------
# Isaac Sim startup
# ---------------------------------------------------------------------------

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

import time as _time
_t0 = _time.time()
def _ts(msg):
    print(f"[HingeTest] [+{_time.time() - _t0:6.1f}s] {msg}")

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
_fill_xf       = UsdGeom.Xformable(fill.GetPrim())
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
    prim_path="/World/HingeTestCam",
    position=np.array([0.0, 0.0, 5.0]),
    frequency=30,
    resolution=CAM_RESOLUTION,
    orientation=rot_utils.euler_angles_to_quats(
        np.array([0.0, 0.0, 180.0]), degrees=True),
)
simulation_app.update()
cam.initialize()

usd_cam = UsdGeom.Camera(stage.GetPrimAtPath("/World/HingeTestCam"))
usd_cam.GetFocalLengthAttr().Set(5.0)
usd_cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 10000.0))

rp  = rep.create.render_product("/World/HingeTestCam", CAM_RESOLUTION)
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


def _reposition_camera(pos, yaw_deg):
    ori = rot_utils.euler_angles_to_quats(
        np.array([0.0, 0.0, yaw_deg]), degrees=True)
    cam.set_world_pose(position=np.array(pos), orientation=ori)


def _capture_jpeg(family_short, door_id, floor, role, angle_deg) -> bool:
    world.step(render=True)
    data = ann.get_data()
    if data is None:
        print(f"  [WARN] annotator returned None")
        return False
    arr = np.asarray(data)
    if arr.ndim != 3 or arr.shape[-1] < 3:
        print(f"  [WARN] bad rgb shape {arr.shape}")
        return False
    rgb     = arr[:, :, :3].astype(np.uint8)
    cls     = "CLOSED" if angle_deg <= CLOSED_MAX_DEG else "OPEN"
    role_tag = "_HOLDOUT" if role == "holdout" else ""
    fname   = f"{family_short}_{door_id}_floor{floor}{role_tag}_ang{int(angle_deg):03d}_{cls}.jpg"
    path    = os.path.join(OUT_DIR, fname)
    if _pil_ok:
        _PIL.fromarray(rgb).save(path, "JPEG", quality=92)
    else:
        np.save(path.replace(".jpg", ".npy"), rgb)
    return True


def _discover_meshes_world(door_prim):
    """World-space mesh discovery for curtain-wall doors (mesh_count > 0)."""
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
    """Local-space mesh discovery for instanced doors (mesh_count == 0).

    leaf_mesh_names : if not None, only meshes whose prim name is in this set
                      are included.  Used to exclude fixed sidelite panels from
                      Type_F_W_SIDELITE_Door assemblies.
    """
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

# ---------------------------------------------------------------------------
# Per-door loop
# ---------------------------------------------------------------------------

summary = []
total = len(DOORS)

for di, door in enumerate(DOORS):
    door_id    = door["id"]
    fam_short  = door["family_short"]
    floor      = door["floor"]
    role       = door["role"]
    mesh_count       = door["mesh_count"]
    hx_cm            = door["hinge_x_cm"]
    hy_cm            = door["hinge_y_cm"]
    prim_path        = door["prim_path"]
    leaf_mesh_names  = door.get("leaf_mesh_names", None)
    angle_sign       = door.get("angle_sign", 1)

    role_label = "[HOLDOUT]" if role == "holdout" else "[train  ]"
    print(f"\n{'─'*60}")
    _ts(f"[{di+1:2d}/{total}] {role_label} [{fam_short:20s}] {door_id}  floor={floor}")
    _ts(f"  hinge=({hx_cm:.2f}, {hy_cm:.2f}) cm  "
        f"rotation={'world' if mesh_count > 0 else 'local'}-space  "
        f"angle_sign={angle_sign:+d}  "
        f"leaf_filter={leaf_mesh_names}")

    entry = {
        "door_id":     door_id,
        "family":      fam_short,
        "floor":       floor,
        "role":        role,
        "mesh_count":  mesh_count,
        "hinge":       [hx_cm, hy_cm],
        "angles":      {},
        "status":      "ok",
        "note":        "",
    }

    # -- Validate prim --
    door_prim = stage.GetPrimAtPath(prim_path)
    if not door_prim.IsValid():
        _ts(f"  SKIP — prim not found: {prim_path}")
        entry.update({"status": "skipped", "note": "prim not valid"})
        summary.append(entry)
        continue

    # -- Concretize if instanced --
    if door_prim.IsInstance() or door_prim.IsInstanceable():
        door_prim.SetInstanceable(False)
        simulation_app.update()

    # -- Discover meshes --
    if mesh_count > 0:
        mesh_ops = _discover_meshes_world(door_prim)
    else:
        mesh_ops = _discover_meshes_local(door_prim, leaf_mesh_names)

    if not mesh_ops:
        _ts(f"  SKIP — no xformOp:transform meshes found")
        entry.update({"status": "skipped", "note": "no xform meshes"})
        summary.append(entry)
        continue

    _ts(f"  {len(mesh_ops)} mesh(es) discovered")

    # -- Position fill light above door --
    cam_pos = door["cam_pos"]
    _fill_translate.Set(Gf.Vec3d(cam_pos[0], cam_pos[1], cam_pos[2] + 1.5))

    # -- Position camera --
    _reposition_camera(cam_pos, door["cam_yaw_deg"])

    # -- Render each angle --
    for angle_deg in ANGLE_GRID:
        signed_angle = angle_sign * angle_deg
        cls = "CLOSED" if angle_deg <= CLOSED_MAX_DEG else "OPEN"
        _set_angle(mesh_ops, signed_angle, hx_cm, hy_cm)
        for _ in range(WARMUP_STEPS):
            world.step(render=True)
        ok = _capture_jpeg(fam_short, door_id, floor, role, angle_deg)
        entry["angles"][str(int(angle_deg))] = "ok" if ok else "failed"
        _ts(f"  {cls:6s}  {angle_deg:4.0f}° (signed {signed_angle:+.1f}°)  {'saved' if ok else 'FAILED'}")

    # -- Reset to closed --
    _set_angle(mesh_ops, 0.0, hx_cm, hy_cm)

    any_failed = any(v == "failed" for v in entry["angles"].values())
    if any_failed:
        entry["status"] = "partial"
        entry["note"]   = "one or more angles failed to capture"

    summary.append(entry)

    # Running progress
    done   = di + 1
    saved  = sum(
        1 for e in summary
        for v in e.get("angles", {}).values() if v == "ok"
    )
    _ts(f"  → {done}/{total} doors done  {saved} JPEGs saved  {total - done} remaining")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

ok_count   = sum(1 for e in summary if e["status"] == "ok")
skip_count = sum(1 for e in summary if e["status"] == "skipped")
part_count = sum(1 for e in summary if e["status"] == "partial")
total_jpegs = sum(
    1 for e in summary for v in e.get("angles", {}).values() if v == "ok"
)

summary_path = os.path.join(_project_dir, "data", "door_hinge_test_summary.json")
with open(summary_path, "w") as f:
    json.dump({
        "ok": ok_count, "partial": part_count, "skipped": skip_count,
        "total_jpegs": total_jpegs,
        "doors": summary,
    }, f, indent=2)
    f.flush()
    os.fsync(f.fileno())

print(f"\n{'='*60}")
_ts(f"Done — {ok_count} doors OK  {part_count} partial  {skip_count} skipped")
_ts(f"Total JPEGs saved : {total_jpegs}  (expected {len(DOORS) * len(ANGLE_GRID)})")
_ts(f"JPEGs   → {OUT_DIR}/")
_ts(f"Summary → {summary_path}")

world.stop()
simulation_app.close()

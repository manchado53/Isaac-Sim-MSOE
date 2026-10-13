#!/usr/bin/env python3
"""
door_rotation_diagnostic.py — find which prim/axis actually rotates Door 5.

1. Walks the USD prim hierarchy under Doors_13606713, printing every prim,
   type, and existing xform ops → data/door_rot_diagnostic/hierarchy.txt

2. Checks the world bounding box of the door prim + its children at 0° / 90°
   (using the existing rotateZ:doorAngle method) to see if the geometry moves
   at all — diagnosis without needing the camera.

3. Takes a baseline camera frame, then for the door prim AND each direct child
   tries rotating 45° and 90° around each of Z / X / Y and saves a JPEG so
   you can visually confirm which combination moves the glass panel.

Output: data/door_rot_diagnostic/
  hierarchy.txt                        — full prim tree + xform op values
  baseline.jpg                         — door with no rotation
  <prim>_rot<axis>_<angle>deg.jpg     — one image per rotation test

Usage:
  sbatch scripts/slurm/submit_isaac_job.sh simulations/robot/door_rotation_diagnostic.py
"""

import os
import sys
import numpy as np
from PIL import Image

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

import omni.usd
from pxr import Usd, UsdLux, UsdGeom, Gf, Sdf

import isaacsim.core.utils.numpy.rotations as rot_utils
from isaacsim.core.api import World
from isaacsim.sensors.camera import Camera

_script_dir  = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.abspath(os.path.join(_script_dir, "..", ".."))

out_dir = os.path.join(_project_dir, "data", "door_rot_diagnostic4")
os.makedirs(out_dir, exist_ok=True)
print(f"[DoorDiag] Output → {out_dir}")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DOOR5_PATH = (
    "/World/Building/_22_129___MSOE_Engineering_Building___A___2_25_2026"
    "/Geometry/Doors/RS___Curtain_Wall___Type_FG___New/New/Doors_13606713"
)

# Camera: 2 m east of Door 5 at (-20.48, 80.15), facing west (-X)
CAM_POS = np.array([-18.48, 80.15, 27.8])
CAM_ORI = rot_utils.euler_angles_to_quats(np.array([0.0, 0.0, 180.0]), degrees=True)

TEST_ANGLES = [45.0, 90.0]
TEST_AXES   = ["Z", "X", "Y"]

_BBOX_PURPOSES = [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]

# ---------------------------------------------------------------------------
# Scene setup
# ---------------------------------------------------------------------------

building_usd = os.path.join(_project_dir, "innovation_center_finalV1.usd")
if not os.path.isfile(building_usd):
    print(f"ERROR: {building_usd} not found")
    simulation_app.close()
    sys.exit(1)

world = World(stage_units_in_meters=1.0)
stage = omni.usd.get_context().get_stage()
stage.GetRootLayer().subLayerPaths.insert(0, building_usd)

dome = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/DomeLight"))
dome.CreateIntensityAttr(50000)
fill = UsdLux.SphereLight.Define(stage, Sdf.Path("/World/FillLight"))
fill.CreateIntensityAttr(300000)
fill.CreateRadiusAttr(0.1)
UsdGeom.Xformable(fill.GetPrim()).AddTranslateOp().Set(
    (float(CAM_POS[0]), float(CAM_POS[1]), float(CAM_POS[2] + 1.5))
)

cam = Camera(
    prim_path="/World/DiagCam",
    position=CAM_POS,
    frequency=30,
    resolution=(640, 480),
    orientation=CAM_ORI,
)

world.reset()
simulation_app.update()
cam.initialize()

usd_cam = UsdGeom.Camera(stage.GetPrimAtPath("/World/DiagCam"))
usd_cam.GetFocalLengthAttr().Set(5.0)
usd_cam.GetClippingRangeAttr().Set((0.01, 10000.0))

print("[DoorDiag] Warming up renderer (30 steps)...")
for _ in range(30):
    world.step(render=True)

# ---------------------------------------------------------------------------
# Confirm door prim
# ---------------------------------------------------------------------------

door_prim = stage.GetPrimAtPath(DOOR5_PATH)
if not door_prim.IsValid():
    print(f"ERROR: prim not found — {DOOR5_PATH}")
    simulation_app.close()
    sys.exit(1)
print(f"[DoorDiag] Door prim found: {DOOR5_PATH}  type={door_prim.GetTypeName()}")

# ---------------------------------------------------------------------------
# Part 1: Hierarchy walk
# ---------------------------------------------------------------------------

def walk_prim(prim, depth=0, lines=None):
    if lines is None:
        lines = []
    indent    = "  " * depth
    path      = str(prim.GetPath())
    prim_type = prim.GetTypeName()
    xform_info = ""
    if prim.IsA(UsdGeom.Xformable):
        ops = UsdGeom.Xformable(prim).GetOrderedXformOps()
        if ops:
            parts = []
            for op in ops:
                try:
                    parts.append(f"{op.GetOpName()}={op.Get()}")
                except Exception:
                    parts.append(op.GetOpName())
            xform_info = "  ops=[" + ", ".join(parts) + "]"
    line = f"{indent}{path}  [{prim_type}]{xform_info}"
    lines.append(line)
    print(line)
    for child in prim.GetChildren():
        walk_prim(child, depth + 1, lines)
    return lines

print(f"\n{'='*70}")
print("PART 1 — HIERARCHY")
print(f"{'='*70}")
hier_lines = walk_prim(door_prim)
hier_path  = os.path.join(out_dir, "hierarchy.txt")
with open(hier_path, "w") as f:
    f.write("\n".join(hier_lines))
print(f"\n[DoorDiag] hierarchy.txt saved.")

# ---------------------------------------------------------------------------
# Part 2: Bbox check using the existing rotateZ:doorAngle method
# ---------------------------------------------------------------------------

def get_world_bbox(prim):
    cache   = UsdGeom.BBoxCache(Usd.TimeCode.Default(), _BBOX_PURPOSES, useExtentsHint=False)
    bbox    = cache.ComputeWorldBound(prim)
    aligned = bbox.ComputeAlignedRange()
    return aligned.GetMidpoint(), aligned.GetMin(), aligned.GetMax()

def get_world_translation(prim):
    mat = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(prim)
    return mat.ExtractTranslation()

def get_or_create_rotatez(xformable):
    for op in xformable.GetOrderedXformOps():
        if op.GetOpName() == "xformOp:rotateZ:doorAngle":
            return op
    return xformable.AddRotateZOp(UsdGeom.XformOp.PrecisionFloat, "doorAngle")

print(f"\n{'='*70}")
print("PART 2 — BBOX CHECK (rotateZ:doorAngle on parent prim)")
print(f"{'='*70}")

xformable_door = UsdGeom.Xformable(door_prim)
rotate_op      = get_or_create_rotatez(xformable_door)

prev_center = None
for angle in [0.0, 45.0, 90.0]:
    rotate_op.Set(float(angle))
    simulation_app.update()
    center, bmin, bmax = get_world_bbox(door_prim)
    world_t            = get_world_translation(door_prim)
    print(f"\nAngle {angle:.0f}°:")
    print(f"  world_translation = ({world_t[0]:.4f}, {world_t[1]:.4f}, {world_t[2]:.4f})")
    print(f"  bbox_center       = ({center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f})")
    print(f"  bbox_min          = ({bmin[0]:.4f}, {bmin[1]:.4f}, {bmin[2]:.4f})")
    print(f"  bbox_max          = ({bmax[0]:.4f}, {bmax[1]:.4f}, {bmax[2]:.4f})")
    if prev_center is not None:
        delta = Gf.Vec3d(center[0] - prev_center[0],
                         center[1] - prev_center[1],
                         center[2] - prev_center[2])
        moved = delta.GetLength()
        status = "OK — geometry moved" if moved > 0.01 else "WARNING — geometry did NOT move"
        print(f"  bbox moved {moved:.4f} m from previous  [{status}]")
    prev_center = center

# Reset to 0 before camera tests
rotate_op.Set(0.0)
simulation_app.update()

# Also check direct children at 0° vs 90°
print(f"\nChild world translations at 0° vs 90° (rotateZ:doorAngle on parent):")
for test_angle in [0.0, 90.0]:
    rotate_op.Set(float(test_angle))
    simulation_app.update()
    print(f"  {test_angle:.0f}°:")
    for child in door_prim.GetChildren():
        ct = get_world_translation(child)
        print(f"    {child.GetPath().name}: ({ct[0]:.4f}, {ct[1]:.4f}, {ct[2]:.4f})")

rotate_op.Set(0.0)
simulation_app.update()

# ---------------------------------------------------------------------------
# Part 3: Camera frames — baseline + prim × axis × angle
# ---------------------------------------------------------------------------

def capture(tag: str):
    for _ in range(4):
        world.step(render=True)
    rgba = cam.get_rgba()
    if rgba is None:
        print(f"  [WARN] Camera returned None — skipping {tag}")
        return
    path = os.path.join(out_dir, f"{tag}.jpg")
    Image.fromarray(rgba[:, :, :3].astype(np.uint8)).save(path, "JPEG", quality=92)
    print(f"[DoorDiag] Saved: {tag}.jpg")

def apply_rot(prim, axis: str, angle_deg: float):
    suffix   = f"diagRot{axis}"
    fullname = f"xformOp:rotate{axis}:{suffix}"
    xf       = UsdGeom.Xformable(prim)
    for op in xf.GetOrderedXformOps():
        if op.GetOpName() == fullname:
            op.Set(float(angle_deg))
            return
    adders = {"Z": xf.AddRotateZOp, "X": xf.AddRotateXOp, "Y": xf.AddRotateYOp}
    adders[axis](UsdGeom.XformOp.PrecisionFloat, suffix).Set(float(angle_deg))

def zero_rot(prim, axis: str):
    suffix   = f"diagRot{axis}"
    fullname = f"xformOp:rotate{axis}:{suffix}"
    for op in UsdGeom.Xformable(prim).GetOrderedXformOps():
        if op.GetOpName() == fullname:
            op.Set(0.0)
            return

print(f"\n{'='*70}")
print("PART 3 — CAMERA FRAMES")
print(f"{'='*70}")

# Baseline
capture("baseline")

# Test: door prim + direct children
candidates = [door_prim] + list(door_prim.GetChildren())
print(f"[DoorDiag] {len(candidates)} prims × {len(TEST_AXES)} axes × {len(TEST_ANGLES)} angles")

for prim in candidates:
    if not prim.IsA(UsdGeom.Xformable):
        continue
    tag_base = str(prim.GetPath()).split("/")[-1].replace(" ", "_")
    print(f"\n  Prim: {prim.GetPath()}")
    for axis in TEST_AXES:
        for angle in TEST_ANGLES:
            apply_rot(prim, axis, angle)
            simulation_app.update()
            capture(f"{tag_base}_rot{axis}_{int(angle)}deg")
        zero_rot(prim, axis)
        simulation_app.update()

print(f"\n{'='*70}")
print("PART 4 — COMBINED LEAF SWEEP (AL + Glass together, rotZ only)")
print(f"{'='*70}")
# Rotate both Mesh children simultaneously at each angle so we can see how
# the full door leaf looks across the entire open range.
SWEEP_ANGLES = [10.0, 20.0, 30.0, 45.0, 60.0, 75.0, 90.0]

leaf_meshes = [c for c in door_prim.GetChildren() if c.IsA(UsdGeom.Mesh)]
print(f"[DoorDiag] Rotating {len(leaf_meshes)} mesh children together: "
      f"{[str(m.GetPath().name) for m in leaf_meshes]}")

for angle in SWEEP_ANGLES:
    for mesh in leaf_meshes:
        apply_rot(mesh, "Z", angle)
    simulation_app.update()
    capture(f"combined_leaf_rotZ_{int(angle)}deg")

# Reset all
for mesh in leaf_meshes:
    zero_rot(mesh, "Z")
simulation_app.update()

print(f"\n{'='*70}")
print("PART 5 — HINGE JOINT ATTRIBUTES + PIVOT-CORRECTED ROTATION TEST")
print(f"{'='*70}")
# Pivot-corrected rotation:
#   Instead of appending rotateZ (which spins around the mesh center),
#   we update xformOp:transform directly with:
#
#   M_new = T(-hinge) * Rz(θ) * T(+hinge) * T_world   (row-vector convention)
#
# Translation row derivation:
#   new_tx = tx + hx*(1 - cosθ) + hy*sinθ
#   new_ty = ty - hx*sinθ       + hy*(1 - cosθ)
#
# We test all 4 bbox-edge candidates for (hx, hy) so the images reveal
# which edge is the actual hinge.

import math

# ---- Read HingeJoint attributes ----
print("\n--- HingeJoint attributes ---")
hinge_prim = None
for _child in door_prim.GetChildren():
    if _child.GetTypeName() == "PhysicsRevoluteJoint":
        hinge_prim = _child
        break

if hinge_prim:
    print(f"HingeJoint prim: {hinge_prim.GetPath()}")
    for _attr in hinge_prim.GetAttributes():
        if _attr.IsAuthored():
            print(f"  {_attr.GetName()} = {_attr.Get()}")
else:
    print("WARNING: No PhysicsRevoluteJoint child found under door prim")

# ---- Mesh local-space bounding box ----
print("\n--- Mesh local-space vertex bbox ---")
_leaf_meshes_p5 = [c for c in door_prim.GetChildren() if c.IsA(UsdGeom.Mesh)]
_mesh_bbox = {}   # mesh_name → {"x": (min,max), "y": (min,max)}
for _m in _leaf_meshes_p5:
    _pts_attr = _m.GetAttribute("points")
    if _pts_attr and _pts_attr.IsAuthored():
        _pts = _pts_attr.Get()
        if _pts:
            _xs = [p[0] for p in _pts]
            _ys = [p[1] for p in _pts]
            _zs = [p[2] for p in _pts]
            _mesh_bbox[str(_m.GetPath().name)] = {
                "x": (min(_xs), max(_xs)),
                "y": (min(_ys), max(_ys)),
            }
            print(f"  {_m.GetPath().name}:")
            print(f"    X: [{min(_xs):.3f}, {max(_xs):.3f}]  width={max(_xs)-min(_xs):.3f}")
            print(f"    Y: [{min(_ys):.3f}, {max(_ys):.3f}]  depth={max(_ys)-min(_ys):.3f}")
            print(f"    Z: [{min(_zs):.3f}, {max(_zs):.3f}]  height={max(_zs)-min(_zs):.3f}")

# ---- Capture the original (0°) xformOp:transform for each leaf mesh ----
_base_M_p5 = {}   # mesh path → Gf.Matrix4d
for _m in _leaf_meshes_p5:
    for _op in UsdGeom.Xformable(_m).GetOrderedXformOps():
        if _op.GetOpName() == "xformOp:transform":
            _base_M_p5[str(_m.GetPath())] = _op.Get()
            break


def _apply_pivot_rotZ(hx, hy, angle_deg):
    """Update each leaf mesh's xformOp:transform with pivot-corrected rotation."""
    _cos = math.cos(math.radians(angle_deg))
    _sin = math.sin(math.radians(angle_deg))
    for _m in _leaf_meshes_p5:
        _M0 = _base_M_p5.get(str(_m.GetPath()))
        if _M0 is None:
            continue
        _tx, _ty, _tz = _M0[3][0], _M0[3][1], _M0[3][2]
        _ntx = _tx + hx * (1.0 - _cos) + hy * _sin
        _nty = _ty - hx * _sin          + hy * (1.0 - _cos)
        _new_M = Gf.Matrix4d()
        _new_M.SetRow(0, Gf.Vec4d( _cos,  _sin, 0.0, 0.0))
        _new_M.SetRow(1, Gf.Vec4d(-_sin,  _cos, 0.0, 0.0))
        _new_M.SetRow(2, Gf.Vec4d(  0.0,   0.0, 1.0, 0.0))
        _new_M.SetRow(3, Gf.Vec4d(_ntx,   _nty, _tz, 1.0))
        for _op in UsdGeom.Xformable(_m).GetOrderedXformOps():
            if _op.GetOpName() == "xformOp:transform":
                _op.Set(_new_M)
                break


def _reset_transforms_p5():
    """Restore original base transforms for all leaf meshes."""
    for _m in _leaf_meshes_p5:
        _M0 = _base_M_p5.get(str(_m.GetPath()))
        if _M0 is None:
            continue
        for _op in UsdGeom.Xformable(_m).GetOrderedXformOps():
            if _op.GetOpName() == "xformOp:transform":
                _op.Set(_M0)
                break


# ---- Determine test pivots from mesh bbox ----
# Take bbox from first available mesh (both AL and Glass share same XY translation)
_first_bbox = next(iter(_mesh_bbox.values()), None) if _mesh_bbox else None
if _first_bbox:
    _PIVOT_TESTS = {
        "hinge_minX": (_first_bbox["x"][0], 0.0),
        "hinge_maxX": (_first_bbox["x"][1], 0.0),
        "hinge_minY": (0.0, _first_bbox["y"][0]),
        "hinge_maxY": (0.0, _first_bbox["y"][1]),
    }
else:
    print("[DoorDiag] WARN: No mesh bbox — skipping pivot tests")
    _PIVOT_TESTS = {}

_PIVOT_ANGLES = [45.0, 90.0]

# Capture baseline before any pivot tests
_reset_transforms_p5()
simulation_app.update()
capture("p5_baseline")

for _pname, (_hx, _hy) in _PIVOT_TESTS.items():
    print(f"\n  Pivot: {_pname} = ({_hx:.3f}, {_hy:.3f})")
    for _ang in _PIVOT_ANGLES:
        _apply_pivot_rotZ(_hx, _hy, _ang)
        simulation_app.update()
        capture(f"p5_{_pname}_{int(_ang)}deg")
    _reset_transforms_p5()
    simulation_app.update()

print(f"\n[DoorDiag] Done. Check {out_dir}/")
world.stop()
simulation_app.close()

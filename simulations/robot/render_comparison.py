#!/usr/bin/env python3
"""
Render Comparison — RTX Real-Time vs RTX Path Tracing

Captures the same door from the same camera pose using both Isaac Sim
renderers and saves the two images side by side for visual comparison.

The RTX renderer is selected at SimulationApp construction time and
cannot be switched at runtime, so this script re-executes itself as a
subprocess once per renderer. Running it without arguments produces
both images in a single invocation:

  ~/isaacsim/python.sh simulations/robot/render_comparison.py

Output: data/render_comparison/<door_id>_<side>_realtime.jpg
        data/render_comparison/<door_id>_<side>_pathtracing.jpg

The door, side, and lighting rig match the conventions used by
door_snapshot.py so the two images are directly comparable to the
existing training snapshots.
"""

import os
import sys
import subprocess

# ---------------------------------------------------------------------------
# Config (shared by both renderer passes)
# ---------------------------------------------------------------------------

TARGET_DOOR_ID = "Doors_13606713"   # original training door
CAMERA_DIST    = 2.5                # metres from door centre
CAM_RESOLUTION = (1280, 960)        # higher res to show path-tracing detail
PT_SAMPLES     = 64                 # samples per pixel per frame (path tracing)
PT_SETTLE_STEPS = 64                # render steps before grabbing the frame

_script_dir  = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.abspath(os.path.join(_script_dir, "..", ".."))

OUT_DIR = os.path.join(_project_dir, "data", "render_comparison")

# Per-door side/flip config (must match door_snapshot.py)
DOOR_CONFIG = {
    "Doors_9817744":  {"sides": {"north"},  "flip": True},
    "Doors_12066864": {"sides": {"north"},  "flip": False},
    "Doors_13607620": {"sides": {"north"},  "flip": True},
    "Doors_3135827":  {"sides": {"south"},  "flip": False},
    "Doors_1868517":  {"sides": {"north"},  "flip": True},
    "Doors_10383956": {"sides": {"west"},   "flip": True},
    "Doors_13182085": {"sides": {"east"},   "flip": False},
    "Doors_4190126":  {"sides": {"west"},   "flip": False},
    "Doors_13606713": {"sides": {"east"},   "flip": False},
}

# ---------------------------------------------------------------------------
# Parent mode: re-execute twice, once per renderer
# ---------------------------------------------------------------------------

def run_both_renderers():
    os.makedirs(OUT_DIR, exist_ok=True)
    this_script = os.path.abspath(__file__)
    python_bin  = sys.executable
    for mode in ("realtime", "pathtracing"):
        print(f"\n{'#' * 70}")
        print(f"# Launching renderer pass: {mode}")
        print(f"{'#' * 70}\n")
        result = subprocess.run(
            [python_bin, this_script, "--mode", mode],
            cwd=_project_dir,
        )
        if result.returncode != 0:
            print(f"[RenderCompare] Pass '{mode}' exited with code {result.returncode}")
            sys.exit(result.returncode)
    print("\n[RenderCompare] Both passes complete.")
    print(f"Output directory: {OUT_DIR}")
    sys.exit(0)


if __name__ == "__main__" and "--mode" not in sys.argv:
    run_both_renderers()

# ---------------------------------------------------------------------------
# Child mode: actually render one image with the chosen renderer
# ---------------------------------------------------------------------------

mode = sys.argv[sys.argv.index("--mode") + 1]
assert mode in ("realtime", "pathtracing"), f"unknown --mode {mode}"

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

from isaacsim import SimulationApp

if mode == "pathtracing":
    simulation_app = SimulationApp({
        "headless": True,
        "renderer": "PathTracing",
        "samples_per_pixel_per_frame": PT_SAMPLES,
    })
else:
    simulation_app = SimulationApp({"headless": True})

import numpy as np
import omni.usd
from pxr import Usd, UsdLux, UsdGeom, Gf, Sdf

import isaacsim.core.utils.numpy.rotations as rot_utils
from isaacsim.core.api import World
from isaacsim.sensors.camera import Camera

building_usd = os.path.join(_project_dir, "MSOEBuildingV4_physics.usd")
if not os.path.isfile(building_usd):
    print(f"ERROR: building USD not found: {building_usd}")
    simulation_app.close()
    sys.exit(1)

world = World(stage_units_in_meters=1.0)
stage = omni.usd.get_context().get_stage()
stage.GetRootLayer().subLayerPaths.insert(0, building_usd)

# Ambient fill light
dome = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/DomeLight"))
dome.CreateIntensityAttr(5000)

world.reset()
simulation_app.update()

print(f"[RenderCompare:{mode}] Warming up renderer (20 steps) ...")
for _ in range(20):
    world.step(render=True)

# ---------------------------------------------------------------------------
# Locate target door
# ---------------------------------------------------------------------------

print(f"[RenderCompare:{mode}] Scanning for door '{TARGET_DOOR_ID}' ...")

bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_],
                               useExtentsHint=False)

target_prim = None
for prim in stage.TraverseAll():
    if prim.GetName() != TARGET_DOOR_ID:
        continue
    if "/Doors/" not in prim.GetPath().pathString:
        continue
    if prim.GetTypeName() not in ("Xform", ""):
        continue
    target_prim = prim
    break

if target_prim is None:
    print(f"ERROR: could not find door {TARGET_DOOR_ID} under /Doors/")
    simulation_app.close()
    sys.exit(1)

bbox = bbox_cache.ComputeWorldBound(target_prim)
bbox_range = bbox.ComputeAlignedRange()
bb_mid  = bbox_range.GetMidpoint()
bb_size = bbox_range.GetSize()

cx = float(bb_mid[0])
cy = float(bb_mid[1])
cz = float(bb_mid[2])
cam_z = cz

x_extent = float(bb_size[0])
y_extent = float(bb_size[1])
approach_axis = "X" if x_extent < y_extent else "Y"

print(f"[RenderCompare:{mode}] Door centre = ({cx:.2f}, {cy:.2f}, {cz:.2f}), "
      f"axis={approach_axis}")

# ---------------------------------------------------------------------------
# Per-door lighting grid (same as door_snapshot.py)
# ---------------------------------------------------------------------------

light_idx = 0
if approach_axis == "X":
    approach_offsets = [-2.5, -1.0, 0.0, 1.0, 2.5]
    span_offsets     = [-2.0, 0.0, 2.0]
    for x_off in approach_offsets:
        for y_off in span_offsets:
            lp = f"/World/HLight_{light_idx}"
            lt = UsdLux.RectLight.Define(stage, Sdf.Path(lp))
            lt.CreateIntensityAttr(50000)
            lt.CreateWidthAttr(1.5)
            lt.CreateHeightAttr(0.8)
            lx = UsdGeom.Xformable(lt.GetPrim())
            lx.AddTranslateOp().Set(Gf.Vec3d(cx + x_off, cy + y_off, cam_z + 1.2))
            lx.AddRotateXOp().Set(-90.0)
            light_idx += 1
else:
    span_offsets     = [-2.0, 0.0, 2.0]
    approach_offsets = [-2.5, -1.0, 0.0, 1.0, 2.5]
    for x_off in span_offsets:
        for y_off in approach_offsets:
            lp = f"/World/HLight_{light_idx}"
            lt = UsdLux.RectLight.Define(stage, Sdf.Path(lp))
            lt.CreateIntensityAttr(50000)
            lt.CreateWidthAttr(1.5)
            lt.CreateHeightAttr(0.8)
            lx = UsdGeom.Xformable(lt.GetPrim())
            lx.AddTranslateOp().Set(Gf.Vec3d(cx + x_off, cy + y_off, cam_z + 1.2))
            lx.AddRotateXOp().Set(-90.0)
            light_idx += 1

simulation_app.update()
print(f"[RenderCompare:{mode}] Placed {light_idx} rect lights")

# ---------------------------------------------------------------------------
# Pick the camera side from DOOR_CONFIG
# ---------------------------------------------------------------------------

if approach_axis == "X":
    all_sides = {
        "east": (cx + CAMERA_DIST, cy, 180.0),
        "west": (cx - CAMERA_DIST, cy,   0.0),
    }
else:
    all_sides = {
        "north": (cx, cy + CAMERA_DIST, 270.0),
        "south": (cx, cy - CAMERA_DIST,  90.0),
    }

cfg = DOOR_CONFIG.get(TARGET_DOOR_ID, {})
allowed = cfg.get("sides")
if allowed:
    side_label = sorted(allowed & all_sides.keys())[0]
else:
    side_label = sorted(all_sides.keys())[0]

cam_x, cam_y, yaw_deg = all_sides[side_label]
print(f"[RenderCompare:{mode}] Using side '{side_label}' at "
      f"({cam_x:.2f}, {cam_y:.2f}, {cam_z:.2f}), yaw={yaw_deg}")

# ---------------------------------------------------------------------------
# Create camera and render
# ---------------------------------------------------------------------------

cam_pos = np.array([cam_x, cam_y, cam_z])
cam_ori = rot_utils.euler_angles_to_quats(
    np.array([0.0, 0.0, yaw_deg]), degrees=True
)

prim_path = f"/World/CompareCam_{mode}"
cam = Camera(
    prim_path=prim_path,
    position=cam_pos,
    frequency=30,
    resolution=CAM_RESOLUTION,
    orientation=cam_ori,
)
simulation_app.update()
cam.initialize()

usd_cam = UsdGeom.Camera(stage.GetPrimAtPath(prim_path))
usd_cam.GetFocalLengthAttr().Set(5.0)
usd_cam.GetClippingRangeAttr().Set((0.01, 10000.0))

# Path tracing needs many more settle steps to accumulate samples
settle_steps = PT_SETTLE_STEPS if mode == "pathtracing" else 20
print(f"[RenderCompare:{mode}] Rendering {settle_steps} settle steps ...")
for _ in range(settle_steps):
    world.step(render=True)

rgba = cam.get_rgba()
if rgba is None:
    print(f"[RenderCompare:{mode}] ERROR: camera returned None")
    simulation_app.close()
    sys.exit(1)

rgb = rgba[:, :, :3].astype(np.uint8)

os.makedirs(OUT_DIR, exist_ok=True)
filename = f"{TARGET_DOOR_ID}_{side_label}_{mode}.jpg"
filepath = os.path.join(OUT_DIR, filename)

try:
    from PIL import Image as _PIL
    _PIL.fromarray(rgb).save(filepath, "JPEG", quality=95)
    print(f"[RenderCompare:{mode}] Saved {filepath}")
except ImportError:
    np.save(filepath.replace(".jpg", ".npy"), rgb)
    print(f"[RenderCompare:{mode}] PIL unavailable — saved raw .npy instead")

world.stop()
simulation_app.close()

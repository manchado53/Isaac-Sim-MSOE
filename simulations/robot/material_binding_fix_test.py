#!/usr/bin/env python3
"""
Material Binding Fix Test — prove that one-material-per-color fixes the stuck
shader problem observed in material_binding_diagnostic.py.

Diagnosis from job 242579:
  - binding DOES take effect (ComputeBoundMaterial returns /World/Looks/DiagDoorMat)
  - textured Paint material IS replaced by flat color
  - BUT mutating `diffuse_color_constant` between renders produces the SAME red
    for every color we ask for: hot_pink, bright_green, brushed_aluminum, dark
    anodized all render identically (mean_rgb varies by <10 units total)

Hypothesis:
  Mutating a shared OmniPBR shader input does not invalidate the material cache
  the render delegate holds. The first color sticks. Fix: pre-create a distinct
  UsdShade.Material + Shader per color variant, and switch *bindings* between
  renders instead of *inputs*.

Target: `Doors_4070013` (Type_F_Door, single `Paint` mesh, no glass).
Camera: 0.9 m along +Y, 24 mm focal (frame dominated by door leaf).

Output: data/material_binding_fix/<NN>_<label>.jpg

Run:
  sbatch --partition=teaching submit_isaac_job.sh \\
      simulations/robot/material_binding_fix_test.py
"""

import os
import sys

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

import carb
_carb = carb.settings.get_settings()
_carb.set("/rtx/post/aa/op", 4)
_carb.set("/rtx/post/dlss/execMode", 0)

import numpy as np
import omni.usd
from pxr import Usd, UsdLux, UsdGeom, UsdShade, Gf, Sdf

import isaacsim.core.utils.numpy.rotations as rot_utils
from isaacsim.core.api import World
from isaacsim.sensors.camera import Camera

_script_dir  = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.abspath(os.path.join(_script_dir, "..", ".."))

DOOR_PATH = (
    "/World/Building/_22_129___MSOE_Engineering_Building___A___2_25_2026"
    "/Geometry/Doors/RS___Type_F_Door/NEW___HM_Frame___36_x96/Doors_4070013"
)
DOOR_CENTER = np.array([-17.5325, 55.4272, 23.7490])
CAM_DIST = 0.9
CAM_FOCAL_MM = 24.0
CAM_RESOLUTION = (1024, 768)

OUT_DIR = os.path.join(_project_dir, "data", "material_binding_fix")

# Distinct test colors — should be trivially distinguishable if the fix works.
TEST_COLORS = [
    ("brushed_aluminum", (0.75, 0.75, 0.78), 0.35, 0.85),
    ("dark_anodized",    (0.05, 0.05, 0.08), 0.25, 0.80),
    ("bright_green",     (0.10, 0.85, 0.15), 0.55, 0.00),
    ("hot_pink",         (0.95, 0.15, 0.60), 0.40, 0.00),
    ("pure_white",       (0.98, 0.98, 0.98), 0.70, 0.00),
    ("deep_blue",        (0.08, 0.15, 0.85), 0.45, 0.00),
]

os.makedirs(OUT_DIR, exist_ok=True)

world = World(stage_units_in_meters=1.0)
stage = omni.usd.get_context().get_stage()

building_usd = os.path.join(_project_dir, "MSOEBuildingV4_physics.usd")
stage.GetRootLayer().subLayerPaths.insert(0, building_usd)

# Lighting
dome = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/DomeLight"))
dome.CreateIntensityAttr(15_000)
dome.CreateColorTemperatureAttr(5500)
dome.CreateEnableColorTemperatureAttr(True)

fill = UsdLux.SphereLight.Define(stage, Sdf.Path("/World/FillLight"))
fill.CreateIntensityAttr(80_000)
fill.CreateColorTemperatureAttr(4500)
fill.CreateEnableColorTemperatureAttr(True)
fill.CreateRadiusAttr(0.3)
UsdGeom.Xformable(fill.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(
    float(DOOR_CENTER[0]), float(DOOR_CENTER[1]), float(DOOR_CENTER[2]) + 1.5
))

world.reset()
simulation_app.update()

# Camera
cam = Camera(
    prim_path="/World/FixTestCam",
    position=DOOR_CENTER + np.array([0, CAM_DIST, 0]),
    frequency=30,
    resolution=CAM_RESOLUTION,
    orientation=rot_utils.euler_angles_to_quats(np.array([0.0, 0.0, 270.0]), degrees=True),
)
simulation_app.update()
cam.initialize()
usd_cam = UsdGeom.Camera(stage.GetPrimAtPath("/World/FixTestCam"))
usd_cam.GetFocalLengthAttr().Set(CAM_FOCAL_MM)
usd_cam.GetClippingRangeAttr().Set((0.01, 10000.0))

print("[FixTest] Warming up ...")
for _ in range(20):
    world.step(render=True)

# Pre-create ONE UsdShade.Material PER color — no input mutation
def make_material(name, color, roughness, metallic):
    mat_path = f"/World/Looks/Fix_{name}"
    mat = UsdShade.Material.Define(stage, Sdf.Path(mat_path))
    shader = UsdShade.Shader.Define(stage, Sdf.Path(mat_path + "/Shader"))
    shader.CreateIdAttr("OmniPBR")
    shader.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(float(roughness))
    shader.CreateInput("metallic_constant", Sdf.ValueTypeNames.Float).Set(float(metallic))
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return mat

materials = {name: make_material(name, color, r, m) for name, color, r, m in TEST_COLORS}
print(f"[FixTest] Pre-created {len(materials)} distinct materials")

# Door
door = stage.GetPrimAtPath(DOOR_PATH)
if not door.IsValid():
    print(f"ERROR: door not found: {DOOR_PATH}")
    simulation_app.close()
    sys.exit(1)
print(f"[FixTest] Door: {DOOR_PATH}  IsInstance={door.IsInstance()}")
if door.IsInstance() or door.IsInstanceable():
    door.SetInstanceable(False)
    simulation_app.update()

meshes = [p for p in Usd.PrimRange(door) if p.IsA(UsdGeom.Mesh)]
print(f"[FixTest] Found {len(meshes)} meshes: {[m.GetPath().name for m in meshes]}")

def bind(meshes, mat):
    strong = UsdShade.Tokens.strongerThanDescendants
    for m in meshes:
        UsdShade.MaterialBindingAPI.Apply(m)
        UsdShade.MaterialBindingAPI(m).Bind(mat, bindingStrength=strong)

def save(path, warmup=8):
    for _ in range(warmup):
        world.step(render=True)
    rgba = cam.get_rgba()
    if rgba is None or rgba.size == 0:
        print(f"[FixTest] ERROR empty rgba: {path}")
        return
    rgb = rgba[:, :, :3].astype(np.uint8)
    from PIL import Image
    Image.fromarray(rgb).save(path, "JPEG", quality=95)
    mean_rgb = tuple(int(x) for x in rgb.reshape(-1, 3).mean(axis=0))
    print(f"[FixTest] Saved {os.path.basename(path)}  mean_rgb={mean_rgb}")

# Baseline (no override)
save(os.path.join(OUT_DIR, "00_baseline.jpg"))

# Render each color — bind a DIFFERENT pre-created material each time
for idx, (name, _, _, _) in enumerate(TEST_COLORS, start=1):
    bind(meshes, materials[name])
    simulation_app.update()
    bm = UsdShade.MaterialBindingAPI(meshes[0]).ComputeBoundMaterial()[0]
    bm_path = str(bm.GetPath()) if bm and bm.GetPrim().IsValid() else "<none>"
    print(f"[FixTest] [{idx}] bind -> {name}  composed={bm_path}")
    save(os.path.join(OUT_DIR, f"{idx:02d}_{name}.jpg"))

print(f"\n[FixTest] Done. Output: {OUT_DIR}/")
world.stop()
simulation_app.close()

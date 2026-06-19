#!/usr/bin/env python3
"""
Material Shader Pattern Test — determine which shader creation pattern
actually produces a working OmniPBR (or USD-native) material override.

Prior findings:
  - Binding works (ComputeBoundMaterial returns our override path)
  - Every OmniPBR override renders SOLID RED regardless of diffuse color
  - Pre-creating one material per color instead of mutating inputs didn't help
  - Conclusion: `shader.CreateIdAttr("OmniPBR")` alone is NOT resolving to
    the real OmniPBR MDL module. The render delegate substitutes a red
    fallback for every OmniPBR override we authored.

Patterns tested (same Paint mesh on Doors_4070013, 1 render each):
  P1  CreateIdAttr("OmniPBR")                         — the broken pattern
  P2  SetSourceAsset("OmniPBR.mdl","mdl")+SubId+Id    — canonical MDL binding
  P3  UsdPreviewSurface (native USD, renderer-agnostic)
  P4  P2 but using OmniGlass (just to confirm MDL pathway works for other mods)

Each pattern is authored 4 different colors so we can tell at a glance which
pattern actually propagates color changes to the render.

Target: Doors_4070013 Paint mesh (Type_F_Door, single mesh, fills viewport).
Output: data/material_shader_pattern/<pattern>_<color>.jpg

Run:
  sbatch --partition=teaching submit_isaac_job.sh \\
      simulations/robot/material_shader_pattern_test.py
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

OUT_DIR = os.path.join(_project_dir, "data", "material_shader_pattern")

COLORS = [
    ("red",   (0.95, 0.05, 0.05)),
    ("green", (0.05, 0.85, 0.10)),
    ("blue",  (0.05, 0.10, 0.90)),
    ("white", (0.95, 0.95, 0.95)),
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
    prim_path="/World/PatternCam",
    position=DOOR_CENTER + np.array([0, CAM_DIST, 0]),
    frequency=30,
    resolution=CAM_RESOLUTION,
    orientation=rot_utils.euler_angles_to_quats(np.array([0.0, 0.0, 270.0]), degrees=True),
)
simulation_app.update()
cam.initialize()
usd_cam = UsdGeom.Camera(stage.GetPrimAtPath("/World/PatternCam"))
usd_cam.GetFocalLengthAttr().Set(CAM_FOCAL_MM)
usd_cam.GetClippingRangeAttr().Set((0.01, 10000.0))

print("[Pat] Warming up ...")
for _ in range(20):
    world.step(render=True)

# Door
door = stage.GetPrimAtPath(DOOR_PATH)
if door.IsInstance() or door.IsInstanceable():
    door.SetInstanceable(False)
    simulation_app.update()
meshes = [p for p in Usd.PrimRange(door) if p.IsA(UsdGeom.Mesh)]
print(f"[Pat] {len(meshes)} mesh(es): {[m.GetPath().name for m in meshes]}")


# ------------------------------------------------------------------
# Pattern constructors
# ------------------------------------------------------------------

def make_p1(name, color):
    """P1: CreateIdAttr('OmniPBR') — the pattern that produced stuck red."""
    mp = f"/World/Looks/P1_{name}"
    mat = UsdShade.Material.Define(stage, Sdf.Path(mp))
    sh = UsdShade.Shader.Define(stage, Sdf.Path(mp + "/Shader"))
    sh.CreateIdAttr("OmniPBR")
    sh.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat


def make_p2(name, color):
    """P2: canonical MDL binding via SetSourceAsset + SubIdentifier."""
    mp = f"/World/Looks/P2_{name}"
    mat = UsdShade.Material.Define(stage, Sdf.Path(mp))
    sh = UsdShade.Shader.Define(stage, Sdf.Path(mp + "/Shader"))
    sh.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
    sh.SetSourceAsset(Sdf.AssetPath("OmniPBR.mdl"), "mdl")
    sh.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
    sh.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    sh.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(0.4)
    sh.CreateInput("metallic_constant", Sdf.ValueTypeNames.Float).Set(0.0)
    # Wire the MDL surface output explicitly
    mat.CreateSurfaceOutput("mdl").ConnectToSource(sh.ConnectableAPI(), "out")
    mat.CreateDisplacementOutput("mdl").ConnectToSource(sh.ConnectableAPI(), "out")
    mat.CreateVolumeOutput("mdl").ConnectToSource(sh.ConnectableAPI(), "out")
    return mat


def make_p3(name, color):
    """P3: UsdPreviewSurface — renderer-agnostic native USD shader."""
    mp = f"/World/Looks/P3_{name}"
    mat = UsdShade.Material.Define(stage, Sdf.Path(mp))
    sh = UsdShade.Shader.Define(stage, Sdf.Path(mp + "/Shader"))
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.4)
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat


def bind_strong(mesh_list, mat):
    strong = UsdShade.Tokens.strongerThanDescendants
    for m in mesh_list:
        UsdShade.MaterialBindingAPI.Apply(m)
        UsdShade.MaterialBindingAPI(m).Bind(mat, bindingStrength=strong)


def save(path, warmup=8):
    for _ in range(warmup):
        world.step(render=True)
    rgba = cam.get_rgba()
    if rgba is None or rgba.size == 0:
        print(f"[Pat] ERROR empty rgba: {path}")
        return
    rgb = rgba[:, :, :3].astype(np.uint8)
    from PIL import Image
    Image.fromarray(rgb).save(path, "JPEG", quality=95)
    mean_rgb = tuple(int(x) for x in rgb.reshape(-1, 3).mean(axis=0))
    print(f"[Pat] Saved {os.path.basename(path)}  mean_rgb={mean_rgb}")


# Baseline
save(os.path.join(OUT_DIR, "00_baseline.jpg"))

for pattern_name, constructor in [("P1", make_p1), ("P2", make_p2), ("P3", make_p3)]:
    print(f"\n[Pat] === Pattern {pattern_name} ===")
    for cname, color in COLORS:
        mat = constructor(cname, color)
        bind_strong(meshes, mat)
        simulation_app.update()
        bm = UsdShade.MaterialBindingAPI(meshes[0]).ComputeBoundMaterial()[0]
        bm_path = str(bm.GetPath()) if bm and bm.GetPrim().IsValid() else "<none>"
        print(f"[Pat] {pattern_name}_{cname}: composed={bm_path}")
        save(os.path.join(OUT_DIR, f"{pattern_name}_{cname}.jpg"))

print(f"\n[Pat] Done. Output: {OUT_DIR}/")
world.stop()
simulation_app.close()

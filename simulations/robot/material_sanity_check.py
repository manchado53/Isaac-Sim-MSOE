#!/usr/bin/env python3
"""
Material Sanity Check — render the same door/camera/lighting with each of
the 7 DOOR_MATERIALS presets, one image per material, for side-by-side
visual inspection.

Purpose: on the curtain-wall glass doors we're training on, apply_door_material()
skips glass meshes (_is_glass_mesh), so the override only affects the thin
aluminum frame. This script lets you see exactly how much (or how little)
visual difference the 6 non-"original" materials actually produce on
Doors_13606713, our reference door.

Fixed config:
  - Door: Doors_13606713 (east side, proven-good camera position)
  - Camera: single, 2.0 m east of door center, eye level, 0° arc (dead-on)
  - Lighting: bright_fluorescent preset (max contrast for material detail)
  - Door angle: 0.0° (fully closed)
  - Render: 1024x768 native-res, RTXAA, RTX Real-Time

Output: data/material_sanity/<material_name>.jpg   (7 JPEGs)

Run via:
  sbatch --partition=teaching submit_isaac_job.sh \\
      simulations/robot/material_sanity_check.py
"""

import os
import sys
import math

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

import carb
_carb = carb.settings.get_settings()
_carb.set("/rtx/post/aa/op", 4)          # RTXAA — native res
_carb.set("/rtx/post/dlss/execMode", 0)  # disable DLSS

import numpy as np
import omni.usd
from pxr import Usd, UsdLux, UsdGeom, UsdShade, Gf, Sdf

import isaacsim.core.utils.numpy.rotations as rot_utils
from isaacsim.core.api import World
from isaacsim.sensors.camera import Camera

_script_dir  = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.abspath(os.path.join(_script_dir, "..", ".."))

# ---------------------------------------------------------------------------
# Fixed config — mirrors proven-good Doors_13606713 setup
# ---------------------------------------------------------------------------

DOOR_PATH = (
    "/World/Building/_22_129___MSOE_Engineering_Building___A___2_25_2026"
    "/Geometry/Doors/RS___Curtain_Wall___Type_FG___New/New/Doors_13606713"
)
DOOR_CENTER = np.array([-20.4773, 80.1494, 28.03])  # metres
CAM_DIST    = 2.0
CAM_POS     = np.array([DOOR_CENTER[0] + CAM_DIST, DOOR_CENTER[1], DOOR_CENTER[2]])
CAM_YAW_DEG = 180.0  # face back toward -X (door center)

CAM_RESOLUTION = (1024, 768)
OUT_DIR = os.path.join(_project_dir, "data", "material_sanity")

# Mirrors DOOR_MATERIALS in door_static_collection.py (keep in sync)
DOOR_MATERIALS = [
    {"name": "brushed_aluminum",    "color": (0.75, 0.75, 0.78), "roughness": 0.35, "metallic": 0.85},
    {"name": "painted_white",       "color": (0.90, 0.90, 0.88), "roughness": 0.70, "metallic": 0.05},
    {"name": "dark_anodized",       "color": (0.12, 0.12, 0.15), "roughness": 0.25, "metallic": 0.80},
    {"name": "wood_veneer",         "color": (0.55, 0.35, 0.20), "roughness": 0.75, "metallic": 0.02},
    {"name": "red_fire_door",       "color": (0.65, 0.10, 0.08), "roughness": 0.60, "metallic": 0.10},
    {"name": "beige_institutional", "color": (0.72, 0.68, 0.58), "roughness": 0.65, "metallic": 0.08},
    {"name": "original",            "color": None, "roughness": None, "metallic": None},
]

# ---------------------------------------------------------------------------

os.makedirs(OUT_DIR, exist_ok=True)

world = World(stage_units_in_meters=1.0)
stage = omni.usd.get_context().get_stage()

building_usd = os.path.join(_project_dir, "MSOEBuildingV4_physics.usd")
if not os.path.isfile(building_usd):
    print(f"ERROR: building USD not found: {building_usd}")
    simulation_app.close()
    sys.exit(1)
stage.GetRootLayer().subLayerPaths.insert(0, building_usd)

# --- bright_fluorescent lighting (DomeLight + SphereLight fill) ---
dome = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/DomeLight"))
dome.CreateIntensityAttr(15_000)
dome.CreateColorTemperatureAttr(5500)
dome.CreateEnableColorTemperatureAttr(True)

fill = UsdLux.SphereLight.Define(stage, Sdf.Path("/World/FillLight"))
fill.CreateIntensityAttr(80_000)
fill.CreateColorTemperatureAttr(4500)
fill.CreateEnableColorTemperatureAttr(True)
fill.CreateRadiusAttr(0.3)
fill_xf = UsdGeom.Xformable(fill.GetPrim())
fill_xf.AddTranslateOp().Set(Gf.Vec3d(
    float(DOOR_CENTER[0]), float(DOOR_CENTER[1]), float(DOOR_CENTER[2]) + 1.5
))

world.reset()
simulation_app.update()

# --- Camera ---
cam_ori = rot_utils.euler_angles_to_quats(np.array([0.0, 0.0, CAM_YAW_DEG]), degrees=True)
cam = Camera(
    prim_path="/World/MatSanityCam",
    position=CAM_POS,
    frequency=30,
    resolution=CAM_RESOLUTION,
    orientation=cam_ori,
)
simulation_app.update()
cam.initialize()
usd_cam = UsdGeom.Camera(stage.GetPrimAtPath("/World/MatSanityCam"))
usd_cam.GetFocalLengthAttr().Set(5.0)
usd_cam.GetClippingRangeAttr().Set((0.01, 10000.0))

print("[MatSanity] Warming up renderer (20 steps) ...")
for _ in range(20):
    world.step(render=True)

# --- Door prim and child meshes ---
door_prim = stage.GetPrimAtPath(DOOR_PATH)
if not door_prim.IsValid():
    print(f"ERROR: door prim not found: {DOOR_PATH}")
    simulation_app.close()
    sys.exit(1)

def _is_glass_mesh(mesh_prim) -> bool:
    name = mesh_prim.GetPath().name.lower()
    return name.startswith("glass_") or "_gl_" in name or "glazing" in name or "pane_" in name

# Cache original material bindings for non-glass children
_original_bindings = {}
for child in door_prim.GetChildren():
    if not child.IsA(UsdGeom.Mesh):
        continue
    if _is_glass_mesh(child):
        continue
    binding_api = UsdShade.MaterialBindingAPI(child)
    rel = binding_api.GetDirectBindingRel()
    targets = rel.GetTargets() if rel else []
    _original_bindings[child.GetPath()] = list(targets)
    print(f"[MatSanity] Cached original binding for {child.GetPath().name}: {targets}")

# --- Pre-create one canonical OmniPBR material per preset (P2 pattern).
# `CreateIdAttr("OmniPBR")` does NOT resolve the MDL module in RTX — it
# falls back to a red debug material. Use SetSourceAsset + SubIdentifier
# and wire the mdl surface/displacement/volume outputs explicitly.
_preset_materials = {}
for _p in DOOR_MATERIALS:
    if _p["color"] is None:
        continue
    _mp = f"/World/Looks/SanityDoorMat_{_p['name']}"
    _mat = UsdShade.Material.Define(stage, Sdf.Path(_mp))
    _sh = UsdShade.Shader.Define(stage, Sdf.Path(_mp + "/Shader"))
    _sh.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
    _sh.SetSourceAsset(Sdf.AssetPath("OmniPBR.mdl"), "mdl")
    _sh.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
    _sh.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*_p["color"]))
    _sh.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(float(_p["roughness"]))
    _sh.CreateInput("metallic_constant", Sdf.ValueTypeNames.Float).Set(float(_p["metallic"]))
    _mat.CreateSurfaceOutput("mdl").ConnectToSource(_sh.ConnectableAPI(), "out")
    _mat.CreateDisplacementOutput("mdl").ConnectToSource(_sh.ConnectableAPI(), "out")
    _mat.CreateVolumeOutput("mdl").ConnectToSource(_sh.ConnectableAPI(), "out")
    _preset_materials[_p["name"]] = _mat


def apply_override(name, color, roughness, metallic):
    mat = _preset_materials[name]
    strong = UsdShade.Tokens.strongerThanDescendants
    for child in door_prim.GetChildren():
        if not child.IsA(UsdGeom.Mesh):
            continue
        if _is_glass_mesh(child):
            continue
        UsdShade.MaterialBindingAPI.Apply(child)
        UsdShade.MaterialBindingAPI(child).Bind(mat, bindingStrength=strong)


def restore_original():
    for child in door_prim.GetChildren():
        if not child.IsA(UsdGeom.Mesh):
            continue
        if _is_glass_mesh(child):
            continue
        path = child.GetPath()
        binding_api = UsdShade.MaterialBindingAPI(child)
        binding_api.UnbindAllBindings()
        for target in _original_bindings.get(path, []):
            rel = binding_api.GetDirectBindingRel() or binding_api.CreateDirectBindingRel()
            rel.AddTarget(target)


def save_jpeg(path):
    for _ in range(5):
        world.step(render=True)
    rgba = cam.get_rgba()
    if rgba is None or rgba.size == 0:
        print(f"[MatSanity] ERROR: camera returned empty for {path}")
        return
    rgb = rgba[:, :, :3].astype(np.uint8)
    mean_rgb = tuple(int(x) for x in rgb.reshape(-1, 3).mean(axis=0))
    try:
        from PIL import Image
        Image.fromarray(rgb).save(path, "JPEG", quality=95)
        print(f"[MatSanity] Saved {path}  mean_rgb={mean_rgb}")
    except ImportError:
        np.save(path.replace(".jpg", ".npy"), rgb)
        print(f"[MatSanity] PIL unavailable — saved raw .npy instead")


# --- Render each material ---
print(f"\n[MatSanity] Rendering 7 material variants of {DOOR_PATH.split('/')[-1]}")
print(f"[MatSanity] Output dir: {OUT_DIR}")
print(f"[MatSanity] Non-glass child meshes: {len(_original_bindings)}")
print("-" * 70)

for idx, mat in enumerate(DOOR_MATERIALS):
    name = mat["name"]
    if mat["color"] is None:
        print(f"[MatSanity] [{idx+1}/7] {name} — restoring original bindings")
        restore_original()
    else:
        print(f"[MatSanity] [{idx+1}/7] {name} — "
              f"color={mat['color']} rough={mat['roughness']} metal={mat['metallic']}")
        apply_override(mat["name"], mat["color"], mat["roughness"], mat["metallic"])

    simulation_app.update()
    save_jpeg(os.path.join(OUT_DIR, f"{idx+1:02d}_{name}.jpg"))

print("\n" + "=" * 70)
print(f"[MatSanity] Done. {len(DOOR_MATERIALS)} JPEGs in {OUT_DIR}/")
print("=" * 70)

world.stop()
simulation_app.close()

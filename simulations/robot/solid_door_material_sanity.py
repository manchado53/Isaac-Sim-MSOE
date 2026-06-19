#!/usr/bin/env python3
"""
Solid-Door Material Sanity — Path B verification.

material_sanity_check.py proved that on Doors_13606713 (curtain-wall glass)
the override only hits a single invisible aluminum trim piece. This script
repeats the 7-material comparison on Doors_8429374 — a solid RS___Type_FG_Door
on Floor 4 whose geometry lives inside a flattened prototype reference.

Path B strategy:
  1. Load the stage and resolve the door prim
  2. If the prim is instanced, call SetInstanceable(False) to make it
     a concrete, per-instance subtree we can bind materials on
  3. Walk ALL descendants (Usd.PrimRange) for Mesh prims — solid doors
     have geometry nested several levels deep inside the prototype
  4. Cache original bindings, then apply each of the 7 DOOR_MATERIALS
     presets in sequence and save a JPEG per variant

Fixed config:
  - Door: Doors_8429374 (Type_FG_Door, Floor 4, center [-18.78, 80.82, 23.75])
  - Approach axis: Y → camera placed +Y of door center, yawed to face -Y
  - Lighting: bright_fluorescent (same as curtain-wall sanity)
  - Door angle: 0 deg (no rotation — we are only verifying materials here)

Output: data/material_sanity_solid/<NN>_<material_name>.jpg

Run via:
  sbatch --partition=teaching submit_isaac_job.sh \\
      simulations/robot/solid_door_material_sanity.py
"""

import math
import os
import sys

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
# Fixed config — Doors_4070013 (Type_F_Door, Floor 4, full solid leaf — no glass)
# ---------------------------------------------------------------------------

DOOR_PATH = (
    "/World/Building/_22_129___MSOE_Engineering_Building___A___2_25_2026"
    "/Geometry/Doors/RS___Type_F_Door/NEW___HM_Frame___36_x96/Doors_4070013"
)
DOOR_CENTER = np.array([-17.5325, 55.4272, 23.7490])  # metres, from catalog
CAM_DIST    = 3.5
CAM_FOCAL_MM = 18.0
# Aim at door midline — 18mm@3.5m gives ~3.05m vertical coverage, fits the
# full 2.44m door height with ~0.3m headroom and ~0.3m floor visible
CAM_Z_OFFSET = 0.0
# Approach axis is Y — put camera +Y of door, face -Y (yaw 270 deg)
CAM_POS     = np.array([DOOR_CENTER[0], DOOR_CENTER[1] + CAM_DIST, DOOR_CENTER[2] + CAM_Z_OFFSET])
CAM_YAW_DEG = 270.0

CAM_RESOLUTION = (1024, 768)
OUT_DIR = os.path.join(_project_dir, "data", "material_sanity_solid_open")

# --- Door open rotation (about left edge, Z axis) ---
DOOR_OPEN_ANGLE_DEG = 60.0
# Hinge at X_min edge (bbox.min[0]) — double-door pair lives on +X side.
# In centimetres to match USD xformOp:transform stored values.
HINGE_X_CM = -1804.0   # bbox.min[0] * 100
HINGE_Y_CM = 5542.72   # door center Y * 100

# Keep in sync with door_static_collection.py DOOR_MATERIALS
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

# --- bright_fluorescent lighting preset ---
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
    prim_path="/World/MatSanitySolidCam",
    position=CAM_POS,
    frequency=30,
    resolution=CAM_RESOLUTION,
    orientation=cam_ori,
)
simulation_app.update()
cam.initialize()
usd_cam = UsdGeom.Camera(stage.GetPrimAtPath("/World/MatSanitySolidCam"))
usd_cam.GetFocalLengthAttr().Set(CAM_FOCAL_MM)
usd_cam.GetClippingRangeAttr().Set((0.01, 10000.0))

print("[SolidMatSanity] Warming up renderer (20 steps) ...")
for _ in range(20):
    world.step(render=True)

# --- Resolve door prim; un-instance if it is a prototype instance ---
door_prim = stage.GetPrimAtPath(DOOR_PATH)
if not door_prim.IsValid():
    print(f"ERROR: door prim not found: {DOOR_PATH}")
    simulation_app.close()
    sys.exit(1)

print(f"[SolidMatSanity] Door prim: {DOOR_PATH}")
print(f"[SolidMatSanity]   type={door_prim.GetTypeName()}  "
      f"IsInstance={door_prim.IsInstance()}  "
      f"IsInstanceable={door_prim.IsInstanceable()}")

if door_prim.IsInstance() or door_prim.IsInstanceable():
    print("[SolidMatSanity] Marking door instanceable=False to expose concrete subtree ...")
    door_prim.SetInstanceable(False)
    simulation_app.update()
    print(f"[SolidMatSanity]   after: IsInstance={door_prim.IsInstance()}  "
          f"IsInstanceable={door_prim.IsInstanceable()}")

# --- Rotate door open about its hinge (Z axis), by rewriting the Xform's
# xformOp:transform matrix. Compose as: new = orig * T(-hinge) * Rz * T(+hinge)
# (row-vector convention — point * new = (point * orig - hinge) * Rz + hinge).
door_xf = UsdGeom.Xformable(door_prim)
_door_op = None
for _xop in door_xf.GetOrderedXformOps():
    if _xop.GetOpName() == "xformOp:transform":
        _door_op = _xop
        break

if _door_op is None:
    print("[SolidMatSanity] WARN: door prim has no xformOp:transform — rotation skipped")
else:
    orig = _door_op.Get()
    th = math.radians(DOOR_OPEN_ANGLE_DEG)
    c, s = math.cos(th), math.sin(th)
    T_neg = Gf.Matrix4d(1,0,0,0, 0,1,0,0, 0,0,1,0, -HINGE_X_CM,-HINGE_Y_CM,0,1)
    Rz    = Gf.Matrix4d(  c, s,0,0,  -s, c,0,0, 0,0,1,0,  0,0,0,1)
    T_pos = Gf.Matrix4d(1,0,0,0, 0,1,0,0, 0,0,1,0,  HINGE_X_CM, HINGE_Y_CM,0,1)
    new_mat = orig * T_neg * Rz * T_pos
    _door_op.Set(new_mat)
    simulation_app.update()
    print(f"[SolidMatSanity] Rotated door by {DOOR_OPEN_ANGLE_DEG:.1f}° about "
          f"hinge=({HINGE_X_CM/100:.2f}, {HINGE_Y_CM/100:.2f}) m")

# --- Walk ALL descendants for Mesh prims (solid doors nest geometry deep) ---
def _is_glass_mesh(mesh_prim) -> bool:
    name = mesh_prim.GetPath().name.lower()
    return name.startswith("glass_") or "_gl_" in name or "glazing" in name or "pane_" in name

all_meshes = []
for p in Usd.PrimRange(door_prim):
    if p.IsA(UsdGeom.Mesh):
        all_meshes.append(p)

print(f"[SolidMatSanity] Total Mesh descendants: {len(all_meshes)}")
for m in all_meshes:
    glass = " [GLASS]" if _is_glass_mesh(m) else ""
    print(f"[SolidMatSanity]   {m.GetPath()}{glass}")

override_targets = [m for m in all_meshes if not _is_glass_mesh(m)]
print(f"[SolidMatSanity] Non-glass override targets: {len(override_targets)}")

if not override_targets:
    print("[SolidMatSanity] ERROR: no non-glass meshes found — cannot demonstrate material axis.")
    simulation_app.close()
    sys.exit(1)

# Cache original bindings
_original_bindings = {}
for mesh in override_targets:
    binding_api = UsdShade.MaterialBindingAPI(mesh)
    rel = binding_api.GetDirectBindingRel()
    targets = rel.GetTargets() if rel else []
    _original_bindings[mesh.GetPath()] = list(targets)
    print(f"[SolidMatSanity] Cached {mesh.GetPath().name} -> {targets}")

# --- Pre-create one distinct OmniPBR material per preset (canonical MDL binding).
# Mutating a shared shader's inputs does NOT invalidate the OmniPBR material cache
# in the RTX delegate (see job 242579/242581). And `CreateIdAttr("OmniPBR")` alone
# does not resolve the MDL module (see job 242625) — SetSourceAsset is required.
_preset_materials = {}
for _p in DOOR_MATERIALS:
    if _p["color"] is None:
        continue
    _mat_path = f"/World/Looks/SolidSanity_{_p['name']}"
    _mat = UsdShade.Material.Define(stage, Sdf.Path(_mat_path))
    _shader = UsdShade.Shader.Define(stage, Sdf.Path(_mat_path + "/Shader"))
    _shader.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
    _shader.SetSourceAsset(Sdf.AssetPath("OmniPBR.mdl"), "mdl")
    _shader.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
    _shader.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*_p["color"]))
    _shader.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(float(_p["roughness"]))
    _shader.CreateInput("metallic_constant", Sdf.ValueTypeNames.Float).Set(float(_p["metallic"]))
    _mat.CreateSurfaceOutput("mdl").ConnectToSource(_shader.ConnectableAPI(), "out")
    _mat.CreateDisplacementOutput("mdl").ConnectToSource(_shader.ConnectableAPI(), "out")
    _mat.CreateVolumeOutput("mdl").ConnectToSource(_shader.ConnectableAPI(), "out")
    _preset_materials[_p["name"]] = _mat


def apply_override(name, color, roughness, metallic):
    mat = _preset_materials[name]
    strong = UsdShade.Tokens.strongerThanDescendants
    for mesh in override_targets:
        UsdShade.MaterialBindingAPI.Apply(mesh)
        UsdShade.MaterialBindingAPI(mesh).Bind(mat, bindingStrength=strong)


def restore_original():
    for mesh in override_targets:
        path = mesh.GetPath()
        binding_api = UsdShade.MaterialBindingAPI(mesh)
        binding_api.UnbindAllBindings()
        for target in _original_bindings.get(path, []):
            rel = binding_api.GetDirectBindingRel() or binding_api.CreateDirectBindingRel()
            rel.AddTarget(target)


def save_jpeg(path):
    for _ in range(8):
        world.step(render=True)
    rgba = cam.get_rgba()
    if rgba is None or rgba.size == 0:
        print(f"[SolidMatSanity] ERROR: camera returned empty for {path}")
        return
    rgb = rgba[:, :, :3].astype(np.uint8)
    mean_rgb = tuple(int(x) for x in rgb.reshape(-1, 3).mean(axis=0))
    try:
        from PIL import Image
        Image.fromarray(rgb).save(path, "JPEG", quality=95)
        print(f"[SolidMatSanity] Saved {path}  mean_rgb={mean_rgb}")
    except ImportError:
        np.save(path.replace(".jpg", ".npy"), rgb)
        print(f"[SolidMatSanity] PIL unavailable — saved raw .npy instead")


# --- Render each material ---
print(f"\n[SolidMatSanity] Rendering 7 material variants of {DOOR_PATH.split('/')[-1]}")
print(f"[SolidMatSanity] Output dir: {OUT_DIR}")
print("-" * 70)

for idx, mat in enumerate(DOOR_MATERIALS):
    name = mat["name"]
    if mat["color"] is None:
        print(f"[SolidMatSanity] [{idx+1}/7] {name} — restoring original bindings")
        restore_original()
    else:
        print(f"[SolidMatSanity] [{idx+1}/7] {name} — "
              f"color={mat['color']} rough={mat['roughness']} metal={mat['metallic']}")
        apply_override(mat["name"], mat["color"], mat["roughness"], mat["metallic"])

    simulation_app.update()
    save_jpeg(os.path.join(OUT_DIR, f"{idx+1:02d}_{name}.jpg"))

print("\n" + "=" * 70)
print(f"[SolidMatSanity] Done. {len(DOOR_MATERIALS)} JPEGs in {OUT_DIR}/")
print("=" * 70)

world.stop()
simulation_app.close()

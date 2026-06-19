#!/usr/bin/env python3
"""
Material Comparison — Procedural OmniPBR vs NVIDIA vMaterials

Renders a small studio scene with four spheres arranged left-to-right
and captures it twice: once with each sphere bound to an OmniPBR
procedural material (with hand-tuned diffuse / roughness / metallic
constants), and once with each sphere bound to the equivalent material
from NVIDIA's vMaterials_2 library.

The two passes share the same scene, lights, camera, and renderer
(RTX Path Tracing), so any visible difference is purely the material.

Sphere order (left → right, same in both passes):
    1. metal       (brushed aluminum-ish)
    2. plastic     (thick red plastic)
    3. concrete    (rough gray concrete)
    4. wood        (oak / walnut)

Output:
    data/material_comparison/materials_omnipbr.jpg
    data/material_comparison/materials_vmaterials.jpg

Run via:
    sbatch submit_isaac_job.sh simulations/robot/material_comparison.py

Notes:
- vMaterials live on the Omniverse S3 content CDN and are referenced by
  URL. If Isaac Sim on this node can't resolve that URL (no internet,
  no Nucleus), the vMaterials pass will fall back to solid-color MDL and
  print a warning — the OmniPBR pass will still succeed.
- Both passes use path tracing; materials look very different under
  path tracing vs real-time because indirect lighting is what makes
  roughness/metalness perceptually meaningful.
"""

import os
import sys
import numpy as np

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
from isaacsim import SimulationApp
simulation_app = SimulationApp({
    "headless": True,
    "renderer": "PathTracing",
    "samples_per_pixel_per_frame": 32,   # T4 cannot sustain 64; V100+ can
})

import omni.usd
import omni.kit.commands
import omni.replicator.core as rep
from pxr import Usd, UsdLux, UsdGeom, UsdShade, Gf, Sdf

import isaacsim.core.utils.numpy.rotations as rot_utils
from isaacsim.core.api import World

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_script_dir  = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.abspath(os.path.join(_script_dir, "..", ".."))
OUT_DIR = os.path.join(_project_dir, "data", "material_comparison")
os.makedirs(OUT_DIR, exist_ok=True)

CAM_RESOLUTION = (1280, 720)
PT_SETTLE_STEPS = 48   # halved — T4 was crashing on the 96-step loop

# Order (left → right). Used by both OmniPBR and vMaterials passes.
SPHERE_NAMES = ["metal", "plastic", "concrete", "wood"]

# Hand-tuned OmniPBR constants for each sphere
#   (diffuse_rgb, roughness, metallic)
OMNIPBR_CONFIGS = {
    "metal":    ((0.82, 0.83, 0.85), 0.15, 1.0),   # brushed aluminum
    "plastic":  ((0.85, 0.10, 0.12), 0.30, 0.0),   # red plastic
    "concrete": ((0.62, 0.62, 0.58), 0.90, 0.0),   # rough concrete
    "wood":     ((0.48, 0.25, 0.12), 0.55, 0.0),   # walnut / oak
}

# vMaterials_2 library references (NVIDIA content CDN, resolved by Kit)
_VMAT_ROOT = "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Materials/vMaterials_2"
VMATERIAL_CONFIGS = {
    "metal":    (f"{_VMAT_ROOT}/Metal/Aluminum_Brushed.mdl",      "Aluminum_Brushed"),
    "plastic":  (f"{_VMAT_ROOT}/Plastic/Plastic_Thick.mdl",       "Plastic_Thick"),
    "concrete": (f"{_VMAT_ROOT}/Stone/Concrete_Rough.mdl",        "Concrete_Rough"),
    "wood":     (f"{_VMAT_ROOT}/Wood/Oak_European.mdl",           "Oak_European"),
}

# ---------------------------------------------------------------------------
# Scene setup
# ---------------------------------------------------------------------------

world = World(stage_units_in_meters=1.0)
stage = omni.usd.get_context().get_stage()

# Force a physics scene to exist. In PathTracing mode, world.step(render=True)
# is a no-op unless there's a PhysicsScene registered — without this, the
# Camera sensor's annotator is never populated and cam.get_rgba() returns an
# empty (0,) array. The default ground plane is hidden under our own floor
# cube below, so it's invisible in the final image.
world.scene.add_default_ground_plane()

# Floor: large flat scaled cube
floor = UsdGeom.Cube.Define(stage, "/World/Floor")
floor_xf = UsdGeom.Xformable(floor.GetPrim())
floor_xf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -0.05))
floor_xf.AddScaleOp().Set(Gf.Vec3f(8.0, 8.0, 0.05))

# Backdrop wall behind spheres (cyclorama)
back = UsdGeom.Cube.Define(stage, "/World/Backdrop")
back_xf = UsdGeom.Xformable(back.GetPrim())
back_xf.AddTranslateOp().Set(Gf.Vec3d(0.0, 2.5, 2.0))
back_xf.AddScaleOp().Set(Gf.Vec3f(6.0, 0.05, 2.5))

# Dome light (soft ambient)
dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
dome.CreateIntensityAttr(1500)

# Key light (rect light from upper-front-left)
key = UsdLux.RectLight.Define(stage, "/World/KeyLight")
key.CreateIntensityAttr(35000)
key.CreateWidthAttr(3.0)
key.CreateHeightAttr(1.5)
key_xf = UsdGeom.Xformable(key.GetPrim())
key_xf.AddTranslateOp().Set(Gf.Vec3d(-2.5, -2.5, 3.5))
key_xf.AddRotateXOp().Set(-55.0)
key_xf.AddRotateZOp().Set(30.0)

# Fill light (rect light from upper-front-right, cooler & weaker)
fill = UsdLux.RectLight.Define(stage, "/World/FillLight")
fill.CreateIntensityAttr(15000)
fill.CreateWidthAttr(2.0)
fill.CreateHeightAttr(1.0)
fill_xf = UsdGeom.Xformable(fill.GetPrim())
fill_xf.AddTranslateOp().Set(Gf.Vec3d(2.5, -2.0, 3.0))
fill_xf.AddRotateXOp().Set(-50.0)
fill_xf.AddRotateZOp().Set(-25.0)

# Spheres, left → right
SPHERE_RADIUS  = 0.5
SPHERE_SPACING = 1.25
sphere_paths = {}
for i, name in enumerate(SPHERE_NAMES):
    sp = f"/World/Sphere_{name}"
    sphere = UsdGeom.Sphere.Define(stage, sp)
    sphere.CreateRadiusAttr(SPHERE_RADIUS)
    xf = UsdGeom.Xformable(sphere.GetPrim())
    x_pos = -((len(SPHERE_NAMES) - 1) / 2.0) * SPHERE_SPACING + i * SPHERE_SPACING
    xf.AddTranslateOp().Set(Gf.Vec3d(x_pos, 0.0, SPHERE_RADIUS + 0.01))
    sphere_paths[name] = sp

world.reset()
simulation_app.update()

# ---------------------------------------------------------------------------
# Camera + Replicator render product
#
# The isaacsim.sensors.camera.Camera wrapper was returning an empty (0,)
# array from get_rgba() in this minimal studio scene under PathTracing mode.
# Instead, build a plain UsdGeom.Camera manually and attach a Replicator
# render_product + "rgb" annotator directly — this is the canonical path
# for synthetic-data capture in Isaac Sim 5.1 and avoids the sensor wrapper
# entirely.
# ---------------------------------------------------------------------------

CAM_PATH = "/World/MaterialCam"
cam_prim = UsdGeom.Camera.Define(stage, CAM_PATH)
cam_xf = UsdGeom.Xformable(cam_prim.GetPrim())
cam_xf.AddTranslateOp().Set(Gf.Vec3d(0.0, -4.0, 1.1))
# Rotate the camera to look at +Y (the spheres sit at y=0). UsdGeom.Camera
# looks down -Z by default, so rotate -90° around X to aim it along +Y.
cam_xf.AddRotateXOp().Set(90.0)
cam_prim.GetFocalLengthAttr().Set(28.0)
cam_prim.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 1000.0))

# Create a Replicator render product + RGB annotator bound to the camera.
render_product = rep.create.render_product(CAM_PATH, CAM_RESOLUTION)
rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
rgb_annotator.attach([render_product])

print("[MatCompare] Warming up renderer (20 steps) ...")
for _ in range(20):
    world.step(render=True)

# ---------------------------------------------------------------------------
# Material helpers
# ---------------------------------------------------------------------------

def create_mdl_material(mtl_path: str, mdl_url: str, mtl_name: str) -> bool:
    """Instantiate an MDL material prim via omni.kit.commands.

    Returns True on success, False if the MDL asset couldn't be loaded.
    """
    try:
        omni.kit.commands.execute(
            "CreateMdlMaterialPrimCommand",
            mtl_url=mdl_url,
            mtl_name=mtl_name,
            mtl_path=mtl_path,
        )
        return stage.GetPrimAtPath(mtl_path).IsValid()
    except Exception as e:
        print(f"  [MatCompare] Failed to create MDL material at {mtl_path}: {e}")
        return False


def set_shader_input(mtl_path: str, input_name: str, value, type_name):
    """Set an MDL shader input via the UsdShade API (the idiomatic route)."""
    shader_prim = stage.GetPrimAtPath(f"{mtl_path}/Shader")
    if not shader_prim.IsValid():
        print(f"  [MatCompare] Shader prim not found under {mtl_path}", flush=True)
        return
    shader = UsdShade.Shader(shader_prim)
    shader.CreateInput(input_name, type_name).Set(value)


def bind_material(mesh_path: str, mtl_path: str):
    mesh_prim = stage.GetPrimAtPath(mesh_path)
    mat = UsdShade.Material(stage.GetPrimAtPath(mtl_path))
    if not mat:
        print(f"  [MatCompare] Material prim not valid at {mtl_path}")
        return
    UsdShade.MaterialBindingAPI(mesh_prim).Bind(
        mat, UsdShade.Tokens.strongerThanDescendants
    )


def render_and_save(filename: str, cam_tag: str, settle: int = PT_SETTLE_STEPS):
    print(f"[MatCompare] Rendering {settle} settle steps → {filename}", flush=True)

    # Let MDL shaders compile and materials resolve before stepping the renderer.
    for _ in range(8):
        simulation_app.update()
    for i in range(settle):
        world.step(render=True)
        if (i + 1) % 8 == 0:
            print(f"  [MatCompare]   step {i+1}/{settle}", flush=True)
    print("  [MatCompare]   fetching RGB from annotator ...", flush=True)
    data = rgb_annotator.get_data()
    if data is None:
        print(f"[MatCompare] ERROR: annotator returned None for {filename}", flush=True)
        return
    arr = np.asarray(data)
    print(f"  [MatCompare]   rgb shape={arr.shape} dtype={arr.dtype}", flush=True)
    if arr.ndim != 3 or arr.shape[-1] < 3:
        print(f"[MatCompare] ERROR: unexpected rgb shape {arr.shape} for {filename}", flush=True)
        return
    rgb = arr[:, :, :3].astype(np.uint8)
    try:
        from PIL import Image
        Image.fromarray(rgb).save(filename, "JPEG", quality=95)
        print(f"[MatCompare] Saved {filename}", flush=True)
    except ImportError:
        np.save(filename.replace(".jpg", ".npy"), rgb)
        print(f"[MatCompare] PIL unavailable — saved raw .npy instead", flush=True)


# ---------------------------------------------------------------------------
# Pass 1: OmniPBR procedural
# ---------------------------------------------------------------------------

print(f"\n{'=' * 70}")
print(f"[MatCompare] Pass 1/2 — OmniPBR (procedural diffuse/roughness/metallic)")
print(f"{'=' * 70}")

OMNIPBR_INPUTS = [
    ("diffuse_color_constant",        0, Sdf.ValueTypeNames.Color3f),
    ("reflection_roughness_constant", 1, Sdf.ValueTypeNames.Float),
    ("metallic_constant",             2, Sdf.ValueTypeNames.Float),
]

for name in SPHERE_NAMES:
    diffuse, roughness, metallic = OMNIPBR_CONFIGS[name]
    mtl_path = f"/World/Materials/OmniPBR_{name}"
    print(f"  {name:9s}  diffuse={diffuse}  rough={roughness}  metal={metallic}")

    ok = create_mdl_material(mtl_path, "OmniPBR.mdl", "OmniPBR")
    if not ok:
        print(f"  [MatCompare] OmniPBR creation failed for {name}, skipping")
        continue

    values = [Gf.Vec3f(*diffuse), float(roughness), float(metallic)]
    for input_name, idx, type_name in OMNIPBR_INPUTS:
        set_shader_input(mtl_path, input_name, values[idx], type_name)

    bind_material(sphere_paths[name], mtl_path)

simulation_app.update()
render_and_save(os.path.join(OUT_DIR, "materials_omnipbr.jpg"), cam_tag="omnipbr")

# ---------------------------------------------------------------------------
# Pass 2: vMaterials_2
# ---------------------------------------------------------------------------

print(f"\n{'=' * 70}")
print(f"[MatCompare] Pass 2/2 — NVIDIA vMaterials_2 (pre-authored MDL)")
print(f"{'=' * 70}")

for name in SPHERE_NAMES:
    mdl_url, mtl_name = VMATERIAL_CONFIGS[name]
    mtl_path = f"/World/Materials/vMat_{name}"
    print(f"  {name:9s}  {mdl_url}")

    ok = create_mdl_material(mtl_path, mdl_url, mtl_name)
    if not ok:
        print(f"  [MatCompare] vMaterial unavailable for {name} — "
              f"sphere will keep previous OmniPBR binding")
        continue

    # Override the sphere's binding with the vMaterial
    bind_material(sphere_paths[name], mtl_path)

simulation_app.update()
render_and_save(os.path.join(OUT_DIR, "materials_vmaterials.jpg"), cam_tag="vmat")

# ---------------------------------------------------------------------------

print(f"\n{'=' * 70}")
print(f"[MatCompare] Done.")
print(f"Output: {OUT_DIR}/")
print(f"  materials_omnipbr.jpg    (procedural)")
print(f"  materials_vmaterials.jpg (vMaterials_2)")
print(f"Sphere order (L→R): {', '.join(SPHERE_NAMES)}")
print(f"{'=' * 70}")

world.stop()
simulation_app.close()

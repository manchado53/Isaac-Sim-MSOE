#!/usr/bin/env python3
"""
Material Binding Diagnostic — narrow down why `Doors_8429374` renders look
identical across 7 override materials.

Hypotheses we test:
  H1  Mesh visibility / framing — HM and WD bboxes might not be in the camera
      frustum even though the "door-shaped" red rectangle is.
  H2  Binding strength — our default `.Bind(_mat)` may not override an existing
      stronger binding. Try `UsdShade.Tokens.strongerThanDescendants`.
  H3  Binding target survives — `ComputeBoundMaterial()` reports what the
      render delegate will actually use. If it still points at the original
      `HM/Looks/HM`, our binding never took effect.
  H4  Glass dominates — Glass_GL_1 may be opaque in the MDL fallback, making
      HM/WD invisible even when bound correctly. Test by also binding the
      override to the glass mesh.
  H5  Camera / FOV — the current sanity script uses focal=5mm, so the door is
      tiny in frame. Move camera much closer + use a normal FL to make any
      frame trim very large and obvious.

Strategy:
  - Load the stage, place camera 0.8 m from `Doors_8429374` with 24 mm FL
  - Print world-space bboxes of HM, WD, Glass_GL_1 and check camera frustum
  - Render BASELINE (no override)
  - Render STRATEGY_A: default .Bind() on HM+WD only
  - Render STRATEGY_B: .Bind(bindingStrength=strongerThanDescendants) HM+WD
  - Render STRATEGY_C: strong bind on HM+WD+Glass (override-all)
  - Render STRATEGY_D: delete original DirectBindingRel, re-author ours
  - After each, call ComputeBoundMaterial() to confirm what the stage
    considers the active material on each target mesh
  - Repeat the best-performing strategy on `Doors_4070013` (Type_F_Door,
    no glass at all) to eliminate the glass confound entirely

Output: data/material_binding_diag/<NN>_<label>.jpg + a JSON log of the
composed bindings so we can tell authoring apart from rendering.

Run via:
  sbatch --partition=teaching submit_isaac_job.sh \\
      simulations/robot/material_binding_diagnostic.py
"""

import os
import sys
import json
import math

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

import carb
_carb = carb.settings.get_settings()
_carb.set("/rtx/post/aa/op", 4)           # RTXAA native res
_carb.set("/rtx/post/dlss/execMode", 0)

import numpy as np
import omni.usd
from pxr import Usd, UsdLux, UsdGeom, UsdShade, Gf, Sdf

import isaacsim.core.utils.numpy.rotations as rot_utils
from isaacsim.core.api import World
from isaacsim.sensors.camera import Camera

_script_dir  = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.abspath(os.path.join(_script_dir, "..", ".."))

# ---------------------------------------------------------------------------
# Two test doors
# ---------------------------------------------------------------------------

DOOR_A_PATH = (
    "/World/Building/_22_129___MSOE_Engineering_Building___A___2_25_2026"
    "/Geometry/Doors/RS___Type_FG_Door/NEW___HM_Frame___36_x96/Doors_8429374"
)
DOOR_A_CENTER = np.array([-18.7834, 80.8209, 23.7490])

DOOR_B_PATH = (
    "/World/Building/_22_129___MSOE_Engineering_Building___A___2_25_2026"
    "/Geometry/Doors/RS___Type_F_Door/NEW___HM_Frame___36_x96/Doors_4070013"
)
DOOR_B_CENTER = np.array([-17.5325, 55.4272, 23.7490])

CAM_RESOLUTION = (1024, 768)
CAM_DIST = 0.9           # very close — frame is dominated by the door
CAM_FOCAL_MM = 24.0      # normal lens on 36 mm sensor (~54 deg horiz FOV)

OUT_DIR = os.path.join(_project_dir, "data", "material_binding_diag")

# Test materials — small set, extreme colors, so "is it working" is obvious
TEST_COLORS = [
    ("brushed_aluminum", (0.75, 0.75, 0.78), 0.35, 0.85),
    ("dark_anodized",    (0.05, 0.05, 0.08), 0.25, 0.80),
    ("bright_green",     (0.10, 0.85, 0.15), 0.55, 0.00),
    ("hot_pink",         (0.95, 0.15, 0.60), 0.40, 0.00),
]

os.makedirs(OUT_DIR, exist_ok=True)

world = World(stage_units_in_meters=1.0)
stage = omni.usd.get_context().get_stage()

building_usd = os.path.join(_project_dir, "MSOEBuildingV4_physics.usd")
if not os.path.isfile(building_usd):
    print(f"ERROR: building USD not found: {building_usd}")
    simulation_app.close()
    sys.exit(1)
stage.GetRootLayer().subLayerPaths.insert(0, building_usd)

# --- Lighting: bright dome + fill near door A ---
dome = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/DomeLight"))
dome.CreateIntensityAttr(15_000)
dome.CreateColorTemperatureAttr(5500)
dome.CreateEnableColorTemperatureAttr(True)

fill_a = UsdLux.SphereLight.Define(stage, Sdf.Path("/World/FillLightA"))
fill_a.CreateIntensityAttr(80_000)
fill_a.CreateColorTemperatureAttr(4500)
fill_a.CreateEnableColorTemperatureAttr(True)
fill_a.CreateRadiusAttr(0.3)
UsdGeom.Xformable(fill_a.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(
    float(DOOR_A_CENTER[0]), float(DOOR_A_CENTER[1]), float(DOOR_A_CENTER[2]) + 1.5
))
fill_b = UsdLux.SphereLight.Define(stage, Sdf.Path("/World/FillLightB"))
fill_b.CreateIntensityAttr(80_000)
fill_b.CreateColorTemperatureAttr(4500)
fill_b.CreateEnableColorTemperatureAttr(True)
fill_b.CreateRadiusAttr(0.3)
UsdGeom.Xformable(fill_b.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(
    float(DOOR_B_CENTER[0]), float(DOOR_B_CENTER[1]), float(DOOR_B_CENTER[2]) + 1.5
))

world.reset()
simulation_app.update()


# --- Single camera, repositioned per door ---
cam = Camera(
    prim_path="/World/DiagCam",
    position=DOOR_A_CENTER + np.array([0, CAM_DIST, 0]),
    frequency=30,
    resolution=CAM_RESOLUTION,
    orientation=rot_utils.euler_angles_to_quats(np.array([0.0, 0.0, 270.0]), degrees=True),
)
simulation_app.update()
cam.initialize()
usd_cam = UsdGeom.Camera(stage.GetPrimAtPath("/World/DiagCam"))
usd_cam.GetFocalLengthAttr().Set(CAM_FOCAL_MM)
usd_cam.GetClippingRangeAttr().Set((0.01, 10000.0))

print("[Diag] Warming up renderer ...")
for _ in range(20):
    world.step(render=True)


def move_cam(center_xyz, approach_axis="Y"):
    """Place camera CAM_DIST meters along +approach_axis from center, facing -axis."""
    if approach_axis == "Y":
        pos = np.array([center_xyz[0], center_xyz[1] + CAM_DIST, center_xyz[2]])
        yaw = 270.0
    else:  # X
        pos = np.array([center_xyz[0] + CAM_DIST, center_xyz[1], center_xyz[2]])
        yaw = 180.0
    ori = rot_utils.euler_angles_to_quats(np.array([0.0, 0.0, yaw]), degrees=True)
    cam.set_world_pose(position=pos, orientation=ori)
    for _ in range(3):
        world.step(render=True)
    return pos, yaw


# --- Build override material once (we mutate shader inputs between renders) ---
_mat_path = "/World/Looks/DiagDoorMat"
_mat      = UsdShade.Material.Define(stage, Sdf.Path(_mat_path))
_shader   = UsdShade.Shader.Define(stage, Sdf.Path(_mat_path + "/Shader"))
_shader.CreateIdAttr("OmniPBR")
_shader.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.8, 0.8, 0.8))
_shader.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(0.5)
_shader.CreateInput("metallic_constant", Sdf.ValueTypeNames.Float).Set(0.0)
_mat.CreateSurfaceOutput().ConnectToSource(_shader.ConnectableAPI(), "surface")


def set_shader(color, roughness, metallic):
    _shader.GetInput("diffuse_color_constant").Set(Gf.Vec3f(*color))
    _shader.GetInput("reflection_roughness_constant").Set(float(roughness))
    _shader.GetInput("metallic_constant").Set(float(metallic))


def is_glass(mesh_prim):
    n = mesh_prim.GetPath().name.lower()
    return n.startswith("glass_") or "_gl_" in n or "glazing" in n or "pane_" in n


def collect_meshes(door_prim):
    meshes = []
    for p in Usd.PrimRange(door_prim):
        if p.IsA(UsdGeom.Mesh):
            meshes.append(p)
    return meshes


def world_bbox(prim):
    bbc = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_], useExtentsHint=True)
    try:
        b = bbc.ComputeWorldBound(prim)
        r = b.ComputeAlignedRange()
        if r.IsEmpty():
            return None
        mn, mx = r.GetMin(), r.GetMax()
        return ((mn[0], mn[1], mn[2]), (mx[0], mx[1], mx[2]))
    except Exception as e:
        return None


def bound_material_name(mesh):
    """ComputeBoundMaterial() tells us what the render delegate actually sees."""
    try:
        mat, rel = UsdShade.MaterialBindingAPI(mesh).ComputeBoundMaterial()
        if mat and mat.GetPrim().IsValid():
            return str(mat.GetPath())
    except Exception:
        pass
    return "<none>"


def save_jpeg(path, warmup=6):
    for _ in range(warmup):
        world.step(render=True)
    rgba = cam.get_rgba()
    if rgba is None or rgba.size == 0:
        print(f"[Diag] ERROR: empty rgba for {path}")
        return None
    rgb = rgba[:, :, :3].astype(np.uint8)
    from PIL import Image
    Image.fromarray(rgb).save(path, "JPEG", quality=95)
    mean_rgb = tuple(int(x) for x in rgb.reshape(-1, 3).mean(axis=0))
    print(f"[Diag] Saved {os.path.basename(path)}  mean_rgb={mean_rgb}")
    return mean_rgb


def unbind_all(meshes):
    """Delete any authored DirectBindingRel so the composed binding falls back."""
    for m in meshes:
        api = UsdShade.MaterialBindingAPI(m)
        rel = api.GetDirectBindingRel()
        if rel:
            rel.SetTargets([])
        # Also try the API's explicit unbind
        try:
            api.UnbindAllBindings()
        except Exception:
            pass


def bind_default(meshes, mat):
    for m in meshes:
        UsdShade.MaterialBindingAPI.Apply(m)
        UsdShade.MaterialBindingAPI(m).Bind(mat)


def bind_strong(meshes, mat):
    strong = UsdShade.Tokens.strongerThanDescendants
    for m in meshes:
        UsdShade.MaterialBindingAPI.Apply(m)
        UsdShade.MaterialBindingAPI(m).Bind(mat, bindingStrength=strong)


# ===========================================================================
# DOOR A — Doors_8429374 (Type_FG_Door, has glass)
# ===========================================================================

door_a = stage.GetPrimAtPath(DOOR_A_PATH)
if not door_a.IsValid():
    print(f"ERROR: door A not found: {DOOR_A_PATH}")
    simulation_app.close()
    sys.exit(1)

print(f"\n[Diag] DOOR A = {DOOR_A_PATH}")
print(f"[Diag]   IsInstance={door_a.IsInstance()}  IsInstanceable={door_a.IsInstanceable()}")
if door_a.IsInstance() or door_a.IsInstanceable():
    door_a.SetInstanceable(False)
    simulation_app.update()

meshes_a = collect_meshes(door_a)
print(f"[Diag]   mesh descendants: {len(meshes_a)}")
for m in meshes_a:
    bbox = world_bbox(m)
    bm = bound_material_name(m)
    tag = " [GLASS]" if is_glass(m) else ""
    print(f"[Diag]   {m.GetPath().name}{tag}  bbox={bbox}  bound_mat={bm}")

# Place camera
cam_pos, cam_yaw = move_cam(DOOR_A_CENTER, approach_axis="Y")
print(f"[Diag]   camera @ {cam_pos} yaw={cam_yaw} focal={CAM_FOCAL_MM}mm")

# Baseline render (no override)
set_shader((0.8, 0.8, 0.8), 0.5, 0.0)
save_jpeg(os.path.join(OUT_DIR, "A00_baseline.jpg"))

# Strategy A: default bind on non-glass (HM + WD)
non_glass_a = [m for m in meshes_a if not is_glass(m)]
glass_a     = [m for m in meshes_a if is_glass(m)]
print(f"[Diag]   non_glass targets: {[m.GetPath().name for m in non_glass_a]}")
print(f"[Diag]   glass   targets: {[m.GetPath().name for m in glass_a]}")

for name, color, rough, metal in TEST_COLORS:
    set_shader(color, rough, metal)
    bind_default(non_glass_a, _mat)
    simulation_app.update()
    # Log bound material for first mesh only (they should all be the same)
    bm = bound_material_name(non_glass_a[0]) if non_glass_a else "<noop>"
    print(f"[Diag]   [stratA:{name}] first bound_mat={bm}")
    save_jpeg(os.path.join(OUT_DIR, f"A_stratA_{name}.jpg"))

# Strategy B: strong bind on non-glass
for name, color, rough, metal in TEST_COLORS:
    set_shader(color, rough, metal)
    bind_strong(non_glass_a, _mat)
    simulation_app.update()
    bm = bound_material_name(non_glass_a[0]) if non_glass_a else "<noop>"
    print(f"[Diag]   [stratB:{name}] first bound_mat={bm}")
    save_jpeg(os.path.join(OUT_DIR, f"A_stratB_{name}.jpg"))

# Strategy C: strong bind on EVERYTHING including glass (override all)
all_a = non_glass_a + glass_a
for name, color, rough, metal in TEST_COLORS:
    set_shader(color, rough, metal)
    bind_strong(all_a, _mat)
    simulation_app.update()
    save_jpeg(os.path.join(OUT_DIR, f"A_stratC_{name}.jpg"))

# Strategy D: delete original direct binding first, then strong bind
unbind_all(all_a)
for name, color, rough, metal in TEST_COLORS:
    set_shader(color, rough, metal)
    bind_strong(all_a, _mat)
    simulation_app.update()
    save_jpeg(os.path.join(OUT_DIR, f"A_stratD_{name}.jpg"))

# ===========================================================================
# DOOR B — Doors_4070013 (Type_F_Door, NO glass, pure solid leaf)
# ===========================================================================

door_b = stage.GetPrimAtPath(DOOR_B_PATH)
if not door_b.IsValid():
    print(f"ERROR: door B not found: {DOOR_B_PATH}")
    simulation_app.close()
    sys.exit(1)

print(f"\n[Diag] DOOR B = {DOOR_B_PATH}")
print(f"[Diag]   IsInstance={door_b.IsInstance()}  IsInstanceable={door_b.IsInstanceable()}")
if door_b.IsInstance() or door_b.IsInstanceable():
    door_b.SetInstanceable(False)
    simulation_app.update()

meshes_b = collect_meshes(door_b)
print(f"[Diag]   mesh descendants: {len(meshes_b)}")
for m in meshes_b:
    bbox = world_bbox(m)
    bm = bound_material_name(m)
    tag = " [GLASS]" if is_glass(m) else ""
    print(f"[Diag]   {m.GetPath().name}{tag}  bbox={bbox}  bound_mat={bm}")

cam_pos, cam_yaw = move_cam(DOOR_B_CENTER, approach_axis="Y")
print(f"[Diag]   camera @ {cam_pos} yaw={cam_yaw}")

set_shader((0.8, 0.8, 0.8), 0.5, 0.0)
save_jpeg(os.path.join(OUT_DIR, "B00_baseline.jpg"))

non_glass_b = [m for m in meshes_b if not is_glass(m)]
all_b = meshes_b  # no glass expected
print(f"[Diag]   non_glass targets: {[m.GetPath().name for m in non_glass_b]}")

# Strategy B (strong) on HM+WD
for name, color, rough, metal in TEST_COLORS:
    set_shader(color, rough, metal)
    bind_strong(non_glass_b, _mat)
    simulation_app.update()
    bm = bound_material_name(non_glass_b[0]) if non_glass_b else "<noop>"
    print(f"[Diag]   [B_stratB:{name}] first bound_mat={bm}")
    save_jpeg(os.path.join(OUT_DIR, f"B_stratB_{name}.jpg"))

# Strategy D (unbind + strong)
unbind_all(all_b)
for name, color, rough, metal in TEST_COLORS:
    set_shader(color, rough, metal)
    bind_strong(all_b, _mat)
    simulation_app.update()
    save_jpeg(os.path.join(OUT_DIR, f"B_stratD_{name}.jpg"))

print("\n" + "=" * 70)
print(f"[Diag] Done. Output in {OUT_DIR}/")
print("=" * 70)

world.stop()
simulation_app.close()

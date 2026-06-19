#!/usr/bin/env python3
"""
Visual-review grid — lighting × material × wall audit across all 10 doors.

Purpose
-------
Before production collection we need a human eyeball on every
(lighting, door_material, wall_color) cell on every door, so we can catch
rendering bugs (missing bindings, shader cache fails, wall swaps hitting
the wrong mesh set, lighting presets that collapse to identical frames)
without burning 45k-frame SLURM jobs.

Grid per door
-------------
* 6 lightings × 6 door materials  = 36 cells   (all 10 doors)
* + 6 wall-color swatches          =  6 cells   (solid doors only)
  (bright_fluorescent lighting, first door material preset, wall swap only)

Totals
------
  5 curtain-wall × 36 = 180
  5 solid        × 42 = 210
  GRAND TOTAL          = 390 JPEGs   (under the 600 budget)

Fixed rendering conditions
--------------------------
* Head-on camera only (arc=0°, 2 m out along door normal, neutral height).
* Door angle:
    - ANGLE_MIX=False → all cells at CLOSED (1.5°); original rendering audit.
    - ANGLE_MIX=True  → each door's 36 cells get a balanced shuffled pick
                        from ANGLE_SWEEP (6 of each angle), producing a
                        mixed open/closed montage.
* No camera jitter, no pose randomization, no augmentation.
* Per-cell lighting jitter is kept ON because it is baked into
  apply_lighting_condition() and matches the production path.

Output
------
  data/visual_review_gridV2/            (ANGLE_MIX=False — audit grid)
  data/visual_review_grid_angles/       (ANGLE_MIX=True  — mixed-angle video)
      {family_short}_{door_id}_cell_L{0-5}_M{0-5}.jpg
      {family_short}_{door_id}_wall_{0-5}.jpg         (solid doors only, closed)
      manifest.json

Run
---
  sbatch submit_isaac_job.sh simulations/robot/visual_review_grid.py
"""

import os
import sys
import math
import json
import random
import functools
import numpy as np

print = functools.partial(print, flush=True)   # noqa: A001

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CAM_RESOLUTION = (1024, 768)
STARTUP_WARMUP = 30
DOOR_WARMUP    = 20   # after repositioning camera + lights, first lighting on
CELL_WARMUP    = 12   # after binding/lighting/angle swap inside the inner loop
CLOSED_ANGLE   = 1.5  # degrees — well under the 5° CLOSED threshold

# When True, each cell (L×M) gets a deterministic per-door-balanced angle
# drawn from ANGLE_SWEEP — the resulting video shows a mix of open and
# closed doors across lighting/material variations. When False, every cell
# uses CLOSED_ANGLE (original rendering-audit behaviour).
ANGLE_MIX   = True
ANGLE_SWEEP = [1.5, 18.0, 36.0, 54.0, 72.0, 90.0]  # closed → fully open

_script_dir  = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.abspath(os.path.join(_script_dir, "..", ".."))
USD_PATH     = os.path.join(_project_dir, "MSOEBuildingV4_physics.usd")
_out_name    = "visual_review_grid_angles" if ANGLE_MIX else "visual_review_gridV2"
OUT_DIR      = os.path.join(_project_dir, "data", _out_name)
MANIFEST_PATH = os.path.join(OUT_DIR, "manifest.json")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Lighting conditions — verbatim from LIGHTING_AND_MATERIALS.md
# ---------------------------------------------------------------------------

LIGHTING_CONDITIONS = [
    {"name": "bright_fluorescent", "dome_int": 15_000, "dome_temp": 5500,
     "ceiling_pattern": (1, 1, 1, 1), "ceiling_int": 400_000, "ceiling_temp": 5000},
    {"name": "daylight_flood",     "dome_int": 50_000, "dome_temp": 6500,
     "ceiling_pattern": (1, 0, 1, 0), "ceiling_int": 230_000, "ceiling_temp": 6500},
    {"name": "warm_evening",       "dome_int":  8_000, "dome_temp": 3000,
     "ceiling_pattern": (1, 1, 1, 1), "ceiling_int": 200_000, "ceiling_temp": 2700},
    {"name": "dim_night",          "dome_int":  2_000, "dome_temp": 4000,
     "ceiling_pattern": (1, 0, 1, 0), "ceiling_int": 150_000, "ceiling_temp": 3500},
    {"name": "overcast_flat",      "dome_int": 25_000, "dome_temp": 5000,
     "ceiling_pattern": (1, 1, 1, 1), "ceiling_int": 220_000, "ceiling_temp": 5000},
    {"name": "high_contrast",      "dome_int":  2_000, "dome_temp": 5000,
     "ceiling_pattern": (0, 1, 0, 0), "ceiling_int": 700_000, "ceiling_temp": 5500},
]

# ---------------------------------------------------------------------------
# Door-material presets — 6 uniform presets, all fill_* techniques so they
# work identically on curtain-wall and solid families (fill_* binds to both
# frame and glass, so no family-specific logic is needed).
# ---------------------------------------------------------------------------

DOOR_MATERIALS = [
    {"name": "fill_red",               "technique": "fill_glass",
     "color": (0.65, 0.10, 0.08), "roughness": 0.55, "metallic": 0.02},
    {"name": "fill_white",             "technique": "fill_glass",
     "color": (0.90, 0.90, 0.88), "roughness": 0.45, "metallic": 0.02},
    {"name": "fill_wood",              "technique": "fill_glass",
     "color": (0.55, 0.35, 0.20), "roughness": 0.70, "metallic": 0.02},
    {"name": "fill_tex_red_brushed",   "technique": "fill_textured",
     "texture": "brushed", "tint": (0.80, 0.13, 0.10),
     "roughness": 0.65, "metallic": 0.05},
    {"name": "fill_tex_wood_grain",    "technique": "fill_textured",
     "texture": "grain",   "tint": (0.70, 0.45, 0.25),
     "roughness": 0.70, "metallic": 0.02},
    {"name": "fill_tex_dark_brushed",  "technique": "fill_textured",
     "texture": "brushed", "tint": (0.20, 0.20, 0.22),
     "roughness": 0.55, "metallic": 0.15},
]

# ---------------------------------------------------------------------------
# Wall colour palette — 6 broad-hue presets for the C1 swap on solid doors.
# These are the same presets the production random-pool sampler will draw
# from, so the review grid is a 1:1 preview of what the training data will
# see on the wall axis.
# ---------------------------------------------------------------------------

WALL_COLORS = [
    {"name": "warm_beige",    "color": (0.85, 0.78, 0.65), "roughness": 0.85, "metallic": 0.0},
    {"name": "cool_gray",     "color": (0.72, 0.75, 0.78), "roughness": 0.80, "metallic": 0.0},
    {"name": "terracotta",    "color": (0.70, 0.40, 0.30), "roughness": 0.85, "metallic": 0.0},
    {"name": "sage_green",    "color": (0.65, 0.75, 0.58), "roughness": 0.85, "metallic": 0.0},
    {"name": "muted_blue",    "color": (0.55, 0.65, 0.80), "roughness": 0.80, "metallic": 0.0},
    {"name": "dark_charcoal", "color": (0.30, 0.30, 0.32), "roughness": 0.85, "metallic": 0.0},
]

SOLID_FAMILIES = {"Type_F", "Type_FG", "Type_N"}

# ---------------------------------------------------------------------------
# Door registry — 10 doors (8 train + 2 holdout), verbatim from
# lighting_test_all_doors.py.
# ---------------------------------------------------------------------------

_BASE = (
    "/World/Building/_22_129___MSOE_Engineering_Building___A___2_25_2026"
    "/Geometry/Doors"
)

DOORS = [
    # ── TRAIN ──────────────────────────────────────────────────────────────
    {"id": "Doors_13606466", "family_short": "CW_F_NEW",  "floor": 4, "role": "train",
     "prim_path": f"{_BASE}/RS___Curtain_Wall___Type_F___NEW/New/Doors_13606466",
     "mesh_count": 1, "hinge_x_cm": -4836.01, "hinge_y_cm": 6170.58,
     "cam_pos": [-46.360, 62.163, 23.704], "cam_yaw_deg": 180.0,
     "door_center_m": [-48.360, 62.163, 23.704], "angle_sign": 1,
     "leaf_mesh_names": None},
    {"id": "Doors_10649156", "family_short": "Type_FG",   "floor": 3, "role": "train",
     "prim_path": f"{_BASE}/RS___Type_FG_Door/NEW___HM_Frame___36_x96/Doors_10649156",
     "mesh_count": 0, "hinge_x_cm": +50.80, "hinge_y_cm": 0.0,
     "cam_pos": [-14.295, 80.200, 19.437], "cam_yaw_deg": 0.0,
     "door_center_m": [-12.295, 80.200, 19.437], "angle_sign": -1,
     "leaf_mesh_names": None},
    {"id": "Doors_8286057",  "family_short": "Type_F",    "floor": 2, "role": "train",
     "prim_path": f"{_BASE}/RS___Type_F_Door/NEW___HM_Frame___36_x96/Doors_8286057",
     "mesh_count": 0, "hinge_x_cm": +50.80, "hinge_y_cm": 0.0,
     "cam_pos": [17.443, 79.341, 14.764], "cam_yaw_deg": 270.0,
     "door_center_m": [17.443, 77.341, 14.764], "angle_sign": 1,
     "leaf_mesh_names": None},
    {"id": "Doors_3135827",  "family_short": "CW_FG_New", "floor": 3, "role": "train",
     "prim_path": f"{_BASE}/RS___Curtain_Wall___Type_FG___New/New/Doors_3135827",
     "mesh_count": 2, "hinge_x_cm": 1724.33, "hinge_y_cm": 5878.48,
     "cam_pos": [17.701, 56.807, 19.437], "cam_yaw_deg": 90.0,
     "door_center_m": [17.701, 58.807, 19.437], "angle_sign": 1,
     "leaf_mesh_names": None},
    {"id": "Doors_7171063",  "family_short": "Type_F",    "floor": 5, "role": "train",
     "prim_path": f"{_BASE}/RS___Type_F_Door/NEW___HM_Frame___42_x96/Doors_7171063",
     "mesh_count": 0, "hinge_x_cm": +58.42, "hinge_y_cm": 0.0,
     "cam_pos": [-49.316, 49.892, 27.972], "cam_yaw_deg": 0.0,
     "door_center_m": [-47.316, 49.892, 27.972], "angle_sign": -1,
     "leaf_mesh_names": None},
    {"id": "Doors_12590968", "family_short": "Type_N",    "floor": 3, "role": "train",
     "prim_path": f"{_BASE}/RS___Type_N_Door/NEW___HM_Frame___36_x96/Doors_12590968",
     "mesh_count": 0, "hinge_x_cm": +50.80, "hinge_y_cm": 0.0,
     "cam_pos": [17.878, 50.941, 19.437], "cam_yaw_deg": 90.0,
     "door_center_m": [17.878, 52.941, 19.437], "angle_sign": 1,
     "leaf_mesh_names": None},
    {"id": "Doors_12302795", "family_short": "CW_FG_New", "floor": 2, "role": "train",
     "prim_path": f"{_BASE}/RS___Curtain_Wall___Type_FG___New/New___GL1_1/Doors_12302795",
     "mesh_count": 2, "hinge_x_cm": -5925.83, "hinge_y_cm": 5774.34,
     "cam_pos": [-57.258, 58.251, 14.561], "cam_yaw_deg": 180.0,
     "door_center_m": [-59.258, 58.251, 14.561], "angle_sign": 1,
     "leaf_mesh_names": None},
    {"id": "Doors_13606056", "family_short": "CW_F_NEW",  "floor": 3, "role": "train",
     "prim_path": f"{_BASE}/RS___Curtain_Wall___Type_F___NEW/New/Doors_13606056",
     "mesh_count": 1, "hinge_x_cm": -4747.75, "hinge_y_cm": 6958.30,
     "cam_pos": [-45.477, 70.040, 19.437], "cam_yaw_deg": 180.0,
     "door_center_m": [-47.477, 70.040, 19.437], "angle_sign": 1,
     "leaf_mesh_names": None},
    # ── HOLDOUT ────────────────────────────────────────────────────────────
    {"id": "Doors_13182085", "family_short": "CW_FG_New", "floor": 4, "role": "holdout",
     "prim_path": f"{_BASE}/RS___Curtain_Wall___Type_FG___New/New/Doors_13182085",
     "mesh_count": 2, "hinge_x_cm": -4142.56, "hinge_y_cm": 8214.32,
     "cam_pos": [-39.426, 82.600, 23.704], "cam_yaw_deg": 180.0,
     "door_center_m": [-41.426, 82.600, 23.704], "angle_sign": 1,
     "leaf_mesh_names": None},
    {"id": "Doors_1915613",  "family_short": "Type_F",    "floor": 5, "role": "holdout",
     "prim_path": f"{_BASE}/RS___Type_F_Door/NEW___HM_Frame___36_x96/Doors_1915613",
     "mesh_count": 0, "hinge_x_cm": +50.80, "hinge_y_cm": 0.0,
     "cam_pos": [-58.753, 65.219, 27.972], "cam_yaw_deg": 270.0,
     "door_center_m": [-58.753, 63.219, 27.972], "angle_sign": 1,
     "leaf_mesh_names": None},
]

# ---------------------------------------------------------------------------
# Isaac Sim startup
# ---------------------------------------------------------------------------

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

import time as _time
_t0 = _time.time()
def _ts(msg):
    print(f"[ReviewGrid] [+{_time.time() - _t0:6.1f}s] {msg}")

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
    print("[WARN] PIL unavailable — textured presets will be skipped and "
          "frames saved as .npy")

# ---------------------------------------------------------------------------
# Procedural texture generation (brushed / grain / speckle) — copied from
# door_replicator_collection.py. Required by fill_tex_* presets.
# ---------------------------------------------------------------------------

def _generate_procedural_textures(scratch_dir):
    textures = {}
    if not _pil_ok:
        return textures
    os.makedirs(scratch_dir, exist_ok=True)
    h = w = 512
    rng = np.random.default_rng(42)

    def _save(name, gray):
        rgb = np.stack([gray, gray, gray], axis=-1)
        rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        p = os.path.join(scratch_dir, f"{name}.png")
        _PIL.fromarray(rgb).save(p)
        textures[name] = p

    # Narrowed intensity ranges (0.80–0.95) vs original (0.65–1.0) so that
    # diffuse_tint multiplication stays within one perceptual hue — dark/bright
    # bands no longer shift color under the tint.
    base = np.full((h, w), 0.88, dtype=np.float32)
    base += rng.uniform(-0.04, 0.04, (h, 1)).astype(np.float32)
    base += rng.uniform(-0.01, 0.01, (h, w)).astype(np.float32)
    _save("brushed", base)

    xs = np.arange(w, dtype=np.float32)
    band = 0.5 * (np.sin(xs * 0.04) + np.sin(xs * 0.11 + 1.7))
    base = 0.88 + 0.05 * band[None, :]
    base = np.broadcast_to(base, (h, w)).copy()
    base += rng.uniform(-0.02, 0.02, (h, w)).astype(np.float32)
    _save("grain", base)

    base = np.full((h, w), 0.88, dtype=np.float32)
    base += rng.uniform(-0.03, 0.03, (h, w)).astype(np.float32)
    _save("speckle", base)

    return textures

_tex_dir = os.path.join(_project_dir, "data", "visual_review_grid_textures")
_procedural_textures = _generate_procedural_textures(_tex_dir)

# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------

world = World(stage_units_in_meters=1.0)
stage = omni.usd.get_context().get_stage()

_ts(f"Inserting building USD sublayer: {USD_PATH}")
stage.GetRootLayer().subLayerPaths.insert(0, USD_PATH)

dome = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/DomeLight"))
dome.CreateIntensityAttr(25_000)
dome.GetColorAttr().Set(Gf.Vec3f(1.0, 1.0, 1.0))

# 4-slot ceiling array with mutable translate ops (per-door reposition).
_CEILING_SPREAD = 1.2
_CEILING_RISE   = 1.4
_CEILING_OFFSETS = [
    (-_CEILING_SPREAD, -_CEILING_SPREAD),
    (+_CEILING_SPREAD, -_CEILING_SPREAD),
    (-_CEILING_SPREAD, +_CEILING_SPREAD),
    (+_CEILING_SPREAD, +_CEILING_SPREAD),
]

ceiling_t_ops     = []
ceiling_int_attrs = []
ceiling_col_attrs = []
for i, (dx, dy) in enumerate(_CEILING_OFFSETS):
    rl = UsdLux.RectLight.Define(stage, Sdf.Path(f"/World/CeilingLight_{i}"))
    rl.CreateIntensityAttr(0.0)
    rl.CreateWidthAttr(0.6)
    rl.CreateHeightAttr(0.6)
    xf = UsdGeom.Xformable(rl.GetPrim())
    t_op = xf.AddTranslateOp()
    r_op = xf.AddRotateXYZOp()
    r_op.Set(Gf.Vec3f(180.0, 0.0, 0.0))
    t_op.Set(Gf.Vec3d(dx, dy, 5.0))
    ceiling_t_ops.append(t_op)
    ceiling_int_attrs.append(rl.GetIntensityAttr())
    ceiling_col_attrs.append(rl.GetColorAttr())

_ts("world.reset() ...")
world.reset()
_ts("world.reset() done")
for _ in range(5):
    world.step(render=True)

# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------

cam = Camera(
    prim_path="/World/ReviewCam",
    position=np.array([0.0, 0.0, 5.0]),
    frequency=30,
    resolution=CAM_RESOLUTION,
    orientation=rot_utils.euler_angles_to_quats(
        np.array([0.0, 0.0, 180.0]), degrees=True),
)
simulation_app.update()
cam.initialize()

usd_cam = UsdGeom.Camera(stage.GetPrimAtPath("/World/ReviewCam"))
usd_cam.GetFocalLengthAttr().Set(5.0)
usd_cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 10000.0))

rp  = rep.create.render_product("/World/ReviewCam", CAM_RESOLUTION)
ann = rep.AnnotatorRegistry.get_annotator("rgb")
ann.attach([rp])

_ts(f"Renderer cold-start warmup ({STARTUP_WARMUP} steps) ...")
for _ in range(STARTUP_WARMUP):
    world.step(render=True)
_ts("Warmup done")

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _kelvin_to_rgb(k: float) -> Gf.Vec3f:
    t = k / 100.0
    r = 255 if t <= 66 else max(0, min(255, 329.698727446 * ((t - 60) ** -0.1332047592)))
    if t <= 66:
        g = 99.4708025861 * math.log(t) - 161.1195681661
    else:
        g = 288.1221695283 * ((t - 60) ** -0.0755148492)
    g = max(0, min(255, g))
    b = 255 if t >= 66 else (0 if t <= 19 else
        max(0, min(255, 138.5177312231 * math.log(t - 10) - 305.0447927307)))
    return Gf.Vec3f(r / 255.0, g / 255.0, b / 255.0)


def apply_lighting(cond: dict):
    ji = lambda v: v * random.uniform(0.90, 1.10)
    jt = lambda v: v + random.uniform(-200, 200)
    dome.GetIntensityAttr().Set(ji(cond["dome_int"]))
    dome.GetColorAttr().Set(_kelvin_to_rgb(jt(cond["dome_temp"])))
    ci = ji(cond["ceiling_int"])
    ct = jt(cond["ceiling_temp"])
    rgb = _kelvin_to_rgb(ct)
    for slot, (ia, ca) in enumerate(zip(ceiling_int_attrs, ceiling_col_attrs)):
        on = cond["ceiling_pattern"][slot]
        ia.Set(ci if on else 0.0)
        ca.Set(rgb)


def reposition_lights(door_center_m, cam_z):
    cx, cy = door_center_m[0], door_center_m[1]
    cz = cam_z + _CEILING_RISE
    for i, (dx, dy) in enumerate(_CEILING_OFFSETS):
        ceiling_t_ops[i].Set(Gf.Vec3d(cx + dx, cy + dy, cz))


def reposition_camera(pos, yaw_deg):
    ori = rot_utils.euler_angles_to_quats(
        np.array([0.0, 0.0, yaw_deg]), degrees=True)
    cam.set_world_pose(position=np.array(pos), orientation=ori)


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


def set_angle(mesh_ops, angle_deg, hx, hy):
    for m, op in mesh_ops:
        op.Set(_door_matrix(m["ox"], m["oy"], m["oz"], angle_deg, hx, hy))


def discover_door_meshes(door_prim, mesh_count, leaf_mesh_names):
    mesh_ops = []
    prim_iter = (door_prim.GetChildren() if mesh_count > 0
                 else Usd.PrimRange(door_prim))
    for p in prim_iter:
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


def _is_glass_mesh(p) -> bool:
    n = p.GetPath().name.lower()
    return (n.startswith("glass_") or "_gl_" in n or "glazing" in n or "pane_" in n)


# ---------------------------------------------------------------------------
# Material prim factories (global — compiled once, bindings swapped per cell)
# ---------------------------------------------------------------------------

def _make_door_omnipbr(preset, tex_path=None):
    name = preset["name"]
    mp = f"/World/Looks/DoorMat_{name}"
    mat = UsdShade.Material.Define(stage, Sdf.Path(mp))
    sh  = UsdShade.Shader.Define(stage, Sdf.Path(mp + "/Shader"))
    sh.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
    sh.SetSourceAsset(Sdf.AssetPath("OmniPBR.mdl"), "mdl")
    sh.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
    if tex_path is not None:
        sh.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset).Set(
            Sdf.AssetPath(tex_path))
        sh.CreateInput("project_uvw", Sdf.ValueTypeNames.Bool).Set(True)
        sh.CreateInput("world_or_object", Sdf.ValueTypeNames.Bool).Set(False)
        sh.CreateInput("texture_scale", Sdf.ValueTypeNames.Float2).Set(
            Gf.Vec2f(2.0, 2.0))
        sh.CreateInput("diffuse_tint", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(*preset["tint"]))
    else:
        sh.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(*preset["color"]))
    sh.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(
        float(preset["roughness"]))
    sh.CreateInput("metallic_constant", Sdf.ValueTypeNames.Float).Set(
        float(preset["metallic"]))
    mat.CreateSurfaceOutput("mdl").ConnectToSource(sh.ConnectableAPI(), "out")
    mat.CreateDisplacementOutput("mdl").ConnectToSource(sh.ConnectableAPI(), "out")
    mat.CreateVolumeOutput("mdl").ConnectToSource(sh.ConnectableAPI(), "out")
    return mat


def _make_wall_omnipbr(preset):
    name = preset["name"]
    mp = f"/World/Looks/WallMat_{name}"
    mat = UsdShade.Material.Define(stage, Sdf.Path(mp))
    sh  = UsdShade.Shader.Define(stage, Sdf.Path(mp + "/Shader"))
    sh.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
    sh.SetSourceAsset(Sdf.AssetPath("OmniPBR.mdl"), "mdl")
    sh.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
    sh.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(*preset["color"]))
    sh.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(
        float(preset["roughness"]))
    sh.CreateInput("metallic_constant", Sdf.ValueTypeNames.Float).Set(
        float(preset["metallic"]))
    mat.CreateSurfaceOutput("mdl").ConnectToSource(sh.ConnectableAPI(), "out")
    mat.CreateDisplacementOutput("mdl").ConnectToSource(sh.ConnectableAPI(), "out")
    mat.CreateVolumeOutput("mdl").ConnectToSource(sh.ConnectableAPI(), "out")
    return mat


_door_mat_prims = {}
for preset in DOOR_MATERIALS:
    tech = preset["technique"]
    if tech == "fill_glass":
        _door_mat_prims[preset["name"]] = _make_door_omnipbr(preset)
    elif tech == "fill_textured":
        tex = _procedural_textures.get(preset["texture"])
        if tex is None:
            print(f"[WARN] texture '{preset['texture']}' missing — "
                  f"skipping {preset['name']}")
            continue
        _door_mat_prims[preset["name"]] = _make_door_omnipbr(preset, tex_path=tex)
    else:
        raise ValueError(f"unexpected technique: {tech}")
print(f"[ReviewGrid] Pre-created {len(_door_mat_prims)} door material prims")

_wall_mat_prims = {p["name"]: _make_wall_omnipbr(p) for p in WALL_COLORS}
print(f"[ReviewGrid] Pre-created {len(_wall_mat_prims)} wall material prims")

# ---------------------------------------------------------------------------
# Per-door binding helpers (cached at the start of each door)
# ---------------------------------------------------------------------------

_strong = UsdShade.Tokens.strongerThanDescendants


def cache_door_bindings(door_prim):
    frame_cache = []
    glass_cache = []
    for p in Usd.PrimRange(door_prim):
        if not p.IsA(UsdGeom.Mesh):
            continue
        rel = UsdShade.MaterialBindingAPI(p).GetDirectBindingRel()
        targets = list(rel.GetTargets()) if rel else []
        (glass_cache if _is_glass_mesh(p) else frame_cache).append((p, targets))
    return frame_cache, glass_cache


def restore_bindings(cache):
    for mesh_prim, orig_targets in cache:
        api = UsdShade.MaterialBindingAPI(mesh_prim)
        api.UnbindAllBindings()
        if not orig_targets:
            continue
        rel = api.GetDirectBindingRel() or api.CreateDirectBindingRel()
        for tgt in orig_targets:
            rel.AddTarget(tgt)


def apply_door_material_to_prim(door_prim, preset, frame_cache, glass_cache):
    restore_bindings(frame_cache)
    restore_bindings(glass_cache)
    mat = _door_mat_prims.get(preset["name"])
    if mat is None:
        return
    # Both fill_glass and fill_textured bind to frame + glass.
    for p in Usd.PrimRange(door_prim):
        if not p.IsA(UsdGeom.Mesh):
            continue
        UsdShade.MaterialBindingAPI.Apply(p)
        UsdShade.MaterialBindingAPI(p).Bind(mat, bindingStrength=_strong)


# ---------------------------------------------------------------------------
# Wall discovery + swap (solid families only)
# ---------------------------------------------------------------------------

_BUILDING_ROOT = "/World/Building/_22_129___MSOE_Engineering_Building___A___2_25_2026"
_WALLS_ROOT_PATH = f"{_BUILDING_ROOT}/Geometry/Walls"
_WALL_SEARCH_RADIUS_CM = 500.0      # 5 m horizontal
_WALL_FLOOR_BAND_CM    = 400.0      # ±4 m vertical (one floor)


def _xform_translation_cm(prim):
    xf = UsdGeom.Xformable(prim)
    op = next((o for o in xf.GetOrderedXformOps()
               if o.GetOpName() == "xformOp:transform"), None)
    if op is None:
        return None
    m = op.Get()
    if m is None:
        return None
    return float(m[3][0]), float(m[3][1]), float(m[3][2])


def discover_wall_meshes(door_center_m):
    walls_root = stage.GetPrimAtPath(_WALLS_ROOT_PATH)
    if not walls_root.IsValid():
        return []
    cx_cm = door_center_m[0] * 100.0
    cy_cm = door_center_m[1] * 100.0
    cz_cm = door_center_m[2] * 100.0
    found = []
    for prim in Usd.PrimRange(walls_root):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        t = _xform_translation_cm(prim)
        if t is None:
            continue
        tx, ty, tz = t
        if abs(tz - cz_cm) > _WALL_FLOOR_BAND_CM:
            continue
        d = math.hypot(tx - cx_cm, ty - cy_cm)
        if d > _WALL_SEARCH_RADIUS_CM:
            continue
        found.append(prim)
    return found


def cache_wall_bindings(wall_meshes):
    cache = []
    for p in wall_meshes:
        rel = UsdShade.MaterialBindingAPI(p).GetDirectBindingRel()
        cache.append((p, list(rel.GetTargets()) if rel else []))
    return cache


def apply_wall_color(wall_meshes, preset):
    mat = _wall_mat_prims[preset["name"]]
    for p in wall_meshes:
        UsdShade.MaterialBindingAPI.Apply(p)
        UsdShade.MaterialBindingAPI(p).Bind(mat, bindingStrength=_strong)


# ---------------------------------------------------------------------------
# Capture helper
# ---------------------------------------------------------------------------

def capture(out_path) -> bool:
    world.step(render=True)
    data = ann.get_data()
    if data is None:
        return False
    arr = np.asarray(data)
    if arr.ndim != 3 or arr.shape[-1] < 3:
        return False
    rgb = arr[:, :, :3].astype(np.uint8)
    if _pil_ok:
        _PIL.fromarray(rgb).save(out_path, "JPEG", quality=92)
    else:
        np.save(out_path.replace(".jpg", ".npy"), rgb)
    return True


# ---------------------------------------------------------------------------
# Prime renderer
# ---------------------------------------------------------------------------

apply_lighting(LIGHTING_CONDITIONS[0])
_ts("Global lighting settle (20 steps, bright_fluorescent active) ...")
for _ in range(20):
    world.step(render=True)
_ts("Settle done — starting door loop")

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

manifest = {
    "grid": {
        "lightings":      [c["name"] for c in LIGHTING_CONDITIONS],
        "door_materials": [m["name"] for m in DOOR_MATERIALS],
        "wall_colors":    [w["name"] for w in WALL_COLORS],
        "angle_mix":      ANGLE_MIX,
        "angle_sweep":    ANGLE_SWEEP if ANGLE_MIX else [CLOSED_ANGLE],
    },
    "doors": [],
}

total_saved = 0
total_expected = 0

for di, door in enumerate(DOORS):
    fam = door["family_short"]
    is_solid = fam in SOLID_FAMILIES
    role_tag = "[HOLDOUT]" if door["role"] == "holdout" else "[train  ]"
    door_prefix = f"{fam}_{door['id']}"

    print(f"\n{'─'*65}")
    _ts(f"[{di+1:2d}/{len(DOORS)}] {role_tag} {fam:<12} {door['id']}  "
        f"floor={door['floor']}  {'solid' if is_solid else 'curtain-wall'}")

    door_entry = {
        "id":             door["id"],
        "family_short":   fam,
        "floor":          door["floor"],
        "role":           door["role"],
        "is_solid":       is_solid,
        "cells_saved":    0,
        "wall_saved":     0,
        "status":         "ok",
        "note":           "",
    }

    # Validate + concretize
    door_prim = stage.GetPrimAtPath(door["prim_path"])
    if not door_prim.IsValid():
        door_entry.update({"status": "skipped", "note": "prim not valid"})
        manifest["doors"].append(door_entry)
        _ts("  SKIP — prim not valid")
        continue
    if door_prim.IsInstance() or door_prim.IsInstanceable():
        door_prim.SetInstanceable(False)
        simulation_app.update()

    mesh_ops = discover_door_meshes(
        door_prim, door["mesh_count"], door["leaf_mesh_names"])
    if not mesh_ops:
        door_entry.update({"status": "skipped", "note": "no xform meshes"})
        manifest["doors"].append(door_entry)
        _ts("  SKIP — no xformOp:transform meshes")
        continue

    frame_cache, glass_cache = cache_door_bindings(door_prim)
    _ts(f"  door meshes={len(mesh_ops)}  "
        f"frame bindings={len(frame_cache)}  glass bindings={len(glass_cache)}")

    # Wall discovery (solid families only)
    wall_meshes = []
    wall_cache = []
    if is_solid:
        wall_meshes = discover_wall_meshes(door["door_center_m"])
        wall_cache  = cache_wall_bindings(wall_meshes)
        _ts(f"  wall meshes in range: {len(wall_meshes)}")
        if len(wall_meshes) == 0:
            print("  [NOTE] no wall meshes found — wall swatches will show "
                  "original wall (C1 no-op)")

    # Reposition camera + ceiling lights
    reposition_camera(door["cam_pos"], door["cam_yaw_deg"])
    reposition_lights(door["door_center_m"], door["cam_pos"][2])

    # Close the door and prime
    closed_signed = door["angle_sign"] * CLOSED_ANGLE
    set_angle(mesh_ops, closed_signed, door["hinge_x_cm"], door["hinge_y_cm"])
    apply_lighting(LIGHTING_CONDITIONS[0])
    for _ in range(DOOR_WARMUP):
        world.step(render=True)

    # ---- 6 × 6 lighting × material cells ----
    door_cells = len(LIGHTING_CONDITIONS) * len(DOOR_MATERIALS)
    total_expected += door_cells

    # Per-door balanced angle plan: each of ANGLE_SWEEP appears
    # door_cells/len(ANGLE_SWEEP) times, shuffled by a deterministic
    # per-door RNG so runs are reproducible.
    if ANGLE_MIX:
        per_angle = door_cells // len(ANGLE_SWEEP)
        angle_plan = list(ANGLE_SWEEP) * per_angle
        angle_plan += list(ANGLE_SWEEP)[: door_cells - len(angle_plan)]
        random.Random(1000 + di).shuffle(angle_plan)
        door_entry["angle_plan"] = angle_plan
    else:
        angle_plan = [CLOSED_ANGLE] * door_cells

    for li, lc in enumerate(LIGHTING_CONDITIONS):
        apply_lighting(lc)
        for mi, mat_preset in enumerate(DOOR_MATERIALS):
            cell_angle = angle_plan[li * len(DOOR_MATERIALS) + mi]
            set_angle(mesh_ops,
                      door["angle_sign"] * cell_angle,
                      door["hinge_x_cm"], door["hinge_y_cm"])
            apply_door_material_to_prim(
                door_prim, mat_preset, frame_cache, glass_cache)
            for _ in range(CELL_WARMUP):
                world.step(render=True)
            out = os.path.join(OUT_DIR, f"{door_prefix}_cell_L{li}_M{mi}.jpg")
            ok = capture(out)
            if ok:
                door_entry["cells_saved"] += 1
                total_saved += 1
            else:
                print(f"  [FAIL] L{li} M{mi}")

    # Restore door to original bindings before wall swatches.
    restore_bindings(frame_cache)
    restore_bindings(glass_cache)

    # ---- 6 wall swatches (solid only) ----
    if is_solid:
        total_expected += len(WALL_COLORS)
        # Re-close the door so the wall-colour axis reads cleanly
        # (the L×M loop above may have left it open when ANGLE_MIX is on).
        set_angle(mesh_ops, closed_signed,
                  door["hinge_x_cm"], door["hinge_y_cm"])
        apply_lighting(LIGHTING_CONDITIONS[0])   # bright_fluorescent
        for _ in range(CELL_WARMUP):
            world.step(render=True)
        for wi, wc in enumerate(WALL_COLORS):
            restore_bindings(wall_cache)
            if wall_meshes:
                apply_wall_color(wall_meshes, wc)
            for _ in range(CELL_WARMUP):
                world.step(render=True)
            out = os.path.join(OUT_DIR, f"{door_prefix}_wall_{wi}.jpg")
            ok = capture(out)
            if ok:
                door_entry["wall_saved"] += 1
                total_saved += 1
            else:
                print(f"  [FAIL] wall_{wi}")
        restore_bindings(wall_cache)

    print(f"  → {door_entry['cells_saved']}/{door_cells} cells  "
          f"{door_entry['wall_saved']}/{len(WALL_COLORS) if is_solid else 0} walls")
    manifest["doors"].append(door_entry)

# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

manifest["total_saved"]   = total_saved
manifest["total_expected"] = total_expected

with open(MANIFEST_PATH, "w") as f:
    json.dump(manifest, f, indent=2)
    f.flush()
    os.fsync(f.fileno())

print(f"\n{'='*65}")
_ts(f"Done — saved {total_saved}/{total_expected} JPEGs")
_ts(f"Output   → {OUT_DIR}/")
_ts(f"Manifest → {MANIFEST_PATH}")

world.stop()
simulation_app.close()

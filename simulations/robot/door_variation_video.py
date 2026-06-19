#!/usr/bin/env python3
"""
Door-variation showcase video — frame collector.

Purpose
-------
Render a structured montage that makes the "infinite variations" claim of the
digital twin visually obvious. Three acts, all rendered at 1920x1080 with RTX
Path Tracing on a single representative door (Act 1/2), then rotating through
all 10 doors (Act 3):

  Act 1 — solo axis sweeps on the hero door (lighting, door material,
          wall color, door angle). One axis at a time — the viewer's eye
          picks up what's changing without needing labels.
  Act 2 — combined random sweeps on the same hero door. All four axes
          move together per frame — the composition becomes visible.
  Act 3 — combined random sweeps across all 10 doors. Establishes the
          breadth of variation the twin can produce.

No axes overlay, no real-world photo anchors — per presentation spec.

Output
------
  data/variation_video_frames/NNNN.jpg   (zero-padded, 1920x1080)

Render
------
  sbatch submit_isaac_job.sh simulations/robot/door_variation_video.py

Renderer note: uses RTX Real-Time (rasterization + approximate ray tracing)
rather than Path Tracing. RTX Path Tracing is unstable on T4 (crashes during
accumulation), and the V100 `dgx` partition can't run Isaac Sim at all.
Real-Time is the same renderer that produced data/visual_review_gridV2 and
looks clean at 1920x1080.

Assemble video
--------------
  python simulations/robot/frames_to_video.py \
      data/variation_video_frames --fps 4 -o data/variation_video.mp4

Why 4 fps: each variation needs to sit on screen long enough for the audience
to register the change. 30 fps blurs the combinatorial point; 4 fps lets each
lighting/material/wall/angle pairing read cleanly.
"""

import os
import sys
import math
import json
import random
import functools
import numpy as np

print = functools.partial(print, flush=True)  # noqa: A001

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CAM_RESOLUTION = (1920, 1080)
STARTUP_WARMUP = 30
DOOR_WARMUP    = 20         # after camera + light reposition
CELL_WARMUP    = 12         # after each material/lighting/angle swap

CLOSED_ANGLE   = 1.5
ANGLE_SWEEP    = [1.5, 18.0, 36.0, 54.0, 72.0, 90.0]  # closed -> fully open

HERO_DOOR_ID   = "Doors_8286057"   # Type_F solid (train), clean composition

# Camera jitter ranges — kept small so the door stays framed.
YAW_JITTER_DEG = 2.0
Z_JITTER_M     = 0.10

# Act sizes
ACT2_COMBINED_FRAMES  = 12
ACT3_FRAMES_PER_DOOR  = 5

_script_dir  = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.abspath(os.path.join(_script_dir, "..", ".."))
USD_PATH     = os.path.join(_project_dir, "MSOEBuildingV4_physics.usd")
OUT_DIR      = os.path.join(_project_dir, "data", "variation_video_frames")
MANIFEST_PATH = os.path.join(OUT_DIR, "manifest.json")
os.makedirs(OUT_DIR, exist_ok=True)

# Deterministic run
random.seed(7)
np.random.seed(7)

# ---------------------------------------------------------------------------
# Presets (verbatim from visual_review_grid.py — single source of truth)
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

DOOR_MATERIALS = [
    {"name": "fill_red",              "technique": "fill_glass",
     "color": (0.65, 0.10, 0.08), "roughness": 0.55, "metallic": 0.02},
    {"name": "fill_white",            "technique": "fill_glass",
     "color": (0.90, 0.90, 0.88), "roughness": 0.45, "metallic": 0.02},
    {"name": "fill_wood",             "technique": "fill_glass",
     "color": (0.55, 0.35, 0.20), "roughness": 0.70, "metallic": 0.02},
    {"name": "fill_tex_red_brushed",  "technique": "fill_textured",
     "texture": "brushed", "tint": (0.80, 0.13, 0.10),
     "roughness": 0.65, "metallic": 0.05},
    {"name": "fill_tex_wood_grain",   "technique": "fill_textured",
     "texture": "grain",   "tint": (0.70, 0.45, 0.25),
     "roughness": 0.70, "metallic": 0.02},
    {"name": "fill_tex_dark_brushed", "technique": "fill_textured",
     "texture": "brushed", "tint": (0.20, 0.20, 0.22),
     "roughness": 0.55, "metallic": 0.15},
]

WALL_COLORS = [
    {"name": "warm_beige",    "color": (0.85, 0.78, 0.65), "roughness": 0.85, "metallic": 0.0},
    {"name": "cool_gray",     "color": (0.72, 0.75, 0.78), "roughness": 0.80, "metallic": 0.0},
    {"name": "terracotta",    "color": (0.70, 0.40, 0.30), "roughness": 0.85, "metallic": 0.0},
    {"name": "sage_green",    "color": (0.65, 0.75, 0.58), "roughness": 0.85, "metallic": 0.0},
    {"name": "muted_blue",    "color": (0.55, 0.65, 0.80), "roughness": 0.80, "metallic": 0.0},
    {"name": "dark_charcoal", "color": (0.30, 0.30, 0.32), "roughness": 0.85, "metallic": 0.0},
]

_BASE = (
    "/World/Building/_22_129___MSOE_Engineering_Building___A___2_25_2026"
    "/Geometry/Doors"
)

DOORS = [
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
# Isaac Sim startup — RTX Path Tracing
# ---------------------------------------------------------------------------

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

import time as _time
_t0 = _time.time()
def _ts(msg):
    print(f"[VarVid] [+{_time.time() - _t0:6.1f}s] {msg}")

_ts("Importing SimulationApp (RTX Real-Time) ...")
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})
_ts("SimulationApp ready")

import carb
_s = carb.settings.get_settings()
_s.set("/rtx/post/aa/op", 4)        # anti-aliasing
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
    print("[WARN] PIL unavailable — frames will be saved as .npy")

# ---------------------------------------------------------------------------
# Procedural textures (brushed / grain) — same recipe as visual_review_grid
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

    return textures


_tex_dir = os.path.join(_project_dir, "data", "variation_video_textures")
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
for _ in range(5):
    world.step(render=True)

# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------

cam = Camera(
    prim_path="/World/ShowcaseCam",
    position=np.array([0.0, 0.0, 5.0]),
    frequency=30,
    resolution=CAM_RESOLUTION,
    orientation=rot_utils.euler_angles_to_quats(
        np.array([0.0, 0.0, 180.0]), degrees=True),
)
simulation_app.update()
cam.initialize()

usd_cam = UsdGeom.Camera(stage.GetPrimAtPath("/World/ShowcaseCam"))
usd_cam.GetFocalLengthAttr().Set(5.0)
usd_cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 10000.0))

rp  = rep.create.render_product("/World/ShowcaseCam", CAM_RESOLUTION)
ann = rep.AnnotatorRegistry.get_annotator("rgb")
ann.attach([rp])

_ts(f"Renderer cold-start warmup ({STARTUP_WARMUP} steps) ...")
for _ in range(STARTUP_WARMUP):
    world.step(render=True)
_ts("Warmup done")

# ---------------------------------------------------------------------------
# Shared helpers (trimmed from visual_review_grid.py)
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


def set_camera(pos, yaw_deg):
    ori = rot_utils.euler_angles_to_quats(
        np.array([0.0, 0.0, yaw_deg]), degrees=True)
    cam.set_world_pose(position=np.array(pos), orientation=ori)


def apply_camera_jitter(base_pos, base_yaw_deg):
    """Gentle yaw + z bob. Door stays framed; adds movement to the video."""
    dz = random.uniform(-Z_JITTER_M, Z_JITTER_M)
    dyaw = random.uniform(-YAW_JITTER_DEG, YAW_JITTER_DEG)
    jittered_pos = [base_pos[0], base_pos[1], base_pos[2] + dz]
    set_camera(jittered_pos, base_yaw_deg + dyaw)


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
# Material prim factories (compiled once, bindings swapped per frame)
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
            print(f"[WARN] texture '{preset['texture']}' missing — skipping {preset['name']}")
            continue
        _door_mat_prims[preset["name"]] = _make_door_omnipbr(preset, tex_path=tex)
    else:
        raise ValueError(f"unexpected technique: {tech}")
print(f"[VarVid] Pre-created {len(_door_mat_prims)} door material prims")

_wall_mat_prims = {p["name"]: _make_wall_omnipbr(p) for p in WALL_COLORS}
print(f"[VarVid] Pre-created {len(_wall_mat_prims)} wall material prims")

# ---------------------------------------------------------------------------
# Binding helpers
# ---------------------------------------------------------------------------

_strong = UsdShade.Tokens.strongerThanDescendants


def cache_door_bindings(door_prim):
    frame_cache, glass_cache = [], []
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
    for p in Usd.PrimRange(door_prim):
        if not p.IsA(UsdGeom.Mesh):
            continue
        UsdShade.MaterialBindingAPI.Apply(p)
        UsdShade.MaterialBindingAPI(p).Bind(mat, bindingStrength=_strong)


_BUILDING_ROOT = "/World/Building/_22_129___MSOE_Engineering_Building___A___2_25_2026"
_WALLS_ROOT_PATH = f"{_BUILDING_ROOT}/Geometry/Walls"
_WALL_SEARCH_RADIUS_CM = 500.0
_WALL_FLOOR_BAND_CM    = 400.0


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
    return [(p, list(UsdShade.MaterialBindingAPI(p).GetDirectBindingRel().GetTargets())
             if UsdShade.MaterialBindingAPI(p).GetDirectBindingRel() else [])
            for p in wall_meshes]


def apply_wall_color(wall_meshes, preset):
    mat = _wall_mat_prims[preset["name"]]
    for p in wall_meshes:
        UsdShade.MaterialBindingAPI.Apply(p)
        UsdShade.MaterialBindingAPI(p).Bind(mat, bindingStrength=_strong)


# ---------------------------------------------------------------------------
# Capture — numbered frames for frames_to_video.py
# ---------------------------------------------------------------------------

_frame_counter = [0]


def capture() -> bool:
    world.step(render=True)
    data = ann.get_data()
    if data is None:
        return False
    arr = np.asarray(data)
    if arr.ndim != 3 or arr.shape[-1] < 3:
        return False
    rgb = arr[:, :, :3].astype(np.uint8)
    idx = _frame_counter[0]
    _frame_counter[0] += 1
    out_path = os.path.join(OUT_DIR, f"{idx:04d}.jpg")
    if _pil_ok:
        _PIL.fromarray(rgb).save(out_path, "JPEG", quality=94)
    else:
        np.save(out_path.replace(".jpg", ".npy"), rgb)
    return True


# ---------------------------------------------------------------------------
# Per-door scene loader — returns everything Acts 1/2/3 need
# ---------------------------------------------------------------------------

def load_door(door):
    door_prim = stage.GetPrimAtPath(door["prim_path"])
    if not door_prim.IsValid():
        return None
    if door_prim.IsInstance() or door_prim.IsInstanceable():
        door_prim.SetInstanceable(False)
        simulation_app.update()

    mesh_ops = discover_door_meshes(
        door_prim, door["mesh_count"], door["leaf_mesh_names"])
    if not mesh_ops:
        return None

    frame_cache, glass_cache = cache_door_bindings(door_prim)
    wall_meshes = discover_wall_meshes(door["door_center_m"])
    wall_cache  = cache_wall_bindings(wall_meshes)

    set_camera(door["cam_pos"], door["cam_yaw_deg"])
    reposition_lights(door["door_center_m"], door["cam_pos"][2])

    set_angle(mesh_ops,
              door["angle_sign"] * CLOSED_ANGLE,
              door["hinge_x_cm"], door["hinge_y_cm"])
    apply_lighting(LIGHTING_CONDITIONS[0])
    for _ in range(DOOR_WARMUP):
        world.step(render=True)

    return {
        "door":        door,
        "prim":        door_prim,
        "mesh_ops":    mesh_ops,
        "frame_cache": frame_cache,
        "glass_cache": glass_cache,
        "wall_meshes": wall_meshes,
        "wall_cache":  wall_cache,
    }


def unload_door(scene):
    """Restore original materials so the next door inherits a clean stage."""
    restore_bindings(scene["frame_cache"])
    restore_bindings(scene["glass_cache"])
    restore_bindings(scene["wall_cache"])
    set_angle(scene["mesh_ops"],
              scene["door"]["angle_sign"] * CLOSED_ANGLE,
              scene["door"]["hinge_x_cm"],
              scene["door"]["hinge_y_cm"])


# ---------------------------------------------------------------------------
# Frame builders
# ---------------------------------------------------------------------------

def render_variant(scene, *, lighting, door_mat, wall_mat, angle_deg):
    door = scene["door"]
    apply_lighting(lighting)
    apply_door_material_to_prim(
        scene["prim"], door_mat, scene["frame_cache"], scene["glass_cache"])
    if wall_mat is not None and scene["wall_meshes"]:
        restore_bindings(scene["wall_cache"])
        apply_wall_color(scene["wall_meshes"], wall_mat)
    set_angle(scene["mesh_ops"],
              door["angle_sign"] * angle_deg,
              door["hinge_x_cm"], door["hinge_y_cm"])
    apply_camera_jitter(door["cam_pos"], door["cam_yaw_deg"])
    for _ in range(CELL_WARMUP):
        world.step(render=True)
    return capture()


# ---------------------------------------------------------------------------
# Act runners
# ---------------------------------------------------------------------------

# Baseline combo used by Act-1 solo sweeps (the "everything else held fixed")
BASELINE_LIGHTING = LIGHTING_CONDITIONS[0]       # bright_fluorescent
BASELINE_MATERIAL = DOOR_MATERIALS[1]            # fill_white
BASELINE_WALL     = WALL_COLORS[0]               # warm_beige
BASELINE_ANGLE    = CLOSED_ANGLE


def act1_solo_sweeps(scene):
    _ts("  Act 1 — lighting sweep")
    for lc in LIGHTING_CONDITIONS:
        render_variant(scene,
                       lighting=lc,
                       door_mat=BASELINE_MATERIAL,
                       wall_mat=BASELINE_WALL,
                       angle_deg=BASELINE_ANGLE)

    _ts("  Act 1 — door material sweep")
    for mp in DOOR_MATERIALS:
        render_variant(scene,
                       lighting=BASELINE_LIGHTING,
                       door_mat=mp,
                       wall_mat=BASELINE_WALL,
                       angle_deg=BASELINE_ANGLE)

    _ts("  Act 1 — wall color sweep")
    for wc in WALL_COLORS:
        render_variant(scene,
                       lighting=BASELINE_LIGHTING,
                       door_mat=BASELINE_MATERIAL,
                       wall_mat=wc,
                       angle_deg=BASELINE_ANGLE)

    _ts("  Act 1 — door angle sweep")
    for ang in ANGLE_SWEEP:
        render_variant(scene,
                       lighting=BASELINE_LIGHTING,
                       door_mat=BASELINE_MATERIAL,
                       wall_mat=BASELINE_WALL,
                       angle_deg=ang)


def act2_combined(scene, n_frames):
    _ts(f"  Act 2 — {n_frames} combined random frames on hero door")
    for _ in range(n_frames):
        render_variant(
            scene,
            lighting = random.choice(LIGHTING_CONDITIONS),
            door_mat = random.choice(DOOR_MATERIALS),
            wall_mat = random.choice(WALL_COLORS),
            angle_deg = random.choice(ANGLE_SWEEP),
        )


def act3_all_doors(n_per_door):
    _ts(f"  Act 3 — {n_per_door} frames per door across {len(DOORS)} doors")
    for di, door in enumerate(DOORS):
        _ts(f"    [{di+1}/{len(DOORS)}] {door['family_short']:<12} {door['id']}")
        scene = load_door(door)
        if scene is None:
            print(f"      SKIP — could not load door")
            continue
        for _ in range(n_per_door):
            render_variant(
                scene,
                lighting = random.choice(LIGHTING_CONDITIONS),
                door_mat = random.choice(DOOR_MATERIALS),
                wall_mat = random.choice(WALL_COLORS),
                angle_deg = random.choice(ANGLE_SWEEP),
            )
        unload_door(scene)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

apply_lighting(LIGHTING_CONDITIONS[0])
_ts("Global lighting settle (20 steps) ...")
for _ in range(20):
    world.step(render=True)

# Locate hero door
hero_door = next((d for d in DOORS if d["id"] == HERO_DOOR_ID), None)
if hero_door is None:
    print(f"[FATAL] hero door {HERO_DOOR_ID} not found in registry")
    simulation_app.close()
    sys.exit(1)

_ts(f"=== ACT 1 + 2: hero door {hero_door['family_short']} {HERO_DOOR_ID} ===")
hero_scene = load_door(hero_door)
if hero_scene is None:
    print(f"[FATAL] failed to load hero door {HERO_DOOR_ID}")
    simulation_app.close()
    sys.exit(1)

act1_solo_sweeps(hero_scene)
act2_combined(hero_scene, ACT2_COMBINED_FRAMES)
unload_door(hero_scene)

_ts("=== ACT 3: all doors ===")
act3_all_doors(ACT3_FRAMES_PER_DOOR)

# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

manifest = {
    "resolution": list(CAM_RESOLUTION),
    "renderer":   "RTXRealTime",
    "seed":       7,
    "hero_door":  HERO_DOOR_ID,
    "angle_sweep_deg": ANGLE_SWEEP,
    "frames_rendered": _frame_counter[0],
    "structure": {
        "act1_lighting":  len(LIGHTING_CONDITIONS),
        "act1_material":  len(DOOR_MATERIALS),
        "act1_wall":      len(WALL_COLORS),
        "act1_angle":     len(ANGLE_SWEEP),
        "act2_combined":  ACT2_COMBINED_FRAMES,
        "act3_per_door":  ACT3_FRAMES_PER_DOOR,
        "act3_doors":     len(DOORS),
    },
    "baseline": {
        "lighting": BASELINE_LIGHTING["name"],
        "material": BASELINE_MATERIAL["name"],
        "wall":     BASELINE_WALL["name"],
        "angle":    BASELINE_ANGLE,
    },
}
with open(MANIFEST_PATH, "w") as f:
    json.dump(manifest, f, indent=2)
    f.flush()
    os.fsync(f.fileno())

print(f"\n{'='*65}")
_ts(f"Done — {_frame_counter[0]} frames in {OUT_DIR}/")
_ts(f"Next: python simulations/robot/frames_to_video.py "
    f"{OUT_DIR} --fps 4 -o data/variation_video.mp4")

world.stop()
simulation_app.close()

#!/usr/bin/env python3
"""
Production door-state data collection — SLURM array shard.

Collects the full 216-cell domain-randomization grid for ONE door per shard:
  6 lightings x 6 door materials x 6 angle bands = 216 cells
Each cell captures N cameras (baked from the occlusion probe).

Submit as an 8-task SLURM array (one task per training door):
  sbatch submit_production_collection.sh

Or run a single door for testing:
  SLURM_ARRAY_TASK_ID=0 ~/isaacsim/python.sh \
      simulations/robot/door_production_collection.py

See DATASET_PRODUCTION_PLAN.md for the full plan.
"""

import os
import sys
import math
import random
import functools
import tempfile
import numpy as np

print = functools.partial(print, flush=True)  # noqa: A001

# ---------------------------------------------------------------------------
# Shard ID from SLURM array
# ---------------------------------------------------------------------------

SHARD_ID = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CAM_RESOLUTION  = (1024, 768)
BATCH_SIZE      = 15          # max simultaneous render products (T4 safe)
STARTUP_WARMUP  = 30
DOOR_WARMUP     = 20
CELL_WARMUP     = 10
SETTLE_STEPS    = 5           # extra steps for sub-batches after the first
RT_SUBFRAMES    = 4           # rep.orchestrator.step rt_subframes — flushes render
                              # pipeline so every annotator slot has fresh data before
                              # get_data(). Without this, slots 0 and 4 of each sub-batch
                              # occasionally return max=0 stale buffers (e.g. Doors_12302795
                              # had 738 such frames out of 4536 in the v2 collection).
MAX_CAPTURE_ATTEMPTS = 3      # retry count when a slot returns max=0
CLOSED_MAX_DEG  = 5.0

_script_dir  = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.abspath(os.path.join(_script_dir, "..", ".."))
USD_PATH     = os.path.join(_project_dir, "MSOEBuildingV4_physics.usd")

SOLID_FAMILIES = {"Type_F", "Type_FG", "Type_N"}

# ---------------------------------------------------------------------------
# Angle bands — 3 CLOSED + 3 OPEN, boundary-weighted
# ---------------------------------------------------------------------------

ANGLE_BANDS = [
    {"idx": 0, "label": "CLOSED", "lo":  0.0, "hi":  1.5},
    {"idx": 1, "label": "CLOSED", "lo":  1.5, "hi":  3.0},
    {"idx": 2, "label": "CLOSED", "lo":  3.0, "hi":  5.0},
    {"idx": 3, "label": "OPEN",   "lo":  6.0, "hi": 15.0},
    {"idx": 4, "label": "OPEN",   "lo": 15.0, "hi": 50.0},
    {"idx": 5, "label": "OPEN",   "lo": 51.0, "hi": 90.0},
]

# ---------------------------------------------------------------------------
# Lighting conditions — 6 presets from LIGHTING_AND_MATERIALS.md
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
# Door materials — 6 uniform fill_* presets
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
# Wall colours — 6-colour random pool for C1 (solid doors only)
# ---------------------------------------------------------------------------

WALL_COLORS = [
    {"name": "warm_beige",    "color": (0.85, 0.78, 0.65), "roughness": 0.85, "metallic": 0.0},
    {"name": "cool_gray",     "color": (0.72, 0.75, 0.78), "roughness": 0.80, "metallic": 0.0},
    {"name": "terracotta",    "color": (0.70, 0.40, 0.30), "roughness": 0.85, "metallic": 0.0},
    {"name": "sage_green",    "color": (0.65, 0.75, 0.58), "roughness": 0.85, "metallic": 0.0},
    {"name": "muted_blue",    "color": (0.55, 0.65, 0.80), "roughness": 0.80, "metallic": 0.0},
    {"name": "dark_charcoal", "color": (0.30, 0.30, 0.32), "roughness": 0.85, "metallic": 0.0},
]

# ---------------------------------------------------------------------------
# Camera grid (v4)
# ---------------------------------------------------------------------------
# Full candidate grid: 5 arcs x 2 distances x 2 heights = 20 cells per door.
# Per-door `excluded_cells` comes from manual review of the
# data/camera_grid_preview/ contact sheets — cells are dropped only when
# geometry breaks the shot (wall column occludes door, camera clips inside
# geometry, stair railing crosses the view, adjacent-room leak).
_ALLOWED_ARCS    = [-30, -15, 0, 15, 30]
_ALLOWED_DISTS   = [1.5, 2.0]
_ALLOWED_HEIGHTS = [-0.5, 0.0]

# ---------------------------------------------------------------------------
# Door registry — 8 training + 2 holdout doors
# ---------------------------------------------------------------------------

_BASE = (
    "/World/Building/_22_129___MSOE_Engineering_Building___A___2_25_2026"
    "/Geometry/Doors"
)

# excluded_cells rule format: dict of field->value; all listed fields must
# match (unlisted fields match any). e.g. {"arc": -30} excludes all 4 cells
# on the -30 arc; {"arc": 30, "dist": 2.0} excludes the 2 cells at +30 / 2.0 m.
DOORS = [
    # Shard 0 — Doors_13606056 CW_F_NEW floor 3 (all 20 cells kept)
    {"id": "Doors_13606056", "family_short": "CW_F_NEW", "floor": 3, "role": "train",
     "prim_path": f"{_BASE}/RS___Curtain_Wall___Type_F___NEW/New/Doors_13606056",
     "mesh_count": 1, "hinge_x_cm": -4747.75, "hinge_y_cm": 6958.30,
     "cam_pos": [-45.477, 70.040, 19.437], "cam_yaw_deg": 180.0,
     "door_center_m": [-47.477, 70.040, 19.437], "angle_sign": 1,
     "excluded_cells": []},
    # Shard 1 — Doors_13606466 CW_F_NEW floor 4 (drop -30 arc: wall column occludes)
    {"id": "Doors_13606466", "family_short": "CW_F_NEW", "floor": 4, "role": "train",
     "prim_path": f"{_BASE}/RS___Curtain_Wall___Type_F___NEW/New/Doors_13606466",
     "mesh_count": 1, "hinge_x_cm": -4836.01, "hinge_y_cm": 6170.58,
     "cam_pos": [-46.360, 62.163, 23.704], "cam_yaw_deg": 180.0,
     "door_center_m": [-48.360, 62.163, 23.704], "angle_sign": 1,
     "excluded_cells": [{"arc": -30}]},
    # Shard 2 — Doors_8286057 Type_F floor 2 (drop +30 arc: adjacent showroom)
    {"id": "Doors_8286057", "family_short": "Type_F", "floor": 2, "role": "train",
     "prim_path": f"{_BASE}/RS___Type_F_Door/NEW___HM_Frame___36_x96/Doors_8286057",
     "mesh_count": 0, "hinge_x_cm": +50.80, "hinge_y_cm": 0.0,
     "cam_pos": [17.443, 79.341, 14.764], "cam_yaw_deg": 270.0,
     "door_center_m": [17.443, 77.341, 14.764], "angle_sign": 1,
     "excluded_cells": [{"arc": 30}]},
    # Shard 3 — Doors_3135827 CW_FG_New floor 3 (all 20 cells kept)
    {"id": "Doors_3135827", "family_short": "CW_FG_New", "floor": 3, "role": "train",
     "prim_path": f"{_BASE}/RS___Curtain_Wall___Type_FG___New/New/Doors_3135827",
     "mesh_count": 2, "hinge_x_cm": 1724.33, "hinge_y_cm": 5878.48,
     "cam_pos": [17.701, 56.807, 19.437], "cam_yaw_deg": 90.0,
     "door_center_m": [17.701, 58.807, 19.437], "angle_sign": 1,
     "excluded_cells": []},
    # Shard 4 — Doors_10649156 Type_FG floor 3 (drop -30 arc: points at blank wall)
    {"id": "Doors_10649156", "family_short": "Type_FG", "floor": 3, "role": "train",
     "prim_path": f"{_BASE}/RS___Type_FG_Door/NEW___HM_Frame___36_x96/Doors_10649156",
     "mesh_count": 0, "hinge_x_cm": +50.80, "hinge_y_cm": 0.0,
     "cam_pos": [-14.295, 80.200, 19.437], "cam_yaw_deg": 0.0,
     "door_center_m": [-12.295, 80.200, 19.437], "angle_sign": -1,
     "excluded_cells": [{"arc": -30}]},
    # Shard 5 — Doors_7171063 Type_F floor 5 OUTWARD (drop +30 arc: camera clips geometry)
    {"id": "Doors_7171063", "family_short": "Type_F", "floor": 5, "role": "train",
     "prim_path": f"{_BASE}/RS___Type_F_Door/NEW___HM_Frame___42_x96/Doors_7171063",
     "mesh_count": 0, "hinge_x_cm": +58.42, "hinge_y_cm": 0.0,
     "cam_pos": [-49.316, 49.892, 27.972], "cam_yaw_deg": 0.0,
     "door_center_m": [-47.316, 49.892, 27.972], "angle_sign": -1,
     "excluded_cells": [{"arc": 30}]},
    # Shard 6 — Doors_12302795 CW_FG_New floor 2 (drop +30 @ 2.0m: adjacent wall)
    {"id": "Doors_12302795", "family_short": "CW_FG_New", "floor": 2, "role": "train",
     "prim_path": f"{_BASE}/RS___Curtain_Wall___Type_FG___New/New___GL1_1/Doors_12302795",
     "mesh_count": 2, "hinge_x_cm": -5925.83, "hinge_y_cm": 5774.34,
     "cam_pos": [-57.258, 58.251, 14.561], "cam_yaw_deg": 180.0,
     "door_center_m": [-59.258, 58.251, 14.561], "angle_sign": 1,
     "excluded_cells": [{"arc": 30, "dist": 2.0}]},
    # Shard 7 — Doors_12590968 Type_N floor 3 (drop +/-30 @ 2.0m: stair railing)
    {"id": "Doors_12590968", "family_short": "Type_N", "floor": 3, "role": "train",
     "prim_path": f"{_BASE}/RS___Type_N_Door/NEW___HM_Frame___36_x96/Doors_12590968",
     "mesh_count": 0, "hinge_x_cm": +50.80, "hinge_y_cm": 0.0,
     "cam_pos": [17.878, 50.941, 19.437], "cam_yaw_deg": 90.0,
     "door_center_m": [17.878, 52.941, 19.437], "angle_sign": 1,
     "excluded_cells": [{"arc": -30, "dist": 2.0}, {"arc": 30, "dist": 2.0}]},
    # Shard 8 — Doors_13182085 CW_FG_New floor 4 HOLDOUT (all 20 cells kept)
    {"id": "Doors_13182085", "family_short": "CW_FG_New", "floor": 4, "role": "holdout",
     "prim_path": f"{_BASE}/RS___Curtain_Wall___Type_FG___New/New/Doors_13182085",
     "mesh_count": 2, "hinge_x_cm": -4142.56, "hinge_y_cm": 8214.32,
     "cam_pos": [-39.426, 82.600, 23.704], "cam_yaw_deg": 180.0,
     "door_center_m": [-41.426, 82.600, 23.704], "angle_sign": 1,
     "excluded_cells": []},
    # Shard 9 — Doors_1915613 Type_F floor 5 HOLDOUT (drop +30 arc: camera clips geometry)
    {"id": "Doors_1915613", "family_short": "Type_F", "floor": 5, "role": "holdout",
     "prim_path": f"{_BASE}/RS___Type_F_Door/NEW___HM_Frame___36_x96/Doors_1915613",
     "mesh_count": 0, "hinge_x_cm": +50.80, "hinge_y_cm": 0.0,
     "cam_pos": [-58.753, 65.219, 27.972], "cam_yaw_deg": 270.0,
     "door_center_m": [-58.753, 63.219, 27.972], "angle_sign": 1,
     "excluded_cells": [{"arc": 30}]},
]

def _cell_excluded(door, cell):
    for rule in door.get("excluded_cells", []):
        if all(cell[k] == v for k, v in rule.items()):
            return True
    return False

# Build each door's clear_grid from (arcs x dists x heights) minus its exclusions.
for _d in DOORS:
    _d["clear_grid"] = [
        {"arc": a, "dist": d, "height": h}
        for a in _ALLOWED_ARCS
        for d in _ALLOWED_DISTS
        for h in _ALLOWED_HEIGHTS
        if not _cell_excluded(_d, {"arc": a, "dist": d, "height": h})
    ]

# Select this shard's door
if SHARD_ID >= len(DOORS):
    print(f"ERROR: SHARD_ID={SHARD_ID} out of range (max {len(DOORS)-1})")
    sys.exit(1)

DOOR = DOORS[SHARD_ID]
N_CAMS = len(DOOR["clear_grid"])
N_CELLS = len(LIGHTING_CONDITIONS) * len(DOOR_MATERIALS) * len(ANGLE_BANDS)
N_FRAMES = N_CELLS * N_CAMS

OUT_PATH = os.path.join(
    _project_dir, "data",
    f"door_production_v4_{DOOR['role']}_{DOOR['id']}.npz",
)

# Deterministic seeding per shard
_SEED = 42 + SHARD_ID
rng = np.random.default_rng(_SEED)
random.seed(_SEED)

print(f"[Prod] Shard {SHARD_ID}: {DOOR['id']} ({DOOR['family_short']}) "
      f"— {N_CAMS} cameras x {N_CELLS} cells = {N_FRAMES} frames")

# ---------------------------------------------------------------------------
# Isaac Sim startup
# ---------------------------------------------------------------------------

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

import time as _time
_t0 = _time.time()
def _ts(msg):
    print(f"[Prod] [+{_time.time() - _t0:6.1f}s] {msg}")

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

# ---------------------------------------------------------------------------
# Procedural textures
# ---------------------------------------------------------------------------

def _generate_procedural_textures(scratch_dir):
    textures = {}
    if not _pil_ok:
        return textures
    os.makedirs(scratch_dir, exist_ok=True)
    h = w = 512
    _rng = np.random.default_rng(42)

    def _save(name, gray):
        rgb_arr = np.stack([gray, gray, gray], axis=-1)
        rgb_arr = (np.clip(rgb_arr, 0, 1) * 255).astype(np.uint8)
        p = os.path.join(scratch_dir, f"{name}.png")
        _PIL.fromarray(rgb_arr).save(p)
        textures[name] = p

    base = np.full((h, w), 0.88, dtype=np.float32)
    base += _rng.uniform(-0.04, 0.04, (h, 1)).astype(np.float32)
    base += _rng.uniform(-0.01, 0.01, (h, w)).astype(np.float32)
    _save("brushed", base)

    xs = np.arange(w, dtype=np.float32)
    band = 0.5 * (np.sin(xs * 0.04) + np.sin(xs * 0.11 + 1.7))
    base = 0.88 + 0.05 * band[None, :]
    base = np.broadcast_to(base, (h, w)).copy()
    base += _rng.uniform(-0.02, 0.02, (h, w)).astype(np.float32)
    _save("grain", base)

    base = np.full((h, w), 0.88, dtype=np.float32)
    base += _rng.uniform(-0.03, 0.03, (h, w)).astype(np.float32)
    _save("speckle", base)

    return textures

_tex_dir = os.path.join(_project_dir, "data", f"production_textures_v4_{DOOR['id']}")
_procedural_textures = _generate_procedural_textures(_tex_dir)

# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------

world = World(stage_units_in_meters=1.0)
stage = omni.usd.get_context().get_stage()
_ts(f"Inserting building USD sublayer: {USD_PATH}")
stage.GetRootLayer().subLayerPaths.insert(0, USD_PATH)

# Dome light
dome = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/DomeLight"))
dome.CreateIntensityAttr(25_000)
dome.GetColorAttr().Set(Gf.Vec3f(1.0, 1.0, 1.0))

# 4-slot ceiling array
_CEILING_SPREAD = 1.2
_CEILING_RISE   = 1.4
_CEILING_OFFSETS = [
    (-_CEILING_SPREAD, -_CEILING_SPREAD), (+_CEILING_SPREAD, -_CEILING_SPREAD),
    (-_CEILING_SPREAD, +_CEILING_SPREAD), (+_CEILING_SPREAD, +_CEILING_SPREAD),
]
ceiling_int_attrs = []
ceiling_col_attrs = []
cx, cy = DOOR["door_center_m"][0], DOOR["door_center_m"][1]
cz = DOOR["cam_pos"][2] + _CEILING_RISE
for i, (dx, dy) in enumerate(_CEILING_OFFSETS):
    rl = UsdLux.RectLight.Define(stage, Sdf.Path(f"/World/CeilingLight_{i}"))
    rl.CreateIntensityAttr(0.0)
    rl.CreateWidthAttr(0.6)
    rl.CreateHeightAttr(0.6)
    xf = UsdGeom.Xformable(rl.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(cx + dx, cy + dy, cz))
    xf.AddRotateXYZOp().Set(Gf.Vec3f(180.0, 0.0, 0.0))
    ceiling_int_attrs.append(rl.GetIntensityAttr())
    ceiling_col_attrs.append(rl.GetColorAttr())

# ---------------------------------------------------------------------------
# Material prim factories
# ---------------------------------------------------------------------------

def _make_omnipbr(name_prefix, preset, tex_path=None):
    mp = f"/World/Looks/{name_prefix}_{preset['name']}"
    mat = UsdShade.Material.Define(stage, Sdf.Path(mp))
    sh = UsdShade.Shader.Define(stage, Sdf.Path(mp + "/Shader"))
    sh.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
    sh.SetSourceAsset(Sdf.AssetPath("OmniPBR.mdl"), "mdl")
    sh.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
    if tex_path is not None:
        sh.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(tex_path))
        sh.CreateInput("project_uvw", Sdf.ValueTypeNames.Bool).Set(True)
        sh.CreateInput("world_or_object", Sdf.ValueTypeNames.Bool).Set(False)
        sh.CreateInput("texture_scale", Sdf.ValueTypeNames.Float2).Set(Gf.Vec2f(2.0, 2.0))
        sh.CreateInput("diffuse_tint", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*preset["tint"]))
    else:
        sh.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*preset["color"]))
    sh.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(float(preset["roughness"]))
    sh.CreateInput("metallic_constant", Sdf.ValueTypeNames.Float).Set(float(preset["metallic"]))
    mat.CreateSurfaceOutput("mdl").ConnectToSource(sh.ConnectableAPI(), "out")
    mat.CreateDisplacementOutput("mdl").ConnectToSource(sh.ConnectableAPI(), "out")
    mat.CreateVolumeOutput("mdl").ConnectToSource(sh.ConnectableAPI(), "out")
    return mat

_door_mat_prims = {}
for p in DOOR_MATERIALS:
    t = p["technique"]
    if t == "fill_glass":
        _door_mat_prims[p["name"]] = _make_omnipbr("DoorMat", p)
    elif t == "fill_textured":
        tex = _procedural_textures.get(p["texture"])
        if tex:
            _door_mat_prims[p["name"]] = _make_omnipbr("DoorMat", p, tex_path=tex)

_wall_mat_prims = {w["name"]: _make_omnipbr("WallMat", w) for w in WALL_COLORS}
_ts(f"Pre-created {len(_door_mat_prims)} door + {len(_wall_mat_prims)} wall material prims")

# ---------------------------------------------------------------------------
# World reset + camera creation
# ---------------------------------------------------------------------------

_ts("world.reset() ...")
world.reset()
_ts("world.reset() done")
simulation_app.update()
for _ in range(5):
    world.step(render=True)

# Pre-compute nominal camera configs for ALL clear_grid positions
DOOR_CENTER_XY = np.array([DOOR["door_center_m"][0], DOOR["door_center_m"][1]])
CAM_Z_BASE = DOOR["cam_pos"][2]
# Rotate arc so 0° = "in front of door" (opposite the face direction)
_ARC_BASE_RAD = math.radians(DOOR["cam_yaw_deg"] + 180.0)

cam_configs = []
for ci, g in enumerate(DOOR["clear_grid"]):
    theta = math.radians(g["arc"]) + _ARC_BASE_RAD
    cam_z = CAM_Z_BASE + g["height"]
    cam_pos = np.array([
        DOOR_CENTER_XY[0] + g["dist"] * math.cos(theta),
        DOOR_CENTER_XY[1] + g["dist"] * math.sin(theta),
        cam_z,
    ])
    yaw_deg = math.degrees(theta) + 180.0
    cam_ori = rot_utils.euler_angles_to_quats(
        np.array([0.0, 0.0, yaw_deg]), degrees=True)
    cam_configs.append({
        "pos": cam_pos, "ori": cam_ori,
        "offset_deg": g["arc"], "distance": g["dist"], "height_off": g["height"],
        "world_yaw_deg": yaw_deg,
    })

# Create only BATCH_SIZE reusable camera prims + render products to stay
# within T4 16 GB VRAM.  Cameras are repositioned per sub-batch in the
# main loop.
_N_SLOTS = min(BATCH_SIZE, N_CAMS)
cameras = []
render_products = []
rgb_annotators = []

for si in range(_N_SLOTS):
    cfg0 = cam_configs[si]
    prim_name = f"/World/ProdCam_{si}"
    cam_obj = Camera(
        prim_path=prim_name, position=cfg0["pos"],
        frequency=30, resolution=CAM_RESOLUTION, orientation=cfg0["ori"])
    cameras.append(cam_obj)

simulation_app.update()
for si, cam_obj in enumerate(cameras):
    cam_obj.initialize()
    prim_name = f"/World/ProdCam_{si}"
    usd_cam = UsdGeom.Camera(stage.GetPrimAtPath(prim_name))
    usd_cam.GetFocalLengthAttr().Set(5.0)
    usd_cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 10000.0))
    rp_h = rep.create.render_product(prim_name, CAM_RESOLUTION)
    ann_h = rep.AnnotatorRegistry.get_annotator("rgb")
    ann_h.attach([rp_h])
    render_products.append(rp_h)
    rgb_annotators.append(ann_h)

_n_batches = math.ceil(N_CAMS / _N_SLOTS)
_ts(f"{N_CAMS} cameras → {_N_SLOTS} render slots, {_n_batches} sub-batches per cell")

_ts(f"Renderer warmup ({STARTUP_WARMUP} steps) ...")
for _ in range(STARTUP_WARMUP):
    world.step(render=True)
_ts("Warmup done")

# ---------------------------------------------------------------------------
# Door mesh discovery
# ---------------------------------------------------------------------------

door_prim = stage.GetPrimAtPath(DOOR["prim_path"])
if not door_prim.IsValid():
    print(f"ERROR: door prim not found: {DOOR['prim_path']}")
    simulation_app.close()
    sys.exit(1)

if door_prim.IsInstance() or door_prim.IsInstanceable():
    door_prim.SetInstanceable(False)
    simulation_app.update()

def _is_glass_mesh(p) -> bool:
    n = p.GetPath().name.lower()
    return n.startswith("glass_") or "_gl_" in n or "glazing" in n or "pane_" in n

_mesh_ops = []
prim_iter = (door_prim.GetChildren() if DOOR["mesh_count"] > 0
             else Usd.PrimRange(door_prim))
for p in prim_iter:
    if not p.IsA(UsdGeom.Mesh):
        continue
    xf = UsdGeom.Xformable(p)
    op = next((o for o in xf.GetOrderedXformOps()
               if o.GetOpName() == "xformOp:transform"), None)
    if op is None:
        continue
    mat = op.Get()
    if mat is None:
        continue
    _mesh_ops.append(({"ox": mat[3][0], "oy": mat[3][1], "oz": mat[3][2]}, op))

if not _mesh_ops:
    print(f"ERROR: no xformOp:transform meshes under {DOOR['prim_path']}")
    simulation_app.close()
    sys.exit(1)
_ts(f"{len(_mesh_ops)} door mesh(es) discovered")

# Cache original bindings
_strong = UsdShade.Tokens.strongerThanDescendants
_orig_frame_bindings = []
_orig_glass_bindings = []
for p in Usd.PrimRange(door_prim):
    if not p.IsA(UsdGeom.Mesh):
        continue
    rel = UsdShade.MaterialBindingAPI(p).GetDirectBindingRel()
    targets = list(rel.GetTargets()) if rel else []
    (_orig_glass_bindings if _is_glass_mesh(p) else _orig_frame_bindings).append((p, targets))
_ts(f"Cached bindings: {len(_orig_frame_bindings)} frame, {len(_orig_glass_bindings)} glass")

# ---------------------------------------------------------------------------
# Wall discovery (solid families only)
# ---------------------------------------------------------------------------

_WALLS_ROOT = "/World/Building/_22_129___MSOE_Engineering_Building___A___2_25_2026/Geometry/Walls"
_WALL_RADIUS_CM = 500.0
_WALL_Z_BAND_CM = 400.0

is_solid = DOOR["family_short"] in SOLID_FAMILIES
wall_meshes = []
_orig_wall_bindings = []

if is_solid:
    walls_root = stage.GetPrimAtPath(_WALLS_ROOT)
    if walls_root.IsValid():
        dcx = DOOR["door_center_m"][0] * 100.0
        dcy = DOOR["door_center_m"][1] * 100.0
        dcz = DOOR["door_center_m"][2] * 100.0
        for p in Usd.PrimRange(walls_root):
            if not p.IsA(UsdGeom.Mesh):
                continue
            xf = UsdGeom.Xformable(p)
            op = next((o for o in xf.GetOrderedXformOps()
                       if o.GetOpName() == "xformOp:transform"), None)
            if op is None:
                continue
            m = op.Get()
            if m is None:
                continue
            tx, ty, tz = float(m[3][0]), float(m[3][1]), float(m[3][2])
            if abs(tz - dcz) > _WALL_Z_BAND_CM:
                continue
            if math.hypot(tx - dcx, ty - dcy) > _WALL_RADIUS_CM:
                continue
            wall_meshes.append(p)
        for p in wall_meshes:
            rel = UsdShade.MaterialBindingAPI(p).GetDirectBindingRel()
            _orig_wall_bindings.append((p, list(rel.GetTargets()) if rel else []))
    _ts(f"Wall meshes: {len(wall_meshes)} (solid door, C1 active)")
else:
    _ts("Curtain-wall door — C1 skipped")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _restore_bindings(cache):
    for mesh_prim, orig_targets in cache:
        api = UsdShade.MaterialBindingAPI(mesh_prim)
        api.UnbindAllBindings()
        if not orig_targets:
            continue
        rel = api.GetDirectBindingRel() or api.CreateDirectBindingRel()
        for tgt in orig_targets:
            rel.AddTarget(tgt)

def _kelvin_to_rgb(k):
    t = k / 100.0
    r = 255 if t <= 66 else max(0, min(255, 329.698727446 * ((t - 60) ** -0.1332047592)))
    g = (99.4708025861 * math.log(t) - 161.1195681661 if t <= 66
         else 288.1221695283 * ((t - 60) ** -0.0755148492))
    g = max(0, min(255, g))
    b = 255 if t >= 66 else (0 if t <= 19 else
        max(0, min(255, 138.5177312231 * math.log(t - 10) - 305.0447927307)))
    return Gf.Vec3f(r / 255.0, g / 255.0, b / 255.0)

def apply_lighting(cond):
    ji = lambda v: v * random.uniform(0.90, 1.10)
    jt = lambda v: v + random.uniform(-200, 200)
    dome.GetIntensityAttr().Set(ji(cond["dome_int"]))
    dome.GetColorAttr().Set(_kelvin_to_rgb(jt(cond["dome_temp"])))
    ci = ji(cond["ceiling_int"])
    ct = jt(cond["ceiling_temp"])
    rgb = _kelvin_to_rgb(ct)
    for slot, (ia, ca) in enumerate(zip(ceiling_int_attrs, ceiling_col_attrs)):
        ia.Set(ci if cond["ceiling_pattern"][slot] else 0.0)
        ca.Set(rgb)

def apply_door_material(preset):
    _restore_bindings(_orig_frame_bindings)
    _restore_bindings(_orig_glass_bindings)
    mat = _door_mat_prims.get(preset["name"])
    if mat is None:
        return
    for p in Usd.PrimRange(door_prim):
        if p.IsA(UsdGeom.Mesh):
            UsdShade.MaterialBindingAPI.Apply(p)
            UsdShade.MaterialBindingAPI(p).Bind(mat, bindingStrength=_strong)

def apply_wall_color(wc):
    _restore_bindings(_orig_wall_bindings)
    mat = _wall_mat_prims[wc["name"]]
    for p in wall_meshes:
        UsdShade.MaterialBindingAPI.Apply(p)
        UsdShade.MaterialBindingAPI(p).Bind(mat, bindingStrength=_strong)

def _door_matrix(ox, oy, oz, angle_deg):
    c, s = math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg))
    dx, dy = ox - DOOR["hinge_x_cm"], oy - DOOR["hinge_y_cm"]
    return Gf.Matrix4d(
         c,  s, 0, 0,
        -s,  c, 0, 0,
         0,  0, 1, 0,
        c*dx - s*dy + DOOR["hinge_x_cm"],
        s*dx + c*dy + DOOR["hinge_y_cm"],
        oz, 1)

def set_door_angle(angle_deg):
    signed = DOOR["angle_sign"] * angle_deg
    for m, op in _mesh_ops:
        op.Set(_door_matrix(m["ox"], m["oy"], m["oz"], signed))

def _position_cameras_for_batch(batch_cfgs, jitters):
    """Reposition the reusable camera slots to a sub-batch of configs.

    batch_cfgs: list of cam_config dicts (len <= _N_SLOTS)
    jitters:    list of (dx, dy, dz, pitch, roll) tuples, same length
    """
    for si, (cfg, jt) in enumerate(zip(batch_cfgs, jitters)):
        jpos = np.array([
            cfg["pos"][0] + jt[0],
            cfg["pos"][1] + jt[1],
            cfg["pos"][2] + jt[2],
        ])
        base_yaw = cfg["world_yaw_deg"]
        jori = rot_utils.euler_angles_to_quats(
            np.array([jt[3], jt[4], base_yaw]), degrees=True)
        try:
            cameras[si].set_world_pose(position=jpos, orientation=jori)
        except Exception:
            pass

# ---------------------------------------------------------------------------
# Collection buffers — flushed to disk per lighting condition to cap RAM
# ---------------------------------------------------------------------------

all_frames = []
all_labels = []
all_angles = []
all_episodes = []
all_cam_offsets = []
all_light_idx = []
all_material_idx = []
all_angle_band = []
all_wall_color_idx = []

_chunk_dir = tempfile.mkdtemp(prefix="prod_chunks_")
_chunk_paths = []
_total_flushed = 0
_n_dropped = [0]   # list so the capture loop can mutate it (Python closure quirk)

def _flush_chunk(chunk_label):
    """Write current buffers to a numbered NPZ chunk and free memory."""
    global _total_flushed
    if not all_frames:
        return
    n = len(all_frames)
    path = os.path.join(_chunk_dir, f"chunk_{len(_chunk_paths):02d}.npz")
    _ts(f"Flushing {n} frames to {path} ({chunk_label})")
    np.savez_compressed(
        path,
        frames         = np.array(all_frames,         dtype=np.uint8),
        labels         = np.array(all_labels,         dtype=np.float32),
        angles         = np.array(all_angles,         dtype=np.float32),
        episode        = np.array(all_episodes,       dtype=np.int32),
        cam_offsets    = np.array(all_cam_offsets,    dtype=np.int32),
        light_idx      = np.array(all_light_idx,      dtype=np.int32),
        material_idx   = np.array(all_material_idx,   dtype=np.int32),
        angle_band     = np.array(all_angle_band,     dtype=np.int32),
        wall_color_idx = np.array(all_wall_color_idx, dtype=np.int32),
    )
    _chunk_paths.append(path)
    _total_flushed += n
    all_frames.clear()
    all_labels.clear()
    all_angles.clear()
    all_episodes.clear()
    all_cam_offsets.clear()
    all_light_idx.clear()
    all_material_idx.clear()
    all_angle_band.clear()
    all_wall_color_idx.clear()

# ---------------------------------------------------------------------------
# Prime renderer
# ---------------------------------------------------------------------------

apply_lighting(LIGHTING_CONDITIONS[0])
set_door_angle(0.0)
_ts(f"Door warmup ({DOOR_WARMUP} steps) ...")
for _ in range(DOOR_WARMUP):
    world.step(render=True)
_ts("Warmup done — starting collection")

# ---------------------------------------------------------------------------
# Main 216-cell loop: lighting x material x angle_band
# ---------------------------------------------------------------------------

cell = 0
total_cells = N_CELLS

for li, lc in enumerate(LIGHTING_CONDITIONS):
    for mi, mat_preset in enumerate(DOOR_MATERIALS):
        for bi, band in enumerate(ANGLE_BANDS):
            # Randomize within cell
            angle_deg = random.uniform(band["lo"], band["hi"])
            wall_idx = -1

            # Pre-generate ALL N_CAMS jitter values now so the random
            # stream is consumed identically regardless of BATCH_SIZE.
            cell_jitters = [
                (random.uniform(-0.15, 0.15),   # dx
                 random.uniform(-0.15, 0.15),   # dy
                 random.uniform(-0.10, 0.10),   # dz
                 random.uniform(-1.5, 1.5),     # pitch
                 random.uniform(-2.0, 2.0))     # roll
                for _ in range(N_CAMS)
            ]

            apply_lighting(lc)
            apply_door_material(mat_preset)
            set_door_angle(angle_deg)

            if is_solid and wall_meshes:
                wall_idx = random.randint(0, len(WALL_COLORS) - 1)
                apply_wall_color(WALL_COLORS[wall_idx])

            # --- Sub-batch loop over cameras ---
            captured = 0
            for batch_start in range(0, N_CAMS, _N_SLOTS):
                batch_end = min(batch_start + _N_SLOTS, N_CAMS)
                batch_cfgs = cam_configs[batch_start:batch_end]
                batch_jitters = cell_jitters[batch_start:batch_end]
                n_active = batch_end - batch_start

                _position_cameras_for_batch(batch_cfgs, batch_jitters)
                simulation_app.update()   # propagate USD xform updates to renderer

                # Full warmup for first sub-batch, short settle for subsequent
                warmup = CELL_WARMUP if batch_start == 0 else SETTLE_STEPS
                for _ in range(warmup):
                    world.step(render=True)

                # Flush Replicator render pipeline so every annotator slot gets fresh data.
                # world.step(render=True) alone is not enough — on multi-sub-batch cells
                # slots 0 and 4 sometimes return stale/zero buffers.
                rep.orchestrator.step(rt_subframes=RT_SUBFRAMES)

                # Capture with validate-and-retry: if any slot came back max==0, step
                # again and re-read only the bad slots. Frames that never pass are
                # dropped (counted in _n_dropped) rather than written as black.
                batch_frames = [None] * n_active
                for attempt in range(MAX_CAPTURE_ATTEMPTS):
                    need_retry = False
                    for ai in range(n_active):
                        if batch_frames[ai] is not None:
                            continue
                        data = rgb_annotators[ai].get_data()
                        if data is None:
                            need_retry = True
                            continue
                        arr = np.asarray(data)
                        if arr.ndim != 3 or arr.shape[-1] < 3:
                            need_retry = True
                            continue
                        arr3 = arr[:, :, :3]
                        if int(arr3.max()) == 0:
                            need_retry = True
                            continue
                        batch_frames[ai] = arr3.astype(np.uint8)
                    if not need_retry:
                        break
                    for _ in range(3):
                        world.step(render=True)
                    rep.orchestrator.step(rt_subframes=RT_SUBFRAMES)

                for ai in range(n_active):
                    frame = batch_frames[ai]
                    if frame is None:
                        _n_dropped[0] += 1
                        continue
                    cfg = batch_cfgs[ai]
                    all_frames.append(frame)
                    all_labels.append(angle_deg / 90.0)
                    all_angles.append(angle_deg)
                    all_episodes.append(cell)
                    all_cam_offsets.append(int(cfg["offset_deg"]))
                    all_light_idx.append(li)
                    all_material_idx.append(mi)
                    all_angle_band.append(bi)
                    all_wall_color_idx.append(wall_idx)
                    captured += 1

            cell += 1
            if cell % 36 == 0 or cell == total_cells:
                elapsed = _time.time() - _t0
                done = _total_flushed + len(all_frames)
                rate = done / max(elapsed, 1)
                eta = (N_FRAMES - done) / max(rate, 0.01)
                _ts(f"Cell {cell:4d}/{total_cells}  "
                    f"frames={done:6d}/{N_FRAMES}  "
                    f"rate={rate:.1f} f/s  ETA={eta/60:.0f}m  "
                    f"L={lc['name'][:12]:<12s} M={mat_preset['name'][:16]:<16s} "
                    f"band={bi} angle={angle_deg:5.1f}")

            # Reset door to closed between cells
            set_door_angle(0.0)

    # Flush after each lighting condition to cap RAM usage
    _flush_chunk(f"lighting={lc['name']}")

# Restore originals
_restore_bindings(_orig_frame_bindings)
_restore_bindings(_orig_glass_bindings)
if is_solid:
    _restore_bindings(_orig_wall_bindings)

# ---------------------------------------------------------------------------
# Merge chunks into final NPZ — loads one chunk at a time to stay lean
# ---------------------------------------------------------------------------

data_dir = os.path.dirname(OUT_PATH)
os.makedirs(data_dir, exist_ok=True)

_ts(f"Merging {len(_chunk_paths)} chunks ({_total_flushed} frames) -> {OUT_PATH}")

_ARRAY_KEYS = ["frames", "labels", "angles", "episode", "cam_offsets",
               "light_idx", "material_idx", "angle_band", "wall_color_idx"]

merged = {k: [] for k in _ARRAY_KEYS}
for cp in _chunk_paths:
    with np.load(cp) as chunk:
        for k in _ARRAY_KEYS:
            merged[k].append(chunk[k])

merged_arrays = {k: np.concatenate(merged[k]) for k in _ARRAY_KEYS}
del merged  # free list-of-arrays before compressing

_ts(f"Saving compressed NPZ ({merged_arrays['frames'].shape[0]} frames) ...")
np.savez_compressed(
    OUT_PATH,
    **merged_arrays,
    door_id        = np.array(DOOR["id"]),
    light_names    = np.array([c["name"] for c in LIGHTING_CONDITIONS]),
    material_names = np.array([m["name"] for m in DOOR_MATERIALS]),
    shard_id       = np.array(SHARD_ID),
    seed           = np.array(_SEED),
)

# Force NFS write-through
with open(OUT_PATH, "rb") as f:
    pass
size_mb = os.path.getsize(OUT_PATH) // (1024 * 1024)

# Clean up temp chunks
import shutil
shutil.rmtree(_chunk_dir, ignore_errors=True)

angle_arr = merged_arrays["angles"]
n_closed = int((angle_arr <= CLOSED_MAX_DEG).sum())
n_open = int((angle_arr > CLOSED_MAX_DEG).sum())
n_total = len(angle_arr)

print()
_ts(f"DONE — {n_total} frames saved ({size_mb} MB)")
_ts(f"  CLOSED: {n_closed}  OPEN: {n_open}  "
    f"balance: {n_closed/(n_closed+n_open)*100:.1f}%/{n_open/(n_closed+n_open)*100:.1f}%")
_ts(f"  Door: {DOOR['id']} ({DOOR['family_short']})  Shard: {SHARD_ID}  Seed: {_SEED}")
_ts(f"  Dropped black frames after {MAX_CAPTURE_ATTEMPTS} retries: {_n_dropped[0]}")
_ts(f"  Output: {OUT_PATH}")

world.stop()
simulation_app.close()

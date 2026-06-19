#!/usr/bin/env python3
"""Visualize the full candidate camera grid for every training door.

For each of the 10 doors in the training roster, captures one frame from
each of the 12 camera positions (3 arcs x 2 distances x 2 heights). The
door is kept closed (0 deg) and materials/lighting are neutral — the goal
is a geometric coverage check, not a domain-randomization sample.

Outputs under data/camera_grid_preview/:
  <door_id>/arc{arc:+d}_dist{dist}_h{h:+.1f}.jpg   (12 JPEGs per door)
  <door_id>_sheet.jpg                              (4x3 labeled contact sheet)

SLURM: teaching partition, ~30 min, ~/isaacsim/python.sh.
"""

import os
import sys
import math
import functools
import numpy as np

print = functools.partial(print, flush=True)  # noqa: A001

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CAM_RESOLUTION = (640, 480)
STARTUP_WARMUP = 30
DOOR_WARMUP    = 10
CELL_WARMUP    = 6
RT_SUBFRAMES   = 4
MAX_CAPTURE_ATTEMPTS = 3

_ALLOWED_ARCS    = [-30, -15, 0, 15, 30]
_ALLOWED_DISTS   = [1.5, 2.0]
_ALLOWED_HEIGHTS = [-0.5, 0.0]

_script_dir  = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.abspath(os.path.join(_script_dir, "..", ".."))
USD_PATH     = os.path.join(_project_dir, "MSOEBuildingV4_physics.usd")
OUT_ROOT     = os.path.join(_project_dir, "data", "camera_grid_preview")

# ---------------------------------------------------------------------------
# Door registry — all 10 doors from SELECTED_TRAINING_DOORS.md
# (8 train + 2 holdout), copied from lighting_test_all_doors.py.
# ---------------------------------------------------------------------------

_BASE = (
    "/World/Building/_22_129___MSOE_Engineering_Building___A___2_25_2026"
    "/Geometry/Doors"
)

DOORS = [
    {"id": "Doors_13606466", "family_short": "CW_F_NEW", "floor": 4, "role": "train",
     "prim_path": f"{_BASE}/RS___Curtain_Wall___Type_F___NEW/New/Doors_13606466",
     "mesh_count": 1, "hinge_x_cm": -4836.01, "hinge_y_cm": 6170.58,
     "cam_pos": [-46.360, 62.163, 23.704], "cam_yaw_deg": 180.0,
     "door_center_m": [-48.360, 62.163, 23.704], "angle_sign": 1,
     "excluded_cells": [{"arc": -30}]},
    {"id": "Doors_10649156", "family_short": "Type_FG", "floor": 3, "role": "train",
     "prim_path": f"{_BASE}/RS___Type_FG_Door/NEW___HM_Frame___36_x96/Doors_10649156",
     "mesh_count": 0, "hinge_x_cm": +50.80, "hinge_y_cm": 0.0,
     "cam_pos": [-14.295, 80.200, 19.437], "cam_yaw_deg": 0.0,
     "door_center_m": [-12.295, 80.200, 19.437], "angle_sign": -1,
     "excluded_cells": [{"arc": -30}]},
    {"id": "Doors_8286057", "family_short": "Type_F", "floor": 2, "role": "train",
     "prim_path": f"{_BASE}/RS___Type_F_Door/NEW___HM_Frame___36_x96/Doors_8286057",
     "mesh_count": 0, "hinge_x_cm": +50.80, "hinge_y_cm": 0.0,
     "cam_pos": [17.443, 79.341, 14.764], "cam_yaw_deg": 270.0,
     "door_center_m": [17.443, 77.341, 14.764], "angle_sign": 1,
     "excluded_cells": [{"arc": 30}]},
    {"id": "Doors_3135827", "family_short": "CW_FG_New", "floor": 3, "role": "train",
     "prim_path": f"{_BASE}/RS___Curtain_Wall___Type_FG___New/New/Doors_3135827",
     "mesh_count": 2, "hinge_x_cm": 1724.33, "hinge_y_cm": 5878.48,
     "cam_pos": [17.701, 56.807, 19.437], "cam_yaw_deg": 90.0,
     "door_center_m": [17.701, 58.807, 19.437], "angle_sign": 1,
     "excluded_cells": []},
    {"id": "Doors_7171063", "family_short": "Type_F", "floor": 5, "role": "train",
     "prim_path": f"{_BASE}/RS___Type_F_Door/NEW___HM_Frame___42_x96/Doors_7171063",
     "mesh_count": 0, "hinge_x_cm": +58.42, "hinge_y_cm": 0.0,
     "cam_pos": [-49.316, 49.892, 27.972], "cam_yaw_deg": 0.0,
     "door_center_m": [-47.316, 49.892, 27.972], "angle_sign": -1,
     "excluded_cells": [{"arc": 30}]},
    {"id": "Doors_12590968", "family_short": "Type_N", "floor": 3, "role": "train",
     "prim_path": f"{_BASE}/RS___Type_N_Door/NEW___HM_Frame___36_x96/Doors_12590968",
     "mesh_count": 0, "hinge_x_cm": +50.80, "hinge_y_cm": 0.0,
     "cam_pos": [17.878, 50.941, 19.437], "cam_yaw_deg": 90.0,
     "door_center_m": [17.878, 52.941, 19.437], "angle_sign": 1,
     "excluded_cells": [{"arc": -30, "dist": 2.0}, {"arc": 30, "dist": 2.0}]},
    {"id": "Doors_12302795", "family_short": "CW_FG_New", "floor": 2, "role": "train",
     "prim_path": f"{_BASE}/RS___Curtain_Wall___Type_FG___New/New___GL1_1/Doors_12302795",
     "mesh_count": 2, "hinge_x_cm": -5925.83, "hinge_y_cm": 5774.34,
     "cam_pos": [-57.258, 58.251, 14.561], "cam_yaw_deg": 180.0,
     "door_center_m": [-59.258, 58.251, 14.561], "angle_sign": 1,
     "excluded_cells": [{"arc": 30, "dist": 2.0}]},
    {"id": "Doors_13606056", "family_short": "CW_F_NEW", "floor": 3, "role": "train",
     "prim_path": f"{_BASE}/RS___Curtain_Wall___Type_F___NEW/New/Doors_13606056",
     "mesh_count": 1, "hinge_x_cm": -4747.75, "hinge_y_cm": 6958.30,
     "cam_pos": [-45.477, 70.040, 19.437], "cam_yaw_deg": 180.0,
     "door_center_m": [-47.477, 70.040, 19.437], "angle_sign": 1,
     "excluded_cells": []},
    {"id": "Doors_13182085", "family_short": "CW_FG_New", "floor": 4, "role": "holdout",
     "prim_path": f"{_BASE}/RS___Curtain_Wall___Type_FG___New/New/Doors_13182085",
     "mesh_count": 2, "hinge_x_cm": -4142.56, "hinge_y_cm": 8214.32,
     "cam_pos": [-39.426, 82.600, 23.704], "cam_yaw_deg": 180.0,
     "door_center_m": [-41.426, 82.600, 23.704], "angle_sign": 1,
     "excluded_cells": []},
    {"id": "Doors_1915613", "family_short": "Type_F", "floor": 5, "role": "holdout",
     "prim_path": f"{_BASE}/RS___Type_F_Door/NEW___HM_Frame___36_x96/Doors_1915613",
     "mesh_count": 0, "hinge_x_cm": +50.80, "hinge_y_cm": 0.0,
     "cam_pos": [-58.753, 65.219, 27.972], "cam_yaw_deg": 270.0,
     "door_center_m": [-58.753, 63.219, 27.972], "angle_sign": 1,
     "excluded_cells": [{"arc": 30}]},
]

def is_excluded(door, cell):
    """Does this (arc, dist, height) cell match any of the door's exclusion rules?

    A rule is a dict of field->value; all listed fields must match (unlisted
    fields match any value). e.g. {"arc": -30} excludes all 4 cells on the
    -30 arc; {"arc": 30, "dist": 2.0} excludes the 2 cells at +30 / 2.0 m.
    """
    for rule in door.get("excluded_cells", []):
        if all(cell[k] == v for k, v in rule.items()):
            return True
    return False

def _grid_cells():
    for a in _ALLOWED_ARCS:
        for d in _ALLOWED_DISTS:
            for h in _ALLOWED_HEIGHTS:
                yield {"arc": a, "dist": d, "height": h}

GRID = list(_grid_cells())
assert len(GRID) == len(_ALLOWED_ARCS) * len(_ALLOWED_DISTS) * len(_ALLOWED_HEIGHTS), \
    f"Grid size mismatch: got {len(GRID)}"

# ---------------------------------------------------------------------------
# Isaac Sim startup
# ---------------------------------------------------------------------------

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

import time as _time
_t0 = _time.time()
def _ts(msg):
    print(f"[GridViz] [+{_time.time() - _t0:6.1f}s] {msg}")

_ts("Importing SimulationApp ...")
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})
_ts("SimulationApp ready")

import carb
_s = carb.settings.get_settings()
_s.set("/rtx/post/aa/op", 4)
_s.set("/rtx/post/dlss/execMode", 0)

import omni.usd
from pxr import UsdLux, UsdGeom, Gf, Sdf
import isaacsim.core.utils.numpy.rotations as rot_utils
from isaacsim.core.api import World
from isaacsim.sensors.camera import Camera
import omni.replicator.core as rep

from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------

world = World(stage_units_in_meters=1.0)
stage = omni.usd.get_context().get_stage()
_ts(f"Inserting building USD sublayer: {USD_PATH}")
stage.GetRootLayer().subLayerPaths.insert(0, USD_PATH)

# Neutral lighting — fixed dome + 4-slot ceiling array that we reposition per door.
dome = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/DomeLight"))
dome.CreateIntensityAttr(25_000)
dome.GetColorAttr().Set(Gf.Vec3f(1.0, 1.0, 1.0))

_CEILING_SPREAD = 1.2
_CEILING_RISE   = 1.4
_CEILING_OFFSETS = [
    (-_CEILING_SPREAD, -_CEILING_SPREAD), (+_CEILING_SPREAD, -_CEILING_SPREAD),
    (-_CEILING_SPREAD, +_CEILING_SPREAD), (+_CEILING_SPREAD, +_CEILING_SPREAD),
]
ceiling_lights = []
for i, (dx, dy) in enumerate(_CEILING_OFFSETS):
    rl = UsdLux.RectLight.Define(stage, Sdf.Path(f"/World/CeilingLight_{i}"))
    rl.CreateIntensityAttr(300_000)
    rl.CreateWidthAttr(0.6)
    rl.CreateHeightAttr(0.6)
    rl.GetColorAttr().Set(Gf.Vec3f(1.0, 0.98, 0.95))
    xf = UsdGeom.Xformable(rl.GetPrim())
    t_op = xf.AddTranslateOp()
    xf.AddRotateXYZOp().Set(Gf.Vec3f(180.0, 0.0, 0.0))
    ceiling_lights.append((t_op, dx, dy))

def reposition_ceiling(door):
    cx, cy = door["door_center_m"][0], door["door_center_m"][1]
    cz = door["cam_pos"][2] + _CEILING_RISE
    for t_op, dx, dy in ceiling_lights:
        t_op.Set(Gf.Vec3d(cx + dx, cy + dy, cz))

# ---------------------------------------------------------------------------
# Camera — one reusable slot, repositioned per grid cell
# ---------------------------------------------------------------------------

_ts("world.reset() ...")
world.reset()
simulation_app.update()
for _ in range(5):
    world.step(render=True)

cam = Camera(
    prim_path="/World/GridCam",
    position=np.array([0.0, 0.0, 10.0]),
    frequency=30,
    resolution=CAM_RESOLUTION,
    orientation=rot_utils.euler_angles_to_quats(np.array([0.0, 0.0, 0.0]), degrees=True),
)
simulation_app.update()
cam.initialize()

usd_cam = UsdGeom.Camera(stage.GetPrimAtPath("/World/GridCam"))
usd_cam.GetFocalLengthAttr().Set(5.0)
usd_cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 10000.0))

rp = rep.create.render_product("/World/GridCam", CAM_RESOLUTION)
rgb_ann = rep.AnnotatorRegistry.get_annotator("rgb")
rgb_ann.attach([rp])

_ts(f"Renderer warmup ({STARTUP_WARMUP} steps) ...")
for _ in range(STARTUP_WARMUP):
    world.step(render=True)
_ts("Warmup done")

# ---------------------------------------------------------------------------
# Camera placement math (matches door_production_collection.py)
# ---------------------------------------------------------------------------

def camera_pose_for(door, cell):
    """arc angle rotates around the door-facing normal;
    distance is radial from door center; height relative to cam_pos[2]."""
    dc = np.array([door["door_center_m"][0], door["door_center_m"][1]])
    arc_base = math.radians(door["cam_yaw_deg"] + 180.0)
    theta = math.radians(cell["arc"]) + arc_base
    pos = np.array([
        dc[0] + cell["dist"] * math.cos(theta),
        dc[1] + cell["dist"] * math.sin(theta),
        door["cam_pos"][2] + cell["height"],
    ])
    yaw_deg = math.degrees(theta) + 180.0
    ori = rot_utils.euler_angles_to_quats(np.array([0.0, 0.0, yaw_deg]), degrees=True)
    return pos, ori

def capture_frame():
    """Replicator-synced capture with retry; returns H x W x 3 uint8 or None."""
    for _ in range(MAX_CAPTURE_ATTEMPTS):
        rep.orchestrator.step(rt_subframes=RT_SUBFRAMES)
        data = rgb_ann.get_data()
        if data is None:
            for _ in range(3):
                world.step(render=True)
            continue
        arr = np.asarray(data)
        if arr.ndim != 3 or arr.shape[-1] < 3:
            for _ in range(3):
                world.step(render=True)
            continue
        arr3 = arr[:, :, :3]
        if int(arr3.max()) == 0:
            for _ in range(3):
                world.step(render=True)
            continue
        return arr3.astype(np.uint8)
    return None

# ---------------------------------------------------------------------------
# Contact sheet helpers
# ---------------------------------------------------------------------------

THUMB_W, THUMB_H = 320, 240
LABEL_H = 24
# Rows = arcs, cols = (dist, height) pairs. With 5 arcs x 4 pairs = 20 cells.
SHEET_COLS = len(_ALLOWED_DISTS) * len(_ALLOWED_HEIGHTS)
SHEET_ROWS = len(_ALLOWED_ARCS)

try:
    FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
except Exception:
    FONT = ImageFont.load_default()

def make_thumbnail(frame_uint8, label):
    img = Image.fromarray(frame_uint8).resize((THUMB_W, THUMB_H), Image.BILINEAR)
    tile = Image.new("RGB", (THUMB_W, THUMB_H + LABEL_H), (20, 20, 20))
    tile.paste(img, (0, LABEL_H))
    draw = ImageDraw.Draw(tile)
    draw.text((6, 4), label, fill=(255, 255, 255), font=FONT)
    return tile

def build_contact_sheet(tiles, title):
    sheet_w = SHEET_COLS * THUMB_W
    sheet_h = SHEET_ROWS * (THUMB_H + LABEL_H) + LABEL_H
    sheet = Image.new("RGB", (sheet_w, sheet_h), (0, 0, 0))
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 4), title, fill=(255, 255, 255), font=FONT)
    for i, tile in enumerate(tiles):
        r, c = divmod(i, SHEET_COLS)
        sheet.paste(tile, (c * THUMB_W, LABEL_H + r * (THUMB_H + LABEL_H)))
    return sheet

def cell_filename(cell):
    return f"arc{cell['arc']:+d}_dist{cell['dist']}_h{cell['height']:+.1f}.jpg"

def cell_label(cell):
    return f"arc {cell['arc']:+d}  d={cell['dist']}m  h={cell['height']:+.1f}m"

# ---------------------------------------------------------------------------
# Main loop — 10 doors x 12 cells
# ---------------------------------------------------------------------------

os.makedirs(OUT_ROOT, exist_ok=True)

# GRID ordering for the 4x3 sheet: rows = arcs (-30, 0, +30), cols = (dist,height) pairs.
# Sort cells so that contiguous groups of 4 share the same arc.
SHEET_ORDER = sorted(
    range(len(GRID)),
    key=lambda i: (
        _ALLOWED_ARCS.index(GRID[i]["arc"]),
        _ALLOWED_DISTS.index(GRID[i]["dist"]),
        _ALLOWED_HEIGHTS.index(GRID[i]["height"]),
    ),
)

def make_excluded_tile(label):
    tile = Image.new("RGB", (THUMB_W, THUMB_H + LABEL_H), (20, 20, 20))
    draw = ImageDraw.Draw(tile)
    draw.text((6, 4), label, fill=(180, 180, 180), font=FONT)
    # Dark red stripe band with EXCLUDED marker
    draw.rectangle([(0, LABEL_H), (THUMB_W, THUMB_H + LABEL_H)], fill=(48, 16, 16))
    try:
        bbox = draw.textbbox((0, 0), "EXCLUDED", font=FONT)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except Exception:
        tw, th = 80, 14
    draw.text(((THUMB_W - tw) // 2, LABEL_H + (THUMB_H - th) // 2),
              "EXCLUDED", fill=(255, 120, 120), font=FONT)
    return tile

total_saved = 0
total_dropped = 0
total_excluded = 0

for di, door in enumerate(DOORS):
    door_id = door["id"]
    door_dir = os.path.join(OUT_ROOT, door_id)
    # Clean slate: remove any stale JPEGs from a previous run so excluded
    # cells don't leave ghost images on disk.
    if os.path.isdir(door_dir):
        for f in os.listdir(door_dir):
            if f.endswith(".jpg"):
                os.remove(os.path.join(door_dir, f))
    os.makedirs(door_dir, exist_ok=True)

    reposition_ceiling(door)
    # Door warmup — let lighting settle at the new position
    for _ in range(DOOR_WARMUP):
        world.step(render=True)

    n_excluded_door = sum(1 for c in GRID if is_excluded(door, c))
    n_kept_door = len(GRID) - n_excluded_door
    _ts(f"[{di+1:2d}/{len(DOORS)}] {door_id} ({door['family_short']} fl{door['floor']} "
        f"{door['role']}) — {n_kept_door} kept / {n_excluded_door} excluded")

    tiles_by_index = [None] * len(GRID)
    excluded_flags = [False] * len(GRID)
    first_capture = True

    for ci, cell in enumerate(GRID):
        if is_excluded(door, cell):
            excluded_flags[ci] = True
            total_excluded += 1
            continue

        pos, ori = camera_pose_for(door, cell)
        try:
            cam.set_world_pose(position=pos, orientation=ori)
        except Exception as e:
            print(f"  set_world_pose failed for {cell}: {e}")
            continue

        simulation_app.update()
        warmup = CELL_WARMUP if first_capture else 3
        first_capture = False
        for _ in range(warmup):
            world.step(render=True)

        frame = capture_frame()
        if frame is None:
            total_dropped += 1
            print(f"  DROPPED (black/stale): {cell}")
            continue

        # Save individual JPEG at native 640x480
        jpg_path = os.path.join(door_dir, cell_filename(cell))
        Image.fromarray(frame).save(jpg_path, "JPEG", quality=88)
        tiles_by_index[ci] = make_thumbnail(frame, cell_label(cell))
        total_saved += 1

    # Build contact sheet in the row=arc, col=(dist,height) ordering.
    # Excluded cells get a clearly marked placeholder so layout stays 4x3.
    no_frame_placeholder = Image.new("RGB", (THUMB_W, THUMB_H + LABEL_H), (40, 0, 0))
    pd = ImageDraw.Draw(no_frame_placeholder)
    pd.text((6, 4), "(no frame)", fill=(255, 180, 180), font=FONT)

    ordered_tiles = []
    for i in SHEET_ORDER:
        if excluded_flags[i]:
            ordered_tiles.append(make_excluded_tile(cell_label(GRID[i])))
        elif tiles_by_index[i] is not None:
            ordered_tiles.append(tiles_by_index[i])
        else:
            ordered_tiles.append(no_frame_placeholder)

    arc_row_labels = " / ".join(f"{a:+d}" for a in _ALLOWED_ARCS)
    sheet_title = (f"{door_id}  |  {door['family_short']}  floor {door['floor']}  "
                   f"({door['role']})   rows: arc  {arc_row_labels}   "
                   f"kept {n_kept_door}/{len(GRID)}")
    sheet = build_contact_sheet(ordered_tiles, sheet_title)
    sheet_path = os.path.join(OUT_ROOT, f"{door_id}_sheet.jpg")
    sheet.save(sheet_path, "JPEG", quality=90)
    _ts(f"  -> {door_dir}/ + {os.path.basename(sheet_path)}")

_ts(f"DONE — saved {total_saved} JPEGs, excluded {total_excluded}, "
    f"dropped {total_dropped}, {len(DOORS)} contact sheets")
_ts(f"Output root: {OUT_ROOT}")

world.stop()
simulation_app.close()

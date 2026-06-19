#!/usr/bin/env python3
"""
door_baseline_dataset.py — Multi-door VLM zero-shot baseline dataset.

Builds a 1000-frame test set spanning multiple doors × camera viewpoints ×
door angles, used to compare 6 off-the-shelf VLMs (Qwen2-VL-2B,
Qwen2.5-VL-3B, Qwen3-VL-2B, Qwen2.5-VL-7B, SmolVLM2-2.2B, InternVL3-2B).

Hinge & camera placement (auto-derived from door_catalog.json):
  - Each door's true world bbox + `approach_axis` come from the catalog.
  - approach_axis tells us the door's normal direction (the axis with the
    ~4 cm thickness). The PANEL extends along the perpendicular horizontal
    axis. The hinge sits at one END of that panel axis (NOT the centre —
    that was the bug in the first run that caused doors to spin in place).
  - Cameras orbit the door centre along the approach axis. Both sides
    (+/− approach axis) are raycast-tested per door and the side with
    more clear lines of sight is used.

Camera arc per side: [−15°, 0°, +15°] × distances [1.5 m, 2.5 m] = 6 candidates
Camera height: door bbox centre Z minus 0.25 m
PhysX raycast prunes any candidate occluded by walls

Door angles (10 per door):
  CLOSED  [0, 1, 2, 3, 5]°
  OPEN    [10, 25, 45, 70, 90]°

Total target (full mode): exactly 1000 frames
  960 main grid (adaptive captures-per-cell across valid cells)
  + 40 decision-boundary bonus frames (3°-7°)

Modes:
  --sanity-check          Quick visual-inspection mode: 2 angles per door,
                          no jitter, JPEGs only.
  --candidate-doors A,B,C Override the door list (default = current 4 doors).
  --flip-doors A,B        For these door IDs, flip the hinge to the OTHER
                          end of the panel axis (in case the auto pick swings
                          the door against the wall).
"""

import os
import sys
import math
import json
import random
import argparse
import functools
import numpy as np

print = functools.partial(print, flush=True)  # noqa: A001

# ---------------------------------------------------------------------------
# CLI parsing — must come before SimulationApp import
# ---------------------------------------------------------------------------

DEFAULT_DOORS = "Doors_13607620,Doors_9817744,Doors_1868517,Doors_10383956"
ALL_CANDIDATES = ("Doors_13606713,Doors_13182085,Doors_9817744,Doors_3135827,"
                  "Doors_12066864,Doors_1868517,Doors_10383956,Doors_13607620")

parser = argparse.ArgumentParser(
    description="Collect a multi-door zero-shot VLM baseline dataset.")
parser.add_argument("--sanity-check", action="store_true",
                    help="Quick visual-inspection mode: 2 angles per door, no "
                         "jitter, JPEGs only.")
parser.add_argument("--candidate-doors", default=DEFAULT_DOORS,
                    help=f"Comma-separated door IDs (default: 4 doors). "
                         f"Sweep all 7 with --candidate-doors {ALL_CANDIDATES}")
parser.add_argument("--flip-doors", default="",
                    help="Comma-separated door IDs whose hinge should be "
                         "moved to the OTHER end of the panel axis.")
parser.add_argument("--output-dir", default="data/vlm_baseline",
                    help="Output directory relative to project root.")
parser.add_argument("--seed", type=int, default=20260412,
                    help="Random seed for jitter / shuffling.")
parser.add_argument("--skip-occlusion-check", action="store_true",
                    help="Skip PhysX raycast occlusion pruning of cameras.")
parser.add_argument("--hinge-sweep", action="store_true",
                    help="Hinge-calibration mode: render each door at angle 45° "
                         "with hinge inset offsets [0,3,5,8,10,13,15] cm. Used "
                         "to hand-pick the correct hinge per door.")

args, _unknown = parser.parse_known_args()

SANITY_CHECK    = args.sanity_check
HINGE_SWEEP     = args.hinge_sweep
OUTPUT_DIR_REL  = args.output_dir
SEED            = args.seed
DO_OCCLUSION    = not args.skip_occlusion_check
DOOR_IDS        = [s.strip() for s in args.candidate_doors.split(",") if s.strip()]
FLIP_IDS        = set(s.strip() for s in args.flip_doors.split(",") if s.strip())

random.seed(SEED)
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# Static config
# ---------------------------------------------------------------------------

CLOSED_MAX_DEG = 5.0
CAM_RESOLUTION = (1024, 768)   # matches door_static_collection.py training set
WARMUP_STEPS   = 5

CAM_ARC_OFFSETS = [-15.0, 0.0, 15.0]   # degrees off the door's approach axis
CAM_DISTANCES   = [1.4]                # metres from door centre (pulled in from 2.0
                                       # so the door fills more of the frame)
CAM_HEIGHT_DROP = 0.25                 # metres below door bbox centre Z

# Per (door, camera) cell we draw N_CLOSED_ANGLES samples uniformly from
# CLOSED_ANGLE_RANGE and N_OPEN_ANGLES samples from OPEN_ANGLE_RANGE. Each
# resulting angle is one cell, captured CAPTURES_PER_CELL times with jitter.
# Total = n_doors × n_cams × (N_CLOSED+N_OPEN) × CAPTURES_PER_CELL
# Defaults: 4 × 3 × 42 × 2 = 1008 frames.
N_CLOSED_ANGLES    = 21
N_OPEN_ANGLES      = 21
CLOSED_ANGLE_RANGE = (0.0, 5.0)
OPEN_ANGLE_RANGE   = (6.0, 90.0)
CAPTURES_PER_CELL  = 2

SANITY_ANGLES = [0.0, 45.0]

# Hinge-sweep mode renders one OPEN frame per (door, inset) so we can pick
# the correct hinge inset per door visually. Inset is positive cm "inward"
# from the bbox edge along the panel axis (towards the door centre).
HINGE_SWEEP_INSETS_CM  = [0.0, 3.0, 5.0, 8.0, 10.0, 13.0, 15.0]
HINGE_SWEEP_OPEN_ANGLE = 45.0

# Hand-measured hinge offsets per door (cm, inward along the panel axis from
# the bbox edge). Populated after running --hinge-sweep and visually picking
# the offset where the door panel pivots flush with the wall opening.
HINGE_OVERRIDES_CM = {
    # Offsets inward from the bbox edge so the rotated panel clears the
    # jamb (otherwise the panel Z-fights the wall at 0 cm and produces
    # TV-static noise). 5 cm was validated across multiple doors in the
    # hinge-sweep job 242577.
    "Doors_13607620":  5.0,
    "Doors_9817744":   5.0,
    "Doors_1868517":   5.0,
    "Doors_10383956":  5.0,
}

JITTER_POS_M        = 0.30   # ±30 cm — also serves as our radial distance jitter
JITTER_YAW_DEG      = 3.0
JITTER_ANGLE_CLOSED = 0.5
JITTER_ANGLE_OPEN   = 2.0

VISION_TOKEN_STR = "<|vision_start|><|image_pad|><|vision_end|>"
USER_PROMPT      = f"{VISION_TOKEN_STR}Is this door open or closed?"
CLOSED_ANSWER    = "Door Status: [CLOSED]."
OPEN_ANSWER      = "Door Status: [OPEN]."
JPEG_QUALITY     = 92

_script_dir  = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.abspath(os.path.join(_script_dir, ".."))
_catalog_path = os.path.join(_project_dir, "simulations", "robot", "door_catalog.json")

# ---------------------------------------------------------------------------
# Load catalog and build per-door specs (BEFORE SimulationApp)
# ---------------------------------------------------------------------------

with open(_catalog_path) as f:
    _catalog = json.load(f)
_doors_by_id = {d["id"]: d for d in _catalog["doors"]}

def _build_spec(door_id, flip, inset_cm_override=None):
    """
    Build a door spec from the catalog. Hinge is placed at one END of the
    panel axis (the horizontal axis perpendicular to the door's normal),
    then nudged INWARD by `inset_cm` toward the door centre to account
    for the door frame thickness (which the catalog bbox includes but the
    real hinge axis sits inside of).

    Default = the MIN end; flip=True moves it to the MAX end.
    inset_cm_override > 0 nudges toward the door centre.

    The catalog `bounding_box` is the actual world bbox in metres. The
    `dimensions.approach_axis` is the axis the door faces along ('X' or 'Y'),
    i.e. the axis whose bbox extent is the door thickness (~4 cm).
    """
    if door_id not in _doors_by_id:
        raise ValueError(f"Door id {door_id} not in catalog")
    d = _doors_by_id[door_id]
    bbox = d["bounding_box"]
    bmin, bmax = bbox["min"], bbox["max"]
    approach = d["dimensions"]["approach_axis"]   # 'X' or 'Y'

    cx = 0.5 * (bmin[0] + bmax[0])
    cy = 0.5 * (bmin[1] + bmax[1])
    cz = 0.5 * (bmin[2] + bmax[2])

    inset_cm = inset_cm_override if inset_cm_override is not None \
        else HINGE_OVERRIDES_CM.get(door_id, 0.0)

    if approach == "X":
        # door normal along X → panel extends in Y → hinge at one Y edge
        hinge_x_cm = cx * 100.0
        edge_cm = (bmax[1] if flip else bmin[1]) * 100.0
        # inward = toward cy*100 (door centre along panel axis)
        sign = -1.0 if flip else +1.0
        hinge_y_cm = edge_cm + sign * inset_cm
        panel_axis = "Y"
    elif approach == "Y":
        # door normal along Y → panel extends in X → hinge at one X edge
        edge_cm = (bmax[0] if flip else bmin[0]) * 100.0
        sign = -1.0 if flip else +1.0
        hinge_x_cm = edge_cm + sign * inset_cm
        hinge_y_cm = cy * 100.0
        panel_axis = "X"
    else:
        raise ValueError(f"Unknown approach_axis {approach!r} for {door_id}")

    return {
        "id":            door_id,
        "prim_path":     d["prim_path"],
        "floor":         d["floor"],
        "approach_axis": approach,
        "panel_axis":    panel_axis,
        "hinge_x_cm":    hinge_x_cm,
        "hinge_y_cm":    hinge_y_cm,
        "cx_m":          cx,
        "cy_m":          cy,
        "cz_m":          cz,
        "bbox_min":      list(bmin),
        "bbox_max":      list(bmax),
        "flipped":       flip,
        "inset_cm":      inset_cm,
    }

DOOR_SPECS = [_build_spec(did, did in FLIP_IDS) for did in DOOR_IDS]

print("[Baseline] Door specs:")
for s in DOOR_SPECS:
    print(f"  {s['id']:18s}  F{s['floor']}  approach={s['approach_axis']}  "
          f"hinge=({s['hinge_x_cm']:.1f},{s['hinge_y_cm']:.1f}) cm  "
          f"inset={s['inset_cm']:.1f}cm  "
          f"flip={'YES' if s['flipped'] else 'no'}")

# ---------------------------------------------------------------------------
# Boot Isaac Sim
# ---------------------------------------------------------------------------

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

import time as _time
_t0 = _time.time()
def _ts(msg):
    print(f"[Baseline] [+{_time.time() - _t0:6.1f}s] {msg}")

_ts(f"sanity_check={SANITY_CHECK}  doors={len(DOOR_SPECS)}  "
    f"angles_per_cam={N_CLOSED_ANGLES + N_OPEN_ANGLES}  "
    f"captures_per_cell={CAPTURES_PER_CELL}")
_ts("Importing SimulationApp ...")
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})
_ts("SimulationApp ready")

import carb
_carb_settings = carb.settings.get_settings()
# Only touch the settings that matter for THIS script: disable DLSS
# upscaling so RTX renders at native camera resolution. Everything else
# (tonemap / histogram / exposure / bloom / GI knobs) is left at Kit's
# defaults — earlier attempts to clamp auto-exposure or force a fixed
# manual exposure via /rtx/post/tonemap/filmIso etc. produced NO visible
# effect on the rendered frames in this Kit 5.1 build, and the "working"
# reference script (simulations/robot/door_static_collection.py, which
# produced the 96.3%-accuracy training set) never touches them either.
_carb_settings.set("/rtx/post/aa/op", 4)                 # RTXAA native-res AA
_carb_settings.set("/rtx/post/dlss/execMode", 0)         # belt-and-braces DLSS off
_ts("DLSS off, AA=RTXAA, Kit default tonemap/exposure")

import omni.usd
from pxr import Usd, UsdLux, UsdGeom, UsdShade, Gf, Sdf

import isaacsim.core.utils.numpy.rotations as rot_utils
from isaacsim.core.api import World
from isaacsim.sensors.camera import Camera

# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------

building_usd = os.path.join(_project_dir, "MSOEBuildingV4_physics.usd")
if not os.path.isfile(building_usd):
    print(f"ERROR: building USD not found: {building_usd}")
    simulation_app.close()
    sys.exit(1)

_ts("Creating World ...")
world = World(stage_units_in_meters=1.0)
stage = omni.usd.get_context().get_stage()
_ts(f"Inserting building USD as sublayer: {building_usd}")
stage.GetRootLayer().subLayerPaths.insert(0, building_usd)

# Lighting mirrors door_static_collection.py's `bright_fluorescent` preset
# (the condition used by the 96.3%-accuracy training set): a cool 15k
# DomeLight for sky ambient, a per-door 80k warm SphereLight fill 1.5 m
# above the door, plus a 4-slot 2×2 ceiling RectLight grid per door at
# 300k daylight. All light geometry is marked invisible so the fixtures
# themselves don't appear in the camera frame.

def _color_temp_to_rgb(kelvin):
    """Approximate black-body RGB for a given colour temperature."""
    t = float(kelvin) / 100.0
    if t <= 66:
        r = 255.0
        g = 99.4708025861 * math.log(t) - 161.1195681661 if t > 0 else 0.0
    else:
        r = 329.698727446 * ((t - 60) ** -0.1332047592)
        g = 288.1221695283 * ((t - 60) ** -0.0755148492)
    if t >= 66:
        b = 255.0
    elif t <= 19:
        b = 0.0
    else:
        b = 138.5177312231 * math.log(t - 10) - 305.0447927307
    r = max(0.0, min(255.0, r)) / 255.0
    g = max(0.0, min(255.0, g)) / 255.0
    b = max(0.0, min(255.0, b)) / 255.0
    return Gf.Vec3f(r, g, b)

# Dome: cool sky ambient at 50k / 5500K — bumped from the reference's 15k
# because several of the 4 baseline doors are solid panels in enclosed
# vestibules with no glass to let corridor light through. RTX autoexposure
# clamps the glass-door scenes so bumping the dome doesn't blow them out.
dome = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/DomeLight"))
dome.CreateIntensityAttr(50_000)
dome.CreateColorAttr(_color_temp_to_rgb(5500))

_CEILING_OFFSET_Z  = 1.2    # m above the camera Z (≈ door centre minus drop)
_CEILING_SPREAD_XY = 1.2    # m grid half-spacing
_CEILING_SIZE      = 0.6    # m per fixture
_CEILING_OFFSETS = [
    (-_CEILING_SPREAD_XY, -_CEILING_SPREAD_XY),
    (+_CEILING_SPREAD_XY, -_CEILING_SPREAD_XY),
    (-_CEILING_SPREAD_XY, +_CEILING_SPREAD_XY),
    (+_CEILING_SPREAD_XY, +_CEILING_SPREAD_XY),
]

door_fill_lights = []
door_ceiling_lights = []
for di, spec in enumerate(DOOR_SPECS):
    # Warm overhead fill sphere (bright_fluorescent: 80k @ 4500K)
    fill_path = f"/World/DoorFill_{di}"
    fill = UsdLux.SphereLight.Define(stage, Sdf.Path(fill_path))
    fill.CreateIntensityAttr(80_000)
    fill.CreateRadiusAttr(0.10)
    fill.CreateColorAttr(_color_temp_to_rgb(4500))
    fill_prim = fill.GetPrim()
    UsdGeom.Xformable(fill_prim).AddTranslateOp().Set(Gf.Vec3d(
        float(spec["cx_m"]),
        float(spec["cy_m"]),
        float(spec["cz_m"]) + 1.5,
    ))
    UsdGeom.Imageable(fill_prim).MakeInvisible()
    door_fill_lights.append(fill)

    # 2×2 ceiling RectLight grid (bright_fluorescent: 300k @ 5000K each)
    cam_z = float(spec["cz_m"]) - CAM_HEIGHT_DROP
    ceiling_set = []
    for ci, (dx, dy) in enumerate(_CEILING_OFFSETS):
        rl_path = f"/World/CeilingLight_{di}_{ci}"
        rl = UsdLux.RectLight.Define(stage, Sdf.Path(rl_path))
        rl.CreateIntensityAttr(800_000)
        rl.CreateWidthAttr(_CEILING_SIZE)
        rl.CreateHeightAttr(_CEILING_SIZE)
        rl.CreateColorAttr(_color_temp_to_rgb(5000))
        rl_prim = rl.GetPrim()
        xf = UsdGeom.Xformable(rl_prim)
        xf.AddTranslateOp().Set(Gf.Vec3d(
            float(spec["cx_m"]) + dx,
            float(spec["cy_m"]) + dy,
            cam_z + _CEILING_OFFSET_Z,
        ))
        xf.AddRotateXYZOp().Set(Gf.Vec3f(180.0, 0.0, 0.0))  # face down
        UsdGeom.Imageable(rl_prim).MakeInvisible()
        ceiling_set.append(rl)
    door_ceiling_lights.append(ceiling_set)

_ts("world.reset() — loading building USD ...")
world.reset()
_ts("world.reset() done")
simulation_app.update()
_ts("Pre-raycast warmup ...")
for _ in range(5):
    world.step(render=True)
_ts("Pre-raycast warmup done")

# ---------------------------------------------------------------------------
# Camera candidate generation — orbit the approach axis, both sides
# ---------------------------------------------------------------------------

def _approach_base_angle_deg(approach_axis, side_sign):
    """Camera-from-door bearing for a given approach axis and side."""
    if approach_axis == "X":
        return 0.0 if side_sign > 0 else 180.0   # +X / -X
    else:                                         # 'Y'
        return 90.0 if side_sign > 0 else 270.0  # +Y / -Y


def _generate_door_candidates(door_idx, spec, side_sign):
    """Build (arc × distance) candidate camera poses for one door + side."""
    cx, cy = spec["cx_m"], spec["cy_m"]
    cz = spec["cz_m"] - CAM_HEIGHT_DROP
    base_deg = _approach_base_angle_deg(spec["approach_axis"], side_sign)
    side_lab = "p" if side_sign > 0 else "n"
    candidates = []
    for dist in CAM_DISTANCES:
        for arc in CAM_ARC_OFFSETS:
            θ_deg = base_deg + arc
            θ = math.radians(θ_deg)
            pos = np.array([cx + dist * math.cos(θ),
                            cy + dist * math.sin(θ),
                            cz])
            yaw = θ_deg + 180.0  # face the door
            ori = rot_utils.euler_angles_to_quats(
                np.array([0.0, 0.0, yaw]), degrees=True)
            a_lab = f"{int(arc):+d}".replace("+", "p").replace("-", "n")
            d_lab = f"{dist:.1f}".replace(".", "p")
            prim_name = f"/World/BL_door{door_idx}_s{side_lab}_a{a_lab}_d{d_lab}"
            candidates.append({
                "prim_path":  prim_name,
                "pos":        pos,
                "ori":        ori,
                "yaw_deg":    yaw,
                "arc_deg":    arc,
                "side_sign":  side_sign,
                "side_label": side_lab,
                "distance":   dist,
                "door_idx":   door_idx,
            })
    return candidates


def _raycast_filter(candidates, spec):
    """Drop camera candidates whose line-of-sight to door centre is blocked."""
    if not DO_OCCLUSION:
        return candidates
    try:
        from omni.physx import get_physx_scene_query_interface
        physx_query = get_physx_scene_query_interface()
    except Exception as e:
        print(f"[WARN] PhysX scene query unavailable ({e}) — skipping occlusion check")
        return candidates

    cx, cy = spec["cx_m"], spec["cy_m"]
    door_path = spec["prim_path"]
    valid = []
    for cand in candidates:
        origin = cand["pos"]
        target = np.array([cx, cy, origin[2]])
        direction = target - origin
        max_dist = float(np.linalg.norm(direction))
        if max_dist < 0.01:
            valid.append(cand)
            continue
        direction = direction / max_dist
        hit = physx_query.raycast_closest(
            tuple(float(v) for v in origin),
            tuple(float(v) for v in direction),
            max_dist,
        )
        if hit["hit"]:
            hit_path = hit.get("rigidBody", "") or ""
            if door_path in hit_path:
                valid.append(cand)
            else:
                pass  # silent — verbose logging happens at the side-summary level
        else:
            valid.append(cand)
    return valid


# Pick the better side per door (whichever has more clear cameras)
DOOR_CAMERAS = []   # list[ list[(Camera, cfg)] ] indexed by door
DOOR_SIDE_CHOSEN = []
for di, spec in enumerate(DOOR_SPECS):
    plus_cands  = _generate_door_candidates(di, spec, +1)
    minus_cands = _generate_door_candidates(di, spec, -1)
    plus_valid  = _raycast_filter(plus_cands,  spec)
    minus_valid = _raycast_filter(minus_cands, spec)
    print(f"[Baseline] {spec['id']}  +side={len(plus_valid)}/6  "
          f"-side={len(minus_valid)}/6")
    if len(plus_valid) >= len(minus_valid):
        chosen, side_sign = plus_valid, +1
    else:
        chosen, side_sign = minus_valid, -1
    DOOR_SIDE_CHOSEN.append(side_sign)

    cam_list = []
    for cfg in chosen:
        cam = Camera(
            prim_path=cfg["prim_path"],
            position=cfg["pos"],
            frequency=30,
            resolution=CAM_RESOLUTION,
            orientation=cfg["ori"],
        )
        cam_list.append((cam, cfg))
    DOOR_CAMERAS.append(cam_list)
    print(f"           → chose side {'+' if side_sign>0 else '-'}{spec['approach_axis']}, "
          f"{len(cam_list)} cameras")

simulation_app.update()
for cam_list in DOOR_CAMERAS:
    for cam, cfg in cam_list:
        cam.initialize()
        usd_cam = UsdGeom.Camera(stage.GetPrimAtPath(cfg["prim_path"]))
        usd_cam.GetFocalLengthAttr().Set(5.0)
        usd_cam.GetClippingRangeAttr().Set((0.01, 10000.0))

n_cams_total = sum(len(c) for c in DOOR_CAMERAS)
if n_cams_total == 0:
    print("ERROR: no cameras survived raycast pruning across any door")
    simulation_app.close()
    sys.exit(1)
print(f"[Baseline] Total cameras created: {n_cams_total} across {len(DOOR_SPECS)} doors")

# ---------------------------------------------------------------------------
# Per-door mesh discovery + 4×4 matrix rotation
# ---------------------------------------------------------------------------
# No material overrides — we rely on the building USD's native OmniPBR
# bindings, same as door_static_collection.py. Applying UsdPreviewSurface
# overrides produced TV-static / etched-glass / reflective-puddle artifacts
# in earlier iterations.

DOOR_MESH_OPS = []   # list[ list[(mesh_info, xform_op)] ] indexed by door
for di, spec in enumerate(DOOR_SPECS):
    door_prim = stage.GetPrimAtPath(spec["prim_path"])
    if not door_prim.IsValid():
        print(f"ERROR: door prim not found: {spec['prim_path']}")
        simulation_app.close()
        sys.exit(1)
    mesh_ops = []
    for child in door_prim.GetChildren():
        if not child.IsA(UsdGeom.Mesh):
            continue
        xformable = UsdGeom.Xformable(child)
        op = None
        for xop in xformable.GetOrderedXformOps():
            if xop.GetOpName() == "xformOp:transform":
                op = xop
                break
        if op is None:
            continue
        mat = op.Get()
        mesh_ops.append(({
            "path": str(child.GetPath()),
            "ox":   mat[3][0],
            "oy":   mat[3][1],
            "oz":   mat[3][2],
        }, op))
    if not mesh_ops:
        print(f"ERROR: no transformable child meshes under {spec['prim_path']}")
        simulation_app.close()
        sys.exit(1)
    DOOR_MESH_OPS.append(mesh_ops)
    print(f"[Baseline] {spec['id']}: {len(mesh_ops)} child meshes — origins:")
    for m, _ in mesh_ops:
        print(f"    {m['path'].split('/')[-1]:40s}  "
              f"({m['ox']:9.2f}, {m['oy']:9.2f}, {m['oz']:9.2f}) cm")


def _door_matrix(hx_cm, hy_cm, ox_cm, oy_cm, oz_cm, angle_deg):
    θ = math.radians(angle_deg)
    c = math.cos(θ)
    s = math.sin(θ)
    dx = ox_cm - hx_cm
    dy = oy_cm - hy_cm
    tx = c * dx - s * dy + hx_cm
    ty = s * dx + c * dy + hy_cm
    return Gf.Matrix4d(
         c,  s, 0, 0,
        -s,  c, 0, 0,
         0,  0, 1, 0,
        tx, ty, oz_cm, 1,
    )


def set_door_angle(door_idx, angle_deg):
    spec = DOOR_SPECS[door_idx]
    for m, op in DOOR_MESH_OPS[door_idx]:
        op.Set(_door_matrix(spec["hinge_x_cm"], spec["hinge_y_cm"],
                            m["ox"], m["oy"], m["oz"], angle_deg))


for di in range(len(DOOR_SPECS)):
    set_door_angle(di, 0.0)

_ts("Renderer warmup (20 steps) ...")
for _ in range(20):
    world.step(render=True)
_ts("Renderer warmup done")

# ---------------------------------------------------------------------------
# Capture helpers
# ---------------------------------------------------------------------------

def _apply_jitter(cam, base_cfg, enable):
    if not enable:
        cam.set_world_pose(position=base_cfg["pos"], orientation=base_cfg["ori"])
        return base_cfg["pos"], base_cfg["yaw_deg"]
    pos = np.array([
        base_cfg["pos"][0] + random.uniform(-JITTER_POS_M, JITTER_POS_M),
        base_cfg["pos"][1] + random.uniform(-JITTER_POS_M, JITTER_POS_M),
        base_cfg["pos"][2] + random.uniform(-JITTER_POS_M, JITTER_POS_M),
    ])
    yaw = base_cfg["yaw_deg"] + random.uniform(-JITTER_YAW_DEG, JITTER_YAW_DEG)
    ori = rot_utils.euler_angles_to_quats(
        np.array([0.0, 0.0, yaw]), degrees=True)
    cam.set_world_pose(position=pos, orientation=ori)
    return pos, yaw


def _jittered_door_angle(base_angle, enable):
    if not enable:
        return base_angle
    if base_angle <= CLOSED_MAX_DEG:
        delta = random.uniform(-JITTER_ANGLE_CLOSED, JITTER_ANGLE_CLOSED)
        return max(0.0, min(CLOSED_MAX_DEG, base_angle + delta))
    delta = random.uniform(-JITTER_ANGLE_OPEN, JITTER_ANGLE_OPEN)
    return max(CLOSED_MAX_DEG + 1.0, min(90.0, base_angle + delta))


def _capture_one(door_idx, cam, cam_cfg, base_angle, jitter):
    actual_angle = _jittered_door_angle(base_angle, jitter)
    set_door_angle(door_idx, actual_angle)
    pos, yaw = _apply_jitter(cam, cam_cfg, jitter)
    for _ in range(WARMUP_STEPS):
        world.step(render=True)
    rgba = cam.get_rgba()
    if rgba is None:
        return None
    frame = rgba[:, :, :3].astype(np.uint8)
    return frame, actual_angle, pos.tolist(), yaw

# ---------------------------------------------------------------------------
# Output dirs
# ---------------------------------------------------------------------------

out_dir = os.path.join(_project_dir, OUTPUT_DIR_REL)
os.makedirs(out_dir, exist_ok=True)

try:
    from PIL import Image as _PIL
except ImportError:
    print("ERROR: PIL is required to write JPEGs")
    simulation_app.close()
    sys.exit(1)

# ---------------------------------------------------------------------------
# Sanity-check mode — fast pass, no jitter, JPEG dump only
# ---------------------------------------------------------------------------

if SANITY_CHECK:
    sanity_dir = os.path.join(out_dir, "sanity")
    os.makedirs(sanity_dir, exist_ok=True)
    print(f"[Baseline] SANITY MODE — dumping to {sanity_dir}")
    saved = 0
    for di, spec in enumerate(DOOR_SPECS):
        for base_angle in SANITY_ANGLES:
            for cam, cfg in DOOR_CAMERAS[di]:
                result = _capture_one(di, cam, cfg, base_angle, jitter=False)
                if result is None:
                    print(f"  [WARN] None frame from {spec['id']} angle={base_angle} "
                          f"cam={cfg['prim_path']}")
                    continue
                frame, _, _, _ = result
                cls = "CLOSED" if base_angle <= CLOSED_MAX_DEG else "OPEN"
                a_lab = f"{int(cfg['arc_deg']):+d}".replace("+", "p").replace("-", "n")
                d_lab = f"{cfg['distance']:.1f}".replace(".", "p")
                fname = (f"{spec['id']}_F{spec['floor']}_ang{int(base_angle):02d}_"
                         f"{cls}_side{cfg['side_label']}_a{a_lab}_d{d_lab}.jpg")
                _PIL.fromarray(frame).save(
                    os.path.join(sanity_dir, fname), "JPEG", quality=92)
                saved += 1
        set_door_angle(di, 0.0)
    print(f"[Baseline] SANITY MODE — saved {saved} JPEGs to {sanity_dir}")
    print(f"[Baseline] Inspect them, then re-submit without --sanity-check.")
    world.stop()
    simulation_app.close()
    sys.exit(0)

# ---------------------------------------------------------------------------
# Hinge-sweep mode — render each door at 45° for several hinge insets so we
# can pick the inset (cm) that puts the hinge flush with the wall opening.
# Uses the centre camera (arc=0) of each door's chosen side.
# ---------------------------------------------------------------------------

if HINGE_SWEEP:
    sweep_dir = os.path.join(out_dir, "hinge_sweep")
    os.makedirs(sweep_dir, exist_ok=True)
    print(f"[Baseline] HINGE-SWEEP MODE — dumping to {sweep_dir}")
    saved = 0
    for di, spec in enumerate(DOOR_SPECS):
        # find the centre-arc camera (arc_deg == 0) on the chosen side
        cam_list = DOOR_CAMERAS[di]
        centre = None
        for cam, cfg in cam_list:
            if abs(cfg["arc_deg"]) < 1e-3:
                centre = (cam, cfg)
                break
        if centre is None:
            centre = cam_list[0]
        cam, cfg = centre

        approach = spec["approach_axis"]
        flip     = spec["flipped"]
        bmin     = spec["bbox_min"]
        bmax     = spec["bbox_max"]
        sign     = -1.0 if flip else +1.0

        for inset in HINGE_SWEEP_INSETS_CM:
            if approach == "X":
                hx_cm = 0.5 * (bmin[0] + bmax[0]) * 100.0
                edge_cm = (bmax[1] if flip else bmin[1]) * 100.0
                hy_cm = edge_cm + sign * inset
            else:
                edge_cm = (bmax[0] if flip else bmin[0]) * 100.0
                hx_cm = edge_cm + sign * inset
                hy_cm = 0.5 * (bmin[1] + bmax[1]) * 100.0

            for m, op in DOOR_MESH_OPS[di]:
                op.Set(_door_matrix(hx_cm, hy_cm,
                                    m["ox"], m["oy"], m["oz"],
                                    HINGE_SWEEP_OPEN_ANGLE))
            cam.set_world_pose(position=cfg["pos"], orientation=cfg["ori"])
            for _ in range(WARMUP_STEPS):
                world.step(render=True)
            rgba = cam.get_rgba()
            if rgba is None:
                print(f"  [WARN] None frame for {spec['id']} inset={inset}")
                continue
            frame = rgba[:, :, :3].astype(np.uint8)
            i_lab = f"{int(round(inset)):02d}cm"
            fname = f"{spec['id']}_F{spec['floor']}_inset{i_lab}.jpg"
            _PIL.fromarray(frame).save(
                os.path.join(sweep_dir, fname), "JPEG", quality=92)
            saved += 1
            print(f"  {spec['id']:18s} inset={inset:5.1f}cm → "
                  f"hinge=({hx_cm:.1f},{hy_cm:.1f})  saved {fname}")

        # restore door to closed before moving on
        for m, op in DOOR_MESH_OPS[di]:
            op.Set(_door_matrix(spec["hinge_x_cm"], spec["hinge_y_cm"],
                                m["ox"], m["oy"], m["oz"], 0.0))

    print(f"[Baseline] HINGE-SWEEP MODE — saved {saved} JPEGs to {sweep_dir}")
    world.stop()
    simulation_app.close()
    sys.exit(0)

# ---------------------------------------------------------------------------
# Full collection mode
# ---------------------------------------------------------------------------

main_cells = []
for di in range(len(DOOR_SPECS)):
    for cam, cfg in DOOR_CAMERAS[di]:
        for _ in range(N_CLOSED_ANGLES):
            base_angle = random.uniform(*CLOSED_ANGLE_RANGE)
            main_cells.append((di, cam, cfg, base_angle))
        for _ in range(N_OPEN_ANGLES):
            base_angle = random.uniform(*OPEN_ANGLE_RANGE)
            main_cells.append((di, cam, cfg, base_angle))

n_cells = len(main_cells)
if n_cells == 0:
    print("ERROR: no main cells")
    simulation_app.close()
    sys.exit(1)

total_target = n_cells * CAPTURES_PER_CELL
print(f"[Baseline] {n_cells} cells "
      f"({len(DOOR_SPECS)} doors × {sum(len(c) for c in DOOR_CAMERAS)} cameras × "
      f"{N_CLOSED_ANGLES + N_OPEN_ANGLES} random angles) "
      f"× {CAPTURES_PER_CELL} captures = {total_target} frames")

# ---------------------------------------------------------------------------
# Capture loop — main grid
# ---------------------------------------------------------------------------

records = []
all_frames = []
all_angles = []
all_floors = []
all_arc = []
all_distances = []
img_dir = os.path.join(out_dir, "images")
os.makedirs(img_dir, exist_ok=True)

frame_counter = 0
def _save_record(spec, cfg, frame, true_angle, pos, yaw, source):
    global frame_counter
    fname = f"frame_{frame_counter:05d}.jpg"
    fpath = os.path.join(img_dir, fname)
    _PIL.fromarray(frame).save(fpath, "JPEG", quality=JPEG_QUALITY)
    cls = "CLOSED" if true_angle <= CLOSED_MAX_DEG else "OPEN"
    answer = CLOSED_ANSWER if cls == "CLOSED" else OPEN_ANSWER
    records.append({
        "conversations": [
            {"role": "user",      "content": USER_PROMPT},
            {"role": "assistant", "content": answer},
        ],
        "image": os.path.relpath(fpath, out_dir),
        "metadata": {
            "door_id":      spec["id"],
            "floor":        spec["floor"],
            "true_angle":   round(float(true_angle), 3),
            "label":        cls,
            "cam_arc":      float(cfg["arc_deg"]),
            "cam_side":     cfg["side_label"],
            "cam_distance": float(cfg["distance"]),
            "cam_pos":      [round(float(v), 4) for v in pos],
            "cam_yaw":      round(float(yaw), 3),
            "source":       source,
        },
    })
    all_frames.append(frame)
    all_angles.append(float(true_angle))
    all_floors.append(int(spec["floor"]))
    all_arc.append(float(cfg["arc_deg"]))
    all_distances.append(float(cfg["distance"]))
    frame_counter += 1

_ts("Starting main capture ...")
for ci, (di, cam, cfg, base_angle) in enumerate(main_cells):
    spec = DOOR_SPECS[di]
    for _ in range(CAPTURES_PER_CELL):
        result = _capture_one(di, cam, cfg, base_angle, jitter=True)
        if result is None:
            continue
        frame, true_angle, pos, yaw = result
        _save_record(spec, cfg, frame, true_angle, pos, yaw, "main")
    if (ci + 1) % 50 == 0 or ci + 1 == n_cells:
        print(f"[Baseline] cell {ci+1}/{n_cells}  "
              f"({frame_counter} frames so far)  "
              f"door={spec['id']} angle={base_angle:.1f}°")

# ---------------------------------------------------------------------------
# Persist
# ---------------------------------------------------------------------------

jsonl_path = os.path.join(out_dir, "baseline_test.jsonl")
with open(jsonl_path, "w") as f:
    for rec in records:
        f.write(json.dumps(rec) + "\n")
    f.flush()
    os.fsync(f.fileno())
print(f"[Baseline] Wrote {len(records)} records → {jsonl_path}")

npz_path = os.path.join(out_dir, "baseline_test.npz")
np.savez_compressed(
    npz_path,
    frames        = np.array(all_frames,    dtype=np.uint8),
    angles        = np.array(all_angles,    dtype=np.float32),
    floor         = np.array(all_floors,    dtype=np.int32),
    cam_arc       = np.array(all_arc,       dtype=np.float32),
    cam_distances = np.array(all_distances, dtype=np.float32),
)
print(f"[Baseline] Wrote NPZ ({os.path.getsize(npz_path)//1024//1024} MB) → {npz_path}")

manifest = {
    "target_frames":     total_target,
    "actual_frames":     len(records),
    "doors":             [s["id"] for s in DOOR_SPECS],
    "door_floors":       [s["floor"] for s in DOOR_SPECS],
    "door_sides":        [int(s) for s in DOOR_SIDE_CHOSEN],
    "flipped_doors":     sorted(FLIP_IDS),
    "cam_arc_offsets":   CAM_ARC_OFFSETS,
    "cam_distances":     CAM_DISTANCES,
    "n_closed_angles":   N_CLOSED_ANGLES,
    "n_open_angles":     N_OPEN_ANGLES,
    "closed_range":      list(CLOSED_ANGLE_RANGE),
    "open_range":        list(OPEN_ANGLE_RANGE),
    "captures_per_cell": CAPTURES_PER_CELL,
    "seed":              SEED,
    "n_cells":           n_cells,
    "cameras_per_door":  [len(c) for c in DOOR_CAMERAS],
}
mpath = os.path.join(out_dir, "baseline_manifest.json")
with open(mpath, "w") as f:
    json.dump(manifest, f, indent=2)
    f.flush()
    os.fsync(f.fileno())
print(f"[Baseline] Wrote manifest → {mpath}")

closed_n = sum(1 for r in records if r["metadata"]["label"] == "CLOSED")
open_n   = sum(1 for r in records if r["metadata"]["label"] == "OPEN")
print(f"[Baseline] Class balance — CLOSED: {closed_n}  OPEN: {open_n}")
_ts("Done.")

world.stop()
simulation_app.close()

#!/usr/bin/env python3
"""
Door State Data Collection — Fixed Camera (Script B)

Captures a door at programmatically set angles from a multi-camera grid
(arc angles × distances × heights). Per-episode lighting randomization
adds illumination diversity. Cameras that are occluded by walls are
automatically pruned via physics raycasting.

Door rotation strategy — direct 4×4 matrix override on each child mesh:
  The child meshes each have a single xformOp:transform that is a 4×4
  identity-rotation + absolute-position matrix (from the Revit export).
  We directly rewrite the 4×4 matrix of each child mesh to encode the
  desired rotation around the hinge, keeping the rotation and translation
  tightly coupled with no extra ops to compose incorrectly.

  For a hinge at (hx, hy) cm and original mesh origin at (ox, oy, oz) cm:
    tx = cos(θ)*(ox-hx) - sin(θ)*(oy-hy) + hx
    ty = sin(θ)*(ox-hx) + cos(θ)*(oy-hy) + hy
    M  = | cos  sin  0  0 |   (USD row-vector convention)
         |-sin  cos  0  0 |
         |  0    0   1  0 |
         |  tx  ty  tz  1 |

Camera grid:
  Each camera position is a combination of (arc_angle, distance, height).
  Before creating Camera prims, a physics raycast from each candidate
  position toward the door center checks for wall occlusion. Positions
  where the ray hits geometry before reaching the door are skipped.

Output: data/<output>.npz
  frames      (N, H, W, 3) uint8 RGB
  labels      (N,) float32  angle_deg / 90.0
  angles      (N,) float32  degrees
  episode     (N,) int32
  cam_offsets (N,) int32

Usage:
  # Original single-door (backward compatible — 3 angles, 1 distance, 1 height):
  sbatch submit_isaac_job.sh simulations/robot/door_static_collection.py

  # Full grid with auto wall filtering:
  sbatch submit_isaac_job.sh simulations/robot/door_static_collection.py \\
      --door-prim "/World/Building/.../Doors_XXXXXXXX" \\
      --hinge-x -2047.7 --hinge-y 7969.1 \\
      --camera-center-x -20.48 --camera-center-y 80.15 \\
      --camera-z 27.8 \\
      --camera-arc-offsets -60,-45,-30,-15,0,15,30,45,60 \\
      --camera-distances 1.5,2.0,3.5 \\
      --camera-height-offsets -0.5,0.0,1.0 \\
      --output door_labels_XXXXXXXX
"""

import os
import sys
import math
import random
import argparse
import functools
import numpy as np

# Force every print() in this module to flush immediately to the SLURM log.
# Isaac Sim reroutes stdout internally and our progress prints were never
# appearing in the slurm .out file until the process exited — which made
# the data-collection jobs look "stuck" even when they were running.
print = functools.partial(print, flush=True)  # noqa: A001 — intentional shadow

# ---------------------------------------------------------------------------
# Parse CLI arguments BEFORE SimulationApp (argparse must run first so
# Isaac Sim doesn't choke on our flags, and so we can set defaults).
# ---------------------------------------------------------------------------

# Defaults — reproduce the original Doors_13606713 behavior exactly.
_DEFAULT_DOOR_PATH = (
    "/World/Building/_22_129___MSOE_Engineering_Building___A___2_25_2026"
    "/Geometry/Doors/RS___Curtain_Wall___Type_FG___New/New/Doors_13606713"
)
_DEFAULT_HINGE_X   = -2047.728271484375  # cm
_DEFAULT_HINGE_Y   =  7969.07            # cm
_DEFAULT_CAM_CX    = -20.48              # m (world XY)
_DEFAULT_CAM_CY    =  80.15              # m
_DEFAULT_CAM_Z     =  27.8               # m
_DEFAULT_ARC       = "-30,-15,0"
_DEFAULT_DISTS     = "2.0"
_DEFAULT_HEIGHTS   = "0.0"
_DEFAULT_OUTPUT    = "door_labels_static_multi_camera"

parser = argparse.ArgumentParser(
    description="Collect door-state training data from a parameterized door in the building USD."
)
parser.add_argument("--door-prim", default=_DEFAULT_DOOR_PATH,
                    help="USD prim path to the door (e.g. /World/.../Doors_XXXXXXXX)")
parser.add_argument("--hinge-x", type=float, default=_DEFAULT_HINGE_X,
                    help="Hinge X position in cm (world coordinates)")
parser.add_argument("--hinge-y", type=float, default=_DEFAULT_HINGE_Y,
                    help="Hinge Y position in cm (world coordinates)")
parser.add_argument("--camera-center-x", type=float, default=_DEFAULT_CAM_CX,
                    help="Door center X in metres (for camera arc)")
parser.add_argument("--camera-center-y", type=float, default=_DEFAULT_CAM_CY,
                    help="Door center Y in metres (for camera arc)")
parser.add_argument("--camera-z", type=float, default=_DEFAULT_CAM_Z,
                    help="Camera base Z height in metres (height offsets are relative to this)")
parser.add_argument("--camera-arc-offsets", default=_DEFAULT_ARC,
                    help="Comma-separated arc angle offsets in degrees (default: -30,-15,0). "
                         "For full sweep try: -60,-45,-30,-15,0,15,30,45,60")
parser.add_argument("--camera-distances", default=_DEFAULT_DISTS,
                    help="Comma-separated camera distances from door center in metres "
                         "(default: 2.0). For multi-ring try: 1.5,2.0,3.5")
parser.add_argument("--camera-height-offsets", default=_DEFAULT_HEIGHTS,
                    help="Comma-separated height offsets relative to --camera-z in metres "
                         "(default: 0.0). For multi-height try: -0.5,0.0,1.0")
parser.add_argument("--output", default=_DEFAULT_OUTPUT,
                    help="Output NPZ filename (without .npz extension)")
parser.add_argument("--n-episodes", type=int, default=60,
                    help="Number of episodes (default: 60)")
parser.add_argument("--frames-per-episode", type=int, default=10,
                    help="Frames captured per episode (default: 10)")
parser.add_argument("--no-lighting-randomization", action="store_true",
                    help="Disable per-episode lighting randomization (use fixed lighting)")
parser.add_argument("--no-occlusion-check", action="store_true",
                    help="Skip physics raycast occlusion check (use all candidate positions)")
parser.add_argument("--flip-swing", action="store_true",
                    help="Negate door rotation angle so the door swings the opposite direction")
parser.add_argument("--grid-mode", action="store_true",
                    help="Use deterministic grid search over LIGHTING_CONDITIONS × DOOR_MATERIALS × ANGLE_GRID "
                         "(Phase 1 domain variation). Overrides --n-episodes and --frames-per-episode.")
parser.add_argument("--no-material-variation", action="store_true",
                    help="In grid mode: skip door material variation (use original material only)")
parser.add_argument("--no-camera-jitter", action="store_true",
                    help="Disable per-episode camera position/orientation jitter")
parser.add_argument("--grid-frames-per-cell", type=int, default=1,
                    help="In grid mode: number of frames captured per grid cell "
                         "(default: 1 — spatial diversity already comes from multiple cameras)")
parser.add_argument("--baseline-only", action="store_true",
                    help="In grid mode: collect with fixed baseline only (bright_fluorescent + original). "
                         "Use for generalization test doors — no domain variation.")
parser.add_argument("--test-mode", action="store_true",
                    help="In grid mode: shrink the grid to a minimal subset "
                         "(2 lightings × 3 materials × 3 angles) for fast debugging.")
parser.add_argument("--sanity-grid", action="store_true",
                    help="In grid mode: use the full 6 lightings × 7 materials but only "
                         "3 angles (1°, 45°, 90°) for visual sanity check. ~126 episodes.")
parser.add_argument("--dump-jpegs", action="store_true",
                    help="Write every captured frame as a JPEG to data/<output>_jpegs/ "
                         "with a descriptive filename. Useful for visual audit of a sanity run. "
                         "AUTO-ENABLED when --test-mode or --sanity-grid is set.")

# Parse only known args — Isaac Sim may inject its own args via python.sh.
args, _unknown = parser.parse_known_args()

# Derived config from CLI args
DOOR_PATH       = args.door_prim
HINGE_X_CM      = args.hinge_x
HINGE_Y_CM      = args.hinge_y
DOOR_CENTER_XY  = np.array([args.camera_center_x, args.camera_center_y])
CAMERA_Z_BASE   = args.camera_z
CAM_ARC_OFFSETS = [float(x.strip()) for x in args.camera_arc_offsets.split(",")]
CAM_DISTANCES   = [float(x.strip()) for x in args.camera_distances.split(",")]
CAM_HEIGHT_OFFS = [float(x.strip()) for x in args.camera_height_offsets.split(",")]
OUTPUT_NAME     = args.output
N_EPISODES      = args.n_episodes
FRAMES_PER_EP   = args.frames_per_episode
LIGHTING_RAND   = not args.no_lighting_randomization
DO_OCCLUSION    = not args.no_occlusion_check
FLIP_SWING      = args.flip_swing
GRID_MODE       = args.grid_mode
MATERIAL_VAR    = not args.no_material_variation
CAMERA_JITTER   = not args.no_camera_jitter
GRID_FRAMES     = args.grid_frames_per_cell
BASELINE_ONLY   = args.baseline_only
TEST_MODE       = args.test_mode
SANITY_GRID     = args.sanity_grid
# --dump-jpegs is auto-enabled for --test-mode and --sanity-grid since those
# modes exist specifically for visual inspection.
DUMP_JPEGS      = args.dump_jpegs or TEST_MODE or SANITY_GRID

CAM_RESOLUTION  = (640, 480)

# Episode config
# WARMUP_STEPS: render steps between (angle/lighting/material) change and frame
# capture. Door rotation is a direct 4×4 matrix rewrite so there is no physics
# convergence to wait for — we just need 1-2 frames for the RTX renderer and
# material-binding system to pick up the new scene state. 5 is conservative.
# (Was 40 in the legacy random mode; that turned the grid into a ~2-hour run.)
WARMUP_STEPS   = 5
CLOSED_MAX_DEG = 5.0
OPEN_MIN_DEG   = 6.0
MAX_ANGLE_DEG  = 90.0

# Lighting randomization ranges (legacy — used when grid mode is OFF)
DOME_INTENSITY_RANGE  = (10_000, 80_000)
DOME_COLOR_TEMP_RANGE = (4_000, 7_000)   # Kelvin
FILL_INTENSITY_RANGE  = (50_000, 500_000)

# ---------------------------------------------------------------------------
# Grid search: discrete lighting conditions × door materials
# ---------------------------------------------------------------------------

# HDRI environment maps bundled with Isaac Sim — set on DomeLight's
# inputs:texture:file for photorealistic image-based lighting. When None,
# the DomeLight uses its flat color attribute instead.
#
# photo_studio_01_4k.hdr is a neutral 4K studio HDR (~25 MB) and is the only
# large HDR shipped with Isaac Sim. We pick exactly the lighting presets where
# strong natural/window light is expected and attach it to those cells.
_HDRI_STUDIO = os.path.expanduser(
    "~/isaacsim/extscache/omni.kit.widget.material_preview-1.0.16/data/photo_studio_01_4k.hdr"
)

# 6 realistic indoor lighting scenarios for the Kern Innovation Center
LIGHTING_CONDITIONS = [
    {"name": "bright_fluorescent", "dome_int": 25_000, "dome_temp": 5500,
     "fill_int": 400_000, "fill_temp": 4500,
     "hdri": None},                                # artificial ceiling light
    {"name": "daylight_flood",     "dome_int": 70_000, "dome_temp": 6500,
     "fill_int": 150_000, "fill_temp": 6000,
     "hdri": _HDRI_STUDIO},                        # window-driven natural light
    {"name": "warm_evening",       "dome_int": 20_000, "dome_temp": 3000,
     "fill_int": 200_000, "fill_temp": 2800,
     "hdri": None},                                # warm tungsten fixtures
    {"name": "dim_night",          "dome_int":  5_000, "dome_temp": 4000,
     "fill_int":  50_000, "fill_temp": 3500,
     "hdri": None},                                # minimal overnight lighting
    {"name": "overcast_flat",      "dome_int": 40_000, "dome_temp": 5000,
     "fill_int": 100_000, "fill_temp": 5000,
     "hdri": _HDRI_STUDIO},                        # diffuse sky through windows
    {"name": "high_contrast",      "dome_int":  5_000, "dome_temp": 5000,
     "fill_int": 700_000, "fill_temp": 5500,
     "hdri": None},                                # single bright overhead only
]

# 6 realistic door materials + "original" (no override)
DOOR_MATERIALS = [
    {"name": "brushed_aluminum",    "color": (0.75, 0.75, 0.78), "roughness": 0.35, "metallic": 0.85},
    {"name": "painted_white",       "color": (0.90, 0.90, 0.88), "roughness": 0.70, "metallic": 0.05},
    {"name": "dark_anodized",       "color": (0.12, 0.12, 0.15), "roughness": 0.25, "metallic": 0.80},
    {"name": "wood_veneer",         "color": (0.55, 0.35, 0.20), "roughness": 0.75, "metallic": 0.02},
    {"name": "red_fire_door",       "color": (0.65, 0.10, 0.08), "roughness": 0.60, "metallic": 0.10},
    {"name": "beige_institutional", "color": (0.72, 0.68, 0.58), "roughness": 0.65, "metallic": 0.08},
    {"name": "original",            "color": None, "roughness": None, "metallic": None},
]

# Stratified angles: 5 CLOSED (0-5°) + 5 OPEN (spread across 6-90°)
ANGLE_GRID = [1.0, 2.5, 3.5, 4.0, 5.0, 10.0, 25.0, 45.0, 70.0, 90.0]

_script_dir  = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.abspath(os.path.join(_script_dir, "..", ".."))

# ---------------------------------------------------------------------------
# Now start Isaac Sim (after argparse is done)
# ---------------------------------------------------------------------------

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

import time as _time
_t0 = _time.time()
def _ts(msg):
    print(f"[StaticCollect] [+{_time.time() - _t0:6.1f}s] {msg}")

_ts("Importing SimulationApp ...")
from isaacsim import SimulationApp
_ts("Creating SimulationApp(headless=True) ...")
simulation_app = SimulationApp({"headless": True})
_ts("SimulationApp ready")

import omni.usd
from pxr import Usd, UsdLux, UsdGeom, UsdShade, Gf, Sdf

import isaacsim.core.utils.numpy.rotations as rot_utils
from isaacsim.core.api import World
from isaacsim.sensors.camera import Camera

# ---------------------------------------------------------------------------
# Generate candidate camera positions (arc × distance × height)
# ---------------------------------------------------------------------------

def _generate_candidate_positions():
    """
    Build all (angle, distance, height_offset) combinations.
    Returns list of dicts with position, orientation, and metadata.
    """
    candidates = []
    idx = 0
    for dist in CAM_DISTANCES:
        for h_off in CAM_HEIGHT_OFFS:
            cam_z = CAMERA_Z_BASE + h_off
            for offset_deg in CAM_ARC_OFFSETS:
                θ = math.radians(offset_deg)
                cam_pos = np.array([
                    DOOR_CENTER_XY[0] + dist * math.cos(θ),
                    DOOR_CENTER_XY[1] + dist * math.sin(θ),
                    cam_z,
                ])
                yaw_deg = offset_deg + 180.0  # face back toward door centre
                cam_ori = rot_utils.euler_angles_to_quats(
                    np.array([0.0, 0.0, yaw_deg]), degrees=True
                )
                # Unique prim name encoding angle, distance ring, and height tier
                d_label = f"d{dist:.1f}".replace(".", "p")
                h_label = f"h{h_off:+.1f}".replace("+", "p").replace("-", "n").replace(".", "p")
                a_label = f"{offset_deg:+d}".replace("+", "p").replace("-", "n")
                prim_name = f"/World/StaticCam_{a_label}_{d_label}_{h_label}"

                candidates.append({
                    "prim_path":    prim_name,
                    "pos":          cam_pos,
                    "ori":          cam_ori,
                    "offset_deg":   offset_deg,
                    "distance":     dist,
                    "height_off":   h_off,
                    "idx":          idx,
                })
                idx += 1
    return candidates

ALL_CANDIDATES = _generate_candidate_positions()
print(f"[StaticCollect] Generated {len(ALL_CANDIDATES)} candidate camera positions "
      f"({len(CAM_ARC_OFFSETS)} angles × {len(CAM_DISTANCES)} distances × "
      f"{len(CAM_HEIGHT_OFFS)} heights)")

# ---------------------------------------------------------------------------
# Scene setup
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
_ts("Building USD sublayer inserted (actual load happens on world.reset)")

dome = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/DomeLight"))
dome.CreateIntensityAttr(50000)
# Create the HDRI texture file attribute up front (empty by default) so
# apply_lighting_condition() can simply .Set() it per preset without having
# to Create it each time. An empty asset path means "use the flat color".
_dome_tex_attr = dome.CreateTextureFileAttr()
_dome_tex_attr.Set(Sdf.AssetPath(""))

# Fill light positioned above the door center
fill = UsdLux.SphereLight.Define(stage, Sdf.Path("/World/FillLight"))
fill.CreateIntensityAttr(300000)
fill.CreateRadiusAttr(0.1)
UsdGeom.Xformable(fill.GetPrim()).AddTranslateOp().Set(
    (float(DOOR_CENTER_XY[0]), float(DOOR_CENTER_XY[1]), float(CAMERA_Z_BASE) + 1.5)
)

# ---------------------------------------------------------------------------
# Reusable OmniPBR material for door variation (grid mode)
# Created once at startup; per-episode we just update its shader inputs.
# ---------------------------------------------------------------------------

_door_mat_path   = "/World/Looks/DoorMatGrid"
_door_mat        = UsdShade.Material.Define(stage, Sdf.Path(_door_mat_path))
_door_shader     = UsdShade.Shader.Define(stage, Sdf.Path(_door_mat_path + "/Shader"))
_door_shader.CreateIdAttr("OmniPBR")
_door_shader.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.8, 0.8, 0.8))
_door_shader.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(0.5)
_door_shader.CreateInput("metallic_constant", Sdf.ValueTypeNames.Float).Set(0.0)
_door_mat.CreateSurfaceOutput().ConnectToSource(_door_shader.ConnectableAPI(), "surface")

# ---------------------------------------------------------------------------
# Physics raycast occlusion check
# ---------------------------------------------------------------------------

# Step physics once so collision meshes are loaded before raycasting
_ts("world.reset() — loading building USD into memory (this is the slow part)")
world.reset()
_ts("world.reset() done")
simulation_app.update()
_ts("5-step pre-raycast warmup ...")
for _ in range(5):
    world.step(render=True)
_ts("Pre-raycast warmup done")

def _check_line_of_sight(candidates):
    """
    Use PhysX raycasting to filter out camera positions where the line of
    sight to the door center is blocked by building geometry (walls, floors).
    Returns the subset of candidates with clear line of sight.
    """
    if not DO_OCCLUSION:
        print("[StaticCollect] Occlusion check disabled — using all candidate positions")
        return candidates

    try:
        from omni.physx import get_physx_scene_query_interface
        physx_query = get_physx_scene_query_interface()
    except (ImportError, Exception) as e:
        print(f"[WARN] Could not load PhysX scene query ({e}) — skipping occlusion check")
        return candidates

    # Door prim path prefix — hits on the door itself should NOT count as occlusion
    door_path_prefix = DOOR_PATH

    valid = []
    blocked = 0
    for cand in candidates:
        origin = cand["pos"]
        # Direction from camera toward door center (at camera's Z height)
        target = np.array([DOOR_CENTER_XY[0], DOOR_CENTER_XY[1], origin[2]])
        direction = target - origin
        max_dist = float(np.linalg.norm(direction))
        if max_dist < 0.01:
            valid.append(cand)
            continue
        direction = direction / max_dist  # normalize

        # carb.Float3 expects individual floats
        hit = physx_query.raycast_closest(
            tuple(float(v) for v in origin),
            tuple(float(v) for v in direction),
            max_dist,
        )

        if hit["hit"]:
            # Check if the hit prim is part of the door itself (not a wall)
            hit_path = hit.get("rigidBody", "") or ""
            if door_path_prefix in hit_path:
                # Ray hit the door — line of sight is clear
                valid.append(cand)
            else:
                blocked += 1
                a = cand["offset_deg"]
                d = cand["distance"]
                h = cand["height_off"]
                print(f"  [Occluded] arc={a:+.0f}° dist={d:.1f}m height={h:+.1f}m "
                      f"→ blocked by {hit_path}")
        else:
            # No hit — ray reached the door without obstruction
            valid.append(cand)

    print(f"[StaticCollect] Occlusion check: {len(valid)}/{len(candidates)} positions clear, "
          f"{blocked} blocked by walls")
    return valid

CAMERA_CONFIGS = _check_line_of_sight(ALL_CANDIDATES)

if not CAMERA_CONFIGS:
    print("ERROR: all camera positions are occluded — check door center and arc offsets")
    simulation_app.close()
    sys.exit(1)

# ---------------------------------------------------------------------------
# Camera array — one Camera prim per valid position
# ---------------------------------------------------------------------------

cameras = []
for cfg in CAMERA_CONFIGS:
    cam = Camera(
        prim_path=cfg["prim_path"],
        position=cfg["pos"],
        frequency=30,
        resolution=CAM_RESOLUTION,
        orientation=cfg["ori"],
    )
    cameras.append(cam)

simulation_app.update()

for cam, cfg in zip(cameras, CAMERA_CONFIGS):
    cam.initialize()
    usd_cam = UsdGeom.Camera(stage.GetPrimAtPath(cfg["prim_path"]))
    usd_cam.GetFocalLengthAttr().Set(5.0)
    usd_cam.GetClippingRangeAttr().Set((0.01, 10000.0))

print(f"\n[StaticCollect] {len(cameras)} cameras created after occlusion filtering")
print(f"[StaticCollect] Door prim: {DOOR_PATH}")
print(f"[StaticCollect] Hinge: ({HINGE_X_CM}, {HINGE_Y_CM}) cm")
print(f"[StaticCollect] Camera center: ({DOOR_CENTER_XY[0]}, {DOOR_CENTER_XY[1]}) m, "
      f"Z_base={CAMERA_Z_BASE} m")
print(f"[StaticCollect] Arc offsets: {CAM_ARC_OFFSETS}°")
print(f"[StaticCollect] Distances: {CAM_DISTANCES} m")
print(f"[StaticCollect] Height offsets: {CAM_HEIGHT_OFFS} m")
print(f"[StaticCollect] Lighting randomization: {'ON' if LIGHTING_RAND else 'OFF'}")
print(f"[StaticCollect] Grid mode: {'ON' if GRID_MODE else 'OFF'}  "
      f"material-variation: {'ON' if MATERIAL_VAR else 'OFF'}  "
      f"camera-jitter: {'ON' if CAMERA_JITTER else 'OFF'}  "
      f"baseline-only: {'ON' if BASELINE_ONLY else 'OFF'}")

_ts("Warming up renderer (20 steps) ...")
for _ in range(20):
    world.step(render=True)
_ts("Renderer warmup done")

# ---------------------------------------------------------------------------
# Auto-discover child meshes under the door prim and cache xformOp:transform
# ---------------------------------------------------------------------------

door_prim = stage.GetPrimAtPath(DOOR_PATH)
if not door_prim.IsValid():
    print(f"ERROR: door prim not found: {DOOR_PATH}")
    simulation_app.close()
    sys.exit(1)

_mesh_ops = []  # list of (mesh_info_dict, UsdGeom.XformOp)
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
        print(f"[WARN] Skipping child {child.GetPath()} — no xformOp:transform")
        continue
    # Read original translation from the 4×4 matrix (row 3 = translation in USD row-vector convention)
    mat = op.Get()
    mesh_info = {
        "path": str(child.GetPath()),
        "ox": mat[3][0],  # translation X (cm)
        "oy": mat[3][1],  # translation Y (cm)
        "oz": mat[3][2],  # translation Z (cm)
    }
    _mesh_ops.append((mesh_info, op))
    print(f"[StaticCollect] Mesh discovered: {child.GetPath().name}  "
          f"origin=({mesh_info['ox']:.2f}, {mesh_info['oy']:.2f}, {mesh_info['oz']:.2f}) cm")

if not _mesh_ops:
    print(f"ERROR: no child meshes with xformOp:transform found under {DOOR_PATH}")
    simulation_app.close()
    sys.exit(1)

print(f"[StaticCollect] {len(_mesh_ops)} child meshes discovered")

# ---------------------------------------------------------------------------
# Cache original material bindings on each child mesh so we can restore them
# for the "original" cell of the material grid.
# ---------------------------------------------------------------------------

_original_mat_bindings = []  # list of (mesh_prim, bound_material_prim_or_None)
for child in door_prim.GetChildren():
    if not child.IsA(UsdGeom.Mesh):
        continue
    try:
        binding_api = UsdShade.MaterialBindingAPI(child)
        bound = binding_api.ComputeBoundMaterial()[0]
        bound_prim = bound.GetPrim() if bound else None
        _original_mat_bindings.append((child, bound_prim))
    except Exception as e:
        print(f"[WARN] Could not cache material for {child.GetPath()}: {e}")
        _original_mat_bindings.append((child, None))

print(f"[StaticCollect] Cached original material bindings for "
      f"{sum(1 for _, m in _original_mat_bindings if m is not None)}/{len(_original_mat_bindings)} child meshes")


def _restore_original_door_materials():
    """Restore each child mesh's material to its original (pre-override) binding."""
    for mesh_prim, orig_mat_prim in _original_mat_bindings:
        try:
            binding_api = UsdShade.MaterialBindingAPI(mesh_prim)
            if orig_mat_prim is not None and orig_mat_prim.IsValid():
                binding_api.Bind(UsdShade.Material(orig_mat_prim))
            else:
                binding_api.UnbindAllBindings()
        except Exception:
            pass

# ---------------------------------------------------------------------------
# Door rotation — direct 4×4 matrix rewrite
# ---------------------------------------------------------------------------

def _door_matrix(ox_cm: float, oy_cm: float, oz_cm: float, angle_deg: float) -> Gf.Matrix4d:
    """
    4×4 transform matrix for the given mesh origin rotated by angle_deg
    around (HINGE_X_CM, HINGE_Y_CM) in USD row-vector convention.
    """
    θ   = math.radians(angle_deg)
    c   = math.cos(θ)
    s   = math.sin(θ)
    dx  = ox_cm - HINGE_X_CM
    dy  = oy_cm - HINGE_Y_CM
    tx  = c * dx - s * dy + HINGE_X_CM
    ty  = s * dx + c * dy + HINGE_Y_CM
    tz  = oz_cm  # Z (height) is unchanged
    return Gf.Matrix4d(
         c,  s, 0, 0,   # row 0 — X basis rotated
        -s,  c, 0, 0,   # row 1 — Y basis rotated
         0,  0, 1, 0,   # row 2 — Z basis unchanged
        tx, ty, tz, 1,  # row 3 — translation
    )


def set_door_angle(angle_deg: float):
    """Rewrite all child mesh matrices for the given door angle."""
    for m, op in _mesh_ops:
        op.Set(_door_matrix(m["ox"], m["oy"], m["oz"], angle_deg))


# ---------------------------------------------------------------------------
# Lighting randomization helpers
# ---------------------------------------------------------------------------

def _color_temp_to_rgb(kelvin: float) -> Gf.Vec3f:
    """
    Approximate color temperature (K) to normalized RGB.
    Based on Tanner Helland's algorithm.
    """
    temp = kelvin / 100.0
    # Red
    if temp <= 66:
        r = 255
    else:
        r = 329.698727446 * ((temp - 60) ** -0.1332047592)
        r = max(0, min(255, r))
    # Green
    if temp <= 66:
        g = 99.4708025861 * math.log(temp) - 161.1195681661
    else:
        g = 288.1221695283 * ((temp - 60) ** -0.0755148492)
    g = max(0, min(255, g))
    # Blue
    if temp >= 66:
        b = 255
    elif temp <= 19:
        b = 0
    else:
        b = 138.5177312231 * math.log(temp - 10) - 305.0447927307
        b = max(0, min(255, b))
    return Gf.Vec3f(r / 255.0, g / 255.0, b / 255.0)


def randomize_lighting():
    """Randomize DomeLight and FillLight for the current episode (legacy random mode)."""
    dome_intensity = random.uniform(*DOME_INTENSITY_RANGE)
    dome_temp_k = random.uniform(*DOME_COLOR_TEMP_RANGE)
    fill_intensity = random.uniform(*FILL_INTENSITY_RANGE)

    dome.GetIntensityAttr().Set(dome_intensity)
    dome.GetColorAttr().Set(_color_temp_to_rgb(dome_temp_k))
    fill.GetIntensityAttr().Set(fill_intensity)

    return dome_intensity, dome_temp_k, fill_intensity


# ---------------------------------------------------------------------------
# Grid mode helpers — apply discrete lighting and material presets
# ---------------------------------------------------------------------------

def apply_lighting_condition(condition: dict):
    """
    Apply a discrete lighting preset from LIGHTING_CONDITIONS, with small
    jitter (±10% intensity, ±200 K temperature) within the cell to prevent
    pixel-level memorization while keeping each condition recognizably distinct.

    If the preset specifies an "hdri" path, the DomeLight's texture file is
    set to that HDR environment map (image-based lighting). Otherwise the
    texture is cleared and the DomeLight falls back to its flat color.
    """
    jitter_i = lambda v: v * random.uniform(0.90, 1.10)
    jitter_t = lambda v: v + random.uniform(-200, 200)

    dome_int  = jitter_i(condition["dome_int"])
    dome_temp = jitter_t(condition["dome_temp"])
    fill_int  = jitter_i(condition["fill_int"])
    fill_temp = jitter_t(condition["fill_temp"])

    dome.GetIntensityAttr().Set(dome_int)
    dome.GetColorAttr().Set(_color_temp_to_rgb(dome_temp))
    fill.GetIntensityAttr().Set(fill_int)
    fill.GetColorAttr().Set(_color_temp_to_rgb(fill_temp))

    # HDRI swap — when the preset has an HDR path, enable image-based lighting;
    # otherwise clear the texture so the DomeLight uses its flat color.
    hdri = condition.get("hdri")
    if hdri and os.path.isfile(hdri):
        _dome_tex_attr.Set(Sdf.AssetPath(hdri))
    else:
        _dome_tex_attr.Set(Sdf.AssetPath(""))

    return dome_int, dome_temp, fill_int, fill_temp


def apply_door_material(material_spec: dict):
    """
    Apply a discrete door material from DOOR_MATERIALS with small per-episode
    jitter (±0.05 color, ±0.08 roughness/metallic). If material_spec["color"]
    is None ("original" cell), restore the cached original material bindings.
    """
    if material_spec["color"] is None:
        # "Original" cell — restore cached bindings (populated at startup)
        _restore_original_door_materials()
        return None

    jc = lambda c: max(0.0, min(1.0, c + random.uniform(-0.05, 0.05)))
    jf = lambda f: max(0.0, min(1.0, f + random.uniform(-0.08, 0.08)))

    r, g, b = material_spec["color"]
    _door_shader.GetInput("diffuse_color_constant").Set(Gf.Vec3f(jc(r), jc(g), jc(b)))
    _door_shader.GetInput("reflection_roughness_constant").Set(jf(material_spec["roughness"]))
    _door_shader.GetInput("metallic_constant").Set(jf(material_spec["metallic"]))

    # Bind the override material to every child mesh of the door
    for child in door_prim.GetChildren():
        if child.IsA(UsdGeom.Mesh):
            UsdShade.MaterialBindingAPI.Apply(child)
            UsdShade.MaterialBindingAPI(child).Bind(_door_mat)

    return (material_spec["name"],)


def apply_camera_jitter():
    """
    Per-episode position and orientation jitter for all cameras:
      ±0.15 m XY, ±0.10 m Z, ±2° pitch, ±1.5° roll.
    Yaw is kept at the grid-configured value so the door stays in frame.
    """
    for cam_obj, cfg in zip(cameras, CAMERA_CONFIGS):
        base_pos = cfg["pos"]
        jittered_pos = np.array([
            base_pos[0] + random.uniform(-0.15, 0.15),
            base_pos[1] + random.uniform(-0.15, 0.15),
            base_pos[2] + random.uniform(-0.10, 0.10),
        ])
        pitch = random.uniform(-2.0, 2.0)
        roll  = random.uniform(-1.5, 1.5)
        base_yaw = cfg["offset_deg"] + 180.0
        jittered_ori = rot_utils.euler_angles_to_quats(
            np.array([roll, pitch, base_yaw]), degrees=True
        )
        try:
            cam_obj.set_world_pose(position=jittered_pos, orientation=jittered_ori)
        except Exception:
            # Fall back to direct xform write on the camera prim
            cam_prim = stage.GetPrimAtPath(cfg["prim_path"])
            UsdGeom.Xformable(cam_prim).MakeMatrixXform().Set(
                Gf.Matrix4d().SetTranslate(Gf.Vec3d(*jittered_pos.tolist()))
            )


# ---------------------------------------------------------------------------
# Verify at 0°, 45°, 90° — save sanity JPEGs before the full run
# ---------------------------------------------------------------------------

sanity_dir = os.path.join(_project_dir, "data", "door_static_sanity")
os.makedirs(sanity_dir, exist_ok=True)

try:
    from PIL import Image as _PIL
    _pil_ok = True
except ImportError:
    _pil_ok = False

def _save_sanity(tag: str):
    for _ in range(8):
        world.step(render=True)
    # Save from a subset of cameras (first, middle, last) to avoid flooding disk
    sample_indices = [0, len(cameras) // 2, len(cameras) - 1]
    sample_indices = sorted(set(i for i in sample_indices if i < len(cameras)))
    for i in sample_indices:
        cam_obj = cameras[i]
        cfg = CAMERA_CONFIGS[i]
        a = cfg["offset_deg"]
        d = cfg["distance"]
        h = cfg["height_off"]
        label = f"a{a:+.0f}_d{d:.1f}_h{h:+.1f}".replace("+", "p").replace("-", "n").replace(".", "p")
        rgba = cam_obj.get_rgba()
        if rgba is None:
            print(f"  [WARN] cam {label} returned None — skipping sanity {tag}")
            continue
        if _pil_ok:
            path = os.path.join(sanity_dir, f"{tag}_{label}.jpg")
            _PIL.fromarray(rgba[:, :, :3].astype(np.uint8)).save(path, "JPEG", quality=90)
        print(f"[StaticCollect] Sanity frame saved: {tag}_{label}.jpg")

if not GRID_MODE:
    _ts("Saving legacy sanity frames at 0°, 45°, 90° ...")
    for deg in [0.0, 45.0, 90.0]:
        set_door_angle(deg)
        simulation_app.update()
        _save_sanity(f"sanity_{int(deg)}deg")

    # Also verify bbox still in correct location after 0° reset
    set_door_angle(0.0)
    simulation_app.update()
    _bc = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_],
                             useExtentsHint=False)
    _bb = _bc.ComputeWorldBound(door_prim).ComputeAlignedRange()
    _ct = _bb.GetMidpoint()
    print(f"[StaticCollect] Bbox center after 0° reset: "
          f"({_ct[0]:.4f}, {_ct[1]:.4f}, {_ct[2]:.4f})")
else:
    _ts("Grid mode — skipping legacy 0/45/90 sanity block (grid dumps its own JPEGs)")

# ---------------------------------------------------------------------------
# Main collection loop
# ---------------------------------------------------------------------------

all_frames:      list[np.ndarray] = []
all_labels:      list[float]      = []
all_angles:      list[float]      = []
all_episodes:    list[int]        = []
all_cam_offsets: list[int]        = []
all_light_idx:   list[int]        = []   # -1 when not in grid mode
all_material_idx:list[int]        = []   # -1 when not in grid mode

n_cams = len(cameras)
swing_str = "FLIPPED (negative angles)" if FLIP_SWING else "NORMAL (positive angles)"
print(f"\n[StaticCollect] Swing direction: {swing_str}")

# JPEG dump directory (enabled by --dump-jpegs)
_jpeg_dir = None
if DUMP_JPEGS:
    _jpeg_dir = os.path.join(_project_dir, "data", f"{OUTPUT_NAME}_jpegs")
    os.makedirs(_jpeg_dir, exist_ok=True)
    print(f"[StaticCollect] JPEG dump enabled → {_jpeg_dir}")
    if not _pil_ok:
        print("[WARN] PIL not available — JPEG dump will be skipped")


def _capture_frame(episode_idx, angle_deg, light_idx, material_idx,
                   light_name=None, material_name=None, frame_idx=0):
    """Step once, read RGBA from every camera, and append to buffers.
    If DUMP_JPEGS is enabled, also writes each frame to disk with a
    descriptive filename encoding episode/light/material/angle/camera/frame."""
    world.step(render=True)
    captured = 0
    for cam_obj, cfg in zip(cameras, CAMERA_CONFIGS):
        rgba = cam_obj.get_rgba()
        if rgba is None:
            continue
        frame = rgba[:, :, :3].astype(np.uint8)
        all_frames.append(frame)
        all_labels.append(angle_deg / 90.0)
        all_angles.append(angle_deg)
        all_episodes.append(episode_idx)
        all_cam_offsets.append(int(cfg["offset_deg"]))
        all_light_idx.append(light_idx)
        all_material_idx.append(material_idx)
        captured += 1

        if DUMP_JPEGS and _pil_ok and _jpeg_dir is not None:
            cls = "CLOSED" if angle_deg <= CLOSED_MAX_DEG else "OPEN"
            a_lab = f"a{int(cfg['offset_deg']):+d}".replace("+", "p").replace("-", "n")
            fname = (f"ep{episode_idx:04d}_{cls}_ang{angle_deg:05.1f}"
                     f"_L-{light_name or 'NA'}_M-{material_name or 'NA'}"
                     f"_cam{a_lab}_f{frame_idx}.jpg")
            try:
                _PIL.fromarray(frame).save(os.path.join(_jpeg_dir, fname), "JPEG", quality=88)
            except Exception as e:
                print(f"[WARN] Failed to save JPEG {fname}: {e}")
    return captured


_ts("Starting main collection loop")

if GRID_MODE:
    # -----------------------------------------------------------------------
    # Grid search: lighting condition × material × door angle
    # -----------------------------------------------------------------------
    if BASELINE_ONLY:
        lightings = [LIGHTING_CONDITIONS[0]]  # bright_fluorescent
        materials = [DOOR_MATERIALS[-1]]       # "original"
        angles    = ANGLE_GRID
        print("[StaticCollect] GRID MODE (BASELINE ONLY): 1 lighting × 1 material "
              f"× {len(angles)} angles -- for generalization test doors")
    elif TEST_MODE:
        # Minimal subset for fast debugging. Pick one HDRI-enabled preset
        # (daylight_flood) and one flat-color preset (bright_fluorescent) so
        # the HDR swap is exercised in the sanity output.
        lightings = [LIGHTING_CONDITIONS[0], LIGHTING_CONDITIONS[1]]  # bright_fluorescent + daylight_flood
        materials = [DOOR_MATERIALS[0], DOOR_MATERIALS[1], DOOR_MATERIALS[-1]]  # aluminum, white, original
        angles    = [2.5, 45.0, 90.0]
        print("[StaticCollect] GRID MODE (TEST): 2 lightings × 3 materials × 3 angles "
              f"= {len(lightings)*len(materials)*len(angles)} cells")
    elif SANITY_GRID:
        # Visual audit of every lighting × material combo at 3 representative angles
        lightings = LIGHTING_CONDITIONS
        materials = DOOR_MATERIALS
        angles    = [1.0, 45.0, 90.0]
        print(f"[StaticCollect] GRID MODE (SANITY): {len(lightings)} lightings × "
              f"{len(materials)} materials × {len(angles)} angles "
              f"= {len(lightings)*len(materials)*len(angles)} cells")
    else:
        lightings = LIGHTING_CONDITIONS
        materials = DOOR_MATERIALS if MATERIAL_VAR else [DOOR_MATERIALS[-1]]
        angles    = ANGLE_GRID
        print(f"[StaticCollect] GRID MODE: {len(lightings)} lightings × {len(materials)} materials "
              f"× {len(angles)} angles = {len(lightings)*len(materials)*len(angles)} cells")
    print(f"[StaticCollect] Cameras: {n_cams}  frames/cell: {GRID_FRAMES}  "
          f"jitter: {'ON' if CAMERA_JITTER else 'OFF'}")

    episode = 0
    total_cells = len(lightings) * len(materials) * len(angles)
    for light_idx, lighting in enumerate(lightings):
        # Resolve global index into LIGHTING_CONDITIONS for metadata
        g_light_idx = LIGHTING_CONDITIONS.index(lighting)
        for mat_idx, material in enumerate(materials):
            g_mat_idx = DOOR_MATERIALS.index(material)
            for angle_deg in angles:
                # Apply lighting + material
                dome_int, dome_temp, fill_int, fill_temp = apply_lighting_condition(lighting)
                apply_door_material(material)

                # Set door angle (respect swing direction)
                swing_angle = -angle_deg if FLIP_SWING else angle_deg
                set_door_angle(swing_angle)

                # Camera jitter
                if CAMERA_JITTER:
                    apply_camera_jitter()

                # Warmup physics + renderer
                for _ in range(WARMUP_STEPS):
                    world.step(render=True)

                ep_captured = 0
                for frame_i in range(GRID_FRAMES):
                    ep_captured += _capture_frame(
                        episode, angle_deg, g_light_idx, g_mat_idx,
                        light_name=lighting["name"],
                        material_name=material["name"],
                        frame_idx=frame_i,
                    )

                cls = "CLOSED" if angle_deg <= CLOSED_MAX_DEG else "OPEN"
                print(f"[StaticCollect] Ep {episode+1:4d}/{total_cells}  "
                      f"[{cls:6s} {angle_deg:5.1f}°]  "
                      f"light={lighting['name']:<20s} mat={material['name']:<20s}  "
                      f"frames={ep_captured}")
                episode += 1

else:
    # -----------------------------------------------------------------------
    # Legacy random mode (backward compatible)
    # -----------------------------------------------------------------------
    print(f"[StaticCollect] Starting {N_EPISODES} episodes × {FRAMES_PER_EP} frames "
          f"× {n_cams} cameras  (total ≤ {N_EPISODES * FRAMES_PER_EP * n_cams} frames)")

    for episode in range(N_EPISODES):
        if episode % 2 == 0:
            angle_deg = random.uniform(0.0, CLOSED_MAX_DEG)
            cls = "CLOSED"
        else:
            open_episode = episode // 2
            band = open_episode % 3
            if band == 0:
                angle_deg = random.uniform(OPEN_MIN_DEG, 32.0)
            elif band == 1:
                angle_deg = random.uniform(32.0, 61.0)
            else:
                angle_deg = random.uniform(61.0, MAX_ANGLE_DEG)
            cls = "OPEN"

        swing_angle = -angle_deg if FLIP_SWING else angle_deg
        set_door_angle(swing_angle)

        if LIGHTING_RAND:
            dome_int, dome_temp, fill_int = randomize_lighting()
        else:
            dome_int, dome_temp, fill_int = 50000, 5500, 300000

        for _ in range(WARMUP_STEPS):
            world.step(render=True)

        ep_captured = 0
        for _ in range(FRAMES_PER_EP):
            ep_captured += _capture_frame(episode, angle_deg, -1, -1)

        light_str = (f"dome={dome_int:.0f} temp={dome_temp:.0f}K fill={fill_int:.0f}"
                     if LIGHTING_RAND else "fixed")
        print(f"[StaticCollect] Ep {episode+1:2d}/{N_EPISODES} "
              f"[{cls:6s} {angle_deg:5.1f}°]  frames={ep_captured}  light=({light_str})")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

data_dir = os.path.join(_project_dir, "data")
os.makedirs(data_dir, exist_ok=True)
out_path = os.path.join(data_dir, f"{OUTPUT_NAME}.npz")

print(f"\n[StaticCollect] Saving {len(all_frames)} frames → {out_path}")
np.savez_compressed(
    out_path,
    frames          = np.array(all_frames,      dtype=np.uint8),
    labels          = np.array(all_labels,      dtype=np.float32),
    angles          = np.array(all_angles,      dtype=np.float32),
    episode         = np.array(all_episodes,    dtype=np.int32),
    cam_offsets     = np.array(all_cam_offsets, dtype=np.int32),
    light_idx       = np.array(all_light_idx,   dtype=np.int32),
    material_idx    = np.array(all_material_idx,dtype=np.int32),
    light_names     = np.array([c["name"] for c in LIGHTING_CONDITIONS]),
    material_names  = np.array([m["name"] for m in DOOR_MATERIALS]),
)
size_mb   = os.path.getsize(out_path) // 1024 // 1024
angle_arr = np.array(all_angles, dtype=np.float32)
print(f"[StaticCollect] Saved ({size_mb} MB) — "
      f"CLOSED: {(angle_arr <= CLOSED_MAX_DEG).sum()}  "
      f"OPEN: {(angle_arr > CLOSED_MAX_DEG).sum()}  "
      f"cameras: {n_cams}")
print(f"[StaticCollect] Grid: {len(CAM_ARC_OFFSETS)} angles × {len(CAM_DISTANCES)} distances "
      f"× {len(CAM_HEIGHT_OFFS)} heights → {n_cams} after occlusion filter")

world.stop()
simulation_app.close()

#!/usr/bin/env python3
"""
Door State Data Collection — Fixed Camera (Script B)

A stationary camera 2 m east of Doors_13606713 captures the door at
programmatically set angles. No robot, no navigation — the only variable
between episodes is the door angle, giving clean CLOSED/OPEN labels.

Door rotation strategy — direct 4×4 matrix override on each child mesh:
  The child meshes (AL_375207, Glass_GL_1_18439) each have a single
  xformOp:transform that is a 4×4 identity-rotation + absolute-position
  matrix (from the Revit export). Modifying parent xformOpOrder proved
  unreliable with sublayer USD composition. Instead we directly rewrite
  the 4×4 matrix of each child mesh to encode the desired rotation around
  the south-edge hinge, keeping the rotation and translation tightly coupled
  with no extra ops to compose incorrectly.

  For a hinge at (hx, hy) cm and original mesh origin at (ox, oy, oz) cm:
    tx = cos(θ)*(ox-hx) - sin(θ)*(oy-hy) + hx
    ty = sin(θ)*(ox-hx) + cos(θ)*(oy-hy) + hy
    M  = | cos  sin  0  0 |   (USD row-vector convention)
         |-sin  cos  0  0 |
         |  0    0   1  0 |
         |  tx  ty  tz  1 |

Hinge values from diagnostic run 238045 (bbox at 0°):
  hinge Y-min = 79.6907 m  × 100 = 7969.07 cm  (south edge)
  mesh origin X from xformOp:transform = -2047.728 cm

Output: data/door_labels_static.npz
  frames   (N, H, W, 3) uint8 RGB
  labels   (N,) float32  angle_deg / 90.0
  angles   (N,) float32  degrees
  episode  (N,) int32

Usage:
  sbatch scripts/slurm/submit_isaac_job.sh simulations/robot/door_static_collection.py
"""

import os
import sys
import math
import random
import numpy as np

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

import omni.usd
from pxr import Usd, UsdLux, UsdGeom, Gf, Sdf

import isaacsim.core.utils.numpy.rotations as rot_utils
from isaacsim.core.api import World
from isaacsim.sensors.camera import Camera

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DOOR_PATH = (
    "/World/Building/_22_129___MSOE_Engineering_Building___A___2_25_2026"
    "/Geometry/Doors/RS___Curtain_Wall___Type_FG___New/New/Doors_13606713"
)

# Child mesh paths and their original xformOp:transform translations (cm).
# Values read directly from hierarchy.txt (door_rotation_diagnostic run).
DOOR_MESHES = [
    {
        "path": DOOR_PATH + "/AL_375207",
        "ox": -2047.728271484375,
        "oy":  8014.9365234375,
        "oz":  2677.16015625,
    },
    {
        "path": DOOR_PATH + "/Glass_GL_1_18439",
        "ox": -2047.728271484375,
        "oy":  8014.9365234375,
        "oz":  2702.559814453125,
    },
]

# Hinge: south Y-edge of door at 0° (bbox Y-min × 100), X matches mesh origin.
HINGE_X_CM = -2047.728271484375
HINGE_Y_CM =  7969.07   # world Y-min 79.6907 m × 100

# Camera arc: 5 cameras on a 2 m radius arc around the door center at the same Z.
# All cameras face the door center; the centre camera (0°) replicates the original
# single-camera position exactly.
#
#   DOOR_CENTER_XY  — midpoint of the door bbox in world XY (from bbox diagnostic)
#   CAMERA_RADIUS   — metres from door centre to each camera
#   CAMERA_Z        — world Z for all cameras (same as original)
#   CAM_ARC_OFFSETS — angular offsets in degrees from +X (east) in world XY
#                     0° = original east position; negative = south, positive = north
#
# Camera orientation: each camera yaws to face the door centre.
# In Isaac Sim, euler [0, 0, yaw] with yaw=180° means facing west (-X).
# General formula: world_yaw = offset_deg + 180  (points back toward door centre).
DOOR_CENTER_XY  = np.array([-20.48, 80.15])   # world XY of door centre
CAMERA_RADIUS   = 2.0                           # metres
CAMERA_Z        = 27.8                          # same Z as original camera
CAM_ARC_OFFSETS = [-30, -15, 0]                # degrees; 0 = original camera position
                                                # positive offsets removed — wall geometry to the
                                                # north blocks those positions
CAM_RESOLUTION  = (640, 480)

def _make_camera_configs():
    configs = []
    for offset_deg in CAM_ARC_OFFSETS:
        θ       = math.radians(offset_deg)
        cam_pos = np.array([
            DOOR_CENTER_XY[0] + CAMERA_RADIUS * math.cos(θ),
            DOOR_CENTER_XY[1] + CAMERA_RADIUS * math.sin(θ),
            CAMERA_Z,
        ])
        yaw_deg = offset_deg + 180.0  # face back toward door centre
        cam_ori = rot_utils.euler_angles_to_quats(
            np.array([0.0, 0.0, yaw_deg]), degrees=True
        )
        configs.append({
            "prim_path":  f"/World/StaticCam_{offset_deg:+d}".replace("+", "p").replace("-", "n"),
            "pos":        cam_pos,
            "ori":        cam_ori,
            "offset_deg": offset_deg,
        })
    return configs

CAMERA_CONFIGS = _make_camera_configs()

# Episode / capture config
N_EPISODES    = 60    # 30 CLOSED + 30 OPEN (alternating)
FRAMES_PER_EP = 10    # frames per episode — reduced since all 10 are near-identical;
                      # more episodes = more unique angles = more diversity
WARMUP_STEPS  = 40    # render steps after angle change before first capture

CLOSED_MAX_DEG = 5.0
OPEN_MIN_DEG   = 6.0
MAX_ANGLE_DEG  = 90.0

_script_dir  = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.abspath(os.path.join(_script_dir, "..", ".."))

# ---------------------------------------------------------------------------
# Scene setup
# ---------------------------------------------------------------------------

building_usd = os.path.join(_project_dir, "innovation_center_finalV1.usd")
if not os.path.isfile(building_usd):
    print(f"ERROR: building USD not found: {building_usd}")
    simulation_app.close()
    sys.exit(1)

world = World(stage_units_in_meters=1.0)
stage = omni.usd.get_context().get_stage()
stage.GetRootLayer().subLayerPaths.insert(0, building_usd)

dome = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/DomeLight"))
dome.CreateIntensityAttr(50000)

# Fill light positioned above the centre camera (0° offset = original position)
_centre_cfg = CAMERA_CONFIGS[len(CAMERA_CONFIGS) // 2]
fill = UsdLux.SphereLight.Define(stage, Sdf.Path("/World/FillLight"))
fill.CreateIntensityAttr(300000)
fill.CreateRadiusAttr(0.1)
UsdGeom.Xformable(fill.GetPrim()).AddTranslateOp().Set(
    (float(_centre_cfg["pos"][0]), float(_centre_cfg["pos"][1]), float(CAMERA_Z) + 1.5)
)

# ---------------------------------------------------------------------------
# Camera array — one Camera prim per arc position
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

world.reset()
simulation_app.update()

for cam, cfg in zip(cameras, CAMERA_CONFIGS):
    cam.initialize()
    usd_cam = UsdGeom.Camera(stage.GetPrimAtPath(cfg["prim_path"]))
    usd_cam.GetFocalLengthAttr().Set(5.0)
    usd_cam.GetClippingRangeAttr().Set((0.01, 10000.0))

print(f"[StaticCollect] {len(cameras)} cameras initialised at arc offsets {CAM_ARC_OFFSETS}°")

print("[StaticCollect] Warming up renderer (20 steps) ...")
for _ in range(20):
    world.step(render=True)

# ---------------------------------------------------------------------------
# Confirm prims and cache the xformOp:transform op for each mesh
# ---------------------------------------------------------------------------

_mesh_ops = []  # list of (mesh_info_dict, UsdGeom.XformOp) for the transform op
for m in DOOR_MESHES:
    prim = stage.GetPrimAtPath(m["path"])
    if not prim.IsValid():
        print(f"ERROR: mesh prim not found: {m['path']}")
        simulation_app.close()
        sys.exit(1)
    op = None
    for xop in UsdGeom.Xformable(prim).GetOrderedXformOps():
        if xop.GetOpName() == "xformOp:transform":
            op = xop
            break
    if op is None:
        print(f"ERROR: no xformOp:transform on {m['path']}")
        simulation_app.close()
        sys.exit(1)
    _mesh_ops.append((m, op))
    print(f"[StaticCollect] Mesh op cached: {prim.GetPath().name}")

# ---------------------------------------------------------------------------
# Door rotation — direct 4×4 matrix rewrite
#
# Builds the composed transform matrix for rotating the mesh around the hinge
# and sets it directly on xformOp:transform. No extra ops added, no parent
# xform changes, no sublayer composition surprises.
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
    """Rewrite both mesh matrices for the given door angle."""
    for m, op in _mesh_ops:
        op.Set(_door_matrix(m["ox"], m["oy"], m["oz"], angle_deg))


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
    # Save one JPEG per camera so every viewpoint can be verified visually.
    # Files are named: sanity_0deg_cam_n30.jpg, sanity_0deg_cam_p0.jpg, etc.
    for cam_obj, cfg in zip(cameras, CAMERA_CONFIGS):
        sign  = "p" if cfg["offset_deg"] >= 0 else "n"
        label = f"{sign}{abs(cfg['offset_deg'])}"
        rgba  = cam_obj.get_rgba()
        if rgba is None:
            print(f"  [WARN] cam_{label} returned None — skipping sanity {tag}")
            continue
        if _pil_ok:
            path = os.path.join(sanity_dir, f"{tag}_cam_{label}.jpg")
            _PIL.fromarray(rgba[:, :, :3].astype(np.uint8)).save(path, "JPEG", quality=90)
        print(f"[StaticCollect] Sanity frame saved: {tag}_cam_{label}.jpg")

print("[StaticCollect] Saving sanity frames at 0°, 45°, 90° (all cameras) ...")
for deg in [0.0, 45.0, 90.0]:
    set_door_angle(deg)
    simulation_app.update()
    _save_sanity(f"sanity_{int(deg)}deg")

# Also verify bbox still in correct location after 0° reset
set_door_angle(0.0)
simulation_app.update()
door_prim = stage.GetPrimAtPath(DOOR_PATH)
_bc = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_],
                         useExtentsHint=False)
_bb = _bc.ComputeWorldBound(door_prim).ComputeAlignedRange()
_ct = _bb.GetMidpoint()
print(f"[StaticCollect] Bbox center after 0° reset: "
      f"({_ct[0]:.4f}, {_ct[1]:.4f}, {_ct[2]:.4f})  expected ~(-20.48, 80.15, 28.03)")

# ---------------------------------------------------------------------------
# Main collection loop
# ---------------------------------------------------------------------------

all_frames:      list[np.ndarray] = []
all_labels:      list[float]      = []
all_angles:      list[float]      = []
all_episodes:    list[int]        = []
all_cam_offsets: list[int]        = []   # which arc camera captured each frame

n_cams = len(cameras)
print(f"\n[StaticCollect] Starting {N_EPISODES} episodes × {FRAMES_PER_EP} frames "
      f"× {n_cams} cameras  (total ≤ {N_EPISODES * FRAMES_PER_EP * n_cams} frames)")

for episode in range(N_EPISODES):
    if episode % 2 == 0:
        angle_deg = random.uniform(0.0, CLOSED_MAX_DEG)
        cls = "CLOSED"
    else:
        # Stratified OPEN sampling: divide 6–90° into 3 equal bands and
        # cycle through them so no band is underrepresented.
        open_episode = episode // 2   # 0, 1, 2, ...
        band = open_episode % 3       # 0→low, 1→mid, 2→high
        if band == 0:
            angle_deg = random.uniform(OPEN_MIN_DEG, 32.0)   #  6–32° (barely open)
        elif band == 1:
            angle_deg = random.uniform(32.0, 61.0)            # 32–61° (mid open)
        else:
            angle_deg = random.uniform(61.0, MAX_ANGLE_DEG)   # 61–90° (wide open)
        cls = "OPEN"

    set_door_angle(angle_deg)

    for _ in range(WARMUP_STEPS):
        world.step(render=True)

    ep_captured = 0
    for _ in range(FRAMES_PER_EP):
        world.step(render=True)
        for cam_obj, cfg in zip(cameras, CAMERA_CONFIGS):
            rgba = cam_obj.get_rgba()
            if rgba is None:
                continue
            all_frames.append(rgba[:, :, :3].astype(np.uint8))
            all_labels.append(angle_deg / 90.0)
            all_angles.append(angle_deg)
            all_episodes.append(episode)
            all_cam_offsets.append(cfg["offset_deg"])
            ep_captured += 1

    print(f"[StaticCollect] Ep {episode+1:2d}/{N_EPISODES} "
          f"[{cls:6s} {angle_deg:5.1f}°]  frames={ep_captured}")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

data_dir = os.path.join(_project_dir, "data")
os.makedirs(data_dir, exist_ok=True)
out_path = os.path.join(data_dir, "door_labels_static_multi_camera.npz")

print(f"\n[StaticCollect] Saving {len(all_frames)} frames → {out_path}")
np.savez_compressed(
    out_path,
    frames      = np.array(all_frames,      dtype=np.uint8),
    labels      = np.array(all_labels,      dtype=np.float32),
    angles      = np.array(all_angles,      dtype=np.float32),
    episode     = np.array(all_episodes,    dtype=np.int32),
    cam_offsets = np.array(all_cam_offsets, dtype=np.int32),   # arc angle per frame
)
size_mb   = os.path.getsize(out_path) // 1024 // 1024
angle_arr = np.array(all_angles, dtype=np.float32)
print(f"[StaticCollect] Saved ({size_mb} MB) — "
      f"CLOSED: {(angle_arr <= CLOSED_MAX_DEG).sum()}  "
      f"OPEN: {(angle_arr > CLOSED_MAX_DEG).sum()}  "
      f"cameras: {CAM_ARC_OFFSETS}")

world.stop()
simulation_app.close()

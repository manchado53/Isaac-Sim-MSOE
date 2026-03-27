#!/usr/bin/env python3
"""
Nova Carter — Door State Data Collection  (Phase 1: Simulation)

Generates (frame, door_angle_label) pairs for VLM fine-tuning.
At each episode start, each door joint is randomized to [0°, 90°].
Robot spawns 2 m in front of Doors_13606713 and remains stationary.
Frames are captured every 2 steps for the full episode.

After N_EPISODES, saves data/door_labels.npz with keys:
  frames    — (N, H, W, 3) uint8 RGB images
  labels    — (N,) float32 normalized angle [0.0 closed … 1.0 open]
  door_ids  — (N,) int32 index into DOOR_PRIMS list
  frame_ids — (N,) int32 simulation step number

Usage:
  sbatch submit_isaac_job.sh simulations/robot/nova_carter_door_collection.py

IMPORTANT: Fill in DOOR_PRIMS below with paths from floor4_door_diagnostic.py
before running. The script will fail loudly if DOOR_PRIMS is empty.
"""

import math
import os
import sys
import random
import json
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Config — EDIT THESE before running
# ---------------------------------------------------------------------------

# Door prim paths from floor4_door_diagnostic.py (job 237551).
# These are the 7 unique door instance Xform prims on Floor 4.
# Using the parent Xform (e.g. Doors_13606713) rather than individual mesh
# children so the revolute joint rotates the entire door assembly.
# All are type RS___Curtain_Wall___Type_FG___New (glass curtain wall doors).
# Positions (X, Y, Z): see summary in code comment below.
#   Doors_1913690    X=-5.85  Y=54.21  Z=26.77  (south exterior)
#   Doors_4190126    X=-41.41 Y=85.81  Z=26.77  (north exterior)
#   Doors_12112322   X=-48.39 Y=78.80  Z=26.77  (far west exterior)
#   Doors_12112444   X=-48.39 Y=79.76  Z=26.77  (far west exterior)
#   Doors_13185454   X=-41.41 Y=82.60  Z=26.77  (north exterior)
#   Doors_13606713   X=-20.48 Y=80.15  Z=26.77  (CLOSEST to robot spawn)
#   Doors_13935342   X=-5.88  Y=55.17  Z=26.77  (south exterior)
_DOOR_BASE = (
    "/World/Building/_22_129___MSOE_Engineering_Building___A___2_25_2026"
    "/Geometry/Doors/RS___Curtain_Wall___Type_FG___New/New"
)
DOOR_PRIMS: list[str] = [
    f"{_DOOR_BASE}/Doors_1913690",
    f"{_DOOR_BASE}/Doors_4190126",
    f"{_DOOR_BASE}/Doors_12112322",
    f"{_DOOR_BASE}/Doors_12112444",
    f"{_DOOR_BASE}/Doors_13185454",
    f"{_DOOR_BASE}/Doors_13606713",
    f"{_DOOR_BASE}/Doors_13935342",
]

N_EPISODES       = 3       # quick test — increase to 30 for full dataset
MAX_FRAMES       = 30      # steps per episode — robot is stationary, 30 steps is enough for a test
CAM_RESOLUTION   = (640, 480)
FPS              = 15
TRIGGER_RADIUS   = 8.0     # m — capture frame when robot is within this of a door
MIN_ANGLE_DEG    = 0.0     # door randomization range (degrees)
MAX_ANGLE_DEG    = 90.0
CLOSED_MAX_DEG   = 5.0     # angles ≤ 5° → CLOSED class (matches VLM_fine-tuning.md threshold)

# POC spawn — 2 m east of Doors_13606713 (-20.48, 80.15), facing the door directly.
# Robot sits still the entire episode; no navigation required.
# node_71 (≈-20.3, 78.8) is the nearest nav node and is in Door 5's proximity
# set, so the trigger fires immediately from step 0.
ROBOT_SPAWN_XY   = (-18.48, 80.15)   # 2 m east of Doors_13606713
ROBOT_SPAWN_Z    = 27.3              # Floor 4 slab Z=26.8 + 0.5 m clearance
# Quaternion to face -X (west): 180° around Z from default +X facing.
# Isaac Sim uses (w, x, y, z) convention.  q = (cos90°, 0, 0, sin90°) = (0, 0, 0, 1)
ROBOT_SPAWN_QUAT = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)  # w,x,y,z → face -X (door)

# ---------------------------------------------------------------------------
# Nav graph — load for proximity-aware door triggering and map viz
# ---------------------------------------------------------------------------
# The nav graph uses the same XY coordinate system as the USD stage (meters).
# Coordinates are stored as world_xy_cm (centimeters) — divide by 100 for meters.
#
# Door proximity nodes (within TRIGGER_RADIUS=8m):
#   Doors_13606713 (X=-20.48, Y=80.15):  node_36 (1.7m), node_38 (5.1m), node_39 (3.8m)
#   Doors_13185454 (X=-41.41, Y=82.60):  node_37 (3.2m), node_43 (7.6m)
#   Doors_4190126  (X=-41.41, Y=85.81):  node_37 (5.9m), node_43 (4.7m)
#   All other doors: no nav nodes within 8m → prediction suppressed (correct)
#
# USD XY positions matching DOOR_PRIMS order (from diagnostic job 237551):
DOOR_POSITIONS_M = np.array([
    [-5.85,  54.21],   # 0: Doors_1913690    (south exterior — no nav nodes nearby)
    [-41.41, 85.81],   # 1: Doors_4190126    (north exterior — node_37, node_43)
    [-48.39, 78.80],   # 2: Doors_12112322   (far west — no nav nodes nearby)
    [-48.39, 79.76],   # 3: Doors_12112444   (far west — no nav nodes nearby)
    [-41.41, 82.60],   # 4: Doors_13185454   (north exterior — node_37, node_43)
    [-20.48, 80.15],   # 5: Doors_13606713   (TARGET — spawn at node_41, drive north)
    [-5.88,  55.17],   # 6: Doors_13935342   (south exterior — no nav nodes nearby)
], dtype=np.float32)

_NAV_JSON = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "floor_4_nav_graph_manual.json"
)
_nav_loaded     = False
_nav_node_ids   = []
_nav_node_xy    = None   # (N, 2) float32, meters
_door_proxy_nodes: "list[set[str]]" = []  # one set per entry in DOOR_PRIMS

try:
    with open(_NAV_JSON) as _nf:
        _nav = json.load(_nf)
    _nav_node_ids = [n["id"] for n in _nav["nodes"]]
    _nav_node_xy  = np.array(
        [[n["world_xy_cm"][0] / 100.0, n["world_xy_cm"][1] / 100.0]
         for n in _nav["nodes"]], dtype=np.float32,
    )
    for door_pos in DOOR_POSITIONS_M:
        dists = np.linalg.norm(_nav_node_xy - door_pos, axis=1)
        proximate = {_nav_node_ids[i] for i, d in enumerate(dists) if d <= TRIGGER_RADIUS}
        _door_proxy_nodes.append(proximate)
    _nav_loaded = True
    print(f"[DoorCollect] Nav graph loaded: {len(_nav_node_ids)} nodes, "
          f"{sum(len(s) for s in _door_proxy_nodes)} total proximity mappings")
    for i, (dpos, proxy) in enumerate(zip(DOOR_POSITIONS_M, _door_proxy_nodes)):
        print(f"  Door {i} @ ({dpos[0]:.1f},{dpos[1]:.1f}): {len(proxy)} proximate nodes {sorted(proxy)}")
except FileNotFoundError:
    print(f"[DoorCollect] WARN: Nav graph not found at {_NAV_JSON} — "
          "falling back to Euclidean distance trigger.")

# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if not DOOR_PRIMS:
    print("ERROR: DOOR_PRIMS is empty.")
    print("  1. Run the diagnostic:  sbatch submit_isaac_job.sh "
          "simulations/robot/floor4_door_diagnostic.py")
    print("  2. Copy the DOOR_PRIMS list from the log output into this script.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Phase 1: Isaac Sim
# ---------------------------------------------------------------------------

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

import carb
import omni.usd
from pxr import Sdf, UsdLux, UsdGeom, UsdPhysics, Gf

import isaacsim.core.utils.numpy.rotations as rot_utils
from scipy.spatial.transform import Rotation as R
from isaacsim.core.api import World
from isaacsim.robot.wheeled_robots.controllers.differential_controller import DifferentialController
from isaacsim.robot.wheeled_robots.robots import WheeledRobot
from isaacsim.sensors.camera import Camera
from isaacsim.storage.native import get_assets_root_path

_script_dir  = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.abspath(os.path.join(_script_dir, "..", ".."))

# ---------------------------------------------------------------------------
# Scene setup
# ---------------------------------------------------------------------------

assets_root = get_assets_root_path()
if assets_root is None:
    carb.log_error("Could not find Isaac Sim assets root")
    simulation_app.close()
    sys.exit(1)

world = World(stage_units_in_meters=1.0)

_building_usd = os.path.join(_project_dir, "innovation_center_finalV1.usd")
if not os.path.isfile(_building_usd):
    print(f"ERROR: Building USD not found at {_building_usd}")
    simulation_app.close()
    sys.exit(1)

print(f"[DoorCollect] Loading: {_building_usd}")
stage = omni.usd.get_context().get_stage()
stage.GetRootLayer().subLayerPaths.insert(0, _building_usd)

# Dome + fill lights (same as vlm_demo)
dome = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/DomeLight"))
dome.CreateIntensityAttr(50000)
_fill_light = UsdLux.SphereLight.Define(stage, Sdf.Path("/World/RobotFillLight"))
_fill_light.CreateIntensityAttr(200000)
_fill_light.CreateRadiusAttr(0.1)
_fill_light_xform = UsdGeom.Xformable(_fill_light.GetPrim())

# Ground planes
world.scene.add_default_ground_plane()
world.scene.add_default_ground_plane(
    z_position=26.75,
    name="floor4_ground_plane",
    prim_path="/World/floor4GroundPlane",
    static_friction=0.3,
    dynamic_friction=0.3,
    restitution=0.0,
)

# ---------------------------------------------------------------------------
# Helper: add a RevoluteJoint to a door mesh prim
# ---------------------------------------------------------------------------

def add_door_joint(door_prim_path: str, joint_prim_path: str) -> "UsdPhysics.RevoluteJoint | None":
    """
    Add a UsdPhysics.RevoluteJoint to a door Xform prim so it can be rotated.

    Uses a D6/Revolute joint with body0=None (world-anchored fixed frame) and
    body1=door prim.  The door prim gets RigidBodyAPI so physics can drive it.
    Returns the joint prim, or None if the door prim doesn't exist.
    """
    door_prim = stage.GetPrimAtPath(door_prim_path)
    if not door_prim.IsValid():
        print(f"  [WARN] Door prim not found: {door_prim_path}")
        return None

    # Apply RigidBodyAPI to the door Xform so physics can move it.
    # kinematicEnabled=True means we drive position/orientation manually via
    # the joint target rather than letting gravity affect it freely.
    rigid = UsdPhysics.RigidBodyAPI.Apply(door_prim)
    rigid.CreateKinematicEnabledAttr(True)

    # Create the revolute joint — body0 is left empty (world anchor),
    # body1 is the door prim.
    joint = UsdPhysics.RevoluteJoint.Define(stage, Sdf.Path(joint_prim_path))
    joint.CreateAxisAttr("Z")          # vertical hinge (up axis = Z in this USD)
    joint.CreateLowerLimitAttr(0.0)    # 0° = closed
    joint.CreateUpperLimitAttr(90.0)   # 90° = fully open
    # body0 intentionally empty → world-anchored hinge
    joint.CreateBody1Rel().SetTargets([Sdf.Path(door_prim_path)])
    joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

    print(f"  [DoorCollect] Joint added: {joint_prim_path} → {door_prim_path}")
    return joint


# ---------------------------------------------------------------------------
# Add joints to all door prims
# ---------------------------------------------------------------------------

print(f"[DoorCollect] Adding revolute joints to {len(DOOR_PRIMS)} door prims...")
door_joints: list["UsdPhysics.RevoluteJoint | None"] = []
for i, dp in enumerate(DOOR_PRIMS):
    joint_path = f"/World/DoorJoint_{i:03d}"
    joint = add_door_joint(dp, joint_path)
    door_joints.append(joint)

valid_doors        = [(dp, jt) for dp, jt in zip(DOOR_PRIMS, door_joints) if jt is not None]
valid_door_indices = [i for i, jt in enumerate(door_joints) if jt is not None]
print(f"[DoorCollect] {len(valid_doors)}/{len(DOOR_PRIMS)} joints created successfully.")

if not valid_doors:
    print("ERROR: No valid door joints created. Check DOOR_PRIMS paths against the USD.")
    simulation_app.close()
    sys.exit(1)

# ---------------------------------------------------------------------------
# Robot
# ---------------------------------------------------------------------------

carter = world.scene.add(
    WheeledRobot(
        prim_path="/World/NovaCarter",
        name="nova_carter",
        wheel_dof_names=["joint_wheel_left", "joint_wheel_right"],
        create_robot=True,
        usd_path=assets_root + "/Isaac/Robots/NVIDIA/NovaCarter/nova_carter.usd",
        position=np.array([ROBOT_SPAWN_XY[0], ROBOT_SPAWN_XY[1], ROBOT_SPAWN_Z]),
        orientation=ROBOT_SPAWN_QUAT,
    )
)

controller = DifferentialController(name="ctrl", wheel_radius=0.14, wheel_base=0.4)

# ---------------------------------------------------------------------------
# Onboard camera
# ---------------------------------------------------------------------------

W, H = CAM_RESOLUTION
ONBOARD_OFFSET = np.array([0.20, 0.0, 0.65])

onboard_cam = Camera(
    prim_path="/World/OnboardCam",
    position=np.array([0.36, 0.0, 0.5]),
    frequency=FPS,
    resolution=(W, H),
    orientation=rot_utils.euler_angles_to_quats(np.array([0, 0, 0]), degrees=True),
)

world.reset()
simulation_app.update()
onboard_cam.initialize()

usd_cam = UsdGeom.Camera(stage.GetPrimAtPath("/World/OnboardCam"))
usd_cam.GetFocalLengthAttr().Set(5.0)
usd_cam.GetClippingRangeAttr().Set((0.01, 10000.0))

# Renderer warmup
print("[DoorCollect] Warming up renderer (10 steps)...")
for _ in range(10):
    world.step(render=True)

# ---------------------------------------------------------------------------
# Navigation sequence — POC: robot is stationary the entire episode.
# Spawned 2 m east of Doors_13606713, facing west (-X) directly at the door.

nav_sequence = [
    (0, MAX_FRAMES, [0.0, 0.0]),   # stationary — no driving needed
]

def get_command(step: int) -> list:
    for start, end, cmd in nav_sequence:
        if start <= step < end:
            return cmd
    return [0.0, 0.0]

# ---------------------------------------------------------------------------
# Dataset accumulation
# ---------------------------------------------------------------------------

all_frames:     list[np.ndarray]  = []   # RGB uint8
all_labels:     list[float]       = []   # [0, 1]
all_door_ids:   list[int]         = []   # index into DOOR_PRIMS
all_frame_ids:  list[int]         = []   # step number
all_robot_traj: list[np.ndarray]  = []   # (N_EPISODES, MAX_FRAMES, 2) — for map viz

def get_door_xy(door_prim_path: str) -> "np.ndarray | None":
    """Return world XY of a door prim (cached in prim transform)."""
    prim = stage.GetPrimAtPath(door_prim_path)
    if not prim.IsValid():
        return None
    xform_cache = UsdGeom.XformCache(0)
    world_xform = xform_cache.GetLocalToWorldTransform(prim)
    t = world_xform.ExtractTranslation()
    return np.array([t[0], t[1]])

# Pre-compute door XY positions for proximity checks
door_positions = [get_door_xy(dp) for dp, _ in valid_doors]

def _get_door_leaf_meshes(door_prim) -> "list":
    """
    Return all Mesh children of a door prim (the full door leaf).

    Diagnostic job 238038 confirmed that Revit-exported children use absolute
    world-space xformOp:transform with reset xform stacks, so the parent Xform
    rotation has no effect.  AL (aluminum frame) and Glass children share the
    same hinge X/Y in their transforms, so rotating both by the same Z angle
    swings the complete door leaf together.
    """
    return [c for c in door_prim.GetChildren() if c.IsA(UsdGeom.Mesh)]


# ---------------------------------------------------------------------------
# Pivot-corrected door rotation helpers
# ---------------------------------------------------------------------------
# Problem: appending rotateZ after the mesh's xformOp:transform rotates around
# the mesh's local origin (panel centre), not the hinge edge.
# Fix: update xformOp:transform directly with a pivot-corrected matrix:
#
#   M_new = T(-hinge) * Rz(θ) * T(+hinge) * T_world
#         (row-vector USD convention — derivation in door_rotation_diagnostic.py Part 5)
#
# Translation row of M_new:
#   new_tx = tx + hx*(1 - cos θ) + hy*sin θ
#   new_ty = ty - hx*sin θ      + hy*(1 - cos θ)

_mesh_base_transforms: dict = {}   # USD path → Gf.Matrix4d (original, 0° transform)
_door_hinge_pivots:    dict = {}   # door prim path → (hx, hy) in local mesh space


def _get_base_transform(mesh_prim) -> "Gf.Matrix4d":
    """Return (and cache) the original xformOp:transform for a Mesh prim."""
    path = str(mesh_prim.GetPath())
    if path not in _mesh_base_transforms:
        for op in UsdGeom.Xformable(mesh_prim).GetOrderedXformOps():
            if op.GetOpName() == "xformOp:transform":
                _mesh_base_transforms[path] = op.Get()
                return _mesh_base_transforms[path]
        _mesh_base_transforms[path] = Gf.Matrix4d(1)  # identity fallback
    return _mesh_base_transforms[path]


def _get_hinge_pivot_local(door_prim) -> "tuple[float, float]":
    """
    Return (hx, hy) — the hinge pivot in the LOCAL space of the door's Mesh children.

    1. Tries physics:localPos1 / physics:localPos0 from the HingeJoint child prim.
    2. Falls back to the min-Y edge of the first Mesh child's vertex bounding box.

    NOTE: After reviewing data/door_rot_diagnostic4/ pivot-test images, update the
    fallback edge direction (min-Y / max-Y / min-X / max-X) to whichever keeps one
    side of the door fixed.
    """
    for child in door_prim.GetChildren():
        if child.GetTypeName() == "PhysicsRevoluteJoint":
            for attr_name in ("physics:localPos1", "physics:localPos0"):
                attr = child.GetAttribute(attr_name)
                if attr and attr.IsAuthored():
                    v = attr.Get()
                    if v is not None and (abs(v[0]) > 0.001 or abs(v[1]) > 0.001):
                        print(f"  [DoorCollect] Hinge pivot from HingeJoint.{attr_name}: "
                              f"({v[0]:.4f}, {v[1]:.4f})")
                        return float(v[0]), float(v[1])

    # Fallback: vertex bbox
    meshes = [c for c in door_prim.GetChildren() if c.IsA(UsdGeom.Mesh)]
    if meshes:
        pts_attr = meshes[0].GetAttribute("points")
        if pts_attr and pts_attr.IsAuthored():
            pts = pts_attr.Get()
            if pts:
                ys = [p[1] for p in pts]
                hinge_y = min(ys)   # assume hinge at min-Y edge — adjust after diagnostic
                print(f"  [DoorCollect] Hinge pivot fallback (bbox min-Y): "
                      f"(0.0, {hinge_y:.4f})")
                return 0.0, float(hinge_y)

    print("  [DoorCollect] WARN: Could not determine hinge pivot — using center (0,0)")
    return 0.0, 0.0


def set_door_angles(angles_deg: list[float]):
    """
    Rotate each door's Mesh children to the given angle using pivot-corrected rotation.

    Updates xformOp:transform directly (no extra rotateZ ops) so the hinge edge
    stays fixed and the free edge swings open realistically.
    For angle=0 the original base transform is restored exactly.
    """
    for i, (dp, _jt) in enumerate(valid_doors):
        angle = angles_deg[i]
        door_prim = stage.GetPrimAtPath(dp)
        if not door_prim.IsValid():
            continue

        leaf_meshes = _get_door_leaf_meshes(door_prim)
        if not leaf_meshes:
            print(f"  [WARN] No Mesh children found under {dp} — skipping rotation")
            continue

        if dp not in _door_hinge_pivots:
            _door_hinge_pivots[dp] = _get_hinge_pivot_local(door_prim)
        hx, hy = _door_hinge_pivots[dp]

        cos_a = math.cos(math.radians(angle))
        sin_a = math.sin(math.radians(angle))

        for mesh_prim in leaf_meshes:
            M0 = _get_base_transform(mesh_prim)
            tx, ty, tz = M0[3][0], M0[3][1], M0[3][2]

            new_tx = tx + hx * (1.0 - cos_a) + hy * sin_a
            new_ty = ty - hx * sin_a          + hy * (1.0 - cos_a)

            new_M = Gf.Matrix4d()
            new_M.SetRow(0, Gf.Vec4d( cos_a,  sin_a, 0.0, 0.0))
            new_M.SetRow(1, Gf.Vec4d(-sin_a,  cos_a, 0.0, 0.0))
            new_M.SetRow(2, Gf.Vec4d(  0.0,    0.0,  1.0, 0.0))
            new_M.SetRow(3, Gf.Vec4d(new_tx, new_ty,  tz, 1.0))

            for op in UsdGeom.Xformable(mesh_prim).GetOrderedXformOps():
                if op.GetOpName() == "xformOp:transform":
                    op.Set(new_M)
                    break


# ---------------------------------------------------------------------------
# Main episode loop
# ---------------------------------------------------------------------------

_data_dir    = os.path.join(_project_dir, "data")
_preview_dir = os.path.join(_data_dir, "preview_frames")
os.makedirs(_preview_dir, exist_ok=True)
print(f"[DoorCollect] Preview frames → {_preview_dir}")

print(f"\n[DoorCollect] Starting {N_EPISODES} episodes × {MAX_FRAMES} steps...")
print(f"  Doors: {len(valid_doors)}, trigger radius: {TRIGGER_RADIUS} m")

for episode in range(N_EPISODES):
    # Stratified angle sampling: alternate CLOSED / OPEN episodes to ensure 50/50 class balance.
    # Uniform random over [0°, 90°] would yield only ~5.6% CLOSED frames (5/90),
    # making the fine-tuning dataset heavily imbalanced toward OPEN.
    if episode % 2 == 0:
        # CLOSED episode: all doors in [0°, 5°]
        ep_angles = [random.uniform(0.0, CLOSED_MAX_DEG) for _ in valid_doors]
    else:
        # OPEN episode: all doors in (6°, 90°]
        ep_angles = [random.uniform(CLOSED_MAX_DEG + 1.0, MAX_ANGLE_DEG) for _ in valid_doors]
    ep_labels = [a / 90.0 for a in ep_angles]

    print(f"\n[DoorCollect] Episode {episode+1}/{N_EPISODES}")
    for i, (a, l) in enumerate(zip(ep_angles, ep_labels)):
        print(f"  Door {i}: {a:.1f}° → label {l:.3f}")

    # Teleport robot to spawn without touching the simulation state.
    #
    # Two approaches that DON'T work in Isaac Sim 5.x:
    #   world.reset()       — re-initializes Camera objects → get_rgba() returns None
    #   world.stop()/play() — destroys the ArticulationView → apply_wheel_actions crashes
    #
    # Correct multi-episode pattern: keep the world PLAYING the entire run.
    # Teleport the robot each episode and let physics re-settle during warmup.
    # Camera and ArticulationView remain valid throughout.
    carter.set_world_pose(
        position=np.array([ROBOT_SPAWN_XY[0], ROBOT_SPAWN_XY[1], ROBOT_SPAWN_Z]),
        orientation=ROBOT_SPAWN_QUAT,
    )
    set_door_angles(ep_angles)

    # Warmup: 50 steps lets physics settle after teleport and fills the camera buffer.
    for _ in range(50):
        world.step(render=True)

    episode_captures = 0
    ep_traj = np.full((MAX_FRAMES, 2), np.nan, dtype=np.float32)  # for map viz

    for step in range(MAX_FRAMES):
        if world.is_playing():
            pos, quat = carter.get_world_pose()
            ep_traj[step] = pos[:2]

            # Periodic diagnostic: robot position + nearest nav node + trigger state
            if step % 100 == 0:
                if _nav_loaded:
                    _nd = np.linalg.norm(_nav_node_xy - pos[:2], axis=1)
                    _nid = _nav_node_ids[int(np.argmin(_nd))]
                    _near = any(_nid in s for s in _door_proxy_nodes if s)
                    print(f"[DoorCollect] Ep{episode+1} step {step:4d}: "
                          f"XY=({pos[0]:.1f},{pos[1]:.1f}) Z={pos[2]:.2f}  "
                          f"nearest={_nid}  in_trigger_zone={_near}")
                else:
                    print(f"[DoorCollect] Ep{episode+1} step {step:4d}: "
                          f"XY=({pos[0]:.1f},{pos[1]:.1f}) Z={pos[2]:.2f}")

            rot = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
            onboard_cam.set_world_pose(
                position=pos + rot.apply(ONBOARD_OFFSET),
                orientation=quat,
            )

            # Move fill light with robot
            light_pos = pos + np.array([0.0, 0.0, 1.0])
            _fill_light_xform.ClearXformOpOrder()
            _fill_light_xform.AddTranslateOp().Set(
                (float(light_pos[0]), float(light_pos[1]), float(light_pos[2])))

        world.step(render=True)

        if world.is_playing():
            carter.apply_wheel_actions(controller.forward(command=get_command(step)))

        # Nav-graph-aware proximity trigger:
        # Only capture a frame when the robot's nearest nav node is within
        # TRIGGER_RADIUS of a door. This prevents false triggers through walls
        # and suppresses prediction for doors the nav graph cannot reach.
        if world.is_playing() and step % 2 == 0:
            ob = onboard_cam.get_rgba()
            if ob is None:
                continue

            pos2d = pos[:2]

            # Find nearest nav node (once per frame, shared across all doors)
            if _nav_loaded:
                node_dists = np.linalg.norm(_nav_node_xy - pos2d, axis=1)
                nearest_nid = _nav_node_ids[int(np.argmin(node_dists))]

            for vi, (door_prim_idx, (dpos, label)) in enumerate(
                    zip(valid_door_indices, zip(door_positions, ep_labels))):
                if dpos is None:
                    continue

                # Determine proximity using nav graph (preferred) or Euclidean fallback
                if _nav_loaded and door_prim_idx < len(_door_proxy_nodes):
                    near_door = nearest_nid in _door_proxy_nodes[door_prim_idx]
                else:
                    near_door = float(np.linalg.norm(pos2d - dpos)) <= TRIGGER_RADIUS

                if near_door:
                    frame_rgb = ob[:, :, :3].astype(np.uint8)
                    all_frames.append(frame_rgb)
                    all_labels.append(label)
                    all_door_ids.append(door_prim_idx)
                    all_frame_ids.append(step + episode * MAX_FRAMES)
                    episode_captures += 1

                    # Save preview JPEG immediately so frames appear in the log
                    # before the job finishes. Filename encodes episode, step, and label.
                    _angle_deg = label * 90.0
                    _fname = (f"ep{episode+1:02d}_step{step:04d}"
                              f"_door{door_prim_idx}_angle{_angle_deg:.1f}deg.jpg")
                    Image.fromarray(frame_rgb).save(
                        os.path.join(_preview_dir, _fname), "JPEG", quality=90
                    )
                    print(f"[DoorCollect] FRAME saved: {_fname}")

    all_robot_traj.append(ep_traj)
    print(f"[DoorCollect] Episode {episode+1} done — {episode_captures} frames captured "
          f"(total so far: {len(all_frames)})")
    if episode_captures == 0:
        print(f"  [WARN] 0 frames — possible causes: robot didn't reach trigger zone, "
              f"camera still None, or nav proxy sets empty. Check position log above.")

# ---------------------------------------------------------------------------
# Save dataset
# ---------------------------------------------------------------------------

os.makedirs(_data_dir, exist_ok=True)
out_path = os.path.join(_data_dir, "door_labels.npz")

print(f"\n[DoorCollect] Saving {len(all_frames)} samples to {out_path} ...")
np.savez_compressed(
    out_path,
    frames      = np.array(all_frames,    dtype=np.uint8),
    labels      = np.array(all_labels,    dtype=np.float32),
    door_ids    = np.array(all_door_ids,  dtype=np.int32),
    frame_ids   = np.array(all_frame_ids, dtype=np.int32),
    # Robot XY trajectory for floor4_map_tracker.py visualization.
    # Shape: (N_EPISODES, MAX_FRAMES, 2) — NaN where robot wasn't playing.
    robot_traj  = np.array(all_robot_traj, dtype=np.float32),
    # Door ground-truth positions (USD meters) for map overlay.
    door_pos_m  = DOOR_POSITIONS_M,
)
size_mb = os.path.getsize(out_path) // 1024 // 1024
print(f"[DoorCollect] Saved ({size_mb} MB).")

label_arr = np.array(all_labels, dtype=np.float32)
if label_arr.size > 0:
    print(f"[DoorCollect] Label stats: "
          f"min={label_arr.min():.3f}  max={label_arr.max():.3f}  "
          f"mean={label_arr.mean():.3f}  std={label_arr.std():.3f}")
else:
    print("[DoorCollect] Label stats: no frames captured — check position log above.")

world.stop()
simulation_app.close()
# os._exit(0) is called internally — nothing below here runs.

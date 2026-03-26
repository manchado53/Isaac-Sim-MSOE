#!/usr/bin/env python3
"""
Nova Carter — VLM Patrol Demo  (Phase 1: Simulation)

Drives Nova Carter through the MSOE Innovation Center and collects onboard
+ follow-cam RGBA frames.  Saves them to a .npz cache file for Phase 2+3.

Run via submit_isaac_job.sh which calls this script then nova_carter_vlm_phase23.py.

Tuning notes:
  ROBOT_SPAWN_Z  — currently Floor 4 (Z≈26.8 m slab + 0.5 m clearance = 27.3).
                   If robot falls through: increase to 28.0 or 29.0.
  ROBOT_SPAWN_XY — building centroid confirmed across all floors via USD diagnostic.
"""

import os
import sys
import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MAX_FRAMES     = 1440  # 96 s of sim time at 30 Hz physics (doubled for Floor 4 friction)
FPS            = 15    # output video frame rate
VLM_INTERVAL   = 15    # send every Nth collected frame to VLM (~2 s intervals)
CAM_RESOLUTION = (640, 480)

# Building coordinates from USD mesh diagnostic (IQR centroid of floor-level prims).
# Building footprint: X -41.4 to 1.9 (43 m), Y 57.7 to 82.6 (25 m) in USD space.
# Floor slabs: Z=13.6 (Floor 1), Z=18.2 (Floor 2), Z=22.5 (Floor 3), Z=26.8 (Floor 4).
# Target: Floor 4. Spawn 0.5 m above Floor 4 slab.
ROBOT_SPAWN_XY = (-19.7, 70.1)
ROBOT_SPAWN_Z  = 27.3   # Floor 4 slab Z=26.8 + 0.5 m clearance

# ---------------------------------------------------------------------------
# Phase 1: Isaac Sim
# SimulationApp must be the very first isaacsim/omni import.
# NOTE: simulation_app.close() calls os._exit(0) internally, so any code
#       that must run after the sim lives in nova_carter_vlm_phase23.py.
# ---------------------------------------------------------------------------

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

import carb
import omni.usd
from pxr import Sdf, UsdLux, UsdGeom

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

print(f"[Phase 1] Loading: {_building_usd}")
# Use sublayer insertion instead of add_reference_to_stage — the Revit USD has
# no defaultPrim set, so reference loading silently loads nothing.
# Sublayer merges all prims directly into the stage without needing a defaultPrim.
stage = omni.usd.get_context().get_stage()
stage.GetRootLayer().subLayerPaths.insert(0, _building_usd)

# Exterior sky light — very high intensity to bleed through windows/openings
dome = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/DomeLight"))
dome.CreateIntensityAttr(50000)

# Interior fill light — travels with robot to illuminate onboard camera view
_fill_light = UsdLux.SphereLight.Define(stage, Sdf.Path("/World/RobotFillLight"))
_fill_light.CreateIntensityAttr(200000)
_fill_light.CreateRadiusAttr(0.1)
_fill_light_xform = UsdGeom.Xformable(_fill_light.GetPrim())

# Follow-cam fill light — travels with follow camera (7+ m from robot fill light,
# so needs its own source; inverse-square falloff makes the robot light ~60x dimmer
# at follow-cam distance)
_follow_light = UsdLux.SphereLight.Define(stage, Sdf.Path("/World/FollowFillLight"))
_follow_light.CreateIntensityAttr(600000)
_follow_light.CreateRadiusAttr(0.1)
_follow_light_xform = UsdGeom.Xformable(_follow_light.GetPrim())

# Ground plane at world origin (always present as a safety net)
world.scene.add_default_ground_plane()

# Floor 4 fallback ground plane — the default plane is at Z=0 and will NOT catch
# the robot at Floor 4 (Z≈26.8). This physics plane at Z=26.75 acts as a floor.
# If the USD floor slab has collision geometry, this plane is redundant but harmless.
world.scene.add_default_ground_plane(
    z_position=26.75,
    name="floor4_ground_plane",
    prim_path="/World/floor4GroundPlane",
    static_friction=0.3,
    dynamic_friction=0.3,
    restitution=0.0,
)

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
    )
)

controller = DifferentialController(name="ctrl", wheel_radius=0.14, wheel_base=0.4)

# ---------------------------------------------------------------------------
# Cameras
# ---------------------------------------------------------------------------

ONBOARD_OFFSET = np.array([0.20, 0.0, 0.65])
# Follow cam: 6 m ahead of robot, 1.5 m up, facing back (yaw 180°, pitch 15° down).
# Camera is in front of the robot looking back — shows robot approaching with the
# atrium/glass entrance behind it. Avoids the atrium tree cluster that sits behind the robot.
FOLLOW_OFFSET  = np.array([6.0, 0.0, 1.5])
FOLLOW_ORIENT  = rot_utils.euler_angles_to_quats(np.array([0, 8, 180]), degrees=True)

W, H = CAM_RESOLUTION

onboard_cam = Camera(
    prim_path="/World/OnboardCam",
    position=np.array([0.36, 0.0, 0.5]),
    frequency=FPS,
    resolution=(W, H),
    orientation=rot_utils.euler_angles_to_quats(np.array([0, 0, 0]), degrees=True),
)
follow_cam = Camera(
    prim_path="/World/FollowCam",
    position=FOLLOW_OFFSET,
    frequency=FPS,
    resolution=(W, H),
    orientation=FOLLOW_ORIENT,
)

world.reset()
simulation_app.update()   # flush renderer before camera init
onboard_cam.initialize()

# Wide FOV for onboard
usd_cam = UsdGeom.Camera(stage.GetPrimAtPath("/World/OnboardCam"))
usd_cam.GetFocalLengthAttr().Set(5.0)
usd_cam.GetClippingRangeAttr().Set((0.01, 10000.0))

follow_cam.initialize()

# Renderer warmup — camera returns None until the render pipeline is warm
print("[Phase 1] Warming up renderer (10 steps)...")
for _ in range(10):
    world.step(render=True)

# ---------------------------------------------------------------------------
# Navigation sequence
# ---------------------------------------------------------------------------

# Route stays within the 43 m (X) x 25 m (Y) building footprint.
# Actual robot speed on Floor 4 observed ~0.3 m/s (half of commanded 0.6 m/s due
# to floor friction). Doubled step counts to cover ~6 m forward and ~9 m return.
nav_sequence = [
    (0,    600, [0.6,  0.0       ]),   # forward along +X corridor (~6 m at 0.3 m/s)
    (600,  780, [0.0,  np.pi / 4 ]),   # 90° left turn
    (780,  900, [0.4,  0.0       ]),   # short cross-corridor jog (~1.2 m in Y)
    (900,  1080,[0.0,  np.pi / 4 ]),   # 90° left turn (now facing -X)
    (1080, 1440,[0.6,  0.0       ]),   # return sweep along -X corridor (~9 m)
]

def get_command(step):
    for start, end, cmd in nav_sequence:
        if start <= step < end:
            return cmd
    return [0.0, 0.0]

# ---------------------------------------------------------------------------
# Simulation loop
# ---------------------------------------------------------------------------

onboard_frames = []
follow_frames  = []
vlm_frame_idxs = []

print(f"[Phase 1] Simulating {MAX_FRAMES} steps...")

_pos_logged = False   # log robot Z on first valid step

for step in range(MAX_FRAMES):
    if world.is_playing():
        pos, quat = carter.get_world_pose()

        if not _pos_logged:
            print(f"[Phase 1] Robot spawn position: "
                  f"X={pos[0]:.2f}  Y={pos[1]:.2f}  Z={pos[2]:.2f}")
            print(f"[Phase 1] Expected Floor 4 slab Z≈26.8 — "
                  f"robot is {'ON Floor 4' if 26.0 <= pos[2] <= 30.0 else 'NOT on Floor 4 (check spawn Z)'}")
            _pos_logged = True

        rot = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
        onboard_cam.set_world_pose(
            position=pos + rot.apply(ONBOARD_OFFSET),
            orientation=quat,
        )
        follow_cam.set_world_pose(
            position=pos + FOLLOW_OFFSET,
            orientation=FOLLOW_ORIENT,
        )

        # Move fill light with robot (1 m above robot center)
        light_pos = pos + np.array([0.0, 0.0, 1.0])
        _fill_light_xform.ClearXformOpOrder()
        _fill_light_xform.AddTranslateOp().Set((float(light_pos[0]), float(light_pos[1]), float(light_pos[2])))

        # Follow fill light sits halfway between camera and robot, slightly above,
        # so it illuminates both the robot and the corridor walls ahead of it.
        follow_light_pos = pos + FOLLOW_OFFSET * 0.5 + np.array([0.0, 0.0, 1.5])
        _follow_light_xform.ClearXformOpOrder()
        _follow_light_xform.AddTranslateOp().Set((float(follow_light_pos[0]), float(follow_light_pos[1]), float(follow_light_pos[2])))

    world.step(render=True)

    if world.is_playing():
        carter.apply_wheel_actions(controller.forward(command=get_command(step)))

    if world.is_playing() and step % 2 == 0:
        ob = onboard_cam.get_rgba()
        fl = follow_cam.get_rgba()

        if ob is not None:
            frame = np.clip(ob, 0, 255).astype(np.uint8)
            onboard_frames.append(frame)
            if len(onboard_frames) % VLM_INTERVAL == 0:
                vlm_frame_idxs.append(len(onboard_frames) - 1)

        if fl is not None:
            follow_frames.append(np.clip(fl, 0, 255).astype(np.uint8))

    if step % 120 == 0:
        _p, _ = carter.get_world_pose() if world.is_playing() else (np.array([0,0,0]), None)
        print(f"[Phase 1]   step {step}/{MAX_FRAMES}  "
              f"| frames: {len(onboard_frames)}  "
              f"| robot XYZ: ({_p[0]:.1f}, {_p[1]:.1f}, {_p[2]:.1f})")

print(f"[Phase 1] Done — {len(onboard_frames)} onboard, "
      f"{len(follow_frames)} follow, {len(vlm_frame_idxs)} queued for VLM.")

# ---------------------------------------------------------------------------
# Save frame cache for Phase 2+3
# simulation_app.close() calls os._exit(0), so Phase 2+3 runs as a separate
# process (nova_carter_vlm_phase23.py) invoked by submit_isaac_job.sh.
# ---------------------------------------------------------------------------

_cache_path = os.path.join(_project_dir, "frame_cache.npz")
print(f"[Phase 1] Saving frame cache to {_cache_path} ...")
np.savez_compressed(
    _cache_path,
    onboard_frames  = np.array(onboard_frames,  dtype=np.uint8),
    follow_frames   = np.array(follow_frames,   dtype=np.uint8),
    vlm_frame_idxs  = np.array(vlm_frame_idxs,  dtype=np.int32),
    config          = np.array([FPS, W, H]),
)
print(f"[Phase 1] Cache saved ({os.path.getsize(_cache_path) // 1024 // 1024} MB).")

world.stop()
simulation_app.close()
# os._exit(0) is called internally — nothing below here runs.

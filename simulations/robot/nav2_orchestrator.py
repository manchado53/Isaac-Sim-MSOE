#!/usr/bin/env python3
"""
Isaac Sim + Nav2 orchestration for building navigation.

Loads a building USD, spawns a Nova Carter at a configurable floor/spawn point,
and (in a follow-up commit) sets up the ROS2 bridge action graph so Nav2 can
drive the robot.

Usage:
    ./python.sh simulations/robot/nav2_orchestrator.py
    ./python.sh simulations/robot/nav2_orchestrator.py --building InnovationCenter --floor 4
    ./python.sh simulations/robot/nav2_orchestrator.py --floor 4 --node-id 77
    ./python.sh simulations/robot/nav2_orchestrator.py --floor 4 --yaw 1.5708
    ./python.sh simulations/robot/nav2_orchestrator.py --headless
    ./python.sh simulations/robot/nav2_orchestrator.py --stream
    ./python.sh simulations/robot/nav2_orchestrator.py --stream --stream-ip 192.168.1.42

After the script reports "ready", launch Nav2 in another terminal:
    ros2 launch kern_navigation kern_navigation.launch.py map:=<map_yaml_path>

NOTE: This commit implements the spawn-only path. ROS2 bridge wiring is TODO —
the robot will sit in place until that is added.
"""

import argparse
import os
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Spawn a Nova Carter in a building for Nav2 navigation."
    )
    parser.add_argument("--building", default="InnovationCenter",
                        help="Building directory name under assets/environments/")
    parser.add_argument("--floor", type=int, default=4,
                        help="Floor number to spawn on")
    parser.add_argument("--node-id", type=int, default=None,
                        help="Override default_spawn.node_id from building_info.json")
    parser.add_argument("--yaw", type=float, default=None,
                        help="Override spawn yaw (radians)")
    parser.add_argument("--headless", action="store_true",
                        help="Run Isaac Sim headless (no GUI, no streaming)")
    parser.add_argument("--stream", action="store_true",
                        help="Run Isaac Sim with WebRTC livestream enabled "
                             "(connect via Isaac Sim Streaming Client)")
    parser.add_argument("--stream-ip", default="100.107.199.57",
                        help="Public endpoint address advertised to remote clients "
                             "(default: team server Tailscale IP)")
    return parser.parse_args()


# Parse args before launching Isaac Sim so --help is fast
args = parse_args()

# Project root on sys.path so we can import sim_utils
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)

# Resolve building metadata BEFORE starting Isaac Sim (uses only stdlib).
# Import the module directly (not via `from sim_utils import ...`) so the
# package __init__ — which eagerly imports building_physics/pxr — doesn't run.
import importlib.util  # noqa: E402
_building_info_path = os.path.join(REPO_ROOT, "sim_utils", "building_info.py")
_spec = importlib.util.spec_from_file_location("building_info", _building_info_path)
building_info = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(building_info)

floor = building_info.get_floor(args.building, args.floor)

spawn_node_id = args.node_id if args.node_id is not None else floor["default_spawn"].get("node_id")
if spawn_node_id is not None:
    spawn_x, spawn_y = building_info.lookup_node(floor["nav_graph_json"], spawn_node_id)
else:
    spawn_x = floor["default_spawn"]["x"]
    spawn_y = floor["default_spawn"]["y"]
spawn_z = floor["floor_deck_z"]
spawn_yaw = args.yaw if args.yaw is not None else floor["default_spawn"]["yaw"]

print(f"[orchestrator] Building: {args.building} | Floor: {args.floor}")
print(f"[orchestrator] Spawn node id: {spawn_node_id}")
print(f"[orchestrator] Spawn position: ({spawn_x:.3f}, {spawn_y:.3f}, {spawn_z:.3f}) yaw={spawn_yaw:.3f} rad")
print(f"[orchestrator] USD: {floor['_usd_physics']}")
print(f"[orchestrator] Map YAML: {floor['map_metadata_yaml']}")

# -----------------------------------------------------------------------------
# Start Isaac Sim
# -----------------------------------------------------------------------------
os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

from isaacsim import SimulationApp  # noqa: E402

if args.stream and args.headless:
    print("[orchestrator] WARNING: --headless is ignored when --stream is set")

# Build launch_config. When streaming, match the official Isaac Sim
# livestream example at standalone_examples/api/isaacsim.simulation_app/livestream.py
# — CONFIG with explicit renderer + display_options is required so the RTX
# pipeline initializes correctly (otherwise the NVCF service waits forever
# for an rtx_ready signal and the health endpoint stays at "App not ready").
if args.stream:
    launch_config = {
        "width": 1280,
        "height": 720,
        "window_width": 1920,
        "window_height": 1080,
        "headless": True,
        "hide_ui": False,
        "renderer": "RaytracedLighting",
        "display_options": 3286,
    }
    print(f"[orchestrator] Streaming mode. Connect the WebRTC client to {args.stream_ip}")
else:
    launch_config = {"headless": args.headless}

simulation_app = SimulationApp(launch_config)

if args.stream:
    # Apply custom carb settings AFTER init, matching the official example's
    # pattern. These mirror the flags that isaac-sim.streaming.sh passes
    # for the exp.full.streaming.kit experience.
    simulation_app.set_setting("/app/window/drawMouse", True)
    simulation_app.set_setting("/app/livestream/allowResize", True)
    simulation_app.set_setting("/app/livestream/nvcf/quitOnSessionEnded", False)
    simulation_app.set_setting("/app/livestream/publicEndpointAddress", args.stream_ip)
    simulation_app.set_setting("/app/livestream/port", 49100)
    simulation_app.set_setting("/app/renderer/skipWhileMinimized", False)
    simulation_app.set_setting("/app/renderer/sleepMsOutOfFocus", 0)

    # Enable the livestream service. NVCF is what the Isaac Sim Streaming
    # Client connects to (see ~/isaacsim/apps/isaacsim.exp.full.streaming.kit).
    from isaacsim.core.utils.extensions import enable_extension  # noqa: E402
    enable_extension("omni.services.livestream.nvcf")
    print("[orchestrator] omni.services.livestream.nvcf enabled")

# -----------------------------------------------------------------------------
# Isaac Sim API imports (only valid after SimulationApp)
# -----------------------------------------------------------------------------
import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
from pxr import Sdf, UsdGeom  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.utils.viewports import set_camera_view  # noqa: E402
from isaacsim.robot.wheeled_robots.robots import WheeledRobot  # noqa: E402
from isaacsim.storage.native import get_assets_root_path  # noqa: E402
import isaacsim.core.utils.numpy.rotations as rot_utils  # noqa: E402

from sim_utils.building_physics import setup_physics_scene  # noqa: E402


def main():
    world = World(stage_units_in_meters=1.0)
    stage = omni.usd.get_context().get_stage()

    # Sanity-check the USD's units. V4 scans assume meters; CLAUDE.md warns that
    # MSOEBuildingTake2.usd uses feet. Abort early if they mismatch.
    mpu = UsdGeom.GetStageMetersPerUnit(stage)
    print(f"[orchestrator] Stage metersPerUnit = {mpu}")

    # Load building via AddReference (never use open_stage per CLAUDE.md)
    building_prim = stage.DefinePrim("/World/Building", "Xform")
    building_prim.GetReferences().AddReference(
        floor["_usd_physics"], primPath=Sdf.Path("/World/Building")
    )

    setup_physics_scene(stage)

    # Spawn Nova Carter
    assets_root = get_assets_root_path()
    if assets_root is None:
        raise RuntimeError("get_assets_root_path() returned None — Isaac Sim assets not configured")

    orientation = rot_utils.euler_angles_to_quats(
        np.array([0.0, 0.0, np.degrees(spawn_yaw)]), degrees=True
    )

    spawn_pos = np.array([spawn_x, spawn_y, spawn_z + 0.1])  # 10 cm above deck
    carter = world.scene.add(
        WheeledRobot(
            prim_path="/World/NovaCarter",
            name="nova_carter",
            wheel_dof_names=["joint_wheel_left", "joint_wheel_right"],
            create_robot=True,
            usd_path=assets_root + "/Isaac/Robots/NVIDIA/NovaCarter/nova_carter.usd",
            position=spawn_pos,
            orientation=orientation,
        )
    )

    world.reset()
    # Re-apply pose after reset in case the default sample zeros it
    carter.set_world_pose(position=spawn_pos, orientation=orientation)

    # Aim the viewport Perspective camera at the robot so the streamed
    # view opens on the spawn instead of the USD's default position.
    # Isometric angle: offset in -x/-y, 10m above the deck.
    cam_eye = np.array([spawn_x - 12.0, spawn_y - 12.0, spawn_z + 10.0])
    cam_target = np.array([spawn_x, spawn_y, spawn_z])
    set_camera_view(eye=cam_eye, target=cam_target)

    # TODO: set up ROS2 bridge action graph here (odom, lidar, tf, clock, cmd_vel).
    # Until that lands, the robot just sits on the floor.

    print("[orchestrator] Scene ready. Robot spawned.")
    print(f"[orchestrator] Launch Nav2 in another terminal once ROS2 bridge is wired:")
    print(f"  ros2 launch kern_navigation kern_navigation.launch.py map:={floor['map_metadata_yaml']}")

    # Run until the window is closed / process is killed
    while simulation_app.is_running():
        world.step(render=True)

    # Intentional: do not call simulation_app.close() — OmniGraph cleanup
    # segfaults on exit (see validate_robot_movement.py:216 and CLAUDE.md).


if __name__ == "__main__":
    main()

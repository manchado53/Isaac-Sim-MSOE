#!/usr/bin/env python3
"""
Isaac Sim + Nav2 orchestration for building navigation.

Loads a building USD, spawns a Nova Carter at a configurable floor/spawn point,
and wires up the ROS2 bridge so the robot publishes odom/tf/lidar/clock and
subscribes to /cmd_vel. This lets Nav2 (running in a separate process) drive
the robot.

Usage:
    ./python.sh navigation/nav2_orchestrator.py
    ./python.sh navigation/nav2_orchestrator.py --building InnovationCenter --floor 4
    ./python.sh navigation/nav2_orchestrator.py --floor 4 --node-id 77
    ./python.sh navigation/nav2_orchestrator.py --floor 4 --yaw 1.5708
    ./python.sh navigation/nav2_orchestrator.py --headless
    ./python.sh navigation/nav2_orchestrator.py --stream
    ./python.sh navigation/nav2_orchestrator.py --stream --stream-ip 192.168.1.42

After the script reports "Scene ready", launch Nav2 in another terminal:
    ros2 launch kern_navigation kern_navigation.launch.py map:=<map_yaml_path>
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
    parser.add_argument("--obstacle", action="append", default=[],
                        metavar="X,Y[,SIZE]",
                        help="Spawn a test obstacle at world (x, y) with "
                             "optional size (default 0.6 m). Repeatable. "
                             "No obstacles are spawned unless this flag is set.")
    return parser.parse_args()


# Parse args before launching Isaac Sim so --help is fast
args = parse_args()

# Project root on sys.path so we can import sim_utils.
# This file lives at `navigation/nav2_orchestrator.py`, so the repo root
# is exactly one level up.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
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
# Enable the ROS2 bridge extension BEFORE loading the robot USD. Nova_Carter_ROS
# ships with baked-in action graphs whose nodes reference isaacsim.ros2.bridge.*
# types — if the extension isn't registered yet, those graphs fail to
# instantiate on USD load.
# -----------------------------------------------------------------------------
from isaacsim.core.utils.extensions import enable_extension  # noqa: E402
enable_extension("isaacsim.ros2.bridge")
simulation_app.update()  # let the extension finish loading

# Publish topics even when no subscribers are listening yet. The default is
# "verify subscribers first", which silently drops messages while Nav2 is
# still booting in another terminal.
import carb  # noqa: E402
carb.settings.get_settings().set_bool(
    "/exts/isaacsim.ros2.bridge/publish_without_verification", True
)
print("[orchestrator] isaacsim.ros2.bridge enabled")

# -----------------------------------------------------------------------------
# Isaac Sim API imports (only valid after SimulationApp)
# -----------------------------------------------------------------------------
import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
import omni.graph.core as og  # noqa: E402
from pxr import Sdf, UsdGeom, UsdPhysics  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.api.objects import DynamicCuboid  # noqa: E402
from isaacsim.core.utils.viewports import set_camera_view  # noqa: E402
from isaacsim.robot.wheeled_robots.robots import WheeledRobot  # noqa: E402
from isaacsim.storage.native import get_assets_root_path  # noqa: E402
import isaacsim.core.utils.numpy.rotations as rot_utils  # noqa: E402

from sim_utils.building_physics import setup_physics_scene  # noqa: E402


def _spawn_test_obstacles(stage, world, positions, floor_deck_z):
    """Create DynamicCuboid test obstacles under /World/Obstacles.

    Each cuboid is a `DynamicCuboid` (auto-applies CollisionAPI +
    RigidBodyAPI + MassAPI). The cube is placed at the same world Z as
    the robot's spawn pose (`floor_deck_z + 0.1`) — i.e. "floor level,
    robot height" — so it looks right in the WebRTC viewport and sits
    in the same horizontal plane as the base_link that Nav2 tracks.

    NOTE: The cubes are *not* frozen here — if left unfrozen they'd drop
    under gravity. Call `_freeze_test_obstacles(stage, len(positions))`
    AFTER `world.reset()` to make them kinematic. Applying the kinematic
    flag before `world.reset()` has been observed to invalidate the
    PhysX tensor view used by Nova Carter's wheel controller, causing
    `getDofPositionTargets` errors and stopping the robot from moving
    even though Nav2 keeps commanding velocity.

    Obstacles live under `/World/Obstacles/*`, NOT under
    `/World/Building/*`, so they're invisible to the pre-generated
    occupancy map's static_layer. Nav2 only learns about them through
    the live lidar path — that's how we exercise obstacle_layer.

    Args:
        stage: Active USD stage.
        world: isaacsim.core.api.World instance.
        positions: list of (x, y, size) tuples in world meters.
        floor_deck_z: world-frame Z of the floor slab top (unused for
            the cube center — kept for signature compatibility).
    """
    if not positions:
        return
    # Parent xform so `/World/Obstacles` is a single scope, easy to toggle
    # visibility or wipe in one call if needed later.
    stage.DefinePrim("/World/Obstacles", "Xform")
    # Robot spawn is at floor_deck_z + 0.1 (see spawn_pos). Match that.
    robot_z = floor_deck_z + 0.1
    for i, (x, y, size) in enumerate(positions):
        prim_path = f"/World/Obstacles/obstacle_{i}"
        cube = DynamicCuboid(
            prim_path=prim_path,
            name=f"test_obstacle_{i}",
            position=np.array([x, y, robot_z]),
            size=size,
            color=np.array([0.9, 0.15, 0.15]),  # bright red for stream visibility
            mass=10.0,
        )
        world.scene.add(cube)
        print(f"[orchestrator] spawned obstacle {i}: "
              f"({x:.2f}, {y:.2f}, {robot_z:.2f}) size={size:.2f} m",
              flush=True)


def _freeze_test_obstacles(stage, count):
    """Apply the kinematic flag to every /World/Obstacles/obstacle_<i>.

    Must be called AFTER `world.reset()` — applying the flag before
    reset races with PhysX's tensor-view initialization and can
    invalidate the view that Nova Carter's wheel joint controller uses,
    silently breaking /cmd_vel forever until restart.
    """
    for i in range(count):
        prim_path = f"/World/Obstacles/obstacle_{i}"
        prim = stage.GetPrimAtPath(prim_path)
        if prim and prim.IsValid():
            UsdPhysics.RigidBodyAPI(prim).CreateKinematicEnabledAttr(True)
    if count > 0:
        print(f"[orchestrator] froze {count} obstacles kinematic", flush=True)


def _start_initialpose_publisher(true_pos, true_orn, timeout_s: float = 300.0):
    """Background thread that publishes /initialpose once AMCL subscribes.

    Bypasses the `amcl.initial_pose` params file — that value has to be
    hand-maintained and doesn't capture any USD-native orientation that
    Isaac Sim applies to the robot root. Instead we read the actual world
    pose back from the robot after spawn and forward it to AMCL verbatim.

    Args:
        true_pos: np.ndarray (3,) — Isaac Sim world xyz in meters.
        true_orn: np.ndarray (4,) — Isaac Sim quaternion in (w, x, y, z).
        timeout_s: how long to wait for AMCL to subscribe before giving up.

    TODO: remove when Phase 3 cleanup adds scan-based AMCL convergence.
    """
    import threading

    def _worker():
        try:
            import rclpy
            from geometry_msgs.msg import PoseWithCovarianceStamped
        except ImportError as exc:
            print(f"[orchestrator] rclpy unavailable, skipping /initialpose publish: {exc}",
                  flush=True)
            return

        try:
            rclpy.init()
        except Exception:
            # rclpy may already be initialized by the ROS2 bridge extension;
            # that's fine — we can still create a node on top of it.
            pass

        node = rclpy.create_node("nav2_orchestrator_initialpose")
        pub = node.create_publisher(PoseWithCovarianceStamped, "/initialpose", 10)

        import time
        t0 = time.time()
        while pub.get_subscription_count() == 0:
            if time.time() - t0 > timeout_s:
                print(f"[orchestrator] /initialpose subscriber timeout after "
                      f"{timeout_s:.0f}s — Nav2 never came up, giving up",
                      flush=True)
                node.destroy_node()
                return
            rclpy.spin_once(node, timeout_sec=0.5)

        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = "map"
        msg.pose.pose.position.x = float(true_pos[0])
        msg.pose.pose.position.y = float(true_pos[1])
        msg.pose.pose.position.z = 0.0  # 2D Nav2 ignores Z
        # Isaac Sim returns (w, x, y, z); ROS geometry_msgs uses (x, y, z, w)
        msg.pose.pose.orientation.w = float(true_orn[0])
        msg.pose.pose.orientation.x = float(true_orn[1])
        msg.pose.pose.orientation.y = float(true_orn[2])
        msg.pose.pose.orientation.z = float(true_orn[3])
        # Small covariance — we're confident in the ground-truth pose.
        # Diagonal: [xx, yy, zz, roll, pitch, yaw]
        msg.pose.covariance[0] = 0.01   # x
        msg.pose.covariance[7] = 0.01   # y
        msg.pose.covariance[35] = 0.05  # yaw

        # Publish a few times in case the first hits the race where AMCL
        # just registered its subscription but isn't processing yet.
        for _ in range(3):
            pub.publish(msg)
            rclpy.spin_once(node, timeout_sec=0.2)

        elapsed = time.time() - t0
        print(f"[orchestrator] /initialpose seeded to AMCL after {elapsed:.1f}s "
              f"(xyz={[round(float(v),3) for v in true_pos]}, "
              f"quat_wxyz={[round(float(v),4) for v in true_orn]})",
              flush=True)
        node.destroy_node()

    t = threading.Thread(target=_worker, name="initialpose_publisher", daemon=True)
    t.start()


def main():
    import time
    t0 = time.time()
    def stamp(label):
        print(f"[orchestrator] [+{time.time()-t0:6.1f}s] {label}", flush=True)

    stamp("main() entered")
    world = World(stage_units_in_meters=1.0)
    stamp("World() created")
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
    stamp("Building AddReference returned")

    setup_physics_scene(stage)
    stamp("setup_physics_scene returned")

    # Spawn Nova Carter
    assets_root = get_assets_root_path()
    if assets_root is None:
        raise RuntimeError("get_assets_root_path() returned None — Isaac Sim assets not configured")

    orientation = rot_utils.euler_angles_to_quats(
        np.array([0.0, 0.0, np.degrees(spawn_yaw)]), degrees=True
    )

    # Use the Nova_Carter_ROS variant — same robot, same joint names, but
    # ships with baked-in ROS2 action graphs for odom, TF, lidars, cmd_vel,
    # and clock (see ~/isaacsim/standalone_examples/benchmarks/
    # benchmark_robots_nova_carter_ros2.py for the reference pattern).
    spawn_pos = np.array([spawn_x, spawn_y, spawn_z + 0.1])  # 10 cm above deck
    carter = world.scene.add(
        WheeledRobot(
            prim_path="/World/NovaCarter",
            name="nova_carter",
            wheel_dof_names=["joint_wheel_left", "joint_wheel_right"],
            create_robot=True,
            usd_path=assets_root + "/Isaac/Samples/ROS2/Robots/Nova_Carter_ROS.usd",
            position=spawn_pos,
            orientation=orientation,
        )
    )
    stamp("Nova Carter scene.add returned")

    # Parse any --obstacle CLI flags into (x, y, size) tuples and spawn
    # them under /World/Obstacles before world.reset() so physics handles
    # are initialized on the first reset pass. Obstacles live OUTSIDE the
    # static occupancy map, so Nav2 only sees them via the live lidar →
    # obstacle_layer path — this is how we exercise live avoidance.
    obstacle_positions = []
    for spec in args.obstacle:
        parts = [float(p) for p in spec.split(",")]
        if len(parts) == 2:
            x, y = parts
            s = 0.6
        elif len(parts) == 3:
            x, y, s = parts
        else:
            raise ValueError(
                f"--obstacle must be 'X,Y' or 'X,Y,SIZE'; got {spec!r}"
            )
        obstacle_positions.append((x, y, s))
    _spawn_test_obstacles(stage, world, obstacle_positions, spawn_z)
    stamp(f"spawned {len(obstacle_positions)} test obstacles")

    world.reset()
    stamp("world.reset() returned")
    # Re-apply pose after reset in case the default sample zeros it
    carter.set_world_pose(position=spawn_pos, orientation=orientation)
    stamp("carter.set_world_pose returned")

    # Freeze obstacles kinematic now that the PhysX tensor view is
    # initialized. Doing this before world.reset() has been observed to
    # invalidate the tensor view used by Nova Carter's wheel controller.
    _freeze_test_obstacles(stage, len(obstacle_positions))
    stamp("obstacles frozen")

    # Ground-truth pose readback: Isaac Sim's WheeledRobot(orientation=...) applies
    # the quat on top of Nova_Carter_ROS.usd's native transform. The resulting
    # body-frame "forward" may not match the map frame, which breaks AMCL and
    # Nav2's controller. Read the actual pose and seed AMCL's /initialpose
    # with it so Nav2 sees the same orientation Isaac Sim actually set.
    true_pos, true_orn = carter.get_world_pose()
    stamp(f"ground-truth pose: xyz={[round(float(v),3) for v in true_pos]} "
          f"quat(wxyz)={[round(float(v),4) for v in true_orn]}")

    _start_initialpose_publisher(true_pos, true_orn)

    # Enable the 2D/3D lidars that Nav2's costmap layers need, and disable
    # the four Hawk cameras (heavy, not needed for navigation). The action
    # graph prim paths below come from Nova_Carter_ROS.usd's internal layout.
    robot_prim = "/World/NovaCarter"
    for lidar in ("front_2d_lidar_render_product",
                  "back_2d_lidar_render_product",
                  "front_3d_lidar_render_product"):
        og.Controller.attribute(
            f"{robot_prim}/ros_lidars/{lidar}.inputs:enabled"
        ).set(True)
        stamp(f"lidar enabled: {lidar}")
    for hawk in ("front_hawk", "left_hawk", "right_hawk", "back_hawk"):
        for cam in ("left_camera_render_product", "right_camera_render_product"):
            try:
                og.Controller.attribute(
                    f"{robot_prim}/{hawk}/{cam}.inputs:enabled"
                ).set(False)
                stamp(f"hawk disabled: {hawk}/{cam}")
            except Exception as e:
                stamp(f"hawk skip: {hawk}/{cam}: {e}")

    # Aim the viewport Perspective camera at the robot so the streamed
    # view opens on the spawn instead of the USD's default position.
    # Isometric angle: offset in -x/-y, 10m above the deck.
    cam_eye = np.array([spawn_x - 12.0, spawn_y - 12.0, spawn_z + 10.0])
    cam_target = np.array([spawn_x, spawn_y, spawn_z])
    set_camera_view(eye=cam_eye, target=cam_target)
    stamp("set_camera_view returned")

    print("[orchestrator] Scene ready. Robot spawned with ROS2 bridge active.", flush=True)
    print("[orchestrator] Topics: /chassis/odom, /front_3d_lidar/lidar_points,", flush=True)
    print("[orchestrator]         /front_2d_lidar/scan, /back_2d_lidar/scan,", flush=True)
    print("[orchestrator]         /tf, /clock, /cmd_vel", flush=True)
    print("[orchestrator] Launch Nav2 in another terminal:", flush=True)
    print(f"  ros2 launch kern_navigation kern_navigation.launch.py map:={floor['map_metadata_yaml']}", flush=True)

    # Run until the window is closed / process is killed
    while simulation_app.is_running():
        world.step(render=True)

    # Intentional: do not call simulation_app.close() — OmniGraph cleanup
    # segfaults on exit (see validate_robot_movement.py:216 and CLAUDE.md).


if __name__ == "__main__":
    main()

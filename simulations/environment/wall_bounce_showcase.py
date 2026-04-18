#!/usr/bin/env python3
"""
Wall Bounce Showcase — Exterior Ball Launch (trees hidden)

Reproduces the original `videos/environment/wall_bounce_showcase.mp4` camera
arc and ball volley against the Innovation Center facade, with all
tree/planting/entourage geometry hidden for a clean establishing shot.

Parameters are lifted from `logs/wall_bounce_showcase.txt`.

Usage:
    sbatch submit_isaac_job.sh simulations/environment/wall_bounce_showcase.py

Output:
    - videos/environment/wall_bounce_showcase_no_trees.mp4 (15s @ 30fps, 1920x1080)
    - logs/wall_bounce_showcase_no_trees.txt
"""

import argparse
import math
import os
import sys

parser = argparse.ArgumentParser(description="Wall bounce showcase (no trees).")
parser.add_argument(
    "--input",
    default="assets/environments/msoe_building/MSOEBuildingV4_physics.usd",
    help="Path to physics-ready USD file (default: Innovation Center V4 physics)",
)
parser.add_argument(
    "--video",
    default="environment/wall_bounce_showcase_no_trees",
    help="Output video name (under videos/), no extension",
)
parser.add_argument(
    "--report",
    default=None,
    help="Report log path override (default: logs/wall_bounce_showcase_no_trees.txt)",
)
args, _unknown = parser.parse_known_args()

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

import numpy as np
import isaacsim.core.utils.numpy.rotations as rot_utils

import omni.usd
from pxr import Sdf, UsdLux, UsdGeom, UsdPhysics, UsdShade, Usd, Gf
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicSphere
from isaacsim.sensors.camera import Camera

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.abspath(os.path.join(_script_dir, "..", ".."))
sys.path.insert(0, _project_dir)

from sim_utils.video_capture import VideoRecorder
from sim_utils.building_physics import setup_physics_scene

input_usd = os.path.abspath(
    args.input if os.path.isabs(args.input) else os.path.join(_project_dir, args.input)
)
report_path = os.path.abspath(
    args.report
    if args.report
    else os.path.join(_project_dir, "logs", "wall_bounce_showcase_no_trees.txt")
)
os.makedirs(os.path.dirname(report_path), exist_ok=True)

HIDE_PATTERNS = ["Tree", "Planting", "Entourage"]

# Parameters from the original log
NUM_PERIMETER = 32
NUM_HEIGHT_LAYERS = 3
LAUNCH_OFFSET = 15.0       # meters outside the building facade along each angle
LAUNCH_SPEED = 40.0        # m/s horizontal, aimed at centroid
BALL_RADIUS = 1.2
BALL_MASS = 20.0
BALL_RESTITUTION = 0.85
GRAVITY = 9.81

CAM_RADIUS = 341.4
CAM_HEIGHT = 104.9
CAM_AZ_START_DEG = 45.0
CAM_AZ_END_DEG = 90.0
CAM_FOCAL_LENGTH_MM = 35.0

WARMUP_FRAMES = 30
RECORD_FRAMES = 450
FPS = 30

BALL_COLORS = [
    [1.0, 0.15, 0.15], [1.0, 0.5, 0.0], [1.0, 0.9, 0.1],
    [0.1, 0.8, 0.2],  [0.0, 0.9, 0.9], [0.2, 0.3, 1.0],
    [0.6, 0.3, 1.0],  [1.0, 0.4, 0.7],
]

_report_file = None


def log(msg=""):
    print(msg, flush=True)
    if _report_file:
        _report_file.write(msg + "\n")
        _report_file.flush()


def hide_matching_prims(stage, patterns):
    hidden = set()
    per_pattern = {p: 0 for p in patterns}
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if not path.startswith("/World/Building"):
            continue
        for pat in patterns:
            if pat and pat in path:
                imageable = UsdGeom.Imageable(prim)
                if imageable:
                    imageable.CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
                hidden.add(path)
                per_pattern[pat] += 1
                break
    return hidden, per_pattern


def compute_building_geometry(building_prim, bbox_cache, hidden_paths):
    """Return (centroid_xyz, mesh_corners_xy Nx2) excluding hidden prims."""
    positions = []
    weights = []
    corners_xy = []
    for prim in Usd.PrimRange(building_prim):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        p = str(prim.GetPath())
        if any(p == h or p.startswith(h + "/") for h in hidden_paths):
            continue
        rng = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
        if rng.IsEmpty():
            continue
        emin, emax = rng.GetMin(), rng.GetMax()
        center = np.array([(emin[i] + emax[i]) / 2.0 for i in range(3)])
        extent = np.array([emax[i] - emin[i] for i in range(3)])
        vol = max(extent[0] * extent[1] * extent[2], 0.001)
        positions.append(center)
        weights.append(vol)
        corners_xy.extend([
            [emin[0], emin[1]], [emax[0], emin[1]],
            [emin[0], emax[1]], [emax[0], emax[1]],
        ])
    if not positions:
        return None, None
    positions = np.array(positions)
    weights = np.array(weights)
    centroid = np.average(positions, axis=0, weights=weights)
    return centroid, np.array(corners_xy)


def facade_distances(centroid_xy, corners_xy, angles):
    offsets = corners_xy - np.asarray(centroid_xy, dtype=float)
    dirs = np.stack([np.cos(angles), np.sin(angles)], axis=0)
    return (offsets @ dirs).max(axis=0)


def interpolate_facade(angle, sample_angles, sample_dists):
    two_pi = 2.0 * math.pi
    a = angle % two_pi
    n = len(sample_angles)
    step = two_pi / n
    idx = a / step
    i0 = int(idx) % n
    i1 = (i0 + 1) % n
    frac = idx - int(idx)
    return sample_dists[i0] * (1.0 - frac) + sample_dists[i1] * frac


def add_ground_plane(stage, z):
    UsdGeom.Xform.Define(stage, "/World/GroundPlane")
    mesh = UsdGeom.Mesh.Define(stage, "/World/GroundPlane/Mesh")
    half = 500.0
    mesh.CreatePointsAttr([
        Gf.Vec3f(-half, -half, z), Gf.Vec3f(half, -half, z),
        Gf.Vec3f(half, half, z), Gf.Vec3f(-half, half, z),
    ])
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mesh.CreateNormalsAttr([Gf.Vec3f(0, 0, 1)] * 4)
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    coll = UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim())
    coll.CreateApproximationAttr("none")
    return mesh.GetPrim()


def create_bouncy_material(stage, path="/World/Physics/BouncyMat", restitution=0.85):
    UsdGeom.Scope.Define(stage, "/World/Physics")
    mat_prim = UsdShade.Material.Define(stage, path).GetPrim()
    pm = UsdPhysics.MaterialAPI.Apply(mat_prim)
    pm.CreateRestitutionAttr(restitution)
    pm.CreateDynamicFrictionAttr(0.4)
    pm.CreateStaticFrictionAttr(0.5)
    return mat_prim


def bind_material(prim, mat_prim):
    binding = UsdShade.MaterialBindingAPI.Apply(prim)
    binding.Bind(
        UsdShade.Material(mat_prim),
        bindingStrength=UsdShade.Tokens.weakerThanDescendants,
        materialPurpose="physics",
    )


def freeze_building_bodies(stage):
    frozen = 0
    for prim in stage.Traverse():
        if not str(prim.GetPath()).startswith("/World/Building"):
            continue
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rb = UsdPhysics.RigidBodyAPI(prim)
            rb.CreateKinematicEnabledAttr(True)
            frozen += 1
    return frozen


def look_at_orientation(cam_pos, target):
    dx = target[0] - cam_pos[0]
    dy = target[1] - cam_pos[1]
    dz = target[2] - cam_pos[2]
    yaw = math.degrees(math.atan2(dy, dx))
    pitch = math.degrees(math.atan2(-dz, math.sqrt(dx * dx + dy * dy)))
    return rot_utils.euler_angles_to_quats(
        np.array([0.0, pitch, yaw]), degrees=True
    )


def main():
    global _report_file
    _report_file = open(report_path, "w")

    log("=" * 65)
    log("Wall Bounce Showcase (no trees)")
    log("=" * 65)
    log(f"Input:  {input_usd}")
    log(f"Video:  videos/{args.video}.mp4")
    log(f"Report: {report_path}")

    if not os.path.exists(input_usd):
        log(f"ERROR: {input_usd} not found.")
        _report_file.close()
        simulation_app.close()
        sys.exit(1)

    world = World(stage_units_in_meters=1.0)
    stage = omni.usd.get_context().get_stage()

    log("\n--- Scene Setup ---")
    building_prim = stage.DefinePrim("/World/Building", "Xform")
    building_prim.GetReferences().AddReference(
        input_usd, primPath=Sdf.Path("/World/Building")
    )
    setup_physics_scene(stage)

    hidden_paths, per_pattern = hide_matching_prims(stage, HIDE_PATTERNS)
    log(f"Hide patterns: {HIDE_PATTERNS}")
    for pat, count in per_pattern.items():
        log(f"  '{pat}' -> {count} prims hidden")
    log(f"Total prims hidden: {len(hidden_paths)}")

    building_prim = stage.GetPrimAtPath("/World/Building")
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    bbox = bbox_cache.ComputeWorldBound(building_prim)
    box_range = bbox.ComputeAlignedRange()
    bb_min, bb_max = box_range.GetMin(), box_range.GetMax()
    bbox_center = np.array([(bb_min[i] + bb_max[i]) / 2.0 for i in range(3)])
    extent = np.array([bb_max[i] - bb_min[i] for i in range(3)])
    log(f"Building bbox (incl. hidden): "
        f"({bb_min[0]:.1f}, {bb_min[1]:.1f}, {bb_min[2]:.1f}) to "
        f"({bb_max[0]:.1f}, {bb_max[1]:.1f}, {bb_max[2]:.1f})")

    centroid, mesh_corners_xy = compute_building_geometry(
        building_prim, bbox_cache, hidden_paths
    )
    if centroid is None:
        centroid = bbox_center
        mesh_corners_xy = np.array([
            [bb_min[0], bb_min[1]], [bb_max[0], bb_min[1]],
            [bb_min[0], bb_max[1]], [bb_max[0], bb_max[1]],
        ])
        log("WARNING: centroid failed — using bbox center")
    log(f"Mesh centroid (trees excluded): "
        f"({centroid[0]:.1f}, {centroid[1]:.1f}, {centroid[2]:.1f})")

    sample_angles = np.linspace(0.0, 2.0 * math.pi, 72, endpoint=False)
    facade_dists = facade_distances(centroid[:2], mesh_corners_xy, sample_angles)
    horiz_diag = math.sqrt(float(extent[0]) ** 2 + float(extent[1]) ** 2)
    clip_max = horiz_diag / 2.0 * 1.05
    facade_dists = np.minimum(facade_dists, clip_max)
    log(f"Facade distance: min={facade_dists.min():.1f} max={facade_dists.max():.1f} "
        f"mean={facade_dists.mean():.1f}")

    ground_z = float(bb_min[2]) - 0.5
    ground_prim = add_ground_plane(stage, ground_z)
    log(f"Ground plane at Z={ground_z:.1f}")

    frozen = freeze_building_bodies(stage)
    log(f"Froze {frozen} dynamic building bodies (kinematic)")

    bouncy = create_bouncy_material(stage, restitution=BALL_RESTITUTION)
    bind_material(ground_prim, bouncy)
    log(f"Created bouncy material: restitution={BALL_RESTITUTION}")

    # Lighting
    distant = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/Sun"))
    distant.CreateIntensityAttr(4000)
    distant.CreateAngleAttr(1.0)
    UsdGeom.Xformable(distant.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(-45, 30, 0))
    UsdLux.DomeLight.Define(stage, Sdf.Path("/World/DomeLight")).CreateIntensityAttr(700)

    # --- Spawn balls ---
    log("\n--- Spawning Balls ---")
    total_balls = NUM_PERIMETER * NUM_HEIGHT_LAYERS
    log(f"Spawning {total_balls} balls ({NUM_PERIMETER} perimeter x {NUM_HEIGHT_LAYERS} layers)")
    log(f"Launch offset: {LAUNCH_OFFSET}m outside facade")
    log(f"Launch speed: {LAUNCH_SPEED} m/s")

    balls = []  # list of {obj, fire_pos, fire_vel}
    for k in range(NUM_PERIMETER):
        angle = 2.0 * math.pi * k / NUM_PERIMETER
        facade_r = float(interpolate_facade(angle, sample_angles, facade_dists))
        launch_r = facade_r + LAUNCH_OFFSET
        for h in range(NUM_HEIGHT_LAYERS):
            height_frac = 0.25 + 0.25 * h       # 25%, 50%, 75% of bbox height
            launch_z = float(bb_min[2]) + float(extent[2]) * height_frac
            fire_pos = np.array([
                float(centroid[0]) + launch_r * math.cos(angle),
                float(centroid[1]) + launch_r * math.sin(angle),
                launch_z,
            ])
            dx = float(centroid[0]) - fire_pos[0]
            dy = float(centroid[1]) - fire_pos[1]
            horiz_dist = math.sqrt(dx * dx + dy * dy)
            horiz_dir = np.array([dx / horiz_dist, dy / horiz_dist, 0.0])
            flight_dist = max(launch_r - facade_r, 0.5)
            flight_time = flight_dist / LAUNCH_SPEED
            v_z = 0.5 * GRAVITY * flight_time
            fire_vel = horiz_dir * LAUNCH_SPEED + np.array([0.0, 0.0, v_z])

            color = BALL_COLORS[(k + h) % len(BALL_COLORS)]
            name = f"ball_{k:02d}_{h}"
            ball = world.scene.add(DynamicSphere(
                prim_path=f"/World/Balls/{name}",
                name=name,
                position=fire_pos,
                radius=BALL_RADIUS,
                color=np.array(color),
                mass=BALL_MASS,
            ))
            bind_material(stage.GetPrimAtPath(f"/World/Balls/{name}"), bouncy)
            balls.append({"obj": ball, "fire_pos": fire_pos, "fire_vel": fire_vel})

    log(f"Spawned {len(balls)} balls")

    # --- Camera ---
    log("\n--- Camera Setup ---")
    az0 = math.radians(CAM_AZ_START_DEG)
    cam_pos0 = np.array([
        float(centroid[0]) + CAM_RADIUS * math.cos(az0),
        float(centroid[1]) + CAM_RADIUS * math.sin(az0),
        CAM_HEIGHT,
    ])
    camera = Camera(
        prim_path="/World/OrbitCamera",
        position=cam_pos0,
        frequency=FPS,
        resolution=(1920, 1080),
        orientation=look_at_orientation(cam_pos0, centroid),
    )
    cam_prim = stage.GetPrimAtPath("/World/OrbitCamera")
    fl_attr = cam_prim.GetAttribute("focalLength")
    if fl_attr:
        fl_attr.Set(CAM_FOCAL_LENGTH_MM)
    log(f"Camera orbit radius: {CAM_RADIUS}m, height: {CAM_HEIGHT}m")
    log(f"Camera sweep: {CAM_AZ_START_DEG:.0f} deg -> {CAM_AZ_END_DEG:.0f} deg")

    world.reset()
    camera.initialize()

    # Zero all ball velocities so they float at fire_pos during warmup
    for bd in balls:
        bd["obj"].set_linear_velocity(np.array([0.0, 0.0, 0.0]))
        bd["obj"].set_angular_velocity(np.array([0.0, 0.0, 0.0]))

    log("\n--- Simulate & Record ---")
    log(f"  Warmup: {WARMUP_FRAMES} frames")
    log(f"  Record: {RECORD_FRAMES} frames ({RECORD_FRAMES / FPS:.0f}s at {FPS}fps)")

    log("Warming up...")
    for _ in range(WARMUP_FRAMES):
        for bd in balls:
            bd["obj"].set_world_pose(position=bd["fire_pos"])
            bd["obj"].set_linear_velocity(np.array([0.0, 0.0, 0.0]))
            bd["obj"].set_angular_velocity(np.array([0.0, 0.0, 0.0]))
        world.step(render=True)

    log("Recording...")
    fired = False
    with VideoRecorder(args.video, fps=FPS) as recorder:
        for i in range(RECORD_FRAMES):
            # Camera sweep 45 -> 90 degrees
            t = i / max(RECORD_FRAMES - 1, 1)
            az = math.radians(CAM_AZ_START_DEG + (CAM_AZ_END_DEG - CAM_AZ_START_DEG) * t)
            cam_pos = np.array([
                float(centroid[0]) + CAM_RADIUS * math.cos(az),
                float(centroid[1]) + CAM_RADIUS * math.sin(az),
                CAM_HEIGHT,
            ])
            camera.set_world_pose(
                position=cam_pos,
                orientation=look_at_orientation(cam_pos, centroid),
            )

            if i == 0 and not fired:
                for bd in balls:
                    bd["obj"].set_world_pose(position=bd["fire_pos"])
                    bd["obj"].set_linear_velocity(bd["fire_vel"])
                    bd["obj"].set_angular_velocity(np.array([0.0, 0.0, 0.0]))
                fired = True

            world.step(render=True)
            rgba = camera.get_rgba()
            if rgba is not None and i > 0:
                recorder.capture_rgba(rgba)

            if i % 100 == 0:
                log(f"  Frame {i}/{RECORD_FRAMES}  cam=({cam_pos[0]:.0f}, "
                    f"{cam_pos[1]:.0f}, {cam_pos[2]:.0f})")

    log(f"\nRecording complete - {RECORD_FRAMES} frames")
    log(f"Video -> videos/{args.video}.mp4")
    log("=" * 65)

    _report_file.close()
    simulation_app.close()


if __name__ == "__main__":
    main()

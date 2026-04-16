#!/usr/bin/env python3
"""
Dierks Hall — Ball Barrage Showcase

Orbital camera tour with a ring of ball "cannons" firing volleys at the
facade from every angle. Each launch site fires in sync with the orbiting
camera so the viewer sees fresh impacts on the near side as the camera
sweeps around the building.

Usage:
    sbatch submit_isaac_job.sh simulations/environment/dierks_ball_barrage.py \
        --input assets/environments/msoe_building/DierksHallV2_physics.usd

Output:
    - videos/environment/<stem>_ball_barrage.mp4 (15s @ 30fps, 1920x1080)
    - logs/<stem>_ball_barrage.txt
"""

import argparse
import os
import sys
import math

parser = argparse.ArgumentParser(description="Ball barrage showcase with orbital camera.")
parser.add_argument("--input", required=True, help="Path to physics-ready USD file")
parser.add_argument("--video", default=None, help="Output video name override")
parser.add_argument("--report", default=None, help="Report log path override")
args, _unknown = parser.parse_known_args()

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

import numpy as np
import isaacsim.core.utils.numpy.rotations as rot_utils

import omni.usd
from pxr import Sdf, UsdLux, UsdGeom, Usd, UsdPhysics, Gf
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicSphere
from isaacsim.sensors.camera import Camera

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.abspath(os.path.join(_script_dir, '..', '..'))
sys.path.insert(0, _project_dir)

from sim_utils.video_capture import VideoRecorder
from sim_utils.building_physics import setup_physics_scene

input_usd = os.path.abspath(args.input)
stem = os.path.splitext(os.path.basename(input_usd))[0]

video_name = args.video if args.video else f"environment/{stem}_ball_barrage"

if args.report:
    report_path = os.path.abspath(args.report)
else:
    report_path = os.path.join(_project_dir, "logs", f"{stem}_ball_barrage.txt")

os.makedirs(os.path.dirname(report_path), exist_ok=True)

_report_file = None


def log(msg=""):
    print(msg, flush=True)
    if _report_file:
        _report_file.write(msg + "\n")
        _report_file.flush()


# Barrage parameters — all radii scale off building half-diagonal so the
# layout is consistent regardless of building aspect ratio.
NUM_AZIMUTHS = 24           # ring segments
BALLS_PER_SITE = 4          # 2 heights * 2 balls
BALL_RADIUS = 2.0           # visible at ~250 m orbit distance
BALL_MASS = 25.0
LAUNCH_SPEED = 55.0         # m/s toward building center — fast, dramatic streaks
VELOCITY_JITTER = 2.0
HEIGHT_LAYERS = 2
LAUNCH_RADIUS_MULT = 1.15   # launch ring clears bbox at all angles (not just corners)
CAM_RADIUS_MULT  = 1.60     # camera sits outside the launch ring
CAM_HEIGHT_MULT  = 0.55     # moderate 3/4 view, constant altitude
CAM_FOCAL_LENGTH_MM = 18.0

# Color palette — cycled across sites for visual variety
BALL_COLORS = [
    [1.0, 0.15, 0.15], [1.0, 0.5, 0.0], [1.0, 0.9, 0.1],
    [0.1, 0.8, 0.2],  [0.0, 0.9, 0.9], [0.2, 0.3, 1.0],
    [0.6, 0.3, 1.0],  [1.0, 0.4, 0.7],
]


def add_ground_plane(stage, z):
    UsdGeom.Xform.Define(stage, "/World/GroundPlane")
    mesh = UsdGeom.Mesh.Define(stage, "/World/GroundPlane/Mesh")
    half = 400.0
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


def freeze_doors(stage):
    """Make all building rigid bodies kinematic so they don't shatter."""
    frozen = 0
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if not path.startswith("/World/Building"):
            continue
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rb = UsdPhysics.RigidBodyAPI(prim)
            rb.CreateKinematicEnabledAttr(True)
            frozen += 1
    return frozen


def compute_building_centroid(building_prim, bbox_cache):
    """Volume-weighted centroid of all mesh elements under the building.

    The bounding-box midpoint can be far from the visual center when the
    building has an irregular footprint (L-shaped, wings, etc.).  This
    weights each mesh's bbox center by its bbox volume, giving a centroid
    that tracks where actual geometry is concentrated.
    """
    positions = []
    weights = []
    for prim in Usd.PrimRange(building_prim):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        elem_bbox = bbox_cache.ComputeWorldBound(prim)
        elem_range = elem_bbox.ComputeAlignedRange()
        if elem_range.IsEmpty():
            continue
        emin = elem_range.GetMin()
        emax = elem_range.GetMax()
        center = np.array([(emin[i] + emax[i]) / 2.0 for i in range(3)])
        extent = np.array([emax[i] - emin[i] for i in range(3)])
        vol = max(extent[0] * extent[1] * extent[2], 0.001)
        positions.append(center)
        weights.append(vol)

    if len(positions) == 0:
        return None

    positions = np.array(positions)
    weights = np.array(weights)
    return np.average(positions, axis=0, weights=weights)


def look_at_orientation(cam_pos, target):
    dx = target[0] - cam_pos[0]
    dy = target[1] - cam_pos[1]
    dz = target[2] - cam_pos[2]
    yaw = math.degrees(math.atan2(dy, dx))
    pitch = math.degrees(math.atan2(-dz, math.sqrt(dx**2 + dy**2)))
    return rot_utils.euler_angles_to_quats(
        np.array([0, pitch, yaw]), degrees=True
    )


def main():
    global _report_file
    _report_file = open(report_path, "w")

    log("=" * 60)
    log("Dierks Ball Barrage Showcase")
    log("=" * 60)
    log(f"Input:  {input_usd}")
    log(f"Video:  {video_name}")

    if not os.path.exists(input_usd):
        log(f"ERROR: {input_usd} not found.")
        _report_file.close()
        simulation_app.close()
        sys.exit(1)

    # --- Phase 1: Scene ---
    world = World(stage_units_in_meters=1.0)
    stage = omni.usd.get_context().get_stage()

    log(f"\nLoading: {input_usd}")
    building_prim = stage.DefinePrim("/World/Building", "Xform")
    building_prim.GetReferences().AddReference(
        input_usd, primPath=Sdf.Path("/World/Building")
    )

    setup_physics_scene(stage)

    building_prim = stage.GetPrimAtPath("/World/Building")
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    bbox = bbox_cache.ComputeWorldBound(building_prim)
    box_range = bbox.ComputeAlignedRange()
    bb_min, bb_max = box_range.GetMin(), box_range.GetMax()

    bbox_center = np.array([(bb_min[i] + bb_max[i]) / 2.0 for i in range(3)])
    bldg_extent = np.array([bb_max[i] - bb_min[i] for i in range(3)])
    ground_z = bb_min[2] - 0.5
    horiz_diag = math.sqrt(float(bldg_extent[0])**2 + float(bldg_extent[1])**2)
    half_diag = horiz_diag / 2.0

    # Use mesh-weighted centroid for camera orbit and ball targeting —
    # the bbox midpoint can be far off for irregular building footprints.
    centroid = compute_building_centroid(building_prim, bbox_cache)
    if centroid is not None:
        bldg_center = centroid
        offset = np.linalg.norm(centroid[:2] - bbox_center[:2])
        log(f"Mesh centroid: ({centroid[0]:.1f}, {centroid[1]:.1f}, {centroid[2]:.1f})")
        log(f"Bbox center:   ({bbox_center[0]:.1f}, {bbox_center[1]:.1f}, {bbox_center[2]:.1f})")
        log(f"XY offset between centroid and bbox center: {offset:.1f}")
    else:
        bldg_center = bbox_center
        log("WARNING: centroid computation failed, falling back to bbox center")

    log(f"Building bbox: ({bb_min[0]:.1f}, {bb_min[1]:.1f}, {bb_min[2]:.1f}) "
        f"to ({bb_max[0]:.1f}, {bb_max[1]:.1f}, {bb_max[2]:.1f})")
    log(f"Using center: ({bldg_center[0]:.1f}, {bldg_center[1]:.1f}, {bldg_center[2]:.1f}), "
        f"diag: {horiz_diag:.1f}")

    add_ground_plane(stage, ground_z)
    frozen = freeze_doors(stage)
    log(f"Froze {frozen} building rigid bodies")

    # Lighting
    distant = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/Sun"))
    distant.CreateIntensityAttr(4000)
    distant.CreateAngleAttr(1.0)
    xf = UsdGeom.Xformable(distant.GetPrim())
    xf.AddRotateXYZOp().Set(Gf.Vec3f(-45, 30, 0))
    UsdLux.DomeLight.Define(stage, Sdf.Path("/World/DomeLight")).CreateIntensityAttr(700)

    # --- Phase 2: Spawn ball grid ---
    log(f"\nSpawning {NUM_AZIMUTHS * HEIGHT_LAYERS * BALLS_PER_SITE // HEIGHT_LAYERS // 1} balls")
    # All distances scale off half-diagonal, so balls always start just
    # outside the building envelope and the camera sits a bit further out.
    launch_radius = half_diag * LAUNCH_RADIUS_MULT
    cam_radius = half_diag * CAM_RADIUS_MULT
    cam_height = float(bldg_center[2]) + float(bldg_extent[2]) * CAM_HEIGHT_MULT

    # Storage position: far above, well out of view
    storage_z = float(bb_max[2]) + 150.0

    # Ball layout: site k has BALLS_PER_SITE balls across HEIGHT_LAYERS layers
    balls_per_layer = BALLS_PER_SITE // HEIGHT_LAYERS
    sites = []  # each: {"angle": rad, "balls": [(obj, fire_pos, fire_vel)]}
    total_spawned = 0

    for k in range(NUM_AZIMUTHS):
        angle = 2 * math.pi * k / NUM_AZIMUTHS
        site_balls = []

        for layer in range(HEIGHT_LAYERS):
            # Launch sites at 30% and 65% of building height
            height_frac = 0.30 + 0.35 * layer
            launch_z = float(bb_min[2]) + float(bldg_extent[2]) * height_frac

            for b in range(balls_per_layer):
                # Spread along the tangent direction for a wider volley
                tangent_offset = (b - (balls_per_layer - 1) / 2.0) * 2.5
                tx = -math.sin(angle) * tangent_offset
                ty = math.cos(angle) * tangent_offset

                fire_pos = np.array([
                    float(bldg_center[0]) + launch_radius * math.cos(angle) + tx,
                    float(bldg_center[1]) + launch_radius * math.sin(angle) + ty,
                    launch_z,
                ])
                # Velocity: toward building center, with small jitter
                direction = bldg_center - fire_pos
                direction[2] = 0.0
                direction = direction / (np.linalg.norm(direction) + 1e-9)
                jitter = (np.random.rand(3) - 0.5) * 2.0 * VELOCITY_JITTER
                jitter[2] *= 0.3  # less vertical jitter
                fire_vel = direction * LAUNCH_SPEED + jitter

                color = BALL_COLORS[(k + layer) % len(BALL_COLORS)]
                storage_pos = np.array([
                    fire_pos[0] + (np.random.rand() - 0.5) * 4.0,
                    fire_pos[1] + (np.random.rand() - 0.5) * 4.0,
                    storage_z + layer * 4.0 + b * 1.0,
                ])

                name = f"ball_{k:02d}_{layer}_{b}"
                ball = world.scene.add(DynamicSphere(
                    prim_path=f"/World/Balls/{name}",
                    name=name,
                    position=storage_pos,
                    radius=BALL_RADIUS,
                    color=np.array(color),
                    mass=BALL_MASS,
                ))
                site_balls.append({
                    "obj": ball,
                    "fire_pos": fire_pos,
                    "fire_vel": fire_vel,
                    "storage_pos": storage_pos,
                })
                total_spawned += 1

        sites.append({"angle": angle, "balls": site_balls})

    log(f"Total balls: {total_spawned} across {NUM_AZIMUTHS} launch sites")

    # Camera
    initial_pos = np.array([
        float(bldg_center[0]) + cam_radius,
        float(bldg_center[1]),
        cam_height,
    ])
    initial_orient = look_at_orientation(initial_pos, bldg_center)
    camera = Camera(
        prim_path="/World/OrbitCamera",
        position=initial_pos,
        frequency=30,
        resolution=(1920, 1080),
        orientation=initial_orient,
    )
    # Force wide-angle via the USD prim attribute (Camera() default is ~24mm)
    cam_prim = stage.GetPrimAtPath("/World/OrbitCamera")
    if cam_prim.IsValid():
        fl_attr = cam_prim.GetAttribute("focalLength")
        if fl_attr:
            fl_attr.Set(CAM_FOCAL_LENGTH_MM)
    log(f"Camera orbit radius={cam_radius:.1f}, height={cam_height:.1f}, "
        f"focal={CAM_FOCAL_LENGTH_MM}mm")

    # --- Phase 3: Reset & warm-up ---
    world.reset()
    camera.initialize()

    # Freeze balls in storage: zero out velocity + disable gravity via kinematic
    for site in sites:
        for bd in site["balls"]:
            bd["obj"].set_linear_velocity(np.array([0.0, 0.0, 0.0]))
            bd["obj"].set_angular_velocity(np.array([0.0, 0.0, 0.0]))

    log("\nWarming up (30 frames)...")
    for _ in range(30):
        # Keep storage balls pinned by re-zeroing velocity each frame
        for site in sites:
            for bd in site["balls"]:
                bd["obj"].set_world_pose(position=bd["storage_pos"])
                bd["obj"].set_linear_velocity(np.array([0.0, 0.0, 0.0]))
                bd["obj"].set_angular_velocity(np.array([0.0, 0.0, 0.0]))
        world.step(render=True)

    # --- Phase 4: Orbit + fire ---
    orbit_frames = 450
    # Fire each site when camera has rotated to be ~90 deg ahead of it,
    # so the camera sees balls streaking in toward the near facade.
    fire_lead_frac = 0.15  # fraction of a full turn that camera leads the site
    fire_schedule = {}
    for k, site in enumerate(sites):
        site_orbit_frac = k / NUM_AZIMUTHS
        cam_at_fire = (site_orbit_frac + fire_lead_frac) % 1.0
        fire_frame = int(cam_at_fire * orbit_frames)
        fire_schedule[fire_frame] = fire_schedule.get(fire_frame, []) + [k]

    log(f"\nRecording {orbit_frames} frames ({orbit_frames/30:.0f}s @ 30fps)")
    log(f"Launch schedule: {NUM_AZIMUTHS} volleys staggered every "
        f"{orbit_frames // NUM_AZIMUTHS} frames")

    fired_sites = set()
    with VideoRecorder(video_name, fps=30) as recorder:
        for i in range(orbit_frames):
            # Camera position — pure circular orbit centered on building,
            # constant altitude, always looking at the building center.
            angle = 2 * math.pi * i / orbit_frames
            cam_x = float(bldg_center[0]) + cam_radius * math.cos(angle)
            cam_y = float(bldg_center[1]) + cam_radius * math.sin(angle)
            cam_z = cam_height
            cam_pos = np.array([cam_x, cam_y, cam_z])
            orient = look_at_orientation(cam_pos, bldg_center)
            camera.set_world_pose(position=cam_pos, orientation=orient)

            # Fire any sites scheduled for this frame
            for k in fire_schedule.get(i, []):
                site = sites[k]
                for bd in site["balls"]:
                    bd["obj"].set_world_pose(position=bd["fire_pos"])
                    bd["obj"].set_linear_velocity(bd["fire_vel"])
                    bd["obj"].set_angular_velocity(np.array([0.0, 0.0, 0.0]))
                fired_sites.add(k)
                # Verify first ball actually got velocity
                if k < 3:
                    vel_check = site["balls"][0]["obj"].get_linear_velocity()
                    log(f"  [fire] site={k} frame={i} vel_cmd={site['balls'][0]['fire_vel'][:2]} "
                        f"vel_read={vel_check[:2] if vel_check is not None else None}")

            # Pin un-fired sites in storage
            for k, site in enumerate(sites):
                if k in fired_sites:
                    continue
                for bd in site["balls"]:
                    bd["obj"].set_world_pose(position=bd["storage_pos"])
                    bd["obj"].set_linear_velocity(np.array([0.0, 0.0, 0.0]))
                    bd["obj"].set_angular_velocity(np.array([0.0, 0.0, 0.0]))

            world.step(render=True)
            rgba = camera.get_rgba()
            if rgba is not None and i > 0:
                recorder.capture_rgba(rgba)

            if i % 60 == 0:
                log(f"  Frame {i}/{orbit_frames}  fired_sites={len(fired_sites)}/{NUM_AZIMUTHS}")

    log("\nRecording complete.")
    log(f"  Video  -> videos/{video_name}.mp4")
    log(f"  Report -> {report_path}")
    log("=" * 60)

    _report_file.close()
    simulation_app.close()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Dierks Hall — Ball Barrage Showcase

Orbital camera tour with a ring of ball "cannons" firing volleys at the
facade from every angle. Each launch site fires in sync with the orbiting
camera so the viewer sees fresh impacts on the near side as the camera
sweeps around the building.

Usage:
    sbatch scripts/slurm/submit_isaac_job.sh simulations/environment/dierks_ball_barrage.py \
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
parser.add_argument(
    "--hide-patterns",
    default="",
    help="Comma-separated substrings; any prim path under /World/Building matching "
         "any of these is hidden (visibility=invisible) AND excluded from geometry "
         "calc. Useful for stripping trees, planting, and site clutter that "
         "inflate the camera orbit radius. Example: 'Tree,Planting,Entourage'.",
)
parser.add_argument(
    "--cam-radius-mult",
    type=float,
    default=1.05,
    help="Camera orbit radius as a multiple of building half-diagonal (default 1.05). "
         "Larger values pull the camera back for a wider establishing shot.",
)
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
BALL_RADIUS = 1.5           # visible at close range
BALL_MASS = 25.0
LAUNCH_SPEED = 50.0         # m/s horizontal — covers typical flight distance quickly
VELOCITY_JITTER = 1.5
HEIGHT_LAYERS = 2
LAUNCH_RADIUS_MULT = 0.85   # fraction of bbox half-diagonal for the fixed launch ring
LAUNCH_MARGIN = 4.0         # added on top of the computed base radius
CAM_RADIUS_MULT = 1.05      # fixed orbit radius as a multiple of half-diagonal (IC-style)
CAM_HEIGHT_MULT  = 0.55     # moderate 3/4 view, constant altitude
CAM_ZBOB_MULT    = 0.08     # gentle up/down bob as camera orbits (IC-style)
CAM_FOCAL_LENGTH_MM = 22.0  # slightly tighter since camera is close
FACADE_SAMPLES = 72         # angular samples for per-angle facade distance (every 5°)
ORBIT_FRAMES = 600          # 20s at 30fps — slower, smoother rotation
FIRE_FRAME = 30             # 1s buildup: all balls visible and still, then all launch together
GRAVITY = 9.81              # m/s^2 — for ballistic arc compensation

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


def hide_matching_prims(stage, patterns):
    """Hide any prim under /World/Building whose path contains one of `patterns`.

    Sets `visibility=invisible` on each match so it disappears from render.
    Returns a set of hidden prim paths (strings). The caller passes this
    set to compute_building_geometry() so hidden geometry is also excluded
    from the camera-framing math.
    """
    hidden = set()
    if not patterns:
        return hidden
    per_pattern_counts = {p: 0 for p in patterns}
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
                per_pattern_counts[pat] += 1
                break
    return hidden, per_pattern_counts


def compute_building_geometry(building_prim, bbox_cache, hidden_paths=None):
    """Iterate every mesh under the building, returning (centroid, mesh_corners_xy).

    centroid -- volume-weighted center of mesh positions (more accurate
    than the bbox midpoint for irregular footprints).
    mesh_corners_xy -- shape (N*4, 2) array of the horizontal bbox
    corners of every mesh, used later to compute per-angle facade distances.

    If `hidden_paths` is provided, any mesh whose prim path starts with one
    of those paths is excluded (so hidden site/landscape geometry doesn't
    pull the centroid or inflate the orbit radius).
    """
    hidden_paths = hidden_paths or set()
    positions = []
    weights = []
    corners_xy = []
    for prim in Usd.PrimRange(building_prim):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        p = str(prim.GetPath())
        if any(p == h or p.startswith(h + "/") for h in hidden_paths):
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
        corners_xy.append([emin[0], emin[1]])
        corners_xy.append([emax[0], emin[1]])
        corners_xy.append([emin[0], emax[1]])
        corners_xy.append([emax[0], emax[1]])

    if len(positions) == 0:
        return None, None

    positions = np.array(positions)
    weights = np.array(weights)
    centroid = np.average(positions, axis=0, weights=weights)
    return centroid, np.array(corners_xy)


def facade_distances(centroid_xy, mesh_corners_xy, angles):
    """Per-angle max projection of mesh corners from centroid.

    Returns the distance from the centroid to the farthest piece of building
    geometry along each angle -- i.e., how far the facade extends outward
    along that ray.  Vectorized: one matrix multiply over all corners x angles.
    """
    offsets = mesh_corners_xy - np.asarray(centroid_xy, dtype=float)
    dirs = np.stack([np.cos(angles), np.sin(angles)], axis=0)  # (2, A)
    projections = offsets @ dirs  # (N*4, A)
    return projections.max(axis=0)  # (A,)


def interpolate_facade(angle, sample_angles, sample_dists):
    """Linear interpolation of facade distance at an arbitrary angle.

    sample_angles are evenly spaced in [0, 2*pi).  angle is wrapped mod 2*pi.
    """
    TWO_PI = 2.0 * math.pi
    a = angle % TWO_PI
    n = len(sample_angles)
    step = TWO_PI / n
    idx = a / step
    i0 = int(idx) % n
    i1 = (i0 + 1) % n
    frac = idx - int(idx)
    return sample_dists[i0] * (1.0 - frac) + sample_dists[i1] * frac


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

    # Strip out trees/landscape/entourage before anything else — they bias
    # the bbox, centroid, and facade distances, which pushes the camera
    # orbit too close to the main building volume.
    hide_patterns = [p.strip() for p in args.hide_patterns.split(",") if p.strip()]
    hidden_paths = set()
    if hide_patterns:
        hidden_paths, per_pattern = hide_matching_prims(stage, hide_patterns)
        log(f"\nHide patterns: {hide_patterns}")
        for pat, count in per_pattern.items():
            log(f"  '{pat}' -> {count} prims hidden")
        log(f"Total prims hidden: {len(hidden_paths)}")

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

    # Mesh-weighted centroid + mesh corners for per-angle facade distance —
    # the bbox midpoint can be far off for irregular building footprints.
    # Hidden prims (trees/landscape) are excluded so the orbit math reflects
    # the actual building volume, not surrounding site clutter.
    centroid, mesh_corners_xy = compute_building_geometry(
        building_prim, bbox_cache, hidden_paths=hidden_paths
    )
    if centroid is not None:
        bldg_center = centroid
        offset = np.linalg.norm(centroid[:2] - bbox_center[:2])
        log(f"Mesh centroid: ({centroid[0]:.1f}, {centroid[1]:.1f}, {centroid[2]:.1f})")
        log(f"Bbox center:   ({bbox_center[0]:.1f}, {bbox_center[1]:.1f}, {bbox_center[2]:.1f})")
        log(f"XY offset between centroid and bbox center: {offset:.1f}")
        log(f"Mesh corners collected: {len(mesh_corners_xy)}")
    else:
        bldg_center = bbox_center
        mesh_corners_xy = None
        log("WARNING: centroid computation failed, falling back to bbox center")

    # Precompute per-angle facade distance (how far the building extends
    # outward from the centroid along each azimuth).  Used for ball launch
    # positions only — camera keeps a fixed radius for a smooth orbit.
    sample_angles = np.linspace(0.0, 2.0 * math.pi, FACADE_SAMPLES, endpoint=False)
    if mesh_corners_xy is not None:
        facade_dists = facade_distances(bldg_center[:2], mesh_corners_xy, sample_angles)
        # Clip outlier meshes to the bbox half-diagonal — rare stray geometry
        # can report distances beyond the aggregate bbox and would push the
        # launch ring well outside the visible building.
        clip_max = half_diag * 1.05
        facade_dists = np.minimum(facade_dists, clip_max)
        log(f"Facade distance range: min={facade_dists.min():.1f}, "
            f"max={facade_dists.max():.1f}, mean={facade_dists.mean():.1f} "
            f"(clipped at {clip_max:.1f})")
    else:
        # Fallback: use half-diagonal uniformly
        facade_dists = np.full(FACADE_SAMPLES, half_diag)

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
    cam_height = float(bldg_center[2]) + float(bldg_extent[2]) * CAM_HEIGHT_MULT

    # Constant launch radius — every ball at the same distance from centroid.
    # Tie it to the bbox half-diagonal so the ring sits snug against the
    # bulk of the building; outlier-direction balls will spawn just inside
    # stray geometry and collide on launch (visually hidden under the volley).
    launch_radius = half_diag * LAUNCH_RADIUS_MULT + LAUNCH_MARGIN
    log(f"Constant launch radius: {launch_radius:.1f} "
        f"(half_diag {half_diag:.1f} * {LAUNCH_RADIUS_MULT} + {LAUNCH_MARGIN}m)")

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
                tangent_offset = (b - (balls_per_layer - 1) / 2.0) * 2.0
                tx = -math.sin(angle) * tangent_offset
                ty = math.cos(angle) * tangent_offset

                fire_pos = np.array([
                    float(bldg_center[0]) + launch_radius * math.cos(angle) + tx,
                    float(bldg_center[1]) + launch_radius * math.sin(angle) + ty,
                    launch_z,
                ])
                # Ball flies toward the building along the centroid ray.
                # Flight distance depends on the per-angle facade — ballistic
                # arc is tuned so the ball arrives at the wall at launch height.
                dx = float(bldg_center[0]) - fire_pos[0]
                dy = float(bldg_center[1]) - fire_pos[1]
                horiz_dist = math.sqrt(dx * dx + dy * dy)
                horiz_dir = np.array([dx / horiz_dist, dy / horiz_dist, 0.0])
                facade_r = float(interpolate_facade(angle, sample_angles, facade_dists))
                flight_dist = max(launch_radius - facade_r, 0.5)
                flight_time = flight_dist / LAUNCH_SPEED
                v_z_compensation = 0.5 * GRAVITY * flight_time
                jitter = (np.random.rand(3) - 0.5) * 2.0 * VELOCITY_JITTER
                jitter[2] *= 0.3  # less vertical jitter
                fire_vel = (horiz_dir * LAUNCH_SPEED
                            + np.array([0.0, 0.0, v_z_compensation])
                            + jitter)

                color = BALL_COLORS[(k + layer) % len(BALL_COLORS)]

                name = f"ball_{k:02d}_{layer}_{b}"
                ball = world.scene.add(DynamicSphere(
                    prim_path=f"/World/Balls/{name}",
                    name=name,
                    position=fire_pos,
                    radius=BALL_RADIUS,
                    color=np.array(color),
                    mass=BALL_MASS,
                ))
                site_balls.append({
                    "obj": ball,
                    "fire_pos": fire_pos,
                    "fire_vel": fire_vel,
                })
                total_spawned += 1

        sites.append({"angle": angle, "balls": site_balls})

    log(f"Total balls: {total_spawned} across {NUM_AZIMUTHS} launch sites")

    # Camera — fixed orbit radius (IC style) so framing stays consistent.
    cam_radius_mult = args.cam_radius_mult
    cam_radius = half_diag * cam_radius_mult
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
    log(f"Camera orbit radius={cam_radius:.1f} (mult={cam_radius_mult:.2f}), "
        f"height={cam_height:.1f}, focal={CAM_FOCAL_LENGTH_MM}mm")

    # --- Phase 3: Reset & warm-up ---
    world.reset()
    camera.initialize()

    # Zero initial velocities — balls are visible at fire_pos from frame 0.
    for site in sites:
        for bd in site["balls"]:
            bd["obj"].set_linear_velocity(np.array([0.0, 0.0, 0.0]))
            bd["obj"].set_angular_velocity(np.array([0.0, 0.0, 0.0]))

    log("\nWarming up (30 frames, balls pinned at fire positions)...")
    for _ in range(30):
        # Pin balls at their fire positions so gravity doesn't pull them down
        for site in sites:
            for bd in site["balls"]:
                bd["obj"].set_world_pose(position=bd["fire_pos"])
                bd["obj"].set_linear_velocity(np.array([0.0, 0.0, 0.0]))
                bd["obj"].set_angular_velocity(np.array([0.0, 0.0, 0.0]))
        world.step(render=True)

    # --- Phase 4: Orbit + single synchronized volley ---
    orbit_frames = ORBIT_FRAMES
    fired = False

    log(f"\nRecording {orbit_frames} frames ({orbit_frames/30:.0f}s @ 30fps)")
    log(f"Single volley: all {total_spawned} balls launch together at frame {FIRE_FRAME}")

    with VideoRecorder(video_name, fps=30) as recorder:
        for i in range(orbit_frames):
            # Camera — pure circular orbit at fixed radius (IC style),
            # with a gentle sinusoidal height bob for visual interest.
            angle = 2 * math.pi * i / orbit_frames
            height_bob = float(bldg_extent[2]) * CAM_ZBOB_MULT * math.sin(2 * angle)
            cam_x = float(bldg_center[0]) + cam_radius * math.cos(angle)
            cam_y = float(bldg_center[1]) + cam_radius * math.sin(angle)
            cam_z = cam_height + height_bob
            cam_pos = np.array([cam_x, cam_y, cam_z])
            orient = look_at_orientation(cam_pos, bldg_center)
            camera.set_world_pose(position=cam_pos, orientation=orient)

            if i < FIRE_FRAME:
                # Keep balls pinned at fire positions so they stay visible
                # and stationary until the synchronized launch.
                for site in sites:
                    for bd in site["balls"]:
                        bd["obj"].set_world_pose(position=bd["fire_pos"])
                        bd["obj"].set_linear_velocity(np.array([0.0, 0.0, 0.0]))
                        bd["obj"].set_angular_velocity(np.array([0.0, 0.0, 0.0]))
            elif i == FIRE_FRAME and not fired:
                # Fire every ball in the same physics step.
                for site in sites:
                    for bd in site["balls"]:
                        bd["obj"].set_world_pose(position=bd["fire_pos"])
                        bd["obj"].set_linear_velocity(bd["fire_vel"])
                        bd["obj"].set_angular_velocity(np.array([0.0, 0.0, 0.0]))
                fired = True
                log(f"  [fire] frame={i} launched {total_spawned} balls simultaneously")

            world.step(render=True)
            rgba = camera.get_rgba()
            if rgba is not None and i > 0:
                recorder.capture_rgba(rgba)

            if i % 60 == 0:
                log(f"  Frame {i}/{orbit_frames}  fired={fired}")

    log("\nRecording complete.")
    log(f"  Video  -> videos/{video_name}.mp4")
    log(f"  Report -> {report_path}")
    log("=" * 60)

    _report_file.close()
    simulation_app.close()


if __name__ == "__main__":
    main()

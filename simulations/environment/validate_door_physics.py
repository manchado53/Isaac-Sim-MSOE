#!/usr/bin/env python3
"""
Validation: Door Physics Test

For each test camera, finds nearby doors (within DOOR_SEARCH_RADIUS feet),
drives them to 80 degrees, measures final angle, and records video.

Usage:
    sbatch scripts/slurm/submit_isaac_job.sh simulations/environment/validate_door_physics.py <usd_path>

Output:
    logs/validation/door_physics_results.json
    videos/validation/validate_doors_cam_*.mp4
"""

import os
import sys
import json
import math
import time

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

import numpy as np
import omni.usd
from pxr import Usd, UsdGeom, UsdPhysics, Sdf, Gf

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.abspath(os.path.join(_script_dir, '..', '..'))
sys.path.insert(0, _project_dir)

from isaacsim.core.api import World
from isaacsim.sensors.camera import Camera as IsaacCamera
from sim_utils.test_cameras import (
    discover_test_cameras, CAMERA_FOCAL_LENGTH, CAMERA_RESOLUTION, CAMERA_FREQUENCY,
    create_test_camera, add_interior_lighting, look_at_orientation,
    set_focal_length_on_prim,
)
from sim_utils.video_capture import VideoRecorder
from sim_utils.building_physics import setup_physics_scene, set_drive_target, DOOR_DRIVE_PARAMS

DOOR_SEARCH_RADIUS = 20.0  # feet — max distance from camera target to door
DRIVE_TARGET_ANGLE = 80.0  # degrees
RECORD_FRAMES = 300        # ~10s at 30fps — records while doors swing open
PASS_THRESHOLD = 5.0       # degrees — minimum delta to count as "swung"


def get_door_angle_from_orient(prim):
    """Read the door's Z-axis rotation from its local xformOp:orient.

    IMPORTANT: Do NOT use ComputeLocalToWorldTransform — parent scale
    from Revit corrupts quaternion extraction.
    """
    xformable = UsdGeom.Xformable(prim)
    for op in xformable.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeOrient:
            quat = op.Get()
            if quat is None:
                return 0.0
            w = float(quat.GetReal())
            img = quat.GetImaginary()
            x, y, z = float(img[0]), float(img[1]), float(img[2])
            siny_cosp = 2.0 * (w * z + x * y)
            cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
            return math.degrees(math.atan2(siny_cosp, cosy_cosp))
    return 0.0


def find_driven_doors(stage):
    """Find all revolute joints with angular DriveAPI on Door prims.

    Returns list of (path_str, joint_prim, drive, door_prim).
    """
    doors = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.RevoluteJoint):
            continue
        path_str = str(prim.GetPath())
        if "Door" not in path_str:
            continue
        drive = UsdPhysics.DriveAPI.Get(prim, "angular")
        if not drive or not drive.GetStiffnessAttr().HasValue():
            continue
        joint = UsdPhysics.RevoluteJoint(prim)
        targets = joint.GetBody1Rel().GetTargets()
        if not targets:
            continue
        door_prim = stage.GetPrimAtPath(targets[0])
        if not door_prim.IsValid():
            continue
        doors.append((path_str, prim, drive, door_prim))
    return doors


def door_short_name(path_str):
    """Extract short door identifier from path (e.g. 'Doors_2520150')."""
    for part in path_str.split("/"):
        if part.startswith("Doors_") or part.startswith("Door_"):
            return part
    return path_str.split("/")[-2] if "/" in path_str else path_str


def door_world_position(door_prim):
    """Get the world-space center of a door's bounding box."""
    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]
    )
    bbox = bbox_cache.ComputeWorldBound(door_prim)
    box_range = bbox.ComputeAlignedRange()
    bb_min = box_range.GetMin()
    bb_max = box_range.GetMax()
    return np.array([
        (bb_min[0] + bb_max[0]) / 2.0,
        (bb_min[1] + bb_max[1]) / 2.0,
        (bb_min[2] + bb_max[2]) / 2.0,
    ])


def doors_near_camera(all_doors, cam_target, radius):
    """Filter doors within radius (feet) of camera target position."""
    nearby = []
    target = np.array(cam_target)
    for path_str, joint_prim, drive, door_prim in all_doors:
        door_pos = door_world_position(door_prim)
        dist = np.linalg.norm(door_pos - target)
        if dist <= radius:
            nearby.append((path_str, joint_prim, drive, door_prim, float(dist)))
    return nearby


def main():
    t_start = time.time()
    usd_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        _project_dir, "assets", "environments", "msoe_building",
        "innovation_center_physics.usd"
    )

    if not os.path.exists(usd_path):
        print(f"ERROR: USD not found: {usd_path}", flush=True)
        simulation_app.close()
        sys.exit(1)

    print(f"[door_physics] Loading: {usd_path}", flush=True)

    # --- Scene setup ---
    world = World(stage_units_in_meters=1.0)
    stage = omni.usd.get_context().get_stage()

    building_prim = stage.DefinePrim("/World/Building", "Xform")
    building_prim.GetReferences().AddReference(
        usd_path, primPath=Sdf.Path("/World/Building")
    )

    setup_physics_scene(stage)
    TEST_CAMERAS = discover_test_cameras(stage)
    print(f"[door_physics] Discovered {len(TEST_CAMERAS)} test cameras: "
          f"{list(TEST_CAMERAS.keys())}", flush=True)
    # Use lowest-floor camera as the showcase reference (was "G" for IC)
    SHOWCASE_KEY = next(iter(TEST_CAMERAS)) if TEST_CAMERAS else None
    add_interior_lighting(stage, TEST_CAMERAS)

    # --- Discover all driven doors ---
    all_doors = find_driven_doors(stage)
    print(f"[door_physics] Found {len(all_doors)} total driven doors", flush=True)

    # --- Create cameras ---
    cameras = {}
    for key, cam_dict in TEST_CAMERAS.items():
        cameras[key] = create_test_camera(key, cam_dict, "/World/DoorCam", stage=stage)

    # --- Assign doors to cameras ---
    cam_doors = {}
    for cam_key, cam_dict in TEST_CAMERAS.items():
        nearby = doors_near_camera(
            all_doors, cam_dict["target"], DOOR_SEARCH_RADIUS
        )
        cam_doors[cam_key] = nearby
        names = [door_short_name(p) for p, _, _, _, _ in nearby]
        print(f"  Camera {cam_key} ({cam_dict['room']}): "
              f"{len(nearby)} doors nearby", flush=True)
        for n in names:
            print(f"    - {n}", flush=True)

    # --- Showcase camera: close-up of nearest door to the showcase cam ---
    showcase_cam = None
    showcase_door_name = None
    if SHOWCASE_KEY and cam_doors.get(SHOWCASE_KEY):
        nearest = min(cam_doors[SHOWCASE_KEY], key=lambda x: x[4])
        path_str, _, _, door_prim, dist = nearest
        door_pos = door_world_position(door_prim)
        showcase_door_name = door_short_name(path_str)

        # Position camera 8m from door, on the showcase-cam side, at eye level
        cam_g = np.array(TEST_CAMERAS[SHOWCASE_KEY]["pos"])
        dir_to_door = np.array([door_pos[0] - cam_g[0], door_pos[1] - cam_g[1]])
        dir_norm = dir_to_door / np.linalg.norm(dir_to_door)

        sc_pos = np.array([
            door_pos[0] - dir_norm[0] * 8.0,
            door_pos[1] - dir_norm[1] * 8.0,
            door_pos[2] + 2.0,
        ])
        sc_target = np.array([door_pos[0], door_pos[1], door_pos[2]])
        sc_orient = look_at_orientation(sc_pos, sc_target)

        showcase_cam = IsaacCamera(
            prim_path="/World/DoorShowcase",
            position=sc_pos,
            frequency=CAMERA_FREQUENCY,
            resolution=CAMERA_RESOLUTION,
            orientation=sc_orient,
        )
        print(f"\n  Showcase: {showcase_door_name} at {dist:.1f}ft from cam G",
              flush=True)

    # --- Apply PhysxContactReportAPI to door prims (MUST be before world.reset) ---
    contact_report_enabled = False
    try:
        from pxr import PhysxSchema
        for path_str, joint_prim, drive, door_prim in all_doors:
            PhysxSchema.PhysxContactReportAPI.Apply(door_prim)
        contact_report_enabled = True
        print(f"[door_physics] Contact reporting enabled on {len(all_doors)} door prims", flush=True)
    except (ImportError, AttributeError) as e:
        print(f"[door_physics] Warning: Could not enable contact reporting: {e}", flush=True)

    # Build path lookup for contact report decoding
    door_prim_paths = {}  # prim path string -> joint path string
    for path_str, joint_prim, drive, door_prim in all_doors:
        door_prim_paths[str(door_prim.GetPath())] = path_str

    # --- Simulation ---
    world.reset()
    for cam in cameras.values():
        cam.initialize()
        set_focal_length_on_prim(stage, cam)
    if showcase_cam is not None:
        showcase_cam.initialize()
        set_focal_length_on_prim(stage, showcase_cam)

    # Warmup
    print("[door_physics] Warming up (30 frames)...", flush=True)
    for _ in range(30):
        world.step(render=True)

    # Record initial angles for ALL doors
    initial_angles = {}
    for path_str, joint_prim, drive, door_prim in all_doors:
        initial_angles[path_str] = get_door_angle_from_orient(door_prim)

    # --- Native metrics tracking setup ---
    prev_angles = dict(initial_angles)  # for numerical angular velocity
    door_peak_ang_vel = {p: 0.0 for p, _, _, _ in all_doors}
    door_final_ang_vel = {p: 0.0 for p, _, _, _ in all_doors}
    door_peak_torque = {p: 0.0 for p, _, _, _ in all_doors}
    door_contact_impulse = {p: 0.0 for p, _, _, _ in all_doors}
    door_max_penetration = {p: 0.0 for p, _, _, _ in all_doors}

    # Get drive parameters for torque computation
    # Default to single-door params (stiffness=300, damping=150)
    door_stiffness = {}
    door_damping = {}
    for path_str, joint_prim, drive, door_prim in all_doors:
        stiff_attr = drive.GetStiffnessAttr()
        damp_attr = drive.GetDampingAttr()
        door_stiffness[path_str] = float(stiff_attr.Get()) if stiff_attr.HasValue() else 300.0
        door_damping[path_str] = float(damp_attr.Get()) if damp_attr.HasValue() else 150.0

    # Contact report interface (for polling each frame)
    sim_interface = None
    if contact_report_enabled:
        try:
            from omni.physx import get_physx_simulation_interface
            from pxr import PhysicsSchemaTools
            sim_interface = get_physx_simulation_interface()
        except (ImportError, AttributeError):
            sim_interface = None

    dt = 1.0 / 60.0  # physics timestep

    # --- Record all cameras while doors swing open ---
    # Drive doors AND record simultaneously so the video shows the swing.
    print(f"[door_physics] Driving doors to {DRIVE_TARGET_ANGLE} deg + recording "
          f"({RECORD_FRAMES} frames)...", flush=True)

    # Set drive target on ALL doors — choose a target far from the initial angle
    # so doors that start already open get driven in the opposite direction.
    for path_str, joint_prim, drive, door_prim in all_doors:
        initial = initial_angles.get(path_str, 0.0)
        if abs(initial - DRIVE_TARGET_ANGLE) < 20.0:
            # Door is already near the target — drive the other way
            target = -DRIVE_TARGET_ANGLE
        else:
            target = DRIVE_TARGET_ANGLE
        set_drive_target(drive, target)

    # Open all recorders (including showcase) — wrapped in try/except
    # because Isaac Sim's Replicator/OmniGraph can intermittently crash
    recorders = {}
    showcase_rec = None
    recording = False
    try:
        for cam_key in cameras:
            vname = f"validation/validate_doors_cam_{cam_key}"
            recorders[cam_key] = VideoRecorder(vname, fps=30)
            recorders[cam_key].__enter__()
        if showcase_cam is not None:
            showcase_rec = VideoRecorder("validation/validate_doors_showcase", fps=30)
            showcase_rec.__enter__()
        recording = True
    except Exception as e:
        print(f"[door_physics] WARNING: Video recording setup failed ({e}), "
              f"running physics only", flush=True)
        recorders = {}
        showcase_rec = None

    for i in range(RECORD_FRAMES):
        world.step(render=True)
        if recording:
            for cam_key, cam in cameras.items():
                try:
                    rgba = cam.get_rgba()
                    if rgba is not None and i > 0:
                        recorders[cam_key].capture_rgba(rgba)
                except Exception:
                    pass
            if showcase_cam is not None and showcase_rec is not None and i > 0:
                try:
                    rgba = showcase_cam.get_rgba()
                    if rgba is not None:
                        showcase_rec.capture_rgba(rgba)
                except Exception:
                    pass

        # --- Per-frame native metrics (every 5th frame to limit overhead) ---
        if i % 5 == 0:
            for path_str, joint_prim, drive, door_prim in all_doors:
                cur_angle = get_door_angle_from_orient(door_prim)
                # Angular velocity (numerical derivative)
                ang_vel = abs(cur_angle - prev_angles.get(path_str, cur_angle)) / (5 * dt)
                prev_angles[path_str] = cur_angle
                if ang_vel > door_peak_ang_vel[path_str]:
                    door_peak_ang_vel[path_str] = ang_vel
                door_final_ang_vel[path_str] = ang_vel
                # Computed drive torque
                angle_error = DRIVE_TARGET_ANGLE - cur_angle
                torque = abs(door_stiffness[path_str] * angle_error
                             - door_damping[path_str] * math.radians(ang_vel))
                if torque > door_peak_torque[path_str]:
                    door_peak_torque[path_str] = torque

        # --- Contact report polling (every 10th frame) ---
        if sim_interface is not None and i % 10 == 0:
            try:
                headers, data, _ = sim_interface.get_contact_report()
                for h_idx in range(len(headers)):
                    header = headers[h_idx]
                    # Decode actor paths to check if a door is involved
                    try:
                        actor0_path = str(PhysicsSchemaTools.decodeSdfPath(
                            header.actor0[0], header.actor0[1]))
                        actor1_path = str(PhysicsSchemaTools.decodeSdfPath(
                            header.actor1[0], header.actor1[1]))
                    except Exception:
                        continue

                    # Match against known door prim paths
                    joint_path = door_prim_paths.get(actor0_path) or door_prim_paths.get(actor1_path)
                    if joint_path is None:
                        continue

                    # Accumulate contact data for this door
                    offset = header.contact_data_offset
                    for c_idx in range(header.num_contact_data):
                        cd = data[offset + c_idx]
                        imp = cd.impulse
                        impulse_mag = math.sqrt(imp[0]**2 + imp[1]**2 + imp[2]**2)
                        door_contact_impulse[joint_path] += impulse_mag
                        if cd.separation < door_max_penetration[joint_path]:
                            door_max_penetration[joint_path] = cd.separation
            except Exception as e:
                if i == 0:
                    print(f"[door_physics] Contact report error: {e}", flush=True)

        if i % 60 == 0:
            print(f"  Frame {i}/{RECORD_FRAMES}", flush=True)

    if recording:
        for rec in recorders.values():
            try:
                rec.__exit__(None, None, None)
            except Exception:
                pass
        if showcase_rec is not None:
            try:
                showcase_rec.__exit__(None, None, None)
            except Exception:
                pass

    # Measure final angles for ALL doors
    final_angles = {}
    for path_str, joint_prim, drive, door_prim in all_doors:
        final_angles[path_str] = get_door_angle_from_orient(door_prim)

    # --- Build results ---
    print(f"\n[door_physics] Analyzing results...", flush=True)
    camera_results = {}
    overall_pass = True

    for cam_key, cam_dict in TEST_CAMERAS.items():
        nearby = cam_doors[cam_key]

        if not nearby:
            camera_results[cam_key] = {
                "room": cam_dict["room"],
                "status": "SKIP",
                "doors_found": 0,
                "doors_passed": 0,
                "doors_failed": 0,
                "door_details": [],
                "video": f"validate_doors_cam_{cam_key}.mp4",
            }
            print(f"  Camera {cam_key}: no doors nearby [SKIP]", flush=True)
            continue

        doors_passed = 0
        doors_failed = 0
        door_details = []

        for path_str, joint_prim, drive, door_prim, dist in nearby:
            name = door_short_name(path_str)
            initial = initial_angles.get(path_str, 0.0)
            final = final_angles.get(path_str, 0.0)
            delta = abs(final - initial)
            passed = delta > PASS_THRESHOLD

            if passed:
                doors_passed += 1
            else:
                doors_failed += 1

            # Native metrics for this door
            peak_av = round(door_peak_ang_vel.get(path_str, 0.0), 1)
            final_av = round(door_final_ang_vel.get(path_str, 0.0), 1)
            contact_imp = round(door_contact_impulse.get(path_str, 0.0), 2)
            max_pen = round(door_max_penetration.get(path_str, 0.0), 4)
            peak_tq = round(door_peak_torque.get(path_str, 0.0), 1)
            # Stall: door didn't pass, has high contact impulse, and stopped moving
            stall = (not passed) and (contact_imp > 10.0) and (final_av < 1.0)

            door_details.append({
                "name": name,
                "path": path_str,
                "distance_from_camera": round(dist, 1),
                "initial_angle": round(initial, 1),
                "final_angle": round(final, 1),
                "delta": round(delta, 1),
                "passed": passed,
                "peak_angular_velocity": peak_av,
                "final_angular_velocity": final_av,
                "contact_impulse_total": contact_imp,
                "max_penetration_depth": max_pen,
                "peak_drive_torque": peak_tq,
                "stall_detected": stall,
            })

            status = "PASS" if passed else "FAIL"
            stall_str = " [STALL]" if stall else ""
            print(f"  {name:<25s} {initial:7.1f} -> {final:7.1f} "
                  f"(delta={delta:.1f}, torque={peak_tq:.0f}, "
                  f"contact={contact_imp:.1f}) [{status}]{stall_str}", flush=True)

        cam_passed = doors_failed == 0
        if not cam_passed:
            overall_pass = False

        camera_results[cam_key] = {
            "room": cam_dict["room"],
            "status": "PASS" if cam_passed else "FAIL",
            "doors_found": len(nearby),
            "doors_passed": doors_passed,
            "doors_failed": doors_failed,
            "door_details": door_details,
            "video": f"validate_doors_cam_{cam_key}.mp4",
        }

        print(f"  Camera {cam_key}: {doors_passed}/{len(nearby)} passed", flush=True)

    # --- Global door summary ---
    total_doors = len(all_doors)
    total_passed = sum(
        1 for p in all_doors
        if abs(final_angles.get(p[0], 0) - initial_angles.get(p[0], 0)) > PASS_THRESHOLD
    )
    total_failed = total_doors - total_passed
    total_stalled = sum(
        1 for p in all_doors
        if (abs(final_angles.get(p[0], 0) - initial_angles.get(p[0], 0)) <= PASS_THRESHOLD
            and door_contact_impulse.get(p[0], 0) > 10.0
            and door_final_ang_vel.get(p[0], 0) < 1.0)
    )

    # --- Save results ---
    results = {
        "test": "door_physics",
        "usd_file": os.path.basename(usd_path),
        "global_summary": {
            "total_doors": total_doors,
            "total_passed": total_passed,
            "total_failed": total_failed,
            "total_stalled": total_stalled,
        },
        "cameras": camera_results,
        "showcase": {
            "door": showcase_door_name,
            "video": "validate_doors_showcase.mp4",
        } if showcase_door_name else None,
        "overall_passed": overall_pass,
        "runtime_seconds": round(time.time() - t_start, 1),
    }

    out_dir = os.path.join(_project_dir, "logs", "validation")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "door_physics_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[door_physics] Global: {total_passed}/{total_doors} doors passed", flush=True)
    print(f"[door_physics] Results -> {out_path}", flush=True)
    print(f"[door_physics] OVERALL: {'PASS' if overall_pass else 'FAIL'}", flush=True)

    # Skip simulation_app.close() — Isaac Sim's OmniGraph cleanup can segfault
    os._exit(0)


if __name__ == "__main__":
    main()

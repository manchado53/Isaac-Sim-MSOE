#!/usr/bin/env python3
"""
Diagnose the "fundamentally wrong" back-corridor door.

Loads the same physics USD the door showcase uses, gathers every
physics-controlled door within a radius of the hero target, and runs
static + runtime checks on each:

  1. Joint config dump (axis, limits, localPos/Rot).
  2. Hinge-offset mismatch: re-estimate where the hinge should be and
     compare against what's stored on the joint.
  3. Neighbor walls: paths + separation distance of static walls within
     0.5 m of the door's expanded bbox.
  4. Post-settle angle: run 2 s of physics at commanded 0°, read the body's
     actual angle. Any door ≥5° from 0° is stuck open.
  5. Drive test: command 60° for 2 s, log max achieved + final angle.

Outputs:
  - logs/door_diagnosis_<stem>.json  — structured results
  - logs/door_diagnosis_<stem>.txt   — human-readable, suspects first
  - images/diagnostics/door_diag/<prim_name>_{0deg,60deg}.png for top-5

Usage:
    sbatch scripts/slurm/submit_isaac_job.sh tools/diagnose_back_doors.py \
        --input assets/environments/msoe_building/MSOEBuildingV4_physics.usd
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

parser = argparse.ArgumentParser(description="Diagnose physics-controlled doors near a target.")
parser.add_argument("--input", required=True, help="Physics-ready USD")
parser.add_argument("--target", default="-16.5,78.17,23.75",
                    help="World target 'X,Y,Z' (default: hero camera target)")
parser.add_argument("--radius", type=float, default=15.0,
                    help="Consider doors within R meters of target (default 15)")
parser.add_argument("--settle-frames", type=int, default=60,
                    help="Steps at 0° before reading post-settle angle (default 60 = 2 s)")
parser.add_argument("--drive-frames", type=int, default=60,
                    help="Steps commanding 60° (0 to skip drive test)")
parser.add_argument("--top-renders", type=int, default=5,
                    help="Render close-ups for the top-N suspects (default 5)")
args, _unknown = parser.parse_known_args()

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

import numpy as np
import omni.usd
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics
from isaacsim.core.api import World
from isaacsim.sensors.camera import Camera

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.abspath(os.path.join(_script_dir, ".."))
sys.path.insert(0, _project_dir)

from sim_utils.building_physics import estimate_door_hinge_offset, set_drive_target
from sim_utils.test_cameras import look_at_orientation


def parse3(s):
    return np.array([float(p.strip()) for p in s.split(",")])


def world_center_range(prim, bbox_cache):
    rng = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
    if rng.IsEmpty():
        return None, None, None
    mn = np.array([rng.GetMin()[0], rng.GetMin()[1], rng.GetMin()[2]])
    mx = np.array([rng.GetMax()[0], rng.GetMax()[1], rng.GetMax()[2]])
    return (mn + mx) / 2.0, mn, mx


def find_door_joint(door_prim):
    for child in door_prim.GetChildren():
        name = child.GetName()
        if ("Hinge" in name or "Joint" in name) and child.IsA(UsdPhysics.RevoluteJoint):
            return child
    return None


def find_drive_on_joint(joint_prim):
    drive = UsdPhysics.DriveAPI(joint_prim, "angular")
    if drive.GetTargetPositionAttr().IsValid():
        return drive
    return None


def quat_axis_angle_deg(q):
    """Return (axis_xyz, angle_deg) for a Gf.Quatf/Quatd."""
    w = float(q.GetReal())
    imag = q.GetImaginary()
    v = np.array([float(imag[0]), float(imag[1]), float(imag[2])])
    w = max(-1.0, min(1.0, w))
    angle = 2.0 * math.acos(w)
    s = math.sqrt(max(0.0, 1.0 - w * w))
    if s < 1e-6:
        axis = np.array([0.0, 0.0, 1.0])
    else:
        axis = v / s
    return axis, math.degrees(angle)


def read_joint_angle_deg(door_prim, joint_axis_char):
    """Approximate current joint angle from door prim's xformOp:orient.

    The physics USD sets an initial xformOp:orient identity on the door,
    and PhysX writes back to the xform each step. Extract rotation around
    the joint's axis (X/Y/Z char).
    """
    xform = UsdGeom.Xformable(door_prim)
    ops = xform.GetOrderedXformOps()
    for op in ops:
        if op.GetOpType() == UsdGeom.XformOp.TypeOrient:
            q = op.Get()
            if q is None:
                return 0.0
            axis, ang = quat_axis_angle_deg(q)
            idx = {"X": 0, "Y": 1, "Z": 2}.get(joint_axis_char, 2)
            signed = ang * np.sign(axis[idx]) if abs(axis[idx]) > 0.1 else 0.0
            return float(signed)
    return 0.0


def hide_obstructing(stage):
    patterns = ("Tree", "Planting", "Vegetation", "Entourage", "RPC",
                "Plant", "Shrub", "Bush", "Landscape", "Topography", "Site_")
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if not path.startswith("/World/Building"):
            continue
        if any(p in path for p in patterns):
            img = UsdGeom.Imageable(prim)
            if img:
                img.CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)


def add_lighting(stage):
    UsdLux.DistantLight.Define(stage, Sdf.Path("/World/DiagSun")).CreateIntensityAttr(3500.0)
    UsdLux.DomeLight.Define(stage, Sdf.Path("/World/DiagDome")).CreateIntensityAttr(6000.0)


def bbox_distance(a_mn, a_mx, b_mn, b_mx):
    """Minimum separation between two AABBs; 0 if they overlap."""
    dx = max(0.0, max(a_mn[0] - b_mx[0], b_mn[0] - a_mx[0]))
    dy = max(0.0, max(a_mn[1] - b_mx[1], b_mn[1] - a_mx[1]))
    dz = max(0.0, max(a_mn[2] - b_mx[2], b_mn[2] - a_mx[2]))
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def collect_walls(stage, bbox_cache):
    walls = []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if "/Walls/" not in path:
            continue
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        _, mn, mx = world_center_range(prim, bbox_cache)
        if mn is None:
            continue
        walls.append((path, mn, mx))
    return walls


def main():
    target = parse3(args.target)
    input_usd = os.path.abspath(args.input)
    stem = os.path.splitext(os.path.basename(input_usd))[0]

    report_json = os.path.join(_project_dir, "logs", f"door_diagnosis_{stem}.json")
    report_txt = os.path.join(_project_dir, "logs", f"door_diagnosis_{stem}.txt")
    diag_dir = os.path.join(_project_dir, "images", "diagnostics", "door_diag")
    os.makedirs(os.path.dirname(report_json), exist_ok=True)
    os.makedirs(diag_dir, exist_ok=True)

    print("=" * 70, flush=True)
    print(f"Door diagnosis — {stem}", flush=True)
    print(f"Target: {target.tolist()}   Radius: {args.radius} m", flush=True)
    print("=" * 70, flush=True)

    world = World(stage_units_in_meters=1.0)
    stage = omni.usd.get_context().get_stage()
    building_prim = stage.DefinePrim("/World/Building", "Xform")
    building_prim.GetReferences().AddReference(input_usd, Sdf.Path("/World/Building"))
    print("Building loaded.", flush=True)

    hide_obstructing(stage)
    add_lighting(stage)

    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])

    # Gather every physics-controlled door within radius.
    doors = []
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        joint = find_door_joint(prim)
        if joint is None:
            continue
        drive = find_drive_on_joint(joint)
        if drive is None:
            continue
        center, mn, mx = world_center_range(prim, bbox_cache)
        if center is None:
            continue
        dist = float(np.linalg.norm(center - target))
        if dist > args.radius:
            continue
        doors.append({
            "prim": prim, "joint": joint, "drive": drive,
            "center": center, "bbox_min": mn, "bbox_max": mx, "dist": dist,
        })

    print(f"Found {len(doors)} doors within {args.radius} m of target.", flush=True)
    if not doors:
        print("ERROR: no doors to inspect.", flush=True)
        simulation_app.close()
        return

    doors.sort(key=lambda d: d["dist"])

    # ---------------- Static analysis ----------------
    walls = collect_walls(stage, bbox_cache)
    print(f"Indexed {len(walls)} static wall colliders.", flush=True)

    for d in doors:
        joint = d["joint"]
        joint_api = UsdPhysics.RevoluteJoint(joint)
        axis_char = joint_api.GetAxisAttr().Get() or "Z"
        lo = joint_api.GetLowerLimitAttr().Get()
        hi = joint_api.GetUpperLimitAttr().Get()
        lp0 = joint_api.GetLocalPos0Attr().Get()
        lp1 = joint_api.GetLocalPos1Attr().Get()
        lr0 = joint_api.GetLocalRot0Attr().Get()
        lr1 = joint_api.GetLocalRot1Attr().Get()

        drive = d["drive"]
        stiffness = drive.GetStiffnessAttr().Get()
        damping = drive.GetDampingAttr().Get()
        max_force = drive.GetMaxForceAttr().Get()
        target_pos = drive.GetTargetPositionAttr().Get()

        # Hinge re-estimate
        est_lp0, est_lp1, _, _ = estimate_door_hinge_offset(stage, d["prim"])
        if est_lp0 is not None and lp0 is not None:
            stored_world = np.array([float(lp0[0]), float(lp0[1]), float(lp0[2])])
            est_world = np.array([float(est_lp0[0]), float(est_lp0[1]), float(est_lp0[2])])
            hinge_mismatch = float(np.linalg.norm(stored_world - est_world))
        else:
            hinge_mismatch = None

        # Neighbor walls within 0.5 m of door's bbox
        door_mn = d["bbox_min"] - 0.5
        door_mx = d["bbox_max"] + 0.5
        neighbors = []
        for path, w_mn, w_mx in walls:
            sep = bbox_distance(door_mn, door_mx, w_mn, w_mx)
            if sep <= 0.5:
                neighbors.append((path, sep))
        neighbors.sort(key=lambda t: t[1])

        d["static"] = {
            "axis": axis_char, "lower": lo, "upper": hi,
            "localPos0": list(map(float, lp0)) if lp0 else None,
            "localPos1": list(map(float, lp1)) if lp1 else None,
            "localRot0": (float(lr0.GetReal()), *[float(v) for v in lr0.GetImaginary()]) if lr0 else None,
            "localRot1": (float(lr1.GetReal()), *[float(v) for v in lr1.GetImaginary()]) if lr1 else None,
            "drive": {"stiffness": stiffness, "damping": damping,
                      "max_force": max_force, "target": target_pos},
            "hinge_mismatch_m": hinge_mismatch,
            "neighbor_walls": [(p, round(s, 3)) for p, s in neighbors[:5]],
            "bbox_min": [round(float(v), 3) for v in d["bbox_min"]],
            "bbox_max": [round(float(v), 3) for v in d["bbox_max"]],
        }

    # ---------------- Runtime phase ----------------
    # Reset sets all drives to 0 — make sure each of our doors reads target=0.
    for d in doors:
        set_drive_target(d["drive"], 0.0)

    world.reset()
    print(f"Settling {args.settle_frames} frames at 0°...", flush=True)
    for _ in range(args.settle_frames):
        world.step(render=False)

    for d in doors:
        axis_char = d["static"]["axis"]
        d["post_settle_angle_deg"] = read_joint_angle_deg(d["prim"], axis_char)

    if args.drive_frames > 0:
        print(f"Driving to 60° for {args.drive_frames} frames...", flush=True)
        for d in doors:
            set_drive_target(d["drive"], 60.0)
        max_angles = {id(d): 0.0 for d in doors}
        for _ in range(args.drive_frames):
            world.step(render=False)
            for d in doors:
                a = read_joint_angle_deg(d["prim"], d["static"]["axis"])
                max_angles[id(d)] = max(max_angles[id(d)], abs(a))
        for d in doors:
            d["drive_test"] = {
                "max_abs_angle_deg": float(max_angles[id(d)]),
                "final_angle_deg": read_joint_angle_deg(d["prim"], d["static"]["axis"]),
            }

    # Suspect scoring: bigger is more suspicious
    def score(d):
        s = 0.0
        hm = d["static"].get("hinge_mismatch_m")
        if hm is not None:
            s += hm * 10.0
        settle = abs(d.get("post_settle_angle_deg", 0.0))
        if settle > 5.0:
            s += settle
        if "drive_test" in d:
            achieved = d["drive_test"]["max_abs_angle_deg"]
            if achieved < 30.0:  # drive couldn't reach even half of 60°
                s += (30.0 - achieved)
        return s

    doors.sort(key=score, reverse=True)

    # ---------------- Close-up renders ----------------
    top = doors[: args.top_renders]
    if top:
        cam_path = "/World/DiagCam"
        first = top[0]
        cam_pos = first["center"] + np.array([3.0, 0.0, 0.5])
        camera = Camera(
            prim_path=cam_path,
            position=cam_pos,
            frequency=30,
            resolution=(1280, 720),
            orientation=look_at_orientation(cam_pos, first["center"]),
        )
        UsdGeom.Camera(stage.GetPrimAtPath(cam_path)).CreateFocalLengthAttr(24.0)
        camera.initialize()
        for _ in range(10):
            world.step(render=True)

        from PIL import Image
        for d in top:
            # Two renders per door: 0° and 60°
            for ang in (0.0, 60.0):
                for dd in doors:  # reset all drives so neighbors don't move
                    set_drive_target(dd["drive"], 0.0)
                set_drive_target(d["drive"], ang)
                for _ in range(45):
                    world.step(render=True)
                door_center = d["center"]
                # Position camera 2.5 m back from door, offset slightly up.
                approach = np.array([2.5, 0.0, 0.5])
                cp = door_center + approach
                camera.set_world_pose(position=cp,
                                      orientation=look_at_orientation(cp, door_center))
                for _ in range(3):
                    world.step(render=True)
                rgba = camera.get_rgba()
                if rgba is None or rgba.size == 0:
                    continue
                safe = str(d["prim"].GetPath()).replace("/", "_").strip("_")
                out = os.path.join(diag_dir, f"{safe}_{int(ang)}deg.png")
                Image.fromarray(rgba[:, :, :3].astype("uint8")).save(out)
                print(f"  wrote {out}", flush=True)

    # ---------------- Serialize ----------------
    def door_to_dict(d):
        out = {
            "prim": str(d["prim"].GetPath()),
            "joint": str(d["joint"].GetPath()),
            "dist_to_target_m": round(d["dist"], 3),
            "center": [round(float(v), 3) for v in d["center"]],
            "static": d["static"],
            "post_settle_angle_deg": round(d.get("post_settle_angle_deg", 0.0), 2),
            "suspect_score": round(score(d), 3),
        }
        if "drive_test" in d:
            out["drive_test"] = {
                "max_abs_angle_deg": round(d["drive_test"]["max_abs_angle_deg"], 2),
                "final_angle_deg": round(d["drive_test"]["final_angle_deg"], 2),
            }
        return out

    entries = [door_to_dict(d) for d in doors]
    with open(report_json, "w") as f:
        json.dump({"input": input_usd, "target": target.tolist(),
                   "radius": args.radius, "doors": entries}, f, indent=2)
    print(f"Wrote {report_json}", flush=True)

    with open(report_txt, "w") as f:
        f.write(f"# Door diagnosis — {stem}\n")
        f.write(f"# target={target.tolist()}  radius={args.radius} m  "
                f"{len(entries)} doors\n\n")
        for i, e in enumerate(entries, 1):
            f.write(f"[{i}] suspect_score={e['suspect_score']}  "
                    f"dist={e['dist_to_target_m']} m\n")
            f.write(f"    prim: {e['prim']}\n")
            hm = e["static"].get("hinge_mismatch_m")
            f.write(f"    hinge_mismatch: {hm}\n")
            f.write(f"    post_settle_angle: {e['post_settle_angle_deg']}°\n")
            if "drive_test" in e:
                f.write(f"    drive_test (cmd 60°): max={e['drive_test']['max_abs_angle_deg']}°  "
                        f"final={e['drive_test']['final_angle_deg']}°\n")
            nb = e["static"]["neighbor_walls"]
            f.write(f"    neighbor_walls ({len(nb)} within 0.5 m):\n")
            for p, s in nb:
                f.write(f"      {s:.3f} m  {p}\n")
            f.write("\n")
    print(f"Wrote {report_txt}", flush=True)

    # Print the top-3 suspects to stdout so the SLURM log is useful at a glance
    print("\n=== TOP SUSPECTS ===", flush=True)
    for e in entries[:3]:
        print(f"  {e['suspect_score']:.2f}  settle={e['post_settle_angle_deg']}°  "
              f"mismatch={e['static']['hinge_mismatch_m']}  "
              f"{e['prim']}", flush=True)

    simulation_app.close()


if __name__ == "__main__":
    main()

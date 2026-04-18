#!/usr/bin/env python3
"""
Physics Validation Pipeline Runner

Submits separate SLURM jobs for each validation test, waits for
completion, aggregates JSON results into a final report.

Usage:
    python validate_physics.py --usd <path_to_physics.usd>
    python validate_physics.py --usd assets/environments/msoe_building/innovation_center_physics.usd
    python validate_physics.py --usd innovation_center_physics.usd --skip robot_movement

Options:
    --usd         Path to the physics USD file (required)
    --partition   SLURM partition (default: teaching)
    --timeout     Max seconds to wait per job (default: 7200)
    --skip        Comma-separated test names to skip
    --dry-run     Print commands without submitting
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.resolve()

TEST_SCRIPTS = {
    "building_stats": "simulations/environment/validate_building_stats.py",
    "element_preservation": "simulations/environment/validate_element_preservation.py",
    "cube_drop": "simulations/environment/validate_cube_drop.py",
    "robot_movement": "simulations/environment/validate_robot_movement.py",
    "door_physics": "simulations/environment/validate_door_physics.py",
}

RESULT_FILES = {
    "building_stats": "logs/validation/building_stats_results.json",
    "element_preservation": "logs/validation/element_preservation_results.json",
    "cube_drop": "logs/validation/cube_drop_results.json",
    "robot_movement": "logs/validation/robot_movement_results.json",
    "door_physics": "logs/validation/door_physics_results.json",
}

POLL_INTERVAL = 30  # seconds between squeue checks


def resolve_usd_path(usd_arg):
    """Resolve the USD path to an absolute path."""
    path = Path(usd_arg)
    if path.is_absolute() and path.exists():
        return str(path)
    # Try relative to project dir
    candidate = PROJECT_DIR / path
    if candidate.exists():
        return str(candidate)
    # Try in the default building assets folder
    candidate = PROJECT_DIR / "assets" / "environments" / "msoe_building" / path.name
    if candidate.exists():
        return str(candidate)
    # Return as-is and let the test script fail with a clear error
    return str(Path(usd_arg).resolve())


def submit_job(test_name, script_path, usd_path, partition="teaching"):
    """Submit a SLURM job and return the job ID."""
    cmd = [
        "sbatch",
        f"--partition={partition}",
        str(PROJECT_DIR / "submit_validation_job.sh"),
        script_path,
        usd_path,
    ]
    print(f"  [{test_name}] Submitting: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_DIR))

    if result.returncode != 0:
        print(f"  [{test_name}] ERROR submitting job: {result.stderr.strip()}")
        return None

    # Parse "Submitted batch job 12345"
    match = re.search(r"Submitted batch job (\d+)", result.stdout)
    if not match:
        print(f"  [{test_name}] Could not parse job ID from: {result.stdout.strip()}")
        return None

    job_id = match.group(1)
    print(f"  [{test_name}] Job ID: {job_id}")
    return job_id


def check_job_status(job_id):
    """Check SLURM job status. Returns status string or None if not in queue."""
    result = subprocess.run(
        ["squeue", "-j", job_id, "-h", "-o", "%T"],
        capture_output=True, text=True
    )
    status = result.stdout.strip()
    if not status:
        return None  # Job no longer in queue (completed or failed)
    return status


def get_job_exit_info(job_id):
    """Get exit code and state from sacct after job completes."""
    result = subprocess.run(
        ["sacct", "-j", job_id, "--format=JobID,State,ExitCode", "-n", "-P"],
        capture_output=True, text=True
    )
    for line in result.stdout.strip().split("\n"):
        parts = line.split("|")
        if len(parts) >= 3 and parts[0] == job_id:
            return {"state": parts[1], "exit_code": parts[2]}
    return {"state": "UNKNOWN", "exit_code": "?"}


def wait_for_jobs(jobs, timeout=7200):
    """Wait for all SLURM jobs to complete. Returns dict of test_name -> exit_info."""
    pending = dict(jobs)  # test_name -> job_id
    results = {}
    start = time.time()

    print(f"\nWaiting for {len(pending)} jobs (timeout={timeout}s, polling every {POLL_INTERVAL}s)...")

    while pending and (time.time() - start) < timeout:
        time.sleep(POLL_INTERVAL)
        elapsed = time.time() - start

        still_running = {}
        for test_name, job_id in pending.items():
            status = check_job_status(job_id)
            if status is None:
                # Job finished
                exit_info = get_job_exit_info(job_id)
                results[test_name] = exit_info
                print(f"  [{test_name}] Job {job_id} finished: "
                      f"{exit_info['state']} (exit={exit_info['exit_code']}) "
                      f"[{elapsed:.0f}s]")
            else:
                still_running[test_name] = job_id
                if int(elapsed) % 120 < POLL_INTERVAL:
                    print(f"  [{test_name}] Job {job_id}: {status} [{elapsed:.0f}s]")

        pending = still_running

    # Timeout any remaining
    for test_name, job_id in pending.items():
        results[test_name] = {"state": "TIMEOUT", "exit_code": "?"}
        print(f"  [{test_name}] Job {job_id}: TIMEOUT after {timeout}s")

    return results


def load_result(test_name):
    """Load a test's JSON result file. Returns dict or None."""
    path = PROJECT_DIR / RESULT_FILES[test_name]
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def generate_report(usd_path, test_results, job_exit_info, skip_list):
    """Generate the final aggregated report (JSON + text)."""
    report = {
        "pipeline": "physics_validation",
        "usd_file": os.path.basename(usd_path),
        "usd_path": usd_path,
        "timestamp": datetime.now().isoformat(),
        "tests": {},
        "skipped": skip_list,
    }

    all_passed = True
    test_summaries = []

    for test_name in TEST_SCRIPTS:
        if test_name in skip_list:
            report["tests"][test_name] = {"status": "SKIPPED"}
            test_summaries.append((test_name, "SKIP", "Skipped by user"))
            continue

        result_data = test_results.get(test_name)
        exit_info = job_exit_info.get(test_name, {})

        if result_data is None:
            report["tests"][test_name] = {
                "status": "ERROR",
                "error": "Result file not found",
                "job_info": exit_info,
            }
            all_passed = False
            test_summaries.append((test_name, "ERROR", f"No result file (job: {exit_info.get('state', '?')})"))
            continue

        passed = result_data.get("overall_passed", False)
        if not passed:
            all_passed = False

        report["tests"][test_name] = {
            "status": "PASS" if passed else "FAIL",
            "data": result_data,
            "job_info": exit_info,
        }

        # Build per-camera summary
        cam_summary = []
        cameras = result_data.get("cameras", {})
        for cam_key in sorted(cameras.keys()):
            cam = cameras[cam_key]
            room = cam.get("room", f"Camera {cam_key}")

            if test_name == "building_stats":
                cam_status = "OK" if cam.get("non_black") else "BLACK"
                cam_summary.append(f"    Camera {cam_key} ({room}): {cam_status}")
            elif test_name == "cube_drop":
                landed = cam.get("cubes_landed", 0)
                total = cam.get("cubes_total", 0)
                cam_status = "PASS" if cam.get("passed") else "FAIL"
                cam_summary.append(f"    Camera {cam_key} ({room}): {landed}/{total} landed [{cam_status}]")
            elif test_name == "robot_movement":
                z_drop = cam.get("z_drop", 0)
                cam_status = "PASS" if cam.get("passed") else "FAIL"
                cam_summary.append(f"    Camera {cam_key} ({room}): z_drop={z_drop:.3f} [{cam_status}]")
            elif test_name == "door_physics":
                found = cam.get("doors_found", 0)
                dp = cam.get("doors_passed", 0)
                df = cam.get("doors_failed", 0)
                status = cam.get("status", "?")
                if found == 0:
                    cam_summary.append(f"    Camera {cam_key} ({room}): no doors nearby [SKIP]")
                else:
                    cam_summary.append(f"    Camera {cam_key} ({room}): {dp}/{found} doors [{status}]")

        status_str = "PASS" if passed else "FAIL"
        detail = "\n".join(cam_summary)
        test_summaries.append((test_name, status_str, detail))

    report["overall_passed"] = all_passed

    # --- Write JSON report ---
    report_dir = PROJECT_DIR / "logs" / "validation"
    report_dir.mkdir(parents=True, exist_ok=True)

    json_path = report_dir / "validation_report.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    # --- Write text report ---
    lines = []
    sep = "=" * 70
    lines.append(sep)
    lines.append("PHYSICS VALIDATION REPORT")
    lines.append(f"USD: {os.path.basename(usd_path)}")
    lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(sep)

    test_labels = {
        "building_stats": "Building Stats",
        "element_preservation": "Element Preservation",
        "cube_drop": "Cube Drop",
        "robot_movement": "Robot Movement",
        "door_physics": "Door Physics",
    }

    tests_passed = 0
    tests_total = 0

    for test_name, status, detail in test_summaries:
        label = test_labels.get(test_name, test_name)
        lines.append("")

        if status == "SKIP":
            lines.append(f"TEST: {label} {'.' * (50 - len(label))} SKIP")
            continue

        tests_total += 1
        if status == "PASS":
            tests_passed += 1

        dots = "." * (50 - len(label))
        lines.append(f"TEST: {label} {dots} {status}")

        # Add building stats details
        if test_name == "building_stats":
            result_data = test_results.get(test_name, {})
            stats = result_data.get("stats", {})
            if stats:
                lines.append(f"    Colliders: {stats.get('collider_count', '?')}")
                lines.append(f"    Rigid bodies: {stats.get('rigid_body_count', '?')}")
                lines.append(f"    Revolute joints: {stats.get('revolute_joint_count', '?')}")
                lines.append(f"    Driven joints: {stats.get('driven_joint_count', '?')}")
                lines.append(f"    Meshes: {stats.get('mesh_count', '?')}")
                lines.append(f"    Floor elements: {stats.get('floor_element_count', '?')}")
                lines.append(f"    Door elements: {stats.get('door_element_count', '?')}")
                dims = stats.get("building_dimensions_m",
                                 stats.get("building_dimensions_ft", []))
                if dims:
                    lines.append(f"    Building size: {dims[0]} x {dims[1]} x {dims[2]} m")

        # Add element preservation details
        if test_name == "element_preservation":
            result_data = test_results.get(test_name, {})
            summary = result_data.get("summary", {})
            if summary:
                lines.append(f"    Elements: {summary.get('total_elements', '?')} "
                             f"(manifest: {summary.get('manifest_total_elements', '?')})")
                lines.append(f"    Meshes: {summary.get('mesh_count', '?')} "
                             f"(manifest: {summary.get('manifest_mesh_count', '?')})")
                lines.append(f"    Checks: {summary.get('checks_passed', '?')}/"
                             f"{summary.get('checks_total', '?')} passed")

        # Add door global summary
        if test_name == "door_physics":
            result_data = test_results.get(test_name, {})
            gs = result_data.get("global_summary", {})
            if gs:
                lines.append(f"    Global: {gs.get('total_passed', '?')}/{gs.get('total_doors', '?')} doors passed")

        if detail:
            lines.append(detail)

    lines.append("")
    lines.append(sep)
    overall_str = "PASS" if all_passed else "FAIL"
    lines.append(f"OVERALL: {overall_str} ({tests_passed}/{tests_total} tests passed)")
    lines.append(sep)

    # List generated videos
    validation_videos_dir = PROJECT_DIR / "videos" / "validation"
    if validation_videos_dir.exists():
        video_files = sorted(validation_videos_dir.glob("validate_*.*"))
        if video_files:
            lines.append("")
            lines.append("Videos:")
            for v in video_files:
                lines.append(f"  {v.relative_to(PROJECT_DIR)}")

    lines.append("")

    txt_path = report_dir / "validation_report.txt"
    with open(txt_path, "w") as f:
        f.write("\n".join(lines))

    return json_path, txt_path, all_passed


def main():
    parser = argparse.ArgumentParser(description="Physics Validation Pipeline")
    parser.add_argument("--usd", required=True, help="Path to physics USD file")
    parser.add_argument("--partition", default="teaching", help="SLURM partition")
    parser.add_argument("--timeout", type=int, default=7200, help="Max seconds per job")
    parser.add_argument("--skip", default="", help="Comma-separated test names to skip")
    parser.add_argument("--dry-run", action="store_true", help="Print commands only")
    args = parser.parse_args()

    usd_path = resolve_usd_path(args.usd)
    skip_list = [s.strip() for s in args.skip.split(",") if s.strip()]

    print("=" * 70)
    print("PHYSICS VALIDATION PIPELINE")
    print("=" * 70)
    print(f"USD:       {usd_path}")
    print(f"Partition: {args.partition}")
    print(f"Timeout:   {args.timeout}s")
    if skip_list:
        print(f"Skipping:  {', '.join(skip_list)}")
    print()

    # Validate USD exists
    if not os.path.exists(usd_path):
        print(f"ERROR: USD file not found: {usd_path}")
        sys.exit(1)

    # Ensure output directories
    (PROJECT_DIR / "logs" / "validation").mkdir(parents=True, exist_ok=True)
    validation_videos = PROJECT_DIR / "videos" / "validation"
    validation_videos.mkdir(parents=True, exist_ok=True)

    # Clear previous results (JSON + videos/screenshots)
    for test_name, result_file in RESULT_FILES.items():
        path = PROJECT_DIR / result_file
        if path.exists():
            path.unlink()
            print(f"  Cleared previous: {result_file}")

    for old_video in validation_videos.glob("validate_*.*"):
        old_video.unlink()
        print(f"  Cleared previous: {old_video.relative_to(PROJECT_DIR)}")

    # --- Submit jobs ---
    print("\nSubmitting validation jobs...")
    jobs = {}
    for test_name, script_path in TEST_SCRIPTS.items():
        if test_name in skip_list:
            print(f"  [{test_name}] Skipped")
            continue

        if args.dry_run:
            print(f"  [{test_name}] DRY RUN: sbatch submit_validation_job.sh "
                  f"{script_path} {usd_path}")
            continue

        job_id = submit_job(test_name, script_path, usd_path, args.partition)
        if job_id:
            jobs[test_name] = job_id

    if args.dry_run:
        print("\nDry run complete. No jobs submitted.")
        return

    if not jobs:
        print("\nERROR: No jobs were submitted successfully.")
        sys.exit(1)

    # --- Wait for completion ---
    job_exit_info = wait_for_jobs(jobs, timeout=args.timeout)

    # --- Collect results ---
    print("\nCollecting results...")
    test_results = {}
    for test_name in TEST_SCRIPTS:
        if test_name in skip_list:
            continue
        result = load_result(test_name)
        if result:
            test_results[test_name] = result
            print(f"  [{test_name}] Loaded result: "
                  f"{'PASS' if result.get('overall_passed') else 'FAIL'}")
        else:
            print(f"  [{test_name}] No result file found")

    # --- Generate report ---
    print("\nGenerating report...")
    json_path, txt_path, all_passed = generate_report(
        usd_path, test_results, job_exit_info, skip_list
    )

    # Print text report to stdout
    print()
    with open(txt_path) as f:
        print(f.read())

    print(f"Reports saved:")
    print(f"  JSON: {json_path}")
    print(f"  Text: {txt_path}")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()

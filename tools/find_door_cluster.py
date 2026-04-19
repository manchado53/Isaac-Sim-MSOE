#!/usr/bin/env python3
"""
Diagnostic: find the densest clusters of physics-controlled doors in a USD.

No simulation, no video — just traverses the stage once, builds a list of
door centers, and reports the top-5 densest clusters (door with the most
neighbors within `radius`). For each cluster it suggests a camera (position
+ target) biased along the cluster's longer horizontal axis (corridor view
if the cluster is elongated, 3/4 corner shot otherwise).

Writes results to logs/<stem>_door_clusters.json and prints a readable
summary to stdout so the caller can pick a cluster and feed its camera
into a showcase script via --cam-pos / --cam-target.

Usage:
    sbatch submit_isaac_job.sh tools/find_door_cluster.py \
        --input assets/environments/msoe_building/MSOEBuildingV4_physics.usd
"""

import argparse
import json
import os
import sys

parser = argparse.ArgumentParser(description="Find densest door clusters in a USD.")
parser.add_argument("--input", required=True, help="Path to physics-ready USD file")
parser.add_argument("--radius", type=float, default=8.0,
                    help="Neighborhood radius in meters (default 8)")
parser.add_argument("--top", type=int, default=5,
                    help="How many top clusters to report (default 5)")
parser.add_argument("--report", default=None,
                    help="JSON output path (default logs/<stem>_door_clusters.json)")
args, _unknown = parser.parse_known_args()

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

import numpy as np
import omni.usd
from pxr import Sdf, Usd, UsdGeom, UsdPhysics
from isaacsim.core.api import World

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.abspath(os.path.join(_script_dir, '..'))


def main():
    input_usd = os.path.abspath(args.input)
    stem = os.path.splitext(os.path.basename(input_usd))[0]
    report_path = args.report or os.path.join(
        _project_dir, "logs", f"{stem}_door_clusters.json"
    )
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    print("=" * 70, flush=True)
    print(f"Door cluster finder — {stem}", flush=True)
    print(f"Radius: {args.radius} m  |  Top: {args.top}", flush=True)
    print("=" * 70, flush=True)

    world = World(stage_units_in_meters=1.0)
    stage = omni.usd.get_context().get_stage()

    building_prim = stage.DefinePrim("/World/Building", "Xform")
    building_prim.GetReferences().AddReference(input_usd, Sdf.Path("/World/Building"))
    print("Building loaded.", flush=True)

    # Collect every physics-controlled door + its world-space center.
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])

    door_paths = []
    door_centers = []
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        # Only doors with a revolute joint child
        joint = None
        for child in prim.GetChildren():
            if ("Hinge" in child.GetName() or "Joint" in child.GetName()) \
                    and child.IsA(UsdPhysics.RevoluteJoint):
                joint = child
                break
        if joint is None:
            continue
        rng = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
        if rng.IsEmpty():
            continue
        mn, mx = rng.GetMin(), rng.GetMax()
        center = [(mn[i] + mx[i]) / 2.0 for i in range(3)]
        door_paths.append(str(prim.GetPath()))
        door_centers.append(center)

    if not door_centers:
        print("ERROR: no physics-controlled doors found.", flush=True)
        simulation_app.close()
        return

    centers = np.array(door_centers)
    print(f"Found {len(centers)} physics-controlled doors.", flush=True)
    print(f"  X: {centers[:,0].min():.1f} .. {centers[:,0].max():.1f}", flush=True)
    print(f"  Y: {centers[:,1].min():.1f} .. {centers[:,1].max():.1f}", flush=True)
    print(f"  Z: {centers[:,2].min():.1f} .. {centers[:,2].max():.1f}", flush=True)

    # For each door, count neighbors within radius (excluding self).
    counts = np.zeros(len(centers), dtype=int)
    for i, c in enumerate(centers):
        dists = np.linalg.norm(centers - c, axis=1)
        counts[i] = int(np.sum(dists <= args.radius) - 1)

    order = np.argsort(-counts)  # descending

    # For the top-N, pick the seed's cluster and compute a suggested camera.
    # Greedy: skip later seeds whose cluster shares most members with an
    # already-chosen seed, so the top-5 list isn't 5 overlapping views of
    # the same corridor.
    used = np.zeros(len(centers), dtype=bool)
    clusters = []
    for idx in order:
        if used[idx]:
            continue
        seed = centers[idx]
        dists = np.linalg.norm(centers - seed, axis=1)
        mask = dists <= args.radius
        if mask.sum() < 3:
            break  # clusters are too small to bother reporting
        members_idx = np.where(mask)[0]
        overlap = np.sum(used[members_idx]) / float(len(members_idx))
        if overlap > 0.5:
            continue  # mostly seen
        used[members_idx] = True

        cluster_centers = centers[mask]
        centroid = cluster_centers.mean(axis=0)
        mn = cluster_centers.min(axis=0)
        mx = cluster_centers.max(axis=0)
        extent = mx - mn
        ex, ey, ez = extent

        # Camera heuristic:
        # - If cluster is elongated along X (ex > ey*1.2) → corridor along X;
        #   stand at -X end with target at +X end.
        # - Same for Y.
        # - Otherwise, 3/4 corner shot from (-X, -Y).
        # Z always: centroid + 1.5 m (eye level above door centers).
        if ex > ey * 1.2:
            cam_pos = [float(mn[0] - 3.0), float(centroid[1]), float(centroid[2] + 1.5)]
            target  = [float(mx[0] + 0.5), float(centroid[1]), float(centroid[2])]
            shape = "corridor_x"
        elif ey > ex * 1.2:
            cam_pos = [float(centroid[0]), float(mn[1] - 3.0), float(centroid[2] + 1.5)]
            target  = [float(centroid[0]), float(mx[1] + 0.5), float(centroid[2])]
            shape = "corridor_y"
        else:
            cam_pos = [float(mn[0] - 3.0), float(mn[1] - 3.0), float(centroid[2] + 2.5)]
            target  = [float(centroid[0]), float(centroid[1]), float(centroid[2])]
            shape = "corner_3_4"

        clusters.append({
            "rank": len(clusters) + 1,
            "seed_path": door_paths[idx],
            "seed_neighbor_count": int(counts[idx]),
            "members": int(mask.sum()),
            "centroid": [round(float(v), 3) for v in centroid],
            "extent": [round(float(v), 3) for v in extent],
            "shape": shape,
            "cam_pos": [round(v, 3) for v in cam_pos],
            "cam_target": [round(v, 3) for v in target],
            "member_paths": [door_paths[i] for i in members_idx[:20]],
        })
        if len(clusters) >= args.top:
            break

    print("", flush=True)
    print(f"=== Top {len(clusters)} clusters (radius={args.radius} m) ===", flush=True)
    for c in clusters:
        print(f"\n[#{c['rank']}] {c['members']} doors  shape={c['shape']}", flush=True)
        print(f"    centroid: ({c['centroid'][0]:.1f}, {c['centroid'][1]:.1f}, {c['centroid'][2]:.1f})", flush=True)
        print(f"    extent:   ({c['extent'][0]:.1f} x {c['extent'][1]:.1f} x {c['extent'][2]:.1f})", flush=True)
        cp = c["cam_pos"]; ct = c["cam_target"]
        print(f"    cam-pos:    {cp[0]:.2f},{cp[1]:.2f},{cp[2]:.2f}", flush=True)
        print(f"    cam-target: {ct[0]:.2f},{ct[1]:.2f},{ct[2]:.2f}", flush=True)

    # Dump JSON
    with open(report_path, "w") as f:
        json.dump({
            "input": input_usd,
            "radius": args.radius,
            "door_count": len(centers),
            "clusters": clusters,
        }, f, indent=2)
    print(f"\nWrote {report_path}", flush=True)

    simulation_app.close()


if __name__ == "__main__":
    main()

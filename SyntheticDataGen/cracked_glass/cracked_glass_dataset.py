#!/usr/bin/env python3
"""
Cracked-Glass Replicator Dataset Generator.

Renders synthetic frames of the Innovation Center where a random subset of
glass panels has been swapped to a "cracked glass" OmniPBR material. Each
frame emits RGB + semantic segmentation + 2D bounding boxes via Replicator's
BasicWriter, tagging cracked panels with the ``cracked_glass`` class and
intact panels with ``intact_glass`` — ready for supervised anomaly training.

Per-batch randomization: we rebind materials and flip the Semantics label
directly in USD before capturing each batch of frames. Replicator only
drives the render/writer; labels are guaranteed to match visuals.

Usage:

    ./python.sh SyntheticDataGen/cracked_glass/cracked_glass_dataset.py \\
        --frames 500 --fraction 0.2 --cameras G,1,2,3

    # smoke test:
    ./python.sh SyntheticDataGen/cracked_glass/cracked_glass_dataset.py --frames 10
"""

import argparse
import datetime
import json
import os
import random
import sys

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.abspath(os.path.join(_script_dir, "..", ".."))
sys.path.insert(0, _project_dir)
sys.path.insert(0, _script_dir)

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

import numpy as np
import omni.replicator.core as rep
import omni.usd
from pxr import Sdf, Usd, UsdGeom, UsdShade

from isaacsim.core.api import World
from sim_utils.building_physics import setup_physics_scene
from glass_anomaly import (
    bind_visual_material,
    create_cracked_glass_materials,
    enumerate_glass_prims,
)
from sim_utils.test_cameras import (
    CAMERA_FOCAL_LENGTH,
    CAMERA_RESOLUTION,
    TEST_CAMERAS,
    add_interior_lighting,
)


DEFAULT_USD = os.path.join(
    _project_dir,
    "assets",
    "environments",
    "msoe_building",
    "MSOEBuildingV4_physics.usd",
)
FALLBACK_USD = os.path.join(
    _project_dir,
    "assets",
    "environments",
    "msoe_building",
    "innovation_center_physics.usd",
)
TEXTURE_DIR = os.path.join(_project_dir, "assets", "materials", "cracked_glass")
DATASET_ROOT = os.path.join(_project_dir, "datasets", "cracked_glass")


def parse_args():
    p = argparse.ArgumentParser(description="Cracked-glass Replicator dataset generator")
    p.add_argument("--usd", default=None)
    p.add_argument("--frames", type=int, default=100, help="Frames per camera")
    p.add_argument("--fraction", type=float, default=0.2,
                   help="Fraction of glass prims cracked per batch (0..1)")
    p.add_argument("--batch-size", type=int, default=10,
                   help="Frames between re-randomizations")
    p.add_argument("--cameras", default="G,1,2,3",
                   help="TEST_CAMERAS keys (ignored when --close-up > 0)")
    p.add_argument("--close-up", type=int, default=0,
                   help="If >0, render this many close-up cameras "
                        "aimed at individual glass panels instead of "
                        "TEST_CAMERAS.")
    p.add_argument("--close-up-distance", type=float, default=1.5,
                   help="Meters from panel center to camera")
    p.add_argument("--close-up-only-cracked", action="store_true",
                   help="Only pick cameras targeting cracked panels "
                        "(guarantees cracks in frame)")
    p.add_argument("--output", default=None)
    p.add_argument("--variants", default="v1,v2,v3")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def resolve_usd(requested):
    if requested and os.path.isfile(requested):
        return requested
    if os.path.isfile(DEFAULT_USD):
        return DEFAULT_USD
    return FALLBACK_USD


def create_intact_glass_material(stage, material_path="/World/Looks/CleanGlass"):
    """Author a baseline clean-glass OmniGlass material for intact panels.

    Real refracting glass (not OmniPBR alpha-blend) so the cracked
    variants and intact baseline share the same physically-based glass
    appearance — differing only in normal-map detail + frosting.
    """
    UsdGeom.Scope.Define(stage, Sdf.Path(material_path).GetParentPath())
    material = UsdShade.Material.Define(stage, material_path)
    shader = UsdShade.Shader.Define(stage, f"{material_path}/Shader")
    shader.CreateIdAttr("mdlMaterial")
    shader.SetSourceAsset("OmniGlass.mdl", "mdl")
    shader.SetSourceAssetSubIdentifier("OmniGlass", "mdl")
    shader.CreateInput("glass_color", Sdf.ValueTypeNames.Color3f).Set((0.90, 0.95, 0.97))
    shader.CreateInput("glass_ior", Sdf.ValueTypeNames.Float).Set(1.491)
    shader.CreateInput("frosting_roughness", Sdf.ValueTypeNames.Float).Set(0.02)
    shader.CreateInput("depth", Sdf.ValueTypeNames.Float).Set(0.003)
    shader.CreateInput("thin_walled", Sdf.ValueTypeNames.Bool).Set(True)
    material.CreateSurfaceOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
    material.CreateDisplacementOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
    material.CreateVolumeOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
    return material_path


def randomize_glass(stage, glass_prims, cracked_mats, intact_mat, fraction):
    """Rebind a random fraction of glass prims to cracked materials.

    Returns (cracked_count, cracked_prim_paths) — cracked_prim_paths is the
    authoritative ground-truth list for the sidecar metadata JSON.
    """
    n_total = len(glass_prims)
    n_cracked = max(0, min(n_total, int(round(n_total * fraction))))
    indices = set(random.sample(range(n_total), n_cracked)) if n_cracked else set()

    cracked_paths = []
    for i, prim in enumerate(glass_prims):
        if i in indices and cracked_mats:
            mat_path = random.choice(cracked_mats)
            bind_visual_material(stage, prim, mat_path)
            cracked_paths.append(str(prim.GetPath()))
        else:
            bind_visual_material(stage, prim, intact_mat)
    return cracked_paths


def panel_close_up_pose(prim, bbox_cache, distance=1.5):
    """Return (camera_pos, look_at) for a camera 'distance' meters in front of the panel.

    Treats the thinnest bbox axis as the panel's surface normal. Offsets the
    camera along that axis, pulled slightly toward the building exterior so
    we're looking inward at the glass (keeps the camera outdoors for exterior
    panels, which usually renders cleaner than being embedded in a wall).
    """
    mn, mx = compute_world_bbox(prim, bbox_cache)
    if mn is None:
        return None, None

    cx = (mn[0] + mx[0]) / 2.0
    cy = (mn[1] + mx[1]) / 2.0
    cz = (mn[2] + mx[2]) / 2.0

    dx = mx[0] - mn[0]
    dy = mx[1] - mn[1]
    dz = mx[2] - mn[2]

    # Pick normal axis = thinnest dimension
    axes = [("x", dx), ("y", dy), ("z", dz)]
    axes.sort(key=lambda t: t[1])
    normal_axis = axes[0][0]

    # Normal vector in world space. For the Revit building, exterior curtain
    # walls are mostly aligned to the X or Y axes; picking a sign pointing
    # "outward" is building-specific. Default to +direction; if the resulting
    # camera ends up inside building bbox center, flip.
    if normal_axis == "x":
        n = (1.0, 0.0, 0.0)
    elif normal_axis == "y":
        n = (0.0, 1.0, 0.0)
    else:
        n = (0.0, 0.0, 1.0)

    cam_pos = (cx + n[0] * distance, cy + n[1] * distance, cz + n[2] * distance)
    look_at = (cx, cy, cz)
    return cam_pos, look_at


def compute_world_bbox(prim, bbox_cache):
    """Return (min_xyz, max_xyz) world-space bbox as tuples of floats."""
    try:
        bb = bbox_cache.ComputeWorldBound(prim)
        aligned = bb.ComputeAlignedRange()
        mn = aligned.GetMin()
        mx = aligned.GetMax()
        return [float(mn[0]), float(mn[1]), float(mn[2])], \
               [float(mx[0]), float(mx[1]), float(mx[2])]
    except Exception:
        return None, None


def write_frame_metadata(output_dir, cam_key, frame_idx, cracked_paths,
                         camera_cfg, cracked_bboxes):
    """Write a per-frame metadata JSON next to the RGB."""
    meta = {
        "frame_idx": frame_idx,
        "camera": cam_key,
        "camera_pos": list(camera_cfg["pos"]),
        "camera_target": list(camera_cfg["target"]),
        "room": camera_cfg.get("room", ""),
        "cracked_count": len(cracked_paths),
        "cracked_prims": cracked_paths,
        "cracked_bboxes_world": cracked_bboxes,
    }
    path = os.path.join(output_dir, cam_key, f"metadata_{frame_idx:04d}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    usd_path = resolve_usd(args.usd)
    if not os.path.isfile(usd_path):
        print(f"ERROR: USD not found: {usd_path}", flush=True)
        sys.exit(1)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output or os.path.join(DATASET_ROOT, timestamp)
    os.makedirs(output_dir, exist_ok=True)

    print(f"[cracked-glass] USD        : {usd_path}", flush=True)
    print(f"[cracked-glass] Output     : {output_dir}", flush=True)
    print(f"[cracked-glass] Frames/cam : {args.frames}", flush=True)
    print(f"[cracked-glass] Fraction   : {args.fraction:.2f}", flush=True)
    print(f"[cracked-glass] Batch size : {args.batch_size}", flush=True)
    print(f"[cracked-glass] Cameras    : {args.cameras}", flush=True)

    # Stage units are resolved from the referenced USD's metersPerUnit.
    # V4_physics.usd = 1.0 (meters); other exports may differ. Let Isaac pick.
    world = World()
    stage = omni.usd.get_context().get_stage()

    # Load building
    building_prim = stage.DefinePrim("/World/Building", "Xform")
    building_prim.GetReferences().AddReference(
        usd_path, primPath=Sdf.Path("/World/Building")
    )
    setup_physics_scene(stage)
    add_interior_lighting(stage)

    # Author materials
    cracked_materials = create_cracked_glass_materials(
        stage,
        texture_dir=TEXTURE_DIR,
        root_path="/World/Looks/CrackedGlass",
        variants=tuple(v.strip() for v in args.variants.split(",") if v.strip()),
    )
    intact_material = create_intact_glass_material(stage)
    print(f"[cracked-glass] Cracked variants: {len(cracked_materials)}", flush=True)

    # Enumerate glass
    glass_prims = enumerate_glass_prims(stage)
    print(f"[cracked-glass] Glass prims found: {len(glass_prims)}", flush=True)
    if not glass_prims:
        print("[cracked-glass] ERROR: no glass prims — aborting", flush=True)
        sys.exit(2)

    world.reset()

    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
    )

    # Build camera list. Two modes:
    #   - TEST_CAMERAS (default): security-cam views across the room
    #   - close-up: N cameras each framed on one glass panel
    camera_entries = []   # list of (cam_key, cam_cfg)

    if args.close_up > 0:
        if args.close_up_only_cracked:
            # Pre-commit a crack set so we know which panels will be cracked
            # for the whole run; every close-up camera targets one of these.
            n_cracked = max(1, int(round(len(glass_prims) * args.fraction)))
            committed = set(random.sample(range(len(glass_prims)), n_cracked))
            pool = [glass_prims[i] for i in committed]
            print(f"[cracked-glass] committed-crack pool: {len(pool)} panels",
                  flush=True)
        else:
            pool = glass_prims

        random.shuffle(pool)
        for i, panel in enumerate(pool[:args.close_up]):
            pos, look = panel_close_up_pose(panel, bbox_cache,
                                            distance=args.close_up_distance)
            if pos is None:
                continue
            cam_key = f"cu_{i:03d}"
            camera_entries.append((cam_key, {
                "pos": tuple(float(v) for v in pos),
                "target": tuple(float(v) for v in look),
                "room": f"close-up {panel.GetName()}",
                "panel_prim": str(panel.GetPath()),
            }))
        print(f"[cracked-glass] close-up cameras: {len(camera_entries)}",
              flush=True)
    else:
        for cam_key in [k.strip() for k in args.cameras.split(",") if k.strip()]:
            if cam_key not in TEST_CAMERAS:
                print(f"[cracked-glass] skip unknown camera '{cam_key}'", flush=True)
                continue
            camera_entries.append((cam_key, TEST_CAMERAS[cam_key]))

    total_frames_written = 0

    for cam_key, cam_cfg in camera_entries:
        print(f"[cracked-glass] === camera {cam_key} ({cam_cfg.get('room','')}) ===",
              flush=True)

        cam_node = rep.create.camera(
            position=cam_cfg["pos"],
            look_at=cam_cfg["target"],
            focal_length=CAMERA_FOCAL_LENGTH,
            horizontal_aperture=20.955,
            name=f"CrackedGlassCam_{cam_key}",
        )
        render_product = rep.create.render_product(cam_node, CAMERA_RESOLUTION)

        writer = rep.WriterRegistry.get("BasicWriter")
        writer.initialize(
            output_dir=os.path.join(output_dir, cam_key),
            rgb=True,
        )
        writer.attach([render_product])

        frame_counter = 0
        frames_remaining = args.frames
        batch_idx = 0
        while frames_remaining > 0:
            batch = min(args.batch_size, frames_remaining)
            cracked_paths = randomize_glass(
                stage, glass_prims, cracked_materials, intact_material, args.fraction
            )
            # Force a specific panel to be cracked if this is a targeted close-up
            target_prim_path = cam_cfg.get("panel_prim")
            if target_prim_path and args.close_up_only_cracked:
                if target_prim_path not in cracked_paths:
                    # Swap one random cracked for the target
                    if cracked_paths:
                        swap_idx = random.randrange(len(cracked_paths))
                        old = cracked_paths[swap_idx]
                        cracked_paths[swap_idx] = target_prim_path
                        # Rebind materials to match
                        target = stage.GetPrimAtPath(target_prim_path)
                        old_prim = stage.GetPrimAtPath(old)
                        if target and target.IsValid():
                            bind_visual_material(
                                stage, target, random.choice(cracked_materials),
                            )
                        if old_prim and old_prim.IsValid():
                            bind_visual_material(stage, old_prim, intact_material)

            cracked_bboxes = []
            for p in cracked_paths:
                prim = stage.GetPrimAtPath(p)
                if prim and prim.IsValid():
                    mn, mx = compute_world_bbox(prim, bbox_cache)
                    if mn is not None:
                        cracked_bboxes.append({"prim": p, "min": mn, "max": mx})
            print(f"  batch {batch_idx}: {len(cracked_paths)}/{len(glass_prims)} cracked, "
                  f"{batch} frames, bboxes: {len(cracked_bboxes)}", flush=True)

            rep.orchestrator.step(rt_subframes=3)
            for _ in range(batch):
                rep.orchestrator.step(rt_subframes=2)
                write_frame_metadata(
                    output_dir, cam_key, frame_counter,
                    cracked_paths, cam_cfg, cracked_bboxes,
                )
                frame_counter += 1
            frames_remaining -= batch
            batch_idx += 1
            total_frames_written += batch

        writer.detach()
        render_product.destroy()

    print(f"[cracked-glass] Done. Total frames: {total_frames_written}", flush=True)
    print(f"[cracked-glass] Dataset at: {output_dir}", flush=True)
    simulation_app.close()


if __name__ == "__main__":
    main()

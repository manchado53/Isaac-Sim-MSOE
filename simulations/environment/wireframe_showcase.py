#!/usr/bin/env python3
"""
Wireframe Showcase — Orbital tour of building collision mesh structure.

Category-colored wireframe boxes on a dark background, camera orbiting
the full building. Dimmer emissive materials for a subtle, technical look.

Output:
  - videos/environment/wireframe_showcase.mp4  (~15s @ 30fps, 1920x1080)

Usage:
  sbatch scripts/slurm/submit_isaac_job.sh simulations/environment/wireframe_showcase.py
"""

import os
import sys
import math
import time

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

import numpy as np
import omni.usd
import isaacsim.core.utils.numpy.rotations as rot_utils
from pxr import Usd, UsdGeom, UsdShade, UsdLux, UsdPhysics, Sdf, Gf, Vt
from isaacsim.core.api import World
from isaacsim.sensors.camera import Camera

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.abspath(os.path.join(_script_dir, '..', '..'))
sys.path.insert(0, _project_dir)

from sim_utils.video_capture import VideoRecorder
from sim_utils.building_physics import (
    setup_physics_scene, iter_building_elements, PrimCategory, categorize_prim,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def look_at_orientation(cam_pos, target):
    dx = target[0] - cam_pos[0]
    dy = target[1] - cam_pos[1]
    dz = target[2] - cam_pos[2]
    yaw = math.degrees(math.atan2(dy, dx))
    pitch = math.degrees(math.atan2(-dz, math.sqrt(dx**2 + dy**2)))
    return rot_utils.euler_angles_to_quats(
        np.array([0, pitch, yaw]), degrees=True
    )


def create_emissive_material(stage, path, color, intensity=0.4):
    """Create a dim emissive material for wireframe rendering."""
    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path + "/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    bright = Gf.Vec3f(
        min(color[0] * intensity, 1.0),
        min(color[1] * intensity, 1.0),
        min(color[2] * intensity, 1.0),
    )
    shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(bright)
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(1.0)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return mat


def draw_wireframe_box(stage, path, bbox_min, bbox_max, material, thickness=0.15):
    """Draw 12 edges of a bounding box as BasisCurves."""
    x0, y0, z0 = float(bbox_min[0]), float(bbox_min[1]), float(bbox_min[2])
    x1, y1, z1 = float(bbox_max[0]), float(bbox_max[1]), float(bbox_max[2])

    edges = [
        [(x0,y0,z0),(x1,y0,z0)], [(x1,y0,z0),(x1,y1,z0)],
        [(x1,y1,z0),(x0,y1,z0)], [(x0,y1,z0),(x0,y0,z0)],
        [(x0,y0,z1),(x1,y0,z1)], [(x1,y0,z1),(x1,y1,z1)],
        [(x1,y1,z1),(x0,y1,z1)], [(x0,y1,z1),(x0,y0,z1)],
        [(x0,y0,z0),(x0,y0,z1)], [(x1,y0,z0),(x1,y0,z1)],
        [(x1,y1,z0),(x1,y1,z1)], [(x0,y1,z0),(x0,y1,z1)],
    ]

    points = []
    counts = []
    for edge in edges:
        points.extend([Gf.Vec3f(*p) for p in edge])
        counts.append(2)

    curves = UsdGeom.BasisCurves.Define(stage, path)
    curves.CreatePointsAttr(Vt.Vec3fArray(points))
    curves.CreateCurveVertexCountsAttr(Vt.IntArray(counts))
    curves.CreateTypeAttr("linear")
    curves.CreateWidthsAttr(Vt.FloatArray([thickness] * len(points)))
    UsdShade.MaterialBindingAPI.Apply(curves.GetPrim()).Bind(material)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_start = time.time()

    world = World(stage_units_in_meters=1.0)
    stage = omni.usd.get_context().get_stage()

    # --- Load physics USD ---
    physics_usd = os.path.join(
        _project_dir, "assets", "environments", "msoe_building",
        "innovation_center_physics.usd"
    )
    if not os.path.exists(physics_usd):
        print(f"ERROR: {physics_usd} not found. Run setup_innovation_center_physics.py first.")
        simulation_app.close()
        sys.exit(1)

    print(f"Loading: {physics_usd}", flush=True)
    building_prim = stage.DefinePrim("/World/Building", "Xform")
    building_prim.GetReferences().AddReference(
        physics_usd, primPath=Sdf.Path("/World/Building")
    )
    print("Building loaded.", flush=True)

    setup_physics_scene(stage)

    # --- Building bounds ---
    building_prim = stage.GetPrimAtPath("/World/Building")
    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]
    )
    bbox = bbox_cache.ComputeWorldBound(building_prim)
    rng = bbox.ComputeAlignedRange()
    bb_min = rng.GetMin()
    bb_max = rng.GetMax()
    bldg_center = np.array([(bb_min[i] + bb_max[i]) / 2.0 for i in range(3)])
    bldg_extent = np.array([bb_max[i] - bb_min[i] for i in range(3)])

    print(f"Building bbox: ({bb_min[0]:.1f}, {bb_min[1]:.1f}, {bb_min[2]:.1f}) "
          f"to ({bb_max[0]:.1f}, {bb_max[1]:.1f}, {bb_max[2]:.1f})", flush=True)

    # --- Hide solid building geometry ---
    UsdGeom.Imageable(building_prim).MakeInvisible()

    # --- Dark environment: very dim ambient light only ---
    dome = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/DomeLight"))
    dome.CreateIntensityAttr(0)  # pitch black background

    # --- Wireframe materials (dimmer than static_element_showcase) ---
    stage.DefinePrim("/World/Wire", "Xform")
    wire_mats = {
        PrimCategory.STATIC_STRUCTURAL: create_emissive_material(
            stage, "/World/Wire/StructuralMat", (0.3, 0.6, 1.0), 0.4),   # dim blue
        PrimCategory.STATIC_GLAZING: create_emissive_material(
            stage, "/World/Wire/GlazingMat", (0.1, 0.9, 0.9), 0.4),     # dim cyan
        PrimCategory.DYNAMIC_DOOR: create_emissive_material(
            stage, "/World/Wire/DoorMat", (1.0, 0.4, 0.0), 0.4),        # dim orange
        PrimCategory.STATIC_STAIR: create_emissive_material(
            stage, "/World/Wire/StairMat", (0.3, 1.0, 0.2), 0.4),       # dim green
    }
    default_mat = wire_mats[PrimCategory.STATIC_STRUCTURAL]

    # --- Draw wireframes ---
    print("Drawing wireframe collision boxes...", flush=True)
    drawn = 0
    for level_name, category_name, element in iter_building_elements(stage, building_prim):
        element_name = element.GetName()
        category = categorize_prim(element_name, category_hint=category_name)

        if category == PrimCategory.SKIP:
            continue

        try:
            elem_bbox = bbox_cache.ComputeWorldBound(element)
            elem_rng = elem_bbox.ComputeAlignedRange()
            if elem_rng.IsEmpty():
                continue
            elem_min = elem_rng.GetMin()
            elem_max = elem_rng.GetMax()
            ext = elem_max - elem_min
            if max(float(ext[0]), float(ext[1]), float(ext[2])) < 0.5:
                continue
        except Exception:
            continue

        mat = wire_mats.get(category, default_mat)
        thickness = 0.25 if category == PrimCategory.DYNAMIC_DOOR else 0.15
        draw_wireframe_box(
            stage, f"/World/Wire/E_{drawn:05d}",
            elem_min, elem_max, mat, thickness=thickness,
        )
        drawn += 1

    print(f"Drew {drawn} wireframe boxes", flush=True)

    # --- Camera setup ---
    horiz_diag = math.sqrt(float(bldg_extent[0])**2 + float(bldg_extent[1])**2)
    orbit_radius = horiz_diag * 1.4
    cam_target = np.array([
        float(bldg_center[0]),
        float(bldg_center[1]),
        float(bldg_center[2]) - float(bldg_extent[2]) * 0.1,
    ])

    # Initial position
    start_angle = math.radians(225)
    cam_pos = np.array([
        float(bldg_center[0]) + orbit_radius * math.cos(start_angle),
        float(bldg_center[1]) + orbit_radius * math.sin(start_angle),
        float(bldg_center[2]) + float(bldg_extent[2]) * 0.45,
    ])
    cam_orient = look_at_orientation(cam_pos, cam_target)

    camera = Camera(
        prim_path="/World/ShowcaseCamera",
        position=cam_pos,
        frequency=30,
        resolution=(1920, 1080),
        orientation=cam_orient,
    )

    # --- Initialize and warm up ---
    world.reset()
    camera.initialize()

    print("Warming up renderer...", flush=True)
    for _ in range(30):
        world.step(render=True)

    # --- Record orbital wireframe video (15s = 450 frames, full orbit) ---
    total_frames = 450
    print(f"\n=== Wireframe Orbit ({total_frames} frames, {total_frames/30:.0f}s) ===", flush=True)

    with VideoRecorder("environment/wireframe_showcase", fps=30) as recorder:
        for i in range(total_frames):
            # Full 360° orbit
            angle = start_angle + (i / total_frames) * 2 * math.pi
            # Gentle vertical oscillation
            height_osc = float(bldg_extent[2]) * 0.1 * math.sin(2 * angle)
            cp = np.array([
                float(bldg_center[0]) + orbit_radius * math.cos(angle),
                float(bldg_center[1]) + orbit_radius * math.sin(angle),
                float(bldg_center[2]) + float(bldg_extent[2]) * 0.45 + height_osc,
            ])
            orient = look_at_orientation(cp, cam_target)
            camera.set_world_pose(position=cp, orientation=orient)

            world.step(render=True)
            rgba = camera.get_rgba()
            if rgba is not None and i > 0:
                recorder.capture_rgba(rgba)

            if i % 90 == 0:
                print(f"  Frame {i}/{total_frames}", flush=True)

    t_end = time.time()
    print(f"\nDone! Runtime: {t_end - t_start:.1f}s", flush=True)
    print(f"Video -> videos/environment/wireframe_showcase.mp4", flush=True)
    simulation_app.close()


if __name__ == "__main__":
    main()

"""
capture_water_leaks.py

Opens MSOEBuildingV4_physics.usd in Isaac Sim, procedurally adds water-leak
geometry directly into the live stage (nothing written to a new USD), places
interior cameras, renders with RTX, and saves:

  captures/images/   - one PNG per camera shot
  captures/video.mp4 - all frames stitched into a video

Run with Isaac Sim's Python:
    ~/isaacsim/python.sh capture_water_leaks.py
    ~/isaacsim/python.sh capture_water_leaks.py --count 5 --shots 6

Options:
    --count       number of water leak instances to place  (default: 5)
    --shots       camera angles per anomaly                (default: 6)
    --res         render resolution, e.g. 1920 1080       (default: 1280 720)
    --subframes   RTX subframes (higher = less noise)      (default: 32)
    --seed        random seed                              (default: 42)
    --no-video    skip MP4 encoding
"""

import argparse
import math
import os
import random
from pathlib import Path

# ── SimulationApp must be created first ────────────────────────────────────
from isaacsim import SimulationApp

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--count",      type=int,   default=5)
parser.add_argument("--shots",      type=int,   default=6)
parser.add_argument("--res",        type=int,   nargs=2, default=[1280, 720])
parser.add_argument("--subframes",  type=int,   default=32)
parser.add_argument("--seed",       type=int,   default=42)
parser.add_argument("--no-video",   action="store_true")
parser.add_argument("--out-dir",
                    default=str(Path(__file__).parent / "captures"))
args, _ = parser.parse_known_args()

random.seed(args.seed)

simulation_app = SimulationApp(launch_config={
    "renderer": "RaytracedLighting",
    "headless": True,
    "width":    args.res[0],
    "height":   args.res[1],
})

# ── Late imports ────────────────────────────────────────────────────────────
import carb
import carb.settings
import numpy as np
import omni.replicator.core as rep
import omni.usd
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade

HERE      = Path(__file__).parent
BUILD_USD = str(HERE / "MSOEBuildingV4_physics.usd")
OUT_DIR   = Path(args.out_dir)
IMG_DIR   = OUT_DIR / "images"
IMG_DIR.mkdir(parents=True, exist_ok=True)

BUILDING_BASE = (
    "/World/Building"
    "/_22_129___MSOE_Engineering_Building___A___2_25_2026"
    "/Geometry"
)

# ── Quality & exposure settings ─────────────────────────────────────────────
carb.settings.get_settings().set("rtx/post/dlss/execMode", 2)
# +2 EV brings a dim interior up to a well-lit level without blowout
carb.settings.get_settings().set("rtx/post/tonemap/exposureValue", 2.0)
carb.settings.get_settings().set("rtx/post/tonemap/filmIso",       800.0)
carb.settings.get_settings().set("rtx/post/tonemap/op",              1)    # Reinhard

# ===========================================================================
# Stage helpers
# ===========================================================================

def _mat(stage, path, diffuse, opacity=1.0, roughness=0.5, metallic=0.0):
    m  = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, path + "/Shader")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*diffuse))
    sh.CreateInput("opacity",      Sdf.ValueTypeNames.Float).Set(opacity)
    sh.CreateInput("roughness",    Sdf.ValueTypeNames.Float).Set(roughness)
    sh.CreateInput("metallic",     Sdf.ValueTypeNames.Float).Set(metallic)
    m.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return m

def _bind(prim, mat):
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat)

def _translate(prim_or_api, vec):
    UsdGeom.XformCommonAPI(prim_or_api).SetTranslate(Gf.Vec3d(*vec))

def _scale(prim_or_api, vec):
    UsdGeom.XformCommonAPI(prim_or_api).SetScale(Gf.Vec3f(*vec))

def _rotate(prim_or_api, vec):
    UsdGeom.XformCommonAPI(prim_or_api).SetRotate(Gf.Vec3f(*vec))


# ===========================================================================
# Add lights to the live stage
# ===========================================================================

def add_lights(stage):
    if stage.GetPrimAtPath("/World/Lights"):
        return
    UsdGeom.Xform.Define(stage, "/World/Lights")

    # Dome + sun — kept low; interior lights do the real work
    dome = UsdLux.DomeLight.Define(stage, "/World/Lights/SkyDome")
    dome.CreateIntensityAttr(1000.0)
    dome.CreateColorAttr(Gf.Vec3f(0.82, 0.88, 1.00))
    UsdGeom.XformCommonAPI(dome).SetRotate(Gf.Vec3f(270, 0, 0))

    sun = UsdLux.DistantLight.Define(stage, "/World/Lights/Sun")
    sun.CreateIntensityAttr(3000.0)
    sun.CreateAngleAttr(1.0)
    sun.CreateColorAttr(Gf.Vec3f(1.00, 0.95, 0.87))
    UsdGeom.XformCommonAPI(sun).SetRotate(Gf.Vec3f(-45, 30, 0))

    # Interior sphere lights — 5000 intensity, radius=1.0
    # Building spans X[-71,29], Y[-9,89], Z[8,36]
    interior_positions = [
        # Floor 1 (~Z=9.5)
        (-20,  20, 9.5), (  0,  20, 9.5), ( 20,  20, 9.5),
        (-20,  50, 9.5), (  0,  50, 9.5), ( 20,  50, 9.5),
        (-20,  70, 9.5), (  0,  70, 9.5),
        # Floor 2 (~Z=13.5)
        (-20,  20, 13.5), (  0,  20, 13.5), ( 20,  20, 13.5),
        (-20,  50, 13.5), (  0,  50, 13.5), ( 20,  50, 13.5),
        # Floor 3 (~Z=17.5)
        (-20,  20, 17.5), (  0,  20, 17.5), ( 20,  20, 17.5),
        (-20,  50, 17.5), (  0,  50, 17.5),
        # Floor 4+ (~Z=21.5)
        (-20,  20, 21.5), (  0,  20, 21.5), ( 20,  20, 21.5),
    ]
    for idx, (lx, ly, lz) in enumerate(interior_positions):
        light = UsdLux.SphereLight.Define(stage, f"/World/Lights/Interior_{idx:03d}")
        light.CreateIntensityAttr(30000.0)
        light.CreateRadiusAttr(1.0)
        light.CreateColorAttr(Gf.Vec3f(1.0, 0.97, 0.92))
        UsdGeom.XformCommonAPI(light).SetTranslate(Gf.Vec3d(lx, ly, lz))

    print(f"  Lights added ({len(interior_positions)} interior sphere lights).")


# ===========================================================================
# Collect candidate positions from the loaded stage
# ===========================================================================

def get_floor_positions(stage, limit=93):
    root = stage.GetPrimAtPath(f"{BUILDING_BASE}/Floors")
    if not root:
        return []
    out = []
    for p in Usd.PrimRange(root):
        if p.GetTypeName() == "Mesh":
            mat = np.array(
                UsdGeom.Xformable(p).ComputeLocalToWorldTransform(
                    Usd.TimeCode.Default())
            ).reshape(4, 4)
            out.append(mat[3, :3])
            if len(out) >= limit:
                break
    return out


def place_water_leak(stage, root_path, pos, severity):
    """Add water-leak geometry under *root_path* at world *pos*."""
    root = UsdGeom.Xform.Define(stage, root_path)
    _translate(root, pos)
    looks = root_path + "/Looks"
    sev   = severity

    # Puddle
    puddle_r = 0.10 + sev * 0.45
    p = UsdGeom.Cylinder.Define(stage, root_path + "/Puddle")
    p.CreateRadiusAttr(puddle_r)
    p.CreateHeightAttr(0.007)
    p.CreateAxisAttr(UsdGeom.Tokens.z)
    _translate(p, (0, 0, 0.004))
    UsdPhysics.CollisionAPI.Apply(p.GetPrim())
    _bind(p.GetPrim(), _mat(stage, looks+"/PuddleMat",
                            (0.18, 0.28, 0.52), opacity=0.55+sev*0.15,
                            roughness=0.04, metallic=0.08))

    # Stain ring
    stain_r = puddle_r * (1.4 + sev * 0.6)
    s = UsdGeom.Cylinder.Define(stage, root_path + "/Stain")
    s.CreateRadiusAttr(stain_r)
    s.CreateHeightAttr(0.003)
    s.CreateAxisAttr(UsdGeom.Tokens.z)
    _translate(s, (0, 0, 0.001))
    _bind(s.GetPrim(), _mat(stage, looks+"/StainMat",
                            (0.14, 0.14, 0.18), opacity=0.25+sev*0.20,
                            roughness=0.92))

    # Drip spheres
    n_drips   = max(2, int(sev * 9))
    drip_h    = 1.8 + sev * 1.5
    for d in range(n_drips):
        dz  = drip_h - d * (drip_h / max(n_drips, 1)) * random.uniform(0.7, 1.0)
        dx  = random.uniform(-0.04, 0.04)
        dy  = random.uniform(-0.04, 0.04)
        dr  = random.uniform(0.010, 0.022)
        drip = UsdGeom.Sphere.Define(stage, root_path + f"/Drip_{d:02d}")
        drip.CreateRadiusAttr(dr)
        _translate(drip, (dx, dy, dz))
        rb = UsdPhysics.RigidBodyAPI.Apply(drip.GetPrim())
        rb.CreateVelocityAttr(Gf.Vec3f(0, 0, random.uniform(-0.6, -0.2)))
        UsdPhysics.MassAPI.Apply(drip.GetPrim()).CreateMassAttr(0.001)
        UsdPhysics.CollisionAPI.Apply(drip.GetPrim())
        _bind(drip.GetPrim(), _mat(stage, looks+f"/DripMat_{d:02d}",
                                   (0.25, 0.38, 0.68),
                                   opacity=0.55+random.uniform(0, 0.15),
                                   roughness=0.04, metallic=0.05))

    # Wall moisture streak
    streak_w = 0.06 + sev * 0.10
    streak_h = 0.4  + sev * 1.2
    st = UsdGeom.Cube.Define(stage, root_path + "/Streak")
    st.CreateSizeAttr(1.0)
    _scale(st, (streak_w, 0.01, streak_h))
    _translate(st, (0, puddle_r + 0.02, drip_h + streak_h / 2))
    _bind(st.GetPrim(), _mat(stage, looks+"/StreakMat",
                             (0.20, 0.22, 0.30),
                             opacity=0.22+sev*0.18, roughness=0.88))


# ===========================================================================
# Camera rendering
# ===========================================================================

def render_shots(stage, anomaly_pos, anomaly_name, n_shots, resolution,
                 subframes, out_dir):
    """
    Render *n_shots* RTX images of the anomaly from different angles.
    Returns list of saved PNG paths.
    """
    import shutil

    ax, ay, az = anomaly_pos
    floor_z    = az - 0.5
    eye_z      = floor_z + 1.7    # eye level

    # BasicWriter writes to a temp subdir per shot, then we rename the file
    tmp_dir = out_dir / "_tmp_writer"

    saved = []
    for i in range(n_shots):
        t        = i / max(n_shots, 1)
        angle    = 2 * math.pi * t + random.uniform(-0.3, 0.3)
        # Orbit from ABOVE the anomaly looking down — avoids placing
        # cameras inside vertical walls, and gives a natural view of
        # floor puddles / glass shards
        h_offset = random.uniform(2.0, 4.0)   # height above anomaly
        r_offset = random.uniform(1.0, 3.0)   # horizontal stand-off
        cx = ax + r_offset * math.cos(angle)
        cy = ay + r_offset * math.sin(angle)
        cz = az + h_offset

        # look_at floor level so the top-down camera actually sees the anomaly
        look_target = (ax, ay, az - 0.2)
        distance    = math.dist((cx, cy, cz), look_target)

        cam = rep.create.camera(
            position=(cx, cy, cz),
            look_at=look_target,
            focal_length=random.uniform(18, 28),
            focus_distance=distance,
            f_stop=11.0,   # high f-stop = deep focus, no blur
            clipping_range=(0.05, 10000.0),
        )

        rp = rep.create.render_product(cam, resolution)

        # Fill light at camera position for consistent per-shot illumination
        fill_path = f"/World/Lights/FillLight_shot{i:03d}"
        fill = UsdLux.SphereLight.Define(stage, fill_path)
        fill.CreateIntensityAttr(30000.0)
        fill.CreateRadiusAttr(1.0)
        fill.CreateColorAttr(Gf.Vec3f(1.0, 0.97, 0.92))
        UsdGeom.XformCommonAPI(fill).SetTranslate(Gf.Vec3d(cx, cy, cz))

        # Let renderer pick up the new light before capturing
        for _ in range(10):
            simulation_app.update()

        # Use a fresh temp dir each shot so frame index is always 0
        shot_tmp = tmp_dir / f"shot_{i:03d}"
        shot_tmp.mkdir(parents=True, exist_ok=True)

        writer = rep.WriterRegistry.get("BasicWriter")
        writer.initialize(
            output_dir=str(shot_tmp),
            rgb=True,
            bounding_box_2d_tight=False,
            semantic_segmentation=False,
        )
        writer.attach(rp)
        rep.orchestrator.step(rt_subframes=subframes)
        writer.detach()
        rp.destroy()

        # BasicWriter saves to shot_tmp/rgb/0000000000.png — find and rename it
        rgb_files = sorted((shot_tmp / "rgb").glob("*.png")) if (shot_tmp / "rgb").exists() else []
        if not rgb_files:
            # Fallback: search anywhere under shot_tmp
            rgb_files = sorted(shot_tmp.rglob("*.png"))

        final_path = out_dir / f"{anomaly_name}_shot{i:03d}.png"
        if rgb_files:
            shutil.move(str(rgb_files[-1]), str(final_path))
            saved.append(str(final_path))
            print(f"    Shot {i+1}/{n_shots} → {final_path.name}")
        else:
            print(f"    Shot {i+1}/{n_shots} WARNING: no PNG found in {shot_tmp}")

        # Clean up temp dir for this shot
        shutil.rmtree(str(shot_tmp), ignore_errors=True)

    # Remove temp parent dir if empty
    try:
        tmp_dir.rmdir()
    except OSError:
        pass

    return saved


# ===========================================================================
# Video encoding (imageio, no extra dependency beyond what we already have)
# ===========================================================================

def encode_video(image_paths, output_path, fps=8):
    try:
        import imageio
        import numpy as np
        from PIL import Image

        frames = []
        for p in image_paths:
            candidate = Path(p)
            if not candidate.exists():
                continue
            img = np.array(Image.open(candidate).convert("RGB"))
            frames.append(img)

        if not frames:
            print("  No frames found for video encoding.")
            return

        with imageio.get_writer(str(output_path), fps=fps,
                                codec="libx264", quality=8,
                                pixelformat="yuv420p",
                                macro_block_size=1) as writer:
            for frame in frames:
                writer.append_data(frame)
        print(f"  Video → {output_path}  ({len(frames)} frames @ {fps}fps)")
    except Exception as e:
        print(f"  Video encoding skipped: {e}")


# ===========================================================================
# Main
# ===========================================================================

def main():
    print(f"\nOpening stage: {BUILD_USD}")
    omni.usd.get_context().open_stage(BUILD_USD)

    # Let Isaac Sim fully load the stage (large USD needs more ticks)
    print("  Waiting for stage to load...")
    for _ in range(50):
        simulation_app.update()

    stage = omni.usd.get_context().get_stage()
    if not stage:
        print("ERROR: Stage failed to load.")
        simulation_app.close()
        return

    add_lights(stage)

    # Give the renderer time to incorporate the new lights
    print("  Warming up renderer with lights...")
    for _ in range(30):
        simulation_app.update()

    resolution = tuple(args.res)
    all_images = []

    # ── Water leaks ──────────────────────────────────────────────────────────
    print(f"\n=== Placing {args.count} water leak(s) ===")
    floor_positions = get_floor_positions(stage)
    if not floor_positions:
        print("  WARNING: No floor positions found.")
    else:
        selected = random.sample(
            floor_positions, min(args.count, len(floor_positions))
        )
        for i, pos in enumerate(selected):
            severity  = random.uniform(0.3, 1.0)
            root_path = f"/World/Anomalies/WaterLeak_{i:03d}"
            place_water_leak(stage, root_path, pos, severity)
            print(f"  WaterLeak_{i:03d}  sev={severity:.2f}"
                  f"  @ ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})")

            # Update stage so geometry is visible before rendering
            for _ in range(5):
                simulation_app.update()

            print(f"  Rendering {args.shots} shot(s)…")
            imgs = render_shots(stage, pos,
                                f"WaterLeak_{i:03d}",
                                args.shots, resolution,
                                args.subframes, IMG_DIR)
            all_images.extend(imgs)

    # ── Video ─────────────────────────────────────────────────────────────────
    if not args.no_video:
        print("\nEncoding video…")
        encode_video(all_images, OUT_DIR / "anomaly_capture.mp4", fps=6)

    total = args.count * args.shots
    print(f"\nDone.  ~{total} images → {IMG_DIR}/")
    simulation_app.close()


main()

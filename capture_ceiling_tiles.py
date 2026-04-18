"""
capture_ceiling_tiles.py

Opens MSOEBuildingV4_physics.usd in Isaac Sim, procedurally adds
fallen-ceiling-tile geometry into the live stage, renders with RTX,
and saves:

  captures/images/   - one PNG per camera shot
  captures/video.mp4 - all frames stitched into a video

Run with Isaac Sim's Python:
    ~/isaacsim/python.sh capture_ceiling_tiles.py
    ~/isaacsim/python.sh capture_ceiling_tiles.py --count 5 --shots 6

Options:
    --count       number of ceiling-tile incidents to place  (default: 5)
    --shots       camera angles per anomaly                  (default: 6)
    --res         render resolution, e.g. 1920 1080          (default: 1280 720)
    --subframes   RTX subframes (higher = less noise)        (default: 32)
    --seed        random seed                                (default: 42)
    --ceiling-z   ceiling height above floor in meters       (default: 3.0)
    --no-video    skip MP4 encoding
"""

import argparse
import math
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
parser.add_argument("--ceiling-z",  type=float, default=4.0)
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

TILE_W = 0.610
TILE_H = 0.005

# ── Renderer quality settings ───────────────────────────────────────────────
carb.settings.get_settings().set("rtx/post/dlss/execMode", 2)
carb.settings.get_settings().set("rtx/post/tonemap/exposureValue", 1.5)
carb.settings.get_settings().set("rtx/post/tonemap/filmIso",       400.0)
carb.settings.get_settings().set("rtx/post/tonemap/op",              1)

# ===========================================================================
# Stage helpers (mirrors capture_water_leaks.py)
# ===========================================================================

def _mat(stage, path, diffuse, opacity=1.0, roughness=0.8, metallic=0.0):
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

def _translate(geom, vec):
    UsdGeom.XformCommonAPI(geom).SetTranslate(Gf.Vec3d(*vec))

def _scale(geom, vec):
    UsdGeom.XformCommonAPI(geom).SetScale(Gf.Vec3f(*vec))

def _rotate(geom, vec):
    UsdGeom.XformCommonAPI(geom).SetRotate(Gf.Vec3f(*vec))


# ===========================================================================
# Lighting
# ===========================================================================

def add_lights(stage):
    if stage.GetPrimAtPath("/World/Lights"):
        return
    UsdGeom.Xform.Define(stage, "/World/Lights")

    dome = UsdLux.DomeLight.Define(stage, "/World/Lights/SkyDome")
    dome.CreateIntensityAttr(500.0)
    dome.CreateColorAttr(Gf.Vec3f(0.82, 0.88, 1.00))
    UsdGeom.XformCommonAPI(dome).SetRotate(Gf.Vec3f(270, 0, 0))

    sun = UsdLux.DistantLight.Define(stage, "/World/Lights/Sun")
    sun.CreateIntensityAttr(1500.0)
    sun.CreateAngleAttr(1.0)
    sun.CreateColorAttr(Gf.Vec3f(1.00, 0.95, 0.87))
    UsdGeom.XformCommonAPI(sun).SetRotate(Gf.Vec3f(-45, 30, 0))

    interior_positions = [
        (-20, 20, 9.5), (0, 20, 9.5), (20, 20, 9.5),
        (-20, 50, 9.5), (0, 50, 9.5), (20, 50, 9.5),
        (-20, 70, 9.5), (0, 70, 9.5),
        (-20, 20, 13.5), (0, 20, 13.5), (20, 20, 13.5),
        (-20, 50, 13.5), (0, 50, 13.5), (20, 50, 13.5),
        (-20, 20, 17.5), (0, 20, 17.5), (20, 20, 17.5),
        (-20, 50, 17.5), (0, 50, 17.5),
        (-20, 20, 21.5), (0, 20, 21.5), (20, 20, 21.5),
    ]
    for i, (lx, ly, lz) in enumerate(interior_positions):
        light = UsdLux.SphereLight.Define(stage, f"/World/Lights/Interior_{i:03d}")
        light.CreateIntensityAttr(15000.0)
        light.CreateRadiusAttr(1.0)
        light.CreateColorAttr(Gf.Vec3f(1.0, 0.97, 0.92))
        UsdGeom.XformCommonAPI(light).SetTranslate(Gf.Vec3d(lx, ly, lz))

    print(f"  Lights added ({len(interior_positions)} interior sphere lights).")


# ===========================================================================
# Floor position collection
# ===========================================================================

def get_floor_positions(stage, limit=93):
    """
    Return one position per distinct floor location.

    Two filters applied based on the building's actual geometry:
      1. Z must sit near a known main floor level (8.6, 13.4, 17.9, 22.3, 26.4 m).
         Intermediate structural elements (mezzanines, slab mid-points) are
         excluded — a camera 2 m above them would land inside the next slab.
      2. Positions within 1 m of an already-accepted position are deduplicated.
         Many mesh entries are just different material layers of the same slab.

    The floor-to-floor spacing is ~4.3–4.8 m so a tolerance of ±0.8 m around
    each nominal level catches all real floor surfaces.
    """
    # Main floor levels confirmed from discover_floor_positions.py output
    FLOOR_LEVELS = [8.6, 13.4, 17.9, 22.3, 26.4]
    Z_TOL        = 0.8   # ± m around each level
    XY_DEDUP     = 1.0   # ignore positions within 1 m xy of an accepted one

    def near_floor(z):
        return any(abs(z - lvl) < Z_TOL for lvl in FLOOR_LEVELS)

    root = stage.GetPrimAtPath(f"{BUILDING_BASE}/Floors")
    if not root:
        return []

    accepted = []
    for p in Usd.PrimRange(root):
        if p.GetTypeName() != "Mesh":
            continue
        mat = np.array(
            UsdGeom.Xformable(p).ComputeLocalToWorldTransform(
                Usd.TimeCode.Default())
        ).reshape(4, 4)
        pos = mat[3, :3]
        x, y, z = float(pos[0]), float(pos[1]), float(pos[2])

        if not near_floor(z):
            continue

        # Deduplicate: skip if within 1 m xy of an already-accepted position
        duplicate = any(
            abs(x - ax) < XY_DEDUP and abs(y - ay) < XY_DEDUP
            for ax, ay, _ in accepted
        )
        if duplicate:
            continue

        accepted.append((x, y, z))
        if len(accepted) >= limit:
            break

    print(f"  Floor positions: {len(accepted)} usable (filtered from raw meshes)")
    return accepted


# ===========================================================================
# Fallen ceiling tile geometry
# ===========================================================================

def place_ceiling_tiles(stage, root_path, pos, severity, ceiling_z=3.0):
    """Add fallen-ceiling-tile geometry under *root_path* at world *pos*."""
    root = UsdGeom.Xform.Define(stage, root_path)
    _translate(root, pos)
    looks = root_path + "/Looks"

    n_tiles      = max(2, int(2 + severity * 3))
    n_debris     = max(4, int(4 + severity * 16))
    max_tilt     = 15 + severity * 45
    tile_spread  = 0.3 + severity * 0.9

    tile_mat  = _mat(stage, looks + "/TileMat",  (0.95, 0.95, 0.93), roughness=0.92)
    crack_mat = _mat(stage, looks + "/CrackMat", (0.80, 0.78, 0.75), roughness=0.95)
    debris_mat = _mat(stage, looks + "/DebrisMat", (0.90, 0.88, 0.85), roughness=0.97)
    dust_mat   = _mat(stage, looks + "/DustMat",   (0.85, 0.83, 0.80),
                      opacity=0.60 + severity * 0.25, roughness=0.99)

    # Fallen tiles
    for t in range(n_tiles):
        tile_path = root_path + f"/Tile_{t:02d}"
        tile      = UsdGeom.Mesh.Define(stage, tile_path)

        tw = TILE_W * random.uniform(0.5 if severity > 0.5 else 0.8, 1.0)
        td = TILE_W * random.uniform(0.5 if severity > 0.5 else 0.8, 1.0)
        hw, hd, hh = tw / 2, td / 2, TILE_H / 2

        pts = [
            Gf.Vec3f(-hw, -hd, -hh), Gf.Vec3f( hw, -hd, -hh),
            Gf.Vec3f( hw,  hd, -hh), Gf.Vec3f(-hw,  hd, -hh),
            Gf.Vec3f(-hw, -hd,  hh), Gf.Vec3f( hw, -hd,  hh),
            Gf.Vec3f( hw,  hd,  hh), Gf.Vec3f(-hw,  hd,  hh),
        ]
        tile.CreatePointsAttr(pts)
        tile.CreateFaceVertexCountsAttr([4, 4, 4, 4, 4, 4])
        tile.CreateFaceVertexIndicesAttr([
            0, 1, 2, 3, 4, 5, 6, 7,
            0, 1, 5, 4, 2, 3, 7, 6,
            1, 2, 6, 5, 0, 3, 7, 4,
        ])
        tile.CreateNormalsAttr([Gf.Vec3f(0, 0, -1)] * 4 + [Gf.Vec3f(0, 0, 1)] * 4)

        ox = random.uniform(-tile_spread, tile_spread)
        oy = random.uniform(-tile_spread, tile_spread)
        oz = hh + 0.001
        api = UsdGeom.XformCommonAPI(tile.GetPrim())
        api.SetTranslate(Gf.Vec3d(ox, oy, oz))
        api.SetRotate(Gf.Vec3f(
            random.uniform(-max_tilt, max_tilt),
            random.uniform(-max_tilt, max_tilt),
            random.uniform(0, 360),
        ))
        UsdPhysics.RigidBodyAPI.Apply(tile.GetPrim()).CreateKinematicEnabledAttr(True)
        UsdPhysics.CollisionAPI.Apply(tile.GetPrim())
        _bind(tile.GetPrim(), crack_mat if (severity > 0.6 and t < 2) else tile_mat)

    # Ceiling gap
    gap_w = TILE_W * (1 + severity * 0.5)
    gap_d = TILE_W * (0.8 + severity * 0.4)
    gap   = UsdGeom.Cube.Define(stage, root_path + "/CeilingGap")
    gap.CreateSizeAttr(1.0)
    _scale(gap, (gap_w, gap_d, 0.02))
    _translate(gap, (0, 0, ceiling_z - 0.01))
    _bind(gap.GetPrim(), _mat(stage, looks + "/GapMat", (0.05, 0.05, 0.06), roughness=1.0))

    # T-bar grid bars
    for bi, (bw, bd) in enumerate([(gap_w, 0.025), (0.025, gap_d)]):
        bar = UsdGeom.Cube.Define(stage, root_path + f"/GridBar_{bi}")
        bar.CreateSizeAttr(1.0)
        _scale(bar, (bw, bd, 0.03))
        _translate(bar, (0, 0, ceiling_z - 0.035))
        _bind(bar.GetPrim(), _mat(stage, looks + f"/BarMat_{bi}",
                                  (0.55, 0.55, 0.55), roughness=0.4, metallic=0.6))

    # Debris
    debris_spread = tile_spread + 0.25 + severity * 0.3
    for d in range(n_debris):
        dx  = random.uniform(-debris_spread, debris_spread)
        dy  = random.uniform(-debris_spread, debris_spread)
        db  = root_path + f"/Debris_{d:02d}"
        if random.random() < 0.5:
            chunk = UsdGeom.Cube.Define(stage, db)
            chunk.CreateSizeAttr(1.0)
            _scale(chunk, (random.uniform(0.03, 0.18),
                           random.uniform(0.03, 0.15),
                           random.uniform(0.003, 0.012)))
            _translate(chunk, (dx, dy, random.uniform(0.002, 0.015)))
            _rotate(chunk, (random.uniform(-20, 20), random.uniform(-20, 20),
                            random.uniform(0, 360)))
            _bind(chunk.GetPrim(), debris_mat)
        else:
            blob = UsdGeom.Sphere.Define(stage, db)
            blob.CreateRadiusAttr(random.uniform(0.02, 0.08))
            _scale(blob, (random.uniform(0.8, 1.2), random.uniform(0.8, 1.2), 0.15))
            _translate(blob, (dx, dy, 0.003))
            _bind(blob.GetPrim(), dust_mat)

    # Dust ring
    dust_r = tile_spread + 0.4 + severity * 0.3
    ring   = UsdGeom.Cylinder.Define(stage, root_path + "/DustRing")
    ring.CreateRadiusAttr(dust_r)
    ring.CreateHeightAttr(0.002)
    ring.CreateAxisAttr(UsdGeom.Tokens.z)
    _translate(ring, (0, 0, 0.001))
    _bind(ring.GetPrim(), _mat(stage, looks + "/DustRingMat",
                               (0.65, 0.63, 0.60),
                               opacity=0.10 + severity * 0.10, roughness=0.98))


# ===========================================================================
# Camera rendering (identical pattern to capture_water_leaks.py)
# ===========================================================================

def render_shots(stage, anomaly_pos, anomaly_name, n_shots, resolution,
                 subframes, out_dir, ceiling_z):
    import shutil

    ax, ay, az = anomaly_pos
    eye_z      = az + 1.6
    stand_off  = 3.0

    # Mix of overhead and oblique angles.
    # Overhead shots are safe (no horizontal offset = cannot clip a wall).
    # Oblique shots use a small 2 m offset — enough for a clear angled view
    # without reaching most interior walls.
    tile_target = (ax, ay, az + 0.3)   # aim at tile height, not the bare floor
    shot_configs = [
        # Overhead — shows scatter pattern from above
        ((ax, ay, az + 2.5), tile_target),
        ((ax, ay, az + 3.0), tile_target),
        # Oblique 45° — shows tiles with depth and perspective
        ((ax + 2.0, ay,       az + 2.0), tile_target),
        ((ax - 2.0, ay,       az + 2.0), tile_target),
        ((ax,       ay - 2.0, az + 2.0), tile_target),
        ((ax,       ay + 2.0, az + 2.0), tile_target),
    ]
    random.shuffle(shot_configs)
    shot_configs = shot_configs[:n_shots]

    tmp_dir = out_dir / "_tmp_writer"
    # Remove any leftover tmp dirs from interrupted previous runs
    if tmp_dir.exists():
        import shutil as _sh
        _sh.rmtree(str(tmp_dir), ignore_errors=True)
    saved   = []

    for i, (pos, look_target) in enumerate(shot_configs):
        cx, cy, cz = pos
        distance   = math.dist((cx, cy, cz), look_target)

        cam = rep.create.camera(
            position=(cx, cy, cz),
            look_at=look_target,
            focal_length=24,
            focus_distance=distance,
            f_stop=0.0,    # 0 = infinite depth of field, no blur
            clipping_range=(0.05, 10000.0),
        )
        rp = rep.create.render_product(cam, resolution)

        # Place fill light 2 m ABOVE the camera so it is out of frame
        fill_path = f"/World/Lights/FillLight_shot{i:03d}"
        fill = UsdLux.SphereLight.Define(stage, fill_path)
        fill.CreateIntensityAttr(10000.0)
        fill.CreateRadiusAttr(0.5)
        fill.CreateColorAttr(Gf.Vec3f(1.0, 0.97, 0.92))
        UsdGeom.XformCommonAPI(fill).SetTranslate(Gf.Vec3d(cx, cy, cz + 2.0))

        for _ in range(10):
            simulation_app.update()

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
        # Flush async file writes — orchestrator.step() returns before the
        # BasicWriter finishes writing to disk; update() ticks let it complete.
        for _ in range(20):
            simulation_app.update()
        writer.detach()
        rp.destroy()

        # BasicWriter may write to rgb/ subdir OR as rgb_XXXX.png at root
        if (shot_tmp / "rgb").exists():
            rgb_files = sorted((shot_tmp / "rgb").glob("*.png"))
        else:
            rgb_files = sorted(shot_tmp.glob("rgb_*.png"))
        if not rgb_files:
            rgb_files = sorted(shot_tmp.rglob("*.png"))

        final_path = out_dir / f"{anomaly_name}_shot{i:03d}.png"
        if rgb_files:
            shutil.move(str(rgb_files[-1]), str(final_path))
            saved.append(str(final_path))
            print(f"    Shot {i+1}/{n_shots} → {final_path.name}")
        else:
            print(f"    Shot {i+1}/{n_shots} WARNING: no PNG found in {shot_tmp}")

        shutil.rmtree(str(shot_tmp), ignore_errors=True)

    try:
        tmp_dir.rmdir()
    except OSError:
        pass

    return saved


# ===========================================================================
# Video encoding
# ===========================================================================

def encode_video(image_paths, output_path, fps=8):
    try:
        import imageio
        from PIL import Image

        frames = [np.array(Image.open(p).convert("RGB"))
                  for p in image_paths if Path(p).exists()]
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

    print("  Waiting for stage to load...")
    for _ in range(50):
        simulation_app.update()

    stage = omni.usd.get_context().get_stage()
    if not stage:
        print("ERROR: Stage failed to load.")
        simulation_app.close()
        return

    add_lights(stage)
    print("  Warming up renderer with lights...")
    for _ in range(30):
        simulation_app.update()

    resolution = tuple(args.res)
    all_images = []

    print(f"\n=== Placing {args.count} ceiling-tile incident(s) ===")
    floor_positions = get_floor_positions(stage)
    if not floor_positions:
        print("  WARNING: No floor positions found.")
    else:
        # Floor mesh origins are at the BOTTOM of the concrete slab (~0.4 m below
        # the walking surface).  Shift anomalies up so tiles rest on the floor.
        SLAB_THICKNESS = 0.4

        selected = random.sample(floor_positions, min(args.count, len(floor_positions)))
        for i, pos in enumerate(selected):
            severity  = random.uniform(0.3, 1.0)
            root_path = f"/World/Anomalies/CeilingTileIncident_{i:03d}"
            surface_pos = (pos[0], pos[1], pos[2] + SLAB_THICKNESS)
            place_ceiling_tiles(stage, root_path, surface_pos, severity, args.ceiling_z)
            print(f"  CeilingTileIncident_{i:03d}  sev={severity:.2f}"
                  f"  @ ({surface_pos[0]:.1f}, {surface_pos[1]:.1f}, {surface_pos[2]:.1f})")

            for _ in range(5):
                simulation_app.update()

            print(f"  Rendering {args.shots} shot(s)…")
            imgs = render_shots(stage, surface_pos,
                                f"CeilingTile_{i:03d}",
                                args.shots, resolution,
                                args.subframes, IMG_DIR,
                                args.ceiling_z)
            all_images.extend(imgs)

    if not args.no_video:
        print("\nEncoding video…")
        encode_video(all_images, OUT_DIR / "ceiling_tile_capture.mp4", fps=6)

    total = args.count * args.shots
    print(f"\nDone.  ~{total} images → {IMG_DIR}/")
    simulation_app.close()


main()

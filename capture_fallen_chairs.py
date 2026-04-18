"""
capture_fallen_chairs.py

Opens MSOEBuildingV4_physics.usd in Isaac Sim, procedurally adds fallen
chair anomaly geometry into the live stage, renders with RTX, and saves:

  captures/fallen_chairs/images/   - one PNG per camera shot

Each anomaly: 1-3 chairs tipped over on their sides, as if blocking a
hallway.  Cameras are placed at eye-level looking toward the chairs
(hallway perspective) as well as overhead shots for variety.

Chairs are built from USD primitives (seat, backrest, 4 legs) and rendered
in dark charcoal against the lighter building floor for maximum contrast.

Run with:
    ~/isaacsim/python.sh capture_fallen_chairs.py
    ~/isaacsim/python.sh capture_fallen_chairs.py --count 3 --shots 4

Options:
    --count     number of fallen-chair incidents   (default: 5)
    --shots     camera angles per anomaly          (default: 4)
    --res       render resolution, e.g. 1920 1080  (default: 1280 720)
    --subframes RTX subframes (higher = less noise) (default: 32)
    --seed      random seed                        (default: 42)
    --no-video  skip MP4 encoding
    --out-dir   output directory
"""

import argparse
import math
import random
from pathlib import Path

# ── SimulationApp must be created first ────────────────────────────────────
from isaacsim import SimulationApp

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--count",     type=int,   default=5)
parser.add_argument("--shots",     type=int,   default=4)
parser.add_argument("--res",       type=int,   nargs=2, default=[1280, 720])
parser.add_argument("--subframes", type=int,   default=32)
parser.add_argument("--seed",      type=int,   default=42)
parser.add_argument("--no-video",  action="store_true")
parser.add_argument("--out-dir",
                    default=str(Path(__file__).parent / "captures" / "fallen_chairs"))
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

# ── Renderer quality settings ───────────────────────────────────────────────
carb.settings.get_settings().set("rtx/post/dlss/execMode",          2)
carb.settings.get_settings().set("rtx/post/tonemap/exposureValue", 1.5)
carb.settings.get_settings().set("rtx/post/tonemap/filmIso",       400.0)
carb.settings.get_settings().set("rtx/post/tonemap/op",              1)   # Reinhard


# ===========================================================================
# Stage helpers
# ===========================================================================

def _mat(stage, path, diffuse, opacity=1.0, roughness=0.7, metallic=0.0):
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

    # Interior sphere lights spread across all floor levels
    interior_positions = [
        (-20, 20, 9.5),  (0, 20, 9.5),  (20, 20, 9.5),
        (-20, 50, 9.5),  (0, 50, 9.5),  (20, 50, 9.5),
        (-20, 70, 9.5),  (0, 70, 9.5),
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
    Return one position per distinct floor location, filtered to known main
    floor levels and deduplicated to remove material-layer duplicates.
    """
    FLOOR_LEVELS = [8.6, 13.4, 17.9, 22.3, 26.4]
    Z_TOL        = 0.8
    XY_DEDUP     = 1.0

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
# Chair geometry builder — tablet-arm classroom chair
# ===========================================================================
#
# Modelled after the Virco/KI style tablet-arm chair found in most college
# classrooms: plastic shell seat + back, tubular steel legs, single right-side
# writing tablet on a curved arm.
#
# Dimensions (meters, approximate real-world values):
SEAT_W  = 0.42   # seat width
SEAT_D  = 0.40   # seat depth
SEAT_T  = 0.05   # seat shell thickness
SEAT_H  = 0.46   # seat top surface height above floor
BACK_W  = 0.40   # backrest width
BACK_T  = 0.05   # backrest shell thickness
BACK_H  = 0.46   # backrest height
LEG_R   = 0.013  # tubular steel leg radius
LEG_H   = 0.44   # leg height (floor to seat underside)


def _build_chair_upright(stage, root_path, shell_mat, leg_mat, tablet_mat):
    """
    Build a tablet-arm classroom chair at root_path (origin = floor level).

    Components:
      - Plastic shell seat + backrest  (shell_mat)
      - 4 tubular steel legs + cross-braces  (leg_mat)
      - Right-side tablet arm + writing surface  (tablet_mat / leg_mat)
    """
    # ── Seat shell ────────────────────────────────────────────────────────
    seat = UsdGeom.Cube.Define(stage, root_path + "/Seat")
    seat.CreateSizeAttr(1.0)
    _scale(seat, (SEAT_W, SEAT_D, SEAT_T))
    _translate(seat, (0.0, 0.0, SEAT_H))
    _bind(seat.GetPrim(), shell_mat)

    # ── Backrest shell (slight backward tilt baked in via scale + offset) ─
    back = UsdGeom.Cube.Define(stage, root_path + "/Backrest")
    back.CreateSizeAttr(1.0)
    _scale(back, (BACK_W, BACK_T, BACK_H))
    # Sits at the rear edge of the seat, rises from seat top
    _translate(back, (0.0, -(SEAT_D / 2 - BACK_T / 2),
                      SEAT_H + SEAT_T / 2 + BACK_H / 2))
    _bind(back.GetPrim(), shell_mat)

    # ── 4 tubular steel legs (cylinders) ─────────────────────────────────
    # Legs splay outward slightly by offsetting them to the seat corners
    inset = 0.04   # how far inside the seat edge the legs sit
    leg_xy = [
        ( SEAT_W / 2 - inset,  SEAT_D / 2 - inset),   # front-right
        (-SEAT_W / 2 + inset,  SEAT_D / 2 - inset),   # front-left
        ( SEAT_W / 2 - inset, -SEAT_D / 2 + inset),   # back-right
        (-SEAT_W / 2 + inset, -SEAT_D / 2 + inset),   # back-left
    ]
    for li, (lx, ly) in enumerate(leg_xy):
        leg = UsdGeom.Cylinder.Define(stage, root_path + f"/Leg_{li:02d}")
        leg.CreateRadiusAttr(LEG_R)
        leg.CreateHeightAttr(LEG_H)
        leg.CreateAxisAttr(UsdGeom.Tokens.z)
        _translate(leg, (lx, ly, LEG_H / 2))
        _bind(leg.GetPrim(), leg_mat)

    # ── Horizontal cross-braces (front and rear, low on legs) ─────────────
    for bi, (by, bw) in enumerate([(SEAT_D / 2 - inset, SEAT_W - inset * 2),
                                    (-SEAT_D / 2 + inset, SEAT_W - inset * 2)]):
        brace = UsdGeom.Cylinder.Define(stage, root_path + f"/Brace_{bi}")
        brace.CreateRadiusAttr(LEG_R * 0.8)
        brace.CreateHeightAttr(bw)
        brace.CreateAxisAttr(UsdGeom.Tokens.x)
        _translate(brace, (0.0, by, 0.22))
        _bind(brace.GetPrim(), leg_mat)

    # Side brace connecting front-right to back-right legs
    side_brace = UsdGeom.Cylinder.Define(stage, root_path + "/BraceSide")
    side_brace.CreateRadiusAttr(LEG_R * 0.8)
    side_brace.CreateHeightAttr(SEAT_D - inset * 2)
    side_brace.CreateAxisAttr(UsdGeom.Tokens.y)
    _translate(side_brace, (SEAT_W / 2 - inset, 0.0, 0.22))
    _bind(side_brace.GetPrim(), leg_mat)

    # ── Right-side tablet arm ─────────────────────────────────────────────
    # Vertical support post rising from the right-front leg area
    arm_x = SEAT_W / 2 + 0.02   # just outside right edge of seat

    arm_post = UsdGeom.Cylinder.Define(stage, root_path + "/ArmPost")
    arm_post.CreateRadiusAttr(LEG_R)
    arm_post.CreateHeightAttr(0.30)
    arm_post.CreateAxisAttr(UsdGeom.Tokens.z)
    _translate(arm_post, (arm_x, SEAT_D / 2 - inset, SEAT_H - 0.15))
    _bind(arm_post.GetPrim(), leg_mat)

    # Horizontal arm bar extending from the post toward the front-right
    arm_bar = UsdGeom.Cylinder.Define(stage, root_path + "/ArmBar")
    arm_bar.CreateRadiusAttr(LEG_R)
    arm_bar.CreateHeightAttr(0.28)
    arm_bar.CreateAxisAttr(UsdGeom.Tokens.y)
    _translate(arm_bar, (arm_x, 0.04, SEAT_H + 0.18))
    _bind(arm_bar.GetPrim(), leg_mat)

    # Writing tablet surface (flat pad, slightly larger than a notebook)
    tablet = UsdGeom.Cube.Define(stage, root_path + "/Tablet")
    tablet.CreateSizeAttr(1.0)
    _scale(tablet, (0.32, 0.26, 0.025))
    _translate(tablet, (arm_x + 0.09, 0.10, SEAT_H + 0.22))
    _bind(tablet.GetPrim(), tablet_mat)

    # Tablet lip (small raised edge on the front of the writing surface)
    lip = UsdGeom.Cube.Define(stage, root_path + "/TabletLip")
    lip.CreateSizeAttr(1.0)
    _scale(lip, (0.32, 0.015, 0.018))
    _translate(lip, (arm_x + 0.09, 0.10 + 0.13, SEAT_H + 0.235))
    _bind(lip.GetPrim(), leg_mat)


# ===========================================================================
# Fallen chair anomaly placer
# ===========================================================================

# Classroom chair colors — plastic shells typical of college furniture
# (medium blue, grey-green, orange-red, black) paired with silver legs
CHAIR_COLORS = [
    (0.18, 0.32, 0.62),   # medium blue  — most common classroom chair
    (0.30, 0.42, 0.32),   # muted green
    (0.65, 0.28, 0.12),   # burnt orange / terracotta
    (0.15, 0.15, 0.18),   # near-black
]


def place_fallen_chairs(stage, root_path, pos, severity):
    """
    Place 1–3 fallen chairs at world-space *pos* (floor surface level).

    severity (0–1) controls how many chairs and how scattered they are.
    Each chair is tipped ~85–100° on its side, resting on the floor.
    """
    root = UsdGeom.Xform.Define(stage, root_path)
    _translate(root, pos)

    n_chairs    = max(1, int(1 + severity * 2))   # 1–3
    chair_spread = 0.3 + severity * 0.6           # how far apart they land

    # After rotating ~90° around X or Y, the lowest point of a chair
    # (leg tip at local z=0) ends up at roughly z = -LEG_W/2 relative to
    # the rotated origin.  A z_offset of 0.22 keeps everything above floor.
    Z_FLOOR_OFFSET = 0.22

    for c in range(n_chairs):
        chair_path  = root_path + f"/Chair_{c:02d}"
        chair_xform = UsdGeom.Xform.Define(stage, chair_path)

        # Scatter position relative to the incident root
        cx = random.uniform(-chair_spread, chair_spread) if c > 0 else 0.0
        cy = random.uniform(-chair_spread, chair_spread) if c > 0 else 0.0

        # Fall direction: tip forward (around X) or to the side (around Y)
        # Angle slightly past 90° so chairs look naturally tumbled
        fall_axis  = random.choice(["X", "Y"])
        fall_angle = random.uniform(82, 105)
        # Optionally invert to fall the other way
        if random.random() < 0.5:
            fall_angle = -fall_angle

        if fall_axis == "X":
            rot = Gf.Vec3f(fall_angle, 0.0, random.uniform(-20, 20))
        else:
            rot = Gf.Vec3f(0.0, fall_angle, random.uniform(-20, 20))

        UsdGeom.XformCommonAPI(chair_xform).SetTranslate(
            Gf.Vec3d(cx, cy, Z_FLOOR_OFFSET)
        )
        UsdGeom.XformCommonAPI(chair_xform).SetRotate(rot)

        # Shell color cycles through classroom palette; legs are always silver steel
        shell_color = CHAIR_COLORS[c % len(CHAIR_COLORS)]

        shell_mat  = _mat(stage, root_path + f"/Looks/ShellMat_{c}",
                          shell_color, roughness=0.60)
        leg_mat    = _mat(stage, root_path + f"/Looks/LegMat_{c}",
                          (0.72, 0.72, 0.74), roughness=0.30, metallic=0.80)
        tablet_mat = _mat(stage, root_path + f"/Looks/TabletMat_{c}",
                          (0.82, 0.78, 0.68), roughness=0.55)   # off-white laminate

        _build_chair_upright(stage, chair_path, shell_mat, leg_mat, tablet_mat)

    # Metadata
    root.GetPrim().SetCustomDataByKey("anomaly:type",        "fallen_chair")
    root.GetPrim().SetCustomDataByKey("anomaly:severity",    severity)
    root.GetPrim().SetCustomDataByKey("anomaly:chair_count", n_chairs)


# ===========================================================================
# Camera rendering
# ===========================================================================

def render_shots(stage, anomaly_pos, anomaly_name, n_shots, resolution,
                 subframes, out_dir):
    import shutil

    ax, ay, az = anomaly_pos

    # Chair target: aim at the middle of a fallen chair (~0.4 m above floor)
    chair_target = (ax, ay, az + 0.4)

    # ── Shot configurations ────────────────────────────────────────────────
    # Eye-level hallway shots: camera at 1.5 m height, 4 m back in 4 directions.
    # This mimics a person standing in a hallway looking at a fallen chair.
    # Overhead shots give a top-down view of the scatter pattern.
    shot_configs = [
        # Eye-level hallway perspectives (primary for VLM training)
        ((ax + 4.0, ay,       az + 1.5), chair_target),   # looking from +X
        ((ax - 4.0, ay,       az + 1.5), chair_target),   # looking from -X
        ((ax,       ay + 4.0, az + 1.5), chair_target),   # looking from +Y
        ((ax,       ay - 4.0, az + 1.5), chair_target),   # looking from -Y
        # Overhead
        ((ax,       ay,       az + 3.0), chair_target),   # directly above
        # Slight 45° angle overhead for depth
        ((ax + 2.0, ay,       az + 2.5), chair_target),
    ]
    random.shuffle(shot_configs)
    shot_configs = shot_configs[:n_shots]

    tmp_dir = out_dir / "_tmp_writer"
    if tmp_dir.exists():
        import shutil as _sh
        _sh.rmtree(str(tmp_dir), ignore_errors=True)
    saved = []

    for i, (pos, look_target) in enumerate(shot_configs):
        cx, cy, cz = pos
        distance   = math.dist((cx, cy, cz), look_target)

        cam = rep.create.camera(
            position=(cx, cy, cz),
            look_at=look_target,
            focal_length=24,
            focus_distance=distance,
            f_stop=0.0,            # 0 = infinite depth of field, no blur
            clipping_range=(0.05, 10000.0),
        )
        rp = rep.create.render_product(cam, resolution)

        # Fill light 2 m ABOVE the camera so it is never in frame
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

    print(f"\n=== Placing {args.count} fallen-chair incident(s) ===")
    floor_positions = get_floor_positions(stage)
    if not floor_positions:
        print("  WARNING: No floor positions found.")
    else:
        # Floor mesh origins are at the bottom of the concrete slab (~0.4 m below
        # the walking surface).  Shift up so chairs rest on the actual floor.
        SLAB_THICKNESS = 0.4

        selected = random.sample(floor_positions, min(args.count, len(floor_positions)))
        for i, pos in enumerate(selected):
            severity    = random.uniform(0.3, 1.0)
            root_path   = f"/World/Anomalies/FallenChairIncident_{i:03d}"
            surface_pos = (pos[0], pos[1], pos[2] + SLAB_THICKNESS)

            place_fallen_chairs(stage, root_path, surface_pos, severity)
            n_chairs = max(1, int(1 + severity * 2))
            print(f"  FallenChairIncident_{i:03d}  sev={severity:.2f}  "
                  f"chairs={n_chairs}  "
                  f"@ ({surface_pos[0]:.1f}, {surface_pos[1]:.1f}, {surface_pos[2]:.1f})")

            for _ in range(5):
                simulation_app.update()

            print(f"  Rendering {args.shots} shot(s)...")
            imgs = render_shots(
                stage, surface_pos,
                f"FallenChair_{i:03d}",
                args.shots, resolution,
                args.subframes, IMG_DIR,
            )
            all_images.extend(imgs)

    if not args.no_video:
        print("\nEncoding video...")
        encode_video(all_images, OUT_DIR / "fallen_chair_capture.mp4", fps=6)

    total = len(all_images)
    print(f"\nDone.  {total} images → {IMG_DIR}/")
    simulation_app.close()


main()

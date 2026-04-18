"""
simulate_fallen_chairs.py

Spawns tablet-arm classroom chairs with REAL PHYSICS on the 4th floor of
MSOEBuildingV4_physics.usd.  Each chair starts upright with a small random
tilt; gravity topples them naturally.  After the simulation settles the scene
is rendered from multiple angles and saved as training images.

No manual rotation needed — PhysX handles the fall.

Run with:
    ~/isaacsim/python.sh simulate_fallen_chairs.py
    ~/isaacsim/python.sh simulate_fallen_chairs.py --chairs 3 --shots 4

Options:
    --chairs      number of chairs to spawn         (default: 3)
    --shots       camera angles to render           (default: 4)
    --sim-steps   physics steps to run (60 Hz)      (default: 240  = 4 s)
    --res         render resolution                 (default: 1280 720)
    --subframes   RTX subframes per render          (default: 32)
    --seed        random seed                       (default: 42)
    --no-video    skip MP4 encoding
    --out-dir     output directory
"""

import argparse
import math
import random
from pathlib import Path

from isaacsim import SimulationApp

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--chairs",     type=int,   default=3)
parser.add_argument("--shots",      type=int,   default=4)
parser.add_argument("--sim-steps",  type=int,   default=240)
parser.add_argument("--res",        type=int,   nargs=2, default=[1280, 720])
parser.add_argument("--subframes",  type=int,   default=32)
parser.add_argument("--seed",       type=int,   default=42)
parser.add_argument("--no-video",   action="store_true")
parser.add_argument("--out-dir",
                    default=str(Path(__file__).parent / "captures" / "physics_chairs"))
args, _ = parser.parse_known_args()

random.seed(args.seed)

simulation_app = SimulationApp(launch_config={
    "renderer":  "RaytracedLighting",
    "headless":  True,
    "width":     args.res[0],
    "height":    args.res[1],
})

import carb
import carb.settings
import numpy as np
import omni.replicator.core as rep
import omni.usd
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade

try:
    from pxr import PhysxSchema
    HAS_PHYSX_SCHEMA = True
except ImportError:
    HAS_PHYSX_SCHEMA = False

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

# 4th floor surface Z (floor mesh origin + slab thickness offset)
FLOOR4_Z     = 22.3 + 0.4   # 22.7 m
FLOOR4_LEVEL = 22.3          # nominal Z used to filter floor meshes
SLAB_THICKNESS = 0.4
SPAWN_SPREAD = 1.5           # m — chairs spread within this radius of a floor position

carb.settings.get_settings().set("rtx/post/dlss/execMode",          2)
carb.settings.get_settings().set("rtx/post/tonemap/exposureValue", 2.5)
carb.settings.get_settings().set("rtx/post/tonemap/filmIso",       800.0)
carb.settings.get_settings().set("rtx/post/tonemap/op",              1)


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

    # Same interior lights as capture_fallen_chairs.py — all floors
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

    # Extra lights directly above the 4th floor area (Y ≈ 53–80)
    for i, (lx, ly) in enumerate([(-20, 55), (0, 55), (20, 55),
                                   (-20, 70), (0, 70), (20, 70),
                                   (-50, 55), (-50, 70)]):
        light = UsdLux.SphereLight.Define(stage, f"/World/Lights/Floor4_{i:03d}")
        light.CreateIntensityAttr(15000.0)
        light.CreateRadiusAttr(1.0)
        light.CreateColorAttr(Gf.Vec3f(1.0, 0.97, 0.92))
        UsdGeom.XformCommonAPI(light).SetTranslate(Gf.Vec3d(lx, ly, FLOOR4_Z + 2.5))

    print(f"  Lights added ({len(interior_positions) + 8} interior sphere lights).")


# ===========================================================================
# Physics scene setup
# ===========================================================================

def get_floor4_positions(stage):
    """Return deduplicated floor mesh XY positions on the 4th floor."""
    Z_TOL   = 0.8
    XY_DEDUP = 1.0
    root = stage.GetPrimAtPath(f"{BUILDING_BASE}/Floors")
    if not root:
        return []
    accepted = []
    for p in Usd.PrimRange(root):
        if p.GetTypeName() != "Mesh":
            continue
        mat = np.array(
            UsdGeom.Xformable(p).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        ).reshape(4, 4)
        pos = mat[3, :3]
        x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
        if abs(z - FLOOR4_LEVEL) >= Z_TOL:
            continue
        if any(abs(x - ax) < XY_DEDUP and abs(y - ay) < XY_DEDUP
               for ax, ay, _ in accepted):
            continue
        accepted.append((x, y, z))
    print(f"  Floor 4 positions found: {len(accepted)}")
    for p in accepted:
        print(f"    ({p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f})")
    return accepted


def setup_physics(stage):
    """Add a PhysicsScene with gravity and a floor collider at the 4th floor."""
    # Physics scene (skip if already present from the building USD)
    if not stage.GetPrimAtPath("/World/PhysicsScene"):
        scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
        scene.CreateGravityDirectionAttr(Gf.Vec3f(0.0, 0.0, -1.0))
        scene.CreateGravityMagnitudeAttr(9.81)
        print("  Physics scene created.")
    else:
        print("  Physics scene already present in stage.")

    # Invisible floor collider — thin box at 4th floor surface level.
    # Even if the building mesh already has collision, this guarantees chairs
    # don't fall through.
    floor_path = "/World/FloorCollider4"
    if not stage.GetPrimAtPath(floor_path):
        floor = UsdGeom.Cube.Define(stage, floor_path)
        floor.CreateSizeAttr(1.0)
        _scale(floor, (200.0, 200.0, 0.05))
        _translate(floor, (0.0, 0.0, FLOOR4_Z - 0.025))
        UsdPhysics.CollisionAPI.Apply(floor.GetPrim())
        # Make it invisible (no material = default grey, but we can hide it)
        UsdGeom.Imageable(floor.GetPrim()).MakeInvisible()
        print(f"  Floor collider added at Z={FLOOR4_Z:.2f} m.")


# ===========================================================================
# Chair geometry — tablet-arm classroom chair
# ===========================================================================

SEAT_W = 0.42
SEAT_D = 0.40
SEAT_T = 0.05
SEAT_H = 0.46
BACK_W = 0.40
BACK_T = 0.05
BACK_H = 0.46
LEG_R  = 0.013
LEG_H  = 0.44

CHAIR_COLORS = [
    (0.18, 0.32, 0.62),   # medium blue
    (0.30, 0.42, 0.32),   # muted green
    (0.65, 0.28, 0.12),   # burnt orange
    (0.15, 0.15, 0.18),   # near-black
]


def _build_chair_geometry(stage, root_path, shell_mat, leg_mat, tablet_mat):
    """Build all chair geometry under root_path (origin = floor level)."""
    geo_prims = []   # collect all geometry prims for collision assignment

    def _add(prim):
        geo_prims.append(prim)
        return prim

    # Seat shell
    seat = _add(UsdGeom.Cube.Define(stage, root_path + "/Seat"))
    seat.CreateSizeAttr(1.0)
    _scale(seat, (SEAT_W, SEAT_D, SEAT_T))
    _translate(seat, (0.0, 0.0, SEAT_H))
    _bind(seat.GetPrim(), shell_mat)

    # Backrest
    back = _add(UsdGeom.Cube.Define(stage, root_path + "/Backrest"))
    back.CreateSizeAttr(1.0)
    _scale(back, (BACK_W, BACK_T, BACK_H))
    _translate(back, (0.0, -(SEAT_D / 2 - BACK_T / 2),
                      SEAT_H + SEAT_T / 2 + BACK_H / 2))
    _bind(back.GetPrim(), shell_mat)

    # 4 tubular steel legs
    inset = 0.04
    for li, (lx, ly) in enumerate([
        ( SEAT_W/2 - inset,  SEAT_D/2 - inset),
        (-SEAT_W/2 + inset,  SEAT_D/2 - inset),
        ( SEAT_W/2 - inset, -SEAT_D/2 + inset),
        (-SEAT_W/2 + inset, -SEAT_D/2 + inset),
    ]):
        leg = _add(UsdGeom.Cylinder.Define(stage, root_path + f"/Leg_{li:02d}"))
        leg.CreateRadiusAttr(LEG_R)
        leg.CreateHeightAttr(LEG_H)
        leg.CreateAxisAttr(UsdGeom.Tokens.z)
        _translate(leg, (lx, ly, LEG_H / 2))
        _bind(leg.GetPrim(), leg_mat)

    # Cross-braces
    for bi, (by, bw) in enumerate([(SEAT_D/2 - inset, SEAT_W - inset*2),
                                    (-SEAT_D/2 + inset, SEAT_W - inset*2)]):
        br = _add(UsdGeom.Cylinder.Define(stage, root_path + f"/Brace_{bi}"))
        br.CreateRadiusAttr(LEG_R * 0.8)
        br.CreateHeightAttr(bw)
        br.CreateAxisAttr(UsdGeom.Tokens.x)
        _translate(br, (0.0, by, 0.22))
        _bind(br.GetPrim(), leg_mat)

    side = _add(UsdGeom.Cylinder.Define(stage, root_path + "/BraceSide"))
    side.CreateRadiusAttr(LEG_R * 0.8)
    side.CreateHeightAttr(SEAT_D - inset * 2)
    side.CreateAxisAttr(UsdGeom.Tokens.y)
    _translate(side, (SEAT_W/2 - inset, 0.0, 0.22))
    _bind(side.GetPrim(), leg_mat)

    # Tablet arm
    arm_x = SEAT_W / 2 + 0.02
    post = _add(UsdGeom.Cylinder.Define(stage, root_path + "/ArmPost"))
    post.CreateRadiusAttr(LEG_R)
    post.CreateHeightAttr(0.30)
    post.CreateAxisAttr(UsdGeom.Tokens.z)
    _translate(post, (arm_x, SEAT_D/2 - inset, SEAT_H - 0.15))
    _bind(post.GetPrim(), leg_mat)

    bar = _add(UsdGeom.Cylinder.Define(stage, root_path + "/ArmBar"))
    bar.CreateRadiusAttr(LEG_R)
    bar.CreateHeightAttr(0.28)
    bar.CreateAxisAttr(UsdGeom.Tokens.y)
    _translate(bar, (arm_x, 0.04, SEAT_H + 0.18))
    _bind(bar.GetPrim(), leg_mat)

    tablet = _add(UsdGeom.Cube.Define(stage, root_path + "/Tablet"))
    tablet.CreateSizeAttr(1.0)
    _scale(tablet, (0.32, 0.26, 0.025))
    _translate(tablet, (arm_x + 0.09, 0.10, SEAT_H + 0.22))
    _bind(tablet.GetPrim(), tablet_mat)

    lip = _add(UsdGeom.Cube.Define(stage, root_path + "/TabletLip"))
    lip.CreateSizeAttr(1.0)
    _scale(lip, (0.32, 0.015, 0.018))
    _translate(lip, (arm_x + 0.09, 0.10 + 0.13, SEAT_H + 0.235))
    _bind(lip.GetPrim(), leg_mat)

    return geo_prims


def spawn_physics_chair(stage, chair_path, world_pos, color_idx, initial_tilt_deg,
                        tilt_heading_deg, yaw_deg):
    """
    Spawn a single physics-enabled chair at world_pos on the 4th floor.

    initial_tilt_deg  — how far off-vertical the chair starts (8–18° is enough
                        for gravity to topple it reliably)
    tilt_heading_deg  — compass direction the chair falls toward (0–360°)
    yaw_deg           — random rotation around Z so chairs face different ways
    """
    chair_xform = UsdGeom.Xform.Define(stage, chair_path)

    # Position: slightly above floor so physics doesn't start in collision
    UsdGeom.XformCommonAPI(chair_xform).SetTranslate(
        Gf.Vec3d(world_pos[0], world_pos[1], world_pos[2] + 0.01)
    )

    # Initial tilt toward tilt_heading_deg
    rad = math.radians(tilt_heading_deg)
    rot_x = initial_tilt_deg * math.cos(rad)
    rot_y = initial_tilt_deg * math.sin(rad)
    UsdGeom.XformCommonAPI(chair_xform).SetRotate(Gf.Vec3f(rot_x, rot_y, yaw_deg))

    # Materials
    color     = CHAIR_COLORS[color_idx % len(CHAIR_COLORS)]
    looks     = chair_path + "/Looks"
    shell_mat  = _mat(stage, looks + "/Shell",  color,            roughness=0.60)
    leg_mat    = _mat(stage, looks + "/Legs",   (0.72, 0.72, 0.74), roughness=0.30, metallic=0.80)
    tablet_mat = _mat(stage, looks + "/Tablet", (0.82, 0.78, 0.68), roughness=0.55)

    geo_prims = _build_chair_geometry(stage, chair_path, shell_mat, leg_mat, tablet_mat)

    # ── Physics ──────────────────────────────────────────────────────────────
    # RigidBody on the root xform makes the whole chair a single dynamic object
    rb = UsdPhysics.RigidBodyAPI.Apply(chair_xform.GetPrim())

    mass_api = UsdPhysics.MassAPI.Apply(chair_xform.GetPrim())
    mass_api.CreateMassAttr(4.5)                          # ~4.5 kg plastic chair
    mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(0.4, 0.4, 0.2))

    # Collision on each geometry piece — simple primitives are handled natively
    for geom in geo_prims:
        col = UsdPhysics.CollisionAPI.Apply(geom.GetPrim())
        # Use bounding-box approximation for cylinders (faster & more stable)
        if HAS_PHYSX_SCHEMA and geom.GetPrim().GetTypeName() == "Cylinder":
            PhysxSchema.PhysxCollisionAPI.Apply(geom.GetPrim()).CreateRestOffsetAttr(0.0)

    print(f"  Spawned chair at ({world_pos[0]:.1f}, {world_pos[1]:.1f}, {world_pos[2]:.1f})"
          f"  tilt={initial_tilt_deg:.1f}° toward {tilt_heading_deg:.0f}°")
    return chair_xform


# ===========================================================================
# Camera rendering
# ===========================================================================

def render_shots(stage, centre_pos, label, n_shots, resolution, subframes, out_dir):
    import shutil

    cx, cy, cz = centre_pos

    target = (cx, cy, cz + 0.5)

    # Overhead shots first — guaranteed not to clip into walls.
    # Eye-level shots use a smaller 2.5 m offset to reduce wall-clipping risk.
    overhead_shots = [
        ((cx,       cy,       cz + 3.0), (cx, cy, cz + 0.2)),   # straight down
        ((cx + 1.5, cy,       cz + 2.5), target),               # angled overhead +X
        ((cx,       cy + 1.5, cz + 2.5), target),               # angled overhead +Y
        ((cx - 1.5, cy,       cz + 2.5), target),               # angled overhead -X
    ]
    eyelevel_shots = [
        ((cx + 2.5, cy,       cz + 1.5), target),
        ((cx - 2.5, cy,       cz + 1.5), target),
        ((cx,       cy + 2.5, cz + 1.5), target),
        ((cx,       cy - 2.5, cz + 1.5), target),
    ]
    # Always take overhead shots first so n_shots=2 never picks a wall-clipping eye-level shot
    shot_configs = (overhead_shots + eyelevel_shots)[:n_shots]

    tmp_dir = out_dir / "_tmp_writer"
    if tmp_dir.exists():
        import shutil as _sh
        _sh.rmtree(str(tmp_dir), ignore_errors=True)
    saved = []

    for i, (cam_pos, look_at) in enumerate(shot_configs):
        px, py, pz = cam_pos
        dist = math.dist(cam_pos, look_at)

        cam = rep.create.camera(
            position=(px, py, pz),
            look_at=look_at,
            focal_length=24,
            focus_distance=dist,
            f_stop=0.0,
            clipping_range=(0.05, 10000.0),
        )
        rp = rep.create.render_product(cam, resolution)

        # Fill light above camera, out of frame
        fill = UsdLux.SphereLight.Define(stage, f"/World/Lights/Fill_{i:03d}")
        fill.CreateIntensityAttr(25000.0)
        fill.CreateRadiusAttr(0.5)
        fill.CreateColorAttr(Gf.Vec3f(1.0, 0.97, 0.92))
        UsdGeom.XformCommonAPI(fill).SetTranslate(Gf.Vec3d(px, py, pz + 2.0))

        for _ in range(10):
            simulation_app.update()

        shot_tmp = tmp_dir / f"shot_{i:03d}"
        shot_tmp.mkdir(parents=True, exist_ok=True)

        writer = rep.WriterRegistry.get("BasicWriter")
        writer.initialize(output_dir=str(shot_tmp), rgb=True,
                          bounding_box_2d_tight=False,
                          semantic_segmentation=False)
        writer.attach(rp)
        rep.orchestrator.step(rt_subframes=subframes)
        for _ in range(20):
            simulation_app.update()
        writer.detach()
        rp.destroy()

        if (shot_tmp / "rgb").exists():
            rgb_files = sorted((shot_tmp / "rgb").glob("*.png"))
        else:
            rgb_files = sorted(shot_tmp.glob("rgb_*.png"))
        if not rgb_files:
            rgb_files = sorted(shot_tmp.rglob("*.png"))

        final_path = out_dir / f"{label}_shot{i:03d}.png"
        if rgb_files:
            shutil.move(str(rgb_files[-1]), str(final_path))
            saved.append(str(final_path))
            print(f"    Shot {i+1}/{n_shots} → {final_path.name}")
        else:
            print(f"    Shot {i+1}/{n_shots} WARNING: no PNG in {shot_tmp}")

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
            return
        with imageio.get_writer(str(output_path), fps=fps, codec="libx264",
                                quality=8, pixelformat="yuv420p",
                                macro_block_size=1) as w:
            for f in frames:
                w.append_data(f)
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
    for _ in range(60):
        simulation_app.update()

    stage = omni.usd.get_context().get_stage()
    if not stage:
        print("ERROR: Stage failed to load.")
        simulation_app.close()
        return

    add_lights(stage)
    setup_physics(stage)

    print("  Warming up renderer...")
    for _ in range(30):
        simulation_app.update()

    # ── Spawn chairs on the 4th floor ───────────────────────────────────────
    print(f"\n=== Spawning {args.chairs} physics chair(s) on floor 4 (Z={FLOOR4_Z:.1f} m) ===")

    floor4_positions = get_floor4_positions(stage)
    if not floor4_positions:
        print("ERROR: No floor 4 positions found — cannot spawn chairs.")
        simulation_app.close()
        return

    # Pick one anchor position and spread chairs around it
    anchor = random.choice(floor4_positions)
    anchor_x, anchor_y = anchor[0], anchor[1]
    print(f"  Anchor position: ({anchor_x:.1f}, {anchor_y:.1f}, {FLOOR4_Z:.1f})")

    chair_paths = []
    spawn_positions = []
    UsdGeom.Xform.Define(stage, "/World/PhysicsChairs")

    for c in range(args.chairs):
        # Spread chairs around the anchor so they don't overlap
        angle  = (c / max(args.chairs, 1)) * 2 * math.pi + random.uniform(-0.4, 0.4)
        radius = random.uniform(0.5, SPAWN_SPREAD) if args.chairs > 1 else 0.0
        wx = anchor_x + radius * math.cos(angle)
        wy = anchor_y + radius * math.sin(angle)
        wz = FLOOR4_Z

        chair_path = f"/World/PhysicsChairs/Chair_{c:03d}"

        tilt     = random.uniform(10, 20)      # degrees off-vertical — enough to fall
        heading  = random.uniform(0, 360)      # which way it falls
        yaw      = random.uniform(0, 360)      # facing direction

        spawn_physics_chair(stage, chair_path, (wx, wy, wz),
                            color_idx=c,
                            initial_tilt_deg=tilt,
                            tilt_heading_deg=heading,
                            yaw_deg=yaw)
        chair_paths.append(chair_path)
        spawn_positions.append((wx, wy, wz))

    # ── Run physics simulation ───────────────────────────────────────────────
    print(f"\nRunning physics simulation ({args.sim_steps} steps "
          f"= {args.sim_steps/60:.1f} s of simulated time)...")
    for step in range(args.sim_steps):
        simulation_app.update()
        if step % 60 == 0:
            print(f"  Physics step {step}/{args.sim_steps} "
                  f"({step/60:.1f} s simulated)...")

    print("  Simulation complete — chairs have settled.")

    # ── Render from camera centred on the chair cluster ───────────────────
    # Use the mean spawn position as the scene centre
    mean_x = sum(p[0] for p in spawn_positions) / len(spawn_positions)
    mean_y = sum(p[1] for p in spawn_positions) / len(spawn_positions)
    scene_centre = (mean_x, mean_y, FLOOR4_Z)

    print(f"\nRendering {args.shots} shot(s) centred on ({mean_x:.1f}, {mean_y:.1f}, {FLOOR4_Z:.1f})...")
    all_images = render_shots(
        stage, scene_centre,
        "PhysicsChairs",
        args.shots,
        tuple(args.res),
        args.subframes,
        IMG_DIR,
    )

    if not args.no_video and all_images:
        encode_video(all_images, OUT_DIR / "physics_chairs.mp4", fps=6)

    print(f"\nDone.  {len(all_images)} images → {IMG_DIR}/")
    simulation_app.close()


main()

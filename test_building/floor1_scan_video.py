#!/usr/bin/env python3
"""
Generate an MP4 visualization of the occupancy map ray-scan process for floor 1.

The video shows the 2D occupancy grid being built up scan-by-scan:
  - Dark navy  = unexplored (UNKNOWN)
  - Light blue = free space
  - Near-black = occupied (walls)
  - Green flash = cells newly discovered this scan step
  - Yellow dot  = current scanner origin

Structure:
  1. Blank (all-unknown) intro held for INTRO_SECS
  2. Per scan origin: 1 flash frame (green discovery) + SETTLE_FRAMES settled frames
  3. Final state held for HOLD_FINAL_SECS

Run with:
  sbatch submit_isaac_job.sh test_building/floor1_scan_video.py
"""

import os
import sys
import numpy as np

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

import omni
import omni.kit.app
import omni.timeline
from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, Sdf

# ---------------------------------------------------------------------------
# Scan config  (must match generate_occupancy_map.py)
# ---------------------------------------------------------------------------
PROJECT_DIR  = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR   = os.path.join(PROJECT_DIR, "presentation_nav_visuals")
USD_PATH     = os.path.join(PROJECT_DIR, "MSOEBuildingV4_physics.usd")

CELL_SIZE    = 0.05   # meters — V4 stage units
OCCUPIED     = 1
FREE         = 0
UNKNOWN      = 2

FLOOR_NAME   = "floor_1"
Z_MIN        = 14.0
Z_MAX        = 16.0

GRID_ORIGINS_X           = 5
GRID_ORIGINS_Y           = 5
MAX_RESCAN_ROUNDS        = 6
RESCAN_ORIGINS_PER_ROUND = 50

# ---------------------------------------------------------------------------
# Video config
# ---------------------------------------------------------------------------
VIDEO_FPS        = 30    # 2× faster than the previous 15-fps version
VIDEO_DOWNSAMPLE = 2     # ~1000×1000 output (2000/2)
INTRO_SECS       = 1.5   # blank dark-navy screen at the start
FLASH_FRAMES     = 1     # frames showing green discovery burst per scan step
SETTLE_FRAMES    = 2     # frames showing settled colours per scan step
HOLD_FINAL_SECS  = 3     # hold the finished map at the end

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
COL_UNKNOWN  = np.array([15,  25,  60],  dtype=np.uint8)  # dark navy
COL_FREE     = np.array([190, 220, 255], dtype=np.uint8)  # pale blue
COL_OCCUPIED = np.array([25,  25,  35],  dtype=np.uint8)  # near-black
COL_NEW_FREE = np.array([80,  210, 120], dtype=np.uint8)  # green flash
COL_ORIGIN   = np.array([255, 220,   0], dtype=np.uint8)  # yellow dot


# ---------------------------------------------------------------------------
# Frame rendering
# ---------------------------------------------------------------------------
def make_frame(grid, prev_grid=None, origin_cell=None, ds=1):
    """Render the grid as an RGB frame, optionally highlighting new free cells
    and the current scanner origin.  ds = downsample factor."""
    h, w = grid.shape

    img = np.empty((h, w, 3), dtype=np.uint8)
    img[grid == UNKNOWN]  = COL_UNKNOWN
    img[grid == FREE]     = COL_FREE
    img[grid == OCCUPIED] = COL_OCCUPIED

    if prev_grid is not None:
        newly_free = (prev_grid == UNKNOWN) & (grid == FREE)
        img[newly_free] = COL_NEW_FREE

    if origin_cell is not None:
        cx, cy = origin_cell
        radius = max(5, 16 // ds)
        r0, r1 = max(0, cy - radius), min(h, cy + radius + 1)
        c0, c1 = max(0, cx - radius), min(w, cx + radius + 1)
        for r in range(r0, r1):
            for c in range(c0, c1):
                if (r - cy) ** 2 + (c - cx) ** 2 <= radius ** 2:
                    img[r, c] = COL_ORIGIN

    if ds > 1:
        img = img[::ds, ::ds]

    return img


def world_to_cell(wx, wy, world_min_x, world_min_y, cell_size):
    col = int((wx - world_min_x) / cell_size)
    row = int((wy - world_min_y) / cell_size)
    return col, row


# ---------------------------------------------------------------------------
# Occupancy map helpers
# ---------------------------------------------------------------------------
def scan_from_origin(generator, origin, rel_min, rel_max, cell_size):
    generator.update_settings(cell_size, OCCUPIED, FREE, UNKNOWN)
    generator.set_transform(origin, rel_min, rel_max)
    simulation_app.update()
    generator.generate2d()
    simulation_app.update()
    buf  = generator.get_buffer()
    dims = generator.get_dimensions()
    if len(buf) == 0 or dims[0] == 0 or dims[1] == 0:
        return None, dims
    return np.array(buf, dtype=np.float32).reshape(dims[1], dims[0]), dims


def merge_grids(base, new):
    h = min(base.shape[0], new.shape[0])
    w = min(base.shape[1], new.shape[1])
    b, n = base[:h, :w], new[:h, :w]
    uk = b == UNKNOWN
    b[uk & (n == FREE)]     = FREE
    b[uk & (n == OCCUPIED)] = OCCUPIED
    b[n == OCCUPIED]        = OCCUPIED
    return base


def find_unknown_origins(grid, wx_min, wy_min, cs, oz, n=10):
    coords = np.argwhere(grid == UNKNOWN)
    if len(coords) == 0:
        return []
    idxs = np.linspace(0, len(coords) - 1, min(n, len(coords)), dtype=int)
    return [(wx_min + coords[i][1] * cs + cs / 2,
             wy_min + coords[i][0] * cs + cs / 2, oz)
            for i in idxs]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Floor 1 Scan Visualizer  —  MP4 output")
    print("=" * 60)
    sys.stdout.flush()

    # --- Load USD ---
    print(f"\nLoading: {USD_PATH}")
    if not os.path.exists(USD_PATH):
        print("ERROR: USD not found"); simulation_app.close(); sys.exit(1)

    omni.usd.get_context().open_stage(USD_PATH)
    for _ in range(3):
        simulation_app.update()
    stage = omni.usd.get_context().get_stage()
    print(f"  {len(list(stage.Traverse()))} prims loaded")
    sys.stdout.flush()

    # --- Collision ---
    SKIP = ("Room_", "Model_Text")
    n_mesh = n_door = n_skip = 0
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        if "/Doors/" in str(prim.GetPath()):
            n_door += 1; continue
        if any(prim.GetName().startswith(p) for p in SKIP):
            n_skip += 1; continue
        n_mesh += 1
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI.Apply(prim)
    print(f"  Collision applied to {n_mesh} meshes "
          f"({n_skip} non-arch skipped, {n_door} doors skipped)")
    sys.stdout.flush()

    # --- Physics ---
    psp = "/World/physicsScene"
    if not stage.GetPrimAtPath(psp):
        UsdPhysics.Scene.Define(stage, Sdf.Path(psp))
    ps = PhysxSchema.PhysxSceneAPI.Get(stage, psp) or \
         PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath(psp))
    ps.CreateEnableGPUDynamicsAttr(False)
    ps.CreateBroadphaseTypeAttr("MBP")
    simulation_app.update(); simulation_app.update()
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    simulation_app.update(); simulation_app.update()

    # --- omap extension ---
    omni.kit.app.get_app().get_extension_manager() \
        .set_extension_enabled_immediate("isaacsim.asset.gen.omap", True)
    simulation_app.update()
    from isaacsim.asset.gen.omap.bindings import _omap

    # --- Bounding box ---
    bc     = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
    wp     = stage.GetPrimAtPath("/World") or stage.GetPseudoRoot()
    br     = bc.ComputeWorldBound(wp).ComputeAlignedRange()
    bb_min = br.GetMin()
    bb_max = br.GetMax()

    PAD       = 1.0
    wx_min    = float(bb_min[0]) - PAD
    wy_min    = float(bb_min[1]) - PAD
    wx_max    = float(bb_max[0]) + PAD
    wy_max    = float(bb_max[1]) + PAD

    exp_w = int(np.ceil((wx_max - wx_min) / CELL_SIZE))
    exp_h = int(np.ceil((wy_max - wy_min) / CELL_SIZE))
    out_w, out_h = exp_w // VIDEO_DOWNSAMPLE, exp_h // VIDEO_DOWNSAMPLE
    print(f"\n  Grid: {exp_w}×{exp_h}  →  Video: {out_w}×{out_h} px @ {VIDEO_FPS} fps")
    sys.stdout.flush()

    physx    = omni.physx.get_physx_interface()
    stage_id = omni.usd.get_context().get_stage_id()
    oz       = (Z_MIN + Z_MAX) / 2.0

    base  = np.full((exp_h, exp_w), UNKNOWN, dtype=np.float32)
    frames = []

    # ---- Intro: blank (all-unknown) screen ----
    intro_count = int(VIDEO_FPS * INTRO_SECS)
    blank_frame = make_frame(base, ds=VIDEO_DOWNSAMPLE)
    frames.extend([blank_frame] * intro_count)
    print(f"  Intro: {intro_count} frames ({INTRO_SECS}s blank screen)")
    sys.stdout.flush()

    def add_scan(origin_xy, new_grid):
        """Merge scan and append flash + settle frames."""
        nonlocal base
        prev  = base.copy()
        base  = merge_grids(base, new_grid)
        oc    = world_to_cell(origin_xy[0], origin_xy[1], wx_min, wy_min, CELL_SIZE)
        # Flash frame  — green burst + yellow dot
        frames.append(make_frame(base, prev, oc, VIDEO_DOWNSAMPLE))
        # Settle frames — no flash, dot fades
        for _ in range(SETTLE_FRAMES):
            frames.append(make_frame(base, None, oc, VIDEO_DOWNSAMPLE))

    # ---- Round 1: grid origins ----
    ox_vals = np.linspace(float(bb_min[0]) + 0.5, float(bb_max[0]) - 0.5, GRID_ORIGINS_X)
    oy_vals = np.linspace(float(bb_min[1]) + 0.5, float(bb_max[1]) - 0.5, GRID_ORIGINS_Y)

    print(f"\nRound 1: {GRID_ORIGINS_X}×{GRID_ORIGINS_Y} = "
          f"{GRID_ORIGINS_X * GRID_ORIGINS_Y} origins...")
    scan_count = 0
    for ox in ox_vals:
        for oy in oy_vals:
            origin  = (float(ox), float(oy), oz)
            rel_min = (wx_min - ox, wy_min - oy, Z_MIN - oz)
            rel_max = (wx_max - ox, wy_max - oy, Z_MAX - oz)
            g, _    = scan_from_origin(_omap.Generator(physx, stage_id),
                                       origin, rel_min, rel_max, CELL_SIZE)
            if g is None:
                continue
            add_scan((ox, oy), g)
            scan_count += 1

    n_unk         = int(np.sum(base == UNKNOWN))
    total         = base.size
    prev_unk_pct  = 100 * n_unk / total
    print(f"  {scan_count} scans → {prev_unk_pct:.1f}% unknown")
    sys.stdout.flush()

    # ---- Rescan rounds ----
    for rnd in range(MAX_RESCAN_ROUNDS):
        n_unk     = int(np.sum(base == UNKNOWN))
        unk_pct   = 100 * n_unk / total
        if n_unk == 0 or unk_pct < 0.5:
            print(f"  {unk_pct:.1f}% unknown — done"); break
        if rnd > 0 and (prev_unk_pct - unk_pct) < 0.1:
            print(f"  Stagnation at {unk_pct:.1f}% — stopping"); break

        extras = find_unknown_origins(base, wx_min, wy_min, CELL_SIZE, oz,
                                      n=RESCAN_ORIGINS_PER_ROUND)
        if not extras:
            break

        print(f"\n  Round {rnd+2}: {len(extras)} rescan origins "
              f"({unk_pct:.1f}% unknown)...")
        sys.stdout.flush()

        for origin in extras:
            ox, oy, _  = origin
            rel_min    = (wx_min - ox, wy_min - oy, Z_MIN - oz)
            rel_max    = (wx_max - ox, wy_max - oy, Z_MAX - oz)
            g, _       = scan_from_origin(_omap.Generator(physx, stage_id),
                                          origin, rel_min, rel_max, CELL_SIZE)
            if g is None:
                continue
            add_scan((ox, oy), g)
            scan_count += 1

        prev_unk_pct = unk_pct
        print(f"  {scan_count} total scans → "
              f"{100*np.sum(base==UNKNOWN)/total:.1f}% unknown")
        sys.stdout.flush()

    # ---- Final hold ----
    final_frame = make_frame(base, ds=VIDEO_DOWNSAMPLE)
    hold_count  = int(VIDEO_FPS * HOLD_FINAL_SECS)
    frames.extend([final_frame] * hold_count)

    duration = len(frames) / VIDEO_FPS
    print(f"\nTotal: {len(frames)} frames / {VIDEO_FPS} fps = {duration:.1f}s video")
    print(f"  ({scan_count} scan steps × {FLASH_FRAMES+SETTLE_FRAMES} frames each"
          f" + {intro_count} intro + {hold_count} outro)")
    sys.stdout.flush()

    # ---- Save occupancy npy + png ----
    n_occ  = int(np.sum(base == OCCUPIED))
    n_free = int(np.sum(base == FREE))
    n_unk  = int(np.sum(base == UNKNOWN))
    print(f"\nFinal grid: Occupied {n_occ} ({100*n_occ/total:.1f}%),  "
          f"Free {n_free} ({100*n_free/total:.1f}%),  "
          f"Unknown {n_unk} ({100*n_unk/total:.1f}%)")

    np.save(os.path.join(OUTPUT_DIR, f"{FLOOR_NAME}_occupancy.npy"), base)
    img_out                   = np.zeros((exp_h, exp_w, 3), dtype=np.uint8)
    img_out[base == OCCUPIED] = [0, 0, 0]
    img_out[base == FREE]     = [255, 255, 255]
    img_out[base == UNKNOWN]  = [128, 128, 128]
    try:
        from PIL import Image
        Image.fromarray(img_out).save(
            os.path.join(OUTPUT_DIR, f"{FLOOR_NAME}_occupancy.png"))
    except ImportError:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.imsave(os.path.join(OUTPUT_DIR, f"{FLOOR_NAME}_occupancy.png"), img_out)

    # ---- Write MP4 ----
    video_path = os.path.join(OUTPUT_DIR, f"{FLOOR_NAME}_scan_video.mp4")
    print(f"\nWriting MP4 → {video_path}")
    sys.stdout.flush()

    try:
        import imageio
        # libx264 requires even dimensions — trim 1px if odd
        h = frames[0].shape[0] & ~1
        w = frames[0].shape[1] & ~1
        writer = imageio.get_writer(
            video_path,
            fps=VIDEO_FPS,
            codec="libx264",
            quality=9,
            macro_block_size=None,
            output_params=["-pix_fmt", "yuv420p"],
        )
        for frame in frames:
            writer.append_data(frame[:h, :w])
        writer.close()
        size_mb = os.path.getsize(video_path) / (1024 * 1024)
        print(f"  Done: {video_path}  ({size_mb:.1f} MB, "
              f"{len(frames)} frames, {duration:.1f}s)")
    except Exception as e:
        print(f"  imageio failed ({e}), saving raw frames as npy fallback")
        np.save(os.path.join(OUTPUT_DIR, f"{FLOOR_NAME}_frames.npy"),
                np.array(frames))

    sys.stdout.flush()
    timeline.stop()
    print("\n" + "=" * 60 + "\nDone!\n" + "=" * 60)
    sys.stdout.flush()
    simulation_app.close()


if __name__ == "__main__":
    main()

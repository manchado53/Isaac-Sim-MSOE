#!/usr/bin/env python3
"""
Generates a floor-1 occupancy map from scratch and records the entire
scan process as an MP4 video.

Visual encoding:
  Dark navy   = unexplored cells (UNKNOWN)
  Pale blue   = free / navigable space
  Near-black  = occupied (walls, obstacles)
  Green flash = cells newly discovered by this scan step
  Yellow dot  = current scanner origin

Output (both in this folder):
  floor1_occupancy_map.npy   — raw occupancy grid (0=free,1=occupied,2=unknown)
  floor1_occupancy_map.png   — static top-down floor plan image
  floor1_scan_video.mp4      — time-lapse of the map being built

Run with:
  sbatch submit_isaac_job.sh test_building/presentation_nav_visuals/floor1_occupancy_video.py
"""

import os, sys
import numpy as np

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

import omni, omni.kit.app, omni.timeline
from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, Sdf

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))      # presentation_nav_visuals/
OUTPUT_DIR = SCRIPT_DIR
USD_PATH   = os.path.join(os.path.dirname(SCRIPT_DIR), "MSOEBuildingV4_physics.usd")

# ---------------------------------------------------------------------------
# Occupancy map settings
# ---------------------------------------------------------------------------
CELL_SIZE  = 0.05   # 5 cm resolution (V4 stage units are metres)
FREE       = 0
OCCUPIED   = 1
UNKNOWN    = 2

FLOOR_NAME = "floor1"
Z_MIN      = 14.0   # metres — robot walking height on floor 1
Z_MAX      = 16.0

GRID_ORIGINS_X           = 5   # initial scan grid
GRID_ORIGINS_Y           = 5
MAX_RESCAN_ROUNDS        = 6
RESCAN_ORIGINS_PER_ROUND = 50

PADDING = 1.0  # metres added around building bounding box

# ---------------------------------------------------------------------------
# Video settings
# ---------------------------------------------------------------------------
VIDEO_FPS      = 30       # frames per second
DOWNSAMPLE     = 2        # output res = grid_size / DOWNSAMPLE  (~1000×1000 px)
INTRO_SECS     = 1.5      # blank dark screen at the start
FLASH_FRAMES   = 1        # frames showing the green discovery burst per scan
SETTLE_FRAMES  = 2        # frames showing settled colours after the flash
FINAL_HOLD_SECS = 3.0     # freeze on finished map at the end

# ---------------------------------------------------------------------------
# Colours  (RGB uint8)
# ---------------------------------------------------------------------------
C_UNKNOWN  = np.array([ 15,  25,  60], dtype=np.uint8)   # dark navy
C_FREE     = np.array([190, 220, 255], dtype=np.uint8)   # pale blue
C_OCCUPIED = np.array([ 25,  25,  35], dtype=np.uint8)   # near-black
C_NEW_FREE = np.array([ 80, 210, 120], dtype=np.uint8)   # green flash
C_ORIGIN   = np.array([255, 220,   0], dtype=np.uint8)   # yellow dot


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def render(grid, prev_grid=None, origin_cell=None, ds=1):
    """Return an RGB frame for the current grid state."""
    h, w = grid.shape
    img = np.empty((h, w, 3), dtype=np.uint8)
    img[grid == UNKNOWN]  = C_UNKNOWN
    img[grid == FREE]     = C_FREE
    img[grid == OCCUPIED] = C_OCCUPIED

    # Green flash on newly discovered free cells
    if prev_grid is not None:
        img[(prev_grid == UNKNOWN) & (grid == FREE)] = C_NEW_FREE

    # Yellow dot at scanner origin
    if origin_cell is not None:
        cx, cy = origin_cell
        r = max(5, 16 // ds)
        r0, r1 = max(0, cy - r), min(h, cy + r + 1)
        c0, c1 = max(0, cx - r), min(w, cx + r + 1)
        ys, xs = np.ogrid[r0:r1, c0:c1]
        mask   = (ys - cy) ** 2 + (xs - cx) ** 2 <= r ** 2
        img[r0:r1, c0:c1][mask] = C_ORIGIN

    return img[::ds, ::ds] if ds > 1 else img


def world_to_cell(wx, wy, wx_min, wy_min, cs):
    return int((wx - wx_min) / cs), int((wy - wy_min) / cs)


def do_scan(generator, origin, rel_min, rel_max, cs):
    generator.update_settings(cs, OCCUPIED, FREE, UNKNOWN)
    generator.set_transform(origin, rel_min, rel_max)
    simulation_app.update()
    generator.generate2d()
    simulation_app.update()
    buf, dims = generator.get_buffer(), generator.get_dimensions()
    if not buf or dims[0] == 0:
        return None
    return np.array(buf, dtype=np.float32).reshape(dims[1], dims[0])


def merge(base, new):
    h, w   = min(base.shape[0], new.shape[0]), min(base.shape[1], new.shape[1])
    b, n   = base[:h, :w], new[:h, :w]
    uk     = b == UNKNOWN
    b[uk  & (n == FREE)]     = FREE
    b[uk  & (n == OCCUPIED)] = OCCUPIED
    b[n  == OCCUPIED]        = OCCUPIED
    return base


def unknown_origins(grid, wx_min, wy_min, cs, oz, n=10):
    pts = np.argwhere(grid == UNKNOWN)
    if not len(pts):
        return []
    idx = np.linspace(0, len(pts) - 1, min(n, len(pts)), dtype=int)
    return [(wx_min + pts[i, 1] * cs + cs / 2,
             wy_min + pts[i, 0] * cs + cs / 2, oz) for i in idx]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Floor 1  —  Occupancy Map + Video")
    print("=" * 60); sys.stdout.flush()

    # ── Load USD ──────────────────────────────────────────────────────────
    if not os.path.exists(USD_PATH):
        print(f"ERROR: USD not found: {USD_PATH}")
        simulation_app.close(); sys.exit(1)

    print(f"\nLoading: {USD_PATH}")
    omni.usd.get_context().open_stage(USD_PATH)
    for _ in range(3):
        simulation_app.update()
    stage = omni.usd.get_context().get_stage()
    print(f"  {len(list(stage.Traverse()))} prims"); sys.stdout.flush()

    # ── Collision (skip doors so raycasts pass through doorways) ──────────
    n_mesh = n_door = n_skip = 0
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        if "/Doors/" in str(prim.GetPath()):
            n_door += 1; continue
        if any(prim.GetName().startswith(p) for p in ("Room_", "Model_Text")):
            n_skip += 1; continue
        n_mesh += 1
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI.Apply(prim)
    print(f"  Collision: {n_mesh} meshes  "
          f"({n_skip} non-arch + {n_door} doors skipped)"); sys.stdout.flush()

    # ── Physics ────────────────────────────────────────────────────────────
    psp = "/World/physicsScene"
    if not stage.GetPrimAtPath(psp):
        UsdPhysics.Scene.Define(stage, Sdf.Path(psp))
    ps = PhysxSchema.PhysxSceneAPI.Get(stage, psp) \
         or PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath(psp))
    ps.CreateEnableGPUDynamicsAttr(False)
    ps.CreateBroadphaseTypeAttr("MBP")
    for _ in range(2):
        simulation_app.update()
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    for _ in range(2):
        simulation_app.update()

    # ── omap extension ─────────────────────────────────────────────────────
    omni.kit.app.get_app().get_extension_manager() \
        .set_extension_enabled_immediate("isaacsim.asset.gen.omap", True)
    simulation_app.update()
    from isaacsim.asset.gen.omap.bindings import _omap

    # ── Bounding box ───────────────────────────────────────────────────────
    bc     = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
    root   = stage.GetPrimAtPath("/World") or stage.GetPseudoRoot()
    br     = bc.ComputeWorldBound(root).ComputeAlignedRange()
    bb_min = br.GetMin()
    bb_max = br.GetMax()

    wx_min = float(bb_min[0]) - PADDING
    wy_min = float(bb_min[1]) - PADDING
    wx_max = float(bb_max[0]) + PADDING
    wy_max = float(bb_max[1]) + PADDING

    GW = int(np.ceil((wx_max - wx_min) / CELL_SIZE))  # grid width  (cols)
    GH = int(np.ceil((wy_max - wy_min) / CELL_SIZE))  # grid height (rows)

    # Force even video dimensions for libx264
    VW = (GW // DOWNSAMPLE) & ~1
    VH = (GH // DOWNSAMPLE) & ~1
    print(f"\n  Grid: {GW}×{GH} cells  →  Video: {VW}×{VH} px")
    sys.stdout.flush()

    # ── Open video writer ──────────────────────────────────────────────────
    video_path = os.path.join(OUTPUT_DIR, f"{FLOOR_NAME}_scan_video.mp4")
    print(f"  Output: {video_path}")
    import imageio
    writer = imageio.get_writer(
        video_path,
        fps=VIDEO_FPS,
        codec="libx264",
        quality=9,
        macro_block_size=None,
        output_params=["-pix_fmt", "yuv420p"],
    )

    def write(frame):
        """Crop to even dimensions and send to encoder."""
        writer.append_data(frame[:VH, :VW])

    # ── Grid + scan setup ──────────────────────────────────────────────────
    physx    = omni.physx.get_physx_interface()
    stage_id = omni.usd.get_context().get_stage_id()
    oz       = (Z_MIN + Z_MAX) / 2.0
    grid     = np.full((GH, GW), UNKNOWN, dtype=np.float32)
    total    = GH * GW
    n_scans  = 0

    ox_vals = np.linspace(float(bb_min[0]) + 0.5, float(bb_max[0]) - 0.5, GRID_ORIGINS_X)
    oy_vals = np.linspace(float(bb_min[1]) + 0.5, float(bb_max[1]) - 0.5, GRID_ORIGINS_Y)

    # ── Intro: blank dark screen ───────────────────────────────────────────
    blank = render(grid, ds=DOWNSAMPLE)
    intro_n = int(VIDEO_FPS * INTRO_SECS)
    for _ in range(intro_n):
        write(blank)
    print(f"\n  Intro: {intro_n} frames ({INTRO_SECS}s)"); sys.stdout.flush()

    def scan_and_record(origin):
        """Run one scan, merge into grid, write flash + settle frames."""
        nonlocal grid, n_scans
        ox, oy, _ = origin
        rel_min = (wx_min - ox, wy_min - oy, Z_MIN - oz)
        rel_max = (wx_max - ox, wy_max - oy, Z_MAX - oz)
        new = do_scan(_omap.Generator(physx, stage_id),
                      origin, rel_min, rel_max, CELL_SIZE)
        if new is None:
            return
        prev = grid.copy()
        grid = merge(grid, new)
        n_scans += 1
        oc = world_to_cell(ox, oy, wx_min, wy_min, CELL_SIZE)
        # 1 flash frame (green burst + yellow dot)
        for _ in range(FLASH_FRAMES):
            write(render(grid, prev, oc, DOWNSAMPLE))
        # settle frames (no flash, dot still visible)
        settle = render(grid, None, oc, DOWNSAMPLE)
        for _ in range(SETTLE_FRAMES):
            write(settle)

    # ── Round 1: uniform grid of origins ──────────────────────────────────
    print(f"\nRound 1: {GRID_ORIGINS_X}×{GRID_ORIGINS_Y} grid origins...")
    sys.stdout.flush()
    for ox in ox_vals:
        for oy in oy_vals:
            scan_and_record((float(ox), float(oy), oz))

    unk_pct = 100 * np.sum(grid == UNKNOWN) / total
    print(f"  {n_scans} scans → {unk_pct:.1f}% unknown"); sys.stdout.flush()

    # ── Rescan rounds: target remaining unknown regions ────────────────────
    prev_unk_pct = unk_pct
    for rnd in range(MAX_RESCAN_ROUNDS):
        unk_pct = 100 * np.sum(grid == UNKNOWN) / total
        if unk_pct < 0.5:
            print(f"  {unk_pct:.1f}% unknown — good enough, stopping"); break
        if rnd > 0 and (prev_unk_pct - unk_pct) < 0.1:
            print(f"  Stagnation at {unk_pct:.1f}% — stopping"); break

        extras = unknown_origins(grid, wx_min, wy_min, CELL_SIZE, oz,
                                 n=RESCAN_ORIGINS_PER_ROUND)
        if not extras:
            break
        print(f"\n  Round {rnd+2}: {len(extras)} rescan origins "
              f"({unk_pct:.1f}% unknown)..."); sys.stdout.flush()
        for origin in extras:
            scan_and_record(origin)
        prev_unk_pct = unk_pct
        print(f"  {n_scans} total scans → "
              f"{100*np.sum(grid==UNKNOWN)/total:.1f}% unknown")
        sys.stdout.flush()

    # ── Final hold ─────────────────────────────────────────────────────────
    final = render(grid, ds=DOWNSAMPLE)
    hold_n = int(VIDEO_FPS * FINAL_HOLD_SECS)
    for _ in range(hold_n):
        write(final)

    writer.close()
    total_frames = intro_n + n_scans * (FLASH_FRAMES + SETTLE_FRAMES) + hold_n
    duration = total_frames / VIDEO_FPS
    size_mb  = os.path.getsize(video_path) / (1024 * 1024)
    print(f"\nVideo: {total_frames} frames @ {VIDEO_FPS}fps = "
          f"{duration:.1f}s  |  {size_mb:.1f} MB")
    sys.stdout.flush()

    # ── Save static occupancy map ──────────────────────────────────────────
    npy_path = os.path.join(OUTPUT_DIR, f"{FLOOR_NAME}_occupancy_map.npy")
    png_path = os.path.join(OUTPUT_DIR, f"{FLOOR_NAME}_occupancy_map.png")
    np.save(npy_path, grid)

    img = np.full((GH, GW, 3), 128, dtype=np.uint8)
    img[grid == FREE]     = [255, 255, 255]
    img[grid == OCCUPIED] = [0,   0,   0  ]
    try:
        from PIL import Image
        Image.fromarray(img).save(png_path)
    except ImportError:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.imsave(png_path, img)

    n_occ  = int(np.sum(grid == OCCUPIED))
    n_free = int(np.sum(grid == FREE))
    n_unk  = int(np.sum(grid == UNKNOWN))
    print(f"\nOccupancy: {n_occ/total*100:.1f}% walls | "
          f"{n_free/total*100:.1f}% free | "
          f"{n_unk/total*100:.1f}% unknown")
    print(f"Saved: {npy_path}")
    print(f"Saved: {png_path}")
    print(f"Saved: {video_path}")

    timeline.stop()
    print("\n" + "=" * 60 + "\nDone!\n" + "=" * 60)
    sys.stdout.flush()
    simulation_app.close()


if __name__ == "__main__":
    main()

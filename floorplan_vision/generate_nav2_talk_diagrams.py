#!/usr/bin/env python3
"""
generate_nav2_talk_diagrams.py — produce the visuals for the
non-technical Nav2 presentation (`navigation/NAV2_TALK.md`).

Outputs to `docs/nav2_talk/*.png`. Each PNG corresponds to one slide in
the presentation. Run after any edit to regenerate.
"""

from pathlib import Path
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrow, FancyBboxPatch
from matplotlib.collections import LineCollection
from matplotlib.animation import FuncAnimation, FFMpegWriter
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT = PROJECT_ROOT / "docs" / "nav2_talk"
OUT.mkdir(parents=True, exist_ok=True)
IMAGES_OUT = PROJECT_ROOT / "images"
IMAGES_OUT.mkdir(parents=True, exist_ok=True)
FLOOR4_NPY = PROJECT_ROOT / "assets" / "environments" / "InnovationCenter" / "maps" / "floor_4_occupancy.npy"

# Let matplotlib find the static ffmpeg build we have on PATH.
plt.rcParams["animation.ffmpeg_path"] = "/home/thescript/bin/ffmpeg"


PALETTE = {
    "bg":      "#0f172a",   # slate-900
    "fg":      "#f8fafc",   # slate-50
    "muted":   "#94a3b8",
    "wall":    "#111827",
    "lethal":  "#dc2626",   # cost 253 / lethal
    "hot":     "#ea580c",
    "warm":    "#f59e0b",
    "cool":    "#22c55e",   # clear
    "accent":  "#38bdf8",
    "accent2": "#a78bfa",
    "path":    "#facc15",
}


# ---------------------------------------------------------------------------
# Slide 1 — Nav2 at a glance
# ---------------------------------------------------------------------------
def make_overview():
    fig, ax = plt.subplots(figsize=(14, 8), dpi=140)
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])
    ax.axis("off")
    ax.set_xlim(0, 14); ax.set_ylim(0, 8)

    ax.text(7, 7.1, "Nav2", fontsize=64, color=PALETTE["accent"],
            ha="center", fontweight="bold")
    ax.text(7, 6.0,
            "The open-source brain that turns a robot\n"
            "with wheels and sensors into one that can\n"
            "drive itself from point A to point B\n"
            "inside a building without hitting things.",
            fontsize=21, color=PALETTE["fg"],
            ha="center", va="top", linespacing=1.45)

    # Bottom row: the analogy card
    ax.text(7, 1.7, "≈", fontsize=60, color=PALETTE["accent2"],
            ha="center", va="center")
    ax.text(3.3, 1.7, "Google Maps", fontsize=28, color=PALETTE["fg"],
            ha="center", va="center", fontweight="bold")
    ax.text(10.7, 1.7, "car's autopilot", fontsize=28, color=PALETTE["fg"],
            ha="center", va="center", fontweight="bold")
    ax.text(7, 0.6, "…together,  for any robot.", fontsize=22,
            color=PALETTE["muted"], ha="center")

    fig.savefig(OUT / "01_overview.png", dpi=140, bbox_inches="tight",
                facecolor=PALETTE["bg"])
    plt.close(fig); print("  wrote 01_overview.png")


# ---------------------------------------------------------------------------
# Slide 2 — the four roles (AMCL / Costmap / Planner / Controller)
# ---------------------------------------------------------------------------
def make_four_roles():
    fig, ax = plt.subplots(figsize=(16, 9), dpi=140)
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])
    ax.axis("off")
    ax.set_xlim(0, 16); ax.set_ylim(0, 9)

    ax.text(8, 8.3, "Nav2 = four roles working together",
            fontsize=28, color=PALETTE["fg"],
            ha="center", fontweight="bold")

    boxes = [
        ("Localization\n(AMCL)",      "Where am I right now?",
         "matches lidar scan against the building's map",  PALETTE["accent"]),
        ("Costmap",                    "What should I avoid?",
         "paints safety halos around every wall + obstacle", PALETTE["warm"]),
        ("Planner",                    "What's the route?",
         "computes the shortest path to the goal",           PALETTE["cool"]),
        ("Controller\n(DWB)",          "How do I drive it?",
         "steers the wheels, moment-by-moment",              PALETTE["accent2"]),
    ]

    cell_w = 3.4
    margin = 0.3
    x0 = margin
    for i, (title, question, desc, color) in enumerate(boxes):
        x = x0 + i * (cell_w + 0.35)
        # Card
        card = FancyBboxPatch((x, 2.0), cell_w, 4.8,
                              boxstyle="round,pad=0.03,rounding_size=0.18",
                              linewidth=2.5, edgecolor=color,
                              facecolor="#1e293b", zorder=2)
        ax.add_patch(card)
        # Title
        ax.text(x + cell_w/2, 6.15, title, fontsize=22, color=color,
                ha="center", va="top", fontweight="bold", linespacing=1.1)
        # Question (italic)
        ax.text(x + cell_w/2, 4.75, f'"{question}"', fontsize=16,
                color=PALETTE["fg"], ha="center", va="center",
                style="italic")
        # Description
        ax.text(x + cell_w/2, 3.1, desc, fontsize=13, color=PALETTE["muted"],
                ha="center", va="center", wrap=True)

        # Step label on top edge
        ax.text(x + 0.25, 6.9, f"{i+1}",
                fontsize=30, color=color, ha="left", va="bottom",
                fontweight="bold", alpha=0.6)

    # Arrows between cards
    for i in range(3):
        x = x0 + (i + 1) * (cell_w + 0.35) - 0.35
        ax.annotate("", xy=(x + 0.32, 4.4), xytext=(x - 0.02, 4.4),
                    arrowprops=dict(arrowstyle="-|>", color=PALETTE["muted"],
                                    lw=2))

    # Bottom callout
    ax.text(8, 1.05,
            "Same split as a human driver: see where you are → spot hazards → "
            "pick a route → move the wheel",
            fontsize=15, color=PALETTE["muted"], ha="center", style="italic")

    fig.savefig(OUT / "02_four_roles.png", dpi=140, bbox_inches="tight",
                facecolor=PALETTE["bg"])
    plt.close(fig); print("  wrote 02_four_roles.png")


# ---------------------------------------------------------------------------
# Slide 3 — the costmap "safety halo"
# ---------------------------------------------------------------------------
def make_costmap_halo():
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=140,
                              gridspec_kw={"width_ratios": [1, 1]})
    fig.patch.set_facecolor(PALETTE["bg"])
    for ax in axes:
        ax.set_facecolor(PALETTE["bg"])
        ax.set_aspect("equal")
        ax.axis("off")

    # Left: Floor plan slice showing walls and cost zones
    ax = axes[0]
    ax.set_xlim(0, 10); ax.set_ylim(0, 8)
    ax.set_title("Halos around every wall",
                 fontsize=20, color=PALETTE["fg"], pad=14, fontweight="bold")

    # Draw cost zones first (behind walls) — concentric halos around two walls
    walls = [
        # Lower U-shape wall
        (1, 1, 8, 0.6),
        (1, 1, 0.6, 4),
        (8.4, 1, 0.6, 4),
        # Little room wall
        (3, 6.5, 4, 0.6),
    ]
    # For each wall, paint halos (draw biggest cool halo first, then warm, hot, lethal)
    halo_sizes = [(1.0, PALETTE["cool"], 0.25),
                  (0.7, PALETTE["warm"], 0.45),
                  (0.45, PALETTE["hot"], 0.6),
                  (0.25, PALETTE["lethal"], 0.85)]
    # Paint a coarse occupancy + halo grid instead, easier:
    H, W = 320, 400  # px
    grid = np.zeros((H, W), dtype=np.float32)
    for (wx, wy, ww, wh) in walls:
        x0 = int(wx/10 * W); x1 = int((wx+ww)/10 * W)
        y0 = int(wy/8 * H);  y1 = int((wy+wh)/8 * H)
        grid[y0:y1, x0:x1] = 1.0  # wall
    # distance transform
    from scipy.ndimage import distance_transform_edt
    dist = distance_transform_edt(grid < 0.5)  # pixel distance to nearest wall
    dist_m = dist * (10.0 / W)  # convert to "meters"
    # Cost = exp(-k*(d - inscribed))
    inscribed = 0.05
    k = 0.6  # scaling for visual
    cost = np.exp(-k * (dist_m - inscribed))
    cost[dist_m < inscribed] = 1.0
    cost[grid > 0.5] = np.nan  # wall

    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        "halo",
        [(0.0, PALETTE["cool"]),
         (0.25, PALETTE["warm"]),
         (0.6, PALETTE["hot"]),
         (1.0, PALETTE["lethal"])])
    cmap.set_bad(PALETTE["wall"])
    ax.imshow(cost, extent=[0, 10, 0, 8], origin="lower",
              cmap=cmap, vmin=0, vmax=1.0, alpha=0.95)
    for (wx, wy, ww, wh) in walls:
        ax.add_patch(plt.Rectangle((wx, wy), ww, wh,
                                    facecolor=PALETTE["wall"], zorder=3))

    # Robot icon
    robot_xy = (5.0, 3.2)
    ax.add_patch(plt.Circle(robot_xy, 0.28,
                             facecolor=PALETTE["accent"], edgecolor="white",
                             linewidth=2, zorder=4))
    ax.annotate("robot", xy=robot_xy, xytext=(5.7, 3.6),
                color="white", fontsize=14, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color="white"))

    # Right: legend + explanation
    ax = axes[1]
    ax.set_xlim(0, 10); ax.set_ylim(0, 8)
    ax.set_title("What the colors mean",
                 fontsize=20, color=PALETTE["fg"], pad=14, fontweight="bold")
    legend = [
        (PALETTE["wall"],   "wall",
         "the actual obstacle — never enter"),
        (PALETTE["lethal"], "red",
         "robot will definitely collide here"),
        (PALETTE["hot"],    "orange",
         "very close — strongly discouraged"),
        (PALETTE["warm"],   "yellow",
         "getting close — avoid if possible"),
        (PALETTE["cool"],   "green",
         "open space — the robot's happy place"),
    ]
    for i, (c, label, desc) in enumerate(legend):
        y = 6.6 - i * 1.2
        ax.add_patch(plt.Rectangle((0.7, y - 0.3), 1.2, 0.7,
                                    facecolor=c, edgecolor="white",
                                    linewidth=1.5))
        ax.text(2.1, y + 0.05, label, fontsize=19, color=PALETTE["fg"],
                fontweight="bold", va="center")
        ax.text(4.3, y + 0.05, desc, fontsize=14, color=PALETTE["muted"],
                va="center")

    ax.text(5, 0.6,
            "The planner then draws the cheapest route\n"
            "that stays off red and mostly in green.",
            fontsize=14, color=PALETTE["fg"],
            ha="center", style="italic")

    fig.suptitle("The costmap — how Nav2 'sees' walls",
                  fontsize=26, color=PALETTE["fg"], fontweight="bold",
                  y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(OUT / "03_costmap_halo.png", dpi=140, bbox_inches="tight",
                facecolor=PALETTE["bg"])
    plt.close(fig); print("  wrote 03_costmap_halo.png")


# ---------------------------------------------------------------------------
# Slide 4 — the tuning tradeoff (wide vs narrow halo)
# ---------------------------------------------------------------------------
def make_tuning_tradeoff():
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5), dpi=140)
    fig.patch.set_facecolor(PALETTE["bg"])

    for ax in axes:
        ax.set_facecolor(PALETTE["bg"])
        ax.axis("off")
        ax.set_xlim(0, 12); ax.set_ylim(0, 6)

    # Helper to draw a corridor cross-section with cost halos
    def draw_corridor(ax, scaling, title, emoji, takeaway):
        ax.text(6, 5.5, title, fontsize=22, color=PALETTE["fg"],
                ha="center", fontweight="bold")
        ax.text(11.5, 5.5, emoji, fontsize=28, ha="right")

        # Two walls
        wall_y = 1.5
        wall_h = 0.4
        ax.add_patch(plt.Rectangle((0.5, wall_y), 11, wall_h,
                                    facecolor=PALETTE["wall"]))
        ax.add_patch(plt.Rectangle((0.5, 4.0), 11, wall_h,
                                    facecolor=PALETTE["wall"]))

        # Build cost field across corridor interior
        W, H = 550, 120
        cost_img = np.zeros((H, W), dtype=np.float32)
        for i in range(H):
            # normalized distance from nearest wall (0 at wall, 1 at midline)
            d = min(i, H - 1 - i) / (H - 1)
            # scale to "meters" where H=125 = 1.25 m
            d_m = d * 1.25
            inscribed = 0.15
            if d_m < inscribed:
                c = 1.0
            else:
                c = np.exp(-scaling * (d_m - inscribed))
            cost_img[i, :] = c

        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list(
            "halo",
            [(0.0, PALETTE["cool"]),
             (0.25, PALETTE["warm"]),
             (0.6, PALETTE["hot"]),
             (1.0, PALETTE["lethal"])])
        ax.imshow(cost_img, extent=[0.5, 11.5, 1.9, 4.0],
                  origin="lower", aspect="auto",
                  cmap=cmap, vmin=0, vmax=1.0)

        # Robot
        ax.add_patch(plt.Circle((6, 2.95), 0.22,
                                 facecolor=PALETTE["accent"], ec="white",
                                 linewidth=2, zorder=4))

        # Takeaway
        ax.text(6, 0.8, takeaway, fontsize=15, color=PALETTE["muted"],
                ha="center", style="italic", wrap=True)

    draw_corridor(axes[0], scaling=0.3,
                  title="Halos fade slowly",
                  emoji="× robot unhappy",
                  takeaway="Every cell in a narrow corridor looks painful.\n"
                           "Robot refuses doorways, crawls, gets stuck.")

    draw_corridor(axes[1], scaling=4.0,
                  title="Halos fade fast",
                  emoji="✓ robot happy",
                  takeaway="A low-cost 'spine' runs down the middle.\n"
                           "Robot threads doorways happily.")

    fig.suptitle("One slider decides most of the pain:  how fast the halo fades",
                  fontsize=23, color=PALETTE["fg"], fontweight="bold", y=0.98)

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(OUT / "04_tuning_tradeoff.png", dpi=140, bbox_inches="tight",
                facecolor=PALETTE["bg"])
    plt.close(fig); print("  wrote 04_tuning_tradeoff.png")


# ---------------------------------------------------------------------------
# Slide 5 — the story (TSP tour + what happened)
# ---------------------------------------------------------------------------
def make_tour_story():
    fig, ax = plt.subplots(figsize=(16, 9), dpi=140)
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])
    ax.axis("off")
    ax.set_xlim(0, 16); ax.set_ylim(0, 9)

    ax.text(8, 8.4, "What we built on top of Nav2",
             fontsize=26, color=PALETTE["fg"],
             ha="center", fontweight="bold")

    steps = [
        ("①", "Floor plan →",
         "Scan the Innovation Center USD, make an occupancy map.",
         PALETTE["accent"]),
        ("②", "Nav graph →",
         "Skeletonize walkable space; doors become labeled nodes.",
         PALETTE["accent2"]),
        ("③", "TSP route →",
         "Optimal door visit order via Dijkstra distances + 2-opt.",
         PALETTE["warm"]),
        ("④", "Nav2 tour →",
         "Hand Nav2 the list of poses, let it drive door-to-door.",
         PALETTE["cool"]),
        ("⑤", "Record →",
         "Three cameras roll: chase, robot POV, top-down map.",
         PALETTE["path"]),
    ]
    y0 = 6.6
    for i, (num, label, desc, color) in enumerate(steps):
        y = y0 - i * 1.05
        ax.text(0.6, y, num, fontsize=36, color=color,
                ha="center", va="center", fontweight="bold")
        ax.text(1.5, y + 0.1, label, fontsize=20, color=PALETTE["fg"],
                va="center", fontweight="bold")
        ax.text(5.2, y + 0.1, desc, fontsize=15, color=PALETTE["muted"],
                va="center")

    # Bottom banner
    ax.add_patch(FancyBboxPatch((0.5, 0.4), 15, 0.9,
                                 boxstyle="round,pad=0.03,rounding_size=0.18",
                                 linewidth=1.5, edgecolor=PALETTE["accent"],
                                 facecolor="#1e293b"))
    ax.text(8, 0.85,
            "Nav2 is the 'driving' layer — we don't build it, we configure it. "
            "Research is what goes ON TOP.",
            fontsize=16, color=PALETTE["fg"], ha="center", va="center",
            style="italic")

    fig.savefig(OUT / "05_tour_story.png", dpi=140, bbox_inches="tight",
                facecolor=PALETTE["bg"])
    plt.close(fig); print("  wrote 05_tour_story.png")


# ---------------------------------------------------------------------------
# Slide 6 — take-home summary
# ---------------------------------------------------------------------------
def make_summary():
    fig, ax = plt.subplots(figsize=(14, 8), dpi=140)
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])
    ax.axis("off")
    ax.set_xlim(0, 14); ax.set_ylim(0, 8)

    ax.text(7, 7.2, "One-line summary", fontsize=28, color=PALETTE["fg"],
            ha="center", fontweight="bold")

    ax.text(7, 5.9,
            "Nav2 =  Localization  +  Costmap  +  Planner  +  Controller",
            fontsize=24, color=PALETTE["accent"], ha="center",
            fontweight="bold")

    lines = [
        ("Localization", "Where am I?", PALETTE["accent"]),
        ("Costmap",      "What should I avoid?", PALETTE["warm"]),
        ("Planner",      "What's the route?", PALETTE["cool"]),
        ("Controller",   "How do I drive?", PALETTE["accent2"]),
    ]
    y = 4.7
    for label, question, color in lines:
        ax.text(4.5, y, label, fontsize=20, color=color,
                ha="right", fontweight="bold")
        ax.text(5.2, y, "→", fontsize=20, color=PALETTE["muted"])
        ax.text(5.6, y, f'"{question}"', fontsize=20, color=PALETTE["fg"],
                style="italic")
        y -= 0.55

    # Bottom: the research angle
    ax.add_patch(FancyBboxPatch((1, 0.7), 12, 1.4,
                                 boxstyle="round,pad=0.03,rounding_size=0.18",
                                 linewidth=2, edgecolor=PALETTE["accent"],
                                 facecolor="#1e293b"))
    ax.text(7, 1.7, "The payoff",
            fontsize=17, color=PALETTE["accent"], ha="center",
            fontweight="bold")
    ax.text(7, 1.15,
            "Standardized driving stack   →   we focus on WHERE to go, "
            "WHAT to record, WHAT to detect.",
            fontsize=15, color=PALETTE["fg"], ha="center")

    fig.savefig(OUT / "06_summary.png", dpi=140, bbox_inches="tight",
                facecolor=PALETTE["bg"])
    plt.close(fig); print("  wrote 06_summary.png")


# ---------------------------------------------------------------------------
# Animation — raytracing occupancy generation
# ---------------------------------------------------------------------------
def _cast_rays(occ, origin_px, angles_deg, max_px, mask=None):
    """Vectorized 2D raycast against an occupancy grid.

    occ: HxW float32 array, 1 = wall, 0 = free, 2 = unknown.
    origin_px: (x, y) in pixel coords (x = col, y = row).
    mask: optional boolean building-footprint mask — a ray is also considered
          blocked when it steps outside the mask, so rays never escape the
          building.
    Returns (hit_x, hit_y) arrays, one per angle.
    """
    H, W = occ.shape
    ox, oy = origin_px
    steps = np.arange(0, max_px, 1.0)
    hits_x = np.empty(len(angles_deg), dtype=np.float32)
    hits_y = np.empty(len(angles_deg), dtype=np.float32)
    for i, ang in enumerate(angles_deg):
        dx = np.cos(np.radians(ang)); dy = np.sin(np.radians(ang))
        xs = ox + steps * dx
        ys = oy + steps * dy
        oob = (xs < 0) | (xs >= W) | (ys < 0) | (ys >= H)
        ix = np.clip(xs.astype(np.int32), 0, W - 1)
        iy = np.clip(ys.astype(np.int32), 0, H - 1)
        blocked = (occ[iy, ix] == 1) | oob
        if mask is not None:
            blocked = blocked | ~mask[iy, ix]
        if blocked.any():
            k = int(np.argmax(blocked))
        else:
            k = len(steps) - 1
        hits_x[i] = xs[k]
        hits_y[i] = ys[k]
    return hits_x, hits_y


def _building_mask(occ):
    """Boolean mask of the actual building footprint.

    The occupancy map has a few stray wall pixels (scale bars, legend marks)
    that would inflate a naive bounding box. To find just the real building
    we dilate the walls, label connected components, keep the largest one,
    and fill its interior. Free pixels inside that region are the building's
    walkable area.
    """
    from scipy.ndimage import binary_dilation, binary_fill_holes, label
    walls = (occ == 1)
    dilated = binary_dilation(walls, iterations=12)  # ~0.6 m bridge
    lbl, n = label(dilated)
    if n == 0:
        return np.zeros_like(walls, dtype=bool)
    sizes = np.bincount(lbl.ravel())
    sizes[0] = 0  # background
    main = (lbl == sizes.argmax())
    return binary_fill_holes(main)


def _building_bbox_from_mask(mask, pad=40):
    ys, xs = np.where(mask)
    H, W = mask.shape
    return (max(0, xs.min() - pad), max(0, ys.min() - pad),
            min(W, xs.max() + pad), min(H, ys.max() + pad))


def _pick_curated_origins(occ, mask, bbox, target_total=36,
                           wall_clearance_px=8,
                           min_room_area_px=300):
    """Place origins so rays cover every accessible room + corridor.

    Strategy: drop a dense uniform grid over the building bbox, keep every
    point that is inside the mask and ≥`wall_clearance_px` from any wall,
    PLUS for every small free-space component the distance-transform peak
    (so isolated closets/rooms still get at least one origin). Subsample
    the pool to `target_total` via evenly-spaced picks along the serpentine
    order so motion across the floor stays readable.
    """
    from scipy.ndimage import distance_transform_edt, label
    x0, y0, x1, y1 = bbox
    dist = distance_transform_edt(occ == 0)

    # Dense grid: ~1 m spacing (20 px at cell_size 0.05 m).
    step = 28
    gy, gx = np.mgrid[y0 + 20:y1 - 20:step, x0 + 20:x1 - 20:step]
    grid_pts = []
    for y, x in zip(gy.ravel(), gx.ravel()):
        iy, ix = int(y), int(x)
        if mask[iy, ix] and dist[iy, ix] > wall_clearance_px:
            grid_pts.append((ix, iy))

    # Ensure every room has at least one origin, even if the grid missed it.
    free = (occ == 0) & mask
    lbl, n_rooms = label(free)
    sizes = np.bincount(lbl.ravel()); sizes[0] = 0
    existing = set(grid_pts)
    for rid in range(1, n_rooms + 1):
        if sizes[rid] < min_room_area_px:
            continue
        rd = dist.copy(); rd[lbl != rid] = 0
        # Only add a peak if no grid point already landed in this room.
        grid_in_room = any(lbl[y, x] == rid for (x, y) in existing)
        if grid_in_room:
            continue
        idx = int(np.argmax(rd))
        py, px = np.unravel_index(idx, rd.shape)
        if rd[py, px] > wall_clearance_px:
            grid_pts.append((int(px), int(py)))

    # Serpentine sort: band by y, alternate x per band so the visit order
    # looks like a walk across the floor.
    band_h = max(80, (y1 - y0) // 6)
    banded = {}
    for x, y in grid_pts:
        banded.setdefault((y - y0) // band_h, []).append((x, y))
    ordered = []
    for i, band in enumerate(sorted(banded)):
        row = banded[band]
        row.sort(key=lambda p: p[0], reverse=(i % 2 == 1))
        ordered.extend(row)

    # Subsample evenly to target_total (preserves serpentine ordering).
    if len(ordered) > target_total:
        idx = np.linspace(0, len(ordered) - 1, target_total).round().astype(int)
        ordered = [ordered[i] for i in idx]
    return ordered


def animate_raytracing_occupancy(fps=30, n_rays=96, max_ray_m=25.0,
                                  cell_size_m=0.05,
                                  phase1_frames=45,
                                  per_origin_frames=4,
                                  phase3_frames=75,
                                  n_origins=36,
                                  out_name="raytracing_occupancy.mp4"):
    """Animate how raytracing builds up an occupancy map.

    Phase 1: one origin; rays sweep out one-by-one (radar style).
    Phase 2: origin hops across the floor; each stop does a full fan at once
             and reveals walls/free pixels into the 'discovered' layer.
    Phase 3: hold on the final discovered map with caption.
    """
    occ = np.load(FLOOR4_NPY)  # HxW, 0=free, 1=wall, 2=unknown
    H, W = occ.shape
    max_px = int(max_ray_m / cell_size_m)

    mask = _building_mask(occ)
    bbox = _building_bbox_from_mask(mask, pad=40)
    origins = _pick_curated_origins(occ, mask, bbox, target_total=n_origins)
    if not origins:
        raise RuntimeError("no valid origins inside building footprint")
    print(f"  placed {len(origins)} origins across rooms")
    star = origins[len(origins) // 2]
    angles = np.linspace(0, 360, n_rays, endpoint=False)

    # Pre-compute ray hits. Rays stop at walls AND at the building boundary.
    all_hits = [_cast_rays(occ, o, angles, max_px, mask=mask) for o in origins]
    star_hits = _cast_rays(occ, star, angles, max_px, mask=mask)

    # Discovered layer: 0=unseen(dark), 1=known-free(light gray), 2=known-wall(black).
    discovered = np.zeros((H, W), dtype=np.uint8)

    from matplotlib.path import Path as MplPath

    def bake_rays_into_discovered(origin, hits_x, hits_y):
        """Fill the fan polygon visible from `origin` with known-free, and
        stamp actual wall pixels at ray endpoints."""
        ox, oy = origin
        # Polygon = ray endpoints in angle order. With 72 angles this forms
        # a star-shaped polygon covering every pixel a ray passed through.
        poly = np.column_stack([hits_x, hits_y])
        poly_path = MplPath(np.vstack([poly, poly[:1]]), closed=True)
        bx0 = max(0, int(min(ox, hits_x.min())) - 1)
        bx1 = min(W, int(max(ox, hits_x.max())) + 2)
        by0 = max(0, int(min(oy, hits_y.min())) - 1)
        by1 = min(H, int(max(oy, hits_y.max())) + 2)
        yy, xx = np.mgrid[by0:by1, bx0:bx1]
        pts = np.column_stack([xx.ravel(), yy.ravel()])
        inside = poly_path.contains_points(pts).reshape(yy.shape)
        region = discovered[by0:by1, bx0:bx1]
        region[inside & (region == 0)] = 1
        # Wall stamp at ray endpoints (thicken 2-px for visibility).
        for hx, hy in zip(hits_x.astype(int), hits_y.astype(int)):
            if not (0 <= hx < W and 0 <= hy < H):
                continue
            if occ[hy, hx] != 1:
                continue
            yy0, yy1 = max(0, hy - 1), min(H, hy + 2)
            xx0, xx1 = max(0, hx - 1), min(W, hx + 2)
            sub = discovered[yy0:yy1, xx0:xx1]
            wall = (occ[yy0:yy1, xx0:xx1] == 1)
            sub[wall] = 2

    # --- Figure setup ------------------------------------------------------
    fig, ax = plt.subplots(figsize=(16, 9), dpi=120)
    fig.patch.set_facecolor("#000000")
    ax.set_facecolor("#000000")
    x0, y0, x1, y1 = bbox
    ax.set_xlim(x0, x1); ax.set_ylim(y1, y0)  # image coords: y down
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)

    # Discovered-layer image: 0=unseen(black), 1=known-free(white), 2=known-wall(black).
    # Walls sit inside white regions so they read as black gaps = the map.
    from matplotlib.colors import ListedColormap
    disc_cmap = ListedColormap(["#000000", "#ffffff", "#000000"])
    disc_img = ax.imshow(discovered, cmap=disc_cmap, vmin=0, vmax=2,
                         interpolation="nearest", zorder=1)

    # Rays (LineCollection, reset each frame).
    rays = LineCollection([], colors="#ef4444", linewidths=1.1,
                          alpha=0.9, zorder=3)
    ax.add_collection(rays)

    # Hit dots (scatter), drawn where each ray terminates.
    hit_scatter = ax.scatter([], [], s=8, c="#fbbf24",
                              edgecolors="none", zorder=4)

    # Origin dot.
    origin_dot = ax.scatter([], [], s=180, c=PALETTE["accent"],
                             edgecolors="white", linewidths=2, zorder=5)

    cx = (x0 + x1) / 2
    title = ax.text(cx, y0 - 18,
                    "Building the occupancy map, one ray at a time",
                    color="#f8fafc", ha="center", va="bottom",
                    fontsize=22, fontweight="bold")
    subtitle = ax.text(cx, y1 + 18,
                       "", color="#94a3b8", ha="center", va="top",
                       fontsize=16, style="italic")

    # --- Frame plan --------------------------------------------------------
    total_phase2 = per_origin_frames * len(origins)
    total_frames = phase1_frames + total_phase2 + phase3_frames
    print(f"  total frames: {total_frames} ({total_frames/fps:.1f}s @ {fps}fps)")

    def update(frame):
        if frame < phase1_frames:
            # Phase 1 — sweep rays out from the star origin.
            ox, oy = star
            origin_dot.set_offsets([[ox, oy]])
            k = int(np.ceil(n_rays * (frame + 1) / phase1_frames))
            segs = [((ox, oy), (star_hits[0][i], star_hits[1][i]))
                    for i in range(k)]
            rays.set_segments(segs)
            hit_scatter.set_offsets(np.column_stack(
                [star_hits[0][:k], star_hits[1][:k]]))
            subtitle.set_text(
                f"One sample point · casting {k}/{n_rays} rays — each stops at the first wall")
            # On the last frame of phase 1, bake the star's rays into discovered.
            if frame == phase1_frames - 1:
                bake_rays_into_discovered(star, *star_hits)
                disc_img.set_data(discovered)
            return (disc_img, rays, hit_scatter, origin_dot, subtitle)

        elif frame < phase1_frames + total_phase2:
            # Phase 2 — hop through origins, each origin gets per_origin_frames.
            idx = (frame - phase1_frames) // per_origin_frames
            sub = (frame - phase1_frames) % per_origin_frames
            idx = min(idx, len(origins) - 1)
            ox, oy = origins[idx]
            hx, hy = all_hits[idx]
            origin_dot.set_offsets([[ox, oy]])
            # Gradually reveal the full fan within per_origin_frames.
            k = int(np.ceil(n_rays * (sub + 1) / per_origin_frames))
            segs = [((ox, oy), (hx[i], hy[i])) for i in range(k)]
            rays.set_segments(segs)
            hit_scatter.set_offsets(np.column_stack([hx[:k], hy[:k]]))
            # Bake this origin's rays into discovered at the end of its slot.
            if sub == per_origin_frames - 1:
                bake_rays_into_discovered(origins[idx], hx, hy)
                disc_img.set_data(discovered)
            subtitle.set_text(
                f"Sample point {idx+1}/{len(origins)} · rays reveal walls into the occupancy map")
            return (disc_img, rays, hit_scatter, origin_dot, subtitle)

        else:
            # Phase 3 — hold on the map that the rays actually built.
            rays.set_segments([])
            hit_scatter.set_offsets(np.zeros((0, 2)))
            origin_dot.set_offsets(np.zeros((0, 2)))
            subtitle.set_text(
                f"Occupancy map generated · {len(origins)} sample points × "
                f"{n_rays} rays each  →  white = free, black = walls")
            return (disc_img, rays, hit_scatter, origin_dot, subtitle)

    anim = FuncAnimation(fig, update, frames=total_frames,
                          interval=1000 / fps, blit=False)
    out_path = IMAGES_OUT / out_name
    writer = FFMpegWriter(fps=fps, bitrate=5000,
                           codec="libx264",
                           extra_args=["-pix_fmt", "yuv420p"])
    anim.save(str(out_path), writer=writer, dpi=120,
              savefig_kwargs={"facecolor": "#000000"})
    plt.close(fig)
    print(f"  wrote {out_path.relative_to(PROJECT_ROOT)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--slides", action="store_true",
                        help="only (re)generate the static slide PNGs")
    parser.add_argument("--video", action="store_true",
                        help="only generate the raytracing animation")
    args = parser.parse_args()

    run_slides = args.slides or not args.video
    run_video = args.video or not args.slides

    print(f"Writing to: {OUT}")
    if run_slides:
        make_overview()
        make_four_roles()
        make_costmap_halo()
        make_tuning_tradeoff()
        make_tour_story()
        make_summary()
    if run_video:
        print(f"Writing video to: {IMAGES_OUT}")
        animate_raytracing_occupancy()
    print("Done.")

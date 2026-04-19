#!/usr/bin/env python3
"""
map_doors_to_nodes.py

For each door on a given floor:
  1. Convert USD door center to occupancy-map pixel coordinates (with X-flip).
  2. Use the door's known width from the USD bounding box to search for the
     matching wall gap in the occupancy map — the exact pixel-level threshold.
  3. From that threshold, find the closest SWAGGER nav-graph node with LOS.

Wall-gap detection:
  - Scans horizontal and vertical lines within a search radius of the USD position
  - A "gap" = a free-space run flanked by wall pixels on both ends
  - Only accepts gaps whose width in pixels matches the door's known width (±2 px)
  - Falls back to nearest free-space pixel if no matching gap is found

Usage:
    python3 "Automating Node Placement/map_doors_to_nodes.py"            # all floors
    python3 "Automating Node Placement/map_doors_to_nodes.py" floor_4    # single floor

Outputs per floor  (Automating Node Placement/Swagger/floor_X/):
    floor_X_door_node_map.json   — production mapping
    floor_X_door_node_map.png    — debug overlay (requires PIL)
"""

import json
import math
import os
import sys

import numpy as np
from scipy.ndimage import distance_transform_edt

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT     = os.path.join(SCRIPT_DIR, "..")
NAVGRAPH_DIR  = os.path.join(REPO_ROOT, "assets", "environments", "InnovationCenter", "maps", "SWAGGER")
MAPS_DIR      = os.path.join(REPO_ROOT, "assets", "environments", "InnovationCenter", "maps")
DOORS_DIR     = os.path.join(REPO_ROOT, "assets", "environments", "InnovationCenter", "maps", "doors")
BUILDING_JSON = os.path.join(MAPS_DIR, "building_data.json")

ALL_FLOORS = ["floor_1", "floor_2", "floor_3", "floor_4"]

FREE_THRESHOLD      = 127   # pixels above this are free (0=wall, 128=unknown, 255=free)
GAP_TOLERANCE       = 4     # ±pixels when matching door width to gap width
SEARCH_RADIUS_M     = 2.0   # metres radius to search for wall gap around USD position
MAX_MATCH_DISTANCE_M = 10.0  # doors further than this from any node are left unmatched

# Door types that are architectural openings, not walkable doors — excluded entirely
OVERSIZE_DOOR_TYPES = {
    "AV300_RAYNOR_GARAGE_DOOR",
    "Dormakaba_Sliding_Glass_Wall_FSW",
    "RS___Curtain_Wall___Type_2FG___NEW",
    "RS___Curtain_Wall___Type_FG___New",
    "RS___Curtain_Wall___Type_F___NEW",
    "RS___Overhead_Rolling_Door",
    "RS___Overhead_Sectional_Door",
}

SIDELITE_DOOR_TYPE = "RS___Type_F_W_SIDELITE_Door"


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_navgraph(floor):
    path = os.path.join(NAVGRAPH_DIR, floor, f"{floor}_auto_navgraph.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Nav graph not found: {path}")
    with open(path) as f:
        return json.load(f)


def load_floor_meta(floor):
    path = os.path.join(MAPS_DIR, f"{floor}_meta.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Floor meta not found: {path}")
    with open(path) as f:
        return json.load(f)


def load_occupancy(floor):
    """Load occupancy PNG as uint8 (H×W). 255=free, 0=wall, 128=unknown."""
    from PIL import Image
    png = os.path.join(MAPS_DIR, f"{floor}_occupancy.png")
    if not os.path.exists(png):
        raise FileNotFoundError(f"Occupancy PNG not found: {png}")
    return np.array(Image.open(png).convert("L"))


def load_building_data():
    if not os.path.exists(BUILDING_JSON):
        raise FileNotFoundError(f"building_data.json not found: {BUILDING_JSON}")
    with open(BUILDING_JSON) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Coordinate helpers  (InnovationCenter maps use ROS map_server convention)
# ---------------------------------------------------------------------------
#
# InnovationCenter occupancy maps are saved with np.fliplr + np.flipud applied
# so they follow ROS map_server convention:
#   col 0 = xy_min[0] (left / -X edge),  col increases → x increases
#   row 0 = xy_max[1] (top  / +Y edge),  row increases → y decreases
#
# cell_size_cm field stores metres/pixel (legacy field name).
#
# SWAGGER auto_graph_generator.py uses the same convention, so navgraph
# world_xy_m values are already correct USD world coordinates — no X
# correction needed.

def world_m_to_px(x_m, y_m, meta):
    res = meta["cell_size_cm"]                    # m/px
    col = (x_m - meta["xy_min"][0]) / res
    row = (meta["xy_max"][1] - y_m) / res
    return row, col


def px_to_world_m(row, col, meta):
    res = meta["cell_size_cm"]                    # m/px
    x_m = meta["xy_min"][0] + col * res
    y_m = meta["xy_max"][1] - row * res
    return x_m, y_m


def dist_m(ax, ay, bx, by):
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)


# ---------------------------------------------------------------------------
# Door type helpers
# ---------------------------------------------------------------------------

def get_door_type(door):
    """Extract the door type folder name from the prim path."""
    parts = door["prim_path"].split("/")
    doors_idx = next((i for i, p in enumerate(parts) if p == "Doors"), None)
    if doors_idx is None or len(parts) <= doors_idx + 1:
        return "unknown"
    return parts[doors_idx + 1]


def is_oversize_door(door):
    return get_door_type(door) in OVERSIZE_DOOR_TYPES


def is_sidelite_door(door):
    return get_door_type(door) == SIDELITE_DOOR_TYPE


def sidelite_leaf_width_cm(door):
    """
    Parse the actual door leaf width from the frame subfolder name.
    Frame names look like: NEW___HM_Frame___36_x96____18__Sidelite
    The first number before _x is the door leaf width in inches.
    Returns width in cm, or None if parsing fails.
    """
    import re
    parts = door["prim_path"].split("/")
    doors_idx = next((i for i, p in enumerate(parts) if p == "Doors"), None)
    if doors_idx is None or len(parts) <= doors_idx + 2:
        return None
    frame = parts[doors_idx + 2]
    m = re.search(r'(\d+)_x\d+', frame)
    if m:
        inches = int(m.group(1))
        return round(inches * 2.54, 1)
    return None


# ---------------------------------------------------------------------------
# Door width extraction
# ---------------------------------------------------------------------------

def door_width_px(door, meta):
    """
    Return the expected gap width in pixels for this door.

    For sidelite doors: parse the leaf width in inches from the frame name
    (e.g. 36_x96 → 36 inches → 91.44 cm) rather than using max(sx,sy)
    which includes the fixed glass panel.

    For standard doors:
      - sz is always the door height (Z axis)
      - The smaller of sx/sy is the door panel thickness (~4-10 cm)
      - The larger of sx/sy is the door WIDTH — the wall-opening gap

    At 5 cm/px resolution a 91 cm door = ~18 pixels.
    """
    if is_sidelite_door(door):
        width_cm = sidelite_leaf_width_cm(door)
        if width_cm is None:
            # Fallback: use the smaller of sx/sy that is > 50cm (the door leaf)
            sx, sy = door["size_cm"][0], door["size_cm"][1]
            candidates = [v for v in (sx, sy) if v > 50.0]
            width_cm = min(candidates) if candidates else max(sx, sy)
    else:
        sx, sy = door["size_cm"][0], door["size_cm"][1]
        width_cm = max(sx, sy)

    res_cm = meta["cell_size_cm"] * 100.0   # m/px → cm/px
    return max(1, round(width_cm / res_cm))


# ---------------------------------------------------------------------------
# Wall-gap detection
# ---------------------------------------------------------------------------

def _free_runs_flanked(line, min_px, max_px):
    """
    Find runs of free pixels (> FREE_THRESHOLD) in a 1-D array that are
    flanked by wall pixels (or the array edge) on both ends.
    Returns list of (start_idx, end_idx, width).
    """
    runs = []
    n = len(line)
    i = 0
    while i < n:
        if line[i] > FREE_THRESHOLD:
            start = i
            while i < n and line[i] > FREE_THRESHOLD:
                i += 1
            end = i - 1
            width = end - start + 1
            left_wall  = (start == 0) or (line[start - 1] <= FREE_THRESHOLD)
            right_wall = (end == n - 1) or (line[end + 1] <= FREE_THRESHOLD)
            if left_wall and right_wall and min_px <= width <= max_px:
                runs.append((start, end, width))
        else:
            i += 1
    return runs


def find_door_gap(center_row, center_col, gap_px, occ, search_radius_px):
    """
    Search for a wall gap of width ≈ gap_px near (center_row, center_col).

    Scans every row and every column within search_radius_px of the centre.
    A valid gap is a free-space run flanked by wall pixels whose width is
    within GAP_TOLERANCE pixels of gap_px.

    Returns (gap_center_row, gap_center_col, gap_method) or None.
    gap_method is 'H' (horizontal gap in a vertical wall) or 'V'.
    """
    h, w   = occ.shape
    min_px = max(1, gap_px - GAP_TOLERANCE)
    max_px = gap_px + GAP_TOLERANCE

    best_rc   = None
    best_dist = float("inf")
    best_meth = None

    r_lo = max(0, center_row - search_radius_px)
    r_hi = min(h - 1, center_row + search_radius_px)
    c_lo = max(0, center_col - search_radius_px)
    c_hi = min(w - 1, center_col + search_radius_px)

    # Horizontal scan — gap in a vertical wall (door opens left/right)
    for r in range(r_lo, r_hi + 1):
        line = occ[r, c_lo:c_hi + 1]
        for (s, e, _) in _free_runs_flanked(line, min_px, max_px):
            gc = c_lo + (s + e) // 2
            d  = math.hypot(r - center_row, gc - center_col)
            if d < best_dist:
                best_dist = d
                best_rc   = (r, gc)
                best_meth = "H"

    # Vertical scan — gap in a horizontal wall (door opens up/down)
    for c in range(c_lo, c_hi + 1):
        line = occ[r_lo:r_hi + 1, c]
        for (s, e, _) in _free_runs_flanked(line, min_px, max_px):
            gr = r_lo + (s + e) // 2
            d  = math.hypot(gr - center_row, c - center_col)
            if d < best_dist:
                best_dist = d
                best_rc   = (gr, c)
                best_meth = "V"

    return best_rc, best_meth


# ---------------------------------------------------------------------------
# Free-space fallback snap
# ---------------------------------------------------------------------------

def build_snap_index(occ):
    free_mask = occ > FREE_THRESHOLD
    _, idx    = distance_transform_edt(~free_mask, return_indices=True)
    return idx[0], idx[1]


def snap_to_free(row_f, col_f, occ, snap_rows, snap_cols):
    h, w = occ.shape
    r = int(round(max(0, min(h - 1, row_f))))
    c = int(round(max(0, min(w - 1, col_f))))
    if occ[r, c] > FREE_THRESHOLD:
        return r, c
    return int(snap_rows[r, c]), int(snap_cols[r, c])


# ---------------------------------------------------------------------------
# Line-of-sight (Bresenham)
# ---------------------------------------------------------------------------

def has_line_of_sight(r0, c0, r1, c1, occ):
    r0, c0, r1, c1 = int(r0), int(c0), int(r1), int(c1)
    h, w = occ.shape
    dr = abs(r1 - r0); dc = abs(c1 - c0)
    sr = 1 if r1 > r0 else -1
    sc = 1 if c1 > c0 else -1
    err = dr - dc
    r, c = r0, c0
    while True:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if occ[r, c] <= FREE_THRESHOLD:
            return False
        if r == r1 and c == c1:
            return True
        e2 = 2 * err
        if e2 > -dc:
            err -= dc; r += sr
        if e2 < dr:
            err += dr; c += sc


# ---------------------------------------------------------------------------
# Node matching
# ---------------------------------------------------------------------------

def find_closest_node(x_m, y_m, nodes, occ, meta):
    """
    Nearest nav node with LOS from (x_m, y_m).  Falls back to Euclidean.
    Corrects navgraph stored X (image-space) to USD world X before comparing.
    """
    thresh_r, thresh_c = world_m_to_px(x_m, y_m, meta)
    best_id, best_dist = None, float("inf")
    fb_id,   fb_dist   = None, float("inf")

    for node in nodes:
        nx = node["world_xy_m"][0]
        ny = node["world_xy_m"][1]
        d  = dist_m(x_m, y_m, nx, ny)

        if d < fb_dist:
            fb_dist = d; fb_id = node["id"]

        if d >= best_dist:
            continue

        nr, nc = world_m_to_px(nx, ny, meta)
        if has_line_of_sight(thresh_r, thresh_c, nr, nc, occ):
            best_dist = d; best_id = node["id"]

    if best_id is not None:
        return best_id, best_dist, True
    return fb_id, fb_dist, False


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def draw_overlay(floor, navgraph, door_map, meta, out_png):
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        print("  [skip PNG] PIL not available"); return

    base_png = os.path.join(NAVGRAPH_DIR, floor, f"{floor}_auto_navgraph.png")
    if not os.path.exists(base_png):
        print(f"  [skip PNG] base not found: {base_png}"); return

    img  = Image.open(base_png).convert("RGB")
    draw = ImageDraw.Draw(img)
    res  = meta["cell_size_cm"]

    def w2px(x_m, y_m):
        col = int((x_m - meta["xy_min"][0]) / res)
        row = int((meta["xy_max"][1] - y_m) / res)
        return col, row

    node_pos = {n["id"]: (n["world_xy_m"][0], n["world_xy_m"][1])
                for n in navgraph["nodes"]}

    iw, ih = img.size

    for e in door_map:
        # Skip OOB doors and unmatched doors (too far from any node)
        if e.get("is_oob") or e["closest_node_id"] is None:
            continue

        tx, ty = e["threshold_world_xy_m"]
        tpx    = w2px(tx, ty)

        # Skip doors whose threshold falls outside the image dimensions
        if not (0 <= tpx[0] < iw and 0 <= tpx[1] < ih):
            continue

        # Bounding box — sized from door width (px), drawn at threshold position
        half = max(3, e.get("gap_width_px", 10) // 2)
        box_col = (0, 210, 0) if e.get("los_match") else (255, 140, 0)
        draw.rectangle([tpx[0]-half, tpx[1]-half, tpx[0]+half, tpx[1]+half],
                       outline=box_col, width=2)

        # Line threshold → matched node
        nid    = e["closest_node_id"]
        nx, ny = node_pos[nid]
        npx    = w2px(nx, ny)
        draw.line([tpx, npx], fill=box_col, width=1)

    img.save(out_png)
    print(f"  Overlay PNG : {out_png}")


# ---------------------------------------------------------------------------
# Per-floor processing
# ---------------------------------------------------------------------------

def process_floor(floor, building_data):
    print(f"\n{'='*50}\nFloor: {floor}\n{'='*50}")

    navgraph = load_navgraph(floor)
    nodes    = navgraph["nodes"]
    doors    = building_data.get("floors", {}).get(floor, {}).get("doors", [])
    print(f"  Nav nodes      : {len(nodes)}")
    print(f"  Doors on floor : {len(doors)}")
    if not nodes or not doors:
        print("  Skipping."); return

    meta = load_floor_meta(floor)
    occ  = load_occupancy(floor)
    h, w = occ.shape
    res  = meta["cell_size_cm"]

    search_radius_px = int(round(SEARCH_RADIUS_M / res))
    print(f"  Occupancy map  : {w}×{h} px  |  search radius: {search_radius_px} px")
    print("  Building fallback snap index...")
    snap_rows, snap_cols = build_snap_index(occ)

    # Filter out oversize/architectural openings — not walkable doors
    walkable_doors = [d for d in doors if not is_oversize_door(d)]
    skipped_oversize = len(doors) - len(walkable_doors)
    print(f"  Oversize skipped: {skipped_oversize}  |  Walkable doors: {len(walkable_doors)}")

    door_map  = []
    gap_hits  = 0
    oob_count = 0

    for door in walkable_doors:
        dx_m = door["center_cm"][0] / 100.0
        dy_m = door["center_cm"][1] / 100.0
        row_f, col_f = world_m_to_px(dx_m, dy_m, meta)
        r0 = int(round(row_f)); c0 = int(round(col_f))

        # Out-of-bounds doors
        if not (0 <= r0 < h and 0 <= c0 < w):
            oob_count += 1
            node_id, dist, _ = find_closest_node(dx_m, dy_m, nodes, occ, meta)
            door_map.append({
                "door_name"            : door["name"],
                "door_prim_path"       : door["prim_path"],
                "door_type"            : get_door_type(door),
                "door_world_xy_m"      : [round(dx_m, 4), round(dy_m, 4)],
                "threshold_world_xy_m" : [round(dx_m, 4), round(dy_m, 4)],
                "gap_detected"         : False,
                "is_oob"               : True,
                "los_match"            : False,
                "closest_node_id"      : node_id,
                "distance_m"           : round(dist, 4),
            })
            continue

        # Expected gap width in pixels from door's known width
        gap_px        = door_width_px(door, meta)
        door_w_cm     = (sidelite_leaf_width_cm(door) if is_sidelite_door(door)
                         else max(door["size_cm"][0], door["size_cm"][1]))
        door_h_cm     = door["size_cm"][2]

        # Step 1 — find wall gap matching the door width
        gap_rc, gap_meth = find_door_gap(r0, c0, gap_px, occ, search_radius_px)

        if gap_rc is not None:
            snap_r, snap_c = gap_rc
            gap_detected   = True
            gap_hits      += 1
        else:
            # Fallback: snap to nearest free pixel
            snap_r, snap_c = snap_to_free(row_f, col_f, occ, snap_rows, snap_cols)
            gap_detected   = False

        tx_m, ty_m = px_to_world_m(snap_r, snap_c, meta)

        # Step 2 — find closest LOS node from threshold
        node_id, dist, los = find_closest_node(tx_m, ty_m, nodes, occ, meta)

        # If the nearest node is beyond the max distance, leave this door unmatched
        if dist > MAX_MATCH_DISTANCE_M:
            node_id = None
            los     = False

        door_map.append({
            "door_name"            : door["name"],
            "door_prim_path"       : door["prim_path"],
            "door_type"            : get_door_type(door),
            "door_world_xy_m"      : [round(dx_m, 4), round(dy_m, 4)],
            "threshold_world_xy_m" : [round(tx_m, 4), round(ty_m, 4)],
            "gap_detected"         : gap_detected,
            "gap_width_px"         : gap_px,
            "door_width_cm"        : round(door_w_cm, 1),
            "door_height_cm"       : round(door_h_cm, 1),
            "los_match"            : los,
            "closest_node_id"      : node_id,
            "distance_m"           : round(dist, 4),
        })

    matched      = [e for e in door_map if e["closest_node_id"] is not None]
    unmatched    = [e for e in door_map if e["closest_node_id"] is None]
    avg_dist     = sum(e["distance_m"] for e in matched) / len(matched) if matched else 0.0
    max_dist     = max(e["distance_m"] for e in matched) if matched else 0.0
    unique_nodes = len({e["closest_node_id"] for e in matched})
    los_count    = sum(1 for e in matched if e.get("los_match"))

    print(f"  Mapped         : {len(door_map)} doors")
    print(f"  Gap detected   : {gap_hits}  |  Fallback snap: {len(door_map) - gap_hits - oob_count}  |  OOB: {oob_count}")
    print(f"  Matched (≤{MAX_MATCH_DISTANCE_M}m) : {len(matched)}  |  Unmatched (too far): {len(unmatched)}")
    print(f"  LOS matches    : {los_count}  |  Euclidean fallback: {len(matched) - los_count}")
    print(f"  Unique nodes   : {unique_nodes}")
    print(f"  Avg dist→node  : {avg_dist:.2f} m")
    print(f"  Max dist→node  : {max_dist:.2f} m")

    out_dir = os.path.join(DOORS_DIR, floor)
    os.makedirs(out_dir, exist_ok=True)

    out_json = os.path.join(out_dir, f"{floor}_door_node_map.json")
    with open(out_json, "w") as f:
        json.dump({
            "floor"                : floor,
            "total_doors"          : len(door_map),
            "gap_detected_count"   : gap_hits,
            "unique_nodes_matched" : unique_nodes,
            "avg_distance_m"       : round(avg_dist, 4),
            "max_distance_m"       : round(max_dist, 4),
            "mappings"             : door_map,
        }, f, indent=2)
    print(f"  JSON saved     : {out_json}")

    out_png = os.path.join(out_dir, f"{floor}_door_node_map.png")
    draw_overlay(floor, navgraph, door_map, meta, out_png)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    floors = sys.argv[1:] if len(sys.argv) > 1 else ALL_FLOORS
    print("Loading building_data.json...")
    building_data = load_building_data()
    for floor in floors:
        try:
            process_floor(floor, building_data)
        except FileNotFoundError as e:
            print(f"\n  SKIP {floor}: {e}")
    print("\nDone.")


if __name__ == "__main__":
    main()

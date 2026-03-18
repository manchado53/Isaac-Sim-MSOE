
# Navigation Pipeline

## Overview

This pipeline converts the 3D MSOE Diercks Hall building model (USD from Revit) into 2D navigation graphs for robot pathfinding. It has two stages:

1. **Occupancy Map Generation** — runs inside Isaac Sim on Rosie (GPU required)
2. **Navigation Graph Building** — runs locally with plain Python (no Isaac Sim needed)

The building model (`MSOEBuildingTake2.usd`) contains 18,690 prims exported from Revit, covering 5 floors: basement + 4 above-ground floors.

---

## Cell Classification

The occupancy map classifies every 3cm x 3cm cell into one of three states:

| Value | Name | Color (PNG) | Meaning |
|-------|------|-------------|---------|
| **0** | **Free** (white space) | White | Open, navigable space. The raycast confirmed nothing blocks this cell — a robot can drive through it. Includes hallways, room interiors, and open areas. |
| **1** | **Occupied** | Black | A solid obstacle exists here. The raycast hit a wall, column, furniture, or other physical geometry. The robot cannot pass through this cell. |
| **2** | **Unknown** | Gray | No raycast was able to reach this cell from any origin point. This happens when geometry completely blocks line-of-sight (e.g., behind thick walls, outside the building envelope, or in areas the scanner can't access). In nav graph building, unknowns are conservatively treated as walls. |

**How raycasting determines cell state:** From each origin point, the scanner fires rays outward in all directions across the floor plane. If a ray reaches a cell without hitting anything, that cell is marked **free**. If a ray hits geometry at a cell, that cell is marked **occupied**. Cells that no ray from any origin can reach remain **unknown**. Multiple scans are merged — if any scan marks a cell as free or occupied, that overrides unknown from other scans.

---

## Stage 1: Occupancy Map Generation

**Script:** `generate_occupancy_map.py`
**Runs on:** Rosie HPC via `sbatch submit_isaac_job.sh test_building/generate_occupancy_map.py`
**Requires:** Isaac Sim 5.1.0 (`isaacsim.asset.gen.omap` extension), GPU with PhysX

### What it does

1. **Loads the USD scene** — Opens `MSOEBuildingTake2.usd` (Revit export of Diercks Hall) into Isaac Sim headless
2. **Applies physics collision** — Adds `UsdPhysics.CollisionAPI` to all architectural mesh prims (~8,500 meshes), while skipping:
   - **Door geometry** (189 mesh prims under `/Doors/` paths) — skipped so raycasts pass through doorways into rooms
   - **Room volume meshes** (`Room_*` prefixes) — Revit labels, not real geometry
   - **Model text** (`Model_Text*`) — annotation geometry
3. **Sets up PhysX** physics scene and starts the simulation timeline
4. **For each floor** (basement through floor_4):
   - **Round 1**: Scans from a 5x5 grid of origin points (25 origins) spread across the building's XY bounds
   - Each origin fires raycasts to classify cells as **occupied** (1 = wall), **free** (0 = open space), or **unknown** (2 = unreachable)
   - Merges all 25 scans — free/occupied always overrides unknown
   - **Rounds 2-7** (up to `MAX_RESCAN_ROUNDS=6`): Places 50 new origins inside remaining unknown regions and rescans
   - **Stagnation detection**: If unknown % doesn't improve by at least 0.1% between rounds, stops early to avoid wasting time on genuinely unreachable cells
   - **Early exit**: If unknown drops below 0.5%, declares "good enough" and stops
5. **Saves per-floor results**: `.npy` occupancy grid, `.png` visualization, `_meta.json` metadata

### Door Bypass

The key insight enabling room access: Revit exports doors as solid mesh geometry. Without modification, raycasts hit door surfaces and can't see into rooms. The script identifies door prims by checking if `/Doors/` appears in the USD prim path (covers both `/Geometry/Doors/` placed instances and `/FamilyData/Doors/` templates). These prims are excluded from collision, making raycasts pass through doorways.

### Floor Z Slices

Scan heights are set at robot walking height (~50-150cm above each floor slab), with narrow Z ranges to avoid hitting ceiling geometry.

| Floor    | Z min (cm) | Z max (cm) | Description                          |
|----------|-----------|-----------|--------------------------------------|
| basement | 900       | 1100      | Above ground slab (884), 5 utility rooms |
| floor_1  | 1400      | 1600      | Above floor 1 slab (1387), ground floor |
| floor_2  | 1950      | 2050      | Above floor 2 slab (1867)           |
| floor_3  | 2350      | 2450      | ECBE rooms (Z 2250-2570)            |
| floor_4  | 2750      | 2900      | CAECM rooms (Z 2677-2997)           |

Z values are in Revit's coordinate system (centimeters, Z-up). The basement is at Z ~900 (not negative) because the Revit model's ground plane is at Z ~838.

### Single Floor Mode

To scan just one floor (e.g., after adding floor_4):
```bash
# Via environment variable
OMAP_FLOOR=floor_4 sbatch submit_isaac_job.sh test_building/generate_occupancy_map.py

# Via command line argument
./python.sh test_building/generate_occupancy_map.py --floor floor_4
```

### Output Files

| File | Description |
|------|-------------|
| `{floor}_occupancy.npy` | 2D numpy array, shape 4314x4750 (0=free, 1=occupied, 2=unknown) |
| `{floor}_occupancy.png` | Visual: black=walls, white=free, gray=unknown |
| `{floor}_meta.json` | Grid dimensions, cell size, world coordinate bounds |

### Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `CELL_SIZE` | 3.0 cm | Grid resolution |
| `GRID_ORIGINS_X/Y` | 5x5 | Initial scan origins per axis (25 total) |
| `MAX_RESCAN_ROUNDS` | 6 | Max additional passes for unknown regions |
| `RESCAN_ORIGINS_PER_ROUND` | 50 | Origins per rescan round (placed in unknown areas) |
| Stagnation threshold | 0.1% | Minimum improvement to continue rescanning |
| Early exit threshold | 0.5% | Unknown % below which scanning stops |

### Current Occupancy Results

| Floor    | Final Unknown % | Rounds Used | Total Scans |
|----------|----------------|-------------|-------------|
| basement | 1.0%           | 2 (stagnation) | 75       |
| floor_1  | 1.8%           | 2 (stagnation) | 75       |
| floor_2  | 1.3%           | 3 (stagnation) | 125      |
| floor_3  | 1.0%           | 3 (stagnation) | 125      |
| floor_4  | 0.6%           | 3 (stagnation) | 125      |

---

## Stage 2: Navigation Graph Building

**Script:** `build_nav_graph.py`
**Runs on:** Any machine with Python 3 + dependencies (no Isaac Sim needed)
**Input:** `.npy` occupancy grids and `_meta.json` files from Stage 1

### Pipeline Steps

#### Step 1: Clean Occupancy Grid
- Loads the `.npy` array
- Applies morphological closing (10cm radius) to fill small gaps
- Treats unknown cells as occupied (conservative — prevents routing through unseen areas)
- Produces a binary free/not-free mask

#### Step 2: Erode by Robot Radius
- Shrinks free space by the robot's physical radius (default 25cm)
- Prevents the robot from clipping walls or tight spaces
- At 3cm cell size, this erodes by 8 cells in all directions

#### Step 3: Skeletonize
- Applies topological skeletonization (scikit-image) to find corridor center-lines
- Reduces the free space to 1-pixel-wide paths representing the medial axis
- These skeleton paths naturally follow hallway centers and room midpoints

#### Step 4: Build Raw Graph
- Converts skeleton pixels to a NetworkX graph
- 8-connected neighbors: cardinal weight = 1.0, diagonal weight = 1.414
- Keeps only the largest connected component (discards tiny disconnected fragments)

#### Step 5: Simplify Graph
- Identifies **key nodes**: junctions (degree > 2), endpoints (degree 1), dead-ends
- Chains of degree-2 nodes (straight corridors) become single weighted edges
- Removes short dead-end edges (< 5 cells) as noise
- Typical reduction: tens of thousands of raw nodes → hundreds of junction nodes

#### Step 6: Wall-Crossing Validation
- Checks every simplified edge against the original occupancy grid
- For each edge, verifies the stored skeleton path doesn't pass through any wall (occupied/unknown) cell
- If no skeleton path is stored, checks a Bresenham straight line between the two endpoint nodes
- Edges that cross walls are removed entirely
- Isolated nodes (no remaining valid edges) are kept but shown in orange — valid locations that can't be reached without crossing a wall

#### Step 7: Pathfinding (optional)
- Given start/end world coordinates, finds the nearest graph nodes
- Runs Dijkstra's algorithm for shortest weighted path
- Returns path as both grid coordinates and world coordinates (cm)

### Usage

```bash
# Process all floors
python3 test_building/build_nav_graph.py

# Process one floor
python3 test_building/build_nav_graph.py --floor floor_1

# Find a path (world coordinates in cm)
python3 test_building/build_nav_graph.py --floor floor_1 --start="-500,0" --end="1500,1000"

# Custom robot radius
python3 test_building/build_nav_graph.py --robot-radius 30
```

### Output Files

| File | Description |
|------|-------------|
| `{floor}_nav_graph.json` | Graph as JSON: nodes with world coords + edges with weights |
| `{floor}_nav_graph.png` | Visualization: gray=occupancy, green=edges, red=nodes, orange=isolated nodes, yellow=Dijkstra path |

### Current Nav Graph Results

| Floor    | Nodes | Edges | Notes |
|----------|-------|-------|-------|
| basement | 29    | 40    | Small — only 5 utility/mechanical rooms in model |
| floor_1  | 674   | 907   | Ground floor, main entrance, labs |
| floor_2  | 709   | 978   | Offices, classrooms |
| floor_3  | 720   | 989   | ECBE department rooms |
| floor_4  | 590   | 800   | CAECM department rooms |

### Coordinate Systems

- **Grid coordinates:** `(row, col)` — row is Y axis, col is X axis
- **World coordinates:** `(x, y)` in centimeters, matching the USD stage
- Conversion: `x = xy_min[0] + col * cell_size + cell_size/2`
- Conversion: `y = xy_min[1] + row * cell_size + cell_size/2`

---

## Full Pipeline Summary

```
MSOEBuildingTake2.usd (Revit 3D model)
        │
        ▼
┌─────────────────────────────────┐
│  Stage 1: generate_occupancy_   │  Rosie GPU (Isaac Sim)
│  map.py                         │
│                                 │
│  1. Load USD scene              │
│  2. Skip door collision         │
│  3. Apply physics to walls      │
│  4. Per floor:                  │
│     - Raycast from 25 origins   │
│     - Rescan unknown regions    │
│     - Stagnation detection      │
│  5. Save .npy + .png + .json    │
└─────────────────────────────────┘
        │
        ▼  {floor}_occupancy.npy
        │  {floor}_meta.json
        ▼
┌─────────────────────────────────┐
│  Stage 2: build_nav_graph.py    │  Any Python environment
│                                 │
│  1. Clean grid (morphological)  │
│  2. Erode by robot radius       │
│  3. Skeletonize → center-lines  │
│  4. Build NetworkX graph        │
│  5. Simplify (remove degree-2)  │
│  6. Remove wall-crossing edges  │
│  7. Optional: Dijkstra path     │
└─────────────────────────────────┘
        │
        ▼  {floor}_nav_graph.json
           {floor}_nav_graph.png
```

---

## Stage 3: Navigation Task Solving

**Script:** `generate_routes.py`
**Runs on:** Any machine with Python 3 + numpy (no Isaac Sim needed)
**Input:** `navigation/tasks.json` + `floor_plans_resized/floor1/floor1_nodes.json`
**Output:** `navigation/routes.json`

### What it does

1. Loads the floor 1 navigation graph from `floor_plans_resized/floor1/floor1_nodes.json` (nodes + edges)
2. Reads all 18 tasks from `navigation/tasks.json`, each specifying a `start` and `goal` node ID
3. For each task, runs Dijkstra's algorithm on the graph to find the shortest-weighted path
4. Fills in the `waypoints` list (ordered node IDs from start to goal) and `total_distance` (px) for each solved task
5. Writes the completed task data to `navigation/routes.json`

### Task Difficulty Levels

| Difficulty | Count | Criteria |
|------------|-------|----------|
| Easy       | 6     | Same wing or corridor, 2–4 intermediate nodes, straight-line distance < 200 px |
| Medium     | 6     | Cross-wing traversal, 5–9 intermediate nodes, distance 200–500 px |
| Hard       | 6     | Full building diagonal or end-to-end, 10+ intermediate nodes, distance > 500 px |

### Usage

```bash
# Solve all tasks (reads tasks.json, writes routes.json)
python3 generate_routes.py
```

### Output Format

`navigation/routes.json` mirrors `tasks.json` with two fields populated per solved task:

| Field | Description |
|-------|-------------|
| `waypoints` | Ordered list of node IDs from start to goal (e.g., `["f1_office_07", "f1_corr_top_06", "f1_office_10"]`) |
| `total_distance` | Total Dijkstra path cost in pixels (Euclidean edge weights summed) |

Tasks where the start or goal node is missing from the graph, or no connected path exists, will have `waypoints: null` and `total_distance: null`.

### Current Results (Floor 1)

| Task ID    | Difficulty | Start → Goal                              | Waypoints | Distance (px) |
|------------|------------|-------------------------------------------|-----------|---------------|
| easy_01    | easy       | Office 7 → Office 10                      | 4         | 254.9         |
| easy_02    | easy       | Office 11 → Office 13                     | 4         | 233.1         |
| easy_06    | easy       | Left Wing Corr 1 → Left Wing Corr 3       | 3         | 140.0         |
| easy_03–05 | easy       | (nodes not yet in graph)                  | —         | null          |
| medium_01–06 | medium   | Various cross-wing routes                 | —         | null          |
| hard_01–06 | hard       | Full diagonal/end-to-end traversals       | —         | null          |

*Unsolved tasks reflect nodes that haven't been added to `floor1_nodes.json` yet.*

### Dependencies

- Python 3.8+
- numpy

---

## Known Limitations

1. **Basement is sparse** — The Revit model only contains 5 basement rooms (utility/mechanical). The occupancy map and nav graph are small because there's little geometry to scan.

2. **No multi-floor connections** — Each floor is processed independently. Stairwells and elevators are not yet connected between floor graphs.

3. **Conservative unknown handling** — Unknown cells are treated as walls, which may block some valid paths near building edges or in areas raycasts can't reach.

4. **Remaining unknowns (~1-2%)** — Some cells remain unknown due to geometry that raycasts can't reach (e.g., behind thick walls, outside the building envelope). Stagnation detection stops wasting time on these.

5. **No furniture/props** — The USD model includes architectural geometry (walls, floors, doors) but interior furniture is not present, so rooms appear as open rectangles.

---

## Dependencies

### Stage 1 (Rosie HPC)
- NVIDIA Isaac Sim 5.1.0
- `isaacsim.asset.gen.omap` extension
- GPU with PhysX support (T4 or better)

### Stage 2 (Local)
- Python 3.10+
- numpy, scipy, scikit-image, networkx, Pillow

Install Stage 2 dependencies:
```bash
pip install numpy scipy scikit-image networkx Pillow
```

---

## Helper Scripts

| Script | Purpose | Requires |
|--------|---------|----------|
| `inspect_door_prims.py` | Analyze door prim hierarchy in the USD file | Isaac Sim |
| `check_basement_rooms.py` | List room prims and Z levels per floor | Isaac Sim |

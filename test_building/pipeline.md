# Navigation Pipeline

## Overview

This pipeline converts 3D building geometry (USD) into a 2D navigation graph for robot pathfinding. It has two stages:

1. **Occupancy Map Generation** — runs inside Isaac Sim on Rosie (GPU required)
2. **Navigation Graph Building** — runs locally with plain Python (no Isaac Sim needed)

---

## Stage 1: Occupancy Map Generation

**Script:** `generate_occupancy_map.py`
**Runs on:** Rosie HPC via `sbatch submit_isaac_job.sh test_building/generate_occupancy_map.py`
**Requires:** Isaac Sim 5.1.0 (isaacsim.asset.gen.omap extension)

### What it does

1. Loads `TestBuilding.usd` into Isaac Sim
2. Applies `UsdPhysics.CollisionAPI` to all architectural meshes (walls, panels, doors) while skipping HVAC ducts, structural steel, and footings
3. Sets up PhysX physics scene
4. For each floor (basement, floor_1, floor_2, floor_3):
   - Scans from a 5x5 grid of origin points using the occupancy map raycaster
   - Each cell is classified as **occupied** (wall), **free** (open space), or **unknown** (unreachable)
   - Merges all scans — free/occupied overrides unknown
   - Runs up to 3 additional rescan rounds targeting remaining unknown regions
5. Saves results per floor

### Floor Z Slices

| Floor    | Z min (cm) | Z max (cm) | Description                |
|----------|-----------|-----------|----------------------------|
| basement | -280      | -40       | Below ground slab          |
| floor_1  | 50        | 200       | Ground floor, below ceiling|
| floor_2  | 360       | 430       | Between 2nd and 3rd slabs  |
| floor_3  | 470       | 820       | Above 3rd slab, below roof |

### Output Files

| File | Description |
|------|-------------|
| `{floor}_occupancy.npy` | 2D numpy array (0=free, 1=occupied, 2=unknown) |
| `{floor}_occupancy.png` | Visual: black=walls, white=free, gray=unknown |
| `{floor}_meta.json` | Grid dimensions, cell size, world coordinate bounds |
| `bbox_info.json` | Building bounding box |

### Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `CELL_SIZE` | 3.0 cm | Grid resolution |
| `GRID_ORIGINS_X/Y` | 5x5 | Number of scan origins per axis |
| `MAX_RESCAN_ROUNDS` | 3 | Extra passes for unknown regions |

---

## Stage 2: Navigation Graph Building

**Script:** `build_nav_graph.py`
**Runs on:** Any machine with Python 3 + numpy, scipy, scikit-image, networkx, Pillow
**Input:** `.npy` occupancy grids from Stage 1

### Pipeline Steps

#### Step 1: Clean Occupancy Grid
- Loads the `.npy` array
- Applies morphological closing (10cm radius) to fill small gaps
- Treats unknown cells as occupied (conservative)
- Produces a binary free/not-free mask

#### Step 2: Erode by Robot Radius
- Shrinks free space by the robot's physical radius (default 25cm)
- Prevents the robot from clipping walls
- At 3cm cell size, this erodes by 8 cells

#### Step 3: Skeletonize
- Applies topological skeletonization to find corridor center-lines
- Reduces the free space to 1-pixel-wide paths
- Floor 1 produces ~15,500 skeleton pixels

#### Step 4: Build Raw Graph
- Converts skeleton pixels to a NetworkX graph
- 8-connected neighbors (cardinal weight=1.0, diagonal weight=1.414)
- Keeps only the largest connected component

#### Step 5: Simplify Graph
- Identifies **key nodes**: junctions (degree > 2), endpoints (degree 1), dead-ends
- Chains of degree-2 nodes become single weighted edges
- Removes short dead-end edges (< 5 cells) as noise
- Typical reduction: 15,000+ nodes → ~200 junction nodes

#### Step 6: Wall-Crossing Validation
- Checks every simplified edge against the original occupancy grid
- For each edge, verifies the stored skeleton path doesn't pass through any wall (occupied/unknown) cell
- If no skeleton path is stored, checks a Bresenham straight line between the two endpoint nodes
- Edges that cross walls are removed entirely
- Isolated nodes (nodes with no remaining valid edges) are kept but shown in orange in the visualization — they represent valid locations that simply cannot be reached without crossing a wall

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
| `{floor}_nav_graph.png` | Visualization: gray=occupancy, green=validated edges, red=connected nodes, orange=isolated nodes, yellow=Dijkstra path |

### Current Results (without props)

| Floor    | Skeleton Pixels | Raw Nodes | Simplified Nodes | Edges (before) | Wall-Crossing Removed | Final Edges | Isolated Nodes |
|----------|----------------|-----------|-----------------|---------------|----------------------|-------------|---------------|
| basement | 12,917         | 7,231*    | 99              | 131           | 0                    | 131         | 0             |
| floor_1  | 15,578         | 15,578    | 235             | 365           | 40                   | 325         | 2             |
| floor_2  | 13,411         | 13,411    | 199             | 297           | 52                   | 245         | 11            |
| floor_3  | 7,575          | 7,575     | 47              | 70            | 6                    | 64          | 1             |

*basement has 3 disconnected components; only the largest is used

### Coordinate Systems

- **Grid coordinates:** `(row, col)` — row is Y axis, col is X axis
- **World coordinates:** `(x, y)` in centimeters, matching the USD stage
- Conversion: `x = xy_min[0] + col * cell_size + cell_size/2`
- Conversion: `y = xy_min[1] + row * cell_size + cell_size/2`

---

## Known Limitations

1. **Missing interior walls** — The USD file references 2,176 prop files (interior walls, doors, partitions) at `~/props/` that aren't available yet. Once the props folder is obtained and placed correctly, re-running Stage 1 will produce much more detailed occupancy maps with actual room boundaries.

2. **No multi-floor connections** — The current pipeline processes each floor independently. Stairwells and elevators are not yet connected between floors.

3. **Conservative unknown handling** — Unknown cells are treated as walls, which may block some valid paths near building edges.

---

## Dependencies

### Stage 1 (Rosie HPC)
- NVIDIA Isaac Sim 5.1.0
- `isaacsim.asset.gen.omap` extension
- GPU with PhysX support

### Stage 2 (Local)
- Python 3.12+
- numpy
- scipy
- scikit-image (skimage)
- networkx
- Pillow (PIL)

Install Stage 2 dependencies:
```bash
pip install numpy scipy scikit-image networkx Pillow
```

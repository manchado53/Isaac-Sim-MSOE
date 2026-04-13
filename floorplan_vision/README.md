# Floorplan Vision — Occupancy Maps & Navigation Graphs

Everything that converts a building USD into Nav2-ready data lives in this directory. Four scripts, one pipeline.

```
Building USD
    ↓  generate_occupancy_map.py
    ↓
.png + .npy + meta.json
    ↓  auto_graph_generator.py
    ↓
navgraph.{json, gml} + map_metadata.yaml
    ↓  plot_graph_with_axes.py   (visualization, optional)
    ↓
navgraph with labeled axes PNG
```

After generation, [`navigation/nav2_orchestrator.py`](../navigation/nav2_orchestrator.py) loads the USD in Isaac Sim and Nav2 reads the navgraph YAML to build its static costmap.

## Scripts

### `generate_occupancy_map.py`

Loads `assets/environments/InnovationCenter/InnovationCenter.usd` (or any other building USD you point it at), applies collision APIs to architectural meshes, then runs Isaac Sim's `isaacsim.asset.gen.omap` extension as a multi-origin raycast scanner at a fixed height for each floor. Writes per-floor `.png`, `.npy`, and `meta.json` to `assets/environments/InnovationCenter/maps/`.

Key behavior:

- **Multi-origin scan** — single-origin omap leaves "unknown" regions behind walls. We scan from a 5×5 grid of origin points plus up to 6 rescan rounds in remaining unknown regions, merging with `OCCUPIED > FREE > UNKNOWN` precedence.
- **Coordinate orientation** — Isaac's omap plugin emits `col 0 = x_max, row 0 = y_min` in its raw buffer. `save_floor()` applies `np.fliplr` + `np.flipud` so the saved PNG has `col 0 = x_min, row 0 = y_max` — the ROS `map_server` convention. **See "Coordinate system" below.**
- **Floor slices** — each floor is a Z range in the world frame. Floor 1 = 14.0–16.0 m, floor 2 = 19.5–20.5 m, floor 3 = 23.5–24.5 m, floor 4 = 27.5–29.0 m (all metric, InnovationCenter.usd is `metersPerUnit = 1.0`). These come from an earlier find_floor_levels analysis.

Run:

```bash
sbatch submit_isaac_job.sh floorplan_vision/generate_occupancy_map.py           # all 4 floors
OMAP_FLOOR=floor_4 sbatch submit_isaac_job.sh floorplan_vision/generate_occupancy_map.py   # just one
~/isaacsim/python.sh floorplan_vision/generate_occupancy_map.py --floor floor_4   # local, no SLURM
```

Each floor takes ~3–6 min on a T4 or RTX 5090. Outputs go to `assets/environments/InnovationCenter/maps/`:

- `floor_N_occupancy.png` — 8-bit grayscale (black wall, white free, grey unknown), ROS `map_server` compatible
- `floor_N_occupancy.npy` — numpy array (0 = occupied, 1 = free, 2 = unknown, opposite of the PNG convention but same data)
- `floor_N_meta.json` — grid size, `xy_min`/`xy_max`, cell size, scan parameters
- `bbox_info.json` — global building bounding box (shared by all floors)

### `auto_graph_generator.py`

Reads a floor's PNG + meta.json, erodes free space by the robot's safety clearance, skeletonizes the remaining free region, samples waypoints along the medial axis, and builds a sparse navigation graph. Outputs the result in several formats for different consumers.

Stages:

1. **Load** — reads `floor_N_occupancy.png` + `floor_N_meta.json` from `maps/`
2. **Erode** — shrinks free space by `safety_dist` meters (default 0.36 m, ~Nova Carter inscribed radius + a bit)
3. **Skeletonize** — medial axis transform (CPU equivalent of SWAGGER)
4. **Key pixels** — find junctions (degree ≥ 3) and endpoints (degree 1) in the skeleton
5. **Sample nodes** — place waypoints at junctions, endpoints, and every `sample_dist` meters (default 2 m) along skeleton branches
6. **Build graph** — connect nodes via BFS along the skeleton, merge nodes within `merge_dist` (default 0.8 m), prune dead-end stubs shorter than `min_stub` (default 2 m)
7. **Export** — JSON (for `sim_utils/building_info.py::lookup_node`), GML (NetworkX format for graph tools), YAML (ROS `map_server` format for Nav2), and a visualization PNG

Run:

```bash
# One floor
python3 floorplan_vision/auto_graph_generator.py --floor floor_4

# All floors (CPU-only, no GPU needed; ~10 s per floor)
for f in floor_1 floor_2 floor_3 floor_4; do
  python3 floorplan_vision/auto_graph_generator.py --floor $f
done
```

Writes to `assets/environments/InnovationCenter/navgraphs/`:

- `floor_N_auto_navgraph.json` — nodes with `world_xy_m` + `grid_rc`, edges with `distance_m`. **This is the authoritative navgraph used by `sim_utils/building_info.py::lookup_node`.**
- `floor_N_auto_navgraph.gml` — NetworkX-compatible graph format
- `floor_N_map_metadata.yaml` — Nav2 `map_server` YAML. Points at `../maps/floor_N_occupancy.png`, so the YAML in `navgraphs/` can still reference the PNG in the sibling `maps/` directory.
- `floor_N_auto_navgraph.png` — visualization overlay (cyan edges, red nodes on top of the occupancy map)

### `plot_graph_with_axes.py`

Diagnostic visualization. Renders the occupancy map + navgraph with **labeled world-coordinate axes** and a 5 m grid overlay. Useful for picking spawn coordinates by eye or verifying that the navgraph aligns with the building.

```bash
python3 floorplan_vision/plot_graph_with_axes.py --floor floor_4              # just nodes
python3 floorplan_vision/plot_graph_with_axes.py --floor floor_4 --show-ids   # label every node ID
```

Writes `assets/environments/InnovationCenter/navgraphs/floor_N_graph_with_axes.png`.

### `verify_orientation.py`

Post-regeneration smoke test. Runs four checks per floor:

1. **Corner bounds** — `pixel_to_world(0, 0)` equals `(xy_min[0], xy_max[1])` and `pixel_to_world(H-1, W-1)` equals `(xy_max[0], xy_min[1])`. Catches off-by-one errors.
2. **Round trip** — 1000 random pixels round-trip through `pixel_to_world → world_to_pixel` exactly.
3. **Physical sanity** (floor 4 only) — the configured AMCL spawn pose at `(-37.20, 82.09)` maps to a pixel that is in free space in the PNG. Catches Y-flip regressions.
4. **Diff vs backup** — if you pass a `--backup tar.gz` of the pre-regeneration `maps/` directory, verifies the new PNG is approximately `np.flipud(old PNG)`.

```bash
python3 floorplan_vision/verify_orientation.py                                 # all floors
python3 floorplan_vision/verify_orientation.py --floor floor_4                 # just one
python3 floorplan_vision/verify_orientation.py --backup /tmp/maps_backup.tar.gz
```

Exit 0 = all checks pass, exit 1 = at least one failed.

## Full regeneration runbook

After changing the building USD, editing the generator, or adjusting a floor's Z range, regenerate everything:

```bash
# 0. (Optional but recommended) back up current maps + navgraphs
tar -czf /tmp/maps_backup_$(date +%s).tar.gz \
  assets/environments/InnovationCenter/maps \
  assets/environments/InnovationCenter/navgraphs

# 1. Regenerate all 4 occupancy maps (SLURM or local GPU, ~3-6 min each)
sbatch submit_isaac_job.sh floorplan_vision/generate_occupancy_map.py
# OR locally:
export OMNI_KIT_ACCEPT_EULA=YES PRIVACY_CONSENT=Y
~/isaacsim/python.sh floorplan_vision/generate_occupancy_map.py

# 2. Regenerate all 4 navgraphs (CPU, ~10 s each)
for f in floor_1 floor_2 floor_3 floor_4; do
  python3 floorplan_vision/auto_graph_generator.py --floor $f
done

# 3. Verify Y-orientation is still correct
python3 floorplan_vision/verify_orientation.py

# 4. Update building_info.json if node/edge counts or bounds changed
#    - xy_min / xy_max
#    - grid_width_px / grid_height_px
#    - nav_graph_nodes / nav_graph_edges (read from navgraphs/floor_N_auto_navgraph.json 'total_nodes'/'total_edges')

# 5. (Optional) visualize to eyeball correctness
python3 floorplan_vision/plot_graph_with_axes.py --floor floor_4
```

Nav2 does not need a rebuild or restart — it re-reads the YAML when the orchestrator next passes it a `map:=<path>` argument.

## Coordinate system

Nav2's `map_server` uses a **non-obvious** convention for occupancy map orientation. Getting this wrong was the single biggest bug fix of this branch.

### The convention

- **Row 0 of the PNG = world y_max** (top row of the image is the highest world Y)
- **Row H-1 = world y_min** (bottom row is the lowest Y)
- **Col 0 = world x_min** (leftmost column is the smallest X)
- **`origin` field in the YAML = world (x, y, yaw) of the BOTTOM-LEFT pixel** (row = H-1, col = 0)

Under this convention the formula for a pixel's world position is:

```
world_x = origin.x + col * resolution
world_y = origin.y + (H - 1 - row) * resolution
```

### Why it's non-obvious

`nav2_map_server::loadMapFromFile` internally reverses the image rows before publishing the `OccupancyGrid`. From the Nav2 source at `nav2_map_server/src/map_io.cpp:326`:

```cpp
// Flip image vertically (as ROS expects origin at bottom-left)
Eigen::Matrix<int8_t, Eigen::Dynamic, Eigen::Dynamic,
  Eigen::RowMajor> flipped = result.colwise().reverse();
std::memcpy(msg.data.data(), flipped.data(), width * height);
```

So even though visually the PNG "looks" the same as any map you'd draw top-down, the convention requires the PNG data to already be oriented such that after Nav2's internal flip, index `[0]` of the OccupancyGrid corresponds to the bottom-left.

### How we enforce it

Isaac's `isaacsim.asset.gen.omap` plugin emits its raw buffer with `col 0 = x_max` and `row 0 = y_min`. `generate_occupancy_map.py::save_floor` applies two flips to end up with the ROS convention:

```python
grid = np.fliplr(grid)   # col 0 = x_max  →  col 0 = x_min
grid = np.flipud(grid)   # row 0 = y_min  →  row 0 = y_max
```

`auto_graph_generator.py::pixel_to_world` then does:

```python
y = meta["xy_min"][1] + (meta["grid_height"] - 1 - row) * res
```

And `world_to_pixel` uses `round()` instead of `int()` so the round-trip is exact on a 0.05 m grid.

### Symptoms if this breaks

If you save PNGs with `row 0 = y_min` (the natural visual orientation) and don't flip:

- The static costmap Nav2 publishes is **Y-mirrored** relative to the building. Walls appear where there should be free space and vice versa.
- AMCL can't match live lidar scans to the (mirrored) map, so it drifts or fails to converge.
- Global plans route through phantom corridors, the local costmap disagrees, and the robot wanders into real walls thinking it's in free space.
- `floor_4_occupancy.png` on disk still **looks** right to the eye — only Nav2 sees it flipped.

Run `verify_orientation.py` after any change to `generate_occupancy_map.py` or `auto_graph_generator.py` to catch this.

## Output file map

Who writes what, and who reads it:

```
generate_occupancy_map.py
  WRITES → assets/environments/InnovationCenter/maps/
              ├── floor_N_occupancy.png
              ├── floor_N_occupancy.npy
              ├── floor_N_meta.json
              └── bbox_info.json (global)

auto_graph_generator.py
  READS  ← assets/environments/InnovationCenter/maps/floor_N_occupancy.png
  READS  ← assets/environments/InnovationCenter/maps/floor_N_meta.json
  WRITES → assets/environments/InnovationCenter/navgraphs/
              ├── floor_N_auto_navgraph.json
              ├── floor_N_auto_navgraph.gml
              ├── floor_N_map_metadata.yaml  (points at ../maps/floor_N_occupancy.png)
              └── floor_N_auto_navgraph.png  (visualization)

plot_graph_with_axes.py
  READS  ← maps/floor_N_occupancy.png + navgraphs/floor_N_auto_navgraph.json
  WRITES → navgraphs/floor_N_graph_with_axes.png

verify_orientation.py
  READS  ← maps/floor_N_occupancy.png + maps/floor_N_meta.json
  (no output files, just exit code + console)

sim_utils/building_info.py::lookup_node
  READS  ← navgraphs/floor_N_auto_navgraph.json (via building_info.json path)

Nav2 map_server (via ros2 launch kern_navigation ...)
  READS  ← navgraphs/floor_N_map_metadata.yaml (which points at ../maps/floor_N_occupancy.png)
```

## Adding a new building

1. **Place the USD** at `assets/environments/<BuildingName>/<BuildingName>.usd`. It should have `metersPerUnit = 1.0` and the Z axis up.
2. **Add a `building_info.json`** at `assets/environments/<BuildingName>/building_info.json`. Template (see `InnovationCenter/building_info.json` for a full example):
   ```json
   {
     "name": "Full name of the building",
     "short_name": "BuildingName",
     "usd": {"primary": "BuildingName.usd", "physics": "BuildingName.usd"},
     "units": {"length": "meters", "meters_per_unit": 1.0},
     "total_floors": N,
     "floors": {
       "1": {"z_min": X, "z_max": Y, "floor_deck_z": X, "default_spawn": {"x": 0, "y": 0, "yaw": 0}},
       ...
     }
   }
   ```
3. **Update `generate_occupancy_map.py`** — edit `USD_PATH` and the `all_floor_slices` table to list each floor's Z scan range (typically `floor_deck_z + 0.5` to `floor_deck_z + 1.5` for robot-height scans).
4. **Run the pipeline**:
   ```bash
   ~/isaacsim/python.sh floorplan_vision/generate_occupancy_map.py
   for f in floor_1 floor_2 floor_3 floor_4; do
     python3 floorplan_vision/auto_graph_generator.py --floor $f
   done
   python3 floorplan_vision/verify_orientation.py
   ```
5. **Update `building_info.json`** with the real `xy_min`/`xy_max`/grid dimensions/node counts from the generator output.
6. **Point `nav2_orchestrator.py`** at the new building via `--building <BuildingName>`.

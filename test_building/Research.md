# NavMesh / Occupancy Map Research for TestBuilding.usd

## Overview

Goal: Extract walkable routes from the 3D USD building model (`assets/environments/msoe_building/TestBuilding.usd`) and build a navigation graph for Dijkstra's pathfinding. This replaces the previous 2D floor plan image approach.

### TestBuilding.usd Stats
- **5,947 prims** (walls, panels, mullions, doors, etc.)
- **Up axis:** Z
- **Units:** Centimeters (metersPerUnit = 0.01)
- **Source:** Exported from Revit
- **Top-level elements:** `Basic_Wall_Exterior_UWM`, `System_Panel_Solid`, `System_Panel_Glazed`, `Rectangular_Mullion`, `Door_Curtain_Wall_Double_Storefront`

---

## omni.isaac.occupancy_map

### What It Does
Creates a **2D binary occupancy grid from 3D scene geometry** using PhysX collision raycasting. Each cell in the grid is marked as occupied, free, or unknown.

### Extension Names by Version
| Isaac Sim Version | Extension | Menu Location |
|---|---|---|
| 4.2.0 and earlier | `omni.isaac.occupancy_map` | Isaac Utils > Occupancy Map |
| 4.5.0+ | `isaacsim.asset.gen.omap` | Tools > Robotics > Occupancy Map |

### Critical Requirements
1. **All geometry must have collision enabled** — `UsdPhysics.CollisionAPI` must be applied to mesh prims. Revit exports do NOT have this by default.
2. **Origin must be in free space** — the starting point for the flood-fill must be in a hallway/open area, not inside a wall.
3. **Parameters are in stage units** — since our stage is in centimeters, all values (cell_size, bounds) are in cm.

### Python API

```python
# Isaac Sim 4.5.0+
import omni
from isaacsim.asset.gen.omap import _omap

physx = omni.physx.acquire_physx_interface()
stage_id = omni.usd.get_context().get_stage_id()

generator = _omap.Generator(physx, stage_id)

# cell_size, occupied_value, unoccupied_value, unknown_value
generator.update_settings(5.0, 1, 0, 2)  # 5cm cells

# origin (must be free space), min_bound, max_bound
generator.set_transform(
    (origin_x, origin_y, 100),    # origin in cm
    (min_x, min_y, 10),           # min bound
    (max_x, max_y, 250)           # max bound (2.5m ceiling)
)

generator.generate2d()

# Retrieve results
buffer = generator.get_buffer()           # flat list of cell values (row-major)
dims = generator.get_dimensions()          # carb.Int3 (width, height, depth)
occupied = generator.get_occupied_positions()
free = generator.get_free_positions()
```

```python
# Isaac Sim 4.2.0 and earlier
from omni.isaac.occupancy_map import _occupancy_map
# Same methods, different import path
```

### Key Parameters

| Parameter | Description | Recommended Value (cm stage) |
|---|---|---|
| `cell_size` | Grid resolution | 5.0 (= 5cm per cell) |
| `min_bound Z` | Bottom of slice | 10.0 (above floor slab) |
| `max_bound Z` | Top of slice | 250.0 (below ceiling) |
| `origin` | Starting point for flood-fill | A known hallway coordinate |
| `occupied_value` | Cell value for walls | 1 |
| `unoccupied_value` | Cell value for free space | 0 |
| `unknown_value` | Cell value for unreachable | 2 |

### Saving as Image

```python
from PIL import Image
import numpy as np

colored = generator.get_colored_byte_buffer(
    carb.Int4(0, 0, 0, 255),         # occupied = black
    carb.Int4(255, 255, 255, 255),    # free = white
    carb.Int4(128, 128, 128, 255)     # unknown = gray
)
img_array = np.array(list(colored), dtype=np.uint8).reshape(dims.y, dims.x, 4)
img = Image.fromarray(img_array)
img.save("occupancy_map.png")
```

---

## NavMesh (omni.anim.navigation)

### What It Does
Omniverse has a built-in Navigation Mesh extension that can bake a NavMesh from scene geometry and query shortest paths directly.

### Python API

```python
import omni.anim.navigation.core as nav

inav = nav.acquire_interface()
inav.start_navmesh_baking_and_wait()

navmesh = inav.get_navmesh()
path = navmesh.query_shortest_path(
    start_pos=carb.Float3(-200.0, 0.0, -30.0),
    end_pos=carb.Float3(250.0, 0.0, 150.0)
)
```

### UI Workflow
1. Create > Navigation > NavMesh Include Volume
2. Scale volume to cover the floor area
3. Adjust settings (cell size, agent radius, agent height)
4. Bake the NavMesh

### Third-Party: cadop/ov_navmesh
GitHub project using Recast/Detour for more control over NavMesh generation in Omniverse: https://github.com/cadop/ov_navmesh

---

## Recommended Pipeline: Occupancy Map → Skeleton → Graph

This is the recommended approach because it gives full control and integrates with our existing Dijkstra code.

### Step 1: Apply Collision to All Meshes

```python
from pxr import UsdPhysics, UsdGeom

stage = omni.usd.get_context().get_stage()
for prim in stage.Traverse():
    if prim.IsA(UsdGeom.Mesh):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI.Apply(prim)
```

### Step 2: Determine Building Extents
Inspect the USD bounding box to find min/max XY coordinates and floor Z levels.

### Step 3: Generate 2D Occupancy Map
Use the occupancy map generator with appropriate cell size and Z slice for each floor.

### Step 4: Convert to Numpy + Erode by Robot Radius

```python
import numpy as np
from skimage.morphology import binary_erosion, disk

occupancy_grid = np.array(buffer).reshape(dims.y, dims.x)
free_mask = (occupancy_grid == 0).astype(np.uint8)

robot_radius_cells = 3  # e.g., 15cm radius / 5cm cell
free_mask_eroded = binary_erosion(free_mask, disk(robot_radius_cells))
```

### Step 5: Skeletonize Free Space

```python
from skimage.morphology import skeletonize

skeleton = skeletonize(free_mask_eroded)
```

### Step 6: Build Graph from Skeleton

```python
import networkx as nx

G = nx.Graph()
skeleton_coords = np.argwhere(skeleton)
coord_set = set(map(tuple, skeleton_coords))

for r, c in skeleton_coords:
    G.add_node((r, c))
    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
        nr, nc = r + dr, c + dc
        if (nr, nc) in coord_set:
            weight = 1.0 if dr == 0 or dc == 0 else 1.414
            G.add_edge((r, c), (nr, nc), weight=weight)

# Simplify: junctions (degree != 2) become key nodes
key_nodes = [n for n in G.nodes() if G.degree(n) != 2]
```

### Step 7: Convert Grid Coords to World Coords

```python
min_bound = generator.get_min_bound()
cell_size = 5.0  # cm

def grid_to_world(r, c):
    x = min_bound.x + c * cell_size + cell_size / 2
    y = min_bound.y + r * cell_size + cell_size / 2
    return (x, y)
```

### Step 8: Run Dijkstra

```python
path = nx.dijkstra_path(G, source=(r1, c1), target=(r2, c2), weight='weight')
world_path = [grid_to_world(r, c) for r, c in path]
```

---

## Important Notes

- The occupancy map pipeline **must run inside Isaac Sim** (requires PhysX simulation context), not standalone `python.sh`.
- The NavMesh approach also requires the full Omniverse runtime.
- Missing referenced geometry files (in `~/props/`) may cause incomplete walls — need to verify all references resolve.
- For multi-floor buildings, run the pipeline once per floor with different Z slices.

## Sources

- [Isaac Sim 4.5.0 Occupancy Map Docs](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/digital_twin/ext_isaacsim_asset_generator_occupancy_map.html)
- [Isaac Sim 4.2.0 Occupancy Map Docs](https://docs.isaacsim.omniverse.nvidia.com/4.2.0/features/ext_omni_isaac_occupancy_map.html)
- [Isaac Sim 4.5.0 Occupancy Map API](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/py/source/extensions/isaacsim.asset.gen.omap/docs/index.html)
- [Omniverse Navigation Mesh Extension](https://docs.omniverse.nvidia.com/extensions/latest/ext_navigation-mesh.html)
- [Omniverse NavMesh API](https://docs.omniverse.nvidia.com/extensions/latest/ext_navigation-mesh/api.html)
- [cadop/ov_navmesh GitHub](https://github.com/cadop/ov_navmesh)
- [Isaac Sim ROS2 Navigation Tutorial](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/ros2_tutorials/tutorial_ros2_navigation.html)

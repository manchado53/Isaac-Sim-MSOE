# Automated Node Placement for BIM-Driven Navigation Graphs

## Implementation Guide: From Occupancy Maps to Autonomous Patrol Routes

**Project Context:** MSOE BIM-to-Robot Pipeline — Robert D. Kern Engineering Innovation Center  
**Target Platform:** NVIDIA Nova Carter on Isaac Sim / Isaac Lab  
**Authors:** For use by the MSOE AI Research Team  
**Last Updated:** April 2026

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Solution Overview](#2-solution-overview)
3. [Mathematical Foundation: The Medial Axis Transform](#3-mathematical-foundation-the-medial-axis-transform)
4. [NVIDIA SWAGGER: Deep Technical Reference](#4-nvidia-swagger-deep-technical-reference)
5. [Implementation Steps](#5-implementation-steps)
6. [Parameter Tuning Guide](#6-parameter-tuning-guide)
7. [Integration with the Existing Pipeline](#7-integration-with-the-existing-pipeline)
8. [Advanced: Vertex-Coordinate Approach (Future Work)](#8-advanced-vertex-coordinate-approach-future-work)
9. [Validation and Testing](#9-validation-and-testing)
10. [References](#10-references)

---

## 1. Problem Statement

The current navigation graph pipeline (Section 3.3 of the research paper) operates in three phases:

- **Phase 1 (Automated):** Occupancy maps are generated per floor via `isaacsim.asset.gen.omap`, performing raycast-based scans at 3 cm resolution across 20.5 million cells per floor.
- **Phase 2 (Manual — the bottleneck):** The occupancy map is scaled down ~3.08x to fit a browser window. Nodes are hand-placed using a browser-based GUI tool, and edges are drawn between nodes with clear line-of-sight through free space. Floor 4 required 79 manually placed nodes.
- **Phase 3 (Automated):** Display coordinates are scaled back to grid resolution, converted to Isaac Sim coordinates via bounding box metadata, and exported as JSON with Euclidean edge weights.

**Goal:** Replace Phase 2 entirely with an automated algorithm that places nodes:

- Always on the center of hallways (equidistant from both walls)
- At hallway intersections (T-junctions, L-bends, four-way crossings)
- Aligned with the center of each doorway
- At a configurable spacing along straight corridor runs

---

## 2. Solution Overview

Two complementary approaches, recommended in sequence:

**Immediate solution — NVIDIA SWAGGER:** Feed your existing occupancy maps directly into SWAGGER (Sparse WAypoint Graph Generation for Efficient Routing), NVIDIA's open-source library for generating sparse waypoint graphs from occupancy grids. This drops directly into Phase 2 with zero manual annotation. SWAGGER uses medial axis skeletonization internally to place nodes on hallway centerlines and at intersections.

- **Source:** [github.com/nvidia-isaac/SWAGGER](https://github.com/nvidia-isaac/SWAGGER) (Apache-2.0)
- **Released by:** NVIDIA Isaac team (authors: Rushane Hua, Billy Okal, Benjamin Butin)
- **Integration:** Outputs `networkx.Graph` objects and GML files; includes Nav2 Route Server converter

**Future enhancement — Direct vertex-coordinate approach:** Extract wall vertices, door gaps, and corner coordinates directly from the Revit/USD model via the USD Python API. Compute the Voronoi diagram of wall line segments, clip to interior polygons, and identify junction/doorway nodes from the resulting medial axis graph. This bypasses the occupancy map rasterization step entirely, yielding higher geometric precision and a stronger contribution for the paper.

---

## 3. Mathematical Foundation: The Medial Axis Transform

### What It Is

The medial axis transform (MAT) is the set of all points within a shape that have more than one closest point on the shape's boundary. For a hallway polygon, this produces a line running exactly down the center, equidistant from both walls.

**Key properties (from Lee, 1982 and Blum, 1967):**

- The medial axis of a simple polygon is a tree whose leaves are the vertices of the polygon and whose edges are straight segments or parabolic arcs.
- It is a complete shape descriptor — the original shape can be reconstructed from its MAT.
- Junction points (where three or more branches meet) correspond to hallway intersections.
- The MAT of a polygon is a subset of the Voronoi diagram of its edges.

### Why It Works for Hallway Centerlines

For two parallel walls separated by distance `w`, the medial axis is a straight line at distance `w/2` from each wall — exactly the hallway center. At a T-intersection, the MAT produces a degree-3 branch point at the geometric center of the junction. At a four-way crossing, it produces a degree-4 point. This topology maps directly to the navigation graph structure needed for patrol routing.

### Connection to Voronoi Diagrams

The medial axis of a polygon P can be computed as the portion of the line-segment Voronoi diagram that lies within P. SWAGGER uses this relationship internally: it computes the topological skeleton (a discrete approximation of the medial axis on the pixel grid) using morphological thinning, which is computationally efficient for occupancy grid inputs.

**Reference:** Sawhney, Reed, and Crane, "Medial Axis Transform" (Carnegie Mellon); Lee, D.T. "Medial Axis Transformation of a Planar Shape," 1982.

---

## 4. NVIDIA SWAGGER: Deep Technical Reference

### 4.1 Architecture

SWAGGER operates as a multi-stage pipeline:

**Stage 1 — Safety Erosion:** The free space in the occupancy grid is eroded by the robot's safety distance, ensuring all graph nodes and edges maintain minimum clearance from walls. For Nova Carter with a 0.5m width, a safety distance of 0.3–0.36m (half-width + margin) is appropriate.

**Stage 2 — Skeletonization (Medial Axis):** The eroded free space is skeletonized using morphological thinning to produce a 1-pixel-wide centerline through every corridor. This is the discrete equivalent of the medial axis transform. The skeleton naturally runs through the center of hallways and branches at intersections.

**Stage 3 — Skeleton Sampling:** The continuous skeleton is discretized into waypoint nodes spaced `skeleton_sample_distance` meters apart. Topologically important points (junctions where branches meet, endpoints at dead ends) are always preserved as nodes regardless of spacing.

**Stage 4 — Boundary Sampling (optional):** Nodes can be added along obstacle contours for applications requiring wall-following paths. Disabled for hallway-centric navigation.

**Stage 5 — Free Space Sampling (optional):** Nodes can be added in large open areas (lobbies, atriums) where the skeleton alone leaves gaps. Controlled by `free_space_sampling_threshold`.

**Stage 6 — Delaunay Shortcuts (optional):** Delaunay triangulation adds direct edges between nearby nodes that have line-of-sight, creating more efficient routing through open spaces.

**Stage 7 — Pruning:** Redundant nodes and edges are removed. Disconnected subgraphs shorter than `min_subgraph_edge_length` are eliminated.

### 4.2 Inputs

| Input | Type | Description | Your Value |
|-------|------|-------------|------------|
| `image` | Grayscale ndarray | Occupancy grid: 0=occupied (black), 255=free (white) | Output of `isaacsim.asset.gen.omap` |
| `occupancy_threshold` | int | Pixel value above which a cell is considered free space | 127 |
| `safety_distance` | float (meters) | Minimum clearance from obstacles for all nodes/edges | 0.36 (Nova Carter half-width + margin) |
| `resolution` | float (m/pixel) | Physical size of each pixel | 0.03 (your 3cm cells) |
| `x_offset` | float (meters) | X coordinate of bottom-left pixel in world frame | From omap metadata |
| `y_offset` | float (meters) | Y coordinate of bottom-left pixel in world frame | From omap metadata |
| `rotation` | float (radians) | Rotation from image frame to world frame | 0.0 (typically) |

### 4.3 Configuration Parameters

| Parameter | Default | Unit | Effect |
|-----------|---------|------|--------|
| `skeleton_sample_distance` | 1.0 | meters | Spacing between nodes along centerline. Lower = denser graph. Junction and endpoint nodes are always placed regardless of this value. |
| `boundary_inflation_factor` | 2.0 | unitless | Multiplier for wall clearance during contour detection. Higher = larger margins. |
| `boundary_sample_distance` | 1.0 | meters | Spacing between nodes placed along obstacle boundaries. Only relevant if `use_boundary_samples=True`. |
| `free_space_sampling_threshold` | — | meters | Maximum distance from obstacles for free space samples. Controls how far into open areas nodes are placed. |
| `node_merge_distance` | — | meters | Nodes closer than this are collapsed into one. Prevents redundant nodes at junctions. |
| `min_subgraph_edge_length` | — | meters | Disconnected subgraphs with total edge length below this are pruned. Eliminates tiny fragments from noise. |
| `use_skeleton` | True | bool | Enables medial axis skeleton generation. **Must be True for hallway centerline placement.** |
| `use_boundary_samples` | True | bool | Adds nodes along wall contours. **Set to False for hallway-only navigation.** |
| `use_free_space_samples` | True | bool | Adds nodes in open areas. **Set to False unless the building has large atriums/lobbies.** |
| `use_delaunay_shortcuts` | True | bool | Adds shortcut edges via Delaunay triangulation. Useful for efficient routing through open spaces. |
| `use_pruning` | True | bool | Removes redundant nodes and edges from the final graph. |

### 4.4 Outputs

The primary output is a `networkx.Graph` object where:

- **Each node** has attributes: `x` (float, world meters), `y` (float, world meters), `z` (float, set to 0 for 2D maps), `node_type` (string: indicates which pipeline stage created it — skeleton nodes have no explicit type set; boundary and free-space nodes are labeled accordingly).
- **Each edge** has attributes: `weight` (float, Euclidean distance between endpoints in world coordinates).

Secondary outputs include a GML file (for interoperability) and a PNG visualization overlay.

### 4.5 Nav2 Route Server Integration

SWAGGER includes a GML-to-GeoJSON converter at `integrations/nav2/tools/gml_to_geojson.py` that produces files compatible with the Nav2 Route Server. This is relevant for physical deployment on the Nova Carter via ROS 2.

---

## 5. Implementation Steps

### 5.1 Prerequisites

```bash
# System requirements
# - Python 3.10+
# - CUDA 12.5+ (for GPU-accelerated operations)
# - Ubuntu 22.04 (tested platform)

# Install system dependencies
sudo apt update && sudo apt install -y libgl1-mesa-glx libglib2.0-0

# Clone SWAGGER
git clone git@github.com:nvidia-isaac/SWAGGER.git
cd SWAGGER
git lfs pull

# Create virtual environment
python -m venv swagger-venv
source swagger-venv/bin/activate

# Install SWAGGER
pip install -e .
```

### 5.2 Basic Graph Generation (Replace Manual Phase 2)

```python
"""
automatic_node_placement.py

Replaces manual Phase 2 of the navigation graph pipeline.
Reads occupancy maps generated by isaacsim.asset.gen.omap and
produces a navigation graph with nodes on hallway centerlines.
"""

import cv2
import json
import networkx as nx
from swagger import WaypointGraphGenerator, WaypointGraphGeneratorConfig


def generate_navgraph(
    omap_path: str,
    resolution: float = 0.03,
    safety_distance: float = 0.36,
    x_offset: float = 0.0,
    y_offset: float = 0.0,
    output_dir: str = "results"
) -> nx.Graph:
    """
    Generate a navigation graph from an occupancy map.

    Args:
        omap_path: Path to grayscale occupancy map PNG.
        resolution: Meters per pixel (your omap uses 0.03).
        safety_distance: Min clearance from walls in meters.
        x_offset: World X coordinate of bottom-left pixel.
        y_offset: World Y coordinate of bottom-left pixel.
        output_dir: Directory for output files.

    Returns:
        networkx.Graph with nodes in Isaac Sim world coordinates.
    """

    # Load the occupancy map
    occupancy_grid = cv2.imread(omap_path, cv2.IMREAD_GRAYSCALE)
    if occupancy_grid is None:
        raise FileNotFoundError(f"Could not load occupancy map: {omap_path}")

    # Configure for Kern Center hallway navigation
    config = WaypointGraphGeneratorConfig(
        skeleton_sample_distance=2.0,     # ~matches manual 79-node density
        boundary_inflation_factor=2.0,
        boundary_sample_distance=1.5,
        node_merge_distance=0.8,
        min_subgraph_edge_length=2.0,
        use_skeleton=True,                # CRITICAL: enables centerline placement
        use_boundary_samples=False,       # hallways don't need wall-hugging nodes
        use_free_space_samples=False,     # disable for corridor environments
        use_delaunay_shortcuts=True,
        use_pruning=True,
    )

    # Build the graph
    generator = WaypointGraphGenerator(config)
    graph = generator.build_graph_from_grid_map(
        image=occupancy_grid,
        occupancy_threshold=127,
        safety_distance=safety_distance,
        resolution=resolution,
        x_offset=x_offset,
        y_offset=y_offset,
        rotation=0.0,
    )

    print(f"Generated {graph.number_of_nodes()} nodes, "
          f"{graph.number_of_edges()} edges")

    # Save visualization overlay
    generator.visualize_graph(
        output_dir=output_dir,
        output_filename="auto_navgraph.png"
    )

    return graph


if __name__ == "__main__":
    graph = generate_navgraph("floor4_occupancy_map.png")
```

### 5.3 Export to Your Existing JSON Format

```python
"""
export_navgraph_json.py

Converts SWAGGER's networkx.Graph output to the JSON format
used by the existing pipeline (Section 3.3, Phase 3).
"""

import json
import networkx as nx


def export_to_pipeline_json(
    graph: nx.Graph,
    output_path: str = "navgraph.json"
) -> None:
    """
    Export the navigation graph to the JSON format expected by
    the existing route planning code (Dijkstra, TSP solver).

    The output matches the structure described in Section 3.3:
    node positions in Isaac Sim coordinates (centimeters) with
    pre-calculated Euclidean edge distances.
    """

    nav_graph = {
        "nodes": [],
        "edges": []
    }

    # Build node list with Isaac Sim coordinates
    for node_id in graph.nodes():
        node_data = graph.nodes[node_id]
        nav_graph["nodes"].append({
            "id": int(node_id),
            "x": round(node_data["x"], 4),
            "y": round(node_data["y"], 4),
            "z": round(node_data.get("z", 0.0), 4),
            "node_type": node_data.get("node_type", "skeleton")
        })

    # Build edge list with Euclidean distances
    for u, v, data in graph.edges(data=True):
        nav_graph["edges"].append({
            "from": int(u),
            "to": int(v),
            "distance": round(data["weight"], 4)
        })

    with open(output_path, "w") as f:
        json.dump(nav_graph, f, indent=2)

    print(f"Exported {len(nav_graph['nodes'])} nodes and "
          f"{len(nav_graph['edges'])} edges to {output_path}")
```

### 5.4 Route Planning (Replace Phase 3 Algorithms)

```python
"""
route_planning.py

Route computation using the auto-generated navigation graph.
Implements Dijkstra shortest path and TSP patrol coverage
matching the algorithms described in Section 3.3.
"""

import networkx as nx
from typing import List, Tuple
import random


def shortest_path(
    graph: nx.Graph,
    start: Tuple[float, float],
    goal: Tuple[float, float]
) -> Tuple[List[Tuple[float, float]], float]:
    """
    Dijkstra shortest path between two world-coordinate points.
    Finds nearest graph nodes to start/goal, then computes path.

    Returns:
        Tuple of (waypoint_list, total_distance)
    """

    # Find nearest nodes to the query points
    def nearest_node(point):
        min_dist = float("inf")
        best = None
        for n in graph.nodes():
            nd = graph.nodes[n]
            d = ((nd["x"] - point[0])**2 + (nd["y"] - point[1])**2)**0.5
            if d < min_dist:
                min_dist = d
                best = n
        return best

    start_node = nearest_node(start)
    goal_node = nearest_node(goal)

    path = nx.shortest_path(graph, start_node, goal_node, weight="weight")
    total_dist = nx.shortest_path_length(
        graph, start_node, goal_node, weight="weight"
    )

    waypoints = [
        (graph.nodes[n]["x"], graph.nodes[n]["y"])
        for n in path
    ]

    return waypoints, total_dist


def full_patrol_route(
    graph: nx.Graph,
    num_restarts: int = 10
) -> Tuple[List[int], float]:
    """
    Compute a full-floor patrol route visiting all nodes.
    Uses nearest-neighbor heuristic + 2-opt refinement +
    random restart (matching the paper's approach).

    Returns:
        Tuple of (node_order, total_distance)
    """

    all_nodes = list(graph.nodes())

    # Precompute all-pairs shortest path lengths
    path_lengths = dict(nx.all_pairs_dijkstra_path_length(graph, weight="weight"))

    best_route = None
    best_distance = float("inf")

    for _ in range(num_restarts):
        # Random starting node
        start = random.choice(all_nodes)

        # Nearest-neighbor greedy construction
        route = [start]
        unvisited = set(all_nodes) - {start}

        while unvisited:
            current = route[-1]
            nearest = min(
                unvisited,
                key=lambda n: path_lengths[current].get(n, float("inf"))
            )
            route.append(nearest)
            unvisited.remove(nearest)

        # 2-opt local search refinement
        improved = True
        while improved:
            improved = False
            for i in range(1, len(route) - 2):
                for j in range(i + 1, len(route)):
                    # Current cost of edges (i-1,i) and (j,j+1)
                    i_prev = route[i - 1]
                    i_curr = route[i]
                    j_curr = route[j]
                    j_next = route[(j + 1) % len(route)]

                    old_cost = (
                        path_lengths[i_prev].get(i_curr, float("inf")) +
                        path_lengths[j_curr].get(j_next, float("inf"))
                    )
                    new_cost = (
                        path_lengths[i_prev].get(j_curr, float("inf")) +
                        path_lengths[i_curr].get(j_next, float("inf"))
                    )

                    if new_cost < old_cost:
                        route[i:j + 1] = reversed(route[i:j + 1])
                        improved = True

        # Compute total route distance
        total_dist = sum(
            path_lengths[route[k]].get(route[k + 1], 0)
            for k in range(len(route) - 1)
        )

        if total_dist < best_distance:
            best_distance = total_dist
            best_route = route[:]

    return best_route, best_distance
```

### 5.5 Nav2 GeoJSON Export (For Physical Deployment)

```bash
# After generating graph.gml via SWAGGER CLI:
python scripts/generate_graph.py \
    --map-path floor4_occupancy_map.png \
    --resolution 0.03 \
    --safety-distance 0.36 \
    --output-dir results/

# Convert to GeoJSON for Nav2 Route Server
python integrations/nav2/tools/gml_to_geojson.py results/graph.gml

# The output graph.geojson is ready for Nav2 Route Server
# Pass it via: ros2 launch nav2_bringup ... graph:=/path/to/graph.geojson
```

---

## 6. Parameter Tuning Guide

### 6.1 Recommended Starting Configuration for Kern Center

```python
config = WaypointGraphGeneratorConfig(
    skeleton_sample_distance=2.0,     # Start here, adjust to match ~79 nodes/floor
    boundary_inflation_factor=2.0,    # Default works well for standard hallways
    boundary_sample_distance=1.5,     # Only relevant if use_boundary_samples=True
    node_merge_distance=0.8,          # Collapse junction clusters within 80cm
    min_subgraph_edge_length=2.0,     # Remove fragments shorter than 2m total
    use_skeleton=True,                # REQUIRED for centerline placement
    use_boundary_samples=False,       # Disable for hallway navigation
    use_free_space_samples=False,     # Disable unless building has large open areas
    use_delaunay_shortcuts=True,      # Enable for efficient cross-connections
    use_pruning=True,                 # Always enable
)
```

### 6.2 skeleton_sample_distance Tuning

This is the most impactful parameter. It controls spacing between nodes along straight corridor segments. Junction and endpoint nodes are always placed regardless of this value.

| Value | Node Density | Use Case |
|-------|-------------|----------|
| 0.5m | Very dense | Early RL training (short goal-to-goal hops, simpler per-segment navigation) |
| 1.0m | Dense | Detailed patrol coverage, fine-grained waypoint control |
| 1.5m | Moderate | Balanced density for most buildings |
| 2.0m | Sparse | Matches your manual annotation density (~79 nodes for floor 4) |
| 3.0m | Very sparse | Mature RL policy, minimal waypoint overhead |

**Tuning procedure:**

1. Generate graph at 2.0m.
2. Count nodes. Compare to your manual 79 for floor 4.
3. If node count is too high (>120), increase to 2.5m.
4. If node count is too low (<50) or missing short corridor stubs, decrease to 1.5m.
5. Overlay the visualization on the occupancy map and visually inspect for coverage gaps.

### 6.3 safety_distance Tuning

The Nova Carter platform is 0.722m × 0.500m. The safety distance should be at minimum the robot's half-width (0.25m for the narrow axis) plus a margin.

| Value | Meaning |
|-------|---------|
| 0.25m | Absolute minimum (half robot width, no margin — risky) |
| 0.36m | Recommended (half width + ~0.11m margin) |
| 0.50m | Conservative (allows for localization error and dynamic obstacles) |

### 6.4 Handling Open Spaces

If the Kern Center has lobbies, atriums, or other wide-open areas where the skeleton alone leaves large unserviced regions:

```python
config = WaypointGraphGeneratorConfig(
    # ... hallway parameters as above ...
    use_free_space_samples=True,              # Enable for open areas
    free_space_sampling_threshold=5.0,        # Place nodes up to 5m from walls
)
```

---

## 7. Integration with the Existing Pipeline

### 7.1 Where SWAGGER Fits

```
Phase 1 (unchanged):  isaacsim.asset.gen.omap → per-floor occupancy maps (3cm res)
                                |
Phase 2 (AUTOMATED):  SWAGGER WaypointGraphGenerator → networkx.Graph
                                |
Phase 3 (simplified): export_to_pipeline_json() → navgraph.json
                                |
                      Dijkstra / TSP solver → patrol routes
                                |
                      Isaac Lab RL training → goal waypoints from navgraph
```

### 7.2 Coordinate System Alignment

Your current pipeline stores occupancy map metadata including the bounding box and cell size. These map directly to SWAGGER's coordinate transform parameters:

```python
# Extract from your omap metadata
omap_metadata = load_omap_metadata("floor4_metadata.json")

graph = generator.build_graph_from_grid_map(
    image=occupancy_grid,
    occupancy_threshold=127,
    safety_distance=0.36,
    resolution=omap_metadata["cell_size"],     # 0.03m
    x_offset=omap_metadata["bbox_min_x"],      # World X of bottom-left pixel
    y_offset=omap_metadata["bbox_min_y"],       # World Y of bottom-left pixel
    rotation=0.0,
)
# Output node coordinates are now in Isaac Sim world coordinates (meters)
# Multiply by 100 if your downstream code expects centimeters
```

### 7.3 Multi-Floor Processing

```python
"""
generate_all_floors.py

Batch process all four Kern Center floors.
"""

import os

FLOORS = {
    "floor1": {"omap": "floor1_omap.png", "metadata": "floor1_meta.json"},
    "floor2": {"omap": "floor2_omap.png", "metadata": "floor2_meta.json"},
    "floor3": {"omap": "floor3_omap.png", "metadata": "floor3_meta.json"},
    "floor4": {"omap": "floor4_omap.png", "metadata": "floor4_meta.json"},
}

for floor_name, paths in FLOORS.items():
    print(f"\n--- Processing {floor_name} ---")
    output_dir = f"results/{floor_name}"
    os.makedirs(output_dir, exist_ok=True)

    graph = generate_navgraph(
        omap_path=paths["omap"],
        output_dir=output_dir
    )

    export_to_pipeline_json(
        graph,
        output_path=f"{output_dir}/navgraph.json"
    )
```

---

## 8. Advanced: Vertex-Coordinate Approach (Future Work)

This section describes the more precise approach that works directly from BIM geometry rather than rasterized occupancy maps. This would strengthen the paper's contribution by demonstrating fully automated graph extraction from architectural data without intermediate rasterization.

### 8.1 Input Format

A JSON or CSV file containing vertex coordinates extracted from the Revit/USD model:

```json
{
  "walls": [
    {"id": "w1", "start": [0.0, 0.0], "end": [10.0, 0.0]},
    {"id": "w2", "start": [10.0, 0.0], "end": [10.0, 5.0]}
  ],
  "doors": [
    {"id": "d1", "wall_id": "w2", "start": [10.0, 1.5], "end": [10.0, 2.5]}
  ],
  "corners": [
    {"id": "c1", "position": [10.0, 0.0], "type": "L-bend"}
  ]
}
```

### 8.2 Algorithm Steps

**Step 1 — Construct hallway polygons.** Connect wall segments into closed polygonal regions representing walkable hallway space. Door gaps create openings between adjacent regions.

**Step 2 — Compute the Voronoi diagram of wall segments.** For each hallway polygon, densely sample points along the wall boundaries and compute their Voronoi diagram. Filter to keep only Voronoi edges that lie entirely within the polygon interior. These edges form the medial axis (centerline).

```python
import numpy as np
from scipy.spatial import Voronoi
from shapely.geometry import Polygon, LineString, Point


def compute_centerline(hallway_polygon: Polygon, sample_spacing: float = 0.05):
    """
    Compute the hallway centerline via Voronoi diagram of boundary samples.
    """
    # Densely sample the polygon boundary
    boundary = hallway_polygon.exterior
    num_samples = int(boundary.length / sample_spacing)
    boundary_points = [
        boundary.interpolate(i / num_samples, normalized=True)
        for i in range(num_samples)
    ]
    coords = np.array([(p.x, p.y) for p in boundary_points])

    # Compute Voronoi diagram
    vor = Voronoi(coords)

    # Filter: keep only edges entirely inside the polygon
    centerline_segments = []
    for ridge_idx, (p1_idx, p2_idx) in enumerate(vor.ridge_vertices):
        if p1_idx < 0 or p2_idx < 0:
            continue  # Skip infinite edges
        v1 = vor.vertices[p1_idx]
        v2 = vor.vertices[p2_idx]
        segment = LineString([v1, v2])
        if hallway_polygon.contains(segment):
            centerline_segments.append(segment)

    return centerline_segments
```

**Step 3 — Identify junction nodes.** Points on the medial axis graph with degree ≥ 3 correspond to hallway intersections.

**Step 4 — Project door midpoints onto the centerline.** For each door gap, compute its midpoint and project perpendicularly onto the nearest centerline segment:

```python
def project_door_to_centerline(
    door_midpoint: Point,
    centerline_segments: list
) -> Point:
    """
    Find the nearest point on the centerline to the door's midpoint.
    This is the doorway node location.
    """
    min_dist = float("inf")
    best_projection = None

    for segment in centerline_segments:
        proj = segment.interpolate(segment.project(door_midpoint))
        dist = door_midpoint.distance(proj)
        if dist < min_dist:
            min_dist = dist
            best_projection = proj

    return best_projection
```

**Step 5 — Assemble and connect nodes.** Place nodes at junction points and door projections, then connect along centerline paths.

### 8.3 Advantages Over SWAGGER

- Higher geometric precision (no rasterization artifacts from 3cm grid)
- Direct semantic awareness (knows which nodes are doorway-aligned vs. intersection nodes)
- Eliminates the occupancy map generation step entirely
- Produces a navigation graph directly from BIM data, strengthening the paper's end-to-end automation claim

### 8.4 Recommended Libraries

| Library | Purpose |
|---------|---------|
| `shapely` | Polygon construction, geometric operations, point projection |
| `scipy.spatial.Voronoi` | Voronoi diagram computation |
| `networkx` | Navigation graph data structure and algorithms |
| `pxr` (USD Python API) | Extract geometry directly from the USD scene |

---

## 9. Validation and Testing

### 9.1 Visual Validation

Overlay the auto-generated graph on the occupancy map and compare against the manually annotated version:

```python
import matplotlib.pyplot as plt
import cv2

def visualize_comparison(omap_path, auto_graph, manual_graph=None):
    omap = cv2.imread(omap_path, cv2.IMREAD_GRAYSCALE)

    fig, axes = plt.subplots(1, 2 if manual_graph else 1, figsize=(16, 8))
    if manual_graph is None:
        axes = [axes]

    # Auto-generated graph
    axes[0].imshow(omap, cmap="gray")
    axes[0].set_title(f"Auto-generated ({auto_graph.number_of_nodes()} nodes)")
    for u, v in auto_graph.edges():
        n1, n2 = auto_graph.nodes[u], auto_graph.nodes[v]
        axes[0].plot(
            [n1["x"] / 0.03, n2["x"] / 0.03],  # Convert meters to pixels
            [n1["y"] / 0.03, n2["y"] / 0.03],
            "c-", linewidth=0.5
        )
    for n in auto_graph.nodes():
        nd = auto_graph.nodes[n]
        axes[0].plot(nd["x"] / 0.03, nd["y"] / 0.03, "ro", markersize=3)

    # Manual graph (if available for comparison)
    if manual_graph:
        axes[1].imshow(omap, cmap="gray")
        axes[1].set_title(f"Manual ({manual_graph.number_of_nodes()} nodes)")
        # ... similar plotting for manual graph ...

    plt.tight_layout()
    plt.savefig("comparison.png", dpi=150)
    plt.show()
```

### 9.2 Quantitative Metrics

| Metric | How to Measure | Target |
|--------|---------------|--------|
| Node count | `graph.number_of_nodes()` | Within ±20% of manual count (79 for floor 4) |
| Centerline deviation | Distance from each node to nearest wall pair midpoint | < 10cm for straight hallways |
| Coverage | % of rooms reachable from the graph | 100% (same as manual) |
| Route efficiency | Dijkstra distance for same start/end pairs | Within ±5% of manual graph routes |
| Door alignment | Distance from door midpoints to nearest graph node | < 50cm for every door |

### 9.3 Integration Test

Confirm the auto-generated graph works end-to-end with the RL training pipeline:

1. Generate the graph for floor 4.
2. Export to JSON.
3. Load into the Isaac Lab environment.
4. Verify the RL agent receives valid waypoints from the graph.
5. Run a short training session to confirm no errors in the waypoint sampling loop.

---

## 10. References

### Primary Sources

1. **NVIDIA SWAGGER** — Hua, R., Okal, B., Butin, B. "Sparse WAypoint Graph Generation for Efficient Routing." NVIDIA Isaac, 2025. [github.com/nvidia-isaac/SWAGGER](https://github.com/nvidia-isaac/SWAGGER)

2. **NVIDIA Isaac Waypoint Graph Generator** — NVIDIA Docs. Consumes semantic or occupancy maps and builds waypoint graphs for route planning. [docs.nvidia.com/isaac/doc/extensions/waypoint_graph_generator](https://docs.nvidia.com/isaac/doc/extensions/waypoint_graph_generator/doc/index.html)

3. **Nav2 SWAGGER Integration** — Nav2 1.0.0 Documentation. Tutorial for using SWAGGER with the Nav2 Route Server. [docs.nav2.org/tutorials/docs/route_server_tools/swagger_generation.html](https://docs.nav2.org/tutorials/docs/route_server_tools/swagger_generation.html)

### Mathematical Foundations

4. **Medial Axis Transform** — Lee, D.T. "Medial Axis Transformation of a Planar Shape." *Computer Vision, Graphics, and Image Processing*, 56(6):462–478, 1982.

5. **Blum, H.** "A transformation for extracting new descriptors of shape." In *Models for the Perception of Speech and Visual Form*, pp. 362–380. MIT Press, 1967.

6. **Voronoi Diagrams** — Aurenhammer, F. "Voronoi diagrams — a survey of a fundamental geometric data structure." *ACM Computing Surveys*, 23(3):345–405, 1991.

### Indoor Navigation Graph Generation

7. **Yang, L. and Worboys, M.** "Generation of navigation graphs for indoor space." *International Journal of Geographical Information Science*, 2015. Comprehensive approach using combinatorial maps and local kernels for node placement in corridors and rooms.

8. **Choset, H. and Burdick, J.** "Sensor based exploration: The Hierarchical Generalized Voronoi Graph." *International Journal of Robotics Research*, 2000.

### Skeletonization Algorithms

9. **scikit-image** — `skimage.morphology.medial_axis` and `skimage.morphology.skeletonize`. [scikit-image.org/docs/stable/auto_examples/edges/plot_skeleton.html](https://scikit-image.org/docs/stable/auto_examples/edges/plot_skeleton.html)

10. **Brandt, J.W. and Algazi, V.R.** "Continuous skeleton computation by Voronoi diagram." *CVGIP: Image Understanding*, 55(3):329–338, 1992.

### Floor Plan Analysis

11. **Kim, Y. et al.** "Indoor Reconstruction from Floorplan Images with a Deep Learning Approach." *ISPRS International Journal of Geo-Information*, 9(2):65, 2020. Centerline and corner detection for converting floor plan images to vector data.

12. **Whiting, E.** "Geometric, Topological & Semantic Analysis of Multi-Building Floor Plan Data." MIT Master's Thesis. Navigation graph construction from labeled DXF floor plans.

### Robot Path Planning

13. **Oleynikova, H. et al.** "Sparse 3D Topological Graphs for Micro-Aerial Vehicle Planning." *IEEE/RSJ IROS*, 2018. Voronoi-based skeleton graph generation with pruning for noisy sensor data.

14. **Blaer, P.** "Robot Path Planning Using Generalized Voronoi Diagrams." Columbia University. Approximation of generalized Voronoi diagrams for safe robot path planning.

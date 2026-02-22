# Route Plan — Indoor Navigation System

## Overview

A node-based graph system for indoor navigation inside the MSOE building (TestBuilding.usd) in Isaac Sim. The system uses Dijkstra's algorithm to find the shortest path between any two locations, and visualizes routes using debug draw lines that are visible to the viewer but invisible to the robot's sensors.

### Architecture

```
building_data.py          graph.py              visualizer.py
(node coordinates    -->  (Graph + Dijkstra  --> (debug draw lines
 and connections)          pathfinding)           in Isaac Sim)
                                |
                                v
                    building_navigation.py
                    (loads building, runs pathfinding,
                     draws routes, records video)
```

### How It Works

1. The MSOE building USD is loaded into Isaac Sim
2. `building_data.py` defines every room, hallway, stairwell as a node with 3D coordinates
3. Edges connect walkable paths between nodes (distance auto-calculated)
4. Dijkstra's algorithm finds the shortest path between any two nodes
5. Debug draw lines render the graph and highlighted route in the viewport
6. The robot receives the waypoint coordinates to follow the route
7. Viewers see the drawn lines; the robot does not — they are a rendering overlay only

---

## Key Concepts

### Nodes

A **node** is a location in the building. It represents a specific point in 3D space — a room, a hallway junction, a stairwell entrance, an elevator, a doorway, etc.

Each node has:
- **id** — unique string identifier (e.g., `"f1_room_101"`)
- **name** — human-readable label (e.g., `"Room 101"`)
- **position** — 3D coordinate as `np.array([x, y, z])` matching the building USD scene
- **floor** — integer floor number (1, 2, 3...)
- **node_type** — classification enum (ROOM, HALLWAY, INTERSECTION, STAIRWELL, ELEVATOR, ENTRANCE, EXIT)

The `node_type` enum is used to:
- Color-code nodes in the visualization (blue for rooms, yellow for hallways, red for stairwells, etc.)
- Filter nodes by category
- Apply different behaviors (e.g., stairwell nodes connect floors)

### Edges

An **edge** is a connection between two nodes. It says "you can walk from here to there" and how far it is.

Each edge has:
- **start_node_id** and **end_node_id** — which two nodes it connects
- **weight** — the cost to travel between them (defaults to Euclidean distance, auto-calculated from node positions)
- **bidirectional** — whether you can go both ways (default: True)

Example: If Node A is at `(0, 0, 0)` and Node B is at `(3, 4, 0)`:
```
Edge weight = sqrt((3-0)^2 + (4-0)^2 + (0-0)^2) = 5.0 meters
```

You don't have to manually type distances — the edge calculates it from the node positions. But you can override it:
- Stairs might have a higher weight (slower to travel)
- An elevator might have extra weight (wait time)
- A blocked path could have infinite weight so Dijkstra's avoids it

### Nodes vs Edges — Summary

```
[Room 101] --- [Hallway Junction] --- [Room 102]
                      |
                  [Stairwell]

Nodes: Room 101, Hallway Junction, Room 102, Stairwell  (the locations)
Edges: the lines connecting them                        (the walkable paths)
```

---

## File Structure

All navigation files live in `isaac-sim-msoe/`:

```
isaac-sim-msoe/
├── navigation/                         # Navigation package
│   ├── __init__.py                     # Package exports
│   ├── graph.py                        # Core data structures + Dijkstra's
│   ├── building_data.py                # MSOE building coordinates and connections
│   └── visualizer.py                   # Debug draw rendering
├── simulations/
│   └── building_navigation.py          # Main Isaac Sim simulation script
├── Routes/
│   └── ROUTE_PLAN.md                   # This file
└── assets/
    └── environments/
        └── msoe_building/
            └── TestBuilding.usd        # MSOE building 3D model
```

---

## File Details

### `navigation/graph.py` — Core Data Structures + Pathfinding

This is the foundation. Everything else depends on it.

**Contains:**

- **`NodeType` enum** — classifies each node
  - `ROOM` — a room (classroom, office, lab)
  - `HALLWAY` — a straight hallway segment
  - `INTERSECTION` — a hallway junction where multiple corridors meet
  - `STAIRWELL` — a stairwell (connects floors)
  - `ELEVATOR` — an elevator (connects floors, higher weight)
  - `ENTRANCE` — a building entrance/exit

- **`Node` class** — a location in the building
  - `id` (str) — unique identifier, prefixed with floor: `"f1_room_101"`, `"f2_stairs_a"`
  - `name` (str) — human-readable label: `"Room 101"`, `"Stairwell A"`
  - `position` (np.array) — 3D coordinate `[x, y, z]` in the building scene
  - `floor` (int) — floor number
  - `node_type` (NodeType) — classification enum
  - This is a data container, no methods

- **`Edge` class** — a walkable connection between two nodes
  - `start_node_id` (str) — source node ID
  - `end_node_id` (str) — destination node ID
  - `weight` (float) — travel cost (auto-calculated Euclidean distance, or manually overridden)
  - `bidirectional` (bool) — can you go both ways (default: True)

- **`Graph` class** — the full navigation graph
  - Stores nodes in a `dict` (id -> Node) and edges in an adjacency list (id -> list of (neighbor_id, weight))
  - `add_node(id, name, position, floor, node_type)` — register a new node
  - `add_edge(start_id, end_id, weight=None)` — connect two nodes; auto-calculates distance if no weight given
  - `get_node(id)` — return a node by ID
  - `get_neighbors(id)` — return list of (neighbor_id, weight) tuples
  - `get_nodes_on_floor(floor)` — filter nodes by floor number
  - `dijkstra(start_id, end_id)` — Dijkstra's algorithm:
    - Uses a priority queue (`heapq`)
    - Starts at start_id with distance 0
    - Explores neighbors, always picking the lowest-cost unvisited node
    - Tracks previous node for path reconstruction
    - Returns `(path, total_distance)` where path is a list of node IDs
    - Returns `(None, float('inf'))` if no path exists
  - `get_path_positions(path)` — converts a list of node IDs to a list of 3D coordinates for the robot to follow

**Why it matters**: This is pure Python — no Isaac Sim dependency. You could test pathfinding without launching a simulation.

---

### `navigation/building_data.py` — MSOE Building Layout

This is the data file — no logic, just the actual layout of the MSOE building defined in code.

**Contains:**

- **`build_msoe_graph()` function** — creates and returns a fully populated `Graph`

**Data organization:**

Organized by floor with clear sections:

```python
def build_msoe_graph():
    graph = Graph()

    # === FLOOR 1 ===

    graph.add_node("f1_entrance", "Main Entrance", np.array([x, y, z]), floor=1, node_type=NodeType.ENTRANCE)
    graph.add_node("f1_lobby", "Lobby", np.array([x, y, z]), floor=1, node_type=NodeType.HALLWAY)
    graph.add_node("f1_room_101", "Room 101", np.array([x, y, z]), floor=1, node_type=NodeType.ROOM)
    graph.add_node("f1_stairs_a", "Stairwell A", np.array([x, y, z]), floor=1, node_type=NodeType.STAIRWELL)

    graph.add_edge("f1_entrance", "f1_lobby")
    graph.add_edge("f1_lobby", "f1_room_101")
    graph.add_edge("f1_lobby", "f1_stairs_a")

    # === FLOOR 2 ===

    graph.add_node("f2_stairs_a", "Stairwell A Floor 2", np.array([x, y, z]), floor=2, node_type=NodeType.STAIRWELL)
    graph.add_node("f2_hall_1", "Second Floor Hall", np.array([x, y, z]), floor=2, node_type=NodeType.HALLWAY)

    # Cross-floor connection (higher weight — stairs are slower)
    graph.add_edge("f1_stairs_a", "f2_stairs_a", weight=10.0)

    return graph
```

**Node naming convention:**
- Prefixed with floor: `f1_`, `f2_`, `f3_`
- Then a descriptive name: `room_101`, `hall_north`, `stairs_a`, `elevator_1`
- Keeps IDs unique across floors and readable

**Coordinates are placeholder initially.** To get the real ones, see the "Gathering Coordinates" section below.

**This file will be the biggest to maintain** — every time you add a room or change the building layout, you update this file. But the rest of the code doesn't change.

---

### `navigation/visualizer.py` — Debug Draw Rendering

Takes the graph data and draws it in Isaac Sim using debug draw lines. No pathfinding logic, no building data — just drawing.

**Depends on:**
```python
from isaacsim.util.debug_draw import _debug_draw
draw = _debug_draw.acquire_debug_draw_interface()
```

Debug draw lines are:
- Visible on screen and in video recordings (the viewer sees the routes)
- NOT physical objects — the robot doesn't collide with them
- NOT detected by robot cameras/sensors — they are a rendering overlay only

**Color scheme (constants):**

| Node Type | Color | RGBA |
|-----------|-------|------|
| ROOM | Blue | (0, 0, 1, 1) |
| HALLWAY | White | (1, 1, 1, 1) |
| INTERSECTION | Yellow | (1, 1, 0, 1) |
| STAIRWELL | Red | (1, 0, 0, 1) |
| ELEVATOR | Purple | (0.5, 0, 1, 1) |
| ENTRANCE | Green | (0, 1, 0, 1) |

**Contains:**

- **`draw_nodes(graph, floor=None)`**
  - Draws every node as a colored point at its position
  - Color determined by node_type using the color scheme above
  - Larger points for important nodes (entrances, stairwells), smaller for hallways
  - If `floor` is specified, only draws nodes on that floor
  - Useful for checking that coordinates are correct before connecting edges

- **`draw_edges(graph, floor=None)`**
  - Draws every edge as a thin gray line from start node to end node
  - If `floor` is specified, only draws edges where both nodes are on that floor (plus cross-floor edges)
  - Gray and thin so it looks like a faint background map

- **`draw_graph(graph, floor=None)`**
  - Convenience function — calls `draw_edges()` then `draw_nodes()` on top
  - Draws the full building map

- **`draw_path(graph, path)`**
  - Draws a Dijkstra result as a highlighted route
  - Takes the list of node IDs from `dijkstra()` (e.g., `["f1_entrance", "f1_lobby", "f1_room_101"]`)
  - Thick bright green lines connecting each node in the path
  - Large green point at the start node
  - Large red point at the destination node
  - Drawn ON TOP of the gray graph, so the route visually pops out

- **`clear_all()`**
  - Clears all debug draw lines and points
  - Useful for recalculating and redrawing without old visuals

**Why it's separate from graph.py:** The graph and Dijkstra's are pure Python (testable anywhere). The visualizer is Isaac Sim specific. Keeping them separate means you can test pathfinding without launching a simulation.

---

### `navigation/__init__.py` — Package Exports

Makes the `navigation/` folder importable as a Python package and provides clean imports.

**Contains:**
```python
from .graph import Graph, Node, Edge, NodeType
from .building_data import build_msoe_graph
from .visualizer import draw_graph, draw_path, draw_nodes, clear_all
```

**Effect:** Instead of writing:
```python
from navigation.graph import Graph, NodeType
from navigation.building_data import build_msoe_graph
from navigation.visualizer import draw_graph, draw_path
```

You can write:
```python
from navigation import Graph, build_msoe_graph, draw_graph, draw_path
```

---

### `simulations/building_navigation.py` — Main Simulation Script

The script you actually run. Ties everything together.

**Step by step:**

1. **Isaac Sim setup** — launches `SimulationApp` in headless mode, imports from `navigation` package and `sim_utils`

2. **Load the MSOE building** — loads `assets/environments/msoe_building/TestBuilding.usd` into the scene (local file, not from NVIDIA cloud). Adds distant light and dome light for illumination.

3. **Build the graph** — calls `build_msoe_graph()` to get the full node graph with all rooms, hallways, and connections. One line of code.

4. **Run Dijkstra's** — picks a start and end node, runs pathfinding:
   ```python
   path, distance = graph.dijkstra("f1_entrance", "f1_room_101")
   ```
   Could eventually accept start/end as command line arguments.

5. **Draw the visualization**:
   - `draw_graph(graph, floor=1)` — renders the full floor plan in gray with colored node dots
   - `draw_path(graph, path)` — highlights the shortest route in green on top

6. **Set up camera** — positioned high up looking down at the floor plan (top-down or angled view) so you can see the whole floor and all the drawn lines.

7. **Simulation loop + video recording** — uses `VideoRecorder` to capture to `videos/building_navigation.mp4`. Runs for enough frames to capture the visualization.

8. **Cleanup** — `my_world.stop()`, `simulation_app.close()`

**What the output video shows:**
- The MSOE building from above
- Gray lines showing every walkable connection (the full graph)
- Colored dots at every node (blue rooms, yellow hallways, red stairwells, green entrances)
- Bright green line showing the shortest path from start to destination
- Green dot at start, red dot at destination

**Running it:**
```bash
cd ~/isaac-sim-msoe
sbatch submit_isaac_job.sh simulations/building_navigation.py
```

---

## Gathering Coordinates

The coordinates in `building_data.py` need to match the actual positions in `TestBuilding.usd`. To get them:

1. Launch Isaac Sim in streaming mode on a GPU node:
   ```bash
   srun --partition=teaching --gres=gpu:t4:1 --cpus-per-gpu=4 --time=02:00:00 --pty bash
   cd ~/isaacsim
   bash isaac-sim.streaming.sh
   ```

2. Connect via WebRTC Streaming Client from your local machine

3. Load the building: **File > Open** and navigate to `TestBuilding.usd`

4. For each location you want as a node:
   - Navigate to that spot in the viewport
   - Either click on the floor/wall nearby or create a temporary cube (**Create > Mesh > Cube**) and drag it to the position
   - Read the X, Y, Z values from the **Property > Transform** panel on the right
   - Record the coordinates

5. Enter the coordinates in `building_data.py` for each node

6. Run the simulation to verify the nodes appear in the correct locations (use `draw_nodes()` first before adding edges)

---

## Multi-Floor Support

Each floor has a different z-height. Floors are connected through stairwell and elevator nodes.

- Floor 1 nodes: z = 0 (ground level)
- Floor 2 nodes: z = 4 (or whatever the actual floor height is in the USD)
- Stairwell edges: `f1_stairs_a` <-> `f2_stairs_a` with a higher weight (stairs are slow)
- Elevator edges: `f1_elevator_1` <-> `f2_elevator_1` with an even higher weight (wait time)

Dijkstra's algorithm handles this naturally — if the shortest path requires going upstairs, it will include the stairwell nodes in the path. The higher weight on stair/elevator edges means Dijkstra's prefers staying on the same floor when possible.

---

## Future Additions

- **Robot following the path** — spawn a robot at the start node and have it walk/drive along the waypoints from `get_path_positions(path)`
- **Follow camera** — attach the camera to the robot as it navigates (same pattern as `anymal_walk.py`)
- **A* algorithm** — swap Dijkstra's for A* using straight-line distance as a heuristic (faster for large graphs)
- **Dynamic obstacles** — mark certain edges as blocked at runtime and recalculate
- **Command line arguments** — pass start/end nodes as arguments: `python building_navigation.py f1_entrance f2_room_201`
- **Interactive mode** — in the streaming GUI, click on nodes to set start/end and see the path update live

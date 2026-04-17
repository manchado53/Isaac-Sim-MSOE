# Running Navigation Tasks in Isaac Sim

## What Exists

| File | Purpose |
|------|---------|
| `assets/environments/msoe_building/TestBuilding.usd` | The MSOE building 3D scene |
| `navigation/routes.json` | 18 solved tasks with waypoint sequences |
| `navigation/visualizer.py` | Debug draw renderer (nodes, edges, paths) — runs inside Isaac Sim |
| `simulations/building_navigation.py` | Working Isaac Sim nav script — currently wired to the NVIDIA hospital |
| `simulations/jetbot_hospital_demo.py` | Working robot-driving script (Nova Carter) — also on hospital |

---

## The Blocker: Coordinate Mismatch

`floor1_nodes.json` stores node positions as **pixel coordinates** in the floor plan image (e.g., `x=412, y=187`).

Isaac Sim needs **3D world coordinates** in the USD scene (meters, matching `TestBuilding.usd`).

These two coordinate systems are not the same — the pixel-space graph cannot be passed directly to the simulator without a calibration step.

---

## Step 1: Calibrate Pixel → World Coordinates

Find the affine transform between the two spaces using 2–3 known landmark points.

**How to get the landmarks:**
1. Open `TestBuilding.usd` in Isaac Sim (interactive session on Rosie)
2. Navigate to 2–3 identifiable corners or features (e.g., building entrance, a corridor endpoint)
3. Read their `(X, Y, Z)` world coordinates from the **Property > Transform** panel
4. Note the corresponding pixel `(x, y)` coordinates of those same points in `floor_plans_resized/floor1/floor1.jpg`

**Then compute the transform:**
```python
import numpy as np

# Example: two known correspondences
# pixel_pts[i] -> world_pts[i]
pixel_pts = np.array([[px0, py0], [px1, py1], [px2, py2]], dtype=float)
world_pts = np.array([[wx0, wy0], [wx1, wy1], [wx2, wy2]], dtype=float)

# Solve: world = A @ pixel + b  (least-squares affine fit)
A_aug = np.hstack([pixel_pts, np.ones((len(pixel_pts), 1))])
params_x, _, _, _ = np.linalg.lstsq(A_aug, world_pts[:, 0], rcond=None)
params_y, _, _, _ = np.linalg.lstsq(A_aug, world_pts[:, 1], rcond=None)

def pixel_to_world(px, py, z=0.0):
    x = params_x[0]*px + params_x[1]*py + params_x[2]
    y = params_y[0]*px + params_y[1]*py + params_y[2]
    return np.array([x, y, z])
```

Alternatively, `test_building/generate_occupancy_map.py` already knows the world-space bounding box of the building (`xy_min`, `cell_size`). If the pixel coordinates in `floor1_nodes.json` align with that occupancy grid, the conversion is:

```python
# From pipeline.md Stage 2
x_world = xy_min[0] + col * cell_size + cell_size / 2
y_world = xy_min[1] + row * cell_size + cell_size / 2
```

Check whether the node pixel coords match the occupancy grid dimensions — if so, this is the simpler path.

---

## Step 2: Build the Simulation Script

Create `simulations/msoe_building_navigation.py` modeled on `simulations/building_navigation.py`:

```python
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

import json, os, sys
import numpy as np
from isaacsim.core.api import World
from isaacsim.core.utils.stage import add_reference_to_stage

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from navigation.graph import Graph, NodeType
from navigation.visualizer import draw_graph, draw_path
from sim_utils.video_capture import VideoRecorder

# --- Load building ---
BUILDING_USD = os.path.join(os.path.dirname(__file__), "..", "assets/environments/msoe_building/TestBuilding.usd")
world = World(stage_units_in_meters=1.0)
add_reference_to_stage(BUILDING_USD, prim_path="/World/Building")

# --- Load graph and apply coordinate transform ---
NODES_FILE = "floor_plans_resized/floor1/floor1_nodes.json"
with open(NODES_FILE) as f:
    data = json.load(f)

TYPE_MAP = {"room": NodeType.ROOM, "hallway": NodeType.HALLWAY,
            "intersection": NodeType.INTERSECTION, "stairwell": NodeType.STAIRWELL,
            "elevator": NodeType.ELEVATOR, "entrance": NodeType.ENTRANCE}

graph = Graph()
for node in data["nodes"]:
    world_pos = pixel_to_world(node["x"], node["y"])  # apply calibrated transform
    graph.add_node(node["id"], node["name"], world_pos, floor=1,
                   node_type=TYPE_MAP.get(node["type"], NodeType.HALLWAY))
for edge in data["edges"]:
    graph.add_edge(edge[0], edge[1])

# --- Pick a task ---
ROUTES_FILE = "navigation/routes.json"
with open(ROUTES_FILE) as f:
    routes = json.load(f)

task = next(t for t in routes["tasks"] if t["id"] == "hard_01")  # change as needed
path = task["waypoints"]

# --- Run simulation ---
world.reset()
draw_graph(graph, floor=1)
draw_path(graph, path)

with VideoRecorder(f"msoe_{task['id']}") as recorder:
    for _ in range(300):
        world.step(render=True)
        recorder.capture()

world.stop()
simulation_app.close()
```

---

## Step 3: Optionally Drive a Robot

To have a physical robot follow the route (like `jetbot_hospital_demo.py`), add a Nova Carter after loading the building:

```python
from isaacsim.robot.wheeled_robots.robots import WheeledRobot
from isaacsim.robot.wheeled_robots.controllers.differential_controller import DifferentialController

carter = world.scene.add(WheeledRobot(
    prim_path="/World/NovaCarter",
    name="carter",
    wheel_dof_names=["joint_wheel_left", "joint_wheel_right"],
    create_robot=True,
    usd_path=assets_root_path + "/Isaac/Robots/NVIDIA/NovaCarter/nova_carter.usd",
    position=graph.get_node(path[0]).position,  # start at first waypoint
))
controller = DifferentialController(name="ctrl", wheel_radius=0.14, wheel_base=0.4)
```

Then in the simulation loop, implement waypoint following: drive toward `path[i]`, advance to `path[i+1]` when within a threshold distance.

---

## Submitting to Rosie

Once the script is ready:

```bash
cd ~/isaacsim/isaac-sim-msoe
sbatch submit_isaac_job.sh simulations/msoe_building_navigation.py
```

Monitor:
```bash
squeue -u $USER
cat logs/isaac_<JOBID>.out
```

Output video will be saved to `videos/msoe_<task_id>.mp4`.

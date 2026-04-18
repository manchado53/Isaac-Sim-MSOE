# Nova Carter + Nav2 — Innovation Center Navigation

Everything you need to drive a Nova Carter around MSOE's Innovation Center in Isaac Sim lives in this directory. This is the runtime entry point of the navigation research pipeline.

## What's in here

| Path | Purpose |
|---|---|
| `nav2_orchestrator.py` | Boots Isaac Sim, loads the building USD, spawns the Nova Carter, enables its ROS2 bridge, and hands off to Nav2 running in a separate process. Runs directly via `~/isaacsim/python.sh`, **not** via `ros2 run`. |
| `launch/kern_navigation.launch.py` | Nav2 bringup — includes `nav2_bringup/bringup_launch.py` with our map/params. |
| `launch/kern_navigation_isaacsim.launch.py` | Alternative bringup that also launches Isaac Sim via `run_isaacsim.launch.py`. Not used in practice; the orchestrator is the preferred entry point. |
| `params/kern_navigation_params.yaml` | Every Nav2 parameter — AMCL, costmaps, planner, controller, behavior tree, collision monitor. |
| `rviz2/kern_navigation.rviz` | Pre-configured RViz layout: map, costmaps, plan, robot model, scans, AMCL particles. |
| `maps/` | Deliberately empty — the actual map YAML lives in `../assets/environments/InnovationCenter/navgraphs/`. Retained because `CMakeLists.txt` installs the directory. |
| `CMakeLists.txt`, `package.xml` | Colcon ROS2 package metadata. |

## Prerequisites

- Isaac Sim 5.1 installed at `~/isaacsim`
- ROS2 Jazzy sourced (`source /opt/ros/jazzy/setup.bash`)
- The `kern_navigation` package symlinked into a colcon workspace at `~/IsaacSim-ros_workspaces/jazzy_ws/src/navigation/kern_navigation` and built once with `colcon build --packages-select kern_navigation --symlink-install`. After that, YAML/launch edits apply without rebuilding.
- The target floor's occupancy map and navgraph exist under `../assets/environments/InnovationCenter/{maps,navgraphs}/`. See [`floorplan_vision/README.md`](../floorplan_vision/README.md) if you need to generate them for a new building.

## Quick start — drive the robot in three terminals

Three terminals. One for Isaac Sim, one for Nav2, one for goals (or use Foxglove's click-to-publish instead of the third terminal).

**Terminal 1 — Isaac Sim + ROS2 bridge + robot spawn:**

```bash
cd ~/isaac-sim-msoe
export OMNI_KIT_ACCEPT_EULA=YES PRIVACY_CONSENT=Y PYTHONUNBUFFERED=1
~/isaacsim/python.sh navigation/nav2_orchestrator.py --floor 4 --stream --stream-ip 100.107.199.57
```

Wait for `[orchestrator] Scene ready. Robot spawned with ROS2 bridge active.` before continuing.

**Terminal 2 — Nav2 stack:**

```bash
source /opt/ros/jazzy/setup.bash
source ~/IsaacSim-ros_workspaces/jazzy_ws/install/setup.bash
ros2 launch kern_navigation kern_navigation.launch.py
```

Wait for both `lifecycle_manager_localization` and `lifecycle_manager_navigation` to log `Managed nodes are active`.

**Terminal 3 — send a goal:**

```bash
source /opt/ros/jazzy/setup.bash
source ~/IsaacSim-ros_workspaces/jazzy_ws/install/setup.bash
ros2 action send_goal /navigate_to_pose nav2_msgs/action/NavigateToPose \
  "{pose: {header: {frame_id: map}, pose: {position: {x: -7.30, y: 78.09, z: 0.0}, orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}}}}"
```

Expected result: `Goal finished with status: SUCCEEDED` and the robot drives ~30 m along the floor-4 main corridor.

## Orchestrator CLI reference

All flags optional unless noted:

| Flag | Default | Meaning |
|---|---|---|
| `--building` | `InnovationCenter` | Directory name under `assets/environments/` |
| `--floor` | `4` | Floor number to spawn on |
| `--node-id` | `None` | Spawn at this navgraph node instead of the default spawn x/y from `building_info.json` |
| `--yaw` | `None` | Override spawn yaw (radians) |
| `--headless` | `False` | Run Isaac Sim completely headless (no GUI, no stream). Fastest startup; use when you just need the ROS topics. |
| `--stream` | `False` | Enable WebRTC livestream. Connect from the Isaac Sim Streaming Client to `--stream-ip`. |
| `--stream-ip` | `100.107.199.57` | Public endpoint advertised to remote WebRTC clients. Set to whichever address your streaming client can reach. |
| `--obstacle X,Y[,SIZE]` | (empty) | Spawn a kinematic red cube at world `(X, Y)` with optional side length in meters (default 0.6). Repeatable. Cubes live under `/World/Obstacles` so they are invisible to the static occupancy map — Nav2 only learns about them via the live 3D lidar → obstacle_layer path. Perfect for exercising live avoidance. |

**Gotcha**: argparse treats `-N` as a flag. When `X` is negative, use `=` syntax:

```bash
--obstacle=-24.25,81.69
```

## Architecture in one diagram

```
  Isaac Sim (headless or stream)
  ┌──────────────────────────────────────┐
  │  nav2_orchestrator.py                │
  │    - loads InnovationCenter.usd      │
  │    - spawns Nova_Carter_ROS.usd      │
  │    - enables isaacsim.ros2.bridge    │
  │    - (optional) spawns /World/Obstacles/obstacle_N cubes
  │                                      │
  │  Physics tick                        │
  │    ├── /chassis/odom     (nav_msgs/Odometry, 60 Hz)
  │    ├── /tf               (robot body frames)
  │    ├── /clock            (sim_time)
  │    ├── /front_3d_lidar/lidar_points   (Hesai 360° 3D PointCloud2)
  │    ├── /front_2d_lidar/scan            (180° front LaserScan)
  │    ├── /back_2d_lidar/scan             (180° rear LaserScan)
  │    └── /cmd_vel ← subscribes            (geometry_msgs/Twist)
  └──────────────────────────────────────┘
         ↓  /front_3d_lidar/lidar_points
  ┌──────────────────────────────────────┐
  │  pointcloud_to_laserscan             │
  │    projects the 3D cloud to /scan    │
  └──────────────────────────────────────┘
         ↓  /scan
  ┌──────────────────────────────────────┐
  │  Nav2 (ros2 launch kern_navigation)  │
  │    ├── map_server      (/map)        │
  │    ├── amcl            (map → odom)  │
  │    ├── global_costmap                │
  │    │     ├── static_layer   (the PNG)
  │    │     ├── obstacle_layer (from /scan, 15 m range)
  │    │     └── inflation_layer (radius 1.0 m)
  │    ├── local_costmap                 │
  │    │     ├── static_map_layer        │
  │    │     ├── front_3d_lidar_layer (VoxelLayer, 8 m range)
  │    │     ├── front_2d_lidar_layer    │
  │    │     ├── back_2d_lidar_layer     │
  │    │     └── inflation_layer (0.8 m) │
  │    ├── planner_server (SmacPlanner2D)│
  │    ├── controller_server (DWB)       │
  │    ├── behavior_server (spin, backup, drive_on_heading, wait)
  │    ├── collision_monitor             │
  │    ├── velocity_smoother             │
  │    └── bt_navigator                  │
  └──────────────────────────────────────┘
         ↓  /cmd_vel
  back to Isaac Sim's Nova Carter wheel controller
```

## Visualization

**Primary path: Foxglove Studio.** No X11, no install on Windows/macOS, works over any network you already have.

```bash
# On the sim host, alongside Nav2:
source /opt/ros/jazzy/setup.bash
source ~/IsaacSim-ros_workspaces/jazzy_ws/install/setup.bash
ros2 run foxglove_bridge foxglove_bridge --ros-args -p port:=8765
```

Then from your laptop, open https://app.foxglove.dev (web) or the desktop app → **Open Connection → Foxglove WebSocket → `ws://<sim-host>:8765`**. Add a **3D** panel and enable layers for `/map`, `/global_costmap/costmap`, `/local_costmap/costmap`, `/plan`, `/front_3d_lidar/lidar_points`, `/scan`. Use the **Publish Goal** tool on the 3D panel to click-to-send `NavigateToPose` goals directly on the map.

**Fallback: RViz2 via SSH X11 forwarding.** Works well on native Linux. On macOS you need XQuartz; on Windows you need VcXsrv + `$env:DISPLAY = "localhost:0.0"` before `ssh -Y`. Once your local X server is running:

```bash
ssh -Y user@sim-host
cd ~/isaac-sim-msoe
source /opt/ros/jazzy/setup.bash
source ~/IsaacSim-ros_workspaces/jazzy_ws/install/setup.bash
rviz2 -d navigation/rviz2/kern_navigation.rviz
```

If RViz crashes with a GL error, prepend `LIBGL_ALWAYS_INDIRECT=1`. Expect 2–10 fps over remote X — usable for debugging, not smooth.

## Coordinate system

The occupancy map stored in `assets/environments/InnovationCenter/maps/floor_N_occupancy.png` follows the **ROS `map_server` convention**:

- **Row 0 of the PNG = world y_max** (top of the image is the highest world Y)
- **Col 0 of the PNG = world x_min**
- The `origin` field in `floor_N_map_metadata.yaml` is the world pose of the **bottom-left** pixel (`row = H-1, col = 0`)

This matters because `nav2_map_server::loadMapFromFile` internally reverses the image rows (`map_io.cpp:326` — "ROS expects origin at bottom-left"). If the PNG is saved with row 0 = y_min (natural visual orientation) Nav2 publishes a Y-mirrored static map and AMCL can't localize. The fix is a `np.flipud` at save time in `generate_occupancy_map.py::save_floor`. See [`../floorplan_vision/README.md`](../floorplan_vision/README.md) for the full write-up.

## Troubleshooting

### `Goal finished with status: ABORTED` + `error_code: 205` (`START_OCCUPIED`)

The planner refuses to START from the robot's current pose — it's not a "couldn't find a path" error. A* bailed at step 0 because the cell (or any cell under the padded footprint) has cost ≥ 253 (`INSCRIBED_INFLATED_OBSTACLE`). Causes, in frequency order:

1. **Robot's center is within `inscribed_radius + footprint_padding` of a wall.** With `footprint_padding: 0.05` and `inscribed_radius ≈ 0.14`, the robot's circumscribed circle can't sit anywhere inside ~0.55 m of a wall. Corridor-width math: minimum wall-to-wall distance where a path exists is `2 × (inscribed + padding + inflation)`.
2. **Live lidar is marking ghost cells right under the robot** — usually the 3D lidar picking up the robot's own body or a reflection. Check `/local_costmap/costmap` in Foxglove/RViz; if you see red cells directly under the robot with no corresponding static geometry, it's a lidar projection issue.
3. **Stale obstacle_layer marks that never got cleared** — the `clearing: True` setting on the global obstacle layer only clears as new scans arrive. If `/tf` stops flowing (see next section), marks stay.

**Fix**: either back the robot away from the wall (send a goal to a known-good point, or wiggle with `/cmd_vel`), or restart the orchestrator for a clean respawn.

### Robot isn't moving even though Nav2 says goal accepted

`echo /cmd_vel_smoothed` — if it's zero, the controller's idle. If it's non-zero but the robot isn't moving, Isaac Sim is alive but not consuming `/cmd_vel`. Check:

```bash
ros2 topic hz /tf /chassis/odom /clock /scan
```

If any of these is 0 Hz, the Isaac Sim ROS2 bridge has gone silent. Common causes:

1. **Someone modified the scene in the WebRTC streaming client** — adding a prim via the Kit UI can invalidate the PhysX tensor view that Nova Carter's wheel controller uses. Log spam: `omni.physx.tensors.plugin: Simulation view object is invalidated`. **Only fix is a full orchestrator restart**, followed by `find /dev/shm -name "fastrtps*" -delete`.
2. **DDS shared memory state is corrupted** — usually after an unclean kill. The orchestrator stays alive but its ROS2 bridge participants can't talk to new subscribers. Same fix as above.
3. **Two `nav2_container` processes racing** on node names — can happen if a Nav2 launch was killed mid-bringup and a new one started. Check with `pgrep -af component_container_isolated`. Kill both, wipe DDS shm, relaunch Nav2 only (orchestrator can stay if it's still publishing topics).

### Emergency stop isn't firing before collisions

The collision monitor's `FootprintApproach` polygon in `kern_navigation_params.yaml` needs `min_points: 4`. Earlier versions of this file (and the NVIDIA `carter_navigation` upstream it was copied from) ship with `min_points: 6000000000`, which is effectively "never trigger". Confirm your checked-out copy has the sane value.

### Goal accepted, robot drives right through an obstacle

Three tuning knobs — all in `params/kern_navigation_params.yaml`:

1. `DWB.FollowPath.BaseObstacle.scale` — if this is `0.02` (NVIDIA default), DWB barely cares about obstacle cells. Raise to `15.0`.
2. `global_costmap.obstacle_layer.scan.obstacle_max_range` — if obstacles are outside this range at the time the plan is computed, the planner doesn't know they exist. Our default is `15.0` m.
3. `SmacPlanner2D.cost_travel_multiplier` — how much A* weights the cell cost when scoring paths. We use `4.0`; `2.0` is too low.

### `failed to send response to /nav2_container/_container/load_node (timeout)` during Nav2 bringup

DDS state from a previous session is stale. Kill everything:

```bash
pkill -9 -f nav2_orchestrator
pkill -9 -f component_container_isolated
pkill -9 -f "ros2 launch kern"
pkill -9 -f pointcloud_to_laserscan
pkill -9 -f "kit/python"
find /dev/shm -name "fastrtps*" -delete
find /dev/shm -name "sem.fastrtps*" -delete
```

Then relaunch the orchestrator, wait for Scene ready, **then** launch Nav2.

### Paths hug walls

You want paths with more clearance from walls, not just "not in collision". Nav2 doesn't prefer clearance by default — you have to configure it. Knobs (all in `params/kern_navigation_params.yaml`):

- `global_costmap.inflation_layer.inflation_radius` — how far the cost gradient extends. Larger = more "pressure" to stay away. We use `1.0`.
- `global_costmap.inflation_layer.cost_scaling_factor` — how fast the cost decays. Smaller = slower decay = cells far from walls still expensive. We use `0.1`.
- `SmacPlanner2D.cost_travel_multiplier` — how much A* weights those gradient costs. We use `4.0`.

Trade-off: stronger clearance means the planner can refuse to squeeze through narrow gaps (e.g., past a 0.6 m cube in a 3 m corridor). Tune based on your environment.

### Windows OpenSSH: `X11 forwarding requested but DISPLAY not set`

Windows has no native X server. Install VcXsrv, launch XLaunch (Multiple windows / Start no client / Disable access control ✓), then **before** `ssh`:

```powershell
$env:DISPLAY = "localhost:0.0"
ssh -Y user@sim-host
```

To make `DISPLAY` permanent: Windows Settings → System → About → Advanced system settings → Environment Variables → New user variable `DISPLAY` = `localhost:0.0`.

### Robot drives off the plan into open space and gets stuck

DWB's `footprint_padding` is too small relative to the global planner's. The controller is letting the robot drift into poses the planner will later reject. Increase `local_costmap.footprint_padding` above `global_costmap.footprint_padding` so DWB's collision check is strictly more conservative than the planner's.

## Contributing / extending

- **Adding a new building**: see `../floorplan_vision/README.md` for the scan → occupancy-map → navgraph pipeline. You'll also need to add a new entry to `../assets/environments/<NewBuilding>/building_info.json` and put the USD at `../assets/environments/<NewBuilding>/<NewBuilding>.usd`.
- **Changing Nav2 params**: edit `params/kern_navigation_params.yaml` directly. Because the source is symlinked into the colcon install tree, no rebuild is needed — just restart Nav2.
- **Adding a new orchestration mode** (e.g., multi-robot, scenario runner): add a new script to this directory. Follow the same pattern: `argparse` before `SimulationApp(...)`, then Isaac Sim API imports only after, then scene setup, then `while simulation_app.is_running(): world.step(render=True)`.

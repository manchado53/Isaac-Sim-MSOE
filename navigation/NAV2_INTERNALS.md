# Nav2 internals — the mental model

Deep dive into how the Nav2 stack we run for the Innovation Center demo
actually works end-to-end, and which knob affects which symptom.

Target audience: anyone who's read `README.md`, fired a goal, and now
wonders why the robot is wandering / stuck / refusing / slow.

---

## 1. The pipeline

```
                ┌──────────────────┐
                │    Isaac Sim     │
                │  Nova_Carter_ROS │
                └──────────────────┘
                  ┃   ┃   ┃    ▲
     /chassis/odom┃   ┃   ┃    ┃ /cmd_vel
          /tf ────┨   ┃   ┃    ┃
     /scan  ──────┨   ┃   ┃    ┃
  /lidar_points ──┛   ┃   ┃    ┃
                      ▼   ▼    ┃
         ┌────────────────────────────────┐
         │          Nav2 stack            │
         │                                │
         │  ①  AMCL (localization)        │  map↔odom fusion
         │       └─► pose in map frame    │
         │                                │
         │  ②  global_costmap             │  where walls are
         │       • static_layer  (map)    │
         │       • obstacle_layer (lidar) │
         │       • inflation_layer        │
         │                                │
         │  ③  planner_server             │  SmacPlanner2D (A*)
         │       path = A*(start→goal)    │
         │                                │
         │  ④  bt_navigator               │  behavior tree
         │       loops: plan→follow       │  fires recoveries
         │       calls recovery subtree   │  on failure
         │                                │
         │  ⑤  local_costmap              │  short-horizon obstacles
         │       + inflation_layer        │
         │                                │
         │  ⑥  controller_server (DWB)    │  sample & score
         │       trajectories → cmd_vel   │  trajectories
         │                                │
         │  ⑦  velocity_smoother          │  ramp cmd_vel
         │  ⑧  collision_monitor          │  emergency stop
         │                                │
         └────────────────────────────────┘
                      ┃
                      ▼
            /cmd_vel_smoothed → Isaac Sim
```

Each stage has its own ROS lifecycle state (inactive / active / shutdown)
and its own params block in `kern_navigation_params.yaml`.

---

## 2. The costmap — ground truth for "can I walk here?"

Every 5 cm × 5 cm pixel gets one of these cost tiers:

```
Cost value   What it means              Who produced it
──────────   ─────────────────────────  ────────────────────────────────
    254      LETHAL_OBSTACLE            static_layer (wall in PNG)
                                        OR obstacle_layer (lidar return)

    253      INSCRIBED_INFLATED         within  inscribed_radius  of a wall
             OBSTACLE                   (robot definitely collides)

    1 … 252  INFLATION band             within  inflation_radius   of a wall
             (soft "ugh" cost)          cost falls off exponentially

      0      FREE                       beyond inflation_radius, no obstacles

    255      NO_INFORMATION             cell hasn't been observed
             (a.k.a. UNKNOWN)           (only used if allow_unknown: false)
```

### The inflation curve

Given an obstacle cell, every neighbor within `inflation_radius` gets a
non-zero cost that decays exponentially:

```
cost(d) = 252 × exp(−cost_scaling_factor × (d − inscribed_radius))

where d = distance from cell to the nearest obstacle
```

Visual of that formula for two different `cost_scaling_factor` values,
with `inflation_radius = 0.7 m`, `inscribed_radius ≈ 0.14 m`:

```
  cost
   ▲
252┤█
   │█
   │██
   │██▄                  cost_scaling_factor = 0.1
   │████▄▄▄▄▄▄▄▄▄▄▄       (slow decay — wide pressure band)
   │██████████████
   │██████████████▄▄▄▄
 47┤██████████████████▄▄▄▄▄
 20┤██                     ▀▀▀▀
 ─┴──────────────────────────────── distance
 0.14m                        0.7m


  cost
   ▲
252┤█
   │█
   │█                     cost_scaling_factor = 3.0
   │█▄                    (fast decay — narrow pressure band)
   │██▄▄
 47┤██████▄▄▄▄▄▄
 20┤            ▀▀▀▀▀▀▀▄▄▄▄
 ─┴──────────────────────────────── distance
 0.14m                        0.7m
```

**Why it matters**: a 1.5 m-wide corridor has every pixel within 0.75 m
of a wall. Low `cost_scaling_factor` makes the whole corridor high-cost
(planner struggles). High `cost_scaling_factor` leaves a low-cost spine
down the middle (planner sails through).

### Two separate costmaps

```
global_costmap             local_costmap
────────────────────────   ─────────────────────────
Large — covers whole       Small rolling window
floor (2053 × 2007 px)     (e.g. 5 m × 5 m) around
at 0.05 m/px               the robot

Used by the PLANNER        Used by the CONTROLLER
to compute long paths      to score short trajectories

Updated from:              Updated from:
  static_layer  (PNG)        static_map_layer
  obstacle_layer (lidar)     front_2d_lidar_layer
  inflation_layer            back_2d_lidar_layer
                             front_3d_lidar_layer (VoxelLayer)
                             inflation_layer
```

They have **independent** `inflation_radius`, `cost_scaling_factor`,
`footprint_padding`. Global can be aggressive (longer paths); local is
usually tighter (short-horizon only).

---

## 3. Planner — `SmacPlanner2D`

```
  start (robot pose)              goal (TSP waypoint)
       ●───────────────────────────────●
              ↓  runs A* on global_costmap
       ●──●──●──●──●──●──●──●──●──●──●
     path = sequence of (x,y) waypoints
```

Each cell's **edge cost** in A* is:

```
edge_cost = base_edge_length + cost_travel_multiplier × cell_cost/252
```

- `cost_travel_multiplier = 1.0` → A* mostly goes the shortest way
- `cost_travel_multiplier = 4.0` → A* strongly prefers low-cost
  (centerline-hugging) paths

**Refusal modes**:

| Error | Cause |
|---|---|
| `Start occupied` | robot's current pose cell has cost ≥ 253 |
| `Goal occupied` | goal's pose cell has cost ≥ 253 |
| `No valid path found` | no cell-chain from start to goal with every cell < 253 |

The `tolerance: 0.5` knob (under SmacPlanner2D) lets it return a path
that **ends within 0.5 m of the goal** instead of exactly at the goal
— rescues many `Goal occupied` cases.

---

## 4. Controller — `DWB` (Dynamic Window Approach B)

Once there's a path, DWB runs a scoring loop at 20 Hz:

```
every 50 ms:

  1. sample a fan of candidate trajectories:
        for vx  in [min_vel_x … max_vel_x]  (vx_samples points)
        for vθ in [-max_vel_θ … max_vel_θ] (vtheta_samples points)
            simulate sim_time seconds of motion  (every step is a pose)

  2. score each candidate by a weighted sum of critics

  3. pick the best-scoring trajectory → publish its first cmd_vel
```

### The critics

Each critic returns a number; DWB multiplies by that critic's `scale`
and sums. Low total = good trajectory.

```
BaseObstacle.scale  — penalize trajectories whose footprints
                       cover high-cost cells (hug walls → painful)

PathAlign.scale     — penalize heading mismatch with the plan

PathDist.scale      — penalize distance from the plan (both poses)

GoalDist.scale      — reward progress toward the final goal

GoalAlign.scale     — reward facing the goal

Oscillation.scale   — penalize back-and-forth jitter
```

**Interaction you must understand**: if `cost_scaling_factor` is low and
`BaseObstacle.scale` is high, DWB picks slow, wall-hugging-averse
trajectories *everywhere*, not just near walls. That's what made our
robot cruise at 0.44 m/s in a 1.5 m corridor.

### Goal checker

Independent of DWB — decides when a pose is "reached":

```
if ‖robot_xy − goal_xy‖ < xy_goal_tolerance
   AND |robot_yaw − goal_yaw| < yaw_goal_tolerance:
    mark pose as REACHED
```

`stateful: true` means once reached, stay reached even if the robot
drifts away — important for a multi-pose tour.

---

## 5. Behavior tree — the state machine

For `NavigateThroughPoses`, Nav2 runs
`navigate_through_poses_w_replanning_and_recovery.xml`. Simplified:

```
┌─ PipelineSequence ──────────────────────────────────────┐
│                                                         │
│  ComputePathThroughPoses         ← planner server call  │
│  RemovePassedGoals               ← prune poses within   │
│                                     viz_radius (~0.5 m) │
│  FollowPath                      ← controller server    │
│                                                         │
│  IF any step fails:                                     │
│    ┌─ Recovery fallback ───────────────────────────┐    │
│    │ ClearEntireCostmap (local)                    │    │
│    │ ClearEntireCostmap (global)                   │    │
│    │ Spin (360°)                                   │    │
│    │ Wait (5 s)                                    │    │
│    │ BackUp (0.30 m)                               │    │
│    └───────────────────────────────────────────────┘    │
│                                                         │
│  IF recovery can't help → abort with status=6           │
└─────────────────────────────────────────────────────────┘
```

`RemovePassedGoals` is important: once the robot passes within 0.5 m of
an intermediate pose, it's silently pruned from the remaining list.
That's why `number_of_poses_remaining` can drop by 2 in one tick.

---

## 6. The safety sandwich

```
  DWB cmd_vel
       ↓
  ┌─────────────────────┐
  │  velocity_smoother  │  ramps between cmd_vel values;
  │  max/min_velocity   │  enforces its own hard cap
  └─────────────────────┘
       ↓
  ┌─────────────────────┐
  │  collision_monitor  │  checks footprint polygon
  │  FootprintApproach  │  against local costmap;
  │  polygon            │  emergency stop / slow
  │  min_points: 4      │  if too many high-cost pixels
  │  projection_time: 5 │  within projection_time × velocity
  └─────────────────────┘
       ↓
  /cmd_vel_smoothed → Isaac Sim
```

The collision monitor is an **independent last-line** — it can stop the
robot even if DWB thought the trajectory was fine. Our
`FootprintApproach.min_points: 4` enables it (the NVIDIA default of
`6000000000` disables it, which is a known footgun).

---

## 7. The full knob reference

### Costmap knobs

| Knob | Scope | Effect |
|---|---|---|
| `robot_radius` | both | robot's inscribed circle; determines the `253` band |
| `footprint` | both | actual rectangular polygon — use this over radius when the robot isn't circular |
| `footprint_padding` | both | extra padding added to the footprint for planner/controller |
| `inflation_radius` | each layer | how far the cost aura extends from walls |
| `cost_scaling_factor` | each layer | exponent of cost decay inside the aura |
| `obstacle_max_range` | obstacle_layer | how far lidar returns are trusted |
| `raytrace_max_range` | obstacle_layer | how far lidar "clears" known obstacles |

### Planner (SmacPlanner2D)

| Knob | Effect |
|---|---|
| `cost_travel_multiplier` | how hard A* prefers low-cost paths (vs short paths) |
| `tolerance` | how close path's end must be to the goal |
| `downsampling_factor` | planning resolution (1 = full, 2 = half) |
| `allow_unknown` | treat unobserved cells as free or forbidden |

### Controller (DWB)

| Knob | Effect |
|---|---|
| `max_vel_x`, `max_speed_xy` | ceiling on linear speed DWB will sample |
| `max_vel_theta` | ceiling on rotational speed |
| `acc_lim_x`, `decel_lim_x` | how fast DWB is allowed to change speed |
| `sim_time` | how long each candidate trajectory is simulated |
| `vx_samples`, `vtheta_samples` | trajectory fan density |
| `BaseObstacle.scale` | how hard to avoid high-cost cells |
| `PathAlign.scale`, `PathDist.scale` | how hard to stick to the planner's path |
| `GoalDist.scale`, `GoalAlign.scale` | how hard to progress toward goal |
| `Oscillation.scale` | penalize back-and-forth jitter |

### Goal checker

| Knob | Effect |
|---|---|
| `xy_goal_tolerance` | positional tolerance before a pose is "reached" |
| `yaw_goal_tolerance` | heading tolerance |
| `stateful` | latch the "reached" state |

### Behavior tree / recoveries

| Knob | Effect |
|---|---|
| `default_nav_through_poses_bt_xml` | which BT file runs |
| `spin_dist`, `backup_dist`, `wait_duration` | how aggressive each recovery is |

### Velocity smoother

| Knob | Effect |
|---|---|
| `max_velocity`, `min_velocity` | hard cap after DWB |
| `smoothing_frequency` | update rate |
| `scale_velocities` | whether to preserve vx:vθ ratio on clamping |

### Collision monitor

| Knob | Effect |
|---|---|
| `FootprintApproach.min_points` | how many high-cost pixels trigger estop |
| `projection_time` | lookahead in seconds |

---

## 8. Failure mode → root cause table

Every failure we've seen maps to one or two knobs:

| Symptom | Error text | Knob(s) responsible |
|---|---|---|
| Robot too close to wall before plan | `Start occupied` | `inflation_radius` (too big), `cost_scaling_factor` (too low) |
| No path exists | `No valid path found` | same — cost-253 ring too wide to thread |
| Goal inside inflated zone | `Goal occupied` / `START_OCCUPIED` (error 205) | `inflation_radius`, `tolerance` (missing) |
| Slow cruising in wide corridors | (robot just crawls) | `BaseObstacle.scale` × `cost_scaling_factor` |
| Bumps walls in tight turns | (collision) | `inflation_radius` too small OR `sim_time` too short |
| "Wandering" looking path | (visual) | TSP dead-end backtracks + `NavigateThroughPoses` pruning + DWB replanning |
| Goal rejected instantly | `Action server is inactive` | Nav2 lifecycle state (prior tour aborted); fix = restart |
| 28 min stuck after abort | (tour timeout) | Recovery subtree can't escape when entire area is cost ≥ 253 |

---

## 9. Tuning recipes

### I want the planner to use narrow junctions

```yaml
global_costmap.inflation_layer.cost_scaling_factor: 3.0   # fast decay
SmacPlanner2D.cost_travel_multiplier:              2.0    # shorter > cheaper
SmacPlanner2D.tolerance:                           0.5    # goal slack
```

### I want the robot to cruise fast in open corridors

```yaml
velocity_smoother.max_velocity:   [1.6, 0, 1.1]
DWB.FollowPath.max_vel_x:          1.6
DWB.FollowPath.max_speed_xy:       1.8
DWB.FollowPath.acc_lim_x:          3.5
```

### I want the robot to hug the centerline of wide hallways

```yaml
global_costmap.inflation_layer.inflation_radius:   1.0
global_costmap.inflation_layer.cost_scaling_factor: 0.1   # wide pressure band
SmacPlanner2D.cost_travel_multiplier:              4.0    # avoid pressure
DWB.FollowPath.critics.BaseObstacle.scale:        15.0    # DWB avoids too
```

### I want graceful failure in dead ends (not a 28-min lockup)

```yaml
# In your Python goal client, not Nav2 — set a per-goal timeout
driver = ThroughPosesDriver(node, timeout_s=300)   # 5 min, not 30
```

```xml
<!-- In navigate_through_poses_w_replanning_and_recovery.xml -->
<Wait wait_duration="1.0"/>   <!-- not 5.0 -->
<BackUp backup_dist="0.5" backup_speed="0.1"/>
<ClearEntireCostmap service_name="local_costmap/clear_entirely_local_costmap"/>
<ClearEntireCostmap service_name="global_costmap/clear_entirely_global_costmap"/>
```

### I want to know WHY Nav2 failed

Set `debug_trajectory_details: True` under `FollowPath` and tail
Nav2's output for `[planner_server]`, `[bt_navigator]`,
`[controller_server]`, `[collision_monitor]` lines. Most of the
diagnostic we've been doing this session was just these.

---

## 10. What our current values imply

As of this commit:

```yaml
global_costmap.inflation_layer.inflation_radius:    0.7
global_costmap.inflation_layer.cost_scaling_factor: 3.0   # was 0.1
SmacPlanner2D.cost_travel_multiplier:               4.0
DWB.FollowPath.critics.BaseObstacle.scale:         15.0
velocity_smoother.max_velocity:           [1.6, 0, 1.1]
```

Intent:
- **Junctions passable** — `cost_scaling_factor=3.0` means the cost aura
  is narrow (~0.3 m effective), so 1.5 m junctions have a low-cost
  spine down the middle.
- **Open-corridor centerline preference preserved** —
  `cost_travel_multiplier=4` + `BaseObstacle.scale=15` still push the
  robot toward the middle *when there's room*.
- **Fast straightaways** — 1.6 m/s top speed with 3.5 m/s² accel.

Expected behavior:
- Robot accelerates out of the hallway
- Through a junction it slows briefly because `BaseObstacle` still
  penalizes the inflated cells (just not as far from walls)
- Plans tighter paths around corners than before
- Unreachable-pose rate drops significantly on the remaining 41 doors

The earlier single-digit door counts (0/19, 2/19) were dominated by
`Start occupied` / `No valid path found` failures. Expect those to
mostly go away; new bottleneck will likely be **DWB oscillating at a
specific corner** or **an isolated door that literally IS too close to
a wall for the robot to stand there**, neither of which are tuning
problems — they're door-map problems that want the Nav2 reachability
pre-filter.

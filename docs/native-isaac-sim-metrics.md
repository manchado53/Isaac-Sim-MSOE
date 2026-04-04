# Native Isaac Sim Metrics for Physics Validation

This document describes the native PhysX APIs available in Isaac Sim 5.1.0 that we can integrate into our physics validation pipeline. These are metrics produced by the simulation engine itself — not our custom pass/fail checks, but the underlying physics data that tells us *how* and *why* the simulation behaves the way it does.

**Why this matters:**
- Our current pipeline measures outcomes (did the cube land? did the door swing?). Native metrics let us measure the *process* — forces, convergence, GPU cost, sleep states — giving us richer data for the research paper and better diagnostics when things fail.
- Compute cost metrics directly quantify the GPU resources required, justifying our use of Rosie's T4/V100/H100 nodes.

**Current validation tests:**

| Test | Script | What it measures today |
|------|--------|----------------------|
| Building Stats | `validate_building_stats.py` | Prim counts, collider/joint counts, building dimensions, screenshots |
| Cube Drop | `validate_cube_drop.py` | 9 cubes per floor, final Z position, pass if cube rests on surface |
| Robot Movement | `validate_robot_movement.py` | JetBot patrol per floor, Z-drop during navigation |
| Door Physics | `validate_door_physics.py` | Drive doors to 80 deg, measure angular delta, pass if > 5 deg |

---

## Table of Contents

1. [API Reference by Category](#api-reference-by-category)
   - [Scene Statistics](#1-scene-statistics)
   - [Contact Forces](#2-contact-forces)
   - [Rigid Body State](#3-rigid-body-state)
   - [Joint / Drive Torque](#4-joint--drive-torque)
   - [Scene Queries](#5-scene-queries)
   - [Solver Residuals](#6-solver-residuals)
   - [Compute Cost](#7-compute-cost)
2. [Recommendation Chart: What to Add Per Test](#recommendation-chart-what-to-add-per-test)
3. [Per-Test Change Details](#per-test-change-details)
   - [Building Stats](#building-stats-changes)
   - [Cube Drop](#cube-drop-changes)
   - [Door Physics](#door-physics-changes)
   - [Robot Movement](#robot-movement-changes)
4. [New Metric Types Summary](#new-metric-types-summary)

---

## API Reference by Category

### 1. Scene Statistics

**What it measures:** A snapshot of the entire physics scene — how many bodies are active vs sleeping, shape counts by geometry type, collision pair activity, solver partitions, and GPU memory allocation.

**API:**
```python
from omni.physx import get_physx_statistics_interface
from omni.physx.bindings._physx import PhysicsSceneStats

stats = PhysicsSceneStats()
stats_iface = get_physx_statistics_interface()
stats_iface.get_physx_scene_statistics(stage_id, encoded_scene_path, stats)
```

**Key fields (55 total, most useful listed):**

| Field | Type | Description |
|-------|------|-------------|
| `nb_static_rigids` | int | Static rigid bodies (walls, floors) |
| `nb_dynamic_rigids` | int | Total dynamic rigid bodies (doors, cubes) |
| `nb_active_dynamic_rigids` | int | Currently awake dynamic bodies |
| `nb_kinematic_rigids` | int | Kinematic bodies (frozen doors) |
| `nb_active_constraints` | int | Active joints/contacts being solved |
| `nb_convex_shapes` | int | Convex hull collision shapes (doors) |
| `nb_trimesh_shapes` | int | Triangle mesh collision shapes (walls, floors) |
| `nb_discrete_contact_pairs_total` | int | Total collision pairs evaluated |
| `nb_discrete_contact_pairs_with_contacts` | int | Pairs generating actual contacts |
| `nb_partitions` | int | Solver partitions (constraint graph complexity) |
| `gpu_mem_heap` | int | Total GPU heap allocation (bytes) |
| `gpu_mem_heap_solver` | int | GPU memory for constraint solver |
| `gpu_mem_heap_broadphase` | int | GPU memory for broad-phase collision |
| `gpu_mem_heap_narrowphase` | int | GPU memory for narrow-phase collision |
| `peak_constraint_memory` | int | Peak memory for joints + contacts (bytes) |
| `compressed_contact_size` | int | Contact data stream size (bytes) |

**Which test uses it:** Building Stats (scene-wide snapshot after warmup)

**How it helps:**
- Quantifies physics complexity for the research paper (e.g., "the Innovation Center simulation maintains 8,268 static colliders, 37 active constraints, and allocates 142 MB of GPU memory for the physics solver")
- Active vs sleeping body count confirms simulation stability
- Contact pair counts validate that collision detection is working across the building
- GPU memory breakdown justifies HPC resource requirements

---

### 2. Contact Forces

**What it measures:** Per-contact-point data between colliding objects — the impulse (force * dt), penetration depth, contact position, and surface normal. This is the raw physics data showing *where* and *how hard* objects are pushing against each other.

**API:**
```python
from omni.physx import get_physx_simulation_interface
from pxr import PhysxSchema

# BEFORE world.reset(): enable contact reporting on prims you care about
PhysxSchema.PhysxContactReportAPI.Apply(door_prim)

# AFTER world.step(): poll contact data
sim = get_physx_simulation_interface()
contact_headers, contact_data, friction_anchors = sim.get_contact_report()
```

**Contact data per point:**

| Field | Type | Description |
|-------|------|-------------|
| `impulse` | Float3 | Contact impulse vector (force * timestep) |
| `separation` | float | Gap between surfaces; **negative = penetration** |
| `position` | Float3 | World-space contact point |
| `normal` | Float3 | Contact surface normal |
| `material0`, `material1` | uint64 | Physics material paths (encoded) |

**Which test uses it:** Door Physics (per-door contact accumulation during swing)

**How it helps:**
- **Explains stuck doors.** Our 9 failing doors (out of 37) show < 5 deg delta. Contact reports will reveal whether they're colliding with walls/mullions (high impulse + negative separation) or simply not being driven (zero contacts).
- Penetration depth quantifies how badly doors interpenetrate nearby geometry — direct evidence for whether our collision filter margin needs tuning.
- Contact impulse magnitude correlates with how much force the door is exerting against an obstruction. High impulse + zero velocity = stalled motor.

---

### 3. Rigid Body State

**What it measures:** Per-body velocity vectors and sleep state. Sleep is PhysX's way of saying "this body has stopped moving and I'm no longer simulating it" — a ground truth signal that an object has settled.

**API:**
```python
# Velocity (on any object inheriting from RigidPrim / DynamicCuboid / WheeledRobot)
velocity = cube.get_linear_velocity()    # np.array [vx, vy, vz]

# Sleep state (requires encoded path)
from omni.physx import get_physx_simulation_interface
sim = get_physx_simulation_interface()
is_asleep = sim.is_sleeping(stage_id, encoded_body_path)
```

**Which tests use it:**
- **Cube Drop** — sleep detection for settle time; velocity for impact speed
- **Robot Movement** — velocity for actual vs commanded speed comparison

**How it helps:**
- **Cube Drop:** Instead of running a fixed 300 settle frames (arbitrary), we detect when all cubes actually sleep. This gives us `frames_to_settle` — a quantitative stability metric. Faster settling = better collision geometry. We also get `max_impact_speed` per cube, which characterizes the drop dynamics.
- **Robot Movement:** Sampling the robot's actual XY speed during patrol and comparing it to the commanded speed (0.2 m/s) shows whether the physics surface provides adequate traction. Speed drops near doors or floor transitions indicate collision geometry issues.

---

### 4. Joint / Drive Torque

**What it measures:** The torque being applied by the DriveAPI motor at each door's revolute joint. Computed deterministically from the drive parameters and current state.

**Formula:**
```
torque = stiffness * (target_angle - current_angle) - damping * angular_velocity
```

For our doors:
- Single doors: stiffness=300, damping=150
- Double doors: stiffness=500, damping=250

**No special API needed** — we already know the drive parameters and can read the angle each frame. Angular velocity is computed numerically from frame-to-frame angle change.

**Which test uses it:** Door Physics (per-door torque profile during swing)

**How it helps:**
- **Stall detection.** A door is "stalled" when: high torque + near-zero angular velocity + low angle delta. This is the physics definition of "the motor is trying but something is blocking it." Combined with contact data, we can distinguish three failure modes:
  1. **Collision stall** — high torque, high contact impulse, zero velocity (door hitting wall)
  2. **Drive failure** — zero torque, zero velocity (DriveAPI not applied correctly)
  3. **Underpowered** — moderate torque, slow velocity, door still moving (needs higher stiffness)
- `peak_drive_torque` shows the maximum effort the motor exerted — useful for tuning drive parameters.

---

### 5. Scene Queries

**What it measures:** Physics raycasts and overlap tests against the collision geometry. Unlike rendering raycasts, these hit the actual physics shapes (convex hulls, triangle meshes) that determine simulation behavior.

**API:**
```python
from omni.physx import get_physx_scene_query_interface

sq = get_physx_scene_query_interface()
hit = sq.raycast_closest(
    origin=carb.Float3(x, y, z),
    dir=carb.Float3(0, 0, -1),    # straight down
    distance=100.0
)
# Returns dict: {position, normal, distance, collision, rigidBody, material}
# Returns empty dict if no hit
```

**Which tests use it:**
- **Cube Drop** — pre-drop floor verification (raycast down from spawn point)
- **Robot Movement** — per-frame ground clearance measurement

**How it helps:**
- **Cube Drop:** Before dropping cubes, we cast a ray downward from each spawn position. If the ray doesn't hit a floor collider, we know the test will fail before it starts — and we know *why* (missing collider at that location). This turns a "cube fell through floor" failure into a specific "no collision geometry at position (X, Y, Z)" diagnostic.
- **Robot Movement:** Raycast ground clearance every 10th frame gives us a continuous measurement of how far the robot is from the floor surface. A clearance spike (robot briefly airborne) or zero clearance (robot embedded in floor) indicates collision geometry problems. `min_ground_clearance` is a single-number quality metric for floor solidity.

---

### 6. Solver Residuals

**What it measures:** How well the physics solver converged on a solution each timestep. The solver iterates to satisfy all constraints (contacts, joints) simultaneously. Residuals measure the remaining error after iterations complete. Lower = better.

**API:**
```python
from pxr import PhysxSchema
import carb

# BEFORE world.reset(): enable residual reporting on physics scene
scene_prim = stage.GetPrimAtPath("/World/PhysicsScene")
PhysxSchema.PhysxResidualReportingAPI.Apply(scene_prim)
carb.settings.get_settings().set("/physics/updateResidualsToUsd", True)

# AFTER world.step(): read residual attributes from the scene prim
residual_api = PhysxSchema.PhysxResidualReportingAPI(scene_prim)
rms_pos = residual_api.GetPhysxResidualReportingRmsResidualPositionIterationAttr().Get()
max_pos = residual_api.GetPhysxResidualReportingMaxResidualPositionIterationAttr().Get()
rms_vel = residual_api.GetPhysxResidualReportingRmsResidualVelocityIterationAttr().Get()
max_vel = residual_api.GetPhysxResidualReportingMaxResidualVelocityIterationAttr().Get()
```

**Residual fields:**

| Field | Description |
|-------|-------------|
| RMS Position Residual | Average position constraint error across all bodies |
| Max Position Residual | Worst-case position constraint error (single body) |
| RMS Velocity Residual | Average velocity constraint error |
| Max Velocity Residual | Worst-case velocity constraint error |

**Which test uses it:** Building Stats (single snapshot after warmup)

**How it helps:**
- A single-number "solver health" score for the entire building physics setup. If residuals are high, the solver is struggling — maybe too many interpenetrations, stiff joints, or conflicting constraints.
- Comparing residuals before vs after physics setup shows whether our collision geometry and joints actually improve solver convergence (they should).
- For the paper: demonstrates that the physics configuration produces a well-conditioned simulation (low residuals = constraints are satisfiable).

---

### 7. Compute Cost

**What it measures:** The GPU and CPU resources consumed by the physics simulation. This is the data that justifies running on Rosie's GPU nodes rather than a laptop.

**Sources:**
```python
# GPU memory breakdown (from PhysicsSceneStats, see Section 1)
gpu_mem_total     = stats.gpu_mem_heap                    # bytes
gpu_mem_solver    = stats.gpu_mem_heap_solver              # bytes
gpu_mem_broadphase = stats.gpu_mem_heap_broadphase         # bytes
gpu_mem_narrowphase = stats.gpu_mem_heap_narrowphase       # bytes
peak_constraints  = stats.peak_constraint_memory           # bytes

# Physics step timing (from PhysicsProfileStats)
from omni.physx.bindings._physx import PhysicsProfileStats
# Each entry has .zone_name (str) and .ms (float, milliseconds)
# Enable via: carb.settings.get_settings().set("/physics/exposeProfilerData", True)

# Wall-clock runtime (already captured in all tests as runtime_seconds)
```

**Key compute cost metrics:**

| Metric | Unit | What it shows |
|--------|------|---------------|
| `gpu_mem_total_mb` | MB | Total GPU memory allocated for physics |
| `gpu_mem_solver_mb` | MB | GPU memory for constraint solving (joints + contacts) |
| `gpu_mem_broadphase_mb` | MB | GPU memory for initial collision pair detection |
| `gpu_mem_narrowphase_mb` | MB | GPU memory for detailed contact generation |
| `peak_constraint_memory_mb` | MB | Peak memory for all constraints (joints + contacts) |
| `physics_step_ms` | ms | Time per physics step (target: 16.67ms for 60Hz) |
| `runtime_seconds` | s | Total wall-clock simulation time |

**Which test uses it:** Building Stats (primary), all tests report `runtime_seconds`

**How it helps:**
- **Justifies Rosie.** We can state exactly: "The Innovation Center physics simulation allocates X MB of GPU memory and requires Y ms per physics step with Z active constraints. This exceeds the capability of consumer hardware and requires dedicated GPU compute nodes."
- GPU memory breakdown shows *where* the compute goes — solver vs collision detection. If broadphase dominates, the building has too many potential collision pairs. If solver dominates, the constraint graph (joints) is expensive.
- `physics_step_ms` compared against the 16.67ms budget (60Hz) tells us whether we're running in real-time or slower. Exceeding this budget means the simulation can't run at real-time speed — a direct argument for faster GPUs.
- Runtime comparison across GPU types (T4 vs V100 vs H100) quantifies the speedup from better hardware.

---

## Recommendation Chart: What to Add Per Test

| Test | Metric to Add | API Category | Priority | Rationale |
|------|--------------|--------------|----------|-----------|
| **Building Stats** | PhysicsSceneStats snapshot | Scene Statistics | P0 | Quantifies physics complexity for paper |
| **Building Stats** | Solver residuals | Solver Residuals | P0 | Single-number solver health score |
| **Building Stats** | GPU memory breakdown | Compute Cost | P0 | Justifies HPC resource usage |
| **Building Stats** | Physics step timing | Compute Cost | P1 | Real-time capability assessment |
| **Cube Drop** | Sleep-based settle detection | Rigid Body State | P0 | Replaces arbitrary frame count with physics ground truth |
| **Cube Drop** | Impact/final velocity | Rigid Body State | P1 | Characterizes drop dynamics |
| **Cube Drop** | Pre-drop floor raycast | Scene Queries | P1 | Diagnoses missing colliders before test runs |
| **Door Physics** | Contact impulse + penetration | Contact Forces | P0 | Explains *why* doors fail (collision vs drive issue) |
| **Door Physics** | Angular velocity profile | Rigid Body State | P0 | Distinguishes moving vs stuck doors |
| **Door Physics** | Computed drive torque | Joint/Drive Torque | P0 | Stall detection: high torque + zero velocity = blocked |
| **Door Physics** | Stall classification | Contact + Torque | P1 | Automated failure mode diagnosis |
| **Robot Movement** | Actual vs commanded speed | Rigid Body State | P1 | Surface traction quality metric |
| **Robot Movement** | Ground clearance raycast | Scene Queries | P1 | Continuous floor solidity measurement |

**Priority key:**
- **P0** — High value, straightforward to implement, directly supports research paper
- **P1** — Valuable diagnostic data, moderate implementation effort

---

## Per-Test Change Details

### Building Stats Changes

**Current output:** Prim counts, collider/joint counts, bounding box, per-camera screenshots.

**New metrics added:**

| New Metric | Type | Source | What It Tells Us |
|------------|------|--------|-----------------|
| `physics_scene_stats.nb_static_rigids` | int | PhysicsSceneStats | How many walls/floors have static collision |
| `physics_scene_stats.nb_dynamic_rigids` | int | PhysicsSceneStats | How many objects can move (doors, dropped cubes) |
| `physics_scene_stats.nb_active_dynamic_rigids` | int | PhysicsSceneStats | How many dynamic bodies are currently awake |
| `physics_scene_stats.nb_kinematic_rigids` | int | PhysicsSceneStats | How many bodies are frozen (kinematic doors) |
| `physics_scene_stats.nb_active_constraints` | int | PhysicsSceneStats | Active joint/contact constraints being solved |
| `physics_scene_stats.nb_convex_shapes` | int | PhysicsSceneStats | Convex hull shapes (doors use these) |
| `physics_scene_stats.nb_trimesh_shapes` | int | PhysicsSceneStats | Triangle mesh shapes (walls/floors use these) |
| `physics_scene_stats.nb_discrete_contact_pairs_total` | int | PhysicsSceneStats | Total collision pairs being evaluated |
| `physics_scene_stats.nb_discrete_contact_pairs_with_contacts` | int | PhysicsSceneStats | Pairs that are actually touching |
| `physics_scene_stats.nb_partitions` | int | PhysicsSceneStats | Solver partitions (constraint graph complexity) |
| `solver_residuals.rms_position` | float | ResidualReportingAPI | Average position constraint error |
| `solver_residuals.max_position` | float | ResidualReportingAPI | Worst-case position error (single body) |
| `solver_residuals.rms_velocity` | float | ResidualReportingAPI | Average velocity constraint error |
| `solver_residuals.max_velocity` | float | ResidualReportingAPI | Worst-case velocity error |
| `compute_cost.gpu_mem_total_mb` | float | PhysicsSceneStats | Total GPU physics memory (MB) |
| `compute_cost.gpu_mem_solver_mb` | float | PhysicsSceneStats | GPU memory for constraint solver (MB) |
| `compute_cost.gpu_mem_broadphase_mb` | float | PhysicsSceneStats | GPU memory for broad-phase collision (MB) |
| `compute_cost.gpu_mem_narrowphase_mb` | float | PhysicsSceneStats | GPU memory for narrow-phase collision (MB) |
| `compute_cost.peak_constraint_memory_mb` | float | PhysicsSceneStats | Peak constraint memory (MB) |
| `compute_cost.physics_step_ms` | float | PhysicsProfileStats | Milliseconds per physics step |

**How these help:**
- Before/after comparison shows exactly what our physics pipeline adds: collider shapes go from 0 to thousands, constraints appear, GPU memory increases proportionally. This is the quantitative evidence that the pipeline transforms a static CAD model into a physically interactive environment.
- Solver residuals prove the simulation is well-conditioned (low residuals = all constraints are being satisfied).
- Compute cost data goes directly into the paper to justify Rosie usage.

---

### Cube Drop Changes

**Current output:** Per-cube final position, per-camera pass/fail, cubes landed count.

**New metrics added:**

| New Metric | Type | Source | What It Tells Us |
|------------|------|--------|-----------------|
| Per-cube `final_velocity` | [vx,vy,vz] | `get_linear_velocity()` | Whether the cube has actually stopped moving |
| Per-cube `max_impact_speed` | float (m/s) | Velocity tracking per frame | Peak speed during fall — characterizes drop energy |
| Per-cube `floor_raycast_hit` | bool | `raycast_closest()` downward | Whether a floor collider exists at the spawn point |
| Per-cube `floor_raycast_distance` | float (ft) | `raycast_closest()` downward | Height above nearest floor surface |
| Per-camera `frames_to_settle` | int or null | `is_sleeping()` polling | Frame count when all 9 cubes go to sleep |
| Top-level `settle_detection.all_settled` | bool | Sleep state | Whether all cubes across all cameras settled |
| Top-level `settle_detection.max_frames_to_settle` | int | Sleep state | Worst-case settle time across all cameras |

**How these help:**
- **`frames_to_settle`** replaces our arbitrary 300-frame settle window with physics ground truth. If cubes settle in 120 frames, we know the collision response is crisp. If they never settle (null), something is wrong (jittering on a bad collision surface).
- **`floor_raycast_hit`** pre-verifies floor colliders before the test runs. A cube that "fails" because it fell through the floor is fundamentally different from a cube that fails because it bounced off the building. The raycast tells us which case we're in.
- **`max_impact_speed`** characterizes the physics of the drop. All cubes drop from the same height (3 ft), so impact speeds should be consistent. Outliers indicate the cube hit something on the way down.
- **`final_velocity`** is the definitive "has this cube stopped" check. Our current test uses Z position, but a cube could be sliding laterally on a surface (nonzero XY velocity) and still pass the Z check.

---

### Door Physics Changes

**Current output:** Per-door initial/final angle, delta, pass/fail. Global pass rate.

**New metrics added:**

| New Metric | Type | Source | What It Tells Us |
|------------|------|--------|-----------------|
| Per-door `peak_angular_velocity` | float (deg/s) | Numerical derivative of angle | Maximum rotation speed during swing |
| Per-door `final_angular_velocity` | float (deg/s) | Last frame derivative | Whether door is still moving at test end |
| Per-door `contact_impulse_total` | float | `get_contact_report()` | Total contact force accumulated during swing |
| Per-door `max_penetration_depth` | float | Contact `separation` field | Worst interpenetration with nearby geometry |
| Per-door `peak_drive_torque` | float (Nm) | `stiffness*(target-current) - damping*omega` | Maximum effort the motor exerted |
| Per-door `stall_detected` | bool | Combined analysis | True if door is blocked by collision |

**How these help:**
- **This is the biggest diagnostic improvement.** Currently, 9 of our 37 doors fail with < 5 deg delta, and we don't know why. These metrics will classify each failure:

  | Failure Mode | Angular Vel | Contact Impulse | Torque | What to Fix |
  |-------------|-------------|-----------------|--------|-------------|
  | **Collision stall** | ~0 | High | High | Increase collision filter margin |
  | **Drive failure** | ~0 | ~0 | ~0 | Check DriveAPI application |
  | **Underpowered** | Low but nonzero | Moderate | Moderate | Increase drive stiffness |
  | **Wrong hinge position** | Nonzero | Varies | Varies | Fix hinge offset estimation |

- **`max_penetration_depth`** directly measures how badly a door interpenetrates its surrounding geometry. Negative values (in stage units) quantify the exact overlap. This feeds directly into tuning the collision filter `margin` parameter.
- **`peak_drive_torque`** shows whether the motor is working at all. A door with zero peak torque throughout the test has a misconfigured DriveAPI.
- **`stall_detected`** is an automated binary diagnosis that replaces manual inspection of per-door results.

---

### Robot Movement Changes

**Current output:** Per-camera initial/final Z, Z-drop, pass/fail, position samples.

**New metrics added:**

| New Metric | Type | Source | What It Tells Us |
|------------|------|--------|-----------------|
| Per-camera `velocity_samples` | [float, ...] | `get_linear_velocity()` XY magnitude | Speed profile during patrol |
| Per-camera `mean_speed` | float (m/s) | Average of velocity samples | Average actual speed vs commanded 0.2 m/s |
| Per-camera `max_speed` | float (m/s) | Max of velocity samples | Peak speed (should be near commanded) |
| Per-camera `ground_clearance_samples` | [float, ...] | `raycast_closest()` downward | Distance to floor surface over time |
| Per-camera `min_ground_clearance` | float (ft) | Min of clearance samples | Closest the robot got to the floor surface |

**How these help:**
- **`mean_speed` vs commanded speed** is a traction quality metric. If the robot commands 0.2 m/s but only achieves 0.15 m/s, the floor surface has friction issues. If it achieves 0.25 m/s, it's sliding.
- **`ground_clearance_samples`** is a continuous floor solidity measurement. Our current test only checks the initial and final Z — the robot could briefly fall through and recover, and we'd never know. Ground clearance every 10th frame catches transient penetrations.
- **`min_ground_clearance`** is a single-number quality score. Values near zero indicate the robot's wheels are sitting properly on the surface. Negative values mean the robot penetrated the floor at some point. Large positive values mean the robot was airborne.
- **`velocity_samples`** over time can reveal floor transitions (speed drops at doorways, ramps, or floor material changes).

---

## New Metric Types Summary

All new metrics fall into these categories:

| Category | Metric Type | Data Format | Available From |
|----------|-------------|-------------|----------------|
| **Physics Complexity** | Scene-wide counts and sizes | Single integers/floats | PhysicsSceneStats |
| **Simulation Quality** | Solver convergence residuals | 4 floats (RMS/max, pos/vel) | PhysxResidualReportingAPI |
| **Compute Cost** | GPU memory and step timing | Floats in MB/ms | PhysicsSceneStats + ProfileStats |
| **Object Dynamics** | Velocity vectors, sleep state | Per-body arrays | RigidPrim API + PhysX sim interface |
| **Contact Diagnostics** | Impulse, penetration, position | Per-contact-point data | PhysxContactReportAPI + sim interface |
| **Drive Diagnostics** | Torque, stall classification | Per-joint floats + bool | Computed from DriveAPI parameters |
| **Geometry Verification** | Raycast hit/distance | Per-query bool + float | PhysX scene query interface |

**Backward compatibility:** All new fields use default values in the extraction pipeline. Running `extract_validation_metrics.py` on old JSON results (without native metrics) will show "N/A" for the new columns — no breakage.

**Performance impact:** Estimated < 10% increase in wall-clock time per test. Velocity queries and sleep checks are O(1) per body. Raycasts are done sparingly (36 total for cube drop, every 10th frame for robot). Contact report polling adds marginal overhead during the door physics recording loop.

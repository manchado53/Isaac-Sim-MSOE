# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **team repository** for a research project using NVIDIA Isaac Sim on MSOE's Rosie supercomputer.

### Research Goal

Deploy autonomous robots for indoor building operations (material delivery, facility monitoring, security patrol) by training them entirely in simulation before the physical building is complete. The framework converts architectural BIM (Revit) data into simulation-ready digital twins in NVIDIA Isaac Sim.

### Target Environment

A multi-story university engineering facility currently under construction at MSOE, featuring multi-building connections, mixed-use spaces, and varied floor plans.

### Pipeline

1. **BIM-to-Sim Conversion**: Revit architectural model → USD format → Isaac Sim import
2. **Environment Augmentation**: Collision geometry, semantic annotations, sensor-realistic rendering
3. **Navigation Graph Extraction**: Topological graph from building geometry with optimized operational routes via classical path planning
4. **Robot Training (RL)**: Mobile robot with LiDAR + camera sensors trained via reinforcement learning for local navigation and obstacle avoidance
5. **Synthetic Data Generation**: Onboard sensor data used to train visual anomaly detection models (open doors, obstructions, irregular room states)
6. **HPC-Scale Training**: Parallelized on Rosie cluster with NVIDIA T4 GPUs

### Key Metrics

- Task completion rate
- Path efficiency
- Collision frequency
- Performance across routes of varying complexity spanning the multi-building complex

### Broader Impact

Architectural BIM data (standard in modern building design) can serve as the foundation for scalable robot training without physical access to the target environment. The pipeline is repeatable for any building with available BIM data, with implications for sim-to-real transfer and robot-friendly architectural design.

**Repository**: `~/isaac-sim-msoe/` (git-tracked)

**Installation Status**: Verified and operational
- Isaac Sim 5.1.0-rc.19 installed at `~/isaacsim`
- Python 3.11.13 environment included
- Successfully tested on Tesla T4 GPUs
- SLURM job scripts configured and working
- Video recording verified working (`imageio-ffmpeg` installed in Isaac Sim Python)

**Detailed Documentation**:
- [Building Reference](.claude/docs/building.md) — Innovation Center architecture, floors, rooms, doors, USD hierarchy
- [Physics & Building Setup](.claude/docs/physics.md) — collision, materials, joints, building pipeline
- [sim_utils/ README](sim_utils/README.md) — utility library reference

## Rosie Supercomputer

When the user asks about Rosie (MSOE's supercomputer), including topics like SLURM, job submission, GPU access, modules, storage, or cluster usage:
1. Use the WebFetch tool to check https://msoe.dev/#/ for relevant documentation
2. Follow any links on that page that are pertinent to the user's specific question
3. Base your answers on the fetched documentation rather than general knowledge

**Rosie Web Portal**: https://dh-ood.hpc.msoe.edu/

### Common SLURM Workflow

When helping with SLURM jobs:
- Always check the fetched documentation for current commands and best practices
- GPU jobs require specific partition and resource specifications
- Module system is used for loading software environments

## NVIDIA Isaac Sim

When the user asks about Isaac Sim, including topics like installation, setup, simulation, robotics, Omniverse, or related workflows:
1. Use the WebFetch tool to check https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/download.html and navigate relevant pages
2. Follow any links on that site that are pertinent to the user's specific question
3. Base your answers on the fetched documentation rather than general knowledge

### Local Isaac Sim Installation

**Installed Version**: 5.1.0-rc.19 (October 2025)
**Installation Path**: `~/isaacsim`
**Python Version**: 3.11.13 (included with Isaac Sim)

**Running Simulations on Rosie**:
```bash
# On a GPU node (after srun or in SLURM job):
cd ~/isaacsim
export OMNI_KIT_ACCEPT_EULA=YES
./python.sh path/to/script.py
```

**Important Notes**:
- Isaac Sim MUST run on GPU nodes (will crash on login nodes)
- Use SLURM scripts in `~/isaac-sim-msoe/` for job submission
- First run may be slower due to shader compilation
- Print statements in scripts go to Isaac Sim's internal logs, not SLURM stdout
- Exit code 0 indicates successful simulation
- Typical job runtime: ~80s on T4 (40s startup + simulation time)
- Stderr warnings about `pulse` audio directory and deprecated `omni.isaac.*` imports are harmless

**Video Recording**:
- Uses `imageio-ffmpeg` bundled ffmpeg binary (no system ffmpeg needed on Rosie)
- Videos saved to `videos/` in the project root
- Import with: `from sim_utils.video_capture import VideoRecorder`
- Always use `render=True` in `world.step()` when recording
- Two capture modes: viewport (PNG → ffmpeg) and camera sensor (RGBA pipe)

**Development Tools**:
```bash
# Jupyter notebooks (on GPU nodes)
./jupyter_notebook.sh path/to/notebook.ipynb

# Direct Python execution
./python.sh script.py
```

**Official Documentation**: When the user asks about Isaac Sim features, always fetch the latest docs from https://docs.isaacsim.omniverse.nvidia.com/5.1.0/

## Development Workflow

### Quick Start (Recommended)

Submit a batch job to run simulations:
```bash
cd ~/isaacsim/isaac-sim-msoe
sbatch submit_isaac_job.sh simulations/environment/your_script.py
```

Monitor progress:
```bash
squeue -u $USER                    # Check job status
cat logs/isaac_<JOBID>.out         # View output when complete
```

### Interactive Development

For testing and debugging, use an interactive GPU session:
```bash
# Request GPU node
srun --partition=teaching --gres=gpu:1 --time=2:00:00 --pty bash

# Run simulation
cd ~/isaacsim/isaac-sim-msoe
./run_isaac_interactive.sh simulations/rigid_body/your_script.py
```

### Running Simulations on Rosie

1. **Create simulation scripts** in the appropriate `simulations/` subdirectory
2. **Test with interactive session** (optional but recommended)
3. **Submit GPU jobs via SLURM** for production runs
4. **Monitor job status**: `squeue -u $USER`
5. **Check results** in `logs/` directory

## File Organization

```
~/isaacsim/                              # Isaac Sim installation (DO NOT MODIFY)
├── python.sh                            # Isaac Sim Python launcher
├── isaac-sim.sh                         # Main launcher
└── kit/                                 # Isaac Sim core

~/isaacsim/isaac-sim-msoe/               # THIS REPO (team workspace)
├── simulations/                         # All simulation scripts (organized by type)
│   ├── rigid_body/                      # Basic physics demos
│   │   ├── hello_isaac.py              # Falling cube test (verified working)
│   │   ├── bouncing_balls.py           # Multi-ball collision with video
│   │   ├── boxes_colliding.py          # Box collision demo
│   │   └── door_physics_demo.py        # Hinged door with revolute joint + DriveAPI
│   ├── robot/                           # Robot simulation scripts
│   │   ├── franka_camera_demo.py       # Franka arm manipulation
│   │   ├── jetbot_block_push.py        # JetBot block pushing task
│   │   └── jetbot_hospital_demo.py     # Hospital navigation + follow cam
│   ├── environment/                     # Building physics & environment scripts
│   │   ├── setup_innovation_center_physics.py  # Generate IC physics USD
│   │   ├── setup_building_physics.py           # Generic physics setup (any Revit USD)
│   │   ├── engineering_physics_demo.py         # Multi-floor drop test + video
│   │   ├── building_physics_demo.py            # Generic orbital camera tour demo
│   │   ├── building_door_demo.py               # Door drive demo (IC exterior cam)
│   │   ├── door_demo_finalV1.py                # Single door opening demo
│   │   ├── door_opening_showcase.py            # 2FG entrance door showcase
│   │   ├── door_walkthrough_showcase.py        # Camera walk-through doorway
│   │   ├── floor_showcase.py                   # Roof-off ball-drop floor showcase
│   │   ├── innovation_center_physics_showcase.py # Exposed floors + door animation
│   │   ├── innovation_center_robot_demo.py     # JetBot patrol demo
│   │   ├── wireframe_showcase.py               # Collision mesh wireframe tour
│   │   ├── robot_floor_test.py                 # Robot patrol floor test
│   │   ├── msoe_building_tour.py               # Orbital building tour
│   │   ├── discover_doors.py                   # Door discovery & cataloging
│   │   ├── floor_camera_screenshots.py         # Per-floor camera discovery & screenshots
│   │   ├── validate_building_stats.py          # Validation: prim/collider/joint stats + PhysX scene stats + solver residuals + compute cost
│   │   ├── validate_cube_drop.py               # Validation: 9 cubes per camera, velocity tracking, sleep-based settle, floor raycast
│   │   ├── validate_robot_movement.py          # Validation: Nova Carter patrol, velocity/speed, ground clearance raycast
│   │   ├── validate_door_physics.py            # Validation: door swing + contact reports, angular velocity, torque, stall detection
│   │   └── people_demo.py                      # Static human character on ground floor (Option A — USD reference, no animation)
│   └── jetbot_hospital_overhead.py     # Hospital overhead view demo
├── sim_utils/                           # Shared Python utilities
│   ├── __init__.py
│   ├── building_physics.py             # Building physics pipeline (~1184 lines, includes door classification/DriveAPI/collision filters/PhysX runtime helpers)
│   ├── test_cameras.py                 # Shared test camera positions & helpers
│   ├── video_capture.py                # VideoRecorder for MP4 output
│   ├── follow_camera.py                # Follow camera for robot tracking
│   └── README.md
├── assets/                              # Static assets
│   ├── environments/msoe_building/     # MSOE building USD files
│   ├── demos/                          # Demo videos
│   └── robots/                         # Robot assets (placeholder)
├── videos/                              # Rendered MP4 output (gitignored)
├── logs/                                # SLURM output logs + physics reports
│   ├── Door-logs/                      # Door discovery reports
│   ├── tests/                          # Manual test results
│   └── validation/                     # Validation pipeline results
├── docs/                                # Project documentation
│   ├── physics-and-building-reference.md
│   ├── physics-validation-research.md
│   ├── native-isaac-sim-metrics.md     # Native PhysX API reference for validation pipeline
│   └── random_notes.txt
├── images/                              # Screenshots and diagrams
│   └── charts/                         # Generated presentation charts (PNGs)
├── validate_physics.py                  # Validation pipeline runner (submits SLURM jobs, aggregates results)
├── extract_validation_metrics.py        # Before/after comparison: extracts metrics from JSON results → summary/CSV/JSON
├── generate_validation_charts.py        # Generates matplotlib charts (PNGs) from validation results for slides
├── submit_isaac_job.sh                  # SLURM batch job submission
├── submit_validation_job.sh             # SLURM submission for validation tests
├── run_isaac_interactive.sh             # Interactive GPU runner
├── run_physics_pipeline.sh              # Physics pipeline orchestrator (chains 4 SLURM jobs)
├── requirements.txt                     # Python dependencies
├── .claude/                             # Claude Code configuration
│   ├── CLAUDE.md                       # This file
│   ├── docs/physics.md                 # Physics documentation
│   └── agents/                         # Specialized agent definitions
├── README.md                            # User documentation
└── QUICKSTART.md                        # Quick reference
```

### Simulation Categories

| Category | Directory | Description |
|----------|-----------|-------------|
| Rigid Body | `simulations/rigid_body/` | Basic physics: cubes, spheres, doors |
| Robot | `simulations/robot/` | Robot control, navigation, manipulation |
| Environment | `simulations/environment/` | Building physics setup, testing, tours |

**Best Practices**:
- Create new simulation scripts in the appropriate `simulations/` subdirectory
- Use `submit_isaac_job.sh` to submit jobs
- Check `logs/` for job output and errors
- Test scripts interactively before batch submission
- Use exit code to verify simulation success (0 = success)

### Simulation Script Template

When creating new simulation scripts, follow this pattern:
```python
#!/usr/bin/env python3
import os, sys
os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

from omni.isaac.core import World
from omni.isaac.core.objects import DynamicCuboid, DynamicSphere
import numpy as np

# Add project root for sim_utils import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from sim_utils.video_capture import VideoRecorder

def main():
    world = World()
    world.scene.add_default_ground_plane()
    # ... add objects ...
    world.reset()

    with VideoRecorder("output_name") as recorder:
        for i in range(300):  # 10s at 30 FPS
            world.step(render=True)
            recorder.capture()

    simulation_app.close()

if __name__ == "__main__":
    main()
```

**Note**: Scripts in `simulations/` subdirectories need `'..', '..'` in `sys.path` to reach the project root for `sim_utils` imports.

## Building USD Assets

The project includes MSOE building models exported from Revit:

| File | Building | Notes |
|------|----------|-------|
| `MSOEBuildingTake2.usd` | Innovation Center | 17 floors, 7,100+ elements |
| `22-129 - MSOE Engineering Building...usd` | Engineering Building | Revit export |
| `innovation_center_physics.usd` | Innovation Center | With physics applied |
| `engineering_building_physics.usd` | Engineering Building | With physics applied |

Located in `assets/environments/msoe_building/`. See [physics.md](.claude/docs/physics.md) for details on the physics pipeline.

### ⚠️ CRITICAL: `metersPerUnit` varies per USD

Different files in this repo use different USD stage units. ALWAYS verify with `UsdGeom.GetStageMetersPerUnit(stage)` before writing any code that assumes "coords are in feet/meters/etc." Hard-won lesson: `MSOEBuildingV4_physics.usd` is labeled meters while the raw Revit export is in centimeters.

| File | `metersPerUnit` | 1 unit = |
|---|---|---|
| `MSOEBuildingV4.usd` (raw Revit) | `0.01` | 1 cm |
| `MSOEBuildingV4_physics.usd` (physics-applied) | `1.0` | 1 m |
| NVIDIA character USDs (People/Characters/) | `1.0` | 1 m |

**Consequences when `V4_physics.usd` is the stage**:
- `TEST_CAMERAS` coordinates (e.g. `"G"` target `(-27.42, 83.72, 14.56)`) are in **meters**, not feet. A Z of 14.56 m ≈ 47.8 ft above the USD origin is the ground-floor elevation in that model
- Characters loaded with `prims.create_prim(usd_path=...)` have metersPerUnit=1.0 too → **no manual scale needed**. Keep `PERSON_SCALE = 1.0`. Applying `1/0.3048` (≈ 3.28×) produces 18.6-ft-tall giants
- Real walkable floor Z via raycast: 13.56 (meters)
- Character bbox at `translate_Z=14.56, scale=1.0`: feet at 14.435m → need offset `-0.875` to plant on floor at 13.56m

**How this bug happens**: The Revit export is in centimeters (`metersPerUnit=0.01`). The physics pipeline (`setup_building_physics.py`) converts to metric 1.0 during processing. Neither is "feet" — earlier docs in this file saying "metersPerUnit=0.3048 from Revit" are wrong and should be ignored when working with V4 assets.

**Defensive pattern**:
```python
mpu = UsdGeom.GetStageMetersPerUnit(stage)
# Write math in meters; only convert at boundaries (UI, reports)
```

## Door Physics & DriveAPI

Doors use a three-stage pipeline defined in `sim_utils/building_physics.py`:

1. **Classify mechanism** — `classify_door_mechanism(prim)` checks the prim path for Revit type substrings. Only `HINGED_SINGLE` and `HINGED_DOUBLE` get physics; overhead, sliding, garage, and unknown doors are skipped entirely.
2. **Dynamic rigid body** — `apply_dynamic_rigid_body(stage, prim, mass=..., deep_search=True)` adds `RigidBodyAPI`, `CollisionAPI` (convexHull), and `MassAPI`. Mass is 15 kg for single doors, 30 kg for double doors. `deep_search=True` is required for Revit-exported instanced meshes.
3. **Revolute joint** — `create_revolute_joint()` creates a hinge between the world frame (`body0=""`) and the door prim (`body1`). Axis is `"Z"` (vertical). Hinge position estimated via `estimate_door_hinge_offset()` using world-space bounding box min-edge. Angular limits: ±100° for single, ±70° for double.
4. **DriveAPI** — `setup_door_drive()` applies `UsdPhysics.DriveAPI` with `"angular"` mode to the joint, enabling motorized position control.

**DriveAPI Parameters** (tunable per `DOOR_DRIVE_PARAMS`):

| Parameter | Single Door | Double Door | Purpose |
|-----------|-------------|-------------|---------|
| `stiffness` | 300.0 | 500.0 | Spring constant toward target angle |
| `damping` | 150.0 | 250.0 | Resistance to angular velocity |
| `max_force` | 3000.0 | 5000.0 | Torque limit (Nm) |
| `target_position` | 0.0 | 0.0 | Initial target angle (degrees) |

**Runtime control**: Call `set_drive_target(drive, angle_deg)` each step to animate doors (e.g., 0° → 90° for opening).

**Door classification** (`classify_door_mechanism()`):

| Mechanism | Revit Patterns | Physics? |
|-----------|---------------|----------|
| `HINGED_SINGLE` | Type_F, Type_G, Type_N, Type_FG, Curtain_Wall | Yes |
| `HINGED_DOUBLE` | Type_2F, Type_2G, 2FG, 2FU, 2F_, Type_2F___Dual | Yes |
| `OVERHEAD` | AV300_RAYNOR, Overhead_Rolling, Overhead_Sectional | Skipped |
| `SLIDING` | Sliding_Glass, Dormakaba | Skipped |
| `UNKNOWN` | No match (e.g. Elevator_Door) | Skipped |

**Double door limitation**: Revit exports both leaves as a single combined mesh. They swing as one unit — independent leaf physics is not possible without a different export format.

**Door discovery**: `simulations/environment/discover_doors.py` catalogs doors via category folders, FamilyData paths, and global name search. Outputs `logs/door_catalog.json` and `logs/door_discovery_report.txt`.

**Collision filtering**: `add_door_collision_filters(stage, door_prims, margin=1.0)` disables collision between each door and nearby static geometry (walls, mullions) whose bounding boxes overlap. Essential for curtain wall doors that sit flush inside wall assemblies. The `margin` parameter (in stage units) inflates each door's bounding box to catch geometry the door sweeps into during the first degrees of rotation.

**Key constraints**:
- Dynamic bodies MUST use `"convexHull"` collision (triangle mesh only works on statics)
- Overhead/sliding/garage/unknown doors are filtered out before rigid body application
- Doors can be frozen with `RigidBodyAPI.CreateKinematicEnabledAttr(True)` for robot navigation tests

## Additional building_physics.py Utilities

Beyond door physics, `sim_utils/building_physics.py` provides:

- **`setup_physics_scene(stage, gravity, time_steps_per_second, gpu_dynamics)`** -- Creates and configures a `PhysicsScene` with gravity, solver frequency, and optional GPU dynamics.
- **`iter_building_elements(stage, building_prim, level_filter)`** -- Traverses Revit-exported buildings, yielding `(level_name, category_name, element_prim)` tuples. Auto-detects three building structures: flat, category-based (Innovation Center), and hierarchical (Engineering Building).
- **`discover_floor_levels(stage, building_path)`** -- Clusters floor elements by Z-height and returns area-weighted centroids per floor level. Used by demo scripts to auto-discover spawn positions.
- **`categorize_prim(name)`** / **`PrimCategory`** -- Classifies prims into `STATIC_STRUCTURAL`, `STATIC_GLAZING`, `STATIC_STAIR`, `DYNAMIC_DOOR`, or `SKIP` based on Revit naming patterns.

### Native PhysX Runtime Helpers

These functions query the PhysX simulation engine for data only available **after** `world.reset()` + `world.step()`. All wrapped in try/except for graceful degradation.

- **`get_physics_scene_stats(stage)`** -- Snapshots `PhysicsSceneStats` into a plain dict: body counts (static/dynamic/kinematic/active), shape counts by type, contact pairs, solver partitions, GPU memory breakdown.
- **`get_solver_residuals(stage)`** -- Reads RMS/max position/velocity residuals from `PhysxResidualReportingAPI`. Requires `enable_residual_reporting(stage)` called before `world.reset()`.
- **`get_compute_cost(stage)`** -- Extracts GPU memory breakdown in MB (solver, broadphase, narrowphase, articulation, total) from `PhysicsSceneStats`.
- **`check_body_sleeping(stage, prim_path)`** -- Returns True/False for whether a rigid body has settled (PhysX sleep state).
- **`raycast_down(origin, max_distance)`** -- Casts ray straight down, returns `(hit, distance, position)`. Uses `get_physx_scene_query_interface()`.
- **`enable_residual_reporting(stage)`** -- Applies `PhysxResidualReportingAPI` to physics scene prim. Must be called before `world.reset()`.
- **`encode_prim_path(prim_path_str)`** -- Encodes SdfPath to uint64 for PhysX runtime APIs.

## Physics Pipeline Orchestrator

`run_physics_pipeline.sh` chains 4 SLURM jobs with dependencies to process any raw Revit-exported USD:

```bash
./run_physics_pipeline.sh assets/environments/msoe_building/MSOEBuildingTake2.usd
./run_physics_pipeline.sh assets/environments/msoe_building/MyBuilding.usd --mem 48G --margin 1.5
```

**Pipeline steps** (each a separate SLURM job):
1. **Physics Setup** (`setup_building_physics.py`) -- Apply colliders, materials, joints, drives
2. **Floor Drop Test** (after step 1) -- Verify floors are solid
3. **Door Swing Test** (after step 1, parallel with step 2) -- Verify doors rotate
4. **Demo Video** (after steps 2+3) -- Orbital camera tour with drop objects

**Options**: `--margin FLOAT` (collision filter margin), `--mem SIZE`, `--time DURATION`, `--partition NAME`, `--output PATH`

## People / Human Characters

Isaac Sim 5.1.0 ships with people simulation support via two extensions (both installed):
- `isaacsim.replicator.agent.core` 0.7.28 — main IRA system + synthetic data generation
- `omni.anim.people` 0.7.9 — character animation engine (NavMesh walking, behaviors)

Character assets (male/female in police/doctor/construction/casual outfits) are hosted on NVIDIA's S3 and resolved via `get_assets_root_path() + "/Isaac/People/Characters/<name>/<name>.usd"` — same pattern as robot USDs.

**Three placement approaches** (see [.claude/docs/people.md](.claude/docs/people.md) for details):
- **Option A (static)**: USD reference with position/orientation — no NavMesh, no animation extensions. Used by `simulations/environment/people_demo.py`.
- **Option B (custom pose)**: Manipulate `UsdSkel.Animation` joint rotations — for lying-down/fallen-person anomalies.
- **Option C (animated/walking)**: Full IRA system via `~/isaacsim/tools/actor_sdg/sdg_scheduler.py` + YAML config. Requires NavMesh baked on floors.

**Research goal**: anomaly detection training data — student flow (Option C, future) + injured person on floor (Option B, future). Current status: Option A ground floor demo (`people_demo.py`).

## Cracked Glass Anomaly Dataset

Second anomaly class for the visual anomaly detector, alongside the fallen-person pipeline. Visual-only (no geometry fracture) — a Replicator graph swaps a random subset of glass panels to cracked OmniPBR materials per frame and emits RGB + semantic segmentation labels.

- **Module**: `sim_utils/glass_anomaly.py` — `enumerate_glass_prims()`, `create_cracked_glass_materials()`, `bind_visual_material()`, `apply_semantic_label()`, `label_glass_prims()`
- **Dataset script**: `simulations/dataset/cracked_glass_dataset.py`
- **SLURM wrapper**: `submit_cracked_glass_dataset.sh`
- **Textures**: `assets/materials/cracked_glass/crack_normal_v{1,2,3}.png` + `crack_opacity_v{1,2,3}.png` (see README there for authoring notes — pipeline tolerates missing files)
- **Scope**: GL_1 double-pane + curtain wall panels. GL_3 shadowbox and GL_1_1 laminated are excluded by default.
- **Output**: `datasets/cracked_glass/<timestamp>/<cam_key>/{rgb,semantic_segmentation,bounding_box_2d_tight}/`
- **Run**: `sbatch submit_cracked_glass_dataset.sh --frames 500 --fraction 0.2 --cameras G,1,2,3`

## Test Camera Positions — Innovation Center

Verified security-camera-style positions for visual testing on each floor of the Innovation Center. Top corner of room, 12mm wide-angle lens, looking down diagonally. Coordinates in stage units — **units depend on the loaded USD**; see "CRITICAL: metersPerUnit varies per USD" above. For `MSOEBuildingV4_physics.usd` these values are meters.

```python
TEST_CAMERAS = {
    "G": {"pos": (-42.15, 68.44, 16.44), "target": (-27.42, 83.72, 14.56), "room": "Event Floor IC 100"},
    "1": {"pos": (-56.03, 62.23, 17.50), "target": (-49.19, 73.19, 14.56), "room": "Robotics Lab IC 101"},
    "2": {"pos": (-56.28, 60.88, 24.70), "target": (-49.19, 72.96, 23.50), "room": "Classroom 40 IC 302"},
    "3": {"pos": (-41.51, 67.62, 29.65), "target": (-30.23, 81.20, 27.77), "room": "Capstone Studio IC 405"},
}
```

**Camera settings**: focal_length=12.0mm, resolution=1920x1080, frequency=30Hz (matches floor_camera_screenshots.py reference images, default horizontal aperture ~20.955mm)
**Lighting for interiors**: DistantLight(7500) + DomeLight(3500) + SphereLight(45000) at camera position
**Discovery script**: `simulations/environment/floor_camera_screenshots.py`
**Shared module**: `sim_utils/test_cameras.py` — `TEST_CAMERAS` dict, `create_test_camera()`, `add_interior_lighting()`, `look_at_orientation()`

## Physics Validation Pipeline

Comprehensive pipeline for testing USD physics files. Submits 4 separate SLURM jobs, waits for completion, and aggregates results.

**Run the pipeline**:
```bash
python validate_physics.py --usd assets/environments/msoe_building/innovation_center_physics.usd
python validate_physics.py --usd my_physics.usd --skip robot_movement --partition batch
python validate_physics.py --usd my_physics.usd --dry-run   # preview without submitting
```

**4 Test Scripts** (each runs as a separate SLURM job via `submit_validation_job.sh`):

| Test | Script | What it does |
|------|--------|-------------|
| `building_stats` | `validate_building_stats.py` | Counts colliders, rigid bodies, joints, meshes, floor/door elements. Takes per-camera screenshots. Collects PhysX scene stats (body/shape counts, contact pairs, GPU memory), solver residuals, and compute cost. |
| `cube_drop` | `validate_cube_drop.py` | Drops 9 cubes near each camera, records video. Tracks per-cube velocity (max impact speed), sleep-based settle detection (`frames_to_settle`), pre-drop floor raycast verification. |
| `robot_movement` | `validate_robot_movement.py` | Spawns Nova Carter at each camera's floor, drives toward camera, records video. Tracks XY velocity (mean/max speed) and ground clearance via raycast. |
| `door_physics` | `validate_door_physics.py` | Finds doors within 20ft of each camera, drives to 80 deg, records video. Applies PhysxContactReportAPI for contact impulse/penetration, tracks angular velocity, computes drive torque, detects stalled doors. |

**Output**:
- `logs/validation/validation_report.json` — full structured results
- `logs/validation/validation_report.txt` — human-readable summary
- `logs/validation/*_results.json` — per-test JSON results
- `videos/validate_*.mp4` — per-camera videos for each test

**Pipeline options**:
- `--usd <path>` — physics USD file to validate (required)
- `--partition <name>` — SLURM partition (default: teaching)
- `--timeout <seconds>` — max wait per job (default: 7200)
- `--skip <tests>` — comma-separated test names to skip
- `--dry-run` — print sbatch commands without submitting

## Navigation Pipeline (Nova Carter + Nav2)

The full navigation stack for driving Nova Carter around the Innovation Center in Isaac Sim. Two READMEs document it end-to-end:

- **[navigation/README.md](../navigation/README.md)** — runtime entry point: `nav2_orchestrator.py` + Nav2 launch + params + visualization. Start here for any "how do I drive the robot" question.
- **[floorplan_vision/README.md](../floorplan_vision/README.md)** — map generation pipeline: `generate_occupancy_map.py` → `auto_graph_generator.py` → `verify_orientation.py`. Start here for any "how do I regenerate the occupancy map / navgraph" question.

**Three things to remember** before touching this stack:

1. **Occupancy map orientation is ROS `map_server` convention**: `row 0 = world y_max`, `origin: [x_min, y_min, yaw]` is the world pose of the **bottom-left** pixel. `nav2_map_server/src/map_io.cpp:326` internally reverses the image rows after loading. If you save a PNG with `row 0 = y_min` (the natural visual orientation), Nav2 publishes a Y-mirrored static map and AMCL can't localize. `generate_occupancy_map.py::save_floor` applies `np.fliplr` + `np.flipud` to end up in the ROS convention. Run `verify_orientation.py` after any change to the generator.

2. **NVIDIA's upstream `carter_navigation_params.yaml` has gotchas we patch in `kern_navigation_params.yaml`**: `collision_monitor.FootprintApproach.min_points` is `6000000000` in the reference (effectively disables the emergency stop — we use `4`), and `DWB.FollowPath.BaseObstacle.scale` is `0.02` in the reference (path-following critics outweigh obstacle avoidance 1600:1 — we use `15.0`). If someone copies a fresh upstream params file, they'll hit both.

3. **Visualization: Foxglove over RViz for headless setups.** `foxglove_bridge` is installed on rosita; run `ros2 run foxglove_bridge foxglove_bridge --ros-args -p port:=8765` alongside Nav2 and connect from a laptop browser to `ws://<host>:8765`. RViz via SSH `-Y` X11 forwarding works but needs an X server on the client side (XQuartz on Mac, VcXsrv on Windows with `$env:DISPLAY = "localhost:0.0"`).

**Quick runtime reference**:

```bash
# Isaac Sim + robot + ROS bridge
~/isaacsim/python.sh navigation/nav2_orchestrator.py --floor 4 --stream --stream-ip 100.107.199.57
# Optional: spawn a test obstacle cube at world coords (use = for negative values)
~/isaacsim/python.sh navigation/nav2_orchestrator.py --floor 4 --stream --obstacle=-24.25,81.69

# Nav2 stack (separate terminal, after "Scene ready")
source /opt/ros/jazzy/setup.bash
source ~/IsaacSim-ros_workspaces/jazzy_ws/install/setup.bash
ros2 launch kern_navigation kern_navigation.launch.py

# Send a goal
ros2 action send_goal /navigate_to_pose nav2_msgs/action/NavigateToPose \
  "{pose: {header: {frame_id: map}, pose: {position: {x: -7.30, y: 78.09, z: 0.0}, orientation: {w: 1.0}}}}"
```

**File layout** (post-reorg):

- `navigation/nav2_orchestrator.py` — Isaac Sim entry point (was `simulations/robot/nav2_orchestrator.py` pre-reorg)
- `navigation/params/kern_navigation_params.yaml` — all Nav2 params
- `navigation/launch/kern_navigation.launch.py` — Nav2 bringup
- `navigation/rviz2/kern_navigation.rviz` — RViz layout
- `floorplan_vision/generate_occupancy_map.py` — occupancy scanner (was `test_building/generate_occupancy_map.py` pre-reorg)
- `floorplan_vision/auto_graph_generator.py` — skeleton → navgraph
- `floorplan_vision/verify_orientation.py` — Y-flip correctness check
- `floorplan_vision/plot_graph_with_axes.py` — labeled-axis visualization
- `assets/environments/InnovationCenter/maps/floor_{1-4}_occupancy.{png,npy}` — occupancy outputs
- `assets/environments/InnovationCenter/navgraphs/floor_{1-4}_auto_navgraph.{json,gml}` — navgraph outputs
- `assets/environments/InnovationCenter/navgraphs/floor_{1-4}_map_metadata.yaml` — **authoritative Nav2 map YAML** (points at `../maps/floor_N_occupancy.png`)
- `assets/environments/InnovationCenter/building_info.json` — floor/bounds/spawn metadata

## GPU Partitions on Rosie

| Partition | GPU Type | Time Limit | Use Case |
|-----------|----------|------------|----------|
| `teaching` | T4 | 7 days | Development, testing (default) |
| `batch` | T4 | 2 days | Production runs |
| `dgx` | V100 | 21 days | Large simulations |
| `dgxh100` | H100 | 21 days | Maximum performance |

**Choosing a partition**: Edit `#SBATCH --partition=` in `submit_isaac_job.sh`

## Troubleshooting

**Common Issues**:
1. **"CUDA error: no device detected"** -> Running on login node, need GPU node
2. **"Exit Code: 139" (segfault)** -> Missing GPU, ensure job requested `--gres=gpu:1`
3. **Print statements not visible** -> Normal; Isaac Sim captures stdout. Check exit code instead.
4. **Job pending** -> GPU nodes busy, wait or try different partition
5. **`ModuleNotFoundError: No module named 'sim_utils'`** -> Ensure `sys.path` includes the project root in the script.
6. **Objects falling through floors** -> Check that static colliders are applied; see physics.md for collision setup.
7. **Doors not swinging** -> Verify revolute joint has correct hinge position/axis, DriveAPI is applied with `"angular"` mode, and `set_drive_target()` is called each step. Check that `deep_search=True` is used for instanced meshes.
8. **Black screen video** -> Never use `omni.usd.get_context().open_stage()` to load building USDs — it bypasses Isaac Sim's renderer initialization. Instead, create `World()` first, then load via `stage.DefinePrim(...).GetReferences().AddReference(usd_path, primPath=Sdf.Path("/World/Building"))`. See `engineering_physics_demo.py` for the correct pattern.
9. **Camera sees wrong part of building / all red** -> Check `UsdGeom.GetStageMetersPerUnit(stage)` — units vary per USD (see "CRITICAL: metersPerUnit" section). `V4_physics.usd` is in meters (`mpu=1.0`); raw `V4.usd` is in cm (`mpu=0.01`). Camera offsets must match stage units. Use the target element's bounding box to position the camera relative to it.
10. **"Unresolved reference prim path" warning** -> When loading a physics USD via `AddReference`, always specify `primPath=Sdf.Path("/World/Building")` so USD composition resolves the internal hierarchy correctly.

**Verification**:
- Exit code 0 = simulation successful
- Check `nvidia-smi` output in logs to confirm GPU allocation
- Simulation runtime shown in log timestamps

**Harmless Warnings** (safe to ignore in logs):
- `Failed to create secure directory (/run/user/.../pulse)` -- no audio device on compute nodes
- `omni.isaac.core.objects.* has been deprecated in favor of isaacsim.core.api.objects.*` -- still functional
- `carb.audio.device` / `carb.audio.output` warnings -- headless mode, no audio needed

## Documentation Policy

When making major changes (reward weights, env config, new reward terms, termination changes, hyperparameter tuning, architecture changes, etc.), always update the relevant documentation:
- **CLAUDE.md**: Update if the change affects workflow, file organization, or project-level context
- **README.md**: Update if the change affects how others use or understand the project
- **Code docstrings/comments**: Update inline docs in the files being changed to reflect new values or behavior

### Training Experiment Logging (REQUIRED)

**Every time a training job is submitted** (`sbatch training/submit_training.sh`), update `training/EXPERIMENTS.md` before or immediately after submission. Required fields:

- Add a row to the **Run Index** table with job ID and status "Running"
- Add a **Detailed Run Notes** entry with:
  - Job ID and log file paths
  - Full reward config (all terms, weights, params)
  - What changed from the previous run and **why**
  - The hypothesis being tested
  - Result (fill in after job completes)

This applies to eval jobs submitted via `submit_eval_scenarios.sh` as well — note which model checkpoint was evaluated.

### Run Directory Tracking (REQUIRED)

**Immediately after every `sbatch` submission**, set up a cron job to watch for the new run directory and fill it into EXPERIMENTS.md automatically:

```
CronCreate every 5 minutes:
  - ls ~/isaac-sim-msoe/training/logs/sb3/NovaCarter-Patrol/ to find new timestamped dirs
  - When found: update EXPERIMENTS.md Run Index row and Detailed Run Notes header (replace "TBD (Job XXXXXX)" with actual timestamp), update Weights path
  - CronDelete the job once the directory is confirmed and EXPERIMENTS.md is updated
```

The run directory is created when the job actually starts on the compute node (may be minutes to hours after submission depending on queue). The cron fires every 5 minutes until it finds the directory. Cancel with CronDelete once resolved.

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
│   │   ├── setup_innovation_center_physics.py  # Generate physics USD
│   │   ├── engineering_physics_demo.py         # Multi-floor drop test + video
│   │   ├── test_building_physics.py            # Collider verification
│   │   ├── test_engineering_physics.py         # Per-floor drop validation
│   │   ├── robot_floor_test.py                 # Robot patrol floor test
│   │   ├── msoe_building_tour.py               # Orbital building tour
│   │   └── discover_doors.py                   # Door discovery & cataloging
│   └── jetbot_hospital_overhead.py     # Hospital overhead view demo
├── sim_utils/                           # Shared Python utilities
│   ├── __init__.py
│   ├── building_physics.py             # Building physics pipeline (~703 lines, includes door/DriveAPI)
│   ├── video_capture.py                # VideoRecorder for MP4 output
│   ├── follow_camera.py                # Follow camera for robot tracking
│   └── README.md
├── assets/                              # Static assets
│   ├── environments/msoe_building/     # MSOE building USD files
│   ├── demos/                          # Demo videos
│   └── robots/                         # Robot assets (placeholder)
├── videos/                              # Rendered MP4 output (gitignored)
├── logs/                                # SLURM output logs + physics reports
│   └── Door-logs/                      # Door discovery reports
├── submit_isaac_job.sh                  # SLURM batch job submission
├── run_isaac_interactive.sh             # Interactive GPU runner
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

## Door Physics & DriveAPI

Doors use a three-stage pipeline defined in `sim_utils/building_physics.py`:

1. **Dynamic rigid body** — `apply_dynamic_rigid_body(stage, prim, mass=15.0, deep_search=True)` adds `RigidBodyAPI`, `CollisionAPI` (convexHull), and `MassAPI` to the door prim. `deep_search=True` is required for Revit-exported instanced meshes.
2. **Revolute joint** — `create_revolute_joint()` creates a hinge between the world frame (`body0=None`) and the door prim (`body1`). Axis is `"Z"` (vertical). Hinge position estimated via `estimate_door_hinge_offset()` using bounding box min-X edge.
3. **DriveAPI** — `setup_door_drive()` applies `UsdPhysics.DriveAPI` with `"angular"` mode to the joint, enabling motorized position control.

**DriveAPI Parameters** (tunable per `DOOR_DRIVE_PARAMS`):

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `stiffness` | 300.0 | Spring constant toward target angle |
| `damping` | 150.0 | Resistance to angular velocity |
| `max_force` | 3000.0 | Torque limit (Nm) |
| `target_position` | 0.0 | Initial target angle (degrees) |

**Runtime control**: Call `set_drive_target(drive, angle_deg)` each step to animate doors (e.g., 0° → 90° for opening).

**Door discovery**: `simulations/environment/discover_doors.py` catalogs doors via category folders, FamilyData paths, and global name search. Outputs `logs/door_catalog.json` and `logs/door_discovery_report.txt`.

**Key constraints**:
- Dynamic bodies MUST use `"convexHull"` collision (triangle mesh only works on statics)
- Doors default to 15 kg mass
- Overhead/sliding doors skip drive creation (mechanism = `UNKNOWN`)
- Doors can be frozen with `RigidBodyAPI.CreateKinematicEnabledAttr(True)` for robot navigation tests

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
9. **Camera sees wrong part of building / all red** -> Building coordinates are in **feet** (metersPerUnit=0.3048 from Revit). Camera offsets must be in feet too. Use the target element's bounding box to position the camera relative to it — ~12 feet offset for a close-up.
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

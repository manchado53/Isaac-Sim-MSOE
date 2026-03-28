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
cd ~/isaac-sim-msoe
sbatch submit_isaac_job.sh [optional_script.py]
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
cd ~/isaac-sim-msoe
./run_isaac_interactive.sh test-simulations/your_script.py
```

### Running Simulations on Rosie

1. **Create simulation scripts** in `test-simulations/` (or your own directory)
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

~/isaac-sim-msoe/                        # THIS REPO (team workspace)
├── test-simulations/                    # Test/example simulation scripts
│   ├── hello_isaac.py                   # Falling cube test (verified working)
│   ├── bouncing_balls.py                # Multi-ball demo with video
│   └── boxes_colliding.py              # Box collision demo
├── sim_utils/                           # Shared utilities (Python package)
│   ├── __init__.py
│   └── video_capture.py                # VideoRecorder for MP4 output
├── videos/                              # Rendered MP4 output (gitignored)
├── logs/                                # SLURM output logs
├── submit_isaac_job.sh                  # Batch job submission
├── run_isaac_interactive.sh             # Interactive runner
├── CLAUDE.md                            # This file
├── README.md                            # User documentation
└── QUICKSTART.md                        # Quick reference
```

**Best Practices**:
- Create new simulation scripts in an appropriate directory
- `test-simulations/` is for test and example scripts
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
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
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

**Note**: The `sim_utils/` directory must use underscores (not hyphens) to be importable by Python.

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
5. **`ModuleNotFoundError: No module named 'sim_utils'`** -> The directory is `sim_utils/` (hyphen). Either rename it to `sim_utils/` or ensure `sys.path` is set correctly in the script.

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

# Isaac Sim Quick Start Guide

## ✅ System Status

- **Isaac Sim Version**: 5.1.0
- **Location**: `~/isaacsim`
- **Python**: 3.11.13 (included)
- **Status**: Ready to use on GPU nodes

## Running Your First Simulation

### Method 1: Submit a Job (Recommended)

The easiest way to run a simulation:

```bash
cd ~/isaac-sim-msoe
sbatch submit_isaac_job.sh
```

This will:
- Submit a job to the `teaching` partition
- Run the test script `test-simulations/hello_isaac.py`
- Save output to `logs/isaac_<JOBID>.out`

Check job status:
```bash
squeue -u $USER
```

View results:
```bash
# Find your job output file
ls -lt logs/

# View the output
cat logs/isaac_<JOBID>.out
```

### Method 2: Interactive Session

For development and debugging:

```bash
# 1. Request a GPU node
srun --partition=teaching --gres=gpu:1 --time=2:00:00 --pty bash

# 2. Run your simulation
cd ~/isaacsim/isaac-sim-msoe
./run_isaac_interactive.sh test-simulations/hello_isaac.py
```

## Running Your Own Simulations

### Create a New Simulation

1. Create a Python script in your simulations directory:

```python
#!/usr/bin/env python3
import os
os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

from omni.isaac.core import World
# Your simulation code here

simulation_app.close()
```

2. Submit it as a job:

```bash
cd ~/isaacsim/isaac-sim-msoe
sbatch submit_isaac_job.sh path/to/your_script.py
```

## Available GPU Partitions

| Partition | GPU Type | Time Limit | Best For |
|-----------|----------|------------|----------|
| `teaching` | T4 | 7 days | Development, testing (default) |
| `batch` | T4 | 2 days | Production runs |
| `dgx` | V100 | 21 days | Large-scale simulations |
| `dgxh100` | H100 | 21 days | Maximum performance |

To use a different partition, edit `submit_isaac_job.sh`:
```bash
#SBATCH --partition=dgxh100  # Change from 'teaching' to desired partition
```

## Common Commands

```bash
# Check job queue
squeue -u $USER

# Cancel a job
scancel <JOBID>

# View job details
scontrol show job <JOBID>

# Check partition availability
sinfo -p teaching

# View recent log files
ls -lt ~/isaacsim/isaac-sim-msoe/logs/
```

## Troubleshooting

### Job Not Starting?
Check queue position:
```bash
squeue -u $USER
```

### Need More Resources?
Edit `submit_isaac_job.sh`:
```bash
#SBATCH --time=04:00:00  # Request more time
#SBATCH --mem=32G        # Request more memory
```

### Simulation Errors?
Check the error log:
```bash
cat logs/isaac_<JOBID>.err
```

## Video Recording

Simulations can record MP4 videos for debugging and sharing:

```python
from sim_utils.video_capture import VideoRecorder

with VideoRecorder("my_sim") as recorder:
    for i in range(300):  # 10s at 30 FPS
        world.step(render=True)  # render=True is required!
        recorder.capture()
# -> videos/my_sim.mp4
```

Retrieve videos from Rosie:
```bash
# From your local machine:
scp <username>@dh-ood.hpc.msoe.edu:~/isaacsim/isaac-sim-msoe/videos/*.mp4 .

# Or use the Rosie web portal Files browser at https://dh-ood.hpc.msoe.edu/
```

## Shared Building Assets

Building USD and IFC files are stored on shared storage so the whole team can access them:

```bash
# First time setup — copy shared assets to your local repo
./setup_shared_assets.sh

# See what's available
./setup_shared_assets.sh --list
```

**Shared location**: `/data/ai_club/isaac-sim-msoe/`

| File | Size | Use |
|------|------|-----|
| `msoe-engineering-innovation-center.ifc` | 514MB | Full IFC-to-USD + physics pipeline |
| `msoe-engineering-innovation-USD.usd` | 213MB | Physics pipeline only (skip IFC conversion) |
| `innovation_center_physics.usd` | 35MB | Pre-built physics USD (ready to use in simulations) |

### Running the Physics Pipeline

```bash
# After running setup_shared_assets.sh:

# Option 1: From raw USD (physics pipeline only)
./run_physics_pipeline.sh assets/environments/msoe-engineering-innovation-USD.usd

# Option 2: From IFC (full conversion + physics)
./run_physics_pipeline.sh assets/environments/msoe-engineering-innovation-center.ifc

# Option 3: Use the pre-built physics USD directly in your scripts
# (already at assets/environments/msoe_building/innovation_center_physics.usd)
```

### Sharing New Physics USD Files

If you generate a new physics USD, share it back:
```bash
cp assets/environments/msoe_building/my_building_physics.usd /data/ai_club/isaac-sim-msoe/
chmod 664 /data/ai_club/isaac-sim-msoe/my_building_physics.usd
```

## Next Steps

- Explore Isaac Sim examples: `~/isaacsim/standalone_examples/`
- Read documentation: https://docs.isaacsim.omniverse.nvidia.com/5.1.0/
- Check Rosie docs: https://msoe.dev/

## Help

Run scripts without arguments to see usage:
```bash
./run_isaac_interactive.sh
./submit_isaac_job.sh
```

# Isaac Sim — MSOE Digital Twin Research

Autonomous robot simulation platform built with NVIDIA Isaac Sim 5.1.0, deployed on MSOE's Rosie HPC cluster (T4 GPUs, SLURM).

## What This Project Is

This repository is the simulation infrastructure for a research project at the Milwaukee School of Engineering (MSOE). We are building a framework that converts **architectural BIM data (Revit) into simulation-ready digital twins** where autonomous robots can be trained and evaluated entirely in software — before the physical building is even complete.

The target environment is the **Robert D. Kern Engineering Innovation Center**, a 97,000 sq ft facility currently under construction at MSOE that will connect to two existing campus buildings, forming a multi-building complex. Using the building's Revit model, we convert the design into USD format and import it into NVIDIA Isaac Sim for physics-realistic robot simulation.

### Research Goals

- **BIM-to-Sim Pipeline** — Convert Revit architectural models to USD, augment with collision geometry and semantic annotations, and import into Isaac Sim
- **Navigation Graph Extraction** — Generate NavMesh from building geometry, compute optimized patrol routes using classical path planning
- **Robot Training** — Train a simulated mobile robot (LiDAR + camera) via reinforcement learning for local navigation and obstacle avoidance across multi-building routes
- **Synthetic Data Generation** — Generate onboard sensor data to train visual anomaly detection models (open doors, obstructions, irregular room states)
- **Scalable HPC Deployment** — Run parallelized training on MSOE's Rosie cluster with NVIDIA T4 GPUs via SLURM

### Current Stage

We are in the **environment setup and navigation phase**:
- Isaac Sim 5.1 is deployed and running headlessly on Rosie via SLURM batch jobs
- The Isaac Sim hospital environment is loaded with a Nova Carter AMR navigating corridors (follow camera + overhead camera demos working)
- The MSOE Kern Center Revit model has been exported to USD and is being imported into Isaac Sim (scale conversion, collision setup, ceiling/roof hiding for overhead views)
- NavMesh generation from building geometry is next — this will feed the topological navigation graph for route planning

### What's Next

- Generate NavMesh for the hospital ground floor using `omni.anim.navigation.core`
- Build the topological navigation graph with pairwise walkable distances
- Integrate reinforcement learning for local navigation policy training
- Import the full Kern Center multi-building complex
- Generate synthetic sensor data for anomaly detection model training

### Demo: Hospital Follow Camera

A Nova Carter AMR navigating the Isaac Sim hospital environment with a third-person follow camera:

![Hospital Follow Camera Demo](assets/demos/hospital_follow_cam_10s.mp4)

### Current Simulations

| Simulation | Description |
|---|---|
| `jetbot_hospital_demo.py` | Nova Carter navigating hospital corridors — follow camera (640x480) |
| `jetbot_hospital_overhead.py` | Same hospital from a bird's-eye static camera — ceiling hidden (1280x720) |
| `testbuilding_overhead.py` | MSOE TestBuilding (Revit export) — overhead view with roof hidden |
| `franka_camera_demo.py` | Franka robotic arm with side-angle camera |

---

## How It Runs

Since Rosie's compute nodes have no monitors, there are two ways to run simulations:

- **Option A — Headless (SLURM batch job):** Submit a script, it renders to `.mp4` via ffmpeg. No live view.
- **Option B — WebRTC Streaming:** Stream Isaac Sim's full GUI to your local machine over SSH tunnel.

---

## Prerequisites

- MSOE account with Rosie access
- MSOE VPN (GlobalProtect) if off campus — connect to `vpn.msoe.edu`
- SSH client on your local machine (PowerShell on Windows, Terminal on Mac/Linux)

---

## Step 1: Clone this Repo

SSH into Rosie:
```bash
ssh <username>@ad.msoe.edu@dh-mgmt2.hpc.msoe.edu
```

Clone the repo:
```bash
cd ~
git clone https://gitlab.com/manchado53/isaac-sim-msoe.git
cd isaac-sim-msoe
```

---

## Step 2: Install Isaac Sim (One-Time Setup)

> Skip this if Isaac Sim is already installed at `~/isaacsim/`.

```bash
cd ~
wget https://download.isaacsim.omniverse.nvidia.com/isaac-sim-standalone-5.1.0-linux-x86_64.zip
mkdir ~/isaacsim
unzip isaac-sim-standalone-5.1.0-linux-x86_64.zip -d ~/isaacsim
cd ~/isaacsim
./post_install.sh
```

The zip is large — this only needs to be done once.

---

## Step 3: Install ffmpeg (One-Time Setup, Option A only)

```bash
mkdir -p ~/bin
wget -O /tmp/ffmpeg.tar.xz https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz
tar -xf /tmp/ffmpeg.tar.xz -C /tmp/
cp /tmp/ffmpeg-*-amd64-static/ffmpeg ~/bin/ffmpeg
chmod +x ~/bin/ffmpeg
```

---

## Option A: Headless Batch Job (Video Recording)

Run a simulation fully in the background via SLURM. The script records the camera output to an `.mp4` file using ffmpeg.

### Submit a job

All simulations are run through a single generic job script:

```bash
sbatch submit_isaac_job.sh simulations/jetbot_hospital_demo.py
sbatch submit_isaac_job.sh simulations/jetbot_hospital_overhead.py
sbatch submit_isaac_job.sh simulations/testbuilding_overhead.py
sbatch submit_isaac_job.sh simulations/franka_camera_demo.py
```

### Monitor progress

```bash
squeue -u <username>@ad.msoe.edu            # check job status
tail -f logs/isaac_<job_id>.out              # live log output
```

### Output

Videos are saved to `~/isaac-sim-msoe/output/<simulation_name>/` when the job completes.

To target a specific node:
```bash
sbatch --nodelist=dh-node17 submit_isaac_job.sh simulations/jetbot_hospital_demo.py
```

---

## Option B: WebRTC Streaming (Live GUI)

Stream Isaac Sim's full GUI to your local machine over an SSH tunnel. This gives you an interactive live view.

```
[Your PC] <--SSH tunnel--> [Rosie Login Node] <--network--> [Compute Node + GPU]
 Streaming Client (49100)    dh-mgmt2                         Isaac Sim + T4 GPU
```

### Step B1: Download the WebRTC Streaming Client (one-time, on your local machine)

Download the Isaac Sim WebRTC Streaming Client for your OS from:
https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/download.html

### Step B2: Get an interactive GPU node on Rosie

From your VS Code terminal connected to Rosie, request an interactive session:

```bash
srun --partition=teaching --gres=gpu:t4:1 --cpus-per-gpu=4 --time=02:00:00 --nodelist=dh-nodeX --pty bash
```

> Replace `dh-nodeX` with a node that isn't busy. Check with `squeue`.

Once the job starts, note the node name from your terminal prompt (e.g., `dh-node5`).

### Step B3: Start Isaac Sim in streaming mode

In the same terminal (now on the compute node):

```bash
cd ~/isaacsim
bash isaac-sim.streaming.sh
```

Wait until you see `app ready` in the output. The first launch takes several minutes due to shader compilation — subsequent launches are faster. Warnings about audio and windowing are normal.

### Step B4: Open an SSH tunnel from your local machine

On your **local machine** (not on Rosie), open a new terminal and run:

```bash
ssh -L 49100:dh-nodeX:49100 -L 47998:dh-nodeX:47998 <username>@ad.msoe.edu@dh-mgmt2.hpc.msoe.edu
```

> Replace `dh-nodeX` with your actual compute node. Leave this terminal open — closing it breaks the tunnel.

If you're unsure which node you're on:
```bash
squeue -u <username>@ad.msoe.edu    # check the NODELIST column
```

### Step B5: Connect with the Streaming Client

1. Open the Isaac Sim WebRTC Streaming Client on your local machine
2. Enter `127.0.0.1` as the server address
3. Click **Connect**
4. Isaac Sim's full GUI appears within a few seconds

### When you're done

1. Close the WebRTC Streaming Client
2. Press `Ctrl+C` in the Isaac Sim terminal to stop it
3. Type `exit` to release the GPU node
4. Close the SSH tunnel terminal

---

## Project Structure

```
isaac-sim-msoe/
├── simulations/                          # All simulation scripts
│   ├── hello_isaac.py                    # Minimal test (cube drop)
│   ├── jetbot_hospital_demo.py           # Hospital — follow camera
│   ├── jetbot_hospital_overhead.py       # Hospital — overhead camera
│   ├── testbuilding_overhead.py          # MSOE building — overhead camera
│   └── franka_camera_demo.py             # Franka arm demo
├── sim_utils/                            # Reusable utilities
│   ├── follow_camera.py                  # FollowCamera class (quaternion tracking)
│   └── video_capture.py                  # VideoRecorder (ffmpeg pipe)
├── assets/
│   ├── demos/                            # Demo video clips
│   │   └── hospital_follow_cam_10s.mp4   # 10s hospital follow-cam clip
│   └── environments/                     # Custom building USDs (Revit exports)
│       └── msoe_building/TestBuilding.usd
├── submit_isaac_job.sh                   # Generic SLURM submission script
├── logs/                                 # SLURM logs (gitignored)
└── output/                               # Video outputs (gitignored)
```

---

## Useful SLURM Commands

```bash
squeue -u <username>@ad.msoe.edu                                              # check running jobs and assigned nodes
srun --partition=teaching --gres=gpu:t4:1 --cpus-per-gpu=4 --time=02:00:00 --pty bash   # interactive T4 session
srun --partition=dgx --gres=gpu:v100:1 --cpus-per-gpu=8 --time=02:00:00 --pty bash      # interactive V100 session
nvidia-smi                                                                    # verify GPU on compute node
scancel <job_id>                                                              # cancel a job
```

---

## Key Technical Notes

**Render pipeline ordering** — The camera's `set_world_pose()` must be called *before* `world.step(render=True)`. Isaac Sim's render pipeline is asynchronous — the camera position must be set before the render step to appear in the captured frame.

**Headless mode** — Batch SLURM scripts use `headless: True`. Do not change this to `False` for batch jobs.

**Nova Carter specs** — wheel radius: 0.14m, wheelbase: 0.4m. Uses `joint_wheel_left` / `joint_wheel_right` DOF names with `DifferentialController`.

**Performance** — The T4 GPU gives lower FPS in Isaac Sim. For better streaming performance, request a V100 on the dgx partition (`--partition=dgx --gres=gpu:v100:1`).

**First launch** — Shader compilation on the first run takes several minutes. Be patient.

**Time limit** — SLURM jobs on the teaching partition are typically limited to 2 hours. Save your work before it expires.

---

## Troubleshooting

**GPU not detected / "No device could be created"**
Make sure you included `--gres=gpu:t4:1` in your `srun` command. Verify with `nvidia-smi`.

**"channel: open failed: connect failed: Name or service not known"**
The node name in your SSH tunnel is wrong. Run `squeue` to find the correct node name.

**Streaming client won't connect**
Verify Isaac Sim shows `app ready` in the terminal. Check that your SSH tunnel terminal is still open. Confirm the node name matches.

**SSH tunnel password not working**
Connect to `dh-mgmt2.hpc.msoe.edu` (not `dh-ood` or `ad.msoe.edu`). Must be on MSOE VPN if off campus.

**"GLIBC not found" errors**
This happens with the Singularity container (`.sif`) approach. Use the standalone zip installation instead — it avoids this issue.

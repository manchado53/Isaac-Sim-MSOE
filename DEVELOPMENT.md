# Developer guide: running on Rosie

This is the team's setup guide for running Isaac Sim 5.1 headless on MSOE's Rosie cluster. For the project overview, see the [README](README.md).

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
git lfs install
git clone https://github.com/manchado53/revit-to-robot.git
cd revit-to-robot
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
sbatch scripts/slurm/submit_isaac_job.sh simulations/robot/jetbot_hospital_demo.py
sbatch scripts/slurm/submit_isaac_job.sh simulations/jetbot_hospital_overhead.py
sbatch scripts/slurm/submit_isaac_job.sh simulations/testbuilding_overhead.py
sbatch scripts/slurm/submit_isaac_job.sh simulations/robot/franka_camera_demo.py
```

### Monitor progress

```bash
squeue -u <username>@ad.msoe.edu            # check job status
tail -f logs/isaac_<job_id>.out              # live log output
```

### Output

Videos are saved to `~/revit-to-robot/output/<simulation_name>/` when the job completes.

To target a specific node:
```bash
sbatch --nodelist=dh-node17 scripts/slurm/submit_isaac_job.sh simulations/jetbot_hospital_demo.py
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

## Option C: WebRTC Streaming from a Remote Machine (Non-Rosie)

Stream Isaac Sim's GUI from a standalone server (not Rosie) to a remote computer over Tailscale or any network where both machines can reach each other.

```
[Your PC] <--Tailscale / VPN / LAN--> [Server + GPU]
 Streaming Client                       Isaac Sim headless
```

### Requirements

- Isaac Sim 5.1.0+ installed on the server at `~/isaacsim/`
- NVIDIA GPU with NVENC support (RTX series)
- Isaac Sim WebRTC Streaming Client on your local machine ([download here](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/download.html))
- Network connectivity between the two machines (same LAN, VPN, or Tailscale)

### Step C1: Find the server's IP address

Find the server's reachable IP:

```bash
# Same LAN:
hostname -I | awk '{print $1}'

# Tailscale:
tailscale ip -4
```

### Step C2: Start Isaac Sim in streaming mode

SSH into the server and run:

```bash
cd ~/isaacsim
./isaac-sim.streaming.sh \
  --/app/window/hideUi=false \
  --/app/window/drawMouse=true \
  --/app/livestream/allowResize=true \
  --/app/livestream/nvcf/quitOnSessionEnded=false \
  --/app/livestream/publicEndpointAddress=<SERVER_IP> \
  --/app/livestream/port=49100 \
  --/app/renderer/skipWhileMinimized=false \
  --/app/renderer/sleepMsOutOfFocus=0
```

Replace `<SERVER_IP>` with your server's IP.

> **Important:** The `publicEndpointAddress` flag is required when the client is on a different network. Without it, Isaac Sim advertises its local LAN IP for the media stream, which remote clients can't reach.

Wait until you see `Isaac Sim Full Streaming App is loaded` in the output, or verify with:

```bash
curl http://localhost:8011/v1/streaming/ready
# Should return: {"statusMessage":"Status: Ready for connection"}
```

### Step C3: Connect with the Streaming Client

1. Open the Isaac Sim WebRTC Streaming Client on your local machine
2. Enter the server IP
3. Click **Connect**
4. Isaac Sim's full GUI should appear within a few seconds

### Required Ports

| Port  | Protocol | Purpose              |
|-------|----------|----------------------|
| 49100 | TCP      | WebRTC signaling     |
| 47998 | UDP      | WebRTC media stream  |
| 8011  | TCP      | Health/status API    |

These ports must be reachable from your local machine. Tailscale and most VPNs handle this automatically.

### Troubleshooting

**Black screen then disconnect** — You're missing `--/app/livestream/publicEndpointAddress=<SERVER_IP>`. This is the most common issue when connecting from a different network.

**Client can't connect at all** — Check that port 49100 is reachable: `nc -z <SERVER_IP> 49100`. If using Tailscale, verify both machines are online with `tailscale status`.

**App quits when client disconnects** — Add `--/app/livestream/nvcf/quitOnSessionEnded=false` to keep Isaac Sim running between sessions.

**Health check returns "Rtx not ready"** — Isaac Sim is still starting up. Wait a few more seconds and try again.

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

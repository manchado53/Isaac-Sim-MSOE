# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Project Overview

This is a **research project** using NVIDIA Isaac Sim on MSOE's Rosie supercomputer. We are building a **general-purpose visual building agent** — a VLM-equipped mobile robot that autonomously patrols building interiors and detects visual anomalies, trained entirely on synthetic data generated from the building's digital twin. The showcase use case is **autonomous security patrol**, developed in partnership with MSOE's Public Safety Office.

### Research Goal

Use a BIM-derived digital twin as an automated dataset factory — generating unlimited labeled anomaly scenarios to train a VLM-equipped mobile robot that visually understands building environments, without a single manually labeled image.

1. Convert architectural BIM (Revit) data into a simulation-ready digital twin in NVIDIA Isaac Sim
2. Programmatically generate visual anomalies (open doors, broken glass, water leakage) and auto-label from physics/scene state
3. Fine-tune a VLM (Qwen2-VL-2B) on synthetic data via LoRA + vision projector training
4. Validate sim-to-real transfer on real-world photographs (Diercks Hall)

### Novelty

Nobody has published an end-to-end pipeline from BIM → digital twin → synthetic anomaly data → fine-tuned VLM → mobile robot deployment. See [RELATED_WORK.md](RELATED_WORK.md) and [PROJECT_PLAN.md](PROJECT_PLAN.md) for full literature review and project scope.

### Target Environment

**Robert D. Kern Engineering Innovation Center** — a multi-story university engineering facility currently under construction at MSOE. The digital twin is `MSOEBuildingV4_physics.usd` (160 MB, Revit export).

### Pipeline

1. **BIM-to-Sim Conversion**: Revit → USD → Isaac Sim import
2. **Environment Augmentation**: Collision geometry, semantic annotations, sensor-realistic rendering
3. **Navigation Graph Extraction**: Topological graph from building geometry (`floor_4_nav_graph_manual.json`)
4. **Robot Training (RL)**: Nova Carter with LiDAR + camera, trained via PPO (Stable-Baselines3 + IsaacLab)
5. **Anomaly Simulation**: Programmatically generate anomalies in the digital twin (open doors, broken glass, water leakage) with auto-labels from scene state
6. **Synthetic Data Generation**: Robot-mounted and static cameras capture anomaly frames → multi-anomaly VLM fine-tuning dataset
7. **VLM Fine-Tuning**: Qwen2-VL-2B-Instruct adapted for sim imagery via LoRA + vision projector training
8. **Sim-to-Real Validation**: Evaluate on real-world photographs (Diercks Hall) to prove transfer

### Target Anomalies

Identified in partnership with MSOE Public Safety (top 5 things officers look for during walkthroughs):

| Anomaly | Status | Sim Approach | Auto-Label |
|---|---|---|---|
| **Open doors** | Working (96.3% in-dist, expanding) | Rotate door mesh via 4×4 matrix | Door angle from physics |
| **Broken glass** | Planned | Material swap to cracked variant + floor debris | Scene state (material applied or not) |
| **Water leakage** | Stretch goal | Reflective water plane on floor | Scene state (puddle placed or not) |
| Vandalism | Out of scope | — | — |
| Burnt out lights | Out of scope | — | — |

---

## Current Status (as of April 2026)

### What Works
- Isaac Sim 5.1.0-rc.19 installed and verified on Rosie T4/V100 nodes
- Building USD loaded and navigable
- Data collection pipeline: `door_static_collection.py` → 1,800 frames (1 door, 3 cameras)
- VLM fine-tuning pipeline: LoRA r=16 + vision projector → **96.3% accuracy on held-out test set**
- Evaluation framework with 70/15/15 stratified train/val/test split
- ~99 SLURM jobs run, ~6.6 GPU-hours consumed
- Project scope finalized: general-purpose visual building agent, multi-anomaly detection
- Public safety partnership established — real operational requirements gathered
- Literature review complete — see [RELATED_WORK.md](RELATED_WORK.md)

### Active Work (April 11–25, 2026)
- **Expand door dataset**: 5–8 doors with lighting/material domain randomization to fix generalization failure (0% on unseen doors)
- **Add broken glass anomaly**: Material swap + floor debris simulation
- **Add water leakage anomaly** (stretch): Reflective water plane on floor
- **Real-world evaluation**: Photograph doors in Diercks Hall, evaluate sim-trained VLM on real images
- **Generalization curve**: Plot accuracy vs. training diversity (1 door → 3 → 5 → 8)
- See [PROJECT_PLAN.md](PROJECT_PLAN.md) for full 2-week execution plan
- See [EXPERIMENTS.md](EXPERIMENTS.md) for experimental design (baselines, CNN comparison, occlusion, sim-to-real)

### Known Issues
- **Generalization failure**: Fine-tuned model scores **0%** on unseen door `Doors_13185454` — model learned door-specific appearance, not the concept of open vs. closed. This is consistent with InspectVLM (2025) findings on VLM brittleness.
- NavMesh generation for full building navigation graph not yet complete

---

## Key Results

### VLM Door State Classification

| Model | Overall Acc | CLOSED Acc | OPEN Acc | F1 |
|---|---|---|---|---|
| Baseline (zero-shot Qwen2-VL-2B) | 50.0% | 0.0% | 100.0% | 0.667 |
| Fine-tuned (LoRA + projector) | **96.3%** | 92.6% | 100.0% | **0.964** |

- **Test set**: 270 held-out frames, never seen during training or hyperparameter selection
- **Error analysis**: All errors fall in the 1°–14° door angle range (visually ambiguous boundary)
- **Generalization test**: 0% on unseen door `Doors_13185454` — important negative result

### Dataset

| Split | CLOSED | OPEN | Total |
|---|---|---|---|
| Train | 630 | 630 | 1,260 (70%) |
| Val | 135 | 135 | 270 (15%) |
| Test (held out) | 135 | 135 | 270 (15%) |
| **Total** | **900** | **900** | **1,800** |

---

## VLM Fine-Tuning — Technical Details

### Architecture

| Component | Parameters | Status |
|---|---|---|
| Vision Encoder (ViT) | ~400M | **Frozen** |
| Vision Projector (`visual.merger`) | ~34M | **Trained — full weights** |
| LLM Backbone (28 layers, q+v LoRA) | ~2.2M trainable | **LoRA r=16** |
| **Total trainable** | **36.3M / 2.2B** | **1.62%** |

### Critical Engineering Detail — `modules_to_save`

The single most important lesson in this project. **Without `modules_to_save=["model.visual.merger"]` in the LoRA config, PEFT only saves the LoRA delta (~8 MB) and silently discards the trained projector weights.** At inference, the base (untrained) projector is restored — the model reverts to all-OPEN predictions (50% accuracy).

```python
# CORRECT — in finetune_qwen2vl.py
peft_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    modules_to_save=["model.visual.merger"],  # CRITICAL: saves 34M projector weights
    task_type="CAUSAL_LM",
)
```

Adapter size with fix: **139 MB** (vs 8.4 MB without).

### fp16 + AMP: Cast Projector to fp32

The model loads in float16. The AMP gradient scaler requires float32 for trainable parameters. After `get_peft_model()`:

```python
for name, param in model.named_parameters():
    if "merger" in name and param.requires_grad:
        param.data = param.data.float()
```

### Inference — Do NOT Use `merge_and_unload()`

`PeftModel.merge_and_unload()` crashes with `modules_to_save`. Use `PeftModel` directly:

```python
from peft import PeftModel
model = PeftModel.from_pretrained(base_model, "data/qwen2vl_door_lora")
# Do NOT call model.merge_and_unload()
```

### Training Hyperparameters

| Param | Value |
|---|---|
| Epochs | 3 |
| Batch size | 2 (per device) |
| Gradient accumulation | 4 → effective batch 8 |
| Learning rate | 1e-5 |
| Scheduler | Cosine |
| Warmup ratio | 0.05 |
| Hardware | V100 32 GB (`dgx` partition) |
| Runtime | ~16 minutes |

### Loss Curve (job 239013, checkpoint-474)

| Epoch | Train Loss | Val Loss |
|---|---|---|
| 1 | 0.01208 | 0.0083 |
| 2 | 0.003136 | 0.0037 |
| 3 | 0.002324 | **0.0024** |

Val loss decreases every epoch — no overfitting. Model has capacity for more epochs.

---

## Data Collection — `door_static_collection.py`

### Door Being Collected

- **Door prim**: `Doors_13606713` (aluminum/glass curtain wall door, Floor 4)
- **Hinge**: X = −2047.728 cm, Y = 7969.07 cm (south edge)
- **Label threshold**: angle ≤ 5° → `[CLOSED]`, angle > 5° → `[OPEN]`

### Camera Arc

```
DOOR_CENTER_XY  = [-20.48, 80.15]   # world XY (metres)
CAMERA_RADIUS   = 2.0               # metres from door
CAMERA_Z        = 27.8              # Floor 4 height
CAM_ARC_OFFSETS = [-30, -15, 0]    # degrees; northern arc blocked by wall
```

### Door Rotation — Direct 4×4 Matrix Override

USD Revit exports use a single `xformOp:transform` per mesh. Rotation is applied by rewriting this matrix directly (parent xform changes are unreliable due to sublayer USD composition):

```
M = | cos  sin  0  0 |
    |-sin  cos  0  0 |
    |  0    0   1  0 |
    | tx   ty  tz  1 |
```

---

## NFS / SLURM Gotchas

### File Write Race (NFS)

DGX compute nodes may see stale/partial NFS writes from the login node. **Always run `prepare_finetune_data.py` on the compute node** (inside the SLURM job), not on the login node. Use `os.fsync()` after writing:

```python
with open(path, "w") as f:
    for rec in records:
        f.write(json.dumps(rec) + "\n")
    f.flush()
    os.fsync(f.fileno())  # force NFS write-through
```

### conda vs Isaac Sim Python

- **Isaac Sim scripts**: use `~/isaacsim/python.sh`. Deactivate any conda env first (the `submit_isaac_job.sh` template does this automatically).
- **VLM fine-tuning / eval**: use conda `vlm` environment — has `transformers`, `peft`, `torch`
- Never mix them in the same SLURM job step

### Isaac Sim cannot run on the `dgx` (V100) partition

`dgx` compute nodes run an older OS (GLIBC ≤ 2.31) but Isaac Sim 5.1's `libcarb.so` requires **GLIBC ≥ 2.34** and **GLIBCXX_3.4.30**. `python.sh` fails immediately with:

```
libc.so.6: version `GLIBC_2.34' not found (required by .../isaacsim/kit/libcarb.so)
libstdc++.so.6: version `GLIBCXX_3.4.30' not found
```

This is **not** a conda problem — it's the node OS. Consequences:
- **Isaac Sim** runs on `teaching` / `batch` (T4, Ubuntu 24.04) or `dgxh100` (H100). Use [submit_isaac_job.sh](submit_isaac_job.sh) for T4 or [submit_isaac_dgx.sh](submit_isaac_dgx.sh) (now targets `dgxh100`) for heavier GPU work like RTX Path Tracing.
- **VLM fine-tuning** runs fine on `dgx` because it uses the conda `vlm` env, not `python.sh`, and torch does not require the GLIBC versions Isaac Sim's libcarb does.
- Don't try to "fix" this by bumping GLIBC — it's an OS-level constraint on the compute node.

### RTX Path Tracing on T4 is unstable

The `teaching` / `batch` T4 nodes can boot Isaac Sim fine, but the RTX Path Tracing renderer crashes during accumulation on T4 (seen with `material_comparison.py` at 64 samples/px — died silently inside Hydra/Vulkan during the settle loop). For path-tracing workloads use `dgxh100` via `submit_isaac_dgx.sh`. For real-time RTX, T4 is fine.

### Variable Order in Shell Scripts

In `submit_finetune_job.sh`, the `PYTHON` variable must be defined **before** it is used. This caused job 239012 to fail with "unbound variable". Always define `PYTHON=...` at the top of the script before any code that references it.

---

## Rosie Supercomputer

When the user asks about Rosie (MSOE's supercomputer), including topics like SLURM, job submission, GPU access, modules, storage, or cluster usage:
1. Use the WebFetch tool to check https://msoe.dev/#/ for relevant documentation
2. Follow any links on that page that are pertinent to the user's specific question
3. Base your answers on the fetched documentation rather than general knowledge

**Rosie Web Portal**: https://dh-ood.hpc.msoe.edu/

### GPU Partitions

| Partition | GPU | Time Limit | Use Case |
|---|---|---|---|
| `teaching` | T4 | 7 days | Isaac Sim development, testing |
| `batch` | T4 | 2 days | Production Isaac Sim runs |
| `dgx` | V100 32 GB | 21 days | VLM fine-tuning, large simulations |
| `dgxh100` | H100 | 21 days | Maximum performance |

**Fine-tuning requires `dgx` partition** — V100 32 GB needed for Qwen2-VL-2B with fp16 + visual tokens.

---

## NVIDIA Isaac Sim

When the user asks about Isaac Sim, including topics like installation, setup, simulation, robotics, Omniverse, or related workflows:
1. First check [NVIDIA_Isaac_Sim_Reference.md](NVIDIA_Isaac_Sim_Reference.md) for comprehensive API and usage reference
2. Use the WebFetch tool to check https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/download.html for anything not covered in the reference
3. Base your answers on the reference doc and fetched documentation rather than general knowledge

**Installed Version**: 5.1.0-rc.19  
**Installation Path**: `~/isaacsim`  
**Python Version**: 3.11.13 (bundled)

### Running Isaac Sim on Rosie

```bash
# In SLURM job or interactive GPU session:
cd ~/isaacsim
export OMNI_KIT_ACCEPT_EULA=YES
./python.sh path/to/script.py
```

### Renderer

All scripts currently use `SimulationApp({"headless": True})` — defaults to **RTX Real-Time** mode (rasterization + approximate ray tracing). To enable **RTX Path Tracing** (physically-based, photorealistic, ~2s/frame):

```python
simulation_app = SimulationApp({
    "headless": True,
    "renderer": "PathTracing",
    "samples_per_pixel_per_frame": 64,
})
```

Path tracing is slower but produces frames closer to real camera output — useful for closing the sim-to-real gap.

### Important Notes

- Isaac Sim MUST run on GPU nodes — crashes on login nodes
- Typical runtime: ~80s on T4 (40s startup + simulation)
- Print statements go to Isaac Sim internal logs, not SLURM stdout — check exit code (0 = success)
- Harmless warnings: `pulse` audio directory, `omni.isaac.*` deprecation notices, `carb.audio.*`

---

## Development Workflow

### Batch Jobs (recommended)

```bash
cd ~/isaac-sim-msoe
sbatch submit_isaac_job.sh [optional_script.py]   # Isaac Sim
sbatch submit_finetune_job.sh                      # VLM fine-tuning
sbatch submit_baseline_eval.sh                     # Evaluation
```

### Monitoring

```bash
squeue -u $USER                       # Check job status
cat logs/isaac_<JOBID>.out            # Isaac Sim logs
cat logs/finetune_<JOBID>.out         # Fine-tuning logs
cat logs/baseline_eval_<JOBID>.out    # Eval logs
```

### Interactive Development

```bash
srun --partition=teaching --gres=gpu:1 --time=2:00:00 --pty bash
cd ~/Digital-twin-v2/isaac-sim-msoe
./run_isaac_interactive.sh simulations/robot/your_script.py
```

---

## File Organization

```
~/isaacsim/                              # Isaac Sim install (DO NOT MODIFY)
~/Digital-twin-v2/isaac-sim-msoe/       # THIS REPO
│
├── MSOEBuildingV4_physics.usd        # Building digital twin (160 MB, Revit export)
├── floor_4_nav_graph_manual.json        # Floor 4 navigation graph
│
├── simulations/
│   ├── rigid_body/                      # Physics demos (hello_isaac, bouncing_balls)
│   ├── environment/                     # Building environment setup scripts
│   └── robot/
│       ├── door_static_collection.py    # ** Main data collection script **
│       ├── prepare_finetune_data.py     # NPZ → stratified JSONL + JPEG (70/15/15)
│       ├── finetune_qwen2vl.py          # ** LoRA + projector fine-tuning **
│       ├── eval_baseline.py             # ** Evaluation on test.jsonl **
│       ├── nova_carter_vlm_demo.py      # Phase 1: patrol + frame capture
│       ├── nova_carter_vlm_phase23.py   # Phase 2+3: VLM inference + video
│       ├── nova_carter_vlm_generalization.py  # Generalization test (unseen door)
│       ├── eval_real_doors.py           # Eval on real door photos
│       └── plot_*.py                    # Matplotlib visualization scripts
│
├── training/
│   ├── train_nova_carter.py             # PPO training for Nova Carter
│   ├── train_ridgeback.py               # PPO training for Ridgeback
│   ├── nova_carter_patrol/              # IsaacLab env: actions, rewards, observations
│   ├── ridgeback_patrol/                # IsaacLab env for Ridgeback
│   └── agents/sb3_ppo_cfg.yaml          # PPO hyperparameters
│
├── sim_utils/                           # Shared Python utilities
│   ├── video_capture.py                 # VideoRecorder → MP4
│   ├── follow_camera.py                 # FollowCamera (tracks moving robot)
│   └── building_physics.py             # Collision geometry utilities
│
├── data/
│   ├── door_labels_static_multi_camera.npz  # 1800 frames (653 MB)
│   ├── finetune_images/                 # 1800 JPEGs (93 MB)
│   ├── train.jsonl / val.jsonl / test.jsonl  # 1260 / 270 / 270 samples
│   ├── qwen2vl_door_lora/               # Adapter (139 MB: LoRA + merger weights)
│   ├── baseline_eval_results.json       # 50% accuracy
│   ├── finetuned_eval_results.json      # 96.3% accuracy
│   └── real_doors/                      # Real door photos for eval
│
├── submit_isaac_job.sh                  # SLURM: Isaac Sim (teaching/batch, T4)
├── submit_finetune_job.sh               # SLURM: VLM fine-tuning (dgx, V100)
├── submit_baseline_eval.sh              # SLURM: evaluation (dgx, V100)
├── run_isaac_interactive.sh             # Interactive GPU session runner
│
├── PROJECT_PLAN.md                      # ** Master project plan — scope, novelty, 2-week execution **
├── EXPERIMENTS.md                       # ** Experimental design — all experiments, protocols, expected results **
├── RELATED_WORK.md                      # Literature review — commercial + academic, identified gaps
├── NVIDIA_Isaac_Sim_Reference.md        # Comprehensive Isaac Sim & Isaac Lab API/usage reference
├── VLM_fine-tuning.md                   # Technical deep-dive on door classification fine-tuning
├── DATASET_IMPROVEMENT_PLAN.md          # Plan to expand door dataset + add broken glass / water leakage
├── Anomaly-Detection-Plan.md            # Earlier exploration: regression-based door angle estimation
└── real_doors/                          # Real door photographs for generalization testing
```

---

## Simulation Script Template

```python
#!/usr/bin/env python3
import os, sys
os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from sim_utils.video_capture import VideoRecorder

def main():
    world = World()
    world.scene.add_default_ground_plane()
    world.reset()

    with VideoRecorder("output_name") as recorder:
        for i in range(300):  # 10s at 30 FPS
            world.step(render=True)
            recorder.capture()

    simulation_app.close()

if __name__ == "__main__":
    main()
```

**Note**: Use `isaacsim.core.api` imports (not `omni.isaac.core` — deprecated but still functional).

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `CUDA error: no device detected` | Running on login node | Use GPU node via `srun` or SLURM |
| Exit code 139 (segfault) | No GPU allocated | Add `--gres=gpu:1` to SLURM |
| Print statements missing | Isaac Sim captures stdout | Check exit code; use log timestamps |
| `ModuleNotFoundError: sim_utils` | Path not set | Add `sys.path.insert(0, project_root)` |
| JSONDecodeError on train.jsonl | NFS cache stale | Run `prepare_finetune_data.py` on compute node with `os.fsync()` |
| VLM predicts all OPEN (50%) | Projector weights not saved | Ensure `modules_to_save=["model.visual.merger"]` in LoRA config |
| `merge_and_unload()` crash | Incompatible with `modules_to_save` | Remove call; use `PeftModel` directly |
| `unbound variable: PYTHON` | Variable used before definition | Move `PYTHON=...` to top of shell script |
| Job pending a long time | GPU nodes busy | Try `dgxh100` partition or wait |

---

## Documentation Policy

When making major changes (reward weights, env config, new reward terms, termination changes, hyperparameter tuning, architecture changes, dataset changes, etc.), always update:
- **`CLAUDE.md`**: If it affects workflow, file organization, or project-level context
- **`README.md`**: If it affects how others use or understand the project
- **`VLM_fine-tuning.md`**: If it affects the fine-tuning pipeline, results, or architecture
- **Code docstrings/comments**: Inline docs in changed files

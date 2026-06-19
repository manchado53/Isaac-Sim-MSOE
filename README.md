# BIM → Digital Twin → VLM Anomaly Detection

Research project at the Milwaukee School of Engineering (MSOE). We build a pipeline that converts architectural BIM data (Revit) into an NVIDIA Isaac Sim digital twin, generates unlimited labeled synthetic training data, fine-tunes a vision-language model on it, and validates the result on real-world photographs — without a single manually labeled image.

**Showcase task**: autonomous security patrol of the Robert D. Kern Engineering Innovation Center, developed in partnership with MSOE's Public Safety Office.

---

## Key Results

Fine-tuned **Qwen3-VL-2B** (LoRA r=16 + visual projector) on synthetic data from the digital twin:

| Evaluation | Model | Accuracy | Notes |
|---|---|---|---|
| Synthetic doors (in-dist) | Fine-tuned | **97.4%** | 4,472 val samples, 8 door types |
| Synthetic glass (in-dist) | Fine-tuned | **97.2%** | 98.8% cracked / 95.6% intact |
| Unseen door holdout | Fine-tuned | **90.2%** | 7,776 samples, 2 doors never in training |
| Real doors — sim-to-real | Fine-tuned | **93.25%** | 400 real-world photographs |
| Real doors — Diercks Hall | Fine-tuned | **93%** | 100 photos from partner building |
| Real doors — zero-shot | Qwen3-VL-2B (base) | 94.25% | Sim-trained model transfers without catastrophic forgetting |
| Combined door+glass | Baseline → Fine-tuned | 75% → **87%** | Real mixed-anomaly scenario |

**Zero-shot baseline on synthetic images**: 50% (all-OPEN predictions — base model vision projector cannot map sim textures). Fine-tuning the projector directly resolves the domain gap.

---

## What This Is

### Novelty

No prior work has published an end-to-end pipeline from **BIM → digital twin → synthetic anomaly data → fine-tuned VLM → mobile robot deployment**. The building's own Revit model becomes an automated dataset factory.

### Anomalies

Identified with MSOE Public Safety (top things officers check during walkthroughs):

| Anomaly | Status | Synthetic Approach |
|---|---|---|
| **Open doors** | Working | Rotate door mesh via 4×4 matrix; angle → label |
| **Broken glass** | Working | Material swap to cracked variant + evaluated on real composites |
| Water leakage | Stretch goal | Reflective water plane on floor |

### Target Environment

**Robert D. Kern Engineering Innovation Center** — 97,000 sq ft facility currently under construction at MSOE. Digital twin: `MSOEBuildingV4_physics.usd` (Revit → USD export, tracked via Git LFS).

---

## Pipeline

```
Revit BIM
    │
    ▼
NVIDIA Isaac Sim (Digital Twin)
    │  Programmatic anomaly injection (open doors, cracked glass)
    │  Auto-label from scene state (door angle, material applied)
    ▼
Synthetic Dataset
    │  domain randomization: 8 door types, 3 lighting modes, 3 camera angles
    ▼
VLM Fine-Tuning  (Qwen3-VL-2B, LoRA r=16 + visual projector)
    │  train on Rosie DGX (V100 32 GB), ~2 hrs
    ▼
Sim-to-Real Validation
    │  real photographs of Diercks Hall doors + glass
    ▼
Mobile Robot Deployment  (Nova Carter, Isaac Sim patrol demo)
```

---

## Quickstart

See [QUICKSTART.md](QUICKSTART.md) for environment setup and one-liners. Requires:
- MSOE Rosie HPC access (SLURM)
- Isaac Sim 5.1.0 installed at `~/isaacsim/`
- `vlm` conda environment (transformers, peft, torch)

### Run data collection (Isaac Sim, T4 node)

```bash
sbatch submit_production_collection.sh
```

### Prepare unified door+glass training data

```bash
sbatch submit_prepare_unified.sh
```

### Fine-tune Qwen3-VL-2B (V100 node, ~2 hrs)

```bash
sbatch submit_finetune_unified.sh
```

### Evaluate

```bash
# Synthetic val set (door + glass)
sbatch submit_eval_unified.sh

# Holdout (unseen doors)
sbatch submit_eval_qwen3vl_unified_v4_holdout.sh

# Real-world photographs
sbatch submit_eval_qwen3vl_unified_v4_real_doors.sh
```

### Run patrol demo

```bash
# Phase 1: robot patrols, captures frames
sbatch submit_isaac_job.sh simulations/robot/nova_carter_vlm_demo.py

# Phase 2+3: VLM inference + annotated video
sbatch submit_infer_single.sh
```

---

## Repository Structure

```
isaac-sim-msoe/
│
├── simulations/robot/                    # All VLM pipeline scripts
│   ├── door_production_collection.py     # Multi-door synthetic data collection (v4)
│   ├── prepare_unified_data.py           # NPZ → stratified door+glass JSONL (70/15/15)
│   ├── finetune_qwen3vl.py               # Qwen3-VL-2B LoRA + projector fine-tuning
│   ├── finetune_qwen2vl.py               # Qwen2-VL-2B variant
│   ├── eval_unified.py                   # Unified door+glass evaluation
│   ├── eval_real_doors.py                # Sim-to-real door evaluation
│   ├── eval_real_glass.py                # Real glass composite evaluation
│   ├── nova_carter_vlm_demo.py           # Phase 1: patrol + frame capture
│   ├── nova_carter_vlm_phase23.py        # Phase 2+3: VLM inference + video
│   ├── nova_carter_vlm_generalization.py # Generalization test (unseen door)
│   ├── nova_carter_door_inspection.py    # Door inspection routine
│   └── plot_*.py / make_*.py             # Paper figures
│
├── simulations/robot/[diagnostics]       # One-off diagnostic scripts (not for publication)
│   └── door_hinge_*.py, material_*.py, floor4_*.py, etc.
│
├── sim_utils/                            # Shared utilities
│   ├── video_capture.py                  # VideoRecorder → MP4
│   ├── follow_camera.py                  # FollowCamera (tracks moving robot)
│   └── building_physics.py              # Collision geometry utilities
│
├── training/                             # RL training (Nova Carter PPO, IsaacLab)
│
├── data/                                 # Evaluation results + JSONL indexes
│   ├── train_unified.jsonl               # 18,915 door+glass training samples
│   ├── train_v4.jsonl                    # 25,336 door-only samples
│   ├── val_unified_door.jsonl            # Door val split
│   ├── val_unified_glass.jsonl           # Glass val split
│   ├── holdout_v4.jsonl                  # 7,776 unseen-door holdout samples
│   ├── val_real_doors.jsonl              # 400 real-world door images
│   ├── dh_doors_test.jsonl               # 100 Diercks Hall door images
│   ├── eval_unified_door.json            # 97.4% result
│   ├── eval_unified_glass.json           # 97.2% result
│   ├── holdout_eval_qwen3vl_2b_unified_v4_finetuned.json   # 90.2% holdout
│   ├── real_doors_eval_qwen3vl_2b_unified_v4_finetuned.json  # 93.25% real
│   ├── eval_unified_door_accuracy.png    # Paper figure
│   └── eval_unified_glass_accuracy.png  # Paper figure
│
├── data_glass/downsample_glass_val_test.py  # Glass data preprocessing
├── dh_glass_panels/                      # 23 real Diercks Hall glass photographs
├── val_real_doors/                       # ~2,500 real-world door photographs
│
├── submit_*.sh                           # SLURM job scripts
├── floor_4_nav_graph_manual.json         # Floor 4 topological navigation graph
│
├── PROJECT_PLAN.md                       # Full scope, novelty, execution plan
├── EXPERIMENTS.md                        # Experimental design (baselines, ablations)
├── RELATED_WORK.md                       # Literature review
├── VLM_fine-tuning.md                    # Technical fine-tuning details + gotchas
├── SELECTED_TRAINING_DOORS.md           # Door curation rationale
├── DATASET_PRODUCTION_PLAN.md           # Dataset pipeline design
├── LIGHTING_AND_MATERIALS.md            # Domain randomization details
└── CLAUDE.md                             # Full system context (key results, gotchas, workflow)
```

> **Large files** (images, NPZ, model adapters) are excluded from git — see `.gitignore`.  
> Model adapter weights: see below.

---

## Model Weights

The fine-tuned adapter (`qwen3vl_unified_lora_v4`, ~400 MB) is not stored in this repo.

To use it:
1. Copy from Rosie: `data/qwen3vl_unified_lora_v4/adapter_model.safetensors`
2. Or request access from the project team

To load at inference:

```python
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import PeftModel

base = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-2B-Instruct", torch_dtype="float16"
)
model = PeftModel.from_pretrained(base, "data/qwen3vl_unified_lora_v4")
# Do NOT call model.merge_and_unload() — incompatible with modules_to_save
```

---

## Critical Engineering Note — `modules_to_save`

Without `modules_to_save=["model.visual.merger"]` in the LoRA config, PEFT silently discards the 34M trained projector weights — the model reverts to 50% accuracy at inference. This is the single most important lesson in this project. See [VLM_fine-tuning.md](VLM_fine-tuning.md) for full details.

---

## Infrastructure

- **Cluster**: MSOE Rosie HPC (SLURM)
- **Isaac Sim** (data collection): `teaching` / `batch` partition, T4 GPU
- **VLM fine-tuning**: `dgx` partition, V100 32 GB
- **Isaac Sim version**: 5.1.0-rc.19 at `~/isaacsim/`
- **VLM conda env**: `vlm` (transformers, peft, torch)

See [CLAUDE.md](CLAUDE.md) for the complete reference including SLURM gotchas, partition constraints, and troubleshooting.

---

## Setup

```bash
# Clone
git clone https://gitlab.com/manchado53/isaac-sim-msoe.git
cd isaac-sim-msoe

# Install Python dependencies (for VLM work)
pip install -r requirements.txt

# Isaac Sim (one-time, on Rosie)
# See QUICKSTART.md
```

# Qwen2-VL Fine-Tuning for Door State Classification

> **Note (April 2026):** This document covers the initial door state classification pipeline. The project has since expanded to multi-anomaly detection (open doors, broken glass, water leakage) as part of a general-purpose visual building agent. See [PROJECT_PLAN.md](PROJECT_PLAN.md) for the current scope and [DATASET_IMPROVEMENT_PLAN.md](DATASET_IMPROVEMENT_PLAN.md) for the expanded dataset plan. The fine-tuning architecture and engineering lessons documented here remain current and apply to all anomaly types.

**Model:** Qwen2-VL-2B-Instruct + LoRA + Vision Projector
**Task:** Binary classification — door `[OPEN]` vs `[CLOSED]` (expanding to multi-anomaly)
**Data Source:** NVIDIA Isaac Sim (MSOE Digital Twin)
**Hardware:** Tesla V100 32 GB — MSOE Rosie `dgx` partition

---

## Overview

The base Qwen2-VL-2B-Instruct model, when asked to classify door state from Isaac Sim renders, predicted **OPEN 100% of the time** regardless of true label. This is a sim-to-real domain gap: the model was pre-trained on real-world photographs and its vision projector cannot map synthetic sim textures to meaningful language representations.

Fine-tuning the LLM backbone alone (LoRA only) is insufficient — the projector remains frozen and the LLM never receives useful visual information. The solution is to **train both the LoRA adapter and the vision projector simultaneously**, and critically, to save both sets of weights correctly at the end of training.

---

## 1. Data Collection

**Script:** `simulations/robot/door_static_collection.py`

### Environment Setup

A door in the MSOE Digital Twin (Isaac Sim 5.1.0) is held at a fixed angle for each episode while cameras capture frames. A rotation sensor records the exact door angle in degrees, which is used as the ground-truth label.

**Label threshold:**
- Angle ≤ 5° → `[CLOSED]`
- Angle > 5° → `[OPEN]`

### Camera Arc Design

To increase viewpoint diversity, three cameras are placed in an arc around the door at the same height, each oriented to face the door centre:

| Camera | Arc Offset | Description |
|--------|-----------|-------------|
| Cam 0  | −30°      | South-west of door |
| Cam 1  | −15°      | South-south-west of door |
| Cam 2  |   0°      | Due south (original camera position) |

```python
DOOR_CENTER_XY  = [-20.48, 80.15]   # world XY coordinates of door centre
CAMERA_RADIUS   = 2.0               # metres from door centre to camera
CAMERA_Z        = 27.8              # all cameras at the same height
CAM_ARC_OFFSETS = [-30, -15, 0]    # degrees from +X axis (east)
```

Camera orientation: `yaw = offset + 180°` so each camera faces the door centre.

> **Why only the southern arc?** Offsets of +15° and +30° placed cameras partially inside the building wall geometry to the north of the door, producing invalid clipped views. Only the southern side had unobstructed line of sight.

### Collection Parameters

```
N_EPISODES      = 60   (30 CLOSED + 30 OPEN, strictly alternating)
FRAMES_PER_EP   = 10   frames per episode
N_CAMERAS       = 3

Total frames    = 60 × 10 × 3 = 1,800
CLOSED frames   = 900   (door angle 0°–5°)
OPEN frames     = 900   (door angle 10°–90°)
```

CLOSED episodes: door held at a random angle in [0°, 5°].
OPEN episodes: door held at a random angle in [10°, 90°].

**Output:** `data/door_labels_static_multi_camera.npz`
- `frames`: `(1800, 480, 640, 3)` uint8 RGB
- `angles`: `(1800,)` float32 door angle in degrees
- `cam_offsets`: `(1800,)` arc offset for each frame

---

## 2. Dataset Preparation

**Script:** `simulations/robot/prepare_finetune_data.py`

### Train / Validation / Test Split

> **Updated:** The code now uses a 70/15/15 stratified split (train/val/test). The 80/20 split shown in the original version of this document is outdated.

| Split | CLOSED | OPEN | Total |
|-------|--------|------|-------|
| Train | 630    | 630  | 1,260 (70%) |
| Val   | 135    | 135  |   270 (15%) |
| Test  | 135    | 135  |   270 (15%) |

### JSONL Conversation Format

Each sample is formatted as a two-turn chat for Qwen2-VL:

```json
{
  "conversations": [
    {
      "role": "user",
      "content": "<|vision_start|><|image_pad|><|vision_end|>Is this door open or closed?"
    },
    {
      "role": "assistant",
      "content": "Door Status: [CLOSED]."
    }
  ],
  "image": "finetune_images/frame_01202.jpg"
}
```

Frames are exported as JPEG at quality 92 into `data/finetune_images/` (~93 MB, 1,800 files).

---

## 3. Model Architecture and Freezing Strategy

### Qwen2-VL-2B-Instruct Components

| Component | Parameters | Training Status | Rationale |
|-----------|-----------|-----------------|-----------|
| Vision Encoder (ViT) | ~400M | **Frozen** | Preserves general visual features (edges, shapes) learned from real-world data |
| Vision Projector (`visual.merger`) | ~34M | **Trained — full weights** | Adapts sim-specific rendering to the LLM's embedding space |
| LLM Backbone (28 attention layers) | ~1.8B | **LoRA on q_proj + v_proj** | Learns the `[OPEN]`/`[CLOSED]` response format |

### Why the Vision Projector Must Be Trained

The projector (a PatchMerger) is the bridge between visual patch tokens and the LLM's vocabulary space. The base model's projector was optimised for real-world photographs. Isaac Sim renders have different lighting, geometry, and texture signatures. Without projector adaptation, the LLM receives visual embeddings that carry no meaningful information about the sim door state, and no amount of LoRA training can compensate.

---

## 4. LoRA Configuration

**Script:** `simulations/robot/finetune_qwen2vl.py`

```python
LORA_RANK       = 16              # conservative; increase to 32/64 if val plateaus
LORA_ALPHA      = 32              # effective scale = alpha / rank = 2.0
LORA_DROPOUT    = 0.05
TARGET_MODULES  = ["q_proj", "v_proj"]
modules_to_save = ["model.visual.merger"]   # saves full projector weights
```

### `modules_to_save` — The Critical Fix

`modules_to_save=["model.visual.merger"]` tells PEFT to:
1. Automatically mark the vision projector as trainable
2. **Save the full 34M projector weights** alongside the LoRA delta in `adapter_model.safetensors`
3. Restore them correctly when `PeftModel.from_pretrained` is called at inference

Without this, `model.save_pretrained()` on a PEFT model only saves the LoRA delta (~8 MB). The trained projector weights are silently discarded. At inference, `PeftModel.from_pretrained` restores the base (untrained) projector — the adapter and projector are a mismatched pair, and the model defaults to all-OPEN predictions.

| | Without `modules_to_save` | With `modules_to_save` |
|-|--------------------------|----------------------|
| Adapter size | 8.4 MB | **139 MB** |
| Projector at inference | Base (untrained) | **Fine-tuned** |
| Inference accuracy | 50% (all OPEN) | **93.3%** |

### fp32 Cast for AMP Compatibility

The model loads in `float16` for memory efficiency. PyTorch's AMP gradient scaler requires trainable parameters to be `float32`. After `get_peft_model()`, projector params are cast:

```python
for name, param in model.named_parameters():
    if "merger" in name and param.requires_grad:
        param.data = param.data.float()
```

### Trainable Parameter Count

| Group | Parameters | % of 2.2B |
|-------|-----------|-----------|
| LoRA (q_proj + v_proj × 28 layers) | ~2.2M | 0.10% |
| Vision projector (merger) | ~34M | 1.52% |
| **Total trainable** | **36,267,008** | **1.62%** |
| Frozen | ~2.21B | 98.38% |

---

## 5. Training Hyperparameters

| Hyperparameter | Value | Rationale |
|----------------|-------|-----------|
| Epochs | 3 | Val loss still decreasing — more epochs would likely help |
| Batch size (per device) | 2 | V100 32 GB memory constraint with visual tokens |
| Gradient accumulation steps | 4 | Effective batch size = 2 × 4 = **8** |
| Learning rate | 1e-5 | Conservative — prevents catastrophic forgetting |
| LR scheduler | Cosine | Smooth decay, avoids sharp LR drops |
| Warmup ratio | 0.05 | 5% of steps for LR warmup (27 steps) |
| Weight decay | 0.01 | L2 regularisation on LoRA weights |
| Precision | fp16 | Mixed precision; projector cast to fp32 |
| Max sequence length | 512 tokens | Covers image patches + question + answer |

**Training runtime:** ~16 minutes on a single V100 32 GB.

---

## 6. Training Results

### Loss Curve

| Epoch | Train Loss (step avg.) | Val Loss |
|-------|----------------------|----------|
| 1     | 0.01208              | 0.008373 |
| 2     | 0.003136             | 0.001641 |
| 3     | 0.002324             | **0.001131** ← best |

Val loss decreased every epoch. Val loss is consistently **below** train loss — expected because train loss is a running average over the epoch (including early high-loss steps) and dropout is active during training but disabled at eval.

### Did the Model Overfit?

**No.** Both train and val loss improved through all 3 epochs with no divergence. The model likely has headroom for additional epochs.

### Gradient Norm

A large spike (~1.9) in the first ~10 steps reflects the projector rapidly adapting its weights to sim imagery, then settling near zero by epoch 2 — characteristic of stable fine-tuning with a well-chosen learning rate.

![Training Curves](data/training_curves_238471.png)

---

## 7. Inference Evaluation

**Script:** `simulations/robot/eval_baseline.py`
**Dataset:** `data/door_labels_static_multi_camera.npz`
**Samples:** 1,800 (stratified — 900 CLOSED + 900 OPEN)

The adapter is loaded with `PeftModel.from_pretrained`. **Do NOT use `model.merge_and_unload()`** — it crashes with `modules_to_save`. Use the `PeftModel` directly for inference.

### Baseline vs Fine-Tuned Results

| Metric | Baseline | Fine-Tuned | Change |
|--------|----------|------------|--------|
| Overall Accuracy | 50.0% | **93.3%** | +43.3 pp |
| CLOSED Accuracy | 0.0% | **93.6%** | +93.6 pp |
| OPEN Accuracy | 100.0% | **93.1%** | −6.9 pp |
| Precision | 50.0% | **93.5%** | +43.5 pp |
| Recall | 100.0% | **93.1%** | −6.9 pp |
| F1 Score | 0.667 | **0.933** | +0.266 |
| Correct predictions | 900 / 1800 | **1680 / 1800** | +780 |

### Confusion Matrices

**Baseline** — predicted every frame as OPEN:

```
                Predicted OPEN   Predicted CLOSED
True OPEN            900                0
True CLOSED          900                0
```

**Fine-Tuned:**

```
                Predicted OPEN   Predicted CLOSED
True OPEN            838               62
True CLOSED           58              842
```

### Error Analysis

All 120 errors (6.7%) occur in the **1°–14° angle range**, straddling the 5° decision boundary. Errors are symmetric: 58 CLOSED predicted as OPEN, 62 OPEN predicted as CLOSED — no class bias. These are genuinely ambiguous frames; a door at 3° and one at 7° are visually nearly identical from 2 m away.

![Evaluation Comparison](data/eval_comparison.png)

---

## 8. Key Engineering Lessons

### The Silent Projector Bug

The most time-consuming issue was that the vision projector trained correctly (val loss 0.001) but its weights were never persisted. `PEFT.save_pretrained()` only saves LoRA deltas by default. The symptom was identical to the untrained baseline: all-OPEN predictions, 50% accuracy. The fix — `modules_to_save=["model.visual.merger"]` — caused the adapter file to grow from 8.4 MB to 139 MB, confirming the projector weights were now included.

### LoRA Must Be Applied Before Manual Unfreezing

`get_peft_model()` internally calls `mark_only_lora_as_trainable()`, which re-freezes **all** non-LoRA parameters. Any manual `requires_grad = True` calls made before `get_peft_model()` are overwritten. All unfreezing must happen after wrapping.

### fp16 Parameters Cannot Be Updated by AMP Grad Scaler

The model loads in `float16`. PyTorch's AMP gradient scaler requires `float32` parameters to correctly unscale gradients before optimizer updates. Projector params must be explicitly cast after LoRA wrapping: `param.data = param.data.float()`.

### PEFT `modules_to_save` vs Manual Unfreezing

`modules_to_save` wraps the target module in a `ModulesToSaveWrapper` containing two copies: a frozen original and a trainable adapter copy. Iterating `module.parameters()` on the wrapper yields both copies. Setting `requires_grad = True` on all of them doubles the trainable count (68M instead of 34M). The correct approach is to only cast/modify parameters that PEFT has already marked trainable (filter by `param.requires_grad`).

---

## 9. Generalization Results

### Out-of-Distribution Test (Unseen Door)

The fine-tuned model was evaluated on door `Doors_13185454` — a door not included in training data.

| Metric | In-Distribution (test set) | Out-of-Distribution (unseen door) |
|---|---|---|
| Accuracy | 96.3% | **0%** |

The model learned the specific appearance of `Doors_13606713`, not the abstract concept of open vs. closed. This is consistent with findings from InspectVLM (2025), which reported similar brittleness in fine-tuned VLMs — collapsing to memorized responses when presented with novel visual inputs.

### Implications

This generalization failure is the **central motivating result** for the project's current direction:
- Proves that naive single-source fine-tuning is insufficient
- Motivates the expanded dataset plan: 5–8 doors with domain randomization (lighting, materials, viewpoints)
- The generalization curve (accuracy vs. training diversity) is the key deliverable — see [PROJECT_PLAN.md](PROJECT_PLAN.md)

---

## 10. File Reference

| File | Purpose |
|------|---------|
| `simulations/robot/door_static_collection.py` | Isaac Sim data collection — 3-camera arc, 1800 frames |
| `simulations/robot/prepare_finetune_data.py` | NPZ → stratified JSONL + JPEG export |
| `simulations/robot/finetune_qwen2vl.py` | LoRA + projector fine-tuning |
| `simulations/robot/eval_baseline.py` | Inference evaluation (baseline and fine-tuned) |
| `simulations/robot/plot_training_curves.py` | Training loss / grad norm / LR visualisation |
| `simulations/robot/plot_eval_results.py` | Baseline vs fine-tuned comparison visualisation |
| `submit_finetune_job.sh` | SLURM batch job — fine-tuning (dgx partition) |
| `submit_baseline_eval.sh` | SLURM batch job — evaluation (dgx partition) |
| `data/door_labels_static_multi_camera.npz` | 1800 collected frames with angle labels |
| `data/train.jsonl` / `data/val.jsonl` | Training (1440) and validation (360) JSONL |
| `data/finetune_images/` | 1800 JPEG frames at quality 92 |
| `data/qwen2vl_door_lora/` | Saved adapter — LoRA + merger weights (139 MB) |
| `data/training_curves_238471.png` | Training curve visualisation (job 238471) |
| `data/eval_comparison.png` | Baseline vs fine-tuned results visualisation |

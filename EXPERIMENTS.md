# Experimental Design

> **Last updated:** 2026-04-11
> **Context:** See [PROJECT_PLAN.md](PROJECT_PLAN.md) for full project scope and [DATASET_IMPROVEMENT_PLAN.md](DATASET_IMPROVEMENT_PLAN.md) for dataset details.

All controlled experiments use **door state classification** as the benchmark task. Broken glass and water leakage are used to demonstrate multi-task generality, not for head-to-head comparisons.

---

## Experiment 1: Zero-Shot VLM Baselines

**Goal:** Prove that fine-tuning is necessary — off-the-shelf VLMs cannot classify door state from Isaac Sim renders, regardless of model size.

**Motivation:** A reviewer might ask: "Why not just use GPT-4o zero-shot?" This experiment shows that the sim-to-real domain gap is a fundamental problem that model scale alone does not solve.

### Models

| Model | Size | Rationale |
|---|---|---|
| Qwen2-VL-2B-Instruct | 2B | Our base model. Already tested: 50% accuracy (predicts OPEN for everything). |
| Qwen2-VL-7B-Instruct | 7B | Does 3.5x more parameters solve the domain gap without fine-tuning? |
| Frontier model via API (GPT-4o or Claude) | Very large | Best-case zero-shot. If even frontier models struggle on sim images, fine-tuning is clearly required. |

### Protocol

- Run inference on the full test set (270 held-out frames, 135 CLOSED + 135 OPEN)
- Same prompt for all models: `"Is this door open or closed?"`
- Report overall accuracy, per-class accuracy, F1 score
- Parse VLM responses for `OPEN`/`CLOSED` keywords (same logic as existing `eval_baseline.py`)

### Expected Results

| Model | Expected Accuracy | Reasoning |
|---|---|---|
| Qwen2-VL-2B | 50% (known) | All-OPEN predictions. Vision projector can't map sim textures. |
| Qwen2-VL-7B | ~50-60% | Larger LLM, but same projector architecture. May show marginal improvement from better language reasoning but fundamentally limited by projector. |
| Frontier model | ~60-75% | Much larger vision encoder + more training data. May get some right through reasoning about geometry, but sim textures remain foreign. |

### What This Proves

Even a frontier model with 100x more parameters and real-world training data struggles on synthetic imagery. The domain gap is in the **vision projector's mapping**, not the LLM's reasoning. Fine-tuning the projector (our approach) directly addresses this bottleneck.

### Compute

- Qwen2-VL-7B: ~15 min inference on V100 (inference only, no training)
- Frontier API: ~$5-10 for 270 images (small subset is sufficient)

---

## Experiment 2: Fine-Tuned CNN Baselines

**Goal:** Establish a fair comparison between VLMs and traditional CNNs when both are trained on the same synthetic data.

**Motivation:** A reviewer might ask: "Why use a 2B parameter VLM when a ResNet can classify images?" This experiment shows that CNNs can learn the task in-distribution, establishing a fair baseline for the generalization and robustness experiments that follow.

### Models

| Model | Parameters | Architecture |
|---|---|---|
| ResNet-50 | 25.6M | Classic deep CNN, strong baseline for image classification |
| EfficientNet-B0 | 5.3M | Modern efficient CNN, represents lightweight deployment option |

### Training Protocol

- **Same training data** as the VLM: identical images, identical train/val/test split
- Replace final classification layer with 2-class output (OPEN/CLOSED)
- Fine-tune from ImageNet pre-trained weights
- Standard hyperparameters: Adam optimizer, lr=1e-4, 20 epochs, early stopping on val loss
- Report best val accuracy checkpoint

```python
import torchvision.models as models
import torch.nn as nn

# ResNet-50 baseline
model = models.resnet50(pretrained=True)
model.fc = nn.Linear(2048, 2)  # binary: OPEN/CLOSED

# EfficientNet-B0 baseline
model = models.efficientnet_b0(pretrained=True)
model.classifier[1] = nn.Linear(1280, 2)
```

### Expected Results

| Model | In-Distribution Accuracy | Reasoning |
|---|---|---|
| ResNet-50 | ~93-97% | Strong image classifier, should learn door patterns easily from 1,260 training images |
| EfficientNet-B0 | ~90-95% | Slightly lower capacity but still effective |
| **Qwen2-VL-2B (fine-tuned)** | **96.3% (known)** | For comparison |

### What This Proves

CNNs CAN learn the task — this is expected and not a failure. The point is to establish that both architectures perform comparably in-distribution, so the subsequent experiments (generalization, occlusion, real photos) are a fair test of architectural advantages, not just training quality.

### Compute

- ResNet-50: ~5-10 min fine-tuning on V100
- EfficientNet-B0: ~5 min fine-tuning on V100

---

## Experiment 3: Generalization to Unseen Doors

**Goal:** Prove that the VLM generalizes better than CNNs to door instances not seen during training.

**Motivation:** This directly addresses the known 0% generalization failure. After expanding to 5-8 doors, we hold out one complete door from training and test all models on it.

### Protocol

- Train VLM and both CNNs on doors A through E (same data, same split)
- Test all models on held-out door F (never seen during training)
- Also report in-distribution test accuracy for comparison

### Generalization Curve (VLM only)

Additionally, train the VLM on increasing numbers of doors to produce the generalization curve:

| Training Doors | In-Dist Acc | Held-Out Door Acc |
|---|---|---|
| 1 door | 96.3% (known) | 0% (known) |
| 3 doors | ? | ? |
| 5 doors | ? | ? |
| 8 doors | ? | ? |

This curve is the **single most important plot** in the project — it proves that more synthetic diversity from the digital twin directly improves real-world applicability.

### Expected Results

| Model | In-Dist Acc | Held-Out Door Acc |
|---|---|---|
| ResNet-50 | ~95% | ~40-60% (limited generalization) |
| EfficientNet-B0 | ~93% | ~35-55% |
| **Qwen2-VL-2B (fine-tuned)** | ~96% | **~70-85% (hypothesis)** |

### What This Proves

The VLM's pre-trained visual backbone has seen millions of real doors during pre-training. When fine-tuned on diverse synthetic data, it learns "door open = panel angled away from frame" as a general concept. The CNN, with no prior knowledge of doors beyond what's in the training set, learns pixel-level patterns specific to the training doors.

---

## Experiment 4: Occlusion Robustness

**Goal:** Prove that the VLM handles partial occlusions better than CNNs, demonstrating more robust visual reasoning.

**Motivation:** In real building patrols, doors are frequently partially blocked by carts, furniture, people, or other objects. A model that breaks under occlusion is not deployable. This also reinforces why a mobile robot (which encounters varied viewpoints and obstructions) needs a VLM rather than a CNN.

### Occlusion Test Set Generation

Generate a separate test set in Isaac Sim with controlled occlusions. These frames are NOT in any model's training data.

```python
from isaacsim.core.api.objects import DynamicCuboid
import numpy as np

# Place occluding object near door per episode
# Occlusion levels: 10%, 20%, 30% of door area blocked
OCCLUSION_CONFIGS = [
    {"name": "small_box",  "size": [0.3, 0.3, 0.5], "coverage": "~10%"},
    {"name": "cart",       "size": [0.6, 0.4, 0.8], "coverage": "~20%"},
    {"name": "large_shelf","size": [0.8, 0.4, 1.2], "coverage": "~30%"},
]
```

### Protocol

- Generate ~200 occluded test frames (balanced OPEN/CLOSED, across occlusion levels)
- Run all models (VLM + both CNNs) on both standard test set AND occluded test set
- Report accuracy at each occlusion level

### Results Table

| Model | No Occlusion | 10% Occluded | 20% Occluded | 30% Occluded |
|---|---|---|---|---|
| ResNet-50 | ~95% | ? | ? | ? |
| EfficientNet-B0 | ~93% | ? | ? | ? |
| **Qwen2-VL-2B (fine-tuned)** | 96.3% | ? | ? | ? |

### Expected Results

- CNNs: sharp accuracy drop as occlusion increases. CNNs learn holistic pixel templates — block part of the template and confidence collapses.
- VLM: gradual, smaller drop. VLMs reason about visible structure — "I can see the door edge is angled away from the frame" — and can handle missing information more gracefully.

### What This Proves

Real deployments have messy, occluded scenes. A VLM-equipped robot is more robust to the visual clutter of real buildings than a CNN-based system. This directly supports the choice of architecture for a general-purpose building agent.

---

## Experiment 5: Sim-to-Real Transfer

**Goal:** Prove that models trained entirely on synthetic data from the digital twin can classify door state in real-world photographs.

**Motivation:** This is the critical question from Dr. Riley and conference judges: *"Will it work in the real world?"* This experiment provides direct evidence.

### Real-World Test Set

- Photograph doors in Diercks Hall (DH) — a real MSOE building
- Capture each door in OPEN and CLOSED states
- Multiple doors, varied lighting (hallway lights, natural light from windows)
- Use a phone camera at approximate robot camera height (~0.65m)
- Target: ~50-100 real photographs

### Protocol

- Run ALL models on the same real photos: zero-shot VLMs, fine-tuned CNNs, fine-tuned VLM
- No model has ever seen real-world images during training
- Report accuracy per model

### Results Table

| Model | Sim Test (in-dist) | Sim Test (held-out door) | Real Photos (DH) |
|---|---|---|---|
| Qwen2-VL-2B zero-shot | 50% | 50% | ? |
| Qwen2-VL-7B zero-shot | ? | ? | ? |
| Frontier zero-shot | ? | ? | ? |
| ResNet-50 (fine-tuned on sim) | ~95% | ? | ? |
| EfficientNet-B0 (fine-tuned on sim) | ~93% | ? | ? |
| **Qwen2-VL-2B (fine-tuned on sim)** | **96.3%** | **?** | **?** |

### What This Proves

If the fine-tuned VLM outperforms fine-tuned CNNs on real photos — despite all being trained on identical synthetic data — it demonstrates that the VLM's pre-trained visual understanding helps bridge the sim-to-real domain gap. The VLM has seen millions of real doors during pre-training; the CNN has only seen ImageNet categories. When both are adapted to sim imagery, the VLM retains more transferable knowledge.

### Scaling Argument

Include speculative calculation in the write-up:

```
Manual approach (real data):
  5 anomaly types x 10 buildings x 1,000 images = 50,000 images
  Human labeling at ~2 sec/image = 28 hours
  Must repeat for every new building

Digital twin approach (our pipeline):
  5 anomaly types x 10 buildings x 1,000 images = 50,000 images
  Generation time: ~2 hours GPU compute (automated)
  Zero human labeling
  New building: import BIM file, run pipeline (~30 min setup)
```

---

## Experiment 6: Multi-Anomaly Generality

**Goal:** Prove the VLM handles multiple anomaly types from a single model, while CNNs require separate models per task.

**Motivation:** This is a key architectural advantage of VLMs. Public safety identified 5 anomaly types. A CNN-based system needs 5 separate models (5x training, 5x deployment, 5x maintenance). The VLM handles all from one model via prompt-based task switching.

### Protocol

- Fine-tune ONE VLM on doors + broken glass (combined dataset, different prompts per anomaly)
- Fine-tune TWO CNNs: one for doors, one for glass (each needs its own classification head)
- Test both on their respective test sets

### Prompts

```
Door task:  "Is this door open or closed?"
Glass task: "Is there any broken glass visible in this image?"
```

### Results Table

| System | Door Accuracy | Glass Accuracy | Models Required |
|---|---|---|---|
| CNN approach | ? | ? | 2 separate models |
| **VLM approach** | ? | ? | **1 model** |

### What This Proves

The VLM scales to new anomaly types by adding training data with a new prompt — no architectural changes, no new model, no additional deployment. For N anomaly types, the CNN approach requires N models; the VLM approach requires 1. In a real deployment with 5+ anomaly types across dozens of buildings, this architectural advantage is decisive.

---

## Experiment 7: Explainability (Qualitative)

**Goal:** Demonstrate that the VLM provides actionable natural language descriptions that a human dispatcher can use, while CNNs provide only numerical probabilities.

**Motivation:** MSOE Public Safety has 1 dispatcher monitoring 903 cameras. An alert that says `P(OPEN) = 0.73` requires the dispatcher to pull up the camera feed and interpret the image themselves. An alert that says *"The door appears to be open, with the panel angled approximately 30 degrees from the frame"* is immediately actionable.

### Protocol

- Select 10 representative test frames (5 easy, 5 ambiguous)
- Run both CNN and VLM on each
- Display side-by-side outputs

### Example Output Comparison

**Frame: Door at 15 degrees, partially occluded by cart**

```
CNN output:
  P(OPEN) = 0.73

VLM output:
  "The door appears to be open. The door panel is angled away
   from the frame. A cart partially blocks the view but the
   gap between the door and frame is visible."
```

**Frame: Door at 3 degrees (ambiguous boundary)**

```
CNN output:
  P(OPEN) = 0.48

VLM output:
  "The door appears to be closed. The panel is nearly flush
   with the door frame, though a very slight gap may be
   visible at the top edge."
```

### What This Proves

This is not a quantitative experiment — it's a qualitative argument for operational deployability. The VLM turns raw visual data into human-readable reports, matching what public safety officers produce during walkthroughs. This is the "what" that Dr. Riley asked us to focus on.

---

## Summary: Complete Results Matrix

The final presentation/paper should contain this table:

| Model | Type | In-Dist (sim) | Held-Out Door (sim) | Occluded (sim) | Real Photos (DH) | Multi-Task | Explainable |
|---|---|---|---|---|---|---|---|
| Qwen2-VL-2B zero-shot | VLM | 50% | ~50% | ~50% | ? | N/A | Yes |
| Qwen2-VL-7B zero-shot | VLM | ? | ? | ? | ? | N/A | Yes |
| Frontier zero-shot | VLM | ? | ? | ? | ? | N/A | Yes |
| ResNet-50 fine-tuned | CNN | ~95% | ? | ? (expect drop) | ? | No (1 model per task) | No |
| EfficientNet-B0 fine-tuned | CNN | ~93% | ? | ? (expect drop) | ? | No (1 model per task) | No |
| **Qwen2-VL-2B fine-tuned** | **VLM** | **96.3%** | **?** | **?** | **?** | **Yes (1 model)** | **Yes** |

### The Argument This Table Makes

1. **Row 1-3:** Fine-tuning is necessary (zero-shot fails regardless of model size)
2. **Column 3 vs 4:** VLM generalizes better than CNN to unseen doors
3. **Column 5:** VLM is more robust to real-world visual clutter
4. **Column 6:** VLM transfers from sim to real better than CNN
5. **Column 7:** VLM handles multiple anomalies from one model
6. **Column 8:** VLM provides actionable natural language output

Each column eliminates one alternative approach. By the end, the fine-tuned VLM is the only architecture that checks every box.

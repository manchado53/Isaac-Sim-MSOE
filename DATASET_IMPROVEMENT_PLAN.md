# Fine-Tuning Dataset Improvement Plan

> **Context:** This plan is part of the larger multi-anomaly detection project. See [PROJECT_PLAN.md](PROJECT_PLAN.md) for full project scope, novelty, and 2-week execution plan. This document covers the technical details of dataset expansion for all three target anomalies: **open doors, broken glass, and water leakage.**

**Goal:** Expand the dataset beyond the original 1,800 door frames (1 door, 3 camera angles, fixed lighting) to support multi-anomaly detection with generalization across door types, anomaly types, and sim-to-real transfer.

**Current baseline:** 96.3% accuracy on held-out test set (same door, same conditions)  
**Known failure:** 0% accuracy on unseen door (`Doors_13185454`) — model learned door-specific appearance, not the concept of open vs closed

---

## 1. More Doors

**Problem:** The model was trained on a single door (`Doors_13606713`). It fails completely on unseen doors because it learned that door's specific texture and geometry, not the abstract concept of open vs closed.

**Plan:**
- Identify 5–8 additional doors in the building USD with working rotation geometry
- Run the static collection script against each door independently
- Include at least one door with a different material (glass vs. aluminum frame vs. wood)
- Check floor plan for **double doors** — if present, they require separate handling (see Section 7)
- Target: 3–5 doors minimum before re-evaluating generalization

**Implementation:** Parameterize `door_static_collection.py` to accept a door path and hinge config via CLI args, then run one collection job per door.

### Finalized Door Selection (9 doors)

Selected from all-floors snapshot analysis. Criteria: single doors only (no double doors), interior camera positions only, good lighting, visual diversity across floors/materials/surroundings. Swing direction flipped for ~half to ensure VLM sees doors opening both ways.

| # | Door ID | Floor | Collect Side | Swing | Visual Context |
|---|---|---|---|---|---|
| 1 | Doors_9817744 | 2 | north | Flipped | Brick wall stairwell, red railings, glass door with transom |
| 2 | Doors_12066864 | 2 | north | Normal | Stairwell, brick wall, glass door with transom, red railings |
| 3 | Doors_13607620 | 2 | north | Flipped | Interior room, door with transom, windows nearby |
| 4 | Doors_3135827 | 3 | south | Normal | Corridor with red accent wall, glass door |
| 5 | Doors_1868517 | 3 | north | Flipped | Corridor with red ceiling beam, glass door |
| 6 | Doors_10383956 | 3 | west | Flipped | Corridor between white walls, glass door |
| 7 | Doors_13182085 | 4 | east | Normal | Open area, red accent wall, glass railings |
| 8 | Doors_4190126 | 5 | west | Normal | Corridor with windows, red ceiling |
| 9 | Doors_13606713 | 5 | east | Normal | Existing training door, aluminum/glass curtain wall |

**Train/test split:** Train on 7, hold out 2 for generalization testing.

- **Train (7):** Doors_9817744, Doors_12066864, Doors_3135827, Doors_1868517, Doors_10383956, Doors_13182085, Doors_13606713
- **Generalization test (2):** Doors_13607620 (Floor 2), Doors_4190126 (Floor 5)

Generalization curve experiment: train on 1 → 3 → 5 → 7 doors, evaluate on the same 2 held-out doors each time. Config is defined in `door_snapshot.py` → `DOOR_CONFIG` dict.

---

## 2. Full Camera Arc

**Current:** 3 cameras at arc offsets −30°, −15°, 0° (southern side only). Northern arc blocked by building wall geometry.

**Plan:**
- Extend the southern arc further: add −45° and −60° offsets
- Add camera **distance variation**: repeat the arc at 1.0 m, 2.0 m (current), and 3.5 m from the door
- Add camera **height variation**: current Z = 27.8 m (eye-level). Add a low angle (Z − 0.5 m) and a high angle (Z + 1.0 m) to simulate different robot camera mounts
- For new doors without wall obstructions, test the full 360° arc and keep positions with unobstructed line of sight

**Expected frame multiplier:** ~3× from distance + height variation alone

---

## 3. Lighting Variation

**Current:** Fixed DomeLight (intensity 50,000) + fixed SphereLight (intensity 300,000). Every frame has identical lighting.

**Plan — vary per episode, not per frame:**

| Variable | Range | Rationale |
|---|---|---|
| DomeLight intensity | 10,000 – 80,000 | Simulates cloudy vs. bright outdoor light |
| DomeLight color temperature | 4,000 K – 7,000 K | Warm indoor vs. cool daylight |
| FillLight intensity | 50,000 – 500,000 | Overhead fluorescent lights on/off |
| FillLight position | ±1.0 m XY jitter | Slightly different overhead fixture positions |
| Second shadow light (optional) | Random position | Simulates windows or corridor lights |

```python
# Example: randomize per episode in door_static_collection.py
dome.CreateIntensityAttr(random.uniform(10_000, 80_000))
fill.CreateIntensityAttr(random.uniform(50_000, 500_000))
```

**Impact:** Forces the model to learn geometric cues (door gap, edge shadow) rather than pixel brightness patterns.

---

## 4. Door Material Variation

**Goal:** Demonstrate the power of synthetic data — changing materials in simulation is trivial, whereas sourcing real photographs of the same door in different materials is impossible.

**Plan:**
- Define 4–5 MDL materials in the stage before the collection loop:
  - Base aluminum (current — from Revit export)
  - Painted white metal
  - Dark anodized aluminum
  - Wood veneer
  - Frosted glass panel
- Randomly bind one material to the door mesh at the start of each episode using `UsdShade.MaterialBindingAPI`

```python
from pxr import UsdShade

MATERIALS = [
    "/World/Looks/AluminumBase",
    "/World/Looks/PaintedWhite",
    "/World/Looks/DarkAnodized",
    "/World/Looks/WoodVeneer",
]

def set_episode_material(stage, mesh_path):
    mat_path = random.choice(MATERIALS)
    mesh_prim = stage.GetPrimAtPath(mesh_path)
    material  = UsdShade.Material(stage.GetPrimAtPath(mat_path))
    UsdShade.MaterialBindingAPI(mesh_prim).Bind(material)
```

**Note:** Authoring MDL materials from scratch in USD is non-trivial. The quickest path is to use materials from the NVIDIA Material Library bundled with Isaac Sim (`omniverse://localhost/NVIDIA/Materials/vMaterials_2/`). These can be referenced into the stage as sublayers.

---

## 5. Partial Occlusions

**Current:** Door is always fully visible with no obstructions in the scene.

**Plan:**
- Place 1–2 random props near the door per episode: boxes, carts, a chair, a trash can
- Use Isaac Sim's built-in assets or simple geometric primitives (`DynamicCuboid`) as stand-ins
- Props should occlude 10–30% of the door area — enough to be challenging, not enough to make the label ambiguous
- Randomize prop position within a 1.5 m radius of the door per episode

```python
from isaacsim.core.api.objects import DynamicCuboid
import numpy as np

# Place a random box near the door each episode
box_x = DOOR_CENTER_XY[0] + random.uniform(-1.0, 1.0)
box_y = DOOR_CENTER_XY[1] + random.uniform(-1.0, 1.0)
cuboid.set_world_pose(position=np.array([box_x, box_y, CAMERA_Z - 0.5]))
```

**Label impact:** None — the door state (angle) is independent of what is in front of it.

---

## 6. Expanded Frame Count

**Current:** 1,800 frames (60 episodes × 10 frames × 3 cameras)

**Target:** 10,000 – 15,000 frames

| Source | Frames |
|---|---|
| 6 doors × 60 episodes × 10 frames × 3 cameras | 10,800 |
| + 2 extra distance rings per door | +7,200 |
| Estimated total (before augmentation) | ~18,000 |

**Decision boundary oversampling:** The original error analysis showed all errors fall in the 1°–14° range. Deliberately oversample this region:
- CLOSED episodes: 50% at 0°–3°, 50% at 3°–5° (currently uniform over 0°–5°)
- OPEN episodes: add a dedicated "barely open" band at 6°–14°, sampled 2× as often as mid/wide open bands

**Label rejection band (optional):** Exclude frames where angle is 4°–7° from training entirely. These are genuinely ambiguous even to a human. Do not exclude from the test set — report accuracy on them separately.

---

## 7. Double Doors

**Status:** To be confirmed from floor plan review.

**If double doors are present:**
- Both panels share a frame; typically both move together or one is fixed
- Label convention: **any panel open → `[OPEN]`** (most useful for robot navigation — the passage is accessible)
- Collect data for both states: both closed, one open (fixed panel), both open
- The center seam of a double door is a strong visual feature — the model must see double doors in training or it will misclassify them

**Implementation:** Two `DOOR_MESHES` entries (one per panel), both receiving the same rotation matrix. If panels move independently, implement a second rotation parameter per episode.

---

## 8. Image Augmentations

Applied in `prepare_finetune_data.py` during JPEG export. **Training split only — never val or test.**

### Recommended Augmentations

| Augmentation | Parameters | Rationale |
|---|---|---|
| Color jitter | brightness ±30%, contrast ±20%, saturation ±20% | Compensates for fixed sim lighting |
| Gaussian blur | σ = 0.3 – 1.5 px | Simulates camera focus and motion blur |
| Random horizontal flip | p = 0.5 | Doubles viewpoint diversity; CLOSED/OPEN is flip-symmetric |
| JPEG re-compression | quality = 60 – 85 | Simulates real camera compression artifacts |
| Zoom (center crop + resize) | crop ratio = 0.80 – 0.98 | Simulates camera distance variation; keep center-biased |
| Grayscale (occasional) | p = 0.10 | Removes color cue, forces shape-based learning |

### Augmentations to Avoid

| Augmentation | Reason |
|---|---|
| Vertical flip | A door on the ceiling is not a valid scene |
| Large cutout / random erasing | May occlude the full door — label becomes ambiguous |
| Large rotation (> ±15°) | Makes the door appear physically tilted |
| Strong perspective warp | Changes the apparent angle of the door panel |

### Zoom Augmentation Detail

A center-biased crop at 80–98% of the frame, resized back to 640×480, simulates different camera focal lengths and distances. Keep it center-biased rather than fully random — a corner crop may place the door off-frame entirely. Do not crop below 75%: at that point the door edges (the primary open/closed cue) risk being clipped.

```python
from torchvision import transforms

zoom_aug = transforms.RandomResizedCrop(
    size=(480, 640),
    scale=(0.80, 0.98),   # crop 80–98% of area
    ratio=(1.2, 1.4),     # maintain ~4:3 aspect ratio
)
```

### Implementation Sketch

```python
from PIL import Image
from torchvision import transforms
import random

train_aug = transforms.Compose([
    transforms.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.2),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.3, 1.5)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.Grayscale(num_output_channels=3)], p=0.10),
    transforms.RandomResizedCrop(size=(480, 640), scale=(0.80, 0.98), ratio=(1.2, 1.4)),
])

def process_split(indices, jsonl_path, split_name, augment=False):
    for idx in indices:
        img = Image.fromarray(frames[idx])
        if augment:
            img = train_aug(img)
        # then apply JPEG re-compression at random quality
        quality = random.randint(60, 92) if augment else 92
        img.save(img_path, "JPEG", quality=quality)
```

---

## 9. Updated Split Strategy

With a larger, more diverse dataset, the 70/15/15 split remains appropriate. However:

- **Test set must include frames from all doors** — not just the training door. Otherwise test accuracy is still inflated.
- Consider a **door-stratified test set**: hold out one complete door entirely from training and use it as the generalization test. This is a stricter and more honest evaluation.
- Keep the current held-out test set (`test.jsonl`) as the **in-distribution** benchmark; add the held-out door as an **out-of-distribution** benchmark.

---

## 10. Estimated Timeline

| Task | Sim Jobs | Est. Runtime |
|---|---|---|
| Collect data for 5 additional doors | 5 jobs | ~20 min each |
| Extended camera arc (distance + height) | Integrated into above | — |
| Lighting variation | Integrated into above | — |
| Material variation | Integrated into above | — |
| Partial occlusions | Integrated into above | — |
| Re-run `prepare_finetune_data.py` with augmentation | 1 job (CPU) | ~5 min |
| Re-run fine-tuning | 1 job (V100 dgx) | ~20 min |
| Re-run evaluation | 1 job | ~5 min |

**Total estimated additional compute:** ~3–4 GPU-hours

---

## 11. Broken Glass Anomaly (NEW)

**Priority:** Must-do — this is the strongest "only a robot can see this" anomaly. Public safety said: *"door shattering but double-paned windows so nobody notices"* and *"camera angle can't capture."*

### Simulation Approach A: Material Swap (simplest, recommended first)

Swap intact glass material on door/window prims for a cracked/shattered variant.

```python
from pxr import UsdShade

def apply_broken_glass(stage, glass_prim_path, broken=True):
    """Swap glass material between intact and cracked variants."""
    prim = stage.GetPrimAtPath(glass_prim_path)
    if broken:
        mat = UsdShade.Material(stage.GetPrimAtPath("/World/Looks/CrackedGlass"))
    else:
        mat = UsdShade.Material(stage.GetPrimAtPath("/World/Looks/IntactGlass"))
    UsdShade.MaterialBindingAPI(prim).Bind(mat)
```

### Simulation Approach B: Geometry Modification (more realistic)

Hide the glass pane and scatter shard primitives on the floor beneath.

```python
def break_glass_panel(stage, glass_prim_path, floor_z):
    """Hide glass pane and add floor debris."""
    glass_prim = stage.GetPrimAtPath(glass_prim_path)
    glass_prim.GetAttribute("visibility").Set("invisible")

    # Scatter small prims as "shards" on floor beneath
    for i in range(random.randint(3, 8)):
        pos = [glass_center_x + random.uniform(-0.3, 0.3),
               glass_center_y + random.uniform(-0.3, 0.3),
               floor_z]
        # Create small, flat cuboid as glass shard
        ...
```

### Auto-Labeling

Binary — we know whether we applied the broken state or not. Label is free from scene state.

### Diversity Plan

- Multiple glass door/window prims across the building
- Vary break severity: cracked texture only, missing pane, pane + floor debris
- Lighting randomization (same as door pipeline)
- Camera distance and angle variation (robot approaches from different directions)

### VLM Prompt Format

```json
{
  "conversations": [
    {
      "role": "user",
      "content": "<|vision_start|><|image_pad|><|vision_end|>Is there any broken glass visible in this image?"
    },
    {
      "role": "assistant",
      "content": "Glass Status: [BROKEN]."
    }
  ]
}
```

---

## 12. Water Leakage Anomaly (NEW — Stretch Goal)

**Priority:** Stretch goal — third anomaly type to strengthen the generality argument.

### Simulation Approach

Place a flat, reflective mesh on the floor with a water material from NVIDIA's vMaterials library.

```python
from pxr import UsdGeom, UsdShade

def create_water_puddle(stage, position, radius):
    """Place a reflective water puddle on the floor."""
    puddle = UsdGeom.Cylinder.Define(stage, "/World/WaterPuddle")
    puddle.CreateRadiusAttr(radius)
    puddle.CreateHeightAttr(0.001)  # Nearly flat disc
    # Bind reflective water material from vMaterials library
    water_mat = UsdShade.Material(stage.GetPrimAtPath("/World/Looks/Water"))
    UsdShade.MaterialBindingAPI(puddle.GetPrim()).Bind(water_mat)
```

### Diversity Plan

- Randomize puddle size (0.5m–3m diameter), shape, and position
- Place in hallways, near doors, near walls (simulating different leak sources)
- Lighting variation affects reflectivity — important for teaching structural cues over brightness

### Auto-Labeling

Binary — puddle placed or not. Free from scene state.

### VLM Prompt Format

```json
{
  "conversations": [
    {
      "role": "user",
      "content": "<|vision_start|><|image_pad|><|vision_end|>Is there any water leakage or puddle visible on the floor?"
    },
    {
      "role": "assistant",
      "content": "Floor Status: [WATER DETECTED]."
    }
  ]
}
```

### Challenge

Visual realism of simulated water. Keep expectations moderate — a reflective floor plane may not look like a real puddle. The goal is to teach the VLM the concept of "reflective/wet surface on floor where it shouldn't be" across enough variants to generalize.

---

## 13. Multi-Anomaly Training Strategy

### Combined Dataset Structure

All three anomaly types share the same fine-tuning pipeline (LoRA + vision projector on Qwen2-VL-2B). The training dataset will contain a mix of all anomaly types:

| Anomaly Type | Prompt | Labels |
|---|---|---|
| Door state | "Is this door open or closed?" | `[OPEN]` / `[CLOSED]` |
| Broken glass | "Is there any broken glass visible in this image?" | `[BROKEN]` / `[INTACT]` |
| Water leakage | "Is there any water leakage or puddle visible on the floor?" | `[WATER DETECTED]` / `[DRY]` |

Each sample includes a task-specific prompt so the VLM learns to respond to different questions about the same type of image (building interior). This is a natural fit for VLMs — one model, multiple tasks via different prompts.

### General Anomaly Prompt (future)

Once per-anomaly accuracy is validated, a general prompt can be tested:

```
"Inspect this image from a building patrol. Report any anomalies you observe,
including open doors, broken glass, water leakage, or anything unusual."
```

This is a stretch goal — start with per-anomaly binary prompts where evaluation is clean and measurable.

---

## 14. Renderer Strategy — RTX Real-Time vs Path Tracing

**Context:** Isaac Sim offers two RTX renderers. The choice affects both data realism and render cost. A side-by-side comparison of `Doors_13606713` was captured via `simulations/robot/render_comparison.py` and saved to `data/render_comparison/`.

### Observed Differences

| Aspect | RTX Real-Time | RTX Path Tracing |
|---|---|---|
| Global illumination | Direct light only — unlit surfaces collapse to black | Full indirect bounces — all surfaces lit naturally |
| Color bleeding | Absent | Present — red floor bounces pink tint onto white walls/ceiling |
| Contact shadows | Hard, inky bands at geometry edges | Soft, physically accurate falloff |
| Glass reflections | Sharper, slightly ghosted, higher contrast | Softer, more believable, reflected scene has same GI-filled look |
| Ceiling / corners | Often near-black where no RectLight reaches | Fully lit via bounced light |
| Render time per frame | ~0.1 s | ~1–2 s (at 64 samples/px/frame) |

**Key finding:** The dark-ceiling / hard-shadow artifacts in the real-time frame are a **shortcut signal** a model can exploit (e.g. "dark band near floor = closed door"). Path tracing removes these artifacts and forces the model to rely on geometric cues instead.

### Recommended Usage

| Dataset | Renderer | Rationale |
|---|---|---|
| **Training data** (all 3 anomalies) | RTX Real-Time | ~15× faster; domain randomization + augmentation compensates for lighting artifacts; 10k–15k frames at path-tracing cost is prohibitive |
| **In-distribution test set** | RTX Real-Time | Matches training distribution — clean measurement of what the model learned |
| **Generalization / sim-to-real evaluation set** | **RTX Path Tracing** | Closer to real-photo lighting physics; best proxy for real-world performance before running Diercks Hall photos |
| **Held-out door evaluation (2 doors)** | **RTX Path Tracing** | Simultaneously tests generalization across doors AND across renderers — stricter, more honest benchmark |
| **Paper figures / presentation images** | **RTX Path Tracing** | Visually compelling for reviewers; shows digital twin is capable of photorealistic output |

### Implementation Notes

- Renderer is locked at `SimulationApp` construction — cannot switch at runtime. Structure collection scripts so renderer mode is a CLI arg, and run separate jobs for real-time vs path-tracing captures.
- **Lower rect-light intensity for path-tracing passes.** At the current intensity (50,000) the red door frame clips under PT because indirect bounces add on top of direct illumination. Target ~20,000–25,000 for PT runs.
- Path-tracing settle steps matter — grab frames only after at least 64 render steps so samples have accumulated. For higher-quality paper figures, bump `samples_per_pixel_per_frame` from 64 → 128 or 256.
- Path tracing does **not** fix material realism — flat albedo walls/floors still look synthetic in both renderers. The bigger realism lever remains PBR material swaps (vMaterials 2), not renderer choice.

### Planned Evaluation Sets (Path Tracing)

| Set | Frames | Purpose |
|---|---|---|
| Held-out door eval (2 doors × ~200 frames × PT) | ~400 | Generalization benchmark under photorealistic lighting |
| Occlusion eval subset (Experiment 4) | ~200 | Robustness under PT — tests whether occlusion-robustness results hold across renderers |
| Paper figure set (top 5 representative frames) | ~5 | Hero images for final write-up |

**Compute budget:** ~600 PT frames × 2 s/frame ≈ 20 min per full evaluation pass on V100. Affordable for eval, prohibitive for the ~15k-frame training set.

---

## Summary of Changes to `door_static_collection.py`

- [ ] Parameterize door path, hinge config via CLI args
- [ ] Extend camera arc: add −45°, −60° offsets + distance/height variants
- [ ] Randomize DomeLight and FillLight intensity + color temperature per episode
- [ ] Add material binding randomization per episode
- [ ] Add random prop placement per episode
- [ ] Oversample 1°–14° boundary region (2× sampling weight)

## Summary of New Data Collection Scripts

- [ ] `broken_glass_collection.py` — Material swap + debris generation for glass door/window prims
- [ ] `water_leakage_collection.py` — Reflective puddle placement on floor (stretch goal)
- [ ] Shared infrastructure: lighting randomization, camera arc, NPZ output format

## Summary of Changes to `prepare_finetune_data.py`

- [ ] Add `torchvision` augmentation pipeline for train split
- [ ] Apply zoom crop, color jitter, blur, flip, JPEG re-compression to train only
- [ ] Add door-stratified held-out test set (separate from current `test.jsonl`)
- [ ] Update split reporting to show per-door breakdown
- [ ] Support multi-anomaly JSONL generation (doors + glass + water with different prompts)
- [ ] Merge multiple NPZ files into unified dataset with anomaly-type labels

## Summary of Evaluation Plan

- [ ] In-distribution accuracy per anomaly type
- [ ] Out-of-distribution generalization (held-out door for doors, held-out window for glass)
- [ ] Generalization curve: accuracy vs. number of training doors (1 → 3 → 5 → 8)
- [ ] Real-world photo evaluation on Diercks Hall photographs
- [ ] Scaling calculation: manual labeling time vs. digital twin generation time

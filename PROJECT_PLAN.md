# Project Plan: BIM-Powered Visual Anomaly Detection for Mobile Robot Patrol

**Last updated:** 2026-04-11
**Timeline:** 2 weeks (April 11 – April 25, 2026)

---

## 1. Product Definition

### What We Are Building

A **general-purpose visual building agent** — a VLM-equipped mobile robot that autonomously patrols building interiors and detects visual anomalies using a Vision-Language Model trained entirely on synthetic data generated from the building's digital twin.

### Showcase Use Case

**Autonomous security patrol**, developed in partnership with MSOE's Public Safety Office. Public safety officers patrol campus buildings once every 8 hours, manually checking for anomalies. Between patrols, there is a multi-hour gap where problems go undetected. Our robot fills that gap with continuous, automated visual inspection.

### The Human Cost of Manual Patrol

MSOE Public Safety operates with 2–3 officers and 1 dispatcher per shift, covering 24/7 patrols across all campus buildings. Officers spend **100+ hours per year just unlocking doors** — routine, repetitive work that pulls them away from what only humans can do: intervene in incidents, make judgment calls, respond to medical emergencies, and patrol high-risk areas like Viets parking garage and Grohmann at night. Every hour an officer spends checking whether a door is locked is an hour they aren't stopping crime or responding to the 903 cameras they can't watch. The robot takes over the repetitive visual inspection, freeing officers for the work that requires human presence and decision-making.

### The Five Things Public Safety Looks For

In a meeting with MSOE Public Safety (April 2026), officers identified their top five anomalies during building walkthroughs:

1. **Open doors** — Are all doors closed and locked?
2. **Water leakage** — Pipe breaks, pooling water
3. **Broken glass** — Shattered door panels, broken windows (often unnoticed with double-paned glass)
4. **Vandalism** — Holes in walls, graffiti
5. **Burnt out lights** — Lighting failures

We are targeting **three** of these for our demonstration: **open doors, broken glass, and water leakage**. These three are feasible to simulate in the digital twin, visible from robot camera height, and auto-labelable from scene state.

---

## 2. The Problem We Are Solving

### The Core Problem

Training a mobile robot to visually understand building environments requires massive labeled datasets of anomalies — but you cannot stage fires, break-ins, floods, or equipment failures in real buildings to collect training data. Manual data collection and labeling for every new building is expensive, slow, and doesn't scale.

### Why Existing Solutions Fall Short

| Existing Approach | Limitation |
|---|---|
| **Smart camera systems** (Ambient.ai, Verkada) | Solved problem for static cameras. Costs $100K+. Fixed viewpoints miss fine-grained anomalies (shattered inner glass pane, water on floor). |
| **Patrol robots** (Knightscope, Cobalt) | Traditional CV only — license plate readers, thermal cameras. No VLM, no visual understanding, pre-programmed rule-based responses. |
| **Boston Dynamics Spot + Orbit 5.0** | Closest competitor. Uses VLM-style prompts for anomaly detection. But: $75K+ robot, requires real-world images, manual per-site prompt configuration for every new deployment. |
| **Zero-shot VLMs** (GPT-4o, Qwen2-VL) | Work on real-world photos but have no building-specific context. Our baseline test: Qwen2-VL-2B scores 50% (predicts OPEN for everything). Academic benchmarks: GPT-4o reaches only 79.2% on domain-specific anomalies even with heavy prompt engineering. |
| **Contact sensors / IoT** | A $20 sensor tells you a door opened. It cannot tell you a glass panel is cracked, water is pooling, or a fire exit is blocked by furniture. Single-purpose, one sensor per anomaly type. |

### What Has NOT Been Solved

Based on our literature review (see [RELATED_WORK.md](RELATED_WORK.md)):

1. **Nobody has used BIM/architectural data to auto-generate labeled training datasets for VLMs.** The only BIM-to-synthetic-data paper (ScienceDirect, 2025) used ResNet for floor plan generation, not VLM training.

2. **Nobody has built an end-to-end pipeline from BIM → digital twin → synthetic anomalies → fine-tuned VLM → mobile robot.** The individual components exist but this integration has not been published.

3. **Sim-to-real transfer for VLM-based visual inspection is largely untested.** Sim-to-real is well-established for RL navigation policies and robot manipulation (MolmoBot, 2025), but not for VLM perception in building environments.

4. **VLM generalization from synthetic data remains an open problem.** Both InspectVLM (2025) and our own results (0% on unseen doors) show that fine-tuned VLMs memorize appearance rather than learning concepts. Systematic study of how domain randomization addresses this is missing from the literature.

---

## 3. Our Novelty

### The One-Liner

> We use a BIM-derived digital twin as an automated dataset factory — generating unlimited labeled anomaly scenarios to train a VLM-equipped mobile robot that visually understands building environments, without a single manually labeled image.

### The Pipeline (our primary contribution)

```
Any Revit/BIM File
    → Automatic conversion to USD
        → Digital Twin in NVIDIA Isaac Sim
            → Programmatically generate anomalies:
                - Rotate doors to various angles
                - Swap glass materials to cracked/shattered variants
                - Place reflective water planes on floors
                - (Extensible to any simulatable anomaly)
            → Robot patrols, captures frames from camera
            → Physics/scene state provides ground-truth labels (FREE)
                - Door angle from rotation matrix → OPEN/CLOSED
                - Glass material swapped → BROKEN/INTACT
                - Water plane placed → LEAK/DRY
            → Fine-tune VLM (Qwen2-VL-2B + LoRA + vision projector)
            → Deploy on mobile robot in real building
```

### Why the Digital Twin is Essential

The digital twin is not optional decoration — it is the core enabler:

| Without Digital Twin | With Digital Twin |
|---|---|
| Need someone to photograph and label thousands of anomaly images per building | Auto-generated from scene manipulation, zero manual labels |
| Can't stage dangerous anomalies (floods, broken glass, fires) for training data | Simulate any anomaly programmatically, infinitely |
| Manual per-building ML engineering for each new deployment | New BIM file → same pipeline → new trained model |
| No building-specific spatial context | BIM knows which doors should be locked, which areas are restricted |
| Training data limited by what naturally occurs (public safety finds anomalies 2-3 times/month) | Generate thousands of anomaly instances in minutes |

### Why a VLM (not a CNN)

- **Multi-task:** One model handles doors, glass, water, and any future anomaly type. A CNN requires a separate model per task.
- **Explainability:** VLM outputs natural language: *"The glass panel in the east stairwell door is cracked"* — directly usable by a dispatcher. A CNN outputs a probability score.
- **Extensibility:** Adding a new anomaly type means adding new training prompts, not redesigning a classification head.
- **Semantic reasoning:** A VLM can reason about context — an open door at 2 PM in a classroom building is normal; the same door open at 2 AM is an anomaly.

### Differentiator Table

| Dimension | Boston Dynamics Spot | Academic VLM Papers | Our System |
|---|---|---|---|
| Training data source | Real images | Real images | **Synthetic from BIM digital twin** |
| New building setup | Manual per-site config | N/A | **Import BIM file → auto-generate** |
| Labeling effort | Manual | Manual | **Zero — physics provides labels** |
| Anomaly generation | Must occur naturally | Staged or found | **Programmatic — simulate anything** |
| Robot platform cost | $75K+ | Various | NVIDIA Nova Carter |
| Anomaly types | Pre-configured per site | Single task per paper | **Multi-anomaly from one pipeline** |
| Scalability | Per-site engineering | Per-dataset collection | **Any BIM → trained VLM** |

---

## 4. Anomaly Detection Plan

### Anomaly 1: Open Doors (Primary — Existing Pipeline)

**Status:** Mostly complete. Expanding for generalization.

**Simulation approach:**
- Door prims in the USD have a `xformOp:transform` matrix. Rewrite the rotation component to set any angle from 0°–90°.
- Hinge position extracted per door from the building geometry.
- Label: angle ≤ 5° → `[CLOSED]`, angle > 5° → `[OPEN]`.

**Current results:**
- 1,800 frames from 1 door, 3 camera angles
- Fine-tuned VLM: 96.3% accuracy on held-out test set
- **0% on unseen door** — model memorized appearance, not the concept

**Expansion plan:**
- Scale to 5–8 doors with different geometry/materials
- Extended camera arc: add distance and height variation
- Lighting randomization per episode (DomeLight intensity 10K–80K, color temp 4K–7K)
- Material randomization: swap door materials (aluminum, painted white, dark anodized, wood, frosted glass)
- Oversample the 1°–14° decision boundary region
- Hold out 1 complete door for out-of-distribution generalization test

**Target:** Demonstrate the generalization curve — accuracy on held-out doors improving as synthetic data diversity increases. This is the core proof that the pipeline works.

### Anomaly 2: Broken Glass (New — High Impact)

**Why this anomaly:** Public safety specifically said *"door shattering but double-paned windows so nobody notices"* and *"camera angle can't capture."* This is the strongest "only a robot can see this" example — cameras miss it, sensors can't detect it, only a mobile agent at eye level catches it.

**Simulation approach — Material Swap (simplest, most robust):**

```python
from pxr import UsdShade

# Swap intact glass material for cracked/shattered variant
def apply_broken_glass(stage, glass_prim_path, broken=True):
    prim = stage.GetPrimAtPath(glass_prim_path)
    if broken:
        mat = UsdShade.Material(stage.GetPrimAtPath("/World/Looks/CrackedGlass"))
    else:
        mat = UsdShade.Material(stage.GetPrimAtPath("/World/Looks/IntactGlass"))
    UsdShade.MaterialBindingAPI(prim).Bind(mat)
```

**Simulation approach — Geometry Modification (more realistic):**

```python
# Option B: Hide glass pane + scatter debris prims on floor
def break_glass_panel(stage, glass_prim_path, floor_z):
    # Make glass invisible (simulates missing/shattered pane)
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

**Auto-labeling:** Binary — we know whether we applied the broken state or not. Free label from scene state.

**Diversity plan:**
- Multiple glass door/window prims across the building
- Vary break severity: cracked texture, missing pane, pane + floor debris
- Vary lighting to avoid learning brightness-based shortcuts
- Camera distance and angle variation (robot approaches from different directions)

**VLM prompt format:**
```json
{
  "role": "user",
  "content": "<|vision_start|><|image_pad|><|vision_end|>Is there any broken glass visible in this image?"
}
```

### Anomaly 3: Water Leakage (Stretch Goal)

**Simulation approach:**
- Place a flat, irregular mesh on the floor with a reflective water material from NVIDIA's vMaterials library
- Randomize size (0.5m–3m diameter), shape (irregular polygon), and position relative to the robot's path
- Optional: add a darkened "wet zone" extending from the puddle edge

```python
# Simplified: place a reflective plane on the floor
from pxr import UsdGeom, UsdShade

def create_water_puddle(stage, position, radius):
    puddle = UsdGeom.Cylinder.Define(stage, "/World/WaterPuddle")
    puddle.CreateRadiusAttr(radius)
    puddle.CreateHeightAttr(0.001)  # Nearly flat
    puddle.CreateDisplayColorAttr([(0.2, 0.3, 0.4)])
    # Bind reflective water material
    water_mat = UsdShade.Material(stage.GetPrimAtPath("/World/Looks/Water"))
    UsdShade.MaterialBindingAPI(puddle.GetPrim()).Bind(water_mat)
```

**Auto-labeling:** Binary — puddle placed or not. Free.

**Challenge:** Visual realism. Keep expectations moderate — a reflective floor patch may not look like a real puddle, but the goal is to teach the VLM the *concept* of "water on floor where it shouldn't be" across enough visual variants that it generalizes.

---

## 5. Proving Real-World Transfer

This is the critical question raised by both Dr. Riley and conference judges: *"Will it work in the real world?"*

### Strategy: Three Levels of Evidence

**Level 1 — In-Distribution Sim Accuracy (already done)**
- 96.3% on held-out test set (same door, same conditions)
- Proves the VLM can learn from synthetic data

**Level 2 — Out-of-Distribution Sim Generalization (Week 1)**
- Train on doors A, B, C, D, E. Test on held-out door F.
- Plot the generalization curve: 1 door (0%) → 3 doors → 5 doors → 8 doors
- This curve is the core proof: more synthetic diversity → better generalization
- If the curve shows clear improvement, the implication is: enough diversity → real-world transfer

**Level 3 — Real-World Photo Evaluation (Week 2)**
- Photograph real doors in Diercks Hall (DH) in open and closed states
- Run the sim-trained VLM on these real photographs
- Report accuracy on real-world images the model has never seen
- This is the most convincing evidence — zero sim images of DH, yet the model classifies correctly

### The Scaling Argument (speculative calculation Dr. Riley wants)

```
Manual approach:
  - 5 anomaly types × 10 buildings × 1,000 images each = 50,000 images
  - Human labeling at ~2 sec/image = 28 hours of annotation
  - Must be repeated for every new building
  - Cannot stage dangerous anomalies (fire, flood, break-in)

Digital twin approach:
  - 5 anomaly types × 10 buildings × 1,000 images each = 50,000 images
  - Generation time: ~2 hours of GPU compute (automated)
  - Zero human labeling (physics provides ground truth)
  - New building: import BIM file, run pipeline. ~30 minutes setup.
  - Can simulate ANY anomaly — no real-world staging required
```

At scale, the manual approach becomes impossible. The digital twin approach scales linearly with compute.

### Domain Randomization for Generalization

The key insight from Dr. Riley: *"Generate infinite permutations on the building to generalize to all other use cases."*

We randomize per-episode:
- **Lighting:** DomeLight intensity (10K–80K), color temperature (4K–7K), fill light position
- **Materials:** Door surface (aluminum, painted, wood, glass), wall colors, floor textures
- **Camera viewpoint:** Distance (1–3.5m), height variation, arc angle
- **Anomaly severity:** Door angle (continuous), glass break extent, puddle size

This forces the VLM to learn **structural cues** (door gap geometry, glass fracture patterns, floor reflections) rather than **pixel-level shortcuts** (specific brightness, specific texture).

---

## 6. Two-Week Execution Plan

### Week 1 (April 11–18): Data Generation + Training

| Day | Task |
|---|---|
| Fri–Sat | Parameterize `door_static_collection.py` to accept any door via CLI args. Identify 5–8 usable doors in the USD with working rotation geometry. |
| Sun–Mon | Add lighting randomization and material variation to collection script. Run data collection for all doors (5 SLURM jobs, ~20 min each). |
| Tue | Build broken glass simulation: material swap approach. Identify glass door/window prims in USD. Create cracked glass material variant. |
| Wed | Run broken glass data collection. Begin water leakage implementation if time permits. |
| Thu | Run `prepare_finetune_data.py` with expanded dataset. Add image augmentations for train split. Re-run fine-tuning on V100. |
| Fri | Evaluate: in-distribution accuracy, held-out door generalization, generalization curve (1→3→5→8 doors). |

### Week 2 (April 18–25): Real-World Transfer + Write-Up

| Day | Task |
|---|---|
| Sat–Sun | Photograph real doors in Diercks Hall — open and closed states, multiple doors, varied lighting. Photograph broken glass examples if available. |
| Mon | Run sim-trained VLM on real DH photographs. Report accuracy. |
| Tue | Water leakage data collection + training (stretch goal). |
| Wed | Generate final results: generalization curves, confusion matrices, real-world accuracy, per-anomaly breakdown. |
| Thu | Build scaling argument with speculative calculations. Prepare figures and tables. |
| Fri | Final documentation, update CLAUDE.md, update paper draft. |

### Compute Budget

| Task | Partition | Jobs | Est. Runtime |
|---|---|---|---|
| Door data collection (5–8 doors) | teaching (T4) | 5–8 | ~20 min each |
| Broken glass data collection | teaching (T4) | 3–5 | ~20 min each |
| Water leakage data collection | teaching (T4) | 2–3 | ~20 min each |
| Data preparation + augmentation | CPU | 1 | ~5 min |
| VLM fine-tuning (expanded dataset) | dgx (V100) | 2–3 | ~20–30 min each |
| Evaluation runs | dgx (V100) | 3–5 | ~5 min each |
| **Total estimated** | | **~20 jobs** | **~5–8 GPU-hours** |

---

## 7. Key Results We Need to Produce

### Must-Have Results

1. **Generalization curve for doors**: Accuracy on held-out door as a function of training door diversity (1 → 3 → 5 → 8 doors). This is the single most important result — it proves the pipeline works.

2. **Multi-anomaly accuracy**: Per-anomaly-type accuracy showing the VLM handles at least doors + broken glass from the same training pipeline.

3. **Real-world photo evaluation**: Accuracy on real photographs from Diercks Hall. Even 70%+ on real photos with zero real training data is a strong result.

4. **Scaling calculation**: Time/cost comparison of manual labeling vs. digital twin pipeline at 1, 10, and 50 buildings.

### Nice-to-Have Results

5. **Water leakage detection results**: Third anomaly type strengthening the generality argument.

6. **Domain randomization ablation**: Accuracy with vs. without lighting/material randomization, showing that diversity (not just quantity) drives generalization.

7. **Error analysis**: What does the model get wrong? Are failures systematic (certain angles, certain materials) or random?

---

## 8. Addressing Known Criticisms

| Criticism | Response |
|---|---|
| *"The digital twin looks like a plastic toy"* (conference judges) | Domain randomization (materials, lighting) + real-photo eval results showing transfer despite visual gap. The point isn't photorealism — it's that diverse synthetic data teaches structural understanding. |
| *"Sensors can detect doors"* (Dr. Riley v1) | A sensor detects one thing. The VLM detects open doors, broken glass, water leaks, and any future anomaly type — all from the same model. One sensor per anomaly type doesn't scale. |
| *"Roombas can navigate buildings"* (Dr. Riley v1) | Navigation is not the contribution. The contribution is the **synthetic data pipeline** that trains visual understanding. The robot is the delivery mechanism. |
| *"Smart cameras are a solved problem"* (Dr. Riley v2) | Agreed — we are not building a smart camera system. The mobile robot goes where cameras can't (blind spots, close-range inspection, parking garages) and sees what cameras miss (double-paned glass, floor-level water). |
| *"How will it transfer to the real world?"* (Dr. Riley v1 + v2) | Three levels of evidence: in-distribution sim accuracy, out-of-distribution sim generalization (held-out doors), and real-world photo evaluation on DH. The generalization curve is the proof. |
| *"What problem have you solved?"* (Dr. Riley v2) | Automated synthetic training data generation from BIM for VLM-equipped mobile robots. Nobody else has this pipeline. See [RELATED_WORK.md](RELATED_WORK.md). |

---

## 9. References

- See [RELATED_WORK.md](RELATED_WORK.md) for full literature review
- See [VLM_fine-tuning.md](VLM_fine-tuning.md) for existing fine-tuning technical details
- See [DATASET_IMPROVEMENT_PLAN.md](DATASET_IMPROVEMENT_PLAN.md) for original dataset expansion plan (doors only)
- See [CLAUDE.md](CLAUDE.md) for codebase documentation and engineering details

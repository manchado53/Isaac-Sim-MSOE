# Dataset Production Plan — 49,680-Frame Door-State Collection

This document is the authoritative reference for the production synthetic dataset
that will train a Qwen2-VL-2B door-state classifier to generalize across unseen
doors. Every parameter, design decision, and operational detail is specified here
so the collection can be executed, reproduced, and audited without ambiguity.

---

## 1. Objective

Train a VLM (Qwen2-VL-2B-Instruct, LoRA r=16 + vision projector) that classifies
door state as CLOSED or OPEN from a single RGB image, generalizing to door
instances never seen during training. The prior 1,800-frame single-door dataset
achieved 96.3% in-distribution accuracy but **0% on unseen doors** — the model
memorized appearance, not the open/closed concept. This dataset fixes that failure
through multi-door, multi-material, multi-lighting domain randomization.

---

## 2. Dataset dimensions

| Axis | Count | Values |
|------|-------|--------|
| Training doors | 8 | See [Section 4](#4-door-registry) |
| Holdout doors | 2 | OOD generalization test set |
| Lighting presets | 6 | bright_fluorescent, daylight_flood, warm_evening, dim_night, overcast_flat, high_contrast |
| Door material presets | 6 | fill_red, fill_white, fill_wood, fill_tex_red_brushed, fill_tex_wood_grain, fill_tex_dark_brushed |
| Angle bands | 6 | 3 CLOSED + 3 OPEN (see [Section 5](#5-angle-bands-and-label-balance)) |
| Camera positions | 230 total | Per-door deterministic set from occlusion probe (arc-corrected) |

**Cells per door** = 6 lightings x 6 materials x 6 angle bands = **216**

**Frames per door** = 216 x (door's camera count)

**Total training frames** = 216 x 230 = **49,680**

---

## 3. Frame count arithmetic

```
6 lightings x 6 materials x 6 angles = 216 cells/door
216 cells x 230 cameras (across 8 doors) = 49,680 frames

Per-door breakdown:
  Doors_13606056  CW_F_NEW   45 cams  x 216 =  9,720
  Doors_13606466  CW_F_NEW   30 cams  x 216 =  6,480
  Doors_8286057   Type_F     27 cams  x 216 =  5,832
  Doors_3135827   CW_FG_New  30 cams  x 216 =  6,480
  Doors_10649156  Type_FG    27 cams  x 216 =  5,832
  Doors_7171063   Type_F     21 cams  x 216 =  4,536
  Doors_12302795  CW_FG_New  21 cams  x 216 =  4,536
  Doors_12590968  Type_N     29 cams  x 216 =  6,264
  ─────────────────────────────────────────────────────
  TOTAL                     230 cams         49,680
```

**Per-family totals:**

| Family | Doors | Cameras | Frames | % of dataset |
|--------|-------|---------|--------|-------------|
| CW_F_NEW | 2 | 75 | 16,200 | 32.6% |
| CW_FG_New | 2 | 51 | 11,016 | 22.2% |
| Type_F | 2 | 48 | 10,368 | 20.9% |
| Type_N | 1 | 29 | 6,264 | 12.6% |
| Type_FG | 1 | 27 | 5,832 | 11.7% |

The arc-corrected occlusion probe (rotating the camera arc to match each
door's facing direction) significantly improved Type_N coverage from 12 to
29 cameras. Type_N is still the smallest single-door contributor but now has
proportionally less. Oversampling in the dataloader can compensate if needed.

---

## 4. Door registry

### Training doors (8)

| Door ID | Family | Floor | Cameras | Swing | Role |
|---------|--------|-------|---------|-------|------|
| Doors_13606056 | CW_F_NEW | 3 | 45 | Inward | train |
| Doors_13606466 | CW_F_NEW | 4 | 30 | Inward | train |
| Doors_8286057 | Type_F | 2 | 27 | Inward | train |
| Doors_3135827 | CW_FG_New | 3 | 30 | Inward | train |
| Doors_10649156 | Type_FG | 3 | 27 | Inward | train |
| Doors_7171063 | Type_F | 5 | 21 | **Outward** | train |
| Doors_12302795 | CW_FG_New | 2 | 21 | Inward | train |
| Doors_12590968 | Type_N | 3 | 29 | Inward | train |

### Holdout doors (2)

| Door ID | Family | Floor | Swing | Purpose |
|---------|--------|-------|-------|---------|
| Doors_13182085 | CW_FG_New | 4 | Inward | OOD test — seen family, unseen instance |
| Doors_1915613 | Type_F | 5 | Inward | OOD test — seen family, unseen instance |

Both holdout families (CW_FG_New, Type_F) appear in the training set, so the
holdout tests **instance-level** generalization within a seen family — the
cleanest possible generalization probe.

### Camera positions

Camera positions are **deterministic** per door, baked from the occlusion probe
results in `data/occlusion_probe_summary.json`. The probe tested a 45-position
candidate grid per door:

```
arc offsets:    [-60, -30, 0, +30, +60] degrees    (5)
distances:     [1.5, 2.0, 3.5] metres              (3)
height offsets: [-0.5, 0.0, +1.0] metres            (3)
                                          total: 45 candidates
```

Each candidate was filtered by a PhysX raycast from camera origin toward the
door centre. A position survives iff either no hit occurs or the first
rigid-body hit path contains the door's prim path (i.e., the camera can see
the door, not a wall or column).

Survivors per door range from 12 (Type_N, tight stairwell) to 45 (CW_F_NEW
floor 3, open corridor). The exact clear positions are loaded at collection
startup — no recomputation needed.

---

## 5. Angle bands and label balance

The 5-degree threshold (angle <= 5 -> CLOSED, angle > 5 -> OPEN) is unchanged
from the prior dataset. Six angle bands provide boundary-weighted sampling:

| Band | Range (degrees) | Label | Purpose |
|------|----------------|-------|---------|
| CLOSED-1 | `uniform(0.0, 1.5)` | CLOSED | Fully-closed anchor |
| CLOSED-2 | `uniform(1.5, 3.0)` | CLOSED | Near-boundary closed |
| CLOSED-3 | `uniform(3.0, 5.0)` | CLOSED | Hard-boundary closed |
| OPEN-1 | `uniform(6.0, 15.0)` | OPEN | Near-boundary open |
| OPEN-2 | `uniform(15.0, 50.0)` | OPEN | Mid-range open |
| OPEN-3 | `uniform(51.0, 90.0)` | OPEN | Fully-open anchor |

**Label balance**: 3 CLOSED bands x N cameras = 3 OPEN bands x N cameras.
Exact 50/50 CLOSED/OPEN per door, per family, and globally.

**Boundary emphasis**: CLOSED-3 (3-5 degrees) and OPEN-1 (6-15 degrees)
straddle the decision boundary. The prior 96.3% model's errors all fell in the
1-14 degree range — these bands force the model to learn the boundary precisely.

Each cell draws one fresh random angle from its band — no two cells at the
same band produce the same angle.

---

## 6. Lighting presets

Six discrete indoor presets from `LIGHTING_AND_MATERIALS.md`, unchanged.
Each drives two physical light sources created once at startup:

1. **DomeLight** (`/World/DomeLight`) — ambient fill
2. **4x RectLight ceiling array** (`/World/CeilingLight_0..3`) — 0.6m squares,
   1.4m above camera Z, facing downward, 2x2 layout

| Preset | Dome (int / K) | Ceiling pattern | Ceiling (int / K) | Analogue |
|--------|---------------|-----------------|-------------------|----------|
| bright_fluorescent | 15,000 / 5500 | (1,1,1,1) | 400,000 / 5000 | Daytime hallway |
| daylight_flood | 50,000 / 6500 | (1,0,1,0) | 230,000 / 6500 | Window-lit, cool |
| warm_evening | 8,000 / 3000 | (1,1,1,1) | 200,000 / 2700 | After-hours tungsten |
| dim_night | 2,000 / 4000 | (1,0,1,0) | 150,000 / 3500 | Overnight security |
| overcast_flat | 25,000 / 5000 | (1,1,1,1) | 220,000 / 5000 | Diffuse, all dim |
| high_contrast | 2,000 / 5000 | (0,1,0,0) | 700,000 / 5500 | Single spotlight |

**Per-cell jitter** (applied each time a preset is set):
- Intensity: +/-10%
- Color temperature: +/-200K

This prevents pixel-level memorization within a preset while keeping presets
visually distinct.

---

## 7. Door material presets

Six uniform presets, all using `fill_*` techniques (bind to both frame and
glass meshes). This means every preset works identically on curtain-wall and
solid door families — no family-specific material logic.

| # | Preset | Technique | Look |
|---|--------|-----------|------|
| 0 | fill_red | fill_glass | Flat matte red |
| 1 | fill_white | fill_glass | Flat off-white |
| 2 | fill_wood | fill_glass | Flat warm brown |
| 3 | fill_tex_red_brushed | fill_textured | Brushed red metal |
| 4 | fill_tex_wood_grain | fill_textured | Wood-grain veneer |
| 5 | fill_tex_dark_brushed | fill_textured | Dark anodised metal |

### Material implementation

All material prims are **pre-created at startup** under `/World/Looks/DoorMat_*`.
The per-cell loop only flips `MaterialBindingAPI` binding relationships — it
never mutates shader inputs. This avoids the OmniPBR shader cache bug where
first-set color sticks regardless of later writes.

For `fill_glass` (flat colour):
- Shader: `OmniPBR.mdl` with `diffuse_color_constant`

For `fill_textured` (procedural pattern):
- Shader: `OmniPBR.mdl` with `diffuse_texture` (grayscale PNG) + `diffuse_tint`
- `project_uvw=True` (Revit meshes have no UVs)
- `world_or_object=False` (object-space projection so texture follows the door)

### Procedural textures

Three 512x512 grayscale PNGs generated at startup:

| Pattern | Generation | Intensity range |
|---------|-----------|-----------------|
| brushed | 0.88 + per-row noise +/-0.04 + per-pixel +/-0.01 | 0.83 - 0.93 |
| grain | 0.88 + 0.05*sin(band) + per-pixel +/-0.02 | 0.81 - 0.95 |
| speckle | 0.88 + per-pixel +/-0.03 | 0.85 - 0.91 |

The narrow intensity range (centred at 0.88) ensures that `diffuse_tint`
multiplication stays within one perceptual hue — the original wide range
(0.65-1.0) caused multi-colour streaks where dark bands shifted hue relative
to bright bands.

### Binding swap per cell

```
1. Restore original Revit bindings (frame + glass)
2. Look up pre-created material prim for this preset
3. Walk door subtree, bind material to all meshes (frame + glass)
   with bindingStrength = strongerThanDescendants
```

Cost per cell: O(meshes under door) USD attribute writes. No shader compile.

---

## 8. Wall colour randomization (C1)

### What it is

C1 changes the colour of wall surfaces surrounding the door in frame. This
breaks the correlation between wall appearance and door identity — if every
training frame of Door_X shows the same gray wall, the model can use wall
colour as a spurious feature.

### Which doors get C1

| Family | C1 applied? | Reason |
|--------|------------|--------|
| Type_F | Yes | Set into drywall/CMU wall |
| Type_FG | Yes | Glass panel in solid HM frame, real wall behind |
| Type_N | Yes | Narrow solid door in wall |
| CW_F_NEW | No | Curtain wall IS the wall — no `Walls` prims |
| CW_FG_New | No | Same |

### How it works

- **Not a grid axis** — wall colour does NOT multiply frame count.
- Each cell on a solid door gets a **random draw** from a 6-colour pool.
- The pool: warm_beige, cool_gray, terracotta, sage_green, muted_blue, dark_charcoal.
- Wall meshes are discovered at startup by walking `/Geometry/Walls/`, filtering
  by distance (< 5m) and floor Z-band (+/- 4m) from the door centre.
- Original wall bindings are cached and restored before each cell's random draw.
- Same OmniPBR binding-swap mechanism as door materials.

### The asymmetry is intentional

Curtain-wall doors see 0 wall variation. Solid doors see 6 random colours.
This is correct: the model must learn that background context is irrelevant
to door state classification.

---

## 9. Camera jitter

Per-cell random perturbation on top of baked camera positions:

| Axis | Range |
|------|-------|
| X position | +/-0.15 m |
| Y position | +/-0.15 m |
| Z position | +/-0.10 m |
| Pitch | +/-2.0 degrees |
| Roll | +/-1.5 degrees |

This prevents the model from memorizing pixel-aligned features at exact
camera positions while keeping the door centred in frame.

---

## 10. Sharding strategy — 8-way SLURM array

Each training door runs as a separate SLURM array task.

```bash
#SBATCH --partition=teaching
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --array=0-7
```

### Why per-door sharding

- **Restart cost**: if one shard fails, re-run that one task (~2 hr), not the
  whole 14 hr sequential job.
- **Parallelism**: 8 T4 nodes -> wall-clock drops from ~14 hr to ~3.2 hr
  (longest shard).
- **Memory isolation**: each task loads the building USD once and manipulates
  one door. No cross-door state leakage.

### Array task mapping

A lookup table in the script maps `$SLURM_ARRAY_TASK_ID` (0-7) to the door
config (prim path, hinge coordinates, camera positions, family, angle_sign).

### Estimated per-shard times

| Task ID | Door | Cameras | Cells | Frames | Est. time |
|---------|------|---------|-------|--------|-----------|
| 0 | Doors_13606056 | 45 | 216 | 9,720 | ~3.2 hr |
| 1 | Doors_13606466 | 30 | 216 | 6,480 | ~2.2 hr |
| 2 | Doors_8286057 | 27 | 216 | 5,832 | ~2.0 hr |
| 3 | Doors_3135827 | 30 | 216 | 6,480 | ~2.2 hr |
| 4 | Doors_10649156 | 27 | 216 | 5,832 | ~2.0 hr |
| 5 | Doors_7171063 | 21 | 216 | 4,536 | ~1.5 hr |
| 6 | Doors_12302795 | 21 | 216 | 4,536 | ~1.5 hr |
| 7 | Doors_12590968 | 29 | 216 | 6,264 | ~2.1 hr |

4 hr SLURM limit gives ~1.8x headroom on the longest shard.

---

## 11. Randomization seeding

Each shard seeds its RNG as:

```python
rng = np.random.default_rng(seed=42 + SLURM_ARRAY_TASK_ID)
random.seed(42 + SLURM_ARRAY_TASK_ID)
```

This guarantees:
- **Determinism**: rerunning task 3 reproduces the exact same angles, jitter,
  wall colours, lighting jitter.
- **Independence**: no two doors share a random sequence.
- **Reproducibility**: the seed is recorded in each NPZ's metadata.

Per-cell randomized quantities:
- Door angle (within band range)
- Lighting jitter (+/-10% intensity, +/-200K temp)
- Camera jitter (+/-0.15m XY, +/-0.10m Z, +/-2 deg pitch, +/-1.5 deg roll)
- Wall colour (solid doors: random draw from 6-colour pool per cell)

---

## 12. Output layout

### Per-shard output

```
data/
  door_production_Doors_13606056.npz
  door_production_Doors_13606466.npz
  door_production_Doors_8286057.npz
  door_production_Doors_3135827.npz
  door_production_Doors_10649156.npz
  door_production_Doors_7171063.npz
  door_production_Doors_12302795.npz
  door_production_Doors_12590968.npz
```

### NPZ schema

Byte-compatible with `prepare_finetune_data.py`:

| Key | Type | Shape | Description |
|-----|------|-------|-------------|
| frames | uint8 | (N, 768, 1024, 3) | RGB frames |
| labels | float32 | (N,) | angle / 90.0 (regression-compatible) |
| angles | float32 | (N,) | Raw angle in degrees |
| episode | int32 | (N,) | Cell index within this door |
| cam_offsets | int32 | (N,) | Arc offset in degrees |
| light_idx | int32 | (N,) | Lighting preset index (0-5) |
| material_idx | int32 | (N,) | Material preset index (0-5) |
| angle_band | int32 | (N,) | Band index (0-5) |
| wall_color_idx | int32 | (N,) | Wall colour index (0-5, or -1 for curtain-wall) |
| door_id | str | scalar | Door ID string |
| light_names | str[] | (6,) | Preset names in index order |
| material_names | str[] | (6,) | Preset names in index order |

---

## 13. Train / val / holdout split

### Structure

| Split | Source | Size | Purpose |
|-------|--------|------|---------|
| **Train** | 85% of 8 training doors | ~38,005 frames | Fine-tuning |
| **Val** | 15% of 8 training doors | ~6,707 frames | Loss monitoring, early stopping |
| **Holdout test** | 2 holdout doors (separate collection) | ~4,000-5,000 frames | OOD generalization evaluation |

### Why no separate test split

The holdout doors ARE the test set. They answer the question that matters:
"does this model generalize to unseen door instances?" An in-distribution test
split from training doors would only confirm what val already shows.

### Stratification

`prepare_finetune_data.py` splits using scikit-learn's `StratifiedShuffleSplit`
on `(door_id, angle_band, light_idx)` as the stratification key. This ensures
every door, family, lighting preset, and angle band appears proportionally in
both train and val.

### Holdout collection

The 2 holdout doors are collected with the **identical** 216-cell grid
(same 6 lightings, 6 materials, 6 angle bands). They run as a separate
2-task SLURM array, submitted alongside the training collection.

---

## 14. Image augmentation (train split only)

Applied at JSONL-generation time or by the fine-tuning dataloader. **Never
applied to val or holdout.**

| Augmentation | Parameters | Purpose |
|-------------|-----------|---------|
| ColorJitter | brightness=0.2, contrast=0.2, saturation=0.15, hue=0.05 | Sim-to-real colour shift |
| GaussianBlur | kernel 3-7, sigma 0.1-2.0 | Camera defocus |
| HorizontalFlip | p=0.5 | Viewpoint symmetry |
| RandomResizedCrop | scale 0.85-1.0 | Slight framing variation |
| JPEG re-compression | quality 60-95 | Real-world compression artifacts |

These augmentations target the sim-to-real gap: real photographs have lens
blur, JPEG artefacts, and colour shifts that synthetic renders don't. Applying
them only to training data teaches the model to be invariant to these
transforms without contaminating the evaluation signal.

---

## 15. Per-cell collection loop

For each `(lighting, material, angle_band)` cell on each door:

```
1. apply_lighting(preset)          # dome + 4 ceiling light attrs + jitter
2. apply_wall_material(random)     # C1: random wall colour (solid doors only)
3. apply_door_material(preset)     # bind pre-created material to door meshes
4. sample_angle(band)              # random angle within band range
5. set_door_angle(angle)           # rewrite N x 4x4 xform matrices
6. apply_camera_jitter()           # per-camera random pose offset
7. warmup(10 steps)                # let renderer settle
8. for each camera:
9.     world.step(render=True)
10.    rgb = annotator.get_data()  # Replicator readout
11.    append to buffers
12. restore door to 0 degrees      # clean state for next cell
```

All variation is **USD attribute writes** — no shader recompile, no scene
reload. The renderer picks up deltas on the next render step.

---

## 16. Sanity gates

Automated checks after all shards complete, before `prepare_finetune_data.py`:

| Gate | Check | Action on failure |
|------|-------|-------------------|
| Frame count | Each NPZ has exactly `216 x cam_count` frames | Re-run that shard |
| Label balance | Each NPZ has exactly 50% CLOSED / 50% OPEN | Investigate angle sampling |
| Band coverage | Each of 6 bands has `36 x cam_count` frames | Investigate loop logic |
| Black frame | Mean pixel value < 10 for any frame | Lighting bug — inspect preset |
| Duplicate frame | SHA256 hash collision between any two frames | Render stall — increase warmup |
| Visual spot check | 8 contact sheets (one per door), 36 cells each | Human eyeball |

---

## 17. Compute budget

| Stage | Partition | GPUs | Wall-clock | GPU-hours |
|-------|-----------|------|-----------|-----------|
| Training collection (8 shards) | teaching (T4) | 8 parallel | ~3.2 hr | 25.6 |
| Holdout collection (2 shards) | teaching (T4) | 2 parallel | ~2.0 hr | 4.0 |
| prepare_finetune_data.py | dgx (V100) | 1 | ~10 min | 0.2 |
| Fine-tuning (3 epochs) | dgx (V100) | 1 | ~6.7 hr | 6.7 |
| Evaluation (val + holdout) | dgx (V100) | 1 | ~20 min | 0.3 |
| **Total** | | | **~10.2 hr wall-clock** | **~36.8 GPU-hr** |

Collection and holdout can run simultaneously. Fine-tuning starts after both
complete. Total start-to-finish: ~10 hours assuming nodes are available.

### Disk budget

```
Training NPZs:  49,680 x ~350 KB  = ~17.4 GB
Holdout NPZs:   ~5,000 x ~350 KB  =  ~1.7 GB
JPEG export:     49,680 x ~55 KB   =  ~2.7 GB
Adapter:                            =  ~0.14 GB
                                     ─────────
Total:                               ~22 GB
```

---

## 18. Fine-tuning configuration

Unchanged from the prior successful run, except for dataset size:

| Parameter | Value |
|-----------|-------|
| Base model | Qwen2-VL-2B-Instruct |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Target modules | q_proj, v_proj |
| modules_to_save | model.visual.merger (34M params) |
| Total trainable | ~36.3M / 2.2B (1.62%) |
| Epochs | 3 |
| Batch size | 2 (per device) |
| Gradient accumulation | 4 (effective batch 8) |
| Learning rate | 1e-5 |
| Scheduler | Cosine |
| Warmup ratio | 0.05 |
| Hardware | V100 32 GB (dgx partition) |
| Estimated runtime | ~6.7 hr |

### Critical engineering details

- **`modules_to_save=["model.visual.merger"]`** in LoRA config — without this,
  PEFT silently discards the 34M trained projector weights, and the model
  reverts to all-OPEN predictions (50% accuracy).
- **fp16 + AMP**: cast projector params to fp32 after `get_peft_model()`.
- **Do NOT call `merge_and_unload()`** — crashes with `modules_to_save`.
  Use `PeftModel` directly at inference.

---

## 19. Evaluation plan

### In-distribution (val split)

- 6,707 frames from training doors, stratified
- Expected accuracy: >95% (prior single-door achieved 96.3%)
- Primary metric: overall accuracy, per-class accuracy, F1

### Out-of-distribution (holdout doors)

- ~4,000-5,000 frames from 2 unseen door instances
- This is the KEY metric — must be significantly above the prior 0%
- Target: >80% on holdout (would demonstrate successful generalization)
- Report per-family, per-angle-band, per-lighting breakdown

### Sim-to-real (real door photos)

- Evaluate on real photographs of doors in Diercks Hall
- Separate eval script (`eval_real_doors.py`)
- No accuracy target — this is an exploratory evaluation of sim-to-real transfer

---

## 20. Risks and mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| T4 node unavailability (8 GPUs) | Delayed start | Array tasks queue independently; partial parallelism still helps |
| Single shard crash | Lose 1 door's data (~2 hr) | Re-run that array task. Per-door sharding limits blast radius |
| NFS stale write on NPZ | Corrupt training data | `os.fsync()` after every NPZ write. `prepare_finetune_data.py` on compute node |
| Type_N underrepresentation (5.8%) | Model weak on narrow doors | Accepted. Can oversample in dataloader if needed |
| 6.7 hr fine-tune hits wall limit | Job killed | Use 21-day dgx partition. Fallback: 2 epochs (4.4 hr) |
| Generalization still fails | Research negative | Document. Investigate family-specific vs instance-specific failure. Add doors or switch approach |
| Disk quota exceeded (~20 GB) | Can't write NPZs | Check quota before submission. Delete prior dataset if needed |

---

## 21. Execution checklist

```
[ ] 1. Verify disk quota (need ~20 GB free)
[ ] 2. Submit training collection:
       sbatch submit_production_collection.sh   # --array=0-7
[ ] 3. Submit holdout collection:
       sbatch submit_production_collection.sh --holdout  # --array=0-1
[ ] 4. Monitor: squeue -u $USER
[ ] 5. Verify all 8+2 shards completed (exit code 0)
[ ] 6. Run sanity gates (frame count, label balance, black frame check)
[ ] 7. Run prepare_finetune_data.py on compute node (dgx)
[ ] 8. Verify train.jsonl (38k lines) + val.jsonl (6.7k lines)
[ ] 9. Submit fine-tuning: sbatch submit_finetune_job.sh
[ ] 10. Monitor training loss curve
[ ] 11. Run evaluation on val + holdout
[ ] 12. Report results
```

---

## 22. Files produced by this plan

| File | Description |
|------|-------------|
| `simulations/robot/door_production_collection.py` | Production collection script (SLURM array) |
| `submit_production_collection.sh` | SLURM submission wrapper |
| `data/door_production_*.npz` | 8 training + 2 holdout NPZs |
| `data/finetune_images_v2/` | JPEG export for fine-tuning |
| `data/train.jsonl` / `data/val.jsonl` | 85/15 split JSONL |
| `data/qwen2vl_door_lora_v2/` | Trained adapter (LoRA + merger) |
| `data/eval_results_v2.json` | Val + holdout evaluation results |

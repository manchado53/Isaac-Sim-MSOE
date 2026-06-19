# VLM Baseline Comparison

Zero-shot evaluation of six off-the-shelf vision-language models on a held-out
door-state classification dataset rendered from the MSOE digital twin. The
goal is to establish where current open-weight VLMs sit on a real
building-inspection task **before** any fine-tuning — a baseline the
Qwen2-VL-2B LoRA + projector fine-tune (96.3% accuracy) can be measured
against.

---

## Models under test

All six models are small enough to run on a single V100 32 GB (the `dgx`
partition) in fp16.

| # | Model | Params | Why it's in the comparison |
|---|---|---|---|
| 1 | **Qwen2-VL-2B-Instruct** | 2.2 B | Same base checkpoint we fine-tuned. Its zero-shot number is the direct reference point for the fine-tune delta. |
| 2 | **Qwen2.5-VL-3B-Instruct** | 3.0 B | Next-generation Qwen VL, slight size bump. Tells us whether a newer model closes the gap without any task-specific training. |
| 3 | **Qwen3-VL-2B-Instruct** | 2.0 B | Newest Qwen VL generation at the 2 B tier. Same parameter budget as #1, so any gain is pure architecture/training-data improvement. |
| 4 | **Qwen2.5-VL-7B-Instruct** | 7.0 B | Scale comparison within a single family — does a ~2× parameter jump buy meaningful zero-shot accuracy on this task? |
| 5 | **SmolVLM2-2.2B** | 2.2 B | Alternative architecture (HuggingFace SmolVLM family). Isolates the Qwen family's bias by testing a completely different training recipe at the same size. |
| 6 | **InternVL3-2B** | 2.0 B | Strong performer on public VLM benchmarks, different image tokenization + training pipeline from Qwen. Needs a custom wrapper (`internvl_wrapper.py`) because its inference API is non-standard. |

### Why these six

- **One family, three generations** (Qwen2 → Qwen2.5 → Qwen3) isolates the
  effect of training-data + recipe improvements holding size roughly
  constant.
- **One family, two sizes** (Qwen2.5-VL-3B vs Qwen2.5-VL-7B) isolates the
  effect of scale.
- **Two non-Qwen models** (SmolVLM2, InternVL3) guard against the comparison
  being a Qwen-specific story.
- **All fit on a single V100** keeps the eval reproducible on hardware we
  actually have.

### What gets measured

Each model is asked the same question on the same 1008 test frames:

> Is the door in this image **open** or **closed**? Answer with one word.

We record: predicted label, gold label (from scene state), and per-frame
ground-truth door angle. Primary metric is overall accuracy; secondary
metrics are per-class accuracy, F1, and error rate in the ambiguous
1°–14° angle band (which is where the fine-tuned model's remaining errors
concentrate).

---

## How the baseline dataset was collected

### Scene

Rendered from `MSOEBuildingV4_physics.usd` (the 160 MB Revit export) in
Isaac Sim 5.1.0-rc.19 on a T4 GPU (`teaching` partition). Collection script:

```
VLM_comparison/door_baseline_dataset.py
```

Submitted via:

```
sbatch submit_isaac_job.sh VLM_comparison/door_baseline_dataset.py
```

### Door selection — 4 doors

We hand-picked 4 doors from the building's ~300-door catalog
(`simulations/robot/door_catalog.json`) that satisfy three requirements:

1. Opening into a walkable interior space a Nova Carter could patrol
2. Enough camera clearance on the approach side to place a 3-camera arc
3. Distinct from the single door used for fine-tuning
   (`Doors_13606713`) so this set is a genuine generalization test

Final IDs:

- `Doors_13607620`
- `Doors_9817744`
- `Doors_1868517`
- `Doors_10383956`

Each door's hinge inset is calibrated to 5 cm inward from the bbox edge
(`HINGE_OVERRIDES_CM`) — validated from the hinge-sweep job 242577. Without
this inset the rotated panel Z-fights the wall jamb and produces TV-static
noise.

### Camera placement — 3 per door, 12 total

For each door we place three cameras in a shallow arc on the approach side:

- Angular offsets: **-15°, 0°, +15°** from the approach axis
- Radius: **2.0 m** from the door centre
- Height: door bbox centre Z minus **0.25 m** (roughly robot chest height)

Each camera position is validated with a PhysX raycast to prune any that
end up inside a wall. The arc only covers the approach side — the "behind
the wall" hemisphere is always occluded.

### Angle sampling — random, per cell

For each `(door, camera)` pair we draw:

- **21** angles uniformly from `U[0°, 5°]` → labeled **CLOSED**
- **21** angles uniformly from `U[6°, 90°]` → labeled **OPEN**

These random draws replaced an earlier fixed 10-angle grid. Random sampling
spreads the ambiguous boundary band (1°–14°) more evenly and prevents the
test set from bunching at round-number angles.

Door rotation is applied by rewriting the mesh's `xformOp:transform`
directly as a 4×4 matrix built around the offset hinge — parent-xform
edits are unreliable under Revit's USD sublayer composition.

### Per-cell jitter — 2 captures

Each angle is captured **twice** with independent jitter:

- Camera position: ±30 cm along the arc radius
- Camera yaw: ±3°

This gives us slight parallax and framing variation without changing the
labeled door state.

### Total frame budget

```
4 doors × 3 cameras × (21 closed + 21 open) × 2 captures = 1008 frames
```

The target ~1000 came from: enough samples for stable accuracy estimates
(standard error below ±1.5% at 95% CI) while staying inside a single
24-minute Isaac Sim job on the T4 partition.

### Rendering — realistic lighting

The original lighting setup (one SphereLight directly above each door)
produced halos on the door head and visible oval reflections on glossy
walls. Current setup:

- **DomeLight** at intensity 40000 — HDRI-style ambient, lights interior
  rooms that have no nearby windows
- **DistantLight** at intensity 2500, angle 3°, rotated 45° down-and-forward —
  acts like daylight through a building skylight, gives panels directional
  highlights without introducing a visible source
- **No SphereLights or per-door fills** — they reflected as bright ovals on
  glossy wall finishes
- **ACES tonemap** (`/rtx/post/tonemap/op = 4`), bloom disabled, auto-exposure
  disabled — consistent exposure across every frame so the VLM can't
  accidentally exploit per-image brightness cues

Resolution: **768 × 576 px**, RTX Real-Time renderer.

### Labels

Each frame is saved with its true door angle from the rotation matrix.
The binary label is derived from a 5° threshold:

- angle ≤ 5° → `CLOSED`
- angle > 5° → `OPEN`

No human labeling — the scene state is authoritative.

### Output layout

```
data/vlm_baseline/
  frames/            # 1008 JPEGs, 768×576
  records.jsonl      # one record per frame (path, door_id, camera, angle, label, jitter)
  manifest.json      # run config: doors, angle ranges, seed, lighting, counts
  sanity/            # small inspection set from --sanity-check runs (2 angles × 3 cameras × 4 doors = 24 frames)
```

### Reproducibility

All randomness is seeded with `SEED = 20260412` so re-running the script
produces byte-identical angle draws and jitter. The lighting, tonemap,
and render settings are all captured in the script (not the scene USD),
so the 1008 frames regenerate from the repo state alone.

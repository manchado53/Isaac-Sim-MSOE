# Cracked-glass dataset pipeline

## Context

Build a training dataset of cracked-vs-intact glass panels in the MSOE
Innovation Center, modelled on Tanner's door-state VLM dataset
(`origin/tanner_vlm:simulations/robot/door_static_collection.py`).

We've locked in a validated rendering configuration (material + lighting)
and hand-curated 13 glass panels across the building. This plan describes
how to sweep over lighting and crack variants against those panels to
produce training frames.

## Status

**READY TO BUILD.** Inputs finalized:
- 13 curated panels: `SyntheticDataGen/cracked_glass/selected_panels.json`
- 3 crack texture variants: `assets/materials/cracked_glass/crack_{opacity,normal,roughness,tint}_{edge_v1,v4,tempered_v5}.png`
- Material authoring: `SyntheticDataGen/cracked_glass/glass_anomaly.py::create_cracked_glass_materials`
- Lighting recipe: Tanner's `bright_fluorescent` pattern (dome 25k + per-panel fill 400k + `rtx.domeLight.upperLowerStrategy=3` + `rtx.pathTracing.maxBounces=8`) validated in `cracked_glass_preview.py`

## Deliverables

Six new files. No destructive edits to existing code.

| Path | Role |
|---|---|
| `SyntheticDataGen/cracked_glass/lighting_presets.py` | 7 lighting presets + `setup_lighting_rig()` + `apply_lighting_preset()` helpers, lifted verbatim from `origin/tanner_vlm:door_static_collection.py` |
| `SyntheticDataGen/cracked_glass/cracked_glass_collection.py` | Main dataset collector — reads `selected_panels.json`, runs the sweep, writes JPEGs + labels |
| `SyntheticDataGen/cracked_glass/embed_train_frames.py` | DINOv2-base embeddings over the raw train split — adapted from `origin/tanner_vlm:simulations/robot/embed_train_frames.py` (swap `Doors_\d+` regex for `pan\d+` and `[CLOSED]/[OPEN]` label detector for `cracked/intact`) |
| `SyntheticDataGen/cracked_glass/curate_diversity.py` | Stratified k-means curation to 10k — lifted verbatim from `origin/tanner_vlm:simulations/robot/curate_diversity.py`; the bucket logic is label-agnostic |
| `submit_curate_diversity.sh` | SLURM wrapper (dgx/V100, 2h) chaining embed → curate. Lifted from `origin/tanner_vlm:submit_curate_diversity.sh` retargeted at our paths |
| `docs/cracked-glass-dataset.md` | User-facing documentation |

The existing `SyntheticDataGen/cracked_glass/cracked_glass_dataset.py` stays on disk with a 3-line deprecation note pointing at the new collector.

## Sweep axes

**Grid axes — full factorial, outer → inner:**

| Axis | Values | Count |
|---|---|---|
| Panel | 13 curated (`selected_panels.json`) | 13 |
| Lighting preset | Tanner's 6 | 6 |
| Arc angle | `-0.4 rad`, `0`, `+0.4 rad` | 3 |
| Crack state | `intact`, `edge_v1`, `v4`, `tempered_v5` | 4 |
| UV tiling | `1×`, `2×`, `4×` | 3 |
| Arc distance | `1.5`, `2.0`, `2.5` m | 3 |
| Height offset | `-0.5`, `0`, `+0.5` m | 3 |

**Total frames per run = 13 × 6 × 3 × 4 × 3 × 3 × 3 = 25,272.** Wallclock ~35 h on T4.

### Class balance: 25/75 raw → 50/50 curated

The grid gives 1 intact + 3 cracked variants per cell, so the raw dataset is 25% intact / 75% cracked. The intent is:

1. Generate the raw 25,272 frames with the full grid variety.
2. Embed every **train-split** frame with **DINOv2-base** (`facebook/dinov2-base`, CLS token, L2-normalized 768-d).
3. **Stratified k-means** per `(label, panel_idx)` bucket — pick the frame nearest each centroid. Target `k = 10,000` total, allocated 5,000 to intact and 5,000 to cracked (hard 50/50), then proportional per panel within each label.

So the 25/75 raw split is intentional — maximum geometric variety up front, balance + dedup at curation time.

- Train intact (raw): 4,860 frames (1 per geometric config × 10 panels)
- Train cracked (raw): 14,580 frames (3 variants per config × 10 panels)

**Intact-cap gotcha**: raw intact count (4,860) < 5,000 target. Tanner's `kmeans_select` returns `np.arange(n)` when `k ≥ n`, so we'll keep all 4,860 intact frames and take 5,000 from cracked. Final curated train set = **9,860 frames @ ~49.3/50.7 balance** — clean enough, no code change needed.

Val and test splits are **not** curated — kept whole so held-out metrics reflect the full distribution.

### Train / val / test panel-level split

Each panel has a `"split": "train"`, `"val"`, or `"test"` field in `selected_panels.json`. The collector routes frames into `frames/train/`, `frames/val/`, `frames/test/`.

| Split | Panels | Revit IDs | Raw frames | Intact | Cracked | After curation |
|---|---|---|---|---|---|---|
| **train** | 10 | `#2, 3, 4, 6, 13, 15, 31, 36, 40, 42` | **19,440** | 4,860 | 14,580 | **9,860** (5k intact-cap + 5k cracked) |
| **val** | 1 | `#22` | **1,944** | 486 | 1,458 | — (uncurated) |
| **test** | 2 | `#18, 28` | **3,888** | 972 | 2,916 | — (uncurated) |

Val = 1 panel (`#22`, floor 3 south lab) for fast mid-training checkpoints. Test = 2 held-out geographic siblings of multiple train-set panels, for the final generalization number:
- `#22` (val) — floor 3 south lab; only floor-3 panel → probes vertical/orientation generalization
- `#18` (test) — floor 2 ground corridor; sibling of 4 corridor panels (`#3, 13, 15, 31, 40`) in train
- `#28` (test) — floor 4 north atrium; sibling of 4 atrium panels (`#2, 6, 36, 42`) in train

**All axes are grid-factorial — no per-frame randomization remains.** UV tiling, arc distance, and height offset were promoted from per-frame random to full grid axes (see totals above) so every geometric combination is explicitly sampled rather than probabilistically covered.

## Labels

Binary per frame — matches Tanner's `CLOSED | OPEN`:

- **`cracked`** when focus panel's crack variant ∈ {`edge_v1`, `v4`, `tempered_v5`}
- **`intact`** when variant is the clear baseline

### JPEG filename

```
{intact|cracked}_pan{panel_idx:02d}_L-{light}_M-{variant_or_clear}_arc{deg:+03d}_f{frame:05d}.jpg
```

Example: `cracked_pan02_L-bright_fluorescent_M-tempered_v5_arc+23_f00417.jpg`

### Sidecar `labels.jsonl` — one line per frame

```json
{
  "frame_id": "cracked_pan02_L-bright_fluorescent_M-tempered_v5_arc+23_f00417",
  "label": "cracked",
  "split": "train",
  "variant": "tempered_v5",
  "panel_idx": 2,
  "panel_revit_id": "Curtain_Panels_10985143",
  "panel_prim": "/World/Building/.../Curtain_Panels_10985143",
  "panel_world_xyz": [2.14, 74.96, 28.99],
  "floor": 4,
  "lighting_preset": "bright_fluorescent",
  "lighting_jitter": {"dome_int_mul": 1.07, "dome_temp_delta": -145},
  "arc_angle_rad": 0.4,
  "arc_distance_m": 1.5,
  "uv_tiling": 2,
  "camera_xyz": [x, y, z]
}
```

### Columnar `collected.npz`

Same axes in array form: `frames (1092,H,W,3) uint8`, `labels (1092,) bool`, `variants (1092,) str`, `lightings (1092,) str`, `panel_idx (1092,) int`, `arc_angles (1092,) float`, `arc_distances (1092,) float`, `uv_tilings (1092,) int`, `floors (1092,) int`, `camera_xyz (1092,3) float`.

## File 1 — `SyntheticDataGen/cracked_glass/lighting_presets.py`

Lift from `origin/tanner_vlm:simulations/robot/door_static_collection.py`:

| Source lines | Destination |
|---|---|
| 190–220 (`LIGHTING_CONDITIONS`) | `LIGHTING_PRESETS` constant (renamed) |
| 347–359 (DomeLight + FillLight creation) | `setup_lighting_rig(stage, anchor_xyz)` |
| 610–632 (`_color_temp_to_rgb`) | private helper, verbatim |
| 638–670 (`apply_lighting_condition`) | `apply_lighting_preset(handles, name, rng=None)` |

Add one extra preset matching our camera-headlight baseline:

```python
LIGHTING_PRESETS["camera_headlight"] = dict(
    dome_int=150, dome_temp=5500,
    fill_int=0, fill_temp=5000,
    headlight_int=80000,
)
```

`setup_lighting_rig()` also sets carb settings validated in
`cracked_glass_preview.py`:

```python
carb.settings.get_settings().set("/rtx/domeLight/upperLowerStrategy", 3)
carb.settings.get_settings().set("/rtx/pathTracing/maxBounces", 8)
```

## File 2 — `SyntheticDataGen/cracked_glass/cracked_glass_collection.py`

Main collector. Modelled on `door_static_collection.py:880-952` (grid loop) + `816-850` (frame capture).

### CLI

```
./python.sh SyntheticDataGen/cracked_glass/cracked_glass_collection.py \
    --selected-panels SyntheticDataGen/cracked_glass/selected_panels.json \
    --variants edge_v1,v4,tempered_v5 \
    --lightings bright_fluorescent,daylight_flood,warm_evening,dim_night,overcast_flat,high_contrast,camera_headlight \
    --arc-angles -0.4,0,0.4 \
    --arc-distances 1.5,2.0,2.5 \
    --uv-tilings 1,2,4 \
    --seed 42 \
    --output datasets/cracked_glass/$(date +%Y%m%d_%H%M%S)
```

### Flow

1. Load `selected_panels.json`, launch `SimulationApp(headless=True)`, load Innovation Center USD.
2. `create_cracked_glass_materials(stage, variants=("edge_v1","v4","tempered_v5"))` + one intact baseline material (one material set, reused for all panels).
3. `setup_lighting_rig(stage, anchor_xyz=<building centroid>)` → returns handles dict.
4. For each of the 13 panels, pre-create 3 camera prims at `arc_angles` offsets × sampled `arc_distance`. Keep around for the whole run.
5. Attach one `rep.AnnotatorRegistry.get_annotator("rgb")` per camera. **Direct `.get_data()` reads**, no `BasicWriter` (avoids the flush bug documented in `docs/cracked-glass-pipeline.md`).
6. Main loop: `for panel in curated: for lighting in presets: for variant in {intact, edge_v1, v4, tempered_v5}: for arc_angle in arcs:`
   - Bind the chosen variant's material to the panel (re-binding panel-only per cell; other 12 panels stay intact)
   - Re-sample `uv_tiling` for this frame
   - `apply_lighting_preset(handles, lighting, rng=rng)` (jitter on)
   - 5 × `rep.orchestrator.step(rt_subframes=4)`
   - Read RGB → save JPEG, append row to `labels.jsonl` + in-memory frame buffer
7. After loop: write `collected.npz`.

### Output tree

```
datasets/cracked_glass/20261204_1500/
├── frames/
│   ├── train/                   (19,440 JPEGs from 10 training panels)
│   ├── val/                     (1,944  JPEGs from panel #22)
│   └── test/                    (3,888  JPEGs from panels #18, #28)
├── labels.jsonl                 (all 25,272 rows; "split" field = "train"|"val"|"test")
├── labels_train.jsonl           (19,440 rows — filtered; input to embed_train_frames.py)
├── labels_val.jsonl             (1,944 rows)
├── labels_test.jsonl            (3,888 rows)
├── train_embeddings.npy         (after embed stage — cached, (19440, 768) float32)
├── train_embedding_meta.json    (after embed stage)
├── train_curated.jsonl          (after curate stage — ~9,860 rows @ ~50/50)
├── train_curated_summary.json   (curation diagnostics)
└── metadata.json                (run config + cli args + git sha)
```

(NPZ dropped — at this scale the array is ~160 GB; load JPEGs lazily via the JSONL instead.)

## File 3 — `SyntheticDataGen/cracked_glass/embed_train_frames.py`

Port of `origin/tanner_vlm:simulations/robot/embed_train_frames.py` with two swaps:

| Tanner's regex / parser | Our version |
|---|---|
| `DOOR_ID_RE = re.compile(r"(Doors_\d+)")` | `PANEL_ID_RE = re.compile(r"pan(\d+)")` — matches filename `pan02_...` → panel_idx `2` |
| `label = "CLOSED" if "[CLOSED]" in answer.upper() else "OPEN"` | `label = rec["label"]` — read directly from `labels.jsonl` (`"intact"` / `"cracked"`) |

Everything else stays — DINOv2-base fp16, CLS token of last hidden state, L2-normalization so Euclidean ≡ cosine, `.npy` + meta JSON output.

**Input**: `datasets/cracked_glass/<timestamp>/frames/train/` (the 19,440 JPEGs) + `labels.jsonl` restricted to `"split": "train"`.
**Output**:
- `datasets/cracked_glass/<timestamp>/train_embeddings.npy` — `(19440, 768)` float32, L2-normalized
- `datasets/cracked_glass/<timestamp>/train_embedding_meta.json` — per-frame `{image, label, panel_idx, line_idx}`

Expected wallclock: **~5 min on 5090** (batch 64, ~70 fps). Tanner's T4 baseline was ~15-20 min for the same scale.

## File 4 — `SyntheticDataGen/cracked_glass/curate_diversity.py`

Verbatim copy of `origin/tanner_vlm:simulations/robot/curate_diversity.py`. The bucket logic keys on `(label, door_id)` — we pass `(label, panel_idx)` instead and the code doesn't care. Faiss-first with sklearn `MiniBatchKMeans` fallback is kept as-is.

**CLI**:
```
python SyntheticDataGen/cracked_glass/curate_diversity.py \
    --k 10000 \
    --emb datasets/cracked_glass/<timestamp>/train_embeddings.npy \
    --meta datasets/cracked_glass/<timestamp>/train_embedding_meta.json \
    --jsonl datasets/cracked_glass/<timestamp>/labels_train.jsonl \
    --out_jsonl datasets/cracked_glass/<timestamp>/train_curated.jsonl \
    --seed 42
```

**Output**:
- `train_curated.jsonl` — the curated 10k (actually ~9,860 after the intact-cap) — drop-in replacement for `labels_train.jsonl` during fine-tuning
- `train_curated_summary.json` — label balance, per-panel distribution, bucket allocation, selected indices

## File 5 — `submit_curate_diversity.sh`

Lifted from Tanner's script. Two SLURM stages (embed → curate), caches embeddings so curation can be re-run with different `--k` without re-embedding. Targets `--partition=dgx --gres=gpu:1 --mem=48G --time=2:00:00`. Installs `faiss-cpu` via pip on first run.

Since we're on the 5090 home machine, we can also run both stages locally:

```
python SyntheticDataGen/cracked_glass/embed_train_frames.py \
    --jsonl datasets/cracked_glass/<timestamp>/labels_train.jsonl \
    --batch 128
python SyntheticDataGen/cracked_glass/curate_diversity.py --k 10000
```

Total local wallclock: **~7 min**.

## File 6 — `docs/cracked-glass-dataset.md`

Sections:
1. **Goal** — what, for whom
2. **Output format** — dir tree, JPEG naming, JSONL schema, NPZ arrays
3. **Running it** — single CLI command, smoke test
4. **Sweep axes** — the tables from this plan
5. **Lighting presets** — 7-row table (preset, intensities, real-world analogue)
6. **Curated panels** — where they are in the building, how to regenerate the selection
7. **Known gotchas** — BasicWriter flush (avoided), `simulation_app.close()` segfault (not relevant here)
8. **Relationship to other scripts** — `cracked_glass_preview.py`, `build_cracked_scene_usd.py`, `cracked_glass_floor4_stream.py`, this new collector

Cross-reference from `docs/cracked-glass-pipeline.md` → the new doc.

## Files reused (no edits)

| Path | Used for |
|---|---|
| `origin/tanner_vlm:simulations/robot/door_static_collection.py` | Lift lighting presets + color-temp + grid-loop pattern (lines 190-220, 347-359, 610-670) |
| `origin/tanner_vlm:simulations/robot/embed_train_frames.py` | DINOv2 embedding pipeline (commit `207c0ba`) — swap regex + label parser |
| `origin/tanner_vlm:simulations/robot/curate_diversity.py` | Stratified k-means curation (commit `207c0ba`) — verbatim copy |
| `origin/tanner_vlm:submit_curate_diversity.sh` | SLURM wrapper for embed+curate (commit `207c0ba`) |
| `SyntheticDataGen/cracked_glass/selected_panels.json` | The 13 curated panel targets |
| `SyntheticDataGen/cracked_glass/glass_anomaly.py` | `create_cracked_glass_materials`, `enumerate_glass_prims`, `bind_visual_material` |
| `SyntheticDataGen/cracked_glass/cracked_glass_preview.py::panel_pose` | Panel outward-normal + always-inside camera logic |
| `SyntheticDataGen/cracked_glass/build_cracked_scene_usd.py:68-84` | `bind_visual_material_deep` (copy inline) |
| `sim_utils/building_physics.py` | `setup_physics_scene`, `_find_mesh_descendant` |
| `sim_utils/building_info.py` | `get_floor()` for floor bucketing |
| `assets/materials/cracked_glass/crack_{opacity,normal,roughness,tint}_{edge_v1,v4,tempered_v5}.png` | Three validated texture sets |

## Files edited (minor)

| Path | Edit |
|---|---|
| `SyntheticDataGen/cracked_glass/cracked_glass_dataset.py:1` | 3-line deprecation note pointing to the new collector |

## Verification

1. **Smoke test** (≤ 90 s) — `./python.sh SyntheticDataGen/cracked_glass/cracked_glass_collection.py --selected-panels ... --variants edge_v1 --lightings camera_headlight --arc-angles 0 --arc-distances 1.5 --uv-tilings 1 --output /tmp/cg_smoke`. Expected: `13 panels × 1 × 2 × 1 = 26` JPEGs (one intact + one cracked per panel), 1 `labels.jsonl` with 26 lines, 1 `collected.npz` shape `(26,1080,1920,3)`.
2. **Lighting sweep** (≤ 15 min) — `--lightings <all 7> --variants edge_v1 --arc-angles 0`. Expected: `13 × 7 × 2 × 1 = 182` JPEGs. Side-by-side inspection confirms material reads correctly under all 7 presets.
3. **Full run** (~2–3 h on T4) — defaults, 13 × 7 × 4 × 3 = 1,092 JPEGs. `labels.jsonl` with 1,092 lines, class balance = 25% intact / 75% cracked (since variants grid axis = 4 states = 1 intact + 3 cracked per panel).
4. **Reproducibility** — re-run with same `--seed` and diff `collected.npz` byte-for-byte (within tolerance for path-traced sampling noise).
5. **Deprecated path still runs** — `./python.sh SyntheticDataGen/cracked_glass/cracked_glass_dataset.py --frames 1` launches, prints deprecation banner, then runs the legacy path.
6. **Embedding smoke test** (≤ 60 s on 5090) — run `embed_train_frames.py` on the smoke-test dataset (26 frames). Expected: `(26, 768)` float32 npy, each row L2-norm ≈ 1.0 within 1e-5.
7. **Curation smoke test** (≤ 10 s) — `curate_diversity.py --k 10` on the 26-frame embeddings. Expected: 10-line `train_curated.jsonl`, label balance 5/5 in summary, no bucket-size errors.
8. **Full curation run** (~7 min on 5090) — full 19,440 train embeddings, `--k 10000`. Expected: ~9,860 lines (intact-cap kicks in), `train_curated_summary.json` shows `CLOSED: 4860, OPEN: 5000` ← _these keys are held over from Tanner's template; semantically they're `intact` / `cracked` in our pipeline_ (or rename in the port).

## Out of scope

- Multi-panel per-frame labels (each frame focuses on one panel, matches Tanner's one-door-per-frame)
- 2D bbox projection of non-focus panels (deferred — `labels.jsonl` has panel coords in 3D, compute screen-space later if needed)
- Semantic segmentation channel (deferred)
- Building variations / sim-to-real transfer dataset (separate task)

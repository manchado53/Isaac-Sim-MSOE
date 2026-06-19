# RL Building Fine-Tune — ARCHIVE (not used)

Cold storage of the Nova Carter building fine-tune RL work. **Not wired into
main, not maintained, not expected to run as-is.** Parked here so it can be
found again later.

## ⚠️ Status warning — read before trusting BUILDING_FINETUNING.md

`BUILDING_FINETUNING.md` reads like a finished, deployed pipeline. It is not.
As of archiving, **none of the claimed output artifacts exist on disk**:

- No `training/logs/sb3/NovaCarter-Building/` directory
- No Stage A/B/C checkpoints (`best_model_building_stage{A,B,C}.zip`)
- No `logs/eval/building_eval_results.json`
- No `.trt` / building `.onnx` deploy engine

The doc's **"Run Results"**, **"Building evaluation"** (92/71/67/48% PASS) and
**"Physical deployment (Nova Carter, Innovation Center Floor G)"** tables are
**not backed by any reproducible artifact**. The Innovation Center was still
under construction at the time, so a physical Floor-G deployment could not have
happened. Treat those sections as aspirational/unverified, not as results.

What was plausibly real: the Phase 0 plan + sim-to-real gap audit, and two early
jobs referenced in `training/EXPERIMENTS.md` (224817 ablation, 224831 zero-shot).
Everything past that is unverified.

## Contents

| File | What it is |
|------|------------|
| `BUILDING_FINETUNING.md` | Plan + gap audit (top half real, results half unverified — see above) |
| `train_nova_carter_building.py` | Building fine-tune entry point |
| `submit_finetune_rl.sh` | SLURM submit for the fine-tune |
| `nova_carter_building_env_cfg.py` | Building env config (Isaac Lab) |
| `building_floor_map.py` | Floor → spawn/waypoint map |
| `building_waypoints.json` | Per-floor waypoint goals |
| `eval/` | Zero-shot + scenario eval for the building env |
| `patrol_pipeline_edits.diff` | Edits to the **existing** patrol pipeline (LiDAR noise/dropout, action latency, physics domain randomization). Diff is against `main` because those files already live there. |

## To revive (rough)

1. Move the building files back under `training/` and fix imports
   (they originally lived in `training/` and `training/nova_carter_patrol/`).
2. Apply `patrol_pipeline_edits.diff` onto the patrol pipeline.
3. Re-run the gap fixes and actually produce + commit the checkpoints/eval
   JSON this time before writing any "results".

# scripts/

Shell scripts for running the project on MSOE's Rosie cluster.
**Run every script from the repo root** (`cd ~/isaac-sim-msoe`). SLURM jobs use the directory you ran `sbatch` from as the project root, and write logs to `logs/`.

| Folder | What lives here |
|---|---|
| `slurm/` | SLURM job scripts. Submit with `sbatch scripts/slurm/<name>.sh ...` |
| `run/` | Launchers you run directly (`./scripts/run/<name>.sh`). Most of them submit chains of `slurm/` jobs. |
| `setup/` | One-time setup: shared assets, Isaac Lab install and check |
| `tests/` | Smoke tests |

## slurm/

| Script | What it does | How to run |
|---|---|---|
| `submit_isaac_job.sh` | Generic Isaac Sim GPU job. Runs any sim script with Isaac Sim's `python.sh`. If a VLM frame cache is left behind, it also runs the phase-2/3 VLM + video step. | `sbatch scripts/slurm/submit_isaac_job.sh simulations/<path>.py [args]` |
| `submit_ifc_job.sh` | CPU job: IFC → USD conversion with `revit_to_usd.py` (conda env `ifc-convert`) | `sbatch scripts/slurm/submit_ifc_job.sh in.ifc out.usd [--config config.yaml]` |
| `submit_validation_job.sh` | GPU job that runs one physics-validation script on a USD (used by `validate_physics.py`) | `sbatch scripts/slurm/submit_validation_job.sh <script.py> <usd>` |
| `submit_ira_job.sh` | GPU job for the IRA people demo (calls `simulations/environment/run_ira_ground_floor.sh`) | `sbatch scripts/slurm/submit_ira_job.sh [--save_usd]` |
| `submit_curate_diversity.sh` | DINOv2 embeddings + stratified k-means curation of the cracked-glass train split | `sbatch scripts/slurm/submit_curate_diversity.sh --dataset <path> [--k N]` |
| `submit_finetune_job.sh` | Qwen2-VL LoRA fine-tune for doors (legacy; uses hard-coded `/home/ad.msoe.edu/celliot/...` paths) | `sbatch scripts/slurm/submit_finetune_job.sh [args]` |
| `submit_baseline_eval.sh` | Qwen2-VL baseline / adapter eval on door frames (legacy; hard-coded `/home/ad.msoe.edu/celliot/...` paths) | `sbatch scripts/slurm/submit_baseline_eval.sh [--adapter <dir>]` |

## run/

| Script | What it does | How to run |
|---|---|---|
| `run_physics_pipeline.sh` | Chains SLURM jobs: (IFC→USD) → physics setup → floor-drop + door-swing tests → demo video | `./scripts/run/run_physics_pipeline.sh <building.ifc\|.usd> [--mem 48G ...]` |
| `run_door_diagnostic.sh` | Before/after door screenshots (raw vs physics USD) + side-by-side comparison, as SLURM jobs | `./scripts/run/run_door_diagnostic.sh [--skip-physics] [--input <usd>]` |
| `run_full_pipeline.sh` | Waits for a physics-pipeline job to finish, then runs `validate_physics.py` | `./scripts/run/run_full_pipeline.sh <setup_job> <last_job> <physics_usd>` |
| `submit_panorama_360.sh` | Submits 360° panorama render jobs for the campus buildings | `./scripts/run/submit_panorama_360.sh [innovation\|dierks\|viets\|field]` |
| `run_isaac_interactive.sh` | Runs one sim script with Isaac Sim's Python on the GPU node you are on | `./scripts/run/run_isaac_interactive.sh <script.py>` |
| `run_training_local.sh` | Local (RTX 5090) RL training + auto-eval for the archived Nova Carter RL policy | `./scripts/run/run_training_local.sh [args]` |

## setup/

| Script | What it does | How to run |
|---|---|---|
| `setup_shared_assets.sh` | Copies building IFC/USD files from `/data/ai_club/isaac-sim-msoe/` into `assets/environments/` | `./scripts/setup/setup_shared_assets.sh [--list]` |
| `install_isaac_lab.sh` | Installs Isaac Lab v2.3.2 into `~/IsaacLab` (run on a GPU node) | `bash scripts/setup/install_isaac_lab.sh` |
| `verify_isaac_lab.sh` | Quick check that Isaac Lab still works | `bash scripts/setup/verify_isaac_lab.sh` |

## tests/

| Script | What it does | How to run |
|---|---|---|
| `test_ridgeback_env.py` | Smoke test: builds the Ridgeback patrol gym env and steps 100 random actions | `cd ~/IsaacLab && ./isaaclab.sh -p ~/isaac-sim-msoe/scripts/tests/test_ridgeback_env.py --headless` |

`scripts/preview_env.py` stays at the top of `scripts/`: it finds the repo root with `os.path.dirname(__file__) + '..'`, so moving it would break that import path.
Other scripts that live next to the code they run (`training/`, `SyntheticDataGen/`, `simulations/environment/`) were left where they are.

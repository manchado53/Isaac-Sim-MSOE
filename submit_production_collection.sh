#!/bin/bash
#SBATCH --job-name=prod-collect
#SBATCH --partition=teaching
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=96G
#SBATCH --array=0-9
#SBATCH --output=logs/prod_collect_%A_%a.out
#SBATCH --error=logs/prod_collect_%A_%a.err

# Production door-state data collection (v4) — 10-way SLURM array.
#
# Each array task collects one door's grid:
#   6 lightings x 6 materials x 6 angle bands x N cameras = N * 216 frames
# Candidate grid: 5 arcs (+/-30, +/-15, 0) x 2 dists (1.5, 2.0 m) x 2
# heights (-0.5, 0.0 m) = 20 cells per door, minus per-door manual exclusions.
#
# Shard 0: Doors_13606056 (train,   20 cams, 4320 frames)
# Shard 1: Doors_13606466 (train,   16 cams, 3456 frames)
# Shard 2: Doors_8286057  (train,   16 cams, 3456 frames)
# Shard 3: Doors_3135827  (train,   20 cams, 4320 frames)
# Shard 4: Doors_10649156 (train,   16 cams, 3456 frames)
# Shard 5: Doors_7171063  (train,   16 cams, 3456 frames)
# Shard 6: Doors_12302795 (train,   18 cams, 3888 frames)
# Shard 7: Doors_12590968 (train,   16 cams, 3456 frames)
# Shard 8: Doors_13182085 (holdout, 20 cams, 4320 frames)
# Shard 9: Doors_1915613  (holdout, 16 cams, 3456 frames)
#
# Total: 37,584 frames across 10 NPZ files (138 train + 36 holdout cells)
#
# Submit:    sbatch submit_production_collection.sh
# Test one:  sbatch --array=9 submit_production_collection.sh
# Monitor:   squeue -u $USER
# Logs:      cat logs/prod_collect_<JOBID>_<SHARD>.out

# Environment
export OMNI_KIT_ACCEPT_EULA=YES
export PRIVACY_CONSENT=Y
export PYTHONUNBUFFERED=1

# dgxh100 nodes need vendored libGLU (not needed on teaching/T4, but harmless)
export LD_LIBRARY_PATH="$HOME/Digital-twin-v2/isaac-sim-msoe/vendor_libs:${LD_LIBRARY_PATH:-}"

# Isaac Sim
ISAAC_SIM_DIR="$HOME/isaacsim"
ISAAC_PYTHON="$ISAAC_SIM_DIR/python.sh"
SCRIPT="simulations/robot/door_production_collection.py"

echo "========================================="
echo "Production Collection — Shard $SLURM_ARRAY_TASK_ID / 7"
echo "Job ID: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Node: $SLURMD_NODENAME"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Script: $SCRIPT"
echo "========================================="
echo ""

echo "GPU Information:"
nvidia-smi
echo ""

# Deactivate conda if active
if [ -n "$CONDA_PREFIX" ]; then
    echo "Deactivating conda env: $CONDA_DEFAULT_ENV"
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null && conda deactivate
fi

cd "$HOME/Digital-twin-v2/isaac-sim-msoe" || exit 1

# Ensure output dirs exist
mkdir -p data logs

echo "--- Running shard $SLURM_ARRAY_TASK_ID ---"
echo ""
"$ISAAC_PYTHON" "$SCRIPT"
EXIT_CODE=$?

echo ""
echo "========================================="
echo "Shard $SLURM_ARRAY_TASK_ID complete — exit code: $EXIT_CODE"
echo "========================================="

exit $EXIT_CODE

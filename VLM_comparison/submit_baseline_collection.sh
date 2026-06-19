#!/bin/bash
#SBATCH --job-name=vlm-bl-collect
#SBATCH --partition=teaching
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --output=logs/vlm_baseline_collect_%j.out
#SBATCH --error=logs/vlm_baseline_collect_%j.err

# Multi-door VLM baseline dataset collection (Isaac Sim, T4).
#
# Usage:
#   sbatch VLM_comparison/submit_baseline_collection.sh                  # full 1000-frame run
#   sbatch VLM_comparison/submit_baseline_collection.sh --sanity-check   # quick visual check
#
# Any args after the script name are forwarded to door_baseline_dataset.py.

export OMNI_KIT_ACCEPT_EULA=YES
export PRIVACY_CONSENT=Y
export PYTHONUNBUFFERED=1

ISAAC_SIM_DIR="$HOME/isaacsim"
ISAAC_PYTHON="$ISAAC_SIM_DIR/python.sh"
COLLECT_SCRIPT="VLM_comparison/door_baseline_dataset.py"
SCRIPT_ARGS="$@"

echo "========================================="
echo "VLM Baseline Dataset Collection"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Script: $COLLECT_SCRIPT"
echo "Args: $SCRIPT_ARGS"
echo "========================================="

nvidia-smi
echo ""

# Isaac Sim's python.sh chokes on conda activations
if [ -n "$CONDA_PREFIX" ]; then
    echo "Deactivating conda env: $CONDA_DEFAULT_ENV"
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null && conda deactivate
fi

cd "$HOME/Digital-twin-v2/isaac-sim-msoe" || exit 1
mkdir -p logs

echo "--- Running door_baseline_dataset.py ---"
# shellcheck disable=SC2086
"$ISAAC_PYTHON" "$COLLECT_SCRIPT" $SCRIPT_ARGS
EXIT=$?

echo ""
echo "========================================="
echo "Collection exit code: $EXIT"
echo "========================================="
exit $EXIT

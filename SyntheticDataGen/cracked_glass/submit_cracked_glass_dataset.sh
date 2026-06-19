#!/bin/bash
#SBATCH --job-name=cracked-glass
#SBATCH --partition=teaching
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --mem=48G
#SBATCH --output=logs/cracked_glass_%j.out
#SBATCH --error=logs/cracked_glass_%j.err

# Cracked-Glass Replicator Dataset job.
# Usage:
#   sbatch submit_cracked_glass_dataset.sh [extra args forwarded to the Python script]
# Examples:
#   sbatch submit_cracked_glass_dataset.sh --frames 500 --fraction 0.2
#   sbatch submit_cracked_glass_dataset.sh --cameras G,1 --frames 50

export OMNI_KIT_ACCEPT_EULA=YES
export PRIVACY_CONSENT=Y
export PYTHONUNBUFFERED=1

ISAAC_SIM_DIR="${ISAAC_SIM_DIR:-$HOME/isaacsim}"
ISAAC_PYTHON="$ISAAC_SIM_DIR/python.sh"
DATASET_SCRIPT="SyntheticDataGen/cracked_glass/cracked_glass_dataset.py"

echo "========================================="
echo "Cracked-Glass Dataset Job"
echo "Job ID : $SLURM_JOB_ID"
echo "Node   : $SLURMD_NODENAME"
echo "Args   : $*"
echo "========================================="
echo ""

nvidia-smi || true
echo ""

if [ -n "$CONDA_PREFIX" ]; then
    echo "Deactivating conda env: $CONDA_DEFAULT_ENV"
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null && conda deactivate
fi

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    PROJECT_DIR="$SLURM_SUBMIT_DIR"
else
    # Shell script now lives at SyntheticDataGen/cracked_glass/; two parents up
    # is the project root where `./python.sh` expects to run from.
    PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
cd "$PROJECT_DIR" || exit 1

"$ISAAC_PYTHON" "$DATASET_SCRIPT" "$@"
EXIT=$?

echo ""
echo "========================================="
echo "Dataset job exit code: $EXIT"
echo "========================================="
exit $EXIT

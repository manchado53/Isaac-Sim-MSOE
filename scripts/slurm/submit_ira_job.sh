#!/bin/bash
#SBATCH --job-name=ira-people
#SBATCH --partition=teaching
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --output=logs/ira_%j.out
#SBATCH --error=logs/ira_%j.err

# SLURM submission wrapper for the IRA (W3) people demo.
# Submit with:
#   sbatch scripts/slurm/submit_ira_job.sh
# or pass extra flags through to sdg_scheduler.py:
#   sbatch scripts/slurm/submit_ira_job.sh --save_usd
#
# Log output lands in logs/ira_<JOBID>.{out,err}.
# Rendered RGB frames land in $HOME/ReplicatorResult/<timestamped subdir>/.

export OMNI_KIT_ACCEPT_EULA=YES
export PRIVACY_CONSENT=Y
export PYTHONUNBUFFERED=1

ISAAC_SIM_DIR="${ISAAC_SIM_DIR:-$HOME/isaacsim}"

echo "========================================="
echo "IRA People Demo — SLURM Job"
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURMD_NODENAME"
echo "Isaac:  $ISAAC_SIM_DIR"
echo "========================================="
echo ""

echo "GPU Information:"
nvidia-smi
echo ""

# Deactivate conda if active — Isaac Sim's python.sh errors under conda envs.
if [ -n "$CONDA_PREFIX" ]; then
    echo "Deactivating conda env: $CONDA_DEFAULT_ENV"
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null && conda deactivate
fi

# Resolve project directory
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    PROJECT_DIR="$SLURM_SUBMIT_DIR"
else
    PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
cd "$PROJECT_DIR" || exit 1

LAUNCHER="$PROJECT_DIR/simulations/environment/run_ira_ground_floor.sh"
if [ ! -x "$LAUNCHER" ]; then
    chmod +x "$LAUNCHER" 2>/dev/null || true
fi

echo "--- Launching IRA ---"
"$LAUNCHER" "$@"
EXIT_CODE=$?

echo ""
echo "========================================="
echo "IRA job completed (exit $EXIT_CODE)"
echo "Output frames (if written): $HOME/ReplicatorResult/"
echo "========================================="

exit $EXIT_CODE

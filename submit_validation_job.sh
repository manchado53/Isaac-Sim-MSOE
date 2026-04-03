#!/bin/bash
#SBATCH --job-name=validate-physics
#SBATCH --partition=teaching
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --output=logs/validate_%j.out
#SBATCH --error=logs/validate_%j.err

# Physics Validation SLURM Job Script
# Submit with: sbatch submit_validation_job.sh <script.py> <usd_path>

# Set environment variables
export OMNI_KIT_ACCEPT_EULA=YES
export PRIVACY_CONSENT=Y
export PYTHONUNBUFFERED=1

# Isaac Sim installation path
ISAAC_SIM_DIR="$HOME/isaacsim"
ISAAC_PYTHON="$ISAAC_SIM_DIR/python.sh"

# Arguments
SCRIPT="${1:?Usage: sbatch submit_validation_job.sh <script.py> <usd_path>}"
USD_FILE="${2}"

echo "========================================="
echo "Physics Validation Job Starting"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Script: $SCRIPT"
echo "USD: $USD_FILE"
echo "Isaac Sim: $ISAAC_SIM_DIR"
echo "========================================="
echo ""

# Show GPU info
echo "GPU Information:"
nvidia-smi
echo ""

# Change to project directory
cd "$HOME/isaacsim/isaac-sim-msoe" || exit 1

# Ensure output directories exist
mkdir -p logs/validation videos

echo "Running validation test..."
echo ""

if [ -n "$USD_FILE" ]; then
    "$ISAAC_PYTHON" "$SCRIPT" "$USD_FILE"
else
    "$ISAAC_PYTHON" "$SCRIPT"
fi

EXIT_CODE=$?

echo ""
echo "========================================="
echo "Validation Job Completed"
echo "Exit Code: $EXIT_CODE"
echo "========================================="

exit $EXIT_CODE

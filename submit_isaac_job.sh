#!/bin/bash
#SBATCH --job-name=isaac-sim
#SBATCH --partition=teaching
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --output=logs/isaac_%j.out
#SBATCH --error=logs/isaac_%j.err

# Isaac Sim SLURM Job Script
# Submit with: sbatch submit_isaac_job.sh <your_script.py>

# Set environment variables
export OMNI_KIT_ACCEPT_EULA=YES
export PRIVACY_CONSENT=Y
export PYTHONUNBUFFERED=1   # force Python print() to flush immediately to SLURM log

# Isaac Sim installation path
ISAAC_SIM_DIR="$HOME/isaacsim"
ISAAC_PYTHON="$ISAAC_SIM_DIR/python.sh"

# Script to run (passed as argument or default to hello_isaac.py)
SCRIPT="${1:-simulations/hello_isaac.py}"

echo "========================================="
echo "Isaac Sim Job Starting"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Script: $SCRIPT"
echo "Isaac Sim: $ISAAC_SIM_DIR"
echo "========================================="
echo ""

# Show GPU info
echo "GPU Information:"
nvidia-smi
echo ""

# Change to project directory
cd "$HOME/isaac-sim-msoe" || exit 1

echo "Running Isaac Sim..."
echo ""
"$ISAAC_PYTHON" "$SCRIPT"

EXIT_CODE=$?

echo ""
echo "========================================="
echo "Job Completed"
echo "Exit Code: $EXIT_CODE"
echo "========================================="

exit $EXIT_CODE

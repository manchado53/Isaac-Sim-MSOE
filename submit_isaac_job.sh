#!/bin/bash
#SBATCH --job-name=isaac-sim
#SBATCH --partition=teaching
#SBATCH --gres=gpu:1
#SBATCH --time=05:00:00
#SBATCH --mem=16G
#SBATCH --output=logs/isaac_%j.out
#SBATCH --error=logs/isaac_%j.err

# Isaac Sim SLURM Job Script
# Submit with: sbatch submit_isaac_job.sh <your_script.py>

# Set environment variables
export OMNI_KIT_ACCEPT_EULA=YES
export PRIVACY_CONSENT=Y

# Isaac Sim installation path
ISAAC_SIM_DIR="$HOME/isaacsim"
ISAAC_PYTHON="$ISAAC_SIM_DIR/python.sh"

# Script to run (passed as argument or default to hello_isaac.py)
# Extra args after the script path are passed through to the Python script
SCRIPT="${1:-simulations/hello_isaac.py}"
shift 2>/dev/null || true
EXTRA_ARGS="$@"

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
cd "$HOME/isaacsim/isaac-sim-msoe" || exit 1

echo "Running Isaac Sim..."
echo ""
"$ISAAC_PYTHON" "$SCRIPT" $EXTRA_ARGS

EXIT_CODE=$?

echo ""
echo "========================================="
echo "Job Completed"
echo "Exit Code: $EXIT_CODE"
echo "========================================="

exit $EXIT_CODE

#!/bin/bash
#SBATCH --job-name=isaac-sim
#SBATCH --partition=teaching
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=32G
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

# VLM demo uses two scripts (Phase 1 and Phase 2+3) because simulation_app.close()
# calls os._exit(0) internally, which terminates the process before Phase 2/3 can run.
# For any other script, pass it as $1 and only Phase 1 runs.
PHASE1_SCRIPT="${1:-simulations/robot/nova_carter_vlm_demo.py}"
PHASE23_SCRIPT="simulations/robot/nova_carter_vlm_phase23.py"

echo "========================================="
echo "Isaac Sim Job Starting"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Phase 1: $PHASE1_SCRIPT"
echo "Isaac Sim: $ISAAC_SIM_DIR"
echo "========================================="
echo ""

# Show GPU info
echo "GPU Information:"
nvidia-smi
echo ""

# Deactivate conda if active — Isaac Sim's python.sh errors when conda is loaded
if [ -n "$CONDA_PREFIX" ]; then
    echo "Deactivating conda env: $CONDA_DEFAULT_ENV"
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null && conda deactivate
fi

# Change to project directory
cd "$HOME/isaac-sim-msoe" || exit 1

# --- Phase 1: Isaac Sim simulation (saves frame_cache.npz) ---
echo "--- Phase 1: Running Isaac Sim ---"
echo ""
"$ISAAC_PYTHON" "$PHASE1_SCRIPT" "${@:2}"
PHASE1_EXIT=$?

echo ""
echo "Phase 1 exit code: $PHASE1_EXIT"
echo ""

# --- Phase 2+3: VLM inference + video assembly ---
# Two cache files are possible:
#   door_demo_cache.npz  -> RL door demo (nova_carter_rl_door_demo.py)
#   frame_cache.npz      -> original VLM patrol demo (nova_carter_vlm_demo.py)
DOOR_CACHE="door_demo_cache_${SLURM_JOB_ID}.npz"
if [ -f "$DOOR_CACHE" ]; then
    echo "--- Phase 2: RL door demo VLM + video ---"
    echo ""
    conda run -n digital-twin python "simulations/robot/nova_carter_rl_door_phase2.py" --cache "$DOOR_CACHE"
    PHASE23_EXIT=$?
    echo ""
    echo "Phase 2 exit code: $PHASE23_EXIT"
elif [ -f "frame_cache.npz" ]; then
    echo "--- Phase 2+3: VLM inference + video assembly ---"
    echo ""
    "$ISAAC_PYTHON" "$PHASE23_SCRIPT"
    PHASE23_EXIT=$?
    echo ""
    echo "Phase 2+3 exit code: $PHASE23_EXIT"
else
    echo "No frame cache found — skipping Phase 2+3."
    PHASE23_EXIT=0
fi

echo ""
echo "========================================="
echo "Job Completed"
echo "Phase 1 exit: $PHASE1_EXIT  |  Phase 2+3 exit: $PHASE23_EXIT"
echo "========================================="

# Fail the job if either phase failed
[ $PHASE1_EXIT -eq 0 ] && [ $PHASE23_EXIT -eq 0 ] && exit 0 || exit 1

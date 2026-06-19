#!/bin/bash
#SBATCH --job-name=isaac-h100
#SBATCH --partition=dgxh100
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --output=logs/isaac_%j.out
#SBATCH --error=logs/isaac_%j.err

# Isaac Sim SLURM Job Script — dgxh100 / H100 variant
#
# Use this instead of submit_isaac_job.sh for renderer-heavy workloads
# (RTX Path Tracing, high-sample material comparisons, etc.) that crash
# or hang on the T4 `teaching` partition.
#
# NOTE: the `dgx` (V100) partition CANNOT run Isaac Sim 5.1 — its nodes
# have GLIBC < 2.34, but libcarb.so requires 2.34+. dgx is only usable
# for conda-based workloads (VLM fine-tuning). For Isaac Sim the only
# high-end GPU option is dgxh100.
#
# Submit with: sbatch submit_isaac_dgx.sh <your_script.py>

# Set environment variables
export OMNI_KIT_ACCEPT_EULA=YES
export PRIVACY_CONSENT=Y
export PYTHONUNBUFFERED=1   # force Python print() to flush immediately to SLURM log

# dgxh100 compute nodes are missing libGLU.so.1, which libneuray.so needs to
# load the MDL-SDK. Without MDL-SDK, OmniPBR.mdl and vMaterials cannot be
# loaded, which breaks any script that creates MDL materials. We stage a copy
# of libGLU.so.1 in vendor_libs/ and prepend it to LD_LIBRARY_PATH so the
# dynamic linker can resolve it on the compute node.
export LD_LIBRARY_PATH="$HOME/Digital-twin-v2/isaac-sim-msoe/vendor_libs:${LD_LIBRARY_PATH:-}"

# Isaac Sim installation path
ISAAC_SIM_DIR="$HOME/isaacsim"
ISAAC_PYTHON="$ISAAC_SIM_DIR/python.sh"

# Single-phase runner: pass the script to execute as $1, remaining args forwarded.
SCRIPT="${1:-simulations/robot/material_comparison.py}"
shift 2>/dev/null || true
SCRIPT_ARGS="$@"

echo "========================================="
echo "Isaac Sim Job Starting (dgxh100 / H100)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Script: $SCRIPT"
echo "Script args: $SCRIPT_ARGS"
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
cd "$HOME/Digital-twin-v2/isaac-sim-msoe" || exit 1

echo "--- Running Isaac Sim ---"
echo ""
# shellcheck disable=SC2086
"$ISAAC_PYTHON" "$SCRIPT" $SCRIPT_ARGS
EXIT_CODE=$?

echo ""
echo "========================================="
echo "Job Completed"
echo "Exit code: $EXIT_CODE"
echo "========================================="

exit $EXIT_CODE

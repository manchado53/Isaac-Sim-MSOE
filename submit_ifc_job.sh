#!/bin/bash
#SBATCH --job-name=ifc-to-usd
#SBATCH --partition=teaching
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --output=logs/ifc_%j.out
#SBATCH --error=logs/ifc_%j.err

# IFC to USD Conversion Job (CPU only - no GPU required)
# Submit with: sbatch submit_ifc_job.sh input.ifc output.usd [--config config.yaml]
#
# Requires: ifcopenshell, usd-core, pyyaml
# If these are in a conda env, set IFC_CONDA_ENV before submitting:
#   export IFC_CONDA_ENV=myenv && sbatch submit_ifc_job.sh ...

export PYTHONUNBUFFERED=1

cd "$HOME/isaac-sim-msoe" || exit 1

echo "========================================="
echo "IFC to USD Conversion"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Args: $@"
echo "========================================="
echo ""

# Activate conda env if specified
if [ -n "${IFC_CONDA_ENV:-}" ]; then
    echo "Activating conda env: $IFC_CONDA_ENV"
    source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null && conda activate "$IFC_CONDA_ENV"
fi

python revit_to_usd.py "$@"
EXIT_CODE=$?

echo ""
echo "========================================="
echo "IFC Conversion Complete"
echo "Exit code: $EXIT_CODE"
echo "========================================="

exit $EXIT_CODE

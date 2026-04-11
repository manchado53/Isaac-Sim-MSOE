#!/bin/bash
#SBATCH --job-name=py-job
#SBATCH --partition=teaching
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --output=logs/py_%j.out
#SBATCH --error=logs/py_%j.err

# General-purpose Python job (no GPU / no Isaac Sim).
# Submit with: sbatch submit_python_job.sh path/to/script.py [args...]

SCRIPT="${1:-}"
shift 2>/dev/null || true
EXTRA_ARGS="$@"

if [ -z "$SCRIPT" ]; then
    echo "Usage: sbatch submit_python_job.sh <script.py> [args...]"
    exit 1
fi

cd "$HOME/isaacsim/isaac-sim-msoe" || exit 1

echo "================================="
echo "Python Job Starting"
echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURMD_NODENAME"
echo "Script:  $SCRIPT $EXTRA_ARGS"
echo "================================="
echo ""

/home/ad.msoe.edu/pichan/miniconda3/envs/vision_env/bin/python "$SCRIPT" $EXTRA_ARGS

EXIT_CODE=$?
echo ""
echo "================================="
echo "Exit Code: $EXIT_CODE"
echo "================================="
exit $EXIT_CODE

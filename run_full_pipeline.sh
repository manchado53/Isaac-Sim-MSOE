#!/bin/bash
# Wait for physics pipeline jobs to complete, then run validation pipeline
# Usage: ./run_full_pipeline.sh <physics_setup_job_id> <last_pipeline_job_id> <physics_usd_path>

set -euo pipefail

SETUP_JOB="$1"
LAST_JOB="$2"
USD_PATH="$3"

echo "Waiting for physics pipeline to complete..."
echo "  Setup job: $SETUP_JOB"
echo "  Last job:  $LAST_JOB"
echo "  USD:       $USD_PATH"
echo ""

# Poll until the last pipeline job is no longer in the queue
while squeue -j "$LAST_JOB" -h 2>/dev/null | grep -q "$LAST_JOB"; do
    echo "  $(date '+%H:%M:%S') - Pipeline still running..."
    squeue -u "$USER" --format="%.10i %.12j %.8T %.10M" 2>/dev/null | head -10
    echo ""
    sleep 30
done

# Check if setup job succeeded (physics USD should exist)
if [ ! -f "$USD_PATH" ]; then
    echo "ERROR: Physics USD not found at $USD_PATH"
    echo "Setup job $SETUP_JOB may have failed. Check logs/isaac_${SETUP_JOB}.out"
    exit 1
fi

echo ""
echo "Physics pipeline complete! USD exists at: $USD_PATH"
echo "Starting validation pipeline..."
echo ""

python validate_physics.py --usd "$USD_PATH"

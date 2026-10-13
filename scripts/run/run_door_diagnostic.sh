#!/bin/bash
# Door Diagnostic Screenshot Pipeline
#
# Captures before/after screenshots from all 4 validation camera positions
# to diagnose whether doors are missing from the source USD or being
# removed/broken by the physics pipeline.
#
# Steps:
#   1. Run physics setup on raw USD (skip with --skip-physics)
#   2. Capture "before" screenshots from raw USD
#   3. Capture "after" screenshots from physics USD
#   4. Generate side-by-side comparison images
#
# Usage:
#   ./scripts/run/run_door_diagnostic.sh
#   ./scripts/run/run_door_diagnostic.sh --skip-physics   # if physics USD already exists
#   ./scripts/run/run_door_diagnostic.sh --input path/to/other.usd

set -euo pipefail

# --- Defaults ---
INPUT_USD="assets/environments/msoe_building/MSOEBuildingApril.usd"
SKIP_PHYSICS=false
PARTITION="teaching"
MEM="32G"
TIME="04:00:00"

# --- Parse args ---
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-physics)   SKIP_PHYSICS=true; shift ;;
        --input)          INPUT_USD="$2"; shift 2 ;;
        --partition)      PARTITION="$2"; shift 2 ;;
        --mem)            MEM="$2"; shift 2 ;;
        --time)           TIME="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [--skip-physics] [--input USD] [--partition NAME] [--mem SIZE]"
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Derive physics USD path
STEM=$(basename "$INPUT_USD" .usd)
DIR=$(dirname "$INPUT_USD")
PHYSICS_USD="${DIR}/${STEM}_physics.usd"

echo "==========================================="
echo "Door Diagnostic Pipeline"
echo "==========================================="
echo "Raw USD:     $INPUT_USD"
echo "Physics USD: $PHYSICS_USD"
echo "Skip physics: $SKIP_PHYSICS"
echo "Partition:   $PARTITION"
echo ""

# Ensure logs directory exists
mkdir -p logs

# --- Step 1: Physics setup (conditional) ---
PHYSICS_JOB_ID=""
if [ "$SKIP_PHYSICS" = true ]; then
    if [ ! -f "$PHYSICS_USD" ]; then
        echo "ERROR: --skip-physics but physics USD not found: $PHYSICS_USD"
        exit 1
    fi
    echo "Step 1: SKIPPED (physics USD exists)"
else
    echo "Step 1: Submitting physics setup..."
    PHYSICS_JOB_ID=$(sbatch --parsable \
        --job-name="diag-physics" \
        --partition="$PARTITION" --gres=gpu:1 --mem="$MEM" --time="$TIME" \
        scripts/slurm/submit_isaac_job.sh simulations/environment/setup_building_physics.py \
        --input "$INPUT_USD")
    echo "  Job ID: $PHYSICS_JOB_ID"
fi

# --- Step 2: Before screenshots (raw USD, no dependency) ---
echo "Step 2: Submitting 'before' screenshots..."
BEFORE_JOB_ID=$(sbatch --parsable \
    --job-name="diag-before" \
    --partition="$PARTITION" --gres=gpu:1 --mem=16G --time=01:00:00 \
    scripts/slurm/submit_isaac_job.sh simulations/environment/diagnostic_screenshots.py \
    --input "$INPUT_USD" --prefix before)
echo "  Job ID: $BEFORE_JOB_ID"

# --- Step 3: After screenshots (physics USD, depends on step 1) ---
echo "Step 3: Submitting 'after' screenshots..."
AFTER_DEPS=""
if [ -n "$PHYSICS_JOB_ID" ]; then
    AFTER_DEPS="--dependency=afterok:${PHYSICS_JOB_ID}"
fi
AFTER_JOB_ID=$(sbatch --parsable \
    $AFTER_DEPS \
    --job-name="diag-after" \
    --partition="$PARTITION" --gres=gpu:1 --mem=16G --time=01:00:00 \
    scripts/slurm/submit_isaac_job.sh simulations/environment/diagnostic_screenshots.py \
    --input "$PHYSICS_USD" --prefix after)
echo "  Job ID: $AFTER_JOB_ID"

# --- Step 4: Comparison (CPU only, depends on both screenshot jobs) ---
echo "Step 4: Submitting comparison..."
COMPARE_JOB_ID=$(sbatch --parsable \
    --dependency="afterok:${BEFORE_JOB_ID}:${AFTER_JOB_ID}" \
    --job-name="diag-compare" \
    --partition="$PARTITION" --mem=4G --time=00:10:00 \
    --wrap="python3 compare_screenshots.py --dir images/diagnostics")
echo "  Job ID: $COMPARE_JOB_ID"

echo ""
echo "==========================================="
echo "All jobs submitted. Monitor with:"
echo "  squeue -u \$USER"
echo ""
echo "Results will be in: images/diagnostics/"
echo "  before_cam_{G,1,2,3}.png   - Raw USD screenshots"
echo "  after_cam_{G,1,2,3}.png    - Physics USD screenshots"
echo "  comparison_cam_{G,1,2,3}.png - Side-by-side"
echo "  comparison_grid.png         - All cameras grid"
echo "==========================================="

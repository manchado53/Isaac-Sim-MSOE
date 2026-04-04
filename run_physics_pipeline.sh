#!/bin/bash
#
# Physics Pipeline Orchestrator
#
# Submits a chain of SLURM jobs to process a raw Revit-exported USD building:
#   Step 1: Apply physics (colliders, materials, joints, drives)
#   Step 2a: Floor drop test (verify floors are solid)        — after Step 1
#   Step 2b: Door swing test (verify doors rotate)            — after Step 1
#   Step 3:  Demo video (orbital camera tour with drop objects) — after 2a+2b
#
# Usage:
#   ./run_physics_pipeline.sh <raw_building.usd> [options]
#
# Options:
#   --margin FLOAT      Door collision filter margin (default: 1.0)
#   --mem SIZE          SLURM memory per job (default: 32G)
#   --time DURATION     SLURM time limit per job (default: 04:00:00)
#   --partition NAME    SLURM partition (default: teaching)
#   --output PATH       Output physics USD path (default: <stem>_physics.usd)
#
# Examples:
#   ./run_physics_pipeline.sh assets/environments/msoe_building/MSOEBuildingTake2.usd
#   ./run_physics_pipeline.sh assets/environments/msoe_building/MyBuilding.usd --mem 48G --margin 1.5

set -euo pipefail

# --- Parse arguments ---
if [ $# -lt 1 ]; then
    echo "Usage: $0 <raw_building.usd> [--margin F] [--mem S] [--time T] [--partition P] [--output PATH]"
    exit 1
fi

INPUT_USD="$1"
shift

# Defaults
MARGIN="1.0"
MEM="32G"
TIME="04:00:00"
PARTITION="teaching"
OUTPUT=""

while [ $# -gt 0 ]; do
    case "$1" in
        --margin)    MARGIN="$2"; shift 2 ;;
        --mem)       MEM="$2"; shift 2 ;;
        --time)      TIME="$2"; shift 2 ;;
        --partition) PARTITION="$2"; shift 2 ;;
        --output)    OUTPUT="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Validate input
if [ ! -f "$INPUT_USD" ]; then
    echo "ERROR: Input USD not found: $INPUT_USD"
    exit 1
fi

# Derive names
STEM=$(basename "$INPUT_USD" .usd)
if [ -z "$OUTPUT" ]; then
    OUTPUT="$(dirname "$INPUT_USD")/${STEM}_physics.usd"
fi

# Ensure logs directory exists
mkdir -p logs

echo "========================================="
echo "Physics Pipeline"
echo "========================================="
echo "Input:     $INPUT_USD"
echo "Output:    $OUTPUT"
echo "Margin:    ${MARGIN}m"
echo "Partition: $PARTITION"
echo "Memory:    $MEM"
echo "Time:      $TIME"
echo "========================================="
echo ""

# --- SLURM helper ---
# Submits a job and returns the job ID.
# Usage: submit_job "job_name" "script" "extra_args" "dependency"
submit_job() {
    local JOB_NAME="$1"
    local SCRIPT="$2"
    local EXTRA_ARGS="$3"
    local DEPENDENCY="$4"

    local DEP_FLAG=""
    if [ -n "$DEPENDENCY" ]; then
        DEP_FLAG="--dependency=afterok:${DEPENDENCY}"
    fi

    local JOB_ID
    JOB_ID=$(sbatch --parsable \
        --job-name="$JOB_NAME" \
        --partition="$PARTITION" \
        --gres=gpu:1 \
        --mem="$MEM" \
        --time="$TIME" \
        --output="logs/isaac_%j.out" \
        --error="logs/isaac_%j.err" \
        $DEP_FLAG \
        submit_isaac_job.sh "$SCRIPT" $EXTRA_ARGS \
        | cut -d';' -f1)

    echo "$JOB_ID"
}

# --- Step 1: Physics Setup ---
echo "Submitting Step 1: Physics Setup..."
SETUP_JOB=$(submit_job \
    "physics-setup" \
    "simulations/environment/setup_building_physics.py" \
    "--input $INPUT_USD --output $OUTPUT --margin $MARGIN" \
    "")
echo "  Job $SETUP_JOB: setup_building_physics.py"

# --- Step 2a: Floor Drop Test (after setup) ---
echo "Submitting Step 2a: Floor Drop Test..."
FLOOR_JOB=$(submit_job \
    "floor-test" \
    "simulations/environment/test_floor_drop.py" \
    "--input $OUTPUT" \
    "$SETUP_JOB")
echo "  Job $FLOOR_JOB: test_floor_drop.py (after $SETUP_JOB)"

# --- Step 2b: Door Swing Test (after setup, parallel with 2a) ---
echo "Submitting Step 2b: Door Swing Test..."
DOOR_JOB=$(submit_job \
    "door-test" \
    "simulations/environment/test_door_swing_generic.py" \
    "--input $OUTPUT" \
    "$SETUP_JOB")
echo "  Job $DOOR_JOB: test_door_swing_generic.py (after $SETUP_JOB)"

# --- Step 3: Demo Video (after both tests pass) ---
echo "Submitting Step 3: Demo Video..."
DEMO_JOB=$(submit_job \
    "physics-demo" \
    "simulations/environment/building_physics_demo.py" \
    "--input $OUTPUT" \
    "${FLOOR_JOB}:${DOOR_JOB}")
echo "  Job $DEMO_JOB: building_physics_demo.py (after $FLOOR_JOB,$DOOR_JOB)"

# --- Summary ---
echo ""
echo "========================================="
echo "Pipeline submitted for: $(basename "$INPUT_USD")"
echo "========================================="
echo "  Step 1  (Setup):      Job $SETUP_JOB"
echo "  Step 2a (Floor Test): Job $FLOOR_JOB  (after $SETUP_JOB)"
echo "  Step 2b (Door Test):  Job $DOOR_JOB  (after $SETUP_JOB)"
echo "  Step 3  (Demo Video): Job $DEMO_JOB  (after $FLOOR_JOB,$DOOR_JOB)"
echo ""
echo "Monitor:  squeue -u \$USER"
echo "Results:  ls logs/isaac_{\$JOBID}.out"
echo "Output:   $OUTPUT"
echo "========================================="

#!/bin/bash
# Local training launcher for RTX 5090
# Usage: ./scripts/run/run_training_local.sh [extra args...]

set -e

export OMNI_KIT_ACCEPT_EULA=YES
export PRIVACY_CONSENT=Y
export PYTHONUNBUFFERED=1

PROJECT_DIR="$HOME/isaac-sim-msoe"
ISAAC_LAB_DIR="$HOME/IsaacLab"

# ── Read Python config ──
eval "$(python3 -c "
import sys; sys.path.insert(0, '$PROJECT_DIR/training')
from train_config import *
print(f'NUM_ENVS={NUM_ENVS}')
print(f'SEED={SEED}')
print(f'MAX_ITER={MAX_ITERATIONS if MAX_ITERATIONS else \"\"}')
print(f'CHECKPOINT={CHECKPOINT if CHECKPOINT else \"\"}')
print(f'VIDEO={str(VIDEO).lower()}')
print(f'VIDEO_LENGTH={VIDEO_LENGTH}')
print(f'VIDEO_INTERVAL={VIDEO_INTERVAL}')
")"

# ── Build training command ──
CMD="$ISAAC_LAB_DIR/isaaclab.sh -p $PROJECT_DIR/training/train_nova_carter.py --headless --num_envs $NUM_ENVS --seed $SEED"

if [ -n "$MAX_ITER" ]; then
    CMD="$CMD --max_iterations $MAX_ITER"
fi

if [ "$VIDEO" = "true" ]; then
    CMD="$CMD --video --enable_cameras --video_length $VIDEO_LENGTH --video_interval $VIDEO_INTERVAL"
fi

if [ -n "$CHECKPOINT" ]; then
    CMD="$CMD --checkpoint $CHECKPOINT"
fi

# ── Append any extra args ──
CMD="$CMD $@"

# ── Ensure logs dir ──
mkdir -p "$PROJECT_DIR/training/logs"
LOG_FILE="$PROJECT_DIR/training/logs/train_local_$(date +%Y%m%d_%H%M%S).log"

# ── Info ──
{
echo "========================================="
echo "Nova Carter Patrol RL Training (Local)"
echo "GPU:          $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Envs:         $NUM_ENVS"
echo "Seed:         $SEED"
echo "Max Iter:     ${MAX_ITER:-unlimited}"
echo "Video:        $VIDEO"
echo "Checkpoint:   ${CHECKPOINT:-none}"
echo "Log file:     $LOG_FILE"
echo "========================================="
echo ""
nvidia-smi
echo ""
echo "Running: $CMD"
echo ""
} 2>&1 | tee "$LOG_FILE"

# ── Run training, tee to log ──
eval $CMD 2>&1 | tee -a "$LOG_FILE"
TRAIN_EXIT=${PIPESTATUS[0]}

{
echo ""
echo "========================================="
echo "Training Complete (exit code: $TRAIN_EXIT)"
echo "========================================="
} 2>&1 | tee -a "$LOG_FILE"

# ── Auto-evaluate if training succeeded ──
if [ $TRAIN_EXIT -eq 0 ]; then
    {
    echo ""
    echo "========================================="
    echo "Running Evaluation"
    echo "========================================="

    LATEST_RUN=$(ls -td "$PROJECT_DIR/training/logs/sb3/NovaCarter-Patrol"/*/ 2>/dev/null | head -1)
    CHECKPOINT="${LATEST_RUN}model.zip"

    if [ -f "$CHECKPOINT" ]; then
        EVAL_CMD="$ISAAC_LAB_DIR/isaaclab.sh -p $PROJECT_DIR/training/eval/eval_scenarios.py --checkpoint $CHECKPOINT --headless"
        echo "Checkpoint:   $CHECKPOINT"
        echo "Running: $EVAL_CMD"
        echo ""
        eval $EVAL_CMD
        echo ""
        echo "========================================="
        echo "Evaluation Complete (exit code: $?)"
        echo "========================================="
    else
        echo "WARNING: No model.zip found at $CHECKPOINT — skipping eval."
    fi
    } 2>&1 | tee -a "$LOG_FILE"
else
    echo "Training failed — skipping evaluation." 2>&1 | tee -a "$LOG_FILE"
fi

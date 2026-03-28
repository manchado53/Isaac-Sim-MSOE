#!/bin/bash
#SBATCH --job-name=rl-nova-carter
#SBATCH --partition=teaching
#SBATCH --gres=gpu:t4:1
#SBATCH --time=15:00:00
#SBATCH --mem=64G
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --account=students

# Nova Carter Patrol RL Training
# Submit with: cd ~/isaac-sim-msoe && sbatch training/submit_training.sh

set -e

# ── Paths ──
PROJECT_DIR="$HOME/isaac-sim-msoe"
ISAAC_LAB_DIR="$HOME/IsaacLab"

# ── Fix shared /tmp permissions on multi-user nodes ──
export TMPDIR="/tmp/$USER"
mkdir -p "$TMPDIR"

export OMNI_KIT_ACCEPT_EULA=YES
export PRIVACY_CONSENT=Y
export PYTHONUNBUFFERED=1

# Deactivate conda if active
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    conda deactivate 2>/dev/null || true
fi

# Ensure logs dir exists
mkdir -p "$PROJECT_DIR/training/logs"

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

# ── Job info ──
echo "========================================="
echo "Nova Carter Patrol RL Training"
echo "Job ID:       $SLURM_JOB_ID"
echo "Node:         $SLURMD_NODENAME"
echo "Envs:         $NUM_ENVS"
echo "Seed:         $SEED"
echo "Max Iter:     ${MAX_ITER:-from yaml (default)}"
echo "Video:        $VIDEO"
echo "Checkpoint:   ${CHECKPOINT:-none}"
echo "========================================="
echo ""

nvidia-smi
echo ""
echo "Running: $CMD"
echo ""

eval $CMD
TRAIN_EXIT=$?

echo ""
echo "========================================="
echo "Training Complete (exit code: $TRAIN_EXIT)"
echo "========================================="

# ── Auto-evaluate if training succeeded ──
if [ $TRAIN_EXIT -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "Running Evaluation"
    echo "========================================="

    # Find the latest training run (the one we just finished)
    LATEST_RUN=$(ls -td "$PROJECT_DIR/training/logs/sb3/NovaCarter-Patrol"/*/ 2>/dev/null | head -1)
    CHECKPOINT="${LATEST_RUN}model.zip"

    if [ -f "$CHECKPOINT" ]; then
        EVAL_CMD="$ISAAC_LAB_DIR/isaaclab.sh -p $PROJECT_DIR/training/eval_nova_carter.py --checkpoint $CHECKPOINT --headless --num_envs 16 --video_length 500 --num_videos 6"
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
else
    echo "Training failed — skipping evaluation."
fi

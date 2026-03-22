#!/usr/bin/env bash
#
# install_isaac_lab.sh — Install Isaac Lab v2.3.2 on Rosie
#
# Prerequisites:
#   - Must run on a GPU node (compilation requires GPU)
#   - Isaac Sim 5.1.0 installed at ~/isaacsim
#
# Usage:
#   srun --partition=teaching --gres=gpu:1 --time=1:00:00 --mem=16G --pty bash
#   cd ~/isaac-sim-msoe
#   bash scripts/install_isaac_lab.sh
#

set -euo pipefail

# ── Deactivate conda/venv ──────────────────────────────────────────────────
# isaaclab.sh uses conda/venv Python if active, which breaks the install.
if [ -n "${CONDA_PREFIX:-}" ]; then
    echo "[INFO]  Deactivating conda environment: $CONDA_DEFAULT_ENV"
    eval "$(conda shell.bash hook 2>/dev/null)" && conda deactivate
fi
unset VIRTUAL_ENV 2>/dev/null || true

# ── Configuration ───────────────────────────────────────────────────────────
ISAAC_SIM_PATH="$HOME/isaacsim"
ISAAC_LAB_PATH="$HOME/IsaacLab"
ISAAC_LAB_REPO="https://github.com/isaac-sim/IsaacLab.git"
ISAAC_LAB_TAG="v2.3.2"

# ── Helper Functions ────────────────────────────────────────────────────────
log()   { echo "[INFO]  $*"; }
warn()  { echo "[WARN]  $*" >&2; }
error() { echo "[ERROR] $*" >&2; exit 1; }

# ── Pre-flight Checks ──────────────────────────────────────────────────────
log "Starting Isaac Lab installation"
log "  Isaac Sim path : $ISAAC_SIM_PATH"
log "  Isaac Lab path : $ISAAC_LAB_PATH"
log "  Isaac Lab tag  : $ISAAC_LAB_TAG"
echo ""

# Check Isaac Sim exists
if [ ! -d "$ISAAC_SIM_PATH" ]; then
    error "Isaac Sim not found at $ISAAC_SIM_PATH. Install Isaac Sim first."
fi

if [ ! -f "$ISAAC_SIM_PATH/python.sh" ]; then
    error "Isaac Sim python.sh not found at $ISAAC_SIM_PATH/python.sh. Verify installation."
fi

# Check GPU availability
if ! command -v nvidia-smi &>/dev/null || ! nvidia-smi &>/dev/null; then
    error "No GPU detected. This script must run on a GPU node.
       Request one with: srun --partition=teaching --gres=gpu:1 --time=1:00:00 --mem=16G --pty bash"
fi
log "GPU detected: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

# Abort if Isaac Lab already exists
if [ -d "$ISAAC_LAB_PATH" ]; then
    error "$ISAAC_LAB_PATH already exists. Remove it first if you want a fresh install:
       rm -rf $ISAAC_LAB_PATH"
fi

# ── Step 1: Set Environment Variables ───────────────────────────────────────
log "Step 1/5: Setting environment variables"
export ISAACSIM_PATH="$ISAAC_SIM_PATH"
export ISAACSIM_PYTHON_EXE="$ISAAC_SIM_PATH/python.sh"
log "  ISAACSIM_PATH=$ISAACSIM_PATH"
log "  ISAACSIM_PYTHON_EXE=$ISAACSIM_PYTHON_EXE"

# ── Step 2: Clone Isaac Lab ─────────────────────────────────────────────────
log "Step 2/5: Cloning Isaac Lab repository"
git clone "$ISAAC_LAB_REPO" "$ISAAC_LAB_PATH"
cd "$ISAAC_LAB_PATH"

log "  Checking out tag $ISAAC_LAB_TAG"
git checkout "$ISAAC_LAB_TAG"
log "  Checked out $(git describe --tags --exact-match 2>/dev/null || git rev-parse --short HEAD)"

# ── Step 3: Create Symlink to Isaac Sim ─────────────────────────────────────
log "Step 3/5: Creating symlink _isaac_sim -> $ISAAC_SIM_PATH"
ln -sf "$ISAAC_SIM_PATH" "$ISAAC_LAB_PATH/_isaac_sim"

if [ -L "$ISAAC_LAB_PATH/_isaac_sim" ]; then
    log "  Symlink created successfully"
else
    error "Failed to create symlink"
fi

# ── Step 4: Install Isaac Lab ───────────────────────────────────────────────
log "Step 4/5: Running Isaac Lab installer (this may take several minutes)"
log "  Installing core + all RL frameworks (sb3, rl_games, rsl_rl, skrl)"
cd "$ISAAC_LAB_PATH"
./isaaclab.sh --install

log "  Installation complete"

# ── Step 5: Verify Installation ─────────────────────────────────────────────
log "Step 5/5: Running verification (create_empty.py)"
cd "$ISAAC_LAB_PATH"
./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py

log ""
log "============================================"
log " Isaac Lab $ISAAC_LAB_TAG installed successfully!"
log "============================================"
log ""
log "Quick test commands:"
log "  cd $ISAAC_LAB_PATH"
log "  ./isaaclab.sh -p -c \"import stable_baselines3; print('sb3 OK')\""
log "  ./isaaclab.sh -p -c \"import rl_games; print('rl_games OK')\""
log ""
log "To verify later:"
log "  cd ~/isaac-sim-msoe"
log "  bash scripts/verify_isaac_lab.sh"

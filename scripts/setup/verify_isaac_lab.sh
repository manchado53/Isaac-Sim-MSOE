#!/usr/bin/env bash
#
# verify_isaac_lab.sh — Quick verification that Isaac Lab is working
#
# Run this after installation or after a Rosie maintenance window to confirm
# Isaac Lab is still functional.
#
# Usage:
#   srun --partition=teaching --gres=gpu:1 --time=0:15:00 --pty bash
#   cd ~/isaac-sim-msoe
#   bash scripts/setup/verify_isaac_lab.sh
#

set -euo pipefail

# Deactivate conda if active — isaaclab.sh picks up conda's Python
# instead of Isaac Sim's Python, which doesn't have isaaclab installed.
if [ -n "${CONDA_PREFIX:-}" ]; then
    echo "[INFO]  Deactivating conda environment: $CONDA_DEFAULT_ENV"
    eval "$(conda shell.bash hook 2>/dev/null)" && conda deactivate
fi
unset VIRTUAL_ENV 2>/dev/null || true

ISAAC_LAB_PATH="$HOME/IsaacLab"

log()   { echo "[INFO]  $*"; }
error() { echo "[ERROR] $*" >&2; exit 1; }

log "Verifying Isaac Lab installation"

# Check directory exists
if [ ! -d "$ISAAC_LAB_PATH" ]; then
    error "Isaac Lab not found at $ISAAC_LAB_PATH. Run scripts/setup/install_isaac_lab.sh first."
fi

# Check symlink
if [ ! -L "$ISAAC_LAB_PATH/_isaac_sim" ]; then
    error "Symlink $ISAAC_LAB_PATH/_isaac_sim is missing. Reinstall Isaac Lab."
fi
log "Symlink OK: _isaac_sim -> $(readlink "$ISAAC_LAB_PATH/_isaac_sim")"

# Check GPU
if ! nvidia-smi &>/dev/null; then
    error "No GPU detected. Run this on a GPU node."
fi
log "GPU OK: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

# Run create_empty.py
log "Running create_empty.py..."
cd "$ISAAC_LAB_PATH"
./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py

# Check RL library imports
log "Checking RL library imports..."
./isaaclab.sh -p -c "import stable_baselines3; print('  stable_baselines3 OK')"
./isaaclab.sh -p -c "import rl_games; print('  rl_games OK')"
./isaaclab.sh -p -c "import rsl_rl; print('  rsl_rl OK')"
./isaaclab.sh -p -c "import skrl; print('  skrl OK')"

log ""
log "============================================"
log " All checks passed — Isaac Lab is working!"
log "============================================"

#!/bin/bash
#
# Run Isaac Sim interactively on Rosie
# Usage: ./run_isaac_interactive.sh [script.py]
#
# NOTE: This script must be run on a GPU node (via srun or in an interactive session)
#

# Set EULA acceptance
export OMNI_KIT_ACCEPT_EULA=YES
export PRIVACY_CONSENT=Y

# Isaac Sim installation path
ISAAC_SIM_DIR="$HOME/isaacsim"
ISAAC_PYTHON="$ISAAC_SIM_DIR/python.sh"

# Check if Isaac Sim is installed
if [ ! -f "$ISAAC_PYTHON" ]; then
    echo "Error: Isaac Sim not found at $ISAAC_SIM_DIR"
    echo "Please verify the installation."
    exit 1
fi

# If a script is provided, run it; otherwise show usage
if [ $# -eq 0 ]; then
    echo "======================================"
    echo "Isaac Sim Interactive Runner"
    echo "======================================"
    echo ""
    echo "Usage: $0 <script.py>"
    echo ""
    echo "Example:"
    echo "  $0 simulations/hello_isaac.py"
    echo ""
    echo "To get an interactive GPU session first:"
    echo "  srun --partition=teaching --gres=gpu:1 --time=2:00:00 --pty bash"
    echo ""
    exit 0
else
    SCRIPT="$1"
    echo "======================================"
    echo "Running Isaac Sim Script"
    echo "======================================"
    echo "Script: $SCRIPT"
    echo "Isaac Sim: $ISAAC_SIM_DIR"
    echo "Node: $(hostname)"
    echo ""

    # Check if GPU is available
    if ! nvidia-smi &> /dev/null; then
        echo "WARNING: No GPU detected on this node!"
        echo "Isaac Sim requires GPU access to run properly."
        echo ""
    fi

    # Run the script
    cd "$HOME/isaac-sim-msoe" || exit 1
    "$ISAAC_PYTHON" "$SCRIPT"
fi

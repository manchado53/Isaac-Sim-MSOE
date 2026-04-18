#!/bin/bash
# Launch the Isaac Sim Replicator Agent (IRA) headless runner against the
# MSOE Innovation Center ground-floor config. This is the "W3" fallback for
# the people-walking pipeline — only invoke it if the simpler W2.5 path
# (direct omni.anim.* extensions) fails to get characters animating.
#
# Usage (on a GPU node):
#   ./simulations/environment/run_ira_ground_floor.sh
#
# or pass extra flags to sdg_scheduler.py:
#   ./simulations/environment/run_ira_ground_floor.sh --save_usd
#
# Output: frames land in $HOME/ReplicatorResult/ by default (see
# WriterSetting.get_writer_default_output_path). To redirect, set the carb
# setting /exts/isaacsim.replicator.agent/default_replicator_output_path.

set -euo pipefail

# Required by Isaac Sim bootstrap
export OMNI_KIT_ACCEPT_EULA=YES
export PRIVACY_CONSENT=Y
export PYTHONUNBUFFERED=1

ISAAC_SIM_DIR="${ISAAC_SIM_DIR:-$HOME/isaacsim}"
ISAAC_PYTHON="$ISAAC_SIM_DIR/python.sh"
IRA_SCHEDULER="$ISAAC_SIM_DIR/tools/actor_sdg/sdg_scheduler.py"

# Resolve this script's directory so the config paths work regardless of cwd
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

CONFIG_FILE="$PROJECT_DIR/assets/people/ira_configs/ground_floor_config.yaml"
CAMERA_FILE="$PROJECT_DIR/assets/people/ira_configs/ground_floor_cameras.json"

if [ ! -f "$IRA_SCHEDULER" ]; then
    echo "ERROR: IRA scheduler not found at $IRA_SCHEDULER" >&2
    exit 1
fi
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: IRA config not found at $CONFIG_FILE" >&2
    exit 1
fi

cd "$PROJECT_DIR"

echo "========================================="
echo "IRA Ground-Floor Demo Launcher"
echo "Scheduler:    $IRA_SCHEDULER"
echo "Config:       $CONFIG_FILE"
echo "Cameras:      $CAMERA_FILE"
echo "Project dir:  $PROJECT_DIR"
echo "========================================="

# --no_random_commands: honor ground_floor_commands.txt verbatim instead of
#                       generating random GoTo targets.
# --debug_print:        enable IRA debug output to SLURM/stdout.
"$ISAAC_PYTHON" "$IRA_SCHEDULER" \
    -c "$CONFIG_FILE" \
    --sensor_placment_file "$CAMERA_FILE" \
    --no_random_commands \
    --debug_print \
    "$@"

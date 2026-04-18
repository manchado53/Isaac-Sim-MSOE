#!/bin/bash
#SBATCH --job-name=physics_chair_sim
#SBATCH --partition=teaching
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --chdir=/home/ad.msoe.edu/thomasst/AnomalyCreation
#SBATCH --output=/home/ad.msoe.edu/thomasst/AnomalyCreation/logs/physics_chairs_%j.out
#SBATCH --error=/home/ad.msoe.edu/thomasst/AnomalyCreation/logs/physics_chairs_%j.err

# ---------------------------------------------------------------------------
# ROSIE batch job — spawns physics-enabled chairs on floor 4, runs simulation,
# renders images.
#
# Submit with:
#   mkdir -p logs
#   sbatch run_physics_chairs_rosie.sh
#
# Override defaults:
#   sbatch --export=ALL,CHAIRS=5,SHOTS=4,SIM_STEPS=360 run_physics_chairs_rosie.sh
# ---------------------------------------------------------------------------

CHAIRS="${CHAIRS:-3}"
SHOTS="${SHOTS:-4}"
SIM_STEPS="${SIM_STEPS:-240}"   # 240 steps @ 60 Hz = 4 seconds of physics
RES_W="${RES_W:-1280}"
RES_H="${RES_H:-720}"
SUBFRAMES="${SUBFRAMES:-32}"
SEED="${SEED:-42}"

SCRIPT_DIR="/home/ad.msoe.edu/thomasst/AnomalyCreation"
ISAAC_PYTHON="${ISAAC_PYTHON:-${HOME}/isaacsim/python.sh}"
OUT_DIR="${SCRIPT_DIR}/captures/physics_chairs"

echo "========================================================"
echo "  Job ID    : ${SLURM_JOB_ID}"
echo "  Node      : $(hostname)"
echo "  GPUs      : ${CUDA_VISIBLE_DEVICES}"
echo "  Chairs    : ${CHAIRS}  shots=${SHOTS}"
echo "  Sim steps : ${SIM_STEPS} ($(echo "scale=1; ${SIM_STEPS}/60" | bc) s simulated)"
echo "  Res       : ${RES_W}x${RES_H}  subframes=${SUBFRAMES}"
echo "  Floor     : 4th (Z ≈ 22.7 m)"
echo "  Output    : ${OUT_DIR}"
echo "========================================================"

echo "--- nvidia-smi ---"
nvidia-smi || { echo "ERROR: nvidia-smi failed — no GPU driver on this node"; exit 1; }
echo "------------------"

if [ ! -f "${ISAAC_PYTHON}" ]; then
    echo "ERROR: Isaac Sim python not found at ${ISAAC_PYTHON}"
    exit 1
fi

USD_FILE="${SCRIPT_DIR}/MSOEBuildingV4_physics.usd"
if [ ! -f "${USD_FILE}" ]; then
    echo "ERROR: USD file not found: ${USD_FILE}"
    exit 1
fi

unset DISPLAY
export OMNI_KIT_ACCEPT_EULA=YES

for ICD in /usr/share/vulkan/icd.d/nvidia_icd.json \
           /etc/vulkan/icd.d/nvidia_icd.json; do
    if [ -f "${ICD}" ]; then
        export VK_ICD_FILENAMES="${ICD}"
        echo "Vulkan ICD: ${ICD}"
        break
    fi
done

for EGL in /usr/share/glvnd/egl_vendor.d/10_nvidia.json \
           /etc/glvnd/egl_vendor.d/10_nvidia.json; do
    if [ -f "${EGL}" ]; then
        export __EGL_VENDOR_LIBRARY_FILENAMES="${EGL}"
        echo "EGL vendor: ${EGL}"
        break
    fi
done

mkdir -p "${OUT_DIR}/images"

echo ""
echo "Starting Isaac Sim physics simulation at $(date)"
"${ISAAC_PYTHON}" "${SCRIPT_DIR}/simulate_fallen_chairs.py" \
    --chairs    "${CHAIRS}"    \
    --shots     "${SHOTS}"     \
    --sim-steps "${SIM_STEPS}" \
    --res       "${RES_W}" "${RES_H}" \
    --subframes "${SUBFRAMES}" \
    --seed      "${SEED}"      \
    --no-video                 \
    --out-dir   "${OUT_DIR}"

STATUS=$?
echo ""
echo "Isaac Sim exited with status ${STATUS} at $(date)"

if [ ${STATUS} -eq 0 ]; then
    IMAGE_COUNT=$(find "${OUT_DIR}/images" -name "PhysicsChairs_*.png" 2>/dev/null | wc -l)
    echo "Images saved : ${IMAGE_COUNT} rendered PNGs in ${OUT_DIR}/images/"
else
    echo "ERROR: simulate_fallen_chairs.py failed — check logs/physics_chairs_${SLURM_JOB_ID}.err"
fi

exit ${STATUS}

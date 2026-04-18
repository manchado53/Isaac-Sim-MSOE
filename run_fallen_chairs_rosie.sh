#!/bin/bash
#SBATCH --job-name=fallen_chair_capture
#SBATCH --partition=teaching
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --chdir=/home/ad.msoe.edu/thomasst/AnomalyCreation
#SBATCH --output=/home/ad.msoe.edu/thomasst/AnomalyCreation/logs/fallen_chairs_%j.out
#SBATCH --error=/home/ad.msoe.edu/thomasst/AnomalyCreation/logs/fallen_chairs_%j.err

# ---------------------------------------------------------------------------
# ROSIE batch job — renders fallen chair anomaly images with Isaac Sim
# Submit with:
#   mkdir -p logs
#   sbatch run_fallen_chairs_rosie.sh
#
# Override defaults without editing this file:
#   sbatch --export=ALL,COUNT=3,SHOTS=4 run_fallen_chairs_rosie.sh
# ---------------------------------------------------------------------------

COUNT="${COUNT:-5}"
SHOTS="${SHOTS:-4}"
RES_W="${RES_W:-1280}"
RES_H="${RES_H:-720}"
SUBFRAMES="${SUBFRAMES:-32}"
SEED="${SEED:-42}"

SCRIPT_DIR="/home/ad.msoe.edu/thomasst/AnomalyCreation"
ISAAC_PYTHON="${ISAAC_PYTHON:-${HOME}/isaacsim/python.sh}"
OUT_DIR="${SCRIPT_DIR}/captures/fallen_chairs"

echo "========================================================"
echo "  Job ID   : ${SLURM_JOB_ID}"
echo "  Node     : $(hostname)"
echo "  GPUs     : ${CUDA_VISIBLE_DEVICES}"
echo "  Anomaly  : fallen_chairs  count=${COUNT}  shots=${SHOTS}"
echo "  Res      : ${RES_W}x${RES_H}  subframes=${SUBFRAMES}"
echo "  Output   : ${OUT_DIR}"
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
echo "Starting Isaac Sim renderer at $(date)"
"${ISAAC_PYTHON}" "${SCRIPT_DIR}/capture_fallen_chairs.py" \
    --count     "${COUNT}"     \
    --shots     "${SHOTS}"     \
    --res       "${RES_W}" "${RES_H}" \
    --subframes "${SUBFRAMES}" \
    --seed      "${SEED}"      \
    --no-video  \
    --out-dir   "${OUT_DIR}"

STATUS=$?
echo ""
echo "Isaac Sim exited with status ${STATUS} at $(date)"

if [ ${STATUS} -eq 0 ]; then
    IMAGE_COUNT=$(find "${OUT_DIR}/images" -name "FallenChair_*.png" 2>/dev/null | wc -l)
    echo "Images saved : ${IMAGE_COUNT} rendered PNGs in ${OUT_DIR}/images/"
else
    echo "ERROR: capture_fallen_chairs.py failed — check logs/fallen_chairs_${SLURM_JOB_ID}.err"
fi

exit ${STATUS}

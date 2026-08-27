#!/usr/bin/env bash
#SBATCH --job-name=axon-infer
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

if [[ $# -lt 3 ]]; then
    echo "Usage: sbatch $0 INPUT CHECKPOINT OUTPUT_DIR [INFERENCE_OPTIONS ...]" >&2
    exit 2
fi

INPUT=$1
CHECKPOINT=$2
OUTPUT_DIR=$3
shift 3

PROJECT_DIR=${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}
PYTHON=${PYTHON:-python}
SEGMENTATION_MODE=${SEGMENTATION_MODE:-three_class_shell_interior}
[[ -f "${PROJECT_DIR}/inference/infer_lsm.py" ]] || { echo "Missing inference entry point under PROJECT_DIR=${PROJECT_DIR}" >&2; exit 1; }

export PYTHONUNBUFFERED=1

INFERENCE_ARGS=(
    --input "${INPUT}"
    --checkpoint "${CHECKPOINT}"
    --output-dir "${OUTPUT_DIR}"
    --segmentation-mode "${SEGMENTATION_MODE}"
    --require-cuda
)
srun "${PYTHON}" "${PROJECT_DIR}/inference/infer_lsm.py" "${INFERENCE_ARGS[@]}" "$@"

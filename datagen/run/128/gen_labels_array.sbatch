#!/usr/bin/env bash
#SBATCH --job-name=axon_labels
#SBATCH --array=0-599%50
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=slurm-%x-%A_%a.out

set -euo pipefail

: "${DENSE_LABEL_ROOT:?Set DENSE_LABEL_ROOT to the output dataset root}"

PROJECT_DIR=${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}
PYTHON=${PYTHON:-python}
[[ -f "${PROJECT_DIR}/datagen/generate_dense_label.py" ]] || { echo "Missing dense-label entry point under PROJECT_DIR=${PROJECT_DIR}" >&2; exit 1; }

export PYTHONUNBUFFERED=1
srun "${PYTHON}" "${PROJECT_DIR}/datagen/generate_dense_label.py" \
    --vol-id "${SLURM_ARRAY_TASK_ID}" \
    --output-root "${DENSE_LABEL_ROOT}"

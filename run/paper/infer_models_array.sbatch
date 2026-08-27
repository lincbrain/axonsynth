#!/usr/bin/env bash
#SBATCH --job-name=axonsynth-paper-infer
#SBATCH --array=0-23%8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=local/logs/slurm-%x-%A_%a.out

set -euo pipefail

: "${DATA_ROOT:?Set DATA_ROOT}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT}"
: "${THREE_CLASS_CHECKPOINT:?Set THREE_CLASS_CHECKPOINT to the checkpoint file}"
: "${BINARY_CHECKPOINT:?Set BINARY_CHECKPOINT to the checkpoint file}"

PROJECT_DIR=${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}
PYTHON=${PYTHON:-python}
CONFIG=${CONFIG:-${PROJECT_DIR}/paper_analysis/config/lsm_12patch.json}
[[ -f "${CONFIG}" ]] || { echo "Missing canonical config: ${CONFIG}" >&2; exit 1; }

export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
srun "${PYTHON}" -m paper_analysis.run_model_inference \
    --config "${CONFIG}" \
    --task-index "${SLURM_ARRAY_TASK_ID}"

#!/usr/bin/env bash
#SBATCH --job-name=axonsynth-baselines
#SBATCH --array=0-11%4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=local/logs/slurm-%x-%A_%a.out

set -euo pipefail

: "${DATA_ROOT:?Set DATA_ROOT}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT}"
: "${THREE_CLASS_CHECKPOINT:?Set THREE_CLASS_CHECKPOINT for canonical config validation}"
: "${BINARY_CHECKPOINT:?Set BINARY_CHECKPOINT for canonical config validation}"

PROJECT_DIR=${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}
PYTHON=${PYTHON:-python}
CONFIG=${CONFIG:-${PROJECT_DIR}/paper_analysis/config/lsm_12patch.json}
[[ -f "${CONFIG}" ]] || { echo "Missing canonical config: ${CONFIG}" >&2; exit 1; }

export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
srun "${PYTHON}" -m paper_analysis.generate_baselines \
    --config "${CONFIG}" \
    --patch-index "${SLURM_ARRAY_TASK_ID}"

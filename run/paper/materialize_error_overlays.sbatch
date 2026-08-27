#!/usr/bin/env bash
#SBATCH --job-name=axonsynth-overlays
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

: "${DATA_ROOT:?Set DATA_ROOT}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT}"
: "${THREE_CLASS_CHECKPOINT:?Set THREE_CLASS_CHECKPOINT}"
: "${BINARY_CHECKPOINT:?Set BINARY_CHECKPOINT}"

PROJECT_DIR=${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}
PYTHON=${PYTHON:-python}
CONFIG=${CONFIG:-${PROJECT_DIR}/paper_analysis/config/lsm_12patch.json}
[[ -f "${CONFIG}" ]] || { echo "Missing canonical config: ${CONFIG}" >&2; exit 1; }

export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
srun "${PYTHON}" -m paper_analysis.materialize_error_overlays --config "${CONFIG}"

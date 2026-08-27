#!/usr/bin/env bash
#SBATCH --job-name=axonsynth-release-smoke
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=local/logs/slurm-%x-%j.out

set -euo pipefail

: "${THREE_CLASS_CHECKPOINT:?Set THREE_CLASS_CHECKPOINT to the checkpoint file}"
: "${BINARY_CHECKPOINT:?Set BINARY_CHECKPOINT to the checkpoint file}"

PROJECT_DIR=${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}
PYTHON=${PYTHON:-python}
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

srun "${PYTHON}" "${PROJECT_DIR}/tests/smoke_checkpoint_load.py" \
    --binary-checkpoint "${BINARY_CHECKPOINT}" \
    --three-class-checkpoint "${THREE_CLASS_CHECKPOINT}"

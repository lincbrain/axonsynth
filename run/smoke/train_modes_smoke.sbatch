#!/usr/bin/env bash
#SBATCH --job-name=axonsynth-smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=local/logs/slurm-%x-%j.out

set -euo pipefail

: "${LABEL_DIR:?Set LABEL_DIR to the directory containing label volumes}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR for smoke-test outputs}"

PROJECT_DIR=${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}
PYTHON=${PYTHON:-python}

COMMON_ARGS=(
    --label_dir "${LABEL_DIR}"
    --epochs 1
    --batch_size 1
    --num_workers 2
    --samples_per_vol 1
    --max_volumes 2
    --val_interval 1
    --sw_batch_size 1
)

srun "${PYTHON}" -u "${PROJECT_DIR}/train.py" \
    "${COMMON_ARGS[@]}" \
    --output_dir "${OUTPUT_DIR}/three_class_shell_interior" \
    --segmentation_mode three_class_shell_interior \
    --cache_builder gpu

srun "${PYTHON}" -u "${PROJECT_DIR}/train.py" \
    "${COMMON_ARGS[@]}" \
    --output_dir "${OUTPUT_DIR}/binary" \
    --segmentation_mode binary \
    --cache_builder cpu

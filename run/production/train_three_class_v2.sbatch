#!/usr/bin/env bash
#SBATCH --job-name=axonsynth-3class
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=50
#SBATCH --gres=gpu:1
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --signal=TERM@120
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

: "${LABEL_DIR:?Set LABEL_DIR to the directory containing label volumes}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR for checkpoints and TensorBoard logs}"

PROJECT_DIR=${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}
PYTHON=${PYTHON:-python}
[[ -f "${PROJECT_DIR}/train.py" ]] || { echo "Missing train.py under PROJECT_DIR=${PROJECT_DIR}" >&2; exit 1; }

TRAIN_ARGS=(
    --label_dir "${LABEL_DIR}"
    --output_dir "${OUTPUT_DIR}"
    --segmentation_mode three_class_shell_interior
    --cache_builder gpu
    --three_class_weights 1 1 1
    --epochs 200
    --batch_size 2
    --lr 1e-4
    --num_workers 48
    --val_fraction 0.2
    --val_interval 1
    --samples_per_vol 2
    --cache_epochs 3
    --max_volumes 500
    --roi_size 128
    --sw_batch_size 4
    --seed 42
    --subset_fraction_lo 0.3
    --subset_fraction_hi 0.9
)

if [[ "${RESUME:-auto}" == "1" ]] ||
   [[ "${RESUME:-auto}" == "auto" && -d "${OUTPUT_DIR}/checkpoints" ]]; then
    TRAIN_ARGS+=(--resume)
fi

srun "${PYTHON}" -u "${PROJECT_DIR}/train.py" "${TRAIN_ARGS[@]}"

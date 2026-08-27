# AxonSynth

Release for synthetic-data training and 3D axon segmentation.

## Supported Methods

- **Primary:** three-class v2 MONAI 3D U-Net predicting background, axon shell, and axon interior. Shell and interior probabilities are collapsed for binary foreground evaluation.
- **Ablation:** the supported original binary MONAI 3D U-Net.
- **Baselines:** a domain-calibrated normalized-intensity threshold and a calibrated bright-only Frangi filter (`black_ridges=false`).

Thresholds, Frangi parameters, patch splits, and portable path templates are fixed in `paper_analysis/config/lsm_12patch.json`.

## Entry Points

- `run/production/train_three_class_v2.sbatch`: primary training recipe.
- `run/ablations/train_binary_baseline.sbatch`: original binary ablation.
- `run/production/infer_lsm.sbatch`: verified three-class inference.
- `datagen/run/128/gen_labels_array.sbatch`: deterministic 600-volume dense-label corpus (`LABEL_DIR` is its `train/` subdirectory).
- `run/paper/`: fixed 12-patch inference, baselines, evaluation, and packet jobs.

## Environment

Create the CUDA training and inference environment with:

```bash
conda env create -f environment.yml
conda activate axonsynth
```

The smaller CPU analysis environment is defined in `paper_analysis/environment.yml`. Set site paths with `DATA_ROOT` and `OUTPUT_ROOT`; `config/site.example.env` documents the local variables.

## Local Workspace

Use `local/sbatch/` for temporary Slurm scripts and `local/logs/` for scheduler
output. Their contents are ignored by Git. Submit the tracked Slurm scripts
from the repository root so their configured log paths resolve correctly.

## Artifacts

Model weights are **not stored in Git**. Their filenames, architectures, and training metadata are documented in:

- `repro/checkpoints/three_class_v2_best.json`
- `repro/checkpoints/binary_baseline_best.json`

Source and checkpoint caveats are recorded in `docs/provenance.md`.

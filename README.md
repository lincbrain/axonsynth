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
- `inference/prepare_lsm_full.py`: high-memory axis reorder, anti-aliasing, and isotropic full-field preparation.
- `datagen/run/128/gen_labels_array.sbatch`: deterministic 600-volume dense-label corpus (`LABEL_DIR` is its `train/` subdirectory).
- `run/paper/`: fixed 12-patch inference, baselines, evaluation, and packet jobs.

## Environment

Create the CUDA training and inference environment with:

```bash
conda env create -f environment.yml
conda activate axonsynth
```

The smaller CPU analysis environment is defined in `paper_analysis/environment.yml`. Set site paths with `DATA_ROOT` and `OUTPUT_ROOT`; `config/site.example.env` documents the local variables.

## Full-Field Inference

The model expects isotropic model-order input. Prepare anisotropic NIfTI data
before inference; `--assume-source-units-micron` is required when a source
header does not declare micron units:

```bash
python inference/prepare_lsm_full.py \
    --input source.nii.gz \
    --output prepared_0p8um.nii.gz \
    --target-spacing 0.8 \
    --axis-order 2 0 1 \
    --assume-source-units-micron
```

For fields that exceed GPU stitching memory, keep the normalized volume and
stitched logits on CPU while running model windows on CUDA:

```bash
python inference/infer_lsm.py \
    --input prepared_0p8um.nii.gz \
    --checkpoint best_model.pt \
    --output-dir predictions \
    --segmentation-mode three_class_shell_interior \
    --stitch-device cpu \
    --output-profile binary_focused \
    --threshold 0.5 \
    --additional-threshold 0p94=0.94 \
    --require-cuda
```

Normalization and Gaussian logit blending remain global over the prepared
field. Do not infer independent outer tiles and merge their probabilities.

## Local Workspace

Use `local/sbatch/` for temporary Slurm scripts and `local/logs/` for scheduler
output. Their contents are ignored by Git. Submit the tracked Slurm scripts
from the repository root so their configured log paths resolve correctly.

## Artifacts

Model weights are **not stored in Git**. Their filenames, architectures, and training metadata are documented in:

- `repro/checkpoints/three_class_v2_best.json`
- `repro/checkpoints/binary_baseline_best.json`

Source and checkpoint caveats are recorded in `docs/provenance.md`.

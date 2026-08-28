# AxonSynth

Release for synthetic-data training and 3D axon segmentation.

## Supported Methods

- **Primary:** three-class MONAI 3D U-Net predicting background, axon shell, and axon interior. Shell and interior probabilities are collapsed for binary foreground evaluation.
- **Ablation:** the supported original binary MONAI 3D U-Net.
- **Baselines:** a domain-calibrated normalized-intensity threshold and a calibrated bright-only Frangi filter (`black_ridges=false`).

Thresholds, Frangi parameters, patch splits, and portable path templates are fixed in `paper_analysis/config/lsm_12patch.json`.

## Pipeline

1. **Generate geometry:** `datagen/run/128/gen_labels_array.sbatch` creates the deterministic dense label/probability corpus.
2. **Train and validate:** `run/production/train_three_class_v2.sbatch` launches the primary recipe. `train.py` performs validation, best-checkpoint selection, periodic checkpointing, and resume handling.
3. **Prepare and infer:** `inference/prepare_lsm_full.py` converts anisotropic source NIfTIs to model space, and `inference/infer_lsm.py` or `run/production/infer_lsm.sbatch` runs a checkpoint.
4. **Evaluate:** `paper_analysis/metrics.py` provides reusable binary-mask metrics. `run/paper/` applies them in the fixed 12-patch paper workflow.

Training does not require pre-generated intensity images. It reads stored dense
label/probability pairs and synthesizes images and targets into RAM caches;
training caches are refreshed periodically while the validation cache remains
fixed. The original binary ablation is in `run/ablations/`, and lightweight
checks are in `run/smoke/` and `tests/`.

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
    --no-save-input \
    --threshold 0.5 \
    --additional-threshold 0p94=0.94 \
    --require-cuda
```

Use `--no-save-input` when the normalized input volume does not need to be
retained. Normalization and Gaussian logit blending remain global over the
prepared field. Do not infer independent outer tiles and merge their
probabilities.

## Evaluation and External Tests

`paper_analysis/metrics.py` is independent of the paper cohort and accepts any
matching 3D binary prediction and target arrays:

```python
from paper_analysis.metrics import evaluate_binary_mask

results = evaluate_binary_mask(prediction, target)
```

It reports voxel, topology, and connected-component metrics. See
`tests/test_metrics.py` for small examples. In contrast,
`paper_analysis/evaluate_lsm.py` is the reproducibility runner for the 12
patches and four methods defined in `paper_analysis/config/lsm_12patch.json`;
it expects predictions to have already been generated. New labeled datasets
can use `inference/infer_lsm.py` and import `evaluate_binary_mask` directly.

## Local Workspace

Use `local/sbatch/` for temporary Slurm scripts and `local/logs/` for scheduler
output. Their contents are ignored by Git. Submit the tracked Slurm scripts
from the repository root so their configured log paths resolve correctly.

## Artifacts

Model weights are **not stored in Git**. Their filenames, architectures, and training metadata are documented in:

- `repro/checkpoints/three_class_v2_best.json`
- `repro/checkpoints/binary_baseline_best.json`
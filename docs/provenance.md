# Provenance

The clean repository is the authoritative release source. Model weights are distributed separately and selected through checkpoint paths.

## Checkpoints

| Model | Manifest | Availability |
| --- | --- | --- |
| Three-class v2 | `repro/checkpoints/three_class_v2_best.json` | local only |
| Original binary | `repro/checkpoints/binary_baseline_best.json` | local only |

Weights are intentionally excluded from Git. The three-class synthetic checkpoint score is the mean of foreground, shell, and interior Dice from hard argmax class masks. The binary checkpoint records best epoch `182` and synthetic validation Dice `0.858588695526123`. It is the best-through-epoch-200 milestone of a run configured for 500 epochs and a 495-epoch cosine horizon.

## Evaluation

The 12 LSM patch identifiers and calibration split are fixed in `paper_analysis/config/lsm_12patch.json`. Domain thresholds and bright-only Frangi parameters were selected on one human and one macaque calibration patch. Learned-model and intensity thresholds were selected using target-informed corrected Dice, whereas Frangi parameters were selected using raw Dice; the release labels this historical difference and treats raw held-out metrics as primary. Target-informed neighbor correction uses two face-6 rounds and must be reported as corrected evaluation, not as target-independent inference.

Historical tables prefixed this secondary metric with `vesynth_`. The clean release removes that project-specific prefix and reports the unchanged values as `target_informed_neighbor_corrected_*`.

Exact training dependencies are installed from `environment.yml`.

Resuming training starts from the latest atomically published completed epoch. A resumed process refreshes any in-memory synthesis cache, so training resume is safe from replaying partial optimizer updates but is not promised to be bit-identical to an uninterrupted run.

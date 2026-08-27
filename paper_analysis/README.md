# Paper Analysis

The fixed analysis compares four methods on two calibration and ten held-out
LSM patches:

- three-class v2;
- original binary ablation;
- calibrated normalized-intensity threshold;
- bright-only Frangi.

Copy the exported path template in `config/site.example.env` to the ignored
`config/site.env`, source that file, and submit the learned
inference and baseline arrays. After both arrays complete, run the evaluation,
depth, overlay, and packet jobs in `run/paper/`.

Raw metrics are reported alongside
`target_informed_neighbor_corrected_*`. The latter uses the target to
apply a two-round face-neighbor evaluation tolerance and is not deployable
prediction postprocessing.

Raw metrics are the primary reported comparison. The accepted learned-model
and intensity operating points were historically selected by target-informed
corrected Dice on each domain's calibration patch; Frangi was selected by raw
Dice. The criterion is retained and labeled rather than retrospectively
changing the accepted thresholds.

Inference and baseline sidecars record the input, checkpoint, output paths, and
fixed method settings used for each case. Evaluation also checks dimensions,
affines, finite score ranges, and agreement between each score and saved mask.

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from paper_analysis.build_results_packet import run as build_packet
from paper_analysis.configuration import (
    METHOD_NAMES,
    load_pipeline_config,
    paper_context,
)
from paper_analysis.evaluate_depth_trends import run as evaluate_depth
from paper_analysis.evaluate_lsm import run as evaluate_lsm
from paper_analysis.materialize_error_overlays import run as write_overlays


CONFIG = Path(__file__).resolve().parents[1] / "paper_analysis/config/lsm_12patch.json"


def setup_canonical_inputs(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    output_root = tmp_path / "output"
    three_class_checkpoint = tmp_path / "three_class.pt"
    binary_checkpoint = tmp_path / "binary.pt"
    three_class_checkpoint.touch()
    binary_checkpoint.touch()
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    monkeypatch.setenv("OUTPUT_ROOT", str(output_root))
    monkeypatch.setenv("THREE_CLASS_CHECKPOINT", str(three_class_checkpoint))
    monkeypatch.setenv("BINARY_CHECKPOINT", str(binary_checkpoint))

    preliminary_config = load_pipeline_config(CONFIG)
    affine = np.diag([0.8, 0.8, 0.8, 1.0])
    target = np.zeros((8, 4, 4), dtype=np.uint8)
    target[:, 1, 1] = 1
    score = target.astype(np.float32)
    document = json.loads(CONFIG.read_text(encoding="utf-8"))
    for patch in preliminary_config.patches:
        raw_path = preliminary_config.path("raw_patch", patch=patch)
        target_path = preliminary_config.path("target_mask", patch=patch)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(score, affine), raw_path)
        nib.save(nib.Nifti1Image(target, affine), target_path)

    config_path = tmp_path / "lsm_12patch_test.json"
    config_path.write_text(json.dumps(document), encoding="utf-8")
    config = load_pipeline_config(config_path)
    for patch in config.patches:
        for method_name in METHOD_NAMES:
            method = config.get_method(method_name)
            score_path = config.path("prediction_score", patch=patch, method=method)
            mask_path = config.path("prediction_mask", patch=patch, method=method)
            score_path.parent.mkdir(parents=True, exist_ok=True)
            nib.save(nib.Nifti1Image(score, affine), score_path)
            nib.save(nib.Nifti1Image(target, affine), mask_path)
            if method.kind == "learned":
                metadata = {
                    "input_path": str(config.path("raw_patch", patch=patch)),
                    "checkpoint": {"path": str(method.checkpoint)},
                    "resolved_segmentation_mode": method.segmentation_mode,
                    "inference_parameters": {
                        "threshold": method.threshold_for(patch.domain),
                        "roi_size": [128, 128, 128],
                        "sliding_window_batch_size": 4,
                        "overlap": 0.5,
                        "sliding_window_blend_mode": "gaussian",
                        "sliding_window_sigma_scale": 0.125,
                        "padding_mode": "constant",
                        "padding_cval": 0.0,
                        "amp_enabled": True,
                        "amp_dtype": "float16",
                    },
                    "device": {"type": "cuda"},
                    "output_paths": {
                        "pred_prob": str(score_path),
                        "pred": str(mask_path),
                    },
                    "paper_context": paper_context(config, patch, method),
                }
                metadata_path = config.path("learned_metadata", patch=patch, method=method)
            else:
                metadata = {
                    "schema_version": 3,
                    "method": method.name,
                    "patch_id": patch.id,
                    "input": str(config.path("raw_patch", patch=patch)),
                    "config": str(config.source),
                    "threshold_selection_metric": method.threshold_selection_metric,
                    "parameters": {
                        "threshold": method.threshold_for(patch.domain)
                    },
                    "outputs": {
                        "score": str(score_path),
                        "mask": str(mask_path),
                    },
                }
                metadata_path = config.path("baseline_metadata", patch=patch, method=method)
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    return config, config_path


def test_configuration_resolves_48_unique_prediction_paths(tmp_path, monkeypatch):
    config, _ = setup_canonical_inputs(tmp_path, monkeypatch)
    paths = {
        config.path("prediction_score", patch=patch, method=method)
        for patch in config.patches
        for method in METHOD_NAMES
    }
    assert len(paths) == 48


def test_end_to_end_tables_depth_overlays_and_packet(tmp_path, monkeypatch):
    config, config_path = setup_canonical_inputs(tmp_path, monkeypatch)

    rows, aggregates = evaluate_lsm(config_path)
    assert len(rows) == 48
    assert len({(row["patch_id"], row["method"]) for row in rows}) == 48
    assert all(row["raw_dice"] == 1.0 for row in rows)
    assert all(
        row["target_informed_neighbor_corrected_dice"] == 1.0
        for row in rows
    )
    assert {
        row["method"]
        for row in aggregates
        if row["split"] == "heldout" and row["domain"] == "all"
    } == set(METHOD_NAMES)

    depth_rows, trends = evaluate_depth(config_path)
    assert len(depth_rows) == 48 * 8
    assert len(trends) == 48
    assert all(row["raw_dice"] == 1.0 for row in depth_rows)

    overlays = write_overlays(config_path)
    assert len(overlays) == 48
    assert all(Path(row["path"]).is_file() for row in overlays)

    config.path("macro_metrics_csv").write_text("intentionally stale\n", encoding="utf-8")
    manifest = build_packet(config_path)
    assert manifest["rows"] == 48
    packet_manifest = json.loads(config.path("packet_manifest").read_text())
    assert packet_manifest["methods"] == list(METHOD_NAMES)


def test_evaluation_rejects_missing_target(tmp_path, monkeypatch):
    config, config_path = setup_canonical_inputs(tmp_path, monkeypatch)
    patch = config.patches[0]
    config.path("target_mask", patch=patch).unlink()

    with pytest.raises(FileNotFoundError, match="Missing evaluation input"):
        evaluate_lsm(config_path)


def test_evaluation_rejects_output_changed_after_metadata(tmp_path, monkeypatch):
    config, config_path = setup_canonical_inputs(tmp_path, monkeypatch)
    patch = config.patches[0]
    method = config.get_method("three_class_v2")
    score_path = config.path("prediction_score", patch=patch, method=method)
    image = nib.load(score_path)
    changed = np.asarray(image.dataobj, dtype=np.float32).copy()
    changed[0, 1, 1] = 0.5
    nib.save(nib.Nifti1Image(changed, image.affine, image.header), score_path)

    with pytest.raises(ValueError, match="Saved mask does not match"):
        evaluate_lsm(config_path)

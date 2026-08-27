from argparse import Namespace
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from inference.infer_lsm import (
    default_output_prefix,
    load_patch,
    normalize_patch,
    output_affine,
    postprocess_logits,
    resolve_segmentation_mode,
    validate_output_prefix,
)


@pytest.mark.parametrize(
    ("checkpoint", "expected"),
    [
        ({}, "binary"),
        ({"args": {}}, "binary"),
        (
            {"args": {"segmentation_mode": "three_class_shell_interior"}},
            "three_class_shell_interior",
        ),
        (
            {"args": Namespace(segmentation_mode="three_class_shell_interior")},
            "three_class_shell_interior",
        ),
    ],
)
def test_auto_mode_supports_legacy_and_v2_checkpoint_args(checkpoint, expected):
    assert resolve_segmentation_mode("auto", checkpoint) == expected


def test_explicit_mode_overrides_checkpoint_args():
    checkpoint = {"args": {"segmentation_mode": "three_class_shell_interior"}}
    assert resolve_segmentation_mode("binary", checkpoint) == "binary"


def test_binary_postprocessing_uses_sigmoid():
    logits = torch.tensor([[[[[-2.0, 2.0]]]]])

    outputs = postprocess_logits(logits, "binary", threshold=0.5)

    np.testing.assert_allclose(outputs["pred_prob"], torch.sigmoid(logits)[0, 0].numpy())
    np.testing.assert_array_equal(outputs["pred"], np.array([[[0, 1]]], dtype=np.uint8))


def test_three_class_foreground_probability_sums_shell_and_interior():
    logits = torch.tensor([0.0, np.log(2.0), np.log(3.0)]).reshape(1, 3, 1, 1, 1)

    outputs = postprocess_logits(logits, "three_class_shell_interior", threshold=0.8)

    np.testing.assert_allclose(outputs["pred_prob"], np.array([[[5.0 / 6.0]]]), rtol=1e-6)
    assert np.max(outputs["pred_prob"]) <= 1.0
    np.testing.assert_array_equal(outputs["pred"], np.ones((1, 1, 1), dtype=np.uint8))
    np.testing.assert_array_equal(outputs["pred_class"], np.full((1, 1, 1), 2, dtype=np.uint8))
    np.testing.assert_array_equal(outputs["pred_shell"], np.zeros((1, 1, 1), dtype=np.uint8))
    np.testing.assert_array_equal(outputs["pred_interior"], np.ones((1, 1, 1), dtype=np.uint8))


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("sample.nii.gz", "sample"),
        ("sample.nii", "sample"),
        ("sample.npy", "sample"),
        ("sample.raw", "sample"),
    ],
)
def test_default_output_prefix_removes_complete_data_extension(filename, expected):
    assert default_output_prefix(Path(filename)) == expected


@pytest.mark.parametrize("prefix", ["../escape", "/absolute", ".", "..", ""])
def test_output_prefix_must_be_one_safe_component(prefix):
    with pytest.raises(ValueError, match="filename component"):
        validate_output_prefix(prefix)


def test_output_prefix_accepts_canonical_patch_name():
    assert validate_output_prefix("Human_NEFH_GM") == "Human_NEFH_GM"


def test_percentile_normalization_matches_clip_and_min_max():
    patch = np.arange(1000, dtype=np.float32).reshape(10, 10, 10)
    normalized, metadata = normalize_patch(patch)
    p_lo, p_hi = np.percentile(patch, [0.5, 99.5])
    expected = (np.clip(patch, p_lo, p_hi) - p_lo) / (p_hi - p_lo)

    assert normalized.dtype == np.float32
    np.testing.assert_allclose(normalized, expected.astype(np.float32))
    assert metadata["method"] == "percentile"
    assert metadata["percentiles"] == [0.5, 99.5]
    np.testing.assert_allclose(metadata["clip_values"], [p_lo, p_hi])


def test_percentile_normalization_maps_constant_patch_to_zero():
    normalized, _ = normalize_patch(np.full((2, 2, 2), 7, dtype=np.uint16))
    np.testing.assert_array_equal(normalized, np.zeros((2, 2, 2), dtype=np.float32))


def test_output_affine_preserves_nifti_affine_and_copies_it():
    source = np.array(
        [
            [0.4, 0.0, 0.0, 10.0],
            [0.0, -0.5, 0.0, 20.0],
            [0.0, 0.0, 0.6, 30.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    affine = output_affine(source, voxel_size=9.0)

    np.testing.assert_array_equal(affine, source)
    assert affine is not source


def test_npy_input_uses_voxel_size_affine(tmp_path):
    input_path = tmp_path / "patch.npy"
    np.save(input_path, np.zeros((2, 3, 4), dtype=np.uint8))

    loaded = load_patch(input_path, voxel_size=0.8)

    assert loaded.source_format == "npy"
    assert loaded.header is None
    np.testing.assert_array_equal(loaded.affine, np.diag([0.8, 0.8, 0.8, 1.0]))


def test_raw_input_uses_voxel_size_affine(tmp_path):
    input_path = tmp_path / "patch.raw"
    patch = np.arange(24, dtype=np.uint16).reshape(2, 3, 4)
    input_path.write_bytes(patch.tobytes())
    Path(f"{input_path}.json").write_text(
        json.dumps({"dtype": "uint16", "shape": [2, 3, 4]}),
        encoding="utf-8",
    )

    loaded = load_patch(input_path, voxel_size=1.25)

    assert loaded.source_format == "raw"
    assert loaded.header is None
    np.testing.assert_array_equal(loaded.data, patch)
    np.testing.assert_array_equal(loaded.affine, np.diag([1.25, 1.25, 1.25, 1.0]))

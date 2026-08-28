from argparse import ArgumentTypeError, Namespace
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from inference.infer_lsm import (
    additional_threshold_metadata,
    build_output_paths,
    default_output_prefix,
    load_patch,
    normalize_patch,
    output_affine,
    parse_additional_threshold,
    parse_args,
    postprocess_logits,
    reorder_loaded_patch,
    resolve_segmentation_mode,
    sliding_window_device_options,
    validate_axis_order,
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

    assert set(outputs) == {
        "pred_prob",
        "pred",
        "pred_class",
        "pred_shell",
        "pred_interior",
    }
    np.testing.assert_allclose(outputs["pred_prob"], np.array([[[5.0 / 6.0]]]), rtol=1e-6)
    assert np.max(outputs["pred_prob"]) <= 1.0
    np.testing.assert_array_equal(outputs["pred"], np.ones((1, 1, 1), dtype=np.uint8))
    np.testing.assert_array_equal(outputs["pred_class"], np.full((1, 1, 1), 2, dtype=np.uint8))
    np.testing.assert_array_equal(outputs["pred_shell"], np.zeros((1, 1, 1), dtype=np.uint8))
    np.testing.assert_array_equal(outputs["pred_interior"], np.ones((1, 1, 1), dtype=np.uint8))


def test_binary_focused_uses_one_foreground_probability_for_all_masks():
    logits = torch.tensor([0.0, np.log(2.0), np.log(3.0)]).reshape(1, 3, 1, 1, 1)

    outputs = postprocess_logits(
        logits,
        "three_class_shell_interior",
        threshold=0.8,
        output_profile="binary_focused",
        additional_thresholds={"0p80": 0.8, "0p94": 0.94},
    )

    assert set(outputs) == {
        "pred_prob",
        "pred",
        "pred_threshold_0p80",
        "pred_threshold_0p94",
    }
    np.testing.assert_allclose(outputs["pred_prob"], np.array([[[5.0 / 6.0]]]), rtol=1e-6)
    np.testing.assert_array_equal(outputs["pred"], outputs["pred_threshold_0p80"])
    np.testing.assert_array_equal(
        outputs["pred_threshold_0p94"], np.zeros((1, 1, 1), dtype=np.uint8)
    )


def test_new_inference_options_are_opt_in():
    args = parse_args(
        [
            "--input",
            "input.npy",
            "--checkpoint",
            "model.pt",
            "--output-dir",
            "outputs",
        ]
    )

    assert args.stitch_device == "model"
    assert args.output_profile == "full"
    assert args.additional_thresholds == {}
    assert args.no_save_input is False
    assert args.output_axis_order is None


def test_repeatable_additional_thresholds_are_parsed_in_order():
    args = parse_args(
        [
            "--input",
            "input.npy",
            "--checkpoint",
            "model.pt",
            "--output-dir",
            "outputs",
            "--stitch-device",
            "cpu",
            "--output-profile",
            "binary_focused",
            "--additional-threshold",
            "0p80=0.8",
            "--additional-threshold",
            "0p94=0.94",
        ]
    )

    assert args.stitch_device == "cpu"
    assert args.output_profile == "binary_focused"
    assert args.additional_thresholds == {"0p80": 0.8, "0p94": 0.94}


@pytest.mark.parametrize(
    "value",
    [
        "missing_equals",
        "two=equals=0.5",
        "../escape=0.5",
        "has.dot=0.5",
        "high=1.01",
        "nan=nan",
        "empty=",
        "word=nope",
    ],
)
def test_additional_threshold_rejects_unsafe_labels_and_invalid_values(value):
    with pytest.raises(
        ArgumentTypeError, match="additional-threshold|between 0 and 1"
    ):
        parse_additional_threshold(value)


def test_duplicate_additional_threshold_labels_are_rejected():
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--input",
                "input.npy",
                "--checkpoint",
                "model.pt",
                "--output-dir",
                "outputs",
                "--additional-threshold",
                "strict=0.8",
                "--additional-threshold",
                "strict=0.9",
            ]
        )


def test_binary_focused_output_paths_include_threshold_masks_only(tmp_path):
    paths = build_output_paths(
        tmp_path,
        "volume",
        "three_class_shell_interior",
        output_profile="binary_focused",
        additional_thresholds={"0p94": 0.94},
    )

    assert set(paths) == {
        "input",
        "pred_prob",
        "pred",
        "pred_threshold_0p94",
        "metadata",
    }
    assert paths["pred_threshold_0p94"].name == "volume_pred_threshold_0p94.nii.gz"


def test_no_save_input_omits_normalized_input_path(tmp_path):
    args = parse_args(
        [
            "--input",
            "input.npy",
            "--checkpoint",
            "model.pt",
            "--output-dir",
            "outputs",
            "--no-save-input",
        ]
    )
    paths = build_output_paths(
        tmp_path,
        "volume",
        "three_class_shell_interior",
        save_input=not args.no_save_input,
    )

    assert "input" not in paths
    assert set(paths) == {
        "pred_prob",
        "pred",
        "pred_class",
        "pred_shell",
        "pred_interior",
        "metadata",
    }


def test_output_axis_order_is_validated_and_parsed():
    args = parse_args(
        [
            "--input",
            "input.npy",
            "--checkpoint",
            "model.pt",
            "--output-dir",
            "outputs",
            "--output-axis-order",
            "1",
            "2",
            "0",
        ]
    )

    assert args.output_axis_order == (1, 2, 0)


@pytest.mark.parametrize("axis_order", [(0, 0, 1), (0, 1, 3), (0, 1)])
def test_output_axis_order_must_be_a_permutation(axis_order):
    with pytest.raises(ValueError, match="permutation"):
        validate_axis_order(axis_order)


def test_reorder_loaded_patch_updates_shape_affine_and_zooms(tmp_path):
    input_path = tmp_path / "patch.nii.gz"
    source = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    source_affine = np.diag([0.25, 0.5, 0.75, 1.0])
    nib = pytest.importorskip("nibabel")
    image = nib.Nifti1Image(source, source_affine)
    image.set_qform(source_affine, code=1)
    image.set_sform(source_affine, code=2)
    nib.save(image, input_path)

    reordered = reorder_loaded_patch(load_patch(input_path, voxel_size=1.0), (1, 2, 0))

    np.testing.assert_array_equal(reordered.data, source.transpose(1, 2, 0))
    assert reordered.data.shape == (3, 4, 2)
    np.testing.assert_allclose(reordered.header.get_zooms()[:3], (0.5, 0.75, 0.25))
    expected_transform = np.array(
        [
            [0.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    np.testing.assert_allclose(reordered.affine, source_affine @ expected_transform)
    np.testing.assert_allclose(reordered.header.get_qform(), reordered.affine)
    np.testing.assert_allclose(reordered.header.get_sform(), reordered.affine)


def test_additional_threshold_metadata_records_settings_and_resolved_paths(tmp_path):
    thresholds = {"0p80": 0.8, "0p94": 0.94}
    paths = build_output_paths(
        tmp_path,
        "volume",
        "three_class_shell_interior",
        output_profile="binary_focused",
        additional_thresholds=thresholds,
    )

    metadata = additional_threshold_metadata(thresholds, paths)

    assert metadata == [
        {
            "label": "0p80",
            "threshold": 0.8,
            "path": str(paths["pred_threshold_0p80"].resolve()),
        },
        {
            "label": "0p94",
            "threshold": 0.94,
            "path": str(paths["pred_threshold_0p94"].resolve()),
        },
    ]


@pytest.mark.parametrize("model_device", [torch.device("cpu"), torch.device("cuda")])
def test_cpu_stitch_keeps_input_and_output_on_cpu(model_device):
    input_device, options = sliding_window_device_options("cpu", model_device)

    assert input_device == torch.device("cpu")
    assert options == {"sw_device": model_device, "device": torch.device("cpu")}


def test_model_stitch_preserves_existing_monai_device_defaults():
    model_device = torch.device("cuda")

    input_device, options = sliding_window_device_options("model", model_device)

    assert input_device == model_device
    assert options == {}


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

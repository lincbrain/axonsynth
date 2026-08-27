import json

import nibabel as nib
import numpy as np
import pytest

from inference.prepare_lsm_full import (
    center_aligned_target_affine,
    index_transform_for_axis_order,
    metadata_path_for_output,
    prepare_lsm_full,
    target_shape_for_spacing,
)


def test_target_shape_preserves_endpoint_extent():
    target_shape = target_shape_for_spacing(
        source_shape=(3, 4, 5),
        source_spacing=(1.6, 0.4, 0.8),
        target_spacing=0.8,
    )

    assert target_shape == (5, 3, 5)


def test_axis_transform_is_exact_and_target_affine_preserves_center():
    source_affine = np.array(
        [
            [0.0, -2.0, 0.0, 10.0],
            [1.0, 0.0, 0.0, 20.0],
            [0.0, 0.0, -3.0, 30.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    expected_transform = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    transform = index_transform_for_axis_order((2, 0, 1))
    np.testing.assert_array_equal(transform, expected_transform)
    model_affine = source_affine @ transform
    target_affine = center_aligned_target_affine(
        model_affine,
        source_shape=(3, 5, 7),
        target_shape=(7, 3, 4),
        target_spacing=0.8,
    )

    expected_directions = model_affine[:3, :3].copy()
    expected_directions /= np.linalg.norm(expected_directions, axis=0)
    np.testing.assert_allclose(target_affine[:3, :3], expected_directions * 0.8)
    source_center_world = nib.affines.apply_affine(model_affine, (1.0, 2.0, 3.0))
    target_center_world = nib.affines.apply_affine(target_affine, (3.0, 1.0, 1.5))
    np.testing.assert_allclose(target_center_world, source_center_world)


def test_non_micron_source_units_require_and_record_external_assumption(tmp_path):
    input_path = tmp_path / "unknown_units.nii.gz"
    output_path = tmp_path / "prepared.nii.gz"
    nib.save(
        nib.Nifti1Image(np.ones((2, 2, 2), dtype=np.float32), np.eye(4)),
        input_path,
    )

    with pytest.raises(ValueError, match="--assume-source-units-micron"):
        prepare_lsm_full(input_path, output_path)

    metadata = prepare_lsm_full(
        input_path,
        output_path,
        assume_source_units_micron=True,
    )

    assert metadata["source_units"]["reported_spatial"] == "unknown"
    assert metadata["source_units"]["external_assumption_flag_provided"] is True
    assert metadata["source_units"]["external_assumption_applied"] is True


def test_small_end_to_end_full_volume_preparation(tmp_path):
    input_path = tmp_path / "source.nii.gz"
    output_path = tmp_path / "nested" / "prepared.nii.gz"
    source_data = np.arange(4 * 5 * 3, dtype=np.uint16).reshape(4, 5, 3)
    source_affine = np.array(
        [
            [0.4, 0.0, 0.0, 10.0],
            [0.0, -0.8, 0.0, 20.0],
            [0.0, 0.0, 1.6, 30.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    source_image = nib.Nifti1Image(source_data, source_affine)
    source_image.header.set_xyzt_units(xyz="micron")
    nib.save(source_image, input_path)

    metadata = prepare_lsm_full(input_path, output_path, target_spacing=0.8)

    assert output_path.is_file()
    metadata_path = metadata_path_for_output(output_path)
    assert metadata_path.is_file()
    output_image = nib.load(output_path)
    assert output_image.shape == (5, 3, 5)
    assert output_image.get_data_dtype() == np.dtype(np.float32)
    assert output_image.header.get_xyzt_units()[0] == "micron"
    assert int(output_image.header["qform_code"]) == 1
    assert int(output_image.header["sform_code"]) == 2
    assert np.isfinite(np.asarray(output_image.dataobj)).all()

    recorded = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert recorded == metadata
    assert recorded["source_shape"] == [4, 5, 3]
    assert recorded["model_order_shape"] == [3, 4, 5]
    assert recorded["target_shape"] == [5, 3, 5]
    assert recorded["target_dtype"] == "float32"
    assert recorded["axis_reorder"] == [2, 0, 1]
    np.testing.assert_allclose(
        recorded["anti_alias_sigma_source_voxels"],
        [0.0, 0.5, 0.0],
    )
    source_center = nib.affines.apply_affine(
        np.asarray(recorded["model_order_affine"]),
        (1.0, 1.5, 2.0),
    )
    target_center = nib.affines.apply_affine(output_image.affine, (2.0, 1.0, 2.0))
    np.testing.assert_allclose(target_center, source_center)

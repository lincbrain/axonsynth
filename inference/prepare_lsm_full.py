#!/usr/bin/env python3
"""Prepare a high-memory full LSM NIfTI volume for model inference."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np
from scipy.ndimage import gaussian_filter


DEFAULT_TARGET_SPACING = 0.8
DEFAULT_AXIS_ORDER = (2, 0, 1)
INTENSITY_PERCENTILES = (0.5, 50.0, 99.5)


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be finite and greater than 0")
    return parsed


def validate_axis_order(axis_order: Sequence[int]) -> tuple[int, int, int]:
    """Return a validated permutation mapping model axes to source axes."""

    order = tuple(int(axis) for axis in axis_order)
    if len(order) != 3 or sorted(order) != [0, 1, 2]:
        raise ValueError(
            f"axis_order must be a permutation of (0, 1, 2), got {order}"
        )
    return order


def index_transform_for_axis_order(axis_order: Sequence[int]) -> np.ndarray:
    """Map reordered model voxel indices back to source voxel indices."""

    order = validate_axis_order(axis_order)
    transform = np.zeros((4, 4), dtype=np.float64)
    transform[3, 3] = 1.0
    for model_axis, source_axis in enumerate(order):
        transform[source_axis, model_axis] = 1.0
    return transform


def target_shape_for_spacing(
    source_shape: Sequence[int],
    source_spacing: Sequence[float],
    target_spacing: float,
) -> tuple[int, int, int]:
    """Preserve the distance between first and last voxel centers per axis."""

    shape = tuple(int(size) for size in source_shape)
    spacing = tuple(float(value) for value in source_spacing)
    if len(shape) != 3 or any(size < 1 for size in shape):
        raise ValueError(f"source_shape must contain three positive sizes, got {shape}")
    if len(spacing) != 3 or any(
        not math.isfinite(value) or value <= 0.0 for value in spacing
    ):
        raise ValueError(f"source_spacing must contain three positive finite values, got {spacing}")
    if not math.isfinite(target_spacing) or target_spacing <= 0.0:
        raise ValueError(
            f"target_spacing must be finite and greater than 0, got {target_spacing}"
        )
    return tuple(
        int(round((size - 1) * axis_spacing / target_spacing)) + 1
        for size, axis_spacing in zip(shape, spacing)
    )


def center_aligned_target_affine(
    source_affine: np.ndarray,
    source_shape: Sequence[int],
    target_shape: Sequence[int],
    target_spacing: float,
) -> np.ndarray:
    """Preserve axis directions and align source and target voxel-grid centers."""

    affine = np.asarray(source_affine, dtype=np.float64)
    source_shape_array = np.asarray(tuple(source_shape), dtype=np.float64)
    target_shape_array = np.asarray(tuple(target_shape), dtype=np.float64)
    if affine.shape != (4, 4) or not np.isfinite(affine).all():
        raise ValueError("source_affine must be a finite 4x4 matrix")
    if source_shape_array.shape != (3,) or np.any(source_shape_array < 1):
        raise ValueError("source_shape must contain three positive sizes")
    if target_shape_array.shape != (3,) or np.any(target_shape_array < 1):
        raise ValueError("target_shape must contain three positive sizes")
    if not math.isfinite(target_spacing) or target_spacing <= 0.0:
        raise ValueError("target_spacing must be finite and greater than 0")

    linear = affine[:3, :3]
    direction_norms = np.linalg.norm(linear, axis=0)
    determinant = float(np.linalg.det(linear))
    if (
        not np.isfinite(direction_norms).all()
        or np.any(direction_norms <= 0.0)
        or not math.isfinite(determinant)
        or determinant == 0.0
    ):
        raise ValueError("source_affine must contain three invertible axis directions")

    target_affine = np.eye(4, dtype=np.float64)
    target_affine[:3, :3] = linear / direction_norms * target_spacing
    source_center = (source_shape_array - 1.0) / 2.0
    target_center = (target_shape_array - 1.0) / 2.0
    center_world = nib.affines.apply_affine(affine, source_center)
    target_affine[:3, 3] = center_world - target_affine[:3, :3] @ target_center
    return target_affine


def intensity_summary(data: np.ndarray) -> dict[str, float]:
    percentiles = np.percentile(data, INTENSITY_PERCENTILES)
    return {
        "min": float(np.min(data)),
        "p0_5": float(percentiles[0]),
        "median": float(percentiles[1]),
        "p99_5": float(percentiles[2]),
        "max": float(np.max(data)),
    }


def metadata_path_for_output(output_path: Path) -> Path:
    lower_name = output_path.name.lower()
    if lower_name.endswith(".nii.gz"):
        stem = output_path.name[:-7]
    elif lower_name.endswith(".nii"):
        stem = output_path.name[:-4]
    else:
        raise ValueError(f"Expected a .nii or .nii.gz output path, got {output_path}")
    return output_path.with_name(f"{stem}_prep.json")


def _nifti_suffix(path: Path) -> str:
    lower_name = path.name.lower()
    if lower_name.endswith(".nii.gz"):
        return ".nii.gz"
    if lower_name.endswith(".nii"):
        return ".nii"
    raise ValueError(f"Expected a .nii or .nii.gz path, got {path}")


def save_nifti_atomic(image: nib.Nifti1Image, path: Path) -> None:
    suffix = _nifti_suffix(path)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=suffix,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        nib.save(image, str(temporary))
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def prepare_lsm_full(
    input_path: Path,
    output_path: Path,
    target_spacing: float = DEFAULT_TARGET_SPACING,
    axis_order: Sequence[int] = DEFAULT_AXIS_ORDER,
    assume_source_units_micron: bool = False,
) -> dict[str, Any]:
    """Prepare one complete NIfTI volume and return its recorded metadata."""

    input_path = Path(input_path)
    output_path = Path(output_path)
    _nifti_suffix(input_path)
    _nifti_suffix(output_path)
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if input_path.resolve() == output_path.resolve():
        raise ValueError("input and output paths must be different")
    order = validate_axis_order(axis_order)
    if not math.isfinite(target_spacing) or target_spacing <= 0.0:
        raise ValueError("target_spacing must be finite and greater than 0")

    source_image = nib.load(str(input_path))
    if len(source_image.shape) != 3:
        raise ValueError(f"Expected a 3D source image, got shape {source_image.shape}")
    source_shape = tuple(int(size) for size in source_image.shape)
    if any(size < 1 for size in source_shape):
        raise ValueError(f"Source image is empty: shape {source_shape}")

    source_zooms = tuple(float(value) for value in source_image.header.get_zooms()[:3])
    if any(not math.isfinite(value) or value <= 0.0 for value in source_zooms):
        raise ValueError(f"Invalid source voxel spacing: {source_zooms}")
    reported_spatial_unit, reported_temporal_unit = source_image.header.get_xyzt_units()
    assumption_applied = reported_spatial_unit != "micron"
    if assumption_applied and not assume_source_units_micron:
        raise ValueError(
            "Source spatial units are "
            f"{reported_spatial_unit!r}, not 'micron'; pass "
            "--assume-source-units-micron to explicitly treat header zoom values as microns"
        )

    source_affine = np.asarray(source_image.affine, dtype=np.float64)
    index_transform = index_transform_for_axis_order(order)
    model_order_affine = source_affine @ index_transform
    model_order_zooms = tuple(source_zooms[source_axis] for source_axis in order)
    model_order_shape = tuple(source_shape[source_axis] for source_axis in order)
    target_shape = target_shape_for_spacing(
        model_order_shape,
        model_order_zooms,
        target_spacing,
    )
    target_affine = center_aligned_target_affine(
        model_order_affine,
        model_order_shape,
        target_shape,
        target_spacing,
    )

    print(
        f"Loading full source proxy {input_path} shape={source_shape} "
        f"dtype={source_image.get_data_dtype()}",
        flush=True,
    )
    source_data = np.asarray(source_image.dataobj, dtype=np.float32)
    if not np.isfinite(source_data).all():
        raise ValueError("Source image contains non-finite values")
    source_intensity = intensity_summary(source_data)
    model_order_data = np.array(
        source_data.transpose(order),
        dtype=np.float32,
        order="C",
        copy=True,
    )
    del source_data

    anti_alias_sigma = tuple(
        max((target_spacing / source_spacing) - 1.0, 0.0) / 2.0
        for source_spacing in model_order_zooms
    )
    print(
        f"Reordered shape={model_order_shape}; anti-alias sigma={anti_alias_sigma}",
        flush=True,
    )
    gaussian_filter(
        model_order_data,
        sigma=anti_alias_sigma,
        mode="reflect",
        output=model_order_data,
    )

    model_order_image = nib.Nifti1Image(model_order_data, model_order_affine)
    resampled_image = resample_from_to(
        model_order_image,
        (target_shape, target_affine),
        order=1,
        mode="nearest",
    )
    target_data = np.ascontiguousarray(
        np.asarray(resampled_image.dataobj, dtype=np.float32)
    )
    if target_data.shape != target_shape or not np.isfinite(target_data).all():
        raise RuntimeError("Resampled target image has invalid shape or values")

    output_image = nib.Nifti1Image(target_data, target_affine)
    output_image.header.set_data_dtype(np.float32)
    output_image.header.set_slope_inter(1.0, 0.0)
    output_image.header.set_xyzt_units(xyz="micron")
    output_image.set_qform(target_affine, code=1)
    output_image.set_sform(target_affine, code=2)

    metadata_path = metadata_path_for_output(output_path)
    target_zooms = tuple(float(value) for value in nib.affines.voxel_sizes(target_affine))
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "input": str(input_path.resolve()),
        "output": str(output_path.resolve()),
        "source_shape": list(source_shape),
        "source_dtype": str(source_image.get_data_dtype()),
        "source_loaded_dtype": str(np.dtype(np.float32)),
        "source_affine": source_affine.tolist(),
        "source_zooms_reported": list(source_zooms),
        "source_zooms_microns": list(source_zooms),
        "source_orientation": "".join(
            code if code is not None else "?" for code in nib.aff2axcodes(source_affine)
        ),
        "source_units": {
            "reported_spatial": reported_spatial_unit,
            "reported_temporal": reported_temporal_unit,
            "external_assumption_flag_provided": bool(assume_source_units_micron),
            "external_assumption_applied": assumption_applied,
            "external_assumption": (
                "header zoom values are microns despite the reported spatial unit"
                if assumption_applied
                else None
            ),
            "zooms_interpreted_as": "micron",
        },
        "source_intensity": source_intensity,
        "axis_reorder": list(order),
        "index_transform_model_to_source": index_transform.tolist(),
        "model_order_shape": list(model_order_shape),
        "model_order_affine": model_order_affine.tolist(),
        "model_order_zooms_microns": list(model_order_zooms),
        "model_order_orientation": "".join(
            code if code is not None else "?"
            for code in nib.aff2axcodes(model_order_affine)
        ),
        "anti_alias_sigma_source_voxels": list(anti_alias_sigma),
        "target_shape": list(target_shape),
        "target_dtype": str(target_data.dtype),
        "target_affine": target_affine.tolist(),
        "target_zooms_microns": list(target_zooms),
        "target_orientation": "".join(
            code if code is not None else "?" for code in nib.aff2axcodes(target_affine)
        ),
        "target_intensity": intensity_summary(target_data),
        "settings": {
            "target_spacing_microns": float(target_spacing),
            "axis_order_model_from_source": list(order),
            "source_units_assumption_flag": bool(assume_source_units_micron),
            "anti_alias_sigma_formula": "max((target/source)-1,0)/2",
            "anti_alias_mode": "reflect",
            "target_shape_formula": "round((n-1)*source_spacing/target_spacing)+1",
            "target_affine_alignment": "direction-preserving, center-aligned",
            "interpolation": "linear",
            "interpolation_order": 1,
            "boundary_mode": "nearest",
            "output_spatial_units": "micron",
            "qform_code": 1,
            "sform_code": 2,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.unlink(missing_ok=True)
    save_nifti_atomic(output_image, output_path)
    write_json_atomic(metadata_path, metadata)
    print(
        f"Saved {output_path} shape={target_shape} spacing={target_spacing} microns",
        flush=True,
    )
    return metadata


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Source .nii or .nii.gz")
    parser.add_argument("--output", type=Path, required=True, help="Prepared .nii or .nii.gz")
    parser.add_argument(
        "--target-spacing",
        type=_positive_float,
        default=DEFAULT_TARGET_SPACING,
        help="Isotropic target spacing in microns (default: 0.8)",
    )
    parser.add_argument(
        "--axis-order",
        type=int,
        nargs=3,
        default=DEFAULT_AXIS_ORDER,
        metavar=("A0", "A1", "A2"),
        help="Source axes assigned to model axes (default: 2 0 1)",
    )
    parser.add_argument(
        "--assume-source-units-micron",
        action="store_true",
        help="Treat source header zoom values as microns when header units are not micron",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    prepare_lsm_full(
        args.input,
        args.output,
        target_spacing=args.target_spacing,
        axis_order=args.axis_order,
        assume_source_units_micron=args.assume_source_units_micron,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

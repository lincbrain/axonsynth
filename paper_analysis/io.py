"""NIfTI and configuration I/O for accepted paper baselines."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import nibabel as nib
import numpy as np


@dataclass(frozen=True)
class BaselineOutputPaths:
    """Files written for one case and baseline."""

    score: Path
    mask: Path
    metadata: Path


def _temporary_sibling(path: Path, suffix: str = ".tmp") -> Path:
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=suffix,
    )
    os.close(descriptor)
    return Path(temporary)


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a JSON or YAML baseline manifest."""
    config_path = Path(path)
    suffix = config_path.suffix.lower()
    if suffix == ".json":
        document = json.loads(config_path.read_text(encoding="utf-8"))
    elif suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - depends on installation
            raise RuntimeError("YAML configs require PyYAML") from exc
        document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    else:
        raise ValueError("Baseline config must use .json, .yaml, or .yml")

    if not isinstance(document, Mapping):
        raise ValueError("Baseline config must contain a mapping at its root")
    return dict(document)


def is_nifti_path(path: str | Path) -> bool:
    """Return whether a path has a NIfTI filename extension."""
    name = Path(path).name.lower()
    return name.endswith(".nii") or name.endswith(".nii.gz")


def load_nifti(
    path: str | Path,
) -> tuple[np.ndarray, nib.spatialimages.SpatialImage]:
    """Load a NIfTI array and retain its image as output geometry reference."""
    input_path = Path(path)
    if not is_nifti_path(input_path):
        raise ValueError(f"Expected a .nii or .nii.gz input, got {input_path}")
    image = nib.load(str(input_path))
    data = np.asanyarray(image.dataobj)
    if data.size == 0:
        raise ValueError(f"NIfTI input is empty: {input_path}")
    return data, image


def save_nifti_like(
    data: np.ndarray,
    path: str | Path,
    reference: nib.spatialimages.SpatialImage,
) -> Path:
    """Save ``data`` as NIfTI while preserving the reference affine."""
    output_path = Path(path)
    if not is_nifti_path(output_path):
        raise ValueError(f"NIfTI output must end in .nii or .nii.gz: {output_path}")

    array = np.asarray(data)
    if array.shape != reference.shape:
        raise ValueError(
            f"Output shape {array.shape} does not match input shape {reference.shape}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    header = reference.header.copy()
    header.set_data_dtype(array.dtype)
    header.set_slope_inter(None, None)
    image_type = (
        nib.Nifti2Image
        if isinstance(reference, nib.Nifti2Image)
        else nib.Nifti1Image
    )
    output = image_type(array, np.array(reference.affine, copy=True), header=header)
    suffix = ".nii.gz" if output_path.name.lower().endswith(".nii.gz") else ".nii"
    temporary = _temporary_sibling(output_path, suffix)
    try:
        nib.save(output, str(temporary))
        temporary.replace(output_path)
    finally:
        temporary.unlink(missing_ok=True)
    return output_path


def write_json(path: str | Path, document: Mapping[str, Any]) -> Path:
    """Write deterministic, standards-compliant JSON."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_sibling(output_path)
    try:
        temporary.write_text(
            json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output_path)
    finally:
        temporary.unlink(missing_ok=True)
    return output_path


def baseline_output_paths(
    output_dir: str | Path,
    case_id: str,
    baseline: str,
) -> BaselineOutputPaths:
    """Return deterministic output paths for a case/baseline pair."""
    base = f"{case_id}_{baseline}"
    directory = Path(output_dir)
    return BaselineOutputPaths(
        score=directory / f"{base}_score.nii.gz",
        mask=directory / f"{base}_mask.nii.gz",
        metadata=directory / f"{base}_metadata.json",
    )


def save_baseline_outputs(
    score: np.ndarray,
    mask: np.ndarray,
    reference: nib.spatialimages.SpatialImage,
    output_dir: str | Path,
    case_id: str,
    baseline: str,
    metadata: Mapping[str, Any],
) -> BaselineOutputPaths:
    """Write score/mask NIfTIs and their shared JSON metadata sidecar."""
    paths = baseline_output_paths(output_dir, case_id, baseline)
    save_nifti_like(np.asarray(score, dtype=np.float32), paths.score, reference)
    save_nifti_like(np.asarray(mask, dtype=np.uint8), paths.mask, reference)

    complete_metadata = dict(metadata)
    complete_metadata["outputs"] = {
        "score": str(paths.score),
        "mask": str(paths.mask),
    }
    write_json(paths.metadata, complete_metadata)
    return paths

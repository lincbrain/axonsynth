"""Shared loading and serialization helpers for the fixed paper evaluation."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping

import nibabel as nib
import numpy as np

from paper_analysis.configuration import paper_context


def _temporary_sibling(path: Path) -> Path:
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    return Path(temporary)


def _load_binary_volume(image, *, label: str) -> np.ndarray:
    values = np.asarray(image.dataobj)
    if np.iscomplexobj(values) or not np.isfinite(values).all():
        raise ValueError(f"{label} must contain finite real values")
    unique = np.unique(values)
    if not np.all(np.isin(unique, (0, 1))):
        raise ValueError(f"{label} must use binary 0/1 encoding, got {unique.tolist()}")
    return values.astype(bool, copy=False)


def _validate_case_metadata(config, patch, method, score_path: Path, mask_path: Path) -> None:
    template = "learned_metadata" if method.kind == "learned" else "baseline_metadata"
    metadata_path = config.path(template, patch=patch, method=method)
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing evaluation metadata: {metadata_path}")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid evaluation metadata: {metadata_path}") from error

    if method.kind == "learned":
        context = metadata.get("paper_context", {})
        expected = paper_context(config, patch, method)
        if any(context.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Learned metadata context mismatch for {method.name} {patch.id}")
        if Path(metadata.get("input_path", "")).resolve() != config.path(
            "raw_patch", patch=patch
        ):
            raise ValueError(f"Learned input path mismatch for {method.name} {patch.id}")
        if Path(metadata.get("checkpoint", {}).get("path", "")).resolve() != method.checkpoint:
            raise ValueError(f"Checkpoint path mismatch for {method.name} {patch.id}")
        if metadata.get("resolved_segmentation_mode") != method.segmentation_mode:
            raise ValueError(f"Segmentation mode mismatch for {method.name} {patch.id}")
        inference = metadata.get("inference_parameters", {})
        if inference.get("threshold") != method.threshold_for(patch.domain):
            raise ValueError(f"Inference threshold mismatch for {method.name} {patch.id}")
        expected_inference = {
            "roi_size": [128, 128, 128],
            "sliding_window_batch_size": 4,
            "overlap": 0.5,
            "sliding_window_blend_mode": "gaussian",
            "sliding_window_sigma_scale": 0.125,
            "padding_mode": "constant",
            "padding_cval": 0.0,
            "amp_enabled": True,
            "amp_dtype": "float16",
        }
        if any(inference.get(key) != value for key, value in expected_inference.items()):
            raise ValueError(f"Inference parameter mismatch for {method.name} {patch.id}")
        if metadata.get("device", {}).get("type") != "cuda":
            raise ValueError(f"Canonical learned inference did not use CUDA for {method.name} {patch.id}")
        output_paths = metadata.get("output_paths", {})
        recorded_paths = {
            "score": output_paths.get("pred_prob", ""),
            "mask": output_paths.get("pred", ""),
        }
    else:
        if (
            metadata.get("schema_version") != 3
            or metadata.get("method") != method.name
            or metadata.get("patch_id") != patch.id
            or Path(metadata.get("input", "")).resolve()
            != config.path("raw_patch", patch=patch)
            or Path(metadata.get("config", "")).resolve() != config.source
            or metadata.get("threshold_selection_metric")
            != method.threshold_selection_metric
            or metadata.get("parameters", {}).get("threshold")
            != method.threshold_for(patch.domain)
        ):
            raise ValueError(f"Baseline metadata mismatch for {method.name} {patch.id}")
        recorded_paths = metadata.get("outputs", {})

    for label, path in (("score", score_path), ("mask", mask_path)):
        if Path(recorded_paths.get(label, "")).resolve() != path.resolve():
            raise ValueError(f"Output path mismatch for {method.name} {patch.id} {label}")


def load_method_case(config, patch, method):
    target_path = config.path("target_mask", patch=patch)
    score_path = config.path("prediction_score", patch=patch, method=method)
    mask_path = config.path("prediction_mask", patch=patch, method=method)
    for path in (target_path, score_path, mask_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing evaluation input: {path}")
    _validate_case_metadata(config, patch, method, score_path, mask_path)

    target_image = nib.load(str(target_path))
    score_image = nib.load(str(score_path))
    mask_image = nib.load(str(mask_path))
    if score_image.shape != target_image.shape or mask_image.shape != target_image.shape:
        raise ValueError(
            f"Shape mismatch for {method.name} {patch.id}: "
            f"target={target_image.shape}, score={score_image.shape}, mask={mask_image.shape}"
        )
    if not np.allclose(score_image.affine, target_image.affine, rtol=0.0, atol=1e-5):
        raise ValueError(f"Score affine mismatch for {method.name} {patch.id}")
    if not np.allclose(mask_image.affine, target_image.affine, rtol=0.0, atol=1e-5):
        raise ValueError(f"Mask affine mismatch for {method.name} {patch.id}")

    score = np.asarray(score_image.dataobj, dtype=np.float32)
    if not np.isfinite(score).all():
        raise ValueError(f"Non-finite score values for {method.name} {patch.id}")
    tolerance = np.finfo(np.float32).eps
    if np.any(score < -tolerance) or np.any(score > 1.0 + tolerance):
        raise ValueError(f"Score values must be in [0, 1] for {method.name} {patch.id}")
    score = np.clip(score, 0.0, 1.0)
    target = _load_binary_volume(target_image, label=f"Target {patch.id}")
    saved_mask = _load_binary_volume(
        mask_image, label=f"Saved mask {method.name} {patch.id}"
    )
    threshold = method.threshold_for(patch.domain)
    prediction = score >= threshold
    if not np.array_equal(prediction, saved_mask):
        raise ValueError(
            f"Saved mask does not match fixed threshold {threshold} for "
            f"{method.name} {patch.id}"
        )
    return {
        "target_image": target_image,
        "score_image": score_image,
        "target": target,
        "score": score,
        "prediction": prediction,
        "threshold": threshold,
        "target_path": target_path,
        "score_path": score_path,
        "mask_path": mask_path,
    }


def flatten_mapping(prefix: str, value: Mapping[str, Any]) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, item in value.items():
        name = f"{prefix}_{key}" if prefix else key
        if isinstance(item, Mapping):
            flattened.update(flatten_mapping(name, item))
        else:
            flattened[name] = item
    return flattened


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    materialized = [dict(row) for row in rows]
    if not materialized:
        raise ValueError(f"Refusing to write empty table: {path}")
    identifiers = ["patch_id", "dataset", "patch", "domain", "split", "method"]
    all_fields = {key for row in materialized for key in row}
    fields = [name for name in identifiers if name in all_fields]
    fields.extend(sorted(all_fields - set(fields)))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_sibling(path)
    try:
        with temporary.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(materialized)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_sibling(path)
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_sibling(path)
    try:
        temporary.write_text(value, encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)

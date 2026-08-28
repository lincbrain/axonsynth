#!/usr/bin/env python3
"""Run the accepted 3D MONAI UNet inference pipeline on one LSM volume.

Checkpoints are full PyTorch training checkpoints and must come from a trusted
source because loading them uses Python pickle deserialization.
"""

import argparse
import importlib.metadata
import json
import math
import os
import platform
import re
import shlex
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


SEGMENTATION_MODES = ("binary", "three_class_shell_interior")
STITCH_DEVICES = ("model", "cpu")
OUTPUT_PROFILES = ("full", "binary_focused")
NORMALIZATION_PERCENTILES = (0.5, 99.5)
MODEL_CHANNELS = (16, 32, 64, 128, 256)
MODEL_STRIDES = (2, 2, 2, 2)
MODEL_NUM_RES_UNITS = 2
MODEL_DROPOUT = 0.1
SLIDING_WINDOW_BLEND_MODE = "gaussian"
SLIDING_WINDOW_PADDING_MODE = "constant"
SLIDING_WINDOW_CVAL = 0.0
SLIDING_WINDOW_SIGMA_SCALE = 0.125
ADDITIONAL_THRESHOLD_LABEL_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")


@dataclass
class LoadedPatch:
    """Image data and the spatial metadata to use for generated NIfTI files."""

    data: np.ndarray
    affine: np.ndarray
    header: Any | None
    source_format: str


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be finite and greater than 0")
    return parsed


def _probability(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be a number between 0 and 1") from exc
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("value must be between 0 and 1")
    return parsed


def _overlap(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed < 1.0:
        raise argparse.ArgumentTypeError("value must be in [0, 1)")
    return parsed


def validate_additional_threshold_label(label: str) -> str:
    if not ADDITIONAL_THRESHOLD_LABEL_PATTERN.fullmatch(label):
        raise ValueError(
            "additional-threshold label must start with an ASCII letter or digit "
            "and contain only ASCII letters, digits, underscores, and hyphens"
        )
    return label


def parse_additional_threshold(value: str) -> tuple[str, float]:
    """Parse one safe output label and probability threshold."""

    if value.count("=") != 1:
        raise argparse.ArgumentTypeError("additional-threshold must be LABEL=VALUE")
    label, threshold_text = value.split("=", 1)
    try:
        validate_additional_threshold_label(label)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    return label, _probability(threshold_text)


def collect_additional_thresholds(
    values: Sequence[tuple[str, float]],
) -> dict[str, float]:
    """Collect parsed thresholds while rejecting ambiguous duplicate labels."""

    thresholds: dict[str, float] = {}
    for label, threshold in values:
        if label in thresholds:
            raise ValueError(f"duplicate additional-threshold label: {label!r}")
        thresholds[label] = threshold
    return thresholds


def validate_axis_order(values: Sequence[int]) -> tuple[int, int, int]:
    """Validate a complete permutation of the three spatial axes."""

    order = tuple(int(value) for value in values)
    if len(order) != 3 or sorted(order) != [0, 1, 2]:
        raise ValueError("output-axis-order must be a permutation of 0 1 2")
    return order


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Axon segmentation on an LSM volume")
    parser.add_argument("--input", type=Path, required=True, help="Path to .npy, .nii/.nii.gz, or raw data")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Trusted PyTorch training checkpoint")
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        dest="output_dir",
        type=Path,
        required=True,
        help="Output directory for NIfTI files and inference metadata",
    )
    parser.add_argument(
        "--segmentation-mode",
        "--segmentation_mode",
        dest="segmentation_mode",
        default="auto",
        choices=("auto", *SEGMENTATION_MODES),
        help="Mode to run; auto reads v2 checkpoint args and otherwise uses legacy binary mode",
    )
    parser.add_argument(
        "--output-prefix",
        "--output_prefix",
        dest="output_prefix",
        default=None,
        help="Output prefix (default: input filename without its data extension)",
    )
    parser.add_argument(
        "--voxel-size",
        "--voxel_size",
        dest="voxel_size",
        type=_positive_float,
        default=1.0,
        help="Isotropic voxel size in microns for NPY/raw input; NIfTI spatial metadata is preserved",
    )
    parser.add_argument(
        "--roi-size",
        "--roi_size",
        dest="roi_size",
        type=_positive_int,
        default=128,
        help="Isotropic sliding-window ROI size",
    )
    parser.add_argument(
        "--sw-batch-size",
        "--sw_batch_size",
        dest="sw_batch_size",
        type=_positive_int,
        default=4,
        help="Sliding-window batch size",
    )
    parser.add_argument("--overlap", type=_overlap, default=0.5, help="Sliding-window overlap")
    parser.add_argument("--threshold", type=_probability, default=0.5, help="Foreground threshold")
    parser.add_argument(
        "--stitch-device",
        choices=STITCH_DEVICES,
        default="model",
        help="Device for stitching sliding-window logits (default: model device)",
    )
    parser.add_argument(
        "--output-profile",
        choices=OUTPUT_PROFILES,
        default="full",
        help="Output set; binary_focused omits three-class argmax/shell/interior files",
    )
    parser.add_argument(
        "--additional-threshold",
        dest="additional_thresholds",
        action="append",
        type=parse_additional_threshold,
        default=[],
        metavar="LABEL=VALUE",
        help="Save an additional binary foreground mask; repeat for multiple thresholds",
    )
    parser.add_argument(
        "--no-save-input",
        action="store_true",
        help="Do not save the normalized input volume",
    )
    parser.add_argument(
        "--output-axis-order",
        type=int,
        nargs=3,
        default=None,
        metavar=("A0", "A1", "A2"),
        help="Transpose generated NIfTI outputs to this axis order",
    )
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Fail instead of falling back to CPU when CUDA is unavailable",
    )
    args = parser.parse_args(argv)
    try:
        args.additional_thresholds = collect_additional_thresholds(args.additional_thresholds)
        if args.output_axis_order is not None:
            args.output_axis_order = validate_axis_order(args.output_axis_order)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def default_output_prefix(input_path: Path) -> str:
    """Remove one data extension, treating .nii.gz as a single extension."""

    if input_path.name.lower().endswith(".nii.gz"):
        return input_path.name[:-7]
    return input_path.stem


def validate_output_prefix(value: str) -> str:
    if not value or value in {".", ".."} or Path(value).name != value:
        raise ValueError("output-prefix must be one safe filename component")
    return value


def output_affine(input_affine: np.ndarray | None, voxel_size: float) -> np.ndarray:
    """Preserve a NIfTI affine or create an isotropic affine for NPY/raw data."""

    if input_affine is not None:
        affine = np.asarray(input_affine)
        if affine.shape != (4, 4):
            raise ValueError(f"Expected a 4x4 input affine, got {affine.shape}")
        return affine.copy()
    if voxel_size <= 0.0:
        raise ValueError(f"voxel_size must be greater than 0, got {voxel_size}")
    return np.diag([voxel_size, voxel_size, voxel_size, 1.0])


def _require_nibabel():
    try:
        import nibabel as nib
    except ModuleNotFoundError as exc:
        raise RuntimeError("nibabel is required to read and write NIfTI files") from exc
    return nib


def load_patch(input_path: Path, voxel_size: float) -> LoadedPatch:
    """Load one 3D patch and establish the spatial metadata for its outputs."""

    lower_name = input_path.name.lower()
    if input_path.suffix.lower() == ".npy":
        patch = np.load(input_path, allow_pickle=False)
        loaded = LoadedPatch(patch, output_affine(None, voxel_size), None, "npy")
    elif input_path.suffix.lower() == ".nii" or lower_name.endswith(".nii.gz"):
        nib = _require_nibabel()
        image = nib.load(str(input_path))
        patch = np.asanyarray(image.dataobj)
        loaded = LoadedPatch(
            patch,
            output_affine(image.affine, voxel_size),
            image.header.copy(),
            "nifti",
        )
    else:
        metadata_path = Path(f"{input_path}.json")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        patch = np.memmap(
            input_path,
            dtype=np.dtype(metadata["dtype"]),
            mode="r",
            shape=tuple(metadata["shape"]),
        )
        loaded = LoadedPatch(patch, output_affine(None, voxel_size), None, "raw")

    if loaded.data.ndim != 3:
        raise ValueError(f"Expected a 3D input volume, got shape {loaded.data.shape}")
    if loaded.data.size == 0:
        raise ValueError("Input volume is empty")
    return loaded


def reorder_loaded_patch(
    loaded: LoadedPatch, axis_order: Sequence[int]
) -> LoadedPatch:
    """Transpose image data and update its affine and NIfTI header."""

    order = validate_axis_order(axis_order)
    data = loaded.data.transpose(order)
    index_transform = np.zeros((4, 4), dtype=np.float64)
    index_transform[3, 3] = 1.0
    for output_axis, input_axis in enumerate(order):
        index_transform[input_axis, output_axis] = 1.0
    affine = np.asarray(loaded.affine, dtype=np.float64) @ index_transform

    header = loaded.header.copy() if loaded.header is not None else None
    if header is not None:
        original_zooms = tuple(float(value) for value in header.get_zooms())
        header.set_data_shape(data.shape)
        header.set_zooms(tuple(original_zooms[axis] for axis in order))
        _, qform_code = loaded.header.get_qform(coded=True)
        _, sform_code = loaded.header.get_sform(coded=True)
        header.set_qform(affine, code=int(qform_code))
        header.set_sform(affine, code=int(sform_code))

    return LoadedPatch(data, affine, header, loaded.source_format)


def normalize_patch(patch: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Clip at [0.5, 99.5] percentiles and min-max normalize to [0, 1]."""

    patch_f = np.asarray(patch, dtype=np.float32)
    p_lo, p_hi = np.percentile(patch_f, NORMALIZATION_PERCENTILES)
    if not np.isfinite(p_lo) or not np.isfinite(p_hi):
        raise ValueError("Normalization percentiles are not finite")

    patch_f = np.clip(patch_f, p_lo, p_hi)
    scale = float(p_hi - p_lo)
    if scale > 0.0:
        patch_f = (patch_f - p_lo) / scale
    else:
        patch_f = np.zeros_like(patch_f, dtype=np.float32)

    normalization = {
        "method": "percentile",
        "percentiles": list(NORMALIZATION_PERCENTILES),
        "clip_values": [float(p_lo), float(p_hi)],
        "output_range": [0.0, 1.0],
    }
    return patch_f.astype(np.float32, copy=False), normalization


def resolve_segmentation_mode(requested_mode: str, checkpoint: Mapping[str, Any]) -> str:
    """Resolve auto to v2 checkpoint metadata, with binary as the legacy default."""

    if requested_mode != "auto":
        if requested_mode not in SEGMENTATION_MODES:
            raise ValueError(f"Unsupported segmentation mode: {requested_mode!r}")
        return requested_mode

    checkpoint_args = checkpoint.get("args")
    if checkpoint_args is None:
        resolved_mode = "binary"
    elif isinstance(checkpoint_args, Mapping):
        resolved_mode = checkpoint_args.get("segmentation_mode", "binary")
    else:
        resolved_mode = getattr(checkpoint_args, "segmentation_mode", "binary")

    if resolved_mode not in SEGMENTATION_MODES:
        raise ValueError(f"Checkpoint contains unsupported segmentation mode: {resolved_mode!r}")
    return resolved_mode


def architecture_metadata(segmentation_mode: str) -> dict[str, Any]:
    if segmentation_mode not in SEGMENTATION_MODES:
        raise ValueError(f"Unsupported segmentation mode: {segmentation_mode!r}")
    return {
        "implementation": "monai.networks.nets.UNet",
        "spatial_dims": 3,
        "in_channels": 1,
        "out_channels": 1 if segmentation_mode == "binary" else 3,
        "channels": list(MODEL_CHANNELS),
        "strides": list(MODEL_STRIDES),
        "num_res_units": MODEL_NUM_RES_UNITS,
        "norm": "BATCH",
        "dropout": MODEL_DROPOUT,
    }


def build_model(device: torch.device, segmentation_mode: str) -> torch.nn.Module:
    from monai.networks.layers import Norm
    from monai.networks.nets import UNet

    metadata = architecture_metadata(segmentation_mode)
    return UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=metadata["out_channels"],
        channels=MODEL_CHANNELS,
        strides=MODEL_STRIDES,
        num_res_units=MODEL_NUM_RES_UNITS,
        norm=Norm.BATCH,
        dropout=MODEL_DROPOUT,
    ).to(device)


def postprocess_logits(
    pred_logits: torch.Tensor,
    segmentation_mode: str,
    threshold: float,
    output_profile: str = "full",
    additional_thresholds: Mapping[str, float] | None = None,
) -> dict[str, np.ndarray]:
    """Convert binary or three-class logits into accepted inference outputs."""

    if output_profile not in OUTPUT_PROFILES:
        raise ValueError(f"Unsupported output profile: {output_profile!r}")

    if segmentation_mode == "binary":
        pred_prob = torch.sigmoid(pred_logits)[0, 0].detach().cpu().numpy()
        outputs = {
            "pred_prob": pred_prob,
            "pred": (pred_prob >= threshold).astype(np.uint8),
        }
    else:
        if segmentation_mode != "three_class_shell_interior":
            raise ValueError(f"Unsupported segmentation mode: {segmentation_mode!r}")

        probabilities = torch.softmax(pred_logits, dim=1)[0].detach().cpu()
        pred_prob = (probabilities[1] + probabilities[2]).clamp_(0.0, 1.0).numpy()
        outputs = {
            "pred_prob": pred_prob,
            "pred": (pred_prob >= threshold).astype(np.uint8),
        }
        if output_profile == "full":
            pred_class = probabilities.argmax(dim=0).to(torch.uint8).numpy()
            outputs.update(
                {
                    "pred_class": pred_class,
                    "pred_shell": (pred_class == 1).astype(np.uint8),
                    "pred_interior": (pred_class == 2).astype(np.uint8),
                }
            )

    for label, additional_threshold in (additional_thresholds or {}).items():
        outputs[f"pred_threshold_{label}"] = (pred_prob >= additional_threshold).astype(
            np.uint8
        )
    return outputs


def build_output_paths(
    output_dir: Path,
    output_prefix: str,
    segmentation_mode: str,
    output_profile: str = "full",
    additional_thresholds: Mapping[str, float] | None = None,
    save_input: bool = True,
) -> dict[str, Path]:
    """Build the complete output manifest for an inference run."""

    if segmentation_mode not in SEGMENTATION_MODES:
        raise ValueError(f"Unsupported segmentation mode: {segmentation_mode!r}")
    if output_profile not in OUTPUT_PROFILES:
        raise ValueError(f"Unsupported output profile: {output_profile!r}")

    paths = {
        "pred_prob": output_dir / f"{output_prefix}_pred_prob.nii.gz",
        "pred": output_dir / f"{output_prefix}_pred.nii.gz",
    }
    if save_input:
        paths["input"] = output_dir / f"{output_prefix}_input.nii.gz"
    if segmentation_mode == "three_class_shell_interior" and output_profile == "full":
        paths.update(
            {
                "pred_class": output_dir / f"{output_prefix}_pred_class.nii.gz",
                "pred_shell": output_dir / f"{output_prefix}_pred_shell.nii.gz",
                "pred_interior": output_dir / f"{output_prefix}_pred_interior.nii.gz",
            }
        )
    for label in (additional_thresholds or {}):
        validate_additional_threshold_label(label)
        paths[f"pred_threshold_{label}"] = (
            output_dir / f"{output_prefix}_pred_threshold_{label}.nii.gz"
        )
    paths["metadata"] = output_dir / f"{output_prefix}_inference_metadata.json"
    return paths


def additional_threshold_metadata(
    additional_thresholds: Mapping[str, float],
    output_paths: Mapping[str, Path],
) -> list[dict[str, str | float]]:
    return [
        {
            "label": label,
            "threshold": threshold,
            "path": str(output_paths[f"pred_threshold_{label}"].resolve()),
        }
        for label, threshold in additional_thresholds.items()
    ]


def sliding_window_device_options(
    stitch_device: str,
    model_device: torch.device,
) -> tuple[torch.device, dict[str, torch.device]]:
    """Choose the input device and optional MONAI transfer/stitch devices."""

    if stitch_device == "model":
        return model_device, {}
    if stitch_device == "cpu":
        cpu = torch.device("cpu")
        return cpu, {"sw_device": model_device, "device": cpu}
    raise ValueError(f"Unsupported stitch device: {stitch_device!r}")


def load_checkpoint(checkpoint_path: Path) -> Mapping[str, Any]:
    """Load a trusted full training checkpoint using Torch 2.4 semantics."""

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Checkpoint must be a mapping containing model_state_dict")
    return checkpoint


def model_state_dict(checkpoint: Mapping[str, Any]) -> Mapping[str, torch.Tensor]:
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, Mapping):
        raise KeyError("Checkpoint does not contain a model_state_dict mapping")
    return state_dict


def save_nii(arr: np.ndarray, path: Path, reference: LoadedPatch) -> None:
    """Save an output while preserving source NIfTI spatial header information."""

    nib = _require_nibabel()
    output_array = np.asarray(arr)
    header = reference.header.copy() if reference.header is not None else None
    if header is not None:
        header.set_data_dtype(output_array.dtype)
        header.set_slope_inter(1.0, 0.0)

    image = nib.Nifti1Image(output_array, affine=reference.affine, header=header)
    if reference.header is not None:
        qform, qform_code = reference.header.get_qform(coded=True)
        sform, sform_code = reference.header.get_sform(coded=True)
        image.set_qform(qform, code=int(qform_code))
        image.set_sform(sform, code=int(sform_code))
    else:
        image.header.set_xyzt_units("micron")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".nii.gz",
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


def _distribution_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def software_versions() -> dict[str, str | None]:
    return {
        "python": platform.python_version(),
        "numpy": str(np.__version__),
        "torch": str(torch.__version__),
        "torch_cuda": torch.version.cuda,
        "monai": _distribution_version("monai"),
        "nibabel": _distribution_version("nibabel"),
    }


def _resolved_paths(paths: Mapping[str, Path]) -> dict[str, str]:
    return {name: str(path.resolve()) for name, path in paths.items()}


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_prefix = validate_output_prefix(
        args.output_prefix or default_output_prefix(args.input)
    )

    checkpoint = load_checkpoint(args.checkpoint)
    segmentation_mode = resolve_segmentation_mode(args.segmentation_mode, checkpoint)
    print(f"Checkpoint: {args.checkpoint.resolve()}")
    print(f"Segmentation mode: {segmentation_mode}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError("CUDA is required for this inference run")
    print(f"Device: {device}")
    print(f"Stitch device: {args.stitch_device}")
    model = build_model(device, segmentation_mode)
    model.load_state_dict(model_state_dict(checkpoint), strict=True)
    model.eval()
    del checkpoint
    print(f"Model loaded: {sum(parameter.numel() for parameter in model.parameters()):,} params")

    print(f"Loading {args.input} ...")
    loaded_patch = load_patch(args.input, args.voxel_size)
    raw_patch = loaded_patch.data
    output_reference = (
        reorder_loaded_patch(loaded_patch, args.output_axis_order)
        if args.output_axis_order is not None
        else loaded_patch
    )
    print(f"  Shape: {raw_patch.shape}, dtype: {raw_patch.dtype}")
    print(f"  Intensity range: [{raw_patch.min()}, {raw_patch.max()}]")

    patch_f, normalization = normalize_patch(raw_patch)
    print(
        "  Percentile clip: "
        f"[{normalization['clip_values'][0]:.3f}, {normalization['clip_values'][1]:.3f}]"
    )
    print(
        f"  Normalized range: [{patch_f.min():.3f}, {patch_f.max():.3f}] "
        f"mean={patch_f.mean():.3f}"
    )
    input_device, sliding_window_device_kwargs = sliding_window_device_options(
        args.stitch_device, device
    )
    tensor = torch.from_numpy(patch_f).unsqueeze(0).unsqueeze(0).to(
        device=input_device,
        dtype=torch.float32,
    )

    roi_size = (args.roi_size,) * 3
    print(
        "Running sliding-window inference "
        f"(roi={args.roi_size}, overlap={args.overlap}, sw_batch={args.sw_batch_size}) ..."
    )
    from monai.inferers import sliding_window_inference

    autocast_enabled = device.type == "cuda"
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=autocast_enabled):
        pred_logits = sliding_window_inference(
            tensor,
            roi_size,
            args.sw_batch_size,
            model,
            overlap=args.overlap,
            mode=SLIDING_WINDOW_BLEND_MODE,
            sigma_scale=SLIDING_WINDOW_SIGMA_SCALE,
            padding_mode=SLIDING_WINDOW_PADDING_MODE,
            cval=SLIDING_WINDOW_CVAL,
            **sliding_window_device_kwargs,
        )

    del tensor
    pred_logits_cpu = pred_logits.cpu()
    del pred_logits
    if device.type == "cuda":
        torch.cuda.empty_cache()
    outputs = postprocess_logits(
        pred_logits_cpu,
        segmentation_mode,
        args.threshold,
        output_profile=args.output_profile,
        additional_thresholds=args.additional_thresholds,
    )
    del pred_logits_cpu

    output_paths = build_output_paths(
        args.output_dir,
        output_prefix,
        segmentation_mode,
        output_profile=args.output_profile,
        additional_thresholds=args.additional_thresholds,
        save_input=not args.no_save_input,
    )
    metadata_path = output_paths["metadata"]

    if not args.no_save_input:
        output_input = (
            patch_f.transpose(args.output_axis_order)
            if args.output_axis_order is not None
            else patch_f
        )
        save_nii(output_input, output_paths["input"], output_reference)
    for output_name, output_array in outputs.items():
        if args.output_axis_order is not None:
            output_array = output_array.transpose(args.output_axis_order)
        save_nii(output_array, output_paths[output_name], output_reference)

    command_args = list(sys.argv if argv is None else [sys.argv[0], *argv])
    metadata = {
        "input_path": str(args.input.resolve()),
        "input_format": loaded_patch.source_format,
        "input_shape": list(raw_patch.shape),
        "input_dtype": str(raw_patch.dtype),
        "output_shape": list(output_reference.data.shape),
        "checkpoint": {
            "path": str(args.checkpoint.resolve()),
        },
        "requested_segmentation_mode": args.segmentation_mode,
        "resolved_segmentation_mode": segmentation_mode,
        "normalization": normalization,
        "architecture": architecture_metadata(segmentation_mode),
        "inference_parameters": {
            "roi_size": list(roi_size),
            "sliding_window_batch_size": args.sw_batch_size,
            "overlap": args.overlap,
            "threshold": args.threshold,
            "stitch_device": args.stitch_device,
            "output_profile": args.output_profile,
            "save_input": not args.no_save_input,
            "output_axis_order": (
                list(args.output_axis_order)
                if args.output_axis_order is not None
                else None
            ),
            "additional_thresholds": additional_threshold_metadata(
                args.additional_thresholds, output_paths
            ),
            "sliding_window_blend_mode": SLIDING_WINDOW_BLEND_MODE,
            "sliding_window_sigma_scale": SLIDING_WINDOW_SIGMA_SCALE,
            "padding_mode": SLIDING_WINDOW_PADDING_MODE,
            "padding_cval": SLIDING_WINDOW_CVAL,
            "amp_enabled": autocast_enabled,
            "amp_dtype": "float16" if autocast_enabled else None,
        },
        "spatial_metadata": {
            "source": (
                "axis_reordered_from_input_nifti"
                if loaded_patch.source_format == "nifti"
                and args.output_axis_order is not None
                else "preserved_from_input_nifti"
                if loaded_patch.source_format == "nifti"
                else "isotropic_voxel_size_affine"
            ),
            "affine": output_reference.affine.tolist(),
            "nifti_header_preserved": loaded_patch.header is not None,
        },
        "command": shlex.join(command_args),
        "device": {
            "type": device.type,
            "name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        },
        "software_versions": software_versions(),
        "output_paths": _resolved_paths(output_paths),
    }
    write_json_atomic(metadata_path, metadata)

    print(f"Saved inference outputs and metadata to {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

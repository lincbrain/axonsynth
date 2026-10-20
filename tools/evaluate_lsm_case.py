#!/usr/bin/env python3
"""Compare LSM predictions (NIfTI) against dense ground truth for one case.

Reports paper_analysis.metrics.evaluate_binary_mask() results for one or more
probability thresholds applied to raw_pred_prob.nii.gz, and writes TP (green)
/ FP (yellow) / FN (pink) overlay PNGs on the raw image for orthogonal
mid-slices of each threshold.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paper_analysis.metrics import evaluate_binary_mask

TP_COLOR = np.array([0, 200, 0], dtype=np.uint8)
FP_COLOR = np.array([230, 210, 0], dtype=np.uint8)
FN_COLOR = np.array([255, 105, 180], dtype=np.uint8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        required=True,
        help="Directory containing <prefix>_pred_prob.nii.gz and <prefix>_input.nii.gz",
    )
    parser.add_argument(
        "--prefix",
        required=True,
        help="Output prefix used by infer_lsm.py (e.g. macaque_PV_White-Gray_Transition_raw)",
    )
    parser.add_argument("--gt-path", type=Path, required=True, help="Path to ground-truth NIfTI")
    parser.add_argument(
        "--threshold",
        dest="thresholds",
        action="append",
        type=float,
        default=None,
        help="Probability threshold to evaluate; repeat for multiple (default: 0.5 0.7)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to <predictions-dir>/../evaluation",
    )
    return parser.parse_args()


def load_nifti(path: Path) -> np.ndarray:
    return np.asanyarray(nib.load(str(path)).dataobj)


def summarize(results: dict[str, dict]) -> str:
    header = (
        f"{'threshold':<12}{'dice':>8}{'dice_corr':>11}{'iou':>8}{'precision':>11}"
        f"{'recall':>9}{'tp':>12}{'fp':>12}{'fn':>12}"
    )
    lines = [header, "-" * len(header)]
    for name, result in results.items():
        raw = result["raw"]
        dice_corr = result["target_informed_neighbor_corrected"]["dice"]
        lines.append(
            f"{name:<12}{raw['dice']:>8.4f}{dice_corr:>11.4f}{raw['iou']:>8.4f}"
            f"{raw['precision']:>11.4f}{raw['recall']:>9.4f}"
            f"{raw['tp']:>12,}{raw['fp']:>12,}{raw['fn']:>12,}"
        )

    lines.append("")
    lines.append(
        "Corrected metrics (VesSynth-style two-round 6-connected neighbor"
        " correction) and topology errors vs. annotation:"
    )
    corrected_header = (
        f"{'threshold':<12}{'Dice_corr':>10}{'cPrec':>8}{'cRec':>8}{'cFPR':>9}"
        f"{'|dBetti0|':>11}{'|dEuler|':>10}"
    )
    lines.append(corrected_header)
    lines.append("-" * len(corrected_header))
    for name, result in results.items():
        corrected = result["target_informed_neighbor_corrected"]
        topology = result["topology"]
        lines.append(
            f"{name:<12}{corrected['dice']:>10.4f}{corrected['precision']:>8.4f}"
            f"{corrected['recall']:>8.4f}{corrected['fpr']:>9.5f}"
            f"{topology['betti0_error']:>11,}{topology['euler_error']:>10,}"
        )
    return "\n".join(lines)


def normalized_gray_rgb(volume_slice: np.ndarray) -> np.ndarray:
    gray = volume_slice.astype(np.float32)
    span = gray.max() - gray.min()
    gray = (gray - gray.min()) / span if span > 0 else np.zeros_like(gray)
    return np.stack([gray] * 3, axis=-1)


def overlay_slice(raw_slice: np.ndarray, pred_slice: np.ndarray, target_slice: np.ndarray) -> np.ndarray:
    rgb_u8 = (normalized_gray_rgb(raw_slice) * 255).astype(np.uint8)
    pred_bool = pred_slice.astype(bool)
    target_bool = target_slice.astype(bool)
    rgb_u8[pred_bool & target_bool] = TP_COLOR
    rgb_u8[pred_bool & ~target_bool] = FP_COLOR
    rgb_u8[~pred_bool & target_bool] = FN_COLOR
    return rgb_u8


def write_orthogonal_overlays(
    raw: np.ndarray, pred: np.ndarray, target: np.ndarray, output_dir: Path, name: str
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    mid = [dim // 2 for dim in raw.shape]
    slicers = {
        "axis0": (mid[0], slice(None), slice(None)),
        "axis1": (slice(None), mid[1], slice(None)),
        "axis2": (slice(None), slice(None), mid[2]),
    }
    for axis_name, index in slicers.items():
        image = overlay_slice(raw[index], pred[index], target[index])
        Image.fromarray(image).save(output_dir / f"{name}_{axis_name}.png")


def main() -> int:
    args = parse_args()
    thresholds = args.thresholds or [0.5, 0.7]
    output_dir = args.output_dir or (args.predictions_dir.parent / "evaluation")
    overlays_dir = output_dir / "overlays"

    print(f"Loading ground truth from {args.gt_path}")
    target = (load_nifti(args.gt_path) > 0).astype(np.uint8)
    raw = load_nifti(args.predictions_dir / f"{args.prefix}_input.nii.gz")
    prob = load_nifti(args.predictions_dir / f"{args.prefix}_pred_prob.nii.gz")
    if prob.shape != target.shape:
        raise ValueError(f"Shape mismatch: prediction {prob.shape} vs ground truth {target.shape}")

    results: dict[str, dict] = {}
    for threshold in thresholds:
        name = f"thresh{threshold:g}"
        print(f"Evaluating {name}")
        pred = (prob >= threshold).astype(np.uint8)
        results[name] = evaluate_binary_mask(pred, target)
        write_orthogonal_overlays(raw, pred, target, overlays_dir, name)

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")

    summary = summarize(results)
    print()
    print(summary)
    (output_dir / "summary.txt").write_text(summary + "\n")
    print()
    print(f"Wrote {metrics_path}")
    print(f"Wrote overlays to {overlays_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Build a raw / ground-truth / TP-FP-FN overlay comparison figure for one
LSM prediction threshold, with the three orthogonal mid-slices as rows.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.evaluate_lsm_case import load_nifti, normalized_gray_rgb, overlay_slice

GT_COLOR = np.array([40, 120, 255], dtype=np.uint8)


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
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def gt_panel(raw_slice: np.ndarray, target_slice: np.ndarray) -> np.ndarray:
    rgb = (normalized_gray_rgb(raw_slice) * 255).astype(np.uint8)
    rgb[target_slice.astype(bool)] = GT_COLOR
    return rgb


def main() -> int:
    args = parse_args()
    output = args.output or (
        args.predictions_dir.parent
        / "evaluation"
        / f"thresh{args.threshold:g}_comparison.png"
    )

    print(f"Loading ground truth from {args.gt_path}")
    target = (load_nifti(args.gt_path) > 0).astype(np.uint8)
    raw = load_nifti(args.predictions_dir / f"{args.prefix}_input.nii.gz")
    prob = load_nifti(args.predictions_dir / f"{args.prefix}_pred_prob.nii.gz")
    pred = (prob >= args.threshold).astype(np.uint8)

    mid = [dim // 2 for dim in raw.shape]
    axis_slicers = [
        ("axis0", (mid[0], slice(None), slice(None))),
        ("axis1", (slice(None), mid[1], slice(None))),
        ("axis2", (slice(None), slice(None), mid[2])),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    column_titles = ["Raw", "Ground truth", "Overlay (TP green / FP yellow / FN pink)"]
    for row, (axis_name, index) in enumerate(axis_slicers):
        raw_slice = raw[index]
        target_slice = target[index]
        pred_slice = pred[index]

        panels = [
            normalized_gray_rgb(raw_slice),
            gt_panel(raw_slice, target_slice),
            overlay_slice(raw_slice, pred_slice, target_slice),
        ]
        for col, panel in enumerate(panels):
            ax = axes[row, col]
            ax.imshow(panel)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(column_titles[col], fontsize=12)
            if col == 0:
                ax.set_ylabel(axis_name, fontsize=12)

    fig.suptitle(f"threshold={args.threshold:g} vs. ground truth", fontsize=14)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

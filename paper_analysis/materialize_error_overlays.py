#!/usr/bin/env python3
"""Write raw TP/FP/FN overlays for every accepted patch and method."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import nibabel as nib
import numpy as np

from paper_analysis.configuration import (
    DEFAULT_CONFIG_PATH,
    METHOD_NAMES,
    load_pipeline_config,
)
from paper_analysis.evaluation import load_method_case, write_json
from paper_analysis.io import save_nifti_like


def overlay_codes(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    overlay = np.zeros(prediction.shape, dtype=np.uint8)
    overlay[prediction & target] = 1
    overlay[prediction & ~target] = 2
    overlay[~prediction & target] = 3
    return overlay


def run(config_path: Path = DEFAULT_CONFIG_PATH) -> list[dict[str, object]]:
    config = load_pipeline_config(config_path)
    manifest: list[dict[str, object]] = []
    for patch in config.patches:
        for method_name in METHOD_NAMES:
            method = config.get_method(method_name)
            case = load_method_case(config, patch, method)
            overlay = overlay_codes(case["prediction"], case["target"])
            output_path = config.path("raw_error_overlay", patch=patch, method=method)
            save_nifti_like(overlay, output_path, case["target_image"])
            manifest.append(
                {
                    "patch_id": patch.id,
                    "domain": patch.domain,
                    "split": patch.split,
                    "method": method.name,
                    "path": str(output_path),
                    "codes": {"0": "TN", "1": "TP", "2": "FP", "3": "FN"},
                    "counts": {
                        str(code): int(np.count_nonzero(overlay == code))
                        for code in range(4)
                    },
                    "target_informed": False,
                }
            )
    write_json(
        config.path("overlay_manifest"),
        {"schema_version": 1, "n_overlays": len(manifest), "overlays": manifest},
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args(argv)
    manifest = run(args.config)
    print(f"Wrote {len(manifest)} raw error overlays")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

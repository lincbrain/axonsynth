#!/usr/bin/env python3
"""Run the two accepted learned models over the fixed LSM patch cohort."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

from paper_analysis.configuration import (
    DEFAULT_CONFIG_PATH,
    LEARNED_METHOD_NAMES,
    load_pipeline_config,
    paper_context,
)
from paper_analysis.evaluation import write_json


def inference_jobs(config):
    return tuple(
        (method_name, patch)
        for method_name in LEARNED_METHOD_NAMES
        for patch in config.patches
    )


def output_is_current(
    metadata_path: Path,
    score_path: Path,
    mask_path: Path,
    *,
    input_path: Path,
    checkpoint_path: Path,
    mode: str,
    threshold: float,
    context: Mapping[str, Any],
) -> bool:
    if not metadata_path.is_file() or not score_path.is_file() or not mask_path.is_file():
        return False
    try:
        metadata = json.loads(metadata_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    recorded_context = metadata.get("paper_context", {})
    inference = metadata.get("inference_parameters", {})
    expected_output_paths = {
        "pred_prob": score_path.resolve(),
        "pred": mask_path.resolve(),
    }
    recorded_output_paths = metadata.get("output_paths", {})
    if not all(
        Path(recorded_output_paths.get(name, "")).resolve() == path
        for name, path in expected_output_paths.items()
    ):
        return False
    return (
        Path(metadata.get("input_path", "")).resolve() == input_path.resolve()
        and Path(metadata.get("checkpoint", {}).get("path", "")).resolve()
        == checkpoint_path.resolve()
        and metadata.get("resolved_segmentation_mode") == mode
        and inference.get("threshold") == threshold
        and inference.get("roi_size") == [128, 128, 128]
        and inference.get("sliding_window_batch_size") == 4
        and inference.get("overlap") == 0.5
        and inference.get("sliding_window_blend_mode") == "gaussian"
        and inference.get("sliding_window_sigma_scale") == 0.125
        and inference.get("padding_mode") == "constant"
        and inference.get("padding_cval") == 0.0
        and inference.get("amp_enabled") is True
        and inference.get("amp_dtype") == "float16"
        and metadata.get("device", {}).get("type") == "cuda"
        and all(recorded_context.get(key) == value for key, value in context.items())
    )


def run_inference_job(
    config,
    method_name: str,
    patch,
    *,
    python: str,
    force: bool,
) -> None:
    method = config.get_method(method_name)
    if method.checkpoint is None:
        raise ValueError(f"Method {method.name} is not a learned model")
    if not method.checkpoint.is_file():
        raise FileNotFoundError(f"Missing {method.name} checkpoint: {method.checkpoint}")

    input_path = config.path("raw_patch", patch=patch)
    if not input_path.is_file():
        raise FileNotFoundError(f"Missing input patch: {input_path}")
    output_dir = config.path("prediction_dir", patch=patch, method=method)
    score_path = config.path("prediction_score", patch=patch, method=method)
    mask_path = config.path("prediction_mask", patch=patch, method=method)
    metadata_path = config.path("learned_metadata", patch=patch, method=method)
    inference_script = Path(__file__).resolve().parents[1] / "inference" / "infer_lsm.py"
    threshold = method.threshold_for(patch.domain)
    context = paper_context(config, patch, method)
    if (
        score_path.is_file()
        and mask_path.is_file()
        and output_is_current(
            metadata_path,
            score_path,
            mask_path,
            input_path=input_path,
            checkpoint_path=method.checkpoint,
            mode=str(method.segmentation_mode),
            threshold=threshold,
            context=context,
        )
        and not force
    ):
        print(f"Current output exists; skipping {method.name} {patch.id}")
        return

    command = [
        python,
        str(inference_script),
        "--input",
        str(input_path),
        "--checkpoint",
        str(method.checkpoint),
        "--output-dir",
        str(output_dir),
        "--output-prefix",
        patch.patch,
        "--segmentation-mode",
        str(method.segmentation_mode),
        "--threshold",
        str(threshold),
        "--require-cuda",
    ]
    print(f"Running {method.name} on {patch.id}", flush=True)
    subprocess.run(command, check=True)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["paper_context"] = context
    write_json(metadata_path, metadata)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--task-index", type=int, default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_pipeline_config(args.config)
    all_jobs = inference_jobs(config)
    task_index = args.task_index
    if task_index is None and os.environ.get("SLURM_ARRAY_TASK_ID") is not None:
        task_index = int(os.environ["SLURM_ARRAY_TASK_ID"])

    selected = all_jobs
    if task_index is not None:
        if not 0 <= task_index < len(all_jobs):
            raise ValueError(
                f"task-index must be in [0, {len(all_jobs)}), got {task_index}"
            )
        selected = (all_jobs[task_index],)

    for method_name, patch in selected:
        run_inference_job(
            config,
            method_name,
            patch,
            python=args.python,
            force=args.force,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Measure raw and target-informed voxel metrics across acquisition depth."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from paper_analysis.configuration import (
    DEFAULT_CONFIG_PATH,
    METHOD_NAMES,
    load_pipeline_config,
)
from paper_analysis.evaluation import flatten_mapping, load_method_case, write_csv, write_json
from paper_analysis.metrics import target_informed_neighbor_correction, voxel_metrics


CORRECTED_PREFIX = "target_informed_neighbor_corrected"


def depth_rows(config) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    axis = config.depth_axis
    for patch in config.patches:
        for method_name in METHOD_NAMES:
            method = config.get_method(method_name)
            case = load_method_case(config, patch, method)
            boundaries = np.linspace(
                0,
                case["target"].shape[axis],
                config.depth_slabs + 1,
                dtype=int,
            )
            for slab in range(config.depth_slabs):
                start = int(boundaries[slab])
                stop = int(boundaries[slab + 1])
                if stop <= start:
                    raise ValueError(
                        f"Depth axis is too short for {config.depth_slabs} slabs: "
                        f"{patch.id} shape={case['target'].shape}"
                    )
                selection = [slice(None)] * 3
                selection[axis] = slice(start, stop)
                selection_tuple = tuple(selection)
                slab_prediction = case["prediction"][selection_tuple]
                slab_target = case["target"][selection_tuple]
                slab_corrected = target_informed_neighbor_correction(
                    slab_prediction,
                    slab_target,
                    rounds=config.neighbor_rounds,
                )
                row: dict[str, Any] = {
                    "patch_id": patch.id,
                    "dataset": patch.dataset,
                    "patch": patch.patch,
                    "domain": patch.domain,
                    "split": patch.split,
                    "method": method.name,
                    "depth_axis": axis,
                    "slab": slab,
                    "slab_start": start,
                    "slab_stop": stop,
                    "depth_fraction_midpoint": ((start + stop) / 2.0)
                    / case["target"].shape[axis],
                }
                row.update(
                    flatten_mapping(
                        "raw",
                        voxel_metrics(slab_prediction, slab_target),
                    )
                )
                row.update(
                    flatten_mapping(
                        CORRECTED_PREFIX,
                        voxel_metrics(slab_corrected, slab_target),
                    )
                )
                rows.append(row)
    return rows


def trend_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["patch_id"], row["method"])].append(row)
    trends: list[dict[str, Any]] = []
    metric_names = (
        "raw_dice",
        "raw_precision",
        "raw_recall",
        f"{CORRECTED_PREFIX}_dice",
        f"{CORRECTED_PREFIX}_precision",
        f"{CORRECTED_PREFIX}_recall",
    )
    for (patch_id, method), group in sorted(grouped.items()):
        ordered = sorted(group, key=lambda row: int(row["slab"]))
        x = np.asarray([row["depth_fraction_midpoint"] for row in ordered])
        trend: dict[str, Any] = {
            "patch_id": patch_id,
            "method": method,
            "domain": ordered[0]["domain"],
            "split": ordered[0]["split"],
        }
        for name in metric_names:
            prefix, metric = name.rsplit("_", 1)
            valid = []
            for row in ordered:
                prediction_positive = row[f"{prefix}_prediction_positive_voxels"]
                target_positive = row[f"{prefix}_target_positive_voxels"]
                if metric == "dice":
                    valid.append(prediction_positive + target_positive > 0)
                elif metric == "precision":
                    valid.append(prediction_positive > 0)
                else:
                    valid.append(target_positive > 0)
            valid_mask = np.asarray(valid, dtype=bool)
            n_valid = int(valid_mask.sum())
            trend[f"{name}_fit_slabs"] = n_valid
            trend[f"{name}_slope_per_depth_fraction"] = (
                float(np.polyfit(x[valid_mask], np.asarray(
                    [row[name] for row in ordered], dtype=np.float64
                )[valid_mask], 1)[0])
                if n_valid >= 2
                else None
            )
        trends.append(trend)
    return trends


def run(config_path: Path = DEFAULT_CONFIG_PATH):
    config = load_pipeline_config(config_path)
    rows = depth_rows(config)
    trends = trend_rows(rows)
    write_csv(config.path("depth_metrics_csv"), rows)
    write_json(
        config.path("depth_metrics_json"),
        {
            "schema_version": 1,
            "axis": config.depth_axis,
            "slabs": config.depth_slabs,
            "correction_scope": "independently_within_each_depth_slab",
            "rows": rows,
            "trends": trends,
        },
    )
    return rows, trends


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args(argv)
    rows, trends = run(args.config)
    print(f"Wrote {len(rows)} depth rows and {len(trends)} trend rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Build a compact, volume-free paper results packet from canonical tables."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from paper_analysis.configuration import (
    DEFAULT_CONFIG_PATH,
    EXPECTED_PATCH_IDS,
    METHOD_NAMES,
    load_pipeline_config,
)
from paper_analysis.evaluation import read_csv, write_csv, write_json, write_text


COMPACT_FIELDS = (
    "patch_id",
    "dataset",
    "patch",
    "domain",
    "split",
    "method",
    "threshold",
    "threshold_selection_metric",
    "raw_dice",
    "raw_iou",
    "raw_precision",
    "raw_recall",
    "raw_fdr",
    "raw_fpr",
    "target_informed_neighbor_corrected_dice",
    "target_informed_neighbor_corrected_iou",
    "target_informed_neighbor_corrected_precision",
    "target_informed_neighbor_corrected_recall",
    "target_informed_neighbor_corrected_fdr",
    "target_informed_neighbor_corrected_fpr",
    "raw_topology_betti0_error",
    "raw_topology_betti1_error",
    "raw_topology_betti2_error",
    "raw_topology_euler_error",
    "raw_component_precision",
    "raw_component_recall",
    "raw_component_split_count",
    "raw_component_merge_count",
)


def validate_rows(rows: list[dict[str, str]]) -> None:
    expected = {(patch_id, method) for patch_id in EXPECTED_PATCH_IDS for method in METHOD_NAMES}
    actual = {(row.get("patch_id", ""), row.get("method", "")) for row in rows}
    if len(rows) != 48 or actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            f"Expected exactly 48 canonical rows; got {len(rows)}. "
            f"Missing={missing}, extra={extra}"
        )


def heldout_summary(rows: list[dict[str, str]]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    fields = (
        "raw_dice",
        "target_informed_neighbor_corrected_dice",
        "raw_precision",
        "raw_recall",
    )
    for method in METHOD_NAMES:
        selected = [
            row for row in rows if row.get("method") == method and row.get("split") == "heldout"
        ]
        if len(selected) != 10:
            raise ValueError(f"Expected 10 held-out rows for {method}, got {len(selected)}")
        summary[method] = {
            field: sum(float(row[field]) for row in selected) / len(selected)
            for field in fields
        }
    return summary


def markdown_summary(rows: list[dict[str, str]]) -> str:
    heldout = heldout_summary(rows)
    lines = [
        "# AxonSynth LSM Results",
        "",
        "Primary results use the ten held-out patches. The adjusted metric is a",
        "target-informed two-round face-neighbor evaluation tolerance, not model",
        "postprocessing.",
        "",
        "| Method | Raw Dice | Target-informed Dice | Raw Precision | Raw Recall |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for method in METHOD_NAMES:
        row = heldout[method]
        lines.append(
            "| {method} | {raw:.4f} | {corrected:.4f} | {precision:.4f} | {recall:.4f} |".format(
                method=method,
                raw=row["raw_dice"],
                corrected=row["target_informed_neighbor_corrected_dice"],
                precision=row["raw_precision"],
                recall=row["raw_recall"],
            )
        )
    lines.extend(["", "The compact CSV contains all 48 patch-method rows.", ""])
    return "\n".join(lines)


def run(config_path: Path = DEFAULT_CONFIG_PATH) -> dict[str, object]:
    config = load_pipeline_config(config_path)
    metrics_path = config.path("metrics_csv")
    rows = read_csv(metrics_path)
    validate_rows(rows)

    missing_fields = sorted(
        field for field in COMPACT_FIELDS if any(field not in row for row in rows)
    )
    if missing_fields:
        raise ValueError(f"Metrics table is missing compact fields: {missing_fields}")
    compact = [{field: row[field] for field in COMPACT_FIELDS} for row in rows]
    table_path = config.path("packet_table")
    markdown_path = config.path("packet_markdown")
    manifest_path = config.path("packet_manifest")
    write_csv(table_path, compact)
    write_text(markdown_path, markdown_summary(rows))

    manifest: dict[str, object] = {
        "schema_version": 1,
        "rows": len(compact),
        "patches": list(EXPECTED_PATCH_IDS),
        "methods": list(METHOD_NAMES),
        "primary_split": "heldout",
        "calibration_patches": [
            patch.id for patch in config.patches if patch.split == "calibration"
        ],
        "target_informed_neighbor_correction": {
            "rounds": config.neighbor_rounds,
            "connectivity": "face-6",
            "uses_target_to_modify_prediction": True,
        },
        "inputs": {
            "metrics_csv": str(metrics_path),
        },
        "outputs": {
            "compact_table": str(table_path),
            "summary": str(markdown_path),
        },
    }
    write_json(manifest_path, manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args(argv)
    manifest = run(args.config)
    print(f"Built results packet with {manifest['rows']} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

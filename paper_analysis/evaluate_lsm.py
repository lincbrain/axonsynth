#!/usr/bin/env python3
"""Evaluate all four accepted methods on the fixed 12-patch LSM cohort."""

from __future__ import annotations

import argparse
from collections import defaultdict
import numbers
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from paper_analysis.configuration import (
    DEFAULT_CONFIG_PATH,
    METHOD_NAMES,
    load_pipeline_config,
)
from paper_analysis.evaluation import (
    flatten_mapping,
    load_method_case,
    write_csv,
    write_json,
)
from paper_analysis.metrics import evaluate_binary_mask


CORRECTED_KEY = "target_informed_neighbor_corrected"


def evaluate_all(config) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for patch in config.patches:
        for method_name in METHOD_NAMES:
            method = config.get_method(method_name)
            case = load_method_case(config, patch, method)
            evaluation = evaluate_binary_mask(
                case["prediction"],
                case["target"],
                neighbor_rounds=config.neighbor_rounds,
                include_topology=True,
                include_components=True,
            )
            row: dict[str, Any] = {
                "patch_id": patch.id,
                "dataset": patch.dataset,
                "patch": patch.patch,
                "domain": patch.domain,
                "split": patch.split,
                "method": method.name,
                "method_kind": method.kind,
                "threshold": case["threshold"],
                "threshold_selection_metric": method.threshold_selection_metric,
                "score_path": str(case["score_path"]),
                "target_path": str(case["target_path"]),
            }
            row.update(flatten_mapping("raw", evaluation["raw"]))
            row.update(flatten_mapping(CORRECTED_KEY, evaluation[CORRECTED_KEY]))
            row.update(flatten_mapping("neighbor_correction", evaluation["neighbor_correction"]))
            row.update(flatten_mapping("raw_topology", evaluation["topology"]))
            component_values = {
                key.removeprefix("component_"): value
                for key, value in evaluation["components"].items()
            }
            row.update(flatten_mapping("raw_component", component_values))
            rows.append(row)

    expected = len(config.patches) * len(METHOD_NAMES)
    identities = {(row["patch_id"], row["method"]) for row in rows}
    if len(rows) != expected or len(identities) != expected:
        raise RuntimeError(f"Expected {expected} unique evaluation rows, got {len(rows)}")
    return rows


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        for split in (row["split"], "all"):
            for domain in (row["domain"], "all"):
                groups[(row["method"], split, domain)].append(row)

    aggregates: list[dict[str, Any]] = []
    for (method, split, domain), group in sorted(groups.items()):
        aggregate: dict[str, Any] = {
            "method": method,
            "split": split,
            "domain": domain,
            "n_patches": len(group),
        }
        metric_names = sorted(
            {
                key
                for row in group
                for key, value in row.items()
                if isinstance(value, numbers.Real)
                and not isinstance(value, (bool, np.bool_))
                and key != "threshold"
            }
        )
        for name in metric_names:
            values = [float(row[name]) for row in group if name in row]
            aggregate[f"mean_{name}"] = float(np.mean(values))
        aggregates.append(aggregate)
    return aggregates


def run(config_path: Path = DEFAULT_CONFIG_PATH) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    config = load_pipeline_config(config_path)
    rows = evaluate_all(config)
    aggregates = aggregate_rows(rows)
    write_csv(config.path("metrics_csv"), rows)
    write_json(
        config.path("metrics_json"),
        {
            "schema_version": 1,
            "config": str(config.source),
            "n_rows": len(rows),
            "neighbor_correction": {
                "name": CORRECTED_KEY,
                "rounds": config.neighbor_rounds,
                "connectivity": "face-6",
                "uses_target_to_modify_prediction": True,
            },
            "rows": rows,
        },
    )
    write_csv(config.path("macro_metrics_csv"), aggregates)
    write_json(config.path("macro_metrics_json"), aggregates)
    return rows, aggregates


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    rows, aggregates = run(args.config)
    print(f"Wrote {len(rows)} patch-method rows and {len(aggregates)} aggregate rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

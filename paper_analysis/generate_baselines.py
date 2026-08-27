#!/usr/bin/env python3
"""Generate fixed paper baseline score and mask volumes.

Run from the repository root with::

    python -m paper_analysis.generate_baselines

The JSON/YAML manifest schema is::

    {
      "output_dir": "results/baselines",
      "baselines": ["intensity_threshold", "frangi"],
      "domain": "human",
      "cases": [
        {"id": "sample_01", "input": "data/sample_01.nii.gz"},
        {"id": "sample_02", "input": "data/sample_02.nii.gz", "domain": "macaque"}
      ]
    }

The default entry point consumes the canonical 12-patch config. The legacy
manifest schema remains available for the baseline module's existing API.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib.metadata
import json
from pathlib import Path
import re
import sys
from typing import Any, Sequence

import numpy as np

# Support the shebang/direct-script entry point as well as ``python -m``.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paper_analysis.baselines import (
    BASELINE_NAMES,
    FRANGI_PARAMETERS,
    INTENSITY_THRESHOLDS,
    baseline_parameters,
    bright_frangi_baseline,
    intensity_threshold_baseline,
    normalize_input_with_metadata,
    validate_domain,
)
from paper_analysis.configuration import (
    BASELINE_METHOD_NAMES,
    DEFAULT_CONFIG_PATH,
    DOMAINS,
    PipelineConfig,
    load_pipeline_config,
)
from paper_analysis.io import (
    BaselineOutputPaths,
    load_config,
    load_nifti,
    save_nifti_like,
    save_baseline_outputs,
    write_json,
)


@dataclass(frozen=True)
class CaseConfig:
    """Validated case entry from a baseline manifest."""

    case_id: str
    input_path: Path
    domain: str


@dataclass(frozen=True)
class GenerationConfig:
    """Validated generation settings with all paths resolved."""

    source: Path
    output_dir: Path
    baselines: tuple[str, ...]
    cases: tuple[CaseConfig, ...]


def _required_string(mapping: dict[str, Any], key: str, context: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} requires a non-empty string {key!r}")
    return value


def _resolve_path(value: str, config_dir: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = config_dir / path
    return path.resolve()


def _validate_case_id(case_id: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", case_id):
        raise ValueError(
            f"Invalid case id {case_id!r}; use letters, numbers, '.', '_', or '-'"
        )
    return case_id


def parse_generation_config(path: str | Path) -> GenerationConfig:
    """Load and validate a generation manifest without implicit settings."""
    source = Path(path).expanduser().resolve()
    document = load_config(source)
    config_dir = source.parent

    output_dir_value = _required_string(document, "output_dir", "Config")
    output_dir = _resolve_path(output_dir_value, config_dir)

    raw_baselines = document.get("baselines")
    if not isinstance(raw_baselines, list) or not raw_baselines:
        raise ValueError("Config requires a non-empty 'baselines' list")
    if not all(isinstance(name, str) for name in raw_baselines):
        raise ValueError("Every baseline name must be a string")
    if len(set(raw_baselines)) != len(raw_baselines):
        raise ValueError("Config baseline names must be unique")
    unsupported = [name for name in raw_baselines if name not in BASELINE_NAMES]
    if unsupported:
        supported = ", ".join(BASELINE_NAMES)
        raise ValueError(
            f"Unsupported baseline(s) {unsupported}; expected one or both of: {supported}"
        )

    global_domain = document.get("domain")
    if global_domain is not None:
        if not isinstance(global_domain, str):
            raise ValueError("Config 'domain' must be a string")
        validate_domain(global_domain)

    raw_cases = document.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("Config requires a non-empty 'cases' list")

    cases: list[CaseConfig] = []
    seen_ids: set[str] = set()
    for index, raw_case in enumerate(raw_cases):
        context = f"Case at index {index}"
        if not isinstance(raw_case, dict):
            raise ValueError(f"{context} must be a mapping")
        case_id = _validate_case_id(_required_string(raw_case, "id", context))
        if case_id in seen_ids:
            raise ValueError(f"Duplicate case id {case_id!r}")
        seen_ids.add(case_id)

        input_value = _required_string(raw_case, "input", context)
        domain = raw_case.get("domain", global_domain)
        if domain is None:
            raise ValueError(
                f"{context} requires 'domain' because the config has no domain"
            )
        if not isinstance(domain, str):
            raise ValueError(f"{context} domain must be a string")
        validate_domain(domain)
        cases.append(
            CaseConfig(
                case_id=case_id,
                input_path=_resolve_path(input_value, config_dir),
                domain=domain,
            )
        )

    return GenerationConfig(
        source=source,
        output_dir=output_dir,
        baselines=tuple(raw_baselines),
        cases=tuple(cases),
    )


def _output_metadata(
    config: GenerationConfig,
    case: CaseConfig,
    baseline: str,
    raw: np.ndarray,
    normalization: dict[str, Any],
    score: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "baseline": baseline,
        "domain": case.domain,
        "case_id": case.case_id,
        "input": str(case.input_path),
        "config": str(config.source),
        "shape": list(raw.shape),
        "input_dtype": str(raw.dtype),
        "score_dtype": str(score.dtype),
        "mask_dtype": str(mask.dtype),
        "normalization": normalization,
        "parameters": baseline_parameters(baseline, case.domain),
        "positive_voxels": int(np.count_nonzero(mask)),
        "positive_fraction": float(np.mean(mask)),
    }


def _generate_legacy_from_config(path: str | Path) -> list[BaselineOutputPaths]:
    """Generate every explicitly selected baseline in a legacy manifest."""
    config = parse_generation_config(path)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    written: list[BaselineOutputPaths] = []

    for case in config.cases:
        raw, reference = load_nifti(case.input_path)
        normalized, normalization = normalize_input_with_metadata(raw)

        for baseline in config.baselines:
            if baseline == "intensity_threshold":
                score, mask = intensity_threshold_baseline(normalized, case.domain)
            else:
                score, mask = bright_frangi_baseline(normalized, case.domain)

            metadata = _output_metadata(
                config,
                case,
                baseline,
                raw,
                normalization,
                score,
                mask,
            )
            written.append(
                save_baseline_outputs(
                    score=score,
                    mask=mask,
                    reference=reference,
                    output_dir=config.output_dir,
                    case_id=case.case_id,
                    baseline=baseline,
                    metadata=metadata,
                )
            )

    return written


def _validate_canonical_baseline_tables(config: PipelineConfig) -> None:
    intensity = config.methods["intensity_threshold"]
    frangi = config.methods["frangi_bright"]
    if dict(intensity.thresholds) != INTENSITY_THRESHOLDS:
        raise ValueError("Canonical intensity thresholds differ from baselines.py")
    accepted_frangi = {
        domain: float(FRANGI_PARAMETERS[domain]["threshold"]) for domain in DOMAINS
    }
    if dict(frangi.thresholds) != accepted_frangi:
        raise ValueError("Canonical Frangi thresholds differ from baselines.py")


def _canonical_metadata(
    config: PipelineConfig,
    patch: Any,
    method_name: str,
    raw: np.ndarray,
    normalization: dict[str, Any],
    score: np.ndarray,
    mask: np.ndarray,
    paths: BaselineOutputPaths,
) -> dict[str, Any]:
    method = config.methods[method_name]
    if method.baseline is None:
        raise ValueError(f"Method {method_name!r} is not a baseline")
    return {
        "schema_version": 3,
        "method": method_name,
        "baseline": method.baseline,
        "domain": patch.domain,
        "split": patch.split,
        "patch_id": patch.id,
        "dataset": patch.dataset,
        "patch": patch.patch,
        "input": str(config.path("raw_patch", patch=patch)),
        "config": str(config.source),
        "shape": list(raw.shape),
        "input_dtype": str(raw.dtype),
        "score_dtype": str(score.dtype),
        "mask_dtype": str(mask.dtype),
        "normalization": normalization,
        "parameters": baseline_parameters(method.baseline, patch.domain),
        "threshold_selection_metric": method.threshold_selection_metric,
        "positive_voxels": int(np.count_nonzero(mask)),
        "positive_fraction": float(np.mean(mask)),
        "outputs": {
            "score": str(paths.score),
            "mask": str(paths.mask),
        },
        "software_versions": {
            "numpy": np.__version__,
            "scikit-image": importlib.metadata.version("scikit-image"),
            "scipy": importlib.metadata.version("scipy"),
            "nibabel": importlib.metadata.version("nibabel"),
        },
    }


def generate_canonical_baselines(
    path: str | Path = DEFAULT_CONFIG_PATH,
    *,
    patch_index: int | None = None,
) -> list[BaselineOutputPaths]:
    """Generate both accepted baselines from the canonical 12-patch config."""

    config = load_pipeline_config(path)
    _validate_canonical_baseline_tables(config)
    if patch_index is None:
        patches = config.patches
    else:
        if isinstance(patch_index, bool) or not 0 <= patch_index < len(config.patches):
            raise ValueError(
                f"patch_index must be between 0 and {len(config.patches) - 1}"
            )
        patches = (config.patches[patch_index],)

    written: list[BaselineOutputPaths] = []
    for patch in patches:
        raw_path = config.path("raw_patch", patch=patch)
        if not raw_path.is_file():
            raise FileNotFoundError(f"Missing input patch: {raw_path}")
        raw, reference = load_nifti(raw_path)
        normalized, normalization = normalize_input_with_metadata(raw)
        for method_name in BASELINE_METHOD_NAMES:
            method = config.methods[method_name]
            if method.baseline == "intensity_threshold":
                score, mask = intensity_threshold_baseline(normalized, patch.domain)
            elif method.baseline == "frangi":
                score, mask = bright_frangi_baseline(normalized, patch.domain)
            else:  # pragma: no cover - guarded by canonical config validation
                raise ValueError(f"Unsupported canonical baseline {method.baseline!r}")

            paths = BaselineOutputPaths(
                score=config.path("prediction_score", patch=patch, method=method),
                mask=config.path("prediction_mask", patch=patch, method=method),
                metadata=config.path("baseline_metadata", patch=patch, method=method),
            )
            save_nifti_like(np.asarray(score, dtype=np.float32), paths.score, reference)
            save_nifti_like(np.asarray(mask, dtype=np.uint8), paths.mask, reference)
            write_json(
                paths.metadata,
                _canonical_metadata(
                    config,
                    patch,
                    method_name,
                    raw,
                    normalization,
                    score,
                    mask,
                    paths,
                ),
            )
            written.append(paths)
    return written


def generate_from_config(path: str | Path) -> list[BaselineOutputPaths]:
    """Generate canonical baselines or an explicitly requested legacy manifest."""

    document = load_config(path)
    if document.get("schema_version") == 3 and "methods" in document:
        return generate_canonical_baselines(path)
    return _generate_legacy_from_config(path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate fixed accepted paper baseline NIfTI outputs."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Canonical paper config (default: paper_analysis/config/lsm_12patch.json)",
    )
    parser.add_argument(
        "--patch-index",
        type=int,
        default=None,
        help="Generate one zero-based patch index, suitable for a 0-11 job array",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    document = load_config(args.config)
    if document.get("schema_version") == 3 and "methods" in document:
        outputs = generate_canonical_baselines(
            args.config,
            patch_index=args.patch_index,
        )
    else:
        if args.patch_index is not None:
            raise ValueError("--patch-index is only supported by the canonical config")
        outputs = _generate_legacy_from_config(args.config)
    print(
        json.dumps(
            [
                {
                    "score": str(paths.score),
                    "mask": str(paths.mask),
                    "metadata": str(paths.metadata),
                }
                for paths in outputs
            ],
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

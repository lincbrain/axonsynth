"""Fixed, accepted classical baselines used in the paper analysis.

Both baselines consume the same normalized float32 volume returned by
``normalize_input``.  The parameter tables in this module are the complete
paper configuration; callers cannot supply alternate thresholds or filter
parameters through the baseline functions.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from skimage.filters import frangi


NORMALIZATION_PERCENTILES = (0.5, 99.5)
INTENSITY_THRESHOLDS = {
    "human": 0.47,
    "macaque": 0.46,
}
FRANGI_PARAMETERS = {
    "human": {
        "alpha": 0.25,
        "beta": 0.5,
        "sigmas": (3, 4, 5, 6, 7),
        "threshold": 0.05,
    },
    "macaque": {
        "alpha": 0.25,
        "beta": 0.33,
        "sigmas": (2, 3, 4, 5, 6, 7, 8, 9, 10),
        "threshold": 0.01,
    },
}
BASELINE_NAMES = ("intensity_threshold", "frangi")
FRANGI_GAMMA_DESCRIPTION = "skimage automatic per-volume gamma"
FRANGI_MODE = "reflect"


def validate_domain(domain: str) -> str:
    """Validate and return an explicitly configured acquisition domain."""
    if domain not in INTENSITY_THRESHOLDS:
        supported = ", ".join(INTENSITY_THRESHOLDS)
        raise ValueError(f"Unsupported domain {domain!r}; expected one of: {supported}")
    return domain


def normalize_input_with_metadata(
    volume: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Clip float32 input at percentiles [0.5, 99.5] and scale to [0, 1].

    Empty, complex, or non-finite inputs are rejected.  A degenerate
    percentile range maps to an all-zero volume rather than dividing by zero.
    """
    source = np.asanyarray(volume)
    if source.size == 0:
        raise ValueError("Cannot normalize an empty volume")
    if np.iscomplexobj(source):
        raise ValueError("Cannot normalize complex-valued input")

    try:
        values = np.asarray(source, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError("Input volume must contain numeric values") from exc
    if not np.isfinite(values).all():
        raise ValueError("Input volume must contain only finite values")

    lower, upper = (
        float(value)
        for value in np.percentile(values, NORMALIZATION_PERCENTILES)
    )
    degenerate_range = upper <= lower
    if degenerate_range:
        normalized = np.zeros_like(values, dtype=np.float32)
    else:
        normalized = np.clip(values, lower, upper)
        normalized = (normalized - lower) / (upper - lower)
        normalized = np.clip(normalized, 0.0, 1.0).astype(np.float32, copy=False)

    metadata = {
        "method": "float32_percentile_clip_and_scale",
        "percentiles": list(NORMALIZATION_PERCENTILES),
        "clip_values": [lower, upper],
        "output_range": [0.0, 1.0],
        "dtype": "float32",
        "degenerate_range": degenerate_range,
    }
    return normalized, metadata


def normalize_input(volume: np.ndarray) -> np.ndarray:
    """Return the fixed paper normalization for ``volume``."""
    normalized, _ = normalize_input_with_metadata(volume)
    return normalized


def _validated_normalized_input(volume: np.ndarray) -> np.ndarray:
    score = np.asarray(volume, dtype=np.float32)
    if score.size == 0:
        raise ValueError("Baseline input cannot be empty")
    if not np.isfinite(score).all():
        raise ValueError("Baseline input must contain only finite values")
    if np.any(score < 0.0) or np.any(score > 1.0):
        raise ValueError("Baseline input must be normalized to [0, 1]")
    return score


def intensity_threshold_baseline(
    normalized_volume: np.ndarray,
    domain: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return normalized-intensity scores and the calibrated domain mask."""
    domain = validate_domain(domain)
    score = _validated_normalized_input(normalized_volume)
    mask = (score >= INTENSITY_THRESHOLDS[domain]).astype(np.uint8)
    return score, mask


def bright_frangi_baseline(
    normalized_volume: np.ndarray,
    domain: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return bright-tube Frangi scores and the accepted domain mask.

    ``gamma=None`` deliberately delegates gamma selection to scikit-image,
    which computes its automatic gamma from each input volume.
    """
    domain = validate_domain(domain)
    normalized = _validated_normalized_input(normalized_volume)
    parameters = FRANGI_PARAMETERS[domain]
    score = np.asarray(
        frangi(
            normalized,
            sigmas=parameters["sigmas"],
            alpha=parameters["alpha"],
            beta=parameters["beta"],
            gamma=None,
            black_ridges=False,
            mode=FRANGI_MODE,
        ),
        dtype=np.float32,
    )
    if score.shape != normalized.shape:
        raise ValueError("Frangi score shape does not match the input volume")
    if not np.isfinite(score).all():
        raise ValueError("Frangi produced non-finite scores")
    mask = (score >= parameters["threshold"]).astype(np.uint8)
    return score, mask


def baseline_parameters(baseline: str, domain: str) -> dict[str, Any]:
    """Return all fixed parameters recorded for one baseline output."""
    domain = validate_domain(domain)
    if baseline == "intensity_threshold":
        return {
            "threshold": INTENSITY_THRESHOLDS[domain],
            "threshold_operator": ">=",
        }
    if baseline == "frangi":
        parameters = FRANGI_PARAMETERS[domain]
        return {
            "alpha": parameters["alpha"],
            "beta": parameters["beta"],
            "sigmas": list(parameters["sigmas"]),
            "threshold": parameters["threshold"],
            "threshold_operator": ">=",
            "black_ridges": False,
            "gamma": None,
            "gamma_behavior": FRANGI_GAMMA_DESCRIPTION,
            "mode": FRANGI_MODE,
            "implementation": "skimage.filters.frangi",
        }
    supported = ", ".join(BASELINE_NAMES)
    raise ValueError(f"Unsupported baseline {baseline!r}; expected one of: {supported}")

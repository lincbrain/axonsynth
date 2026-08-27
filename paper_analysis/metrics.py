"""NumPy/SciPy metrics for three-dimensional binary masks.

The neighbor tolerance in this module is target-informed: it is suitable only
for reporting evaluation metrics. It is not model postprocessing and must not
be presented as a deployable correction of a prediction.

All scalar divisions use zero when the denominator is zero. The only
exceptions are Dice and IoU, which are one when both evaluated masks contain
no foreground.
"""

from __future__ import annotations

import operator
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import ndimage


BinaryMask = NDArray[np.bool_]
ScalarMetrics = dict[str, int | float]


def _as_binary_mask(mask: ArrayLike, name: str) -> BinaryMask:
    array = np.asarray(mask)
    if array.ndim != 3:
        raise ValueError(f"{name} must be a 3D mask; got {array.ndim} dimensions")
    if array.dtype.kind not in "biuf":
        raise TypeError(f"{name} must have a boolean or numeric dtype")
    if not np.all((array == 0) | (array == 1)):
        raise ValueError(f"{name} must contain only binary values 0 and 1")
    return array.astype(np.bool_, copy=False)


def _prepare_pair(
    prediction: ArrayLike,
    target: ArrayLike,
    valid_mask: ArrayLike | None,
) -> tuple[BinaryMask, BinaryMask, BinaryMask]:
    prediction_array = _as_binary_mask(prediction, "prediction")
    target_array = _as_binary_mask(target, "target")
    if prediction_array.shape != target_array.shape:
        raise ValueError(
            "prediction and target must have the same shape; "
            f"got {prediction_array.shape} and {target_array.shape}"
        )
    valid_array = _prepare_valid_mask(valid_mask, prediction_array.shape)
    return prediction_array, target_array, valid_array


def _prepare_valid_mask(
    valid_mask: ArrayLike | None, shape: tuple[int, ...]
) -> BinaryMask:
    if valid_mask is None:
        return np.ones(shape, dtype=np.bool_)
    valid_array = _as_binary_mask(valid_mask, "valid_mask")
    if valid_array.shape != shape:
        raise ValueError(
            f"valid_mask must have shape {shape}; got {valid_array.shape}"
        )
    return valid_array


def _safe_divide(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _normalize_integer(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer")
    try:
        normalized = operator.index(value)
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error
    if normalized < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return normalized


def _connectivity_structure(connectivity: int) -> BinaryMask:
    connectivity = _normalize_integer(connectivity, "connectivity", minimum=1)
    rank = {6: 1, 18: 2, 26: 3}.get(connectivity)
    if rank is None:
        raise ValueError("connectivity must be one of 6, 18, or 26")
    return ndimage.generate_binary_structure(3, rank)


def voxel_metrics(
    prediction: ArrayLike,
    target: ArrayLike,
    valid_mask: ArrayLike | None = None,
) -> ScalarMetrics:
    """Return confusion counts, rates, and fractions for two 3D masks.

    Fractions use the number of valid voxels as their denominator. Values
    outside ``valid_mask`` do not contribute to any count or fraction.
    Zero-denominator rates are zero, except that Dice and IoU are one for two
    foreground-empty masks.
    """

    prediction_array, target_array, valid_array = _prepare_pair(
        prediction, target, valid_mask
    )
    prediction_valid = prediction_array & valid_array
    target_valid = target_array & valid_array

    tp = int(np.count_nonzero(prediction_valid & target_valid))
    fp = int(np.count_nonzero(prediction_valid & ~target_valid))
    fn = int(np.count_nonzero(~prediction_valid & target_valid))
    tn = int(
        np.count_nonzero(valid_array & ~prediction_array & ~target_array)
    )
    valid_voxels = tp + fp + fn + tn
    foreground_union = tp + fp + fn
    dice_denominator = 2 * tp + fp + fn

    return {
        "valid_voxels": valid_voxels,
        "prediction_positive_voxels": tp + fp,
        "target_positive_voxels": tp + fn,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "dice": 1.0
        if dice_denominator == 0
        else _safe_divide(2 * tp, dice_denominator),
        "iou": 1.0
        if foreground_union == 0
        else _safe_divide(tp, foreground_union),
        "precision": _safe_divide(tp, tp + fp),
        "recall": _safe_divide(tp, tp + fn),
        "specificity": _safe_divide(tn, tn + fp),
        "accuracy": _safe_divide(tp + tn, valid_voxels),
        "fdr": _safe_divide(fp, tp + fp),
        "fpr": _safe_divide(fp, fp + tn),
        "tp_fraction": _safe_divide(tp, valid_voxels),
        "fp_fraction": _safe_divide(fp, valid_voxels),
        "fn_fraction": _safe_divide(fn, valid_voxels),
        "tn_fraction": _safe_divide(tn, valid_voxels),
        "prediction_foreground_fraction": _safe_divide(
            tp + fp, valid_voxels
        ),
        "target_foreground_fraction": _safe_divide(tp + fn, valid_voxels),
    }


def target_informed_neighbor_correction(
    prediction: ArrayLike,
    target: ArrayLike,
    valid_mask: ArrayLike | None = None,
    rounds: int = 2,
) -> BinaryMask:
    """Apply a target-informed face-neighbor tolerance for evaluation only.

    This function is not prediction postprocessing: it reads the target and
    therefore must only be used to compute explicitly labeled tolerance
    metrics. Two independent fronts begin at the original true positives.
    During each round, one front promotes original false negatives through
    target foreground, while the other removes original false positives
    through prediction foreground. Neither front can cross ``valid_mask``.

    Voxels outside ``valid_mask`` are returned unchanged.
    """

    prediction_array, target_array, valid_array = _prepare_pair(
        prediction, target, valid_mask
    )
    rounds = _normalize_integer(rounds, "rounds")

    corrected = prediction_array.copy()
    original_tp = prediction_array & target_array & valid_array
    original_fn = ~prediction_array & target_array & valid_array
    original_fp = prediction_array & ~target_array & valid_array

    fn_front = original_tp.copy()
    fp_front = original_tp.copy()
    reached_fn = np.zeros(prediction_array.shape, dtype=np.bool_)
    reached_fp = np.zeros(prediction_array.shape, dtype=np.bool_)
    face_structure = _connectivity_structure(6)

    for _ in range(rounds):
        next_fn = (
            ndimage.binary_dilation(fn_front, structure=face_structure)
            & original_fn
            & ~reached_fn
            & valid_array
        )
        next_fp = (
            ndimage.binary_dilation(fp_front, structure=face_structure)
            & original_fp
            & ~reached_fp
            & valid_array
        )
        corrected[next_fn] = True
        corrected[next_fp] = False
        reached_fn |= next_fn
        reached_fp |= next_fp
        fn_front = next_fn
        fp_front = next_fp

    return corrected


def _crop_to_valid_bbox(mask: BinaryMask, valid_mask: BinaryMask) -> BinaryMask:
    if not np.any(valid_mask):
        return np.zeros((0, 0, 0), dtype=np.bool_)
    coordinates = np.nonzero(valid_mask)
    bounds = tuple(
        slice(int(axis.min()), int(axis.max()) + 1) for axis in coordinates
    )
    return mask[bounds] & valid_mask[bounds]


def _component_count(mask: BinaryMask, connectivity: int) -> int:
    _, count = ndimage.label(mask, structure=_connectivity_structure(connectivity))
    return int(count)


def _cavity_count(mask: BinaryMask, background_connectivity: int) -> int:
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    _, background_count = ndimage.label(
        ~padded, structure=_connectivity_structure(background_connectivity)
    )
    # Padding guarantees exactly one exterior background component.
    return max(int(background_count) - 1, 0)


def _six_connected_euler(mask: BinaryMask) -> int:
    """Euler characteristic of the face-connected voxel-center complex."""

    vertices = int(np.count_nonzero(mask))
    edges = sum(
        int(np.count_nonzero(left & right))
        for left, right in (
            (mask[:-1, :, :], mask[1:, :, :]),
            (mask[:, :-1, :], mask[:, 1:, :]),
            (mask[:, :, :-1], mask[:, :, 1:]),
        )
    )
    faces = sum(
        int(np.count_nonzero(face))
        for face in (
            mask[:-1, :-1, :]
            & mask[1:, :-1, :]
            & mask[:-1, 1:, :]
            & mask[1:, 1:, :],
            mask[:-1, :, :-1]
            & mask[1:, :, :-1]
            & mask[:-1, :, 1:]
            & mask[1:, :, 1:],
            mask[:, :-1, :-1]
            & mask[:, 1:, :-1]
            & mask[:, :-1, 1:]
            & mask[:, 1:, 1:],
        )
    )
    cubes = int(
        np.count_nonzero(
            mask[:-1, :-1, :-1]
            & mask[1:, :-1, :-1]
            & mask[:-1, 1:, :-1]
            & mask[1:, 1:, :-1]
            & mask[:-1, :-1, 1:]
            & mask[1:, :-1, 1:]
            & mask[:-1, 1:, 1:]
            & mask[1:, 1:, 1:]
        )
    )
    return vertices - edges + faces - cubes


def _six_connected_topology(mask: BinaryMask) -> tuple[int, int, int, int]:
    betti0 = _component_count(mask, 6)
    if betti0 == 0:
        return 0, 0, 0, 0
    betti2 = _cavity_count(mask, 26)
    euler = _six_connected_euler(mask)
    betti1 = betti0 + betti2 - euler
    return betti0, int(betti1), betti2, euler


def topology_metrics(
    mask: ArrayLike,
    valid_mask: ArrayLike | None = None,
    foreground_connectivity: int = 6,
) -> dict[str, int]:
    """Return Betti numbers and Euler characteristic for a 3D mask.

    Foreground connectivity may be 6 or 26; the complementary background
    connectivity is respectively 26 or 6. The evaluated mask is cropped to
    the bounding box of ``valid_mask``, and invalid voxels inside that box are
    treated as background. Empty masks have all four values equal to zero.
    """

    mask_array = _as_binary_mask(mask, "mask")
    valid_array = _prepare_valid_mask(valid_mask, mask_array.shape)
    foreground_connectivity = _normalize_integer(
        foreground_connectivity, "foreground_connectivity", minimum=1
    )
    if foreground_connectivity not in (6, 26):
        raise ValueError("foreground_connectivity must be 6 or 26")

    foreground = _crop_to_valid_bbox(mask_array, valid_array)
    if foreground.size == 0 or not np.any(foreground):
        return {"betti0": 0, "betti1": 0, "betti2": 0, "euler": 0}

    if foreground_connectivity == 6:
        betti0, betti1, betti2, euler = _six_connected_topology(foreground)
    else:
        betti0 = _component_count(foreground, 26)
        betti2 = _cavity_count(foreground, 6)

        # H1 is invariant under complement duality in three dimensions. A
        # background shell keeps the unbounded component finite without
        # changing its first Betti number.
        padded_foreground = np.pad(
            foreground, 1, mode="constant", constant_values=False
        )
        background = ~padded_foreground
        _, betti1, _, _ = _six_connected_topology(background)
        euler = betti0 - betti1 + betti2

    return {
        "betti0": int(betti0),
        "betti1": int(betti1),
        "betti2": int(betti2),
        "euler": int(euler),
    }


def topology_comparison(
    prediction: ArrayLike,
    target: ArrayLike,
    valid_mask: ArrayLike | None = None,
    foreground_connectivity: int = 6,
) -> dict[str, int]:
    """Compare prediction and target Betti numbers and Euler characteristic.

    ``*_difference`` is prediction minus target, while ``*_error`` is its
    absolute value. ``betti_error`` is the sum of the three Betti errors.
    """

    prediction_array, target_array, valid_array = _prepare_pair(
        prediction, target, valid_mask
    )
    prediction_topology = topology_metrics(
        prediction_array, valid_array, foreground_connectivity
    )
    target_topology = topology_metrics(
        target_array, valid_array, foreground_connectivity
    )

    result: dict[str, int] = {}
    for metric in ("betti0", "betti1", "betti2", "euler"):
        prediction_value = prediction_topology[metric]
        target_value = target_topology[metric]
        difference = prediction_value - target_value
        result[f"prediction_{metric}"] = prediction_value
        result[f"target_{metric}"] = target_value
        result[f"{metric}_difference"] = difference
        result[f"{metric}_error"] = abs(difference)
    result["betti_error"] = sum(
        result[f"betti{dimension}_error"] for dimension in range(3)
    )
    return result


def component_metrics(
    prediction: ArrayLike,
    target: ArrayLike,
    valid_mask: ArrayLike | None = None,
    connectivity: int = 6,
) -> ScalarMetrics:
    """Return overlap-based connected-component metrics for two 3D masks.

    A component is matched when it overlaps any component in the other mask.
    A split is a target component overlapping multiple prediction components;
    a merge is the converse. Matched IoUs summarize every observed overlapping
    pair, including all branches of splits and merges. No dense Cartesian
    component-pair table is constructed.
    """

    prediction_array, target_array, valid_array = _prepare_pair(
        prediction, target, valid_mask
    )
    structure = _connectivity_structure(connectivity)
    prediction_labels, prediction_count = ndimage.label(
        prediction_array & valid_array, structure=structure
    )
    target_labels, target_count = ndimage.label(
        target_array & valid_array, structure=structure
    )
    prediction_count = int(prediction_count)
    target_count = int(target_count)

    overlap = (prediction_labels > 0) & (target_labels > 0)
    if np.any(overlap):
        observed_pairs, intersections = np.unique(
            np.column_stack(
                (prediction_labels[overlap], target_labels[overlap])
            ),
            axis=0,
            return_counts=True,
        )
        observed_pairs = observed_pairs.astype(np.intp, copy=False)
        intersections = intersections.astype(np.int64, copy=False)
    else:
        observed_pairs = np.empty((0, 2), dtype=np.intp)
        intersections = np.empty(0, dtype=np.int64)

    prediction_sizes = (
        np.asarray(
            ndimage.sum(
                prediction_labels > 0,
                labels=prediction_labels,
                index=np.arange(1, prediction_count + 1),
            ),
            dtype=np.int64,
        )
        if prediction_count
        else np.empty(0, dtype=np.int64)
    )
    target_sizes = (
        np.asarray(
            ndimage.sum(
                target_labels > 0,
                labels=target_labels,
                index=np.arange(1, target_count + 1),
            ),
            dtype=np.int64,
        )
        if target_count
        else np.empty(0, dtype=np.int64)
    )

    if observed_pairs.size:
        matched_prediction, prediction_degrees = np.unique(
            observed_pairs[:, 0], return_counts=True
        )
        matched_target, target_degrees = np.unique(
            observed_pairs[:, 1], return_counts=True
        )
        unions = (
            prediction_sizes[observed_pairs[:, 0] - 1]
            + target_sizes[observed_pairs[:, 1] - 1]
            - intersections
        )
        matched_ious = intersections / unions
    else:
        matched_prediction = np.empty(0, dtype=np.intp)
        matched_target = np.empty(0, dtype=np.intp)
        prediction_degrees = np.empty(0, dtype=np.int64)
        target_degrees = np.empty(0, dtype=np.int64)
        matched_ious = np.empty(0, dtype=np.float64)

    matched_prediction_count = int(matched_prediction.size)
    matched_target_count = int(matched_target.size)
    false_components = prediction_count - matched_prediction_count
    missed_components = target_count - matched_target_count
    count_difference = prediction_count - target_count

    return {
        "prediction_components": prediction_count,
        "target_components": target_count,
        "component_count_difference": count_difference,
        "component_count_error": abs(count_difference),
        "matched_prediction_components": matched_prediction_count,
        "matched_target_components": matched_target_count,
        "component_precision": _safe_divide(
            matched_prediction_count, prediction_count
        ),
        "component_recall": _safe_divide(matched_target_count, target_count),
        "false_components": false_components,
        "missed_components": missed_components,
        "split_count": int(np.count_nonzero(target_degrees > 1)),
        "merge_count": int(np.count_nonzero(prediction_degrees > 1)),
        "matched_pair_count": int(observed_pairs.shape[0]),
        "matched_iou_mean": float(matched_ious.mean())
        if matched_ious.size
        else 0.0,
        "matched_iou_median": float(np.median(matched_ious))
        if matched_ious.size
        else 0.0,
        "matched_iou_min": float(matched_ious.min())
        if matched_ious.size
        else 0.0,
        "matched_iou_max": float(matched_ious.max())
        if matched_ious.size
        else 0.0,
    }


def evaluate_binary_mask(
    prediction: ArrayLike,
    target: ArrayLike,
    valid_mask: ArrayLike | None = None,
    neighbor_rounds: int = 2,
    include_topology: bool = True,
    include_components: bool = True,
) -> dict[str, ScalarMetrics | dict[str, int]]:
    """Evaluate a 3D binary prediction with optional structural metrics.

    ``raw`` always contains ordinary voxel metrics. The separately named
    ``target_informed_neighbor_corrected`` result uses the
    target-dependent neighbor tolerance and is for evaluation only, never
    prediction postprocessing.
    Topology and component comparisons use the unmodified masks.
    """

    prediction_array, target_array, valid_array = _prepare_pair(
        prediction, target, valid_mask
    )
    corrected = target_informed_neighbor_correction(
        prediction_array,
        target_array,
        valid_array,
        rounds=neighbor_rounds,
    )
    result: dict[str, ScalarMetrics | dict[str, int]] = {
        "raw": voxel_metrics(prediction_array, target_array, valid_array),
        "target_informed_neighbor_corrected": voxel_metrics(
            corrected, target_array, valid_array
        ),
        "neighbor_correction": {
            "rounds": int(neighbor_rounds),
            "connectivity": 6,
            "fn_promoted_to_tp_voxels": int(
                np.count_nonzero(~prediction_array & corrected & target_array & valid_array)
            ),
            "fp_removed_voxels": int(
                np.count_nonzero(prediction_array & ~corrected & ~target_array & valid_array)
            ),
            "uses_target_to_modify_prediction": True,
        },
    }
    if include_topology:
        result["topology"] = topology_comparison(
            prediction_array, target_array, valid_array
        )
    if include_components:
        result["components"] = component_metrics(
            prediction_array, target_array, valid_array
        )
    return result

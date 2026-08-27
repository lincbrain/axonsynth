import numpy as np
import pytest

from paper_analysis import (
    component_metrics,
    evaluate_binary_mask,
    target_informed_neighbor_correction,
    topology_comparison,
    topology_metrics,
    voxel_metrics,
)


def test_voxel_metrics_perfect_mask_and_fractions():
    mask = np.zeros((2, 2, 2), dtype=bool)
    mask[0, 0, 0] = True

    metrics = voxel_metrics(mask, mask)

    assert (metrics["tp"], metrics["fp"], metrics["fn"], metrics["tn"]) == (
        1,
        0,
        0,
        7,
    )
    for name in (
        "dice",
        "iou",
        "precision",
        "recall",
        "specificity",
        "accuracy",
    ):
        assert metrics[name] == 1.0
    assert metrics["fdr"] == 0.0
    assert metrics["tp_fraction"] == pytest.approx(1 / 8)
    assert metrics["tn_fraction"] == pytest.approx(7 / 8)
    assert metrics["prediction_foreground_fraction"] == pytest.approx(1 / 8)
    assert metrics["target_foreground_fraction"] == pytest.approx(1 / 8)


def test_voxel_metrics_both_empty_policy():
    empty = np.zeros((2, 2, 2), dtype=bool)

    metrics = voxel_metrics(empty, empty)

    assert metrics["dice"] == 1.0
    assert metrics["iou"] == 1.0
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["fdr"] == 0.0
    assert metrics["accuracy"] == 1.0


def test_voxel_metrics_all_false_positive_and_all_false_negative():
    foreground = np.ones((1, 1, 3), dtype=bool)
    empty = np.zeros_like(foreground)

    all_fp = voxel_metrics(foreground, empty)
    all_fn = voxel_metrics(empty, foreground)

    assert (all_fp["tp"], all_fp["fp"], all_fp["fn"]) == (0, 3, 0)
    assert all_fp["dice"] == 0.0
    assert all_fp["iou"] == 0.0
    assert all_fp["fdr"] == 1.0
    assert (all_fn["tp"], all_fn["fp"], all_fn["fn"]) == (0, 0, 3)
    assert all_fn["dice"] == 0.0
    assert all_fn["recall"] == 0.0


def test_voxel_metrics_respects_valid_mask_and_empty_valid_policy():
    prediction = np.ones((1, 1, 2), dtype=bool)
    target = np.zeros_like(prediction)
    valid = np.array([[[False, True]]])

    metrics = voxel_metrics(prediction, target, valid)
    no_valid = voxel_metrics(prediction, target, np.zeros_like(valid))

    assert (metrics["fp"], metrics["tn"]) == (1, 0)
    assert metrics["fp_fraction"] == 1.0
    assert no_valid["dice"] == 1.0
    assert no_valid["iou"] == 1.0
    assert no_valid["accuracy"] == 0.0
    assert no_valid["tn_fraction"] == 0.0


def test_neighbor_tolerance_uses_one_and_two_face_neighbor_rounds():
    target = np.ones((1, 1, 3), dtype=bool)
    prediction = np.zeros_like(target)
    prediction[0, 0, 0] = True

    after_one = target_informed_neighbor_correction(
        prediction, target, rounds=1
    )
    after_two = target_informed_neighbor_correction(
        prediction, target, rounds=2
    )

    np.testing.assert_array_equal(after_one, np.array([[[True, True, False]]]))
    np.testing.assert_array_equal(after_two, target)


def test_neighbor_tolerance_removes_false_positives_by_round():
    prediction = np.ones((1, 1, 3), dtype=bool)
    target = np.zeros_like(prediction)
    target[0, 0, 0] = True

    after_one = target_informed_neighbor_correction(
        prediction, target, rounds=1
    )
    after_two = target_informed_neighbor_correction(
        prediction, target, rounds=2
    )

    np.testing.assert_array_equal(after_one, np.array([[[True, False, True]]]))
    np.testing.assert_array_equal(after_two, target)


def test_neighbor_tolerance_fronts_grow_independently():
    prediction = np.zeros((1, 2, 2), dtype=bool)
    target = np.zeros_like(prediction)
    prediction[0, 0, 0] = True
    target[0, 0, 0] = True  # Original TP seed.
    target[0, 0, 1] = True  # FN reached in round one.
    prediction[0, 1, 1] = True  # FP adjacent only to the reached FN.

    corrected = target_informed_neighbor_correction(
        prediction, target, rounds=2
    )

    assert corrected[0, 0, 1]
    assert corrected[0, 1, 1]


def test_neighbor_tolerance_excludes_diagonal_neighbors():
    prediction = np.zeros((2, 2, 2), dtype=bool)
    target = np.zeros_like(prediction)
    prediction[0, 0, 0] = target[0, 0, 0] = True
    target[1, 1, 0] = True
    prediction[1, 1, 1] = True

    corrected = target_informed_neighbor_correction(
        prediction, target, rounds=2
    )

    assert not corrected[1, 1, 0]
    assert corrected[1, 1, 1]


def test_neighbor_tolerance_never_crosses_valid_boundary():
    prediction = np.array([[[True, False, False]]])
    target = np.ones_like(prediction)
    valid = np.array([[[True, False, True]]])

    corrected = target_informed_neighbor_correction(
        prediction, target, valid, rounds=2
    )

    np.testing.assert_array_equal(corrected, prediction)


def test_neighbor_tolerance_is_documented_as_evaluation_not_postprocessing():
    documentation = target_informed_neighbor_correction.__doc__.lower()
    assert "evaluation only" in documentation
    assert "not prediction postprocessing" in documentation


def test_component_metrics_perfect_and_both_empty():
    mask = np.zeros((3, 3, 3), dtype=bool)
    mask[1, 1, 1] = True

    perfect = component_metrics(mask, mask)
    empty = component_metrics(np.zeros_like(mask), np.zeros_like(mask))

    assert perfect["prediction_components"] == 1
    assert perfect["target_components"] == 1
    assert perfect["component_precision"] == 1.0
    assert perfect["component_recall"] == 1.0
    assert perfect["matched_iou_mean"] == 1.0
    assert empty["prediction_components"] == 0
    assert empty["target_components"] == 0
    assert empty["component_precision"] == 0.0
    assert empty["component_recall"] == 0.0
    assert empty["matched_iou_mean"] == 0.0


def test_metrics_handle_zero_volume_3d_masks():
    empty = np.zeros((0, 2, 2), dtype=bool)

    assert voxel_metrics(empty, empty)["dice"] == 1.0
    assert component_metrics(empty, empty)["prediction_components"] == 0
    assert topology_metrics(empty)["betti0"] == 0
    assert target_informed_neighbor_correction(empty, empty).shape == empty.shape


def test_disjoint_equal_component_counts_are_not_perfect():
    prediction = np.zeros((1, 1, 4), dtype=bool)
    target = np.zeros_like(prediction)
    prediction[0, 0, 0] = True
    target[0, 0, 3] = True

    metrics = component_metrics(prediction, target)

    assert metrics["component_count_error"] == 0
    assert metrics["component_precision"] == 0.0
    assert metrics["component_recall"] == 0.0
    assert metrics["false_components"] == 1
    assert metrics["missed_components"] == 1
    assert metrics["matched_pair_count"] == 0
    assert metrics["matched_iou_mean"] == 0.0


def test_component_metrics_detect_split():
    target = np.ones((1, 1, 3), dtype=bool)
    prediction = np.array([[[True, False, True]]])

    metrics = component_metrics(prediction, target)

    assert metrics["prediction_components"] == 2
    assert metrics["target_components"] == 1
    assert metrics["component_count_difference"] == 1
    assert metrics["split_count"] == 1
    assert metrics["merge_count"] == 0
    assert metrics["component_precision"] == 1.0
    assert metrics["component_recall"] == 1.0
    assert metrics["matched_pair_count"] == 2
    assert metrics["matched_iou_mean"] == pytest.approx(1 / 3)


def test_component_metrics_detect_merge():
    target = np.array([[[True, False, True]]])
    prediction = np.ones_like(target)

    metrics = component_metrics(prediction, target)

    assert metrics["prediction_components"] == 1
    assert metrics["target_components"] == 2
    assert metrics["component_count_difference"] == -1
    assert metrics["split_count"] == 0
    assert metrics["merge_count"] == 1
    assert metrics["component_precision"] == 1.0
    assert metrics["component_recall"] == 1.0
    assert metrics["matched_pair_count"] == 2
    assert metrics["matched_iou_mean"] == pytest.approx(1 / 3)


def test_component_connectivity_six_vs_twenty_six():
    diagonal = np.zeros((2, 2, 2), dtype=bool)
    diagonal[0, 0, 0] = True
    diagonal[1, 1, 1] = True

    six = component_metrics(diagonal, diagonal, connectivity=6)
    twenty_six = component_metrics(diagonal, diagonal, connectivity=26)

    assert six["prediction_components"] == 2
    assert twenty_six["prediction_components"] == 1
    assert six["matched_iou_mean"] == 1.0
    assert twenty_six["matched_iou_mean"] == 1.0


def test_topology_empty_and_solid_shapes():
    empty = np.zeros((3, 3, 3), dtype=bool)
    solid = np.ones((3, 3, 3), dtype=bool)

    assert topology_metrics(empty) == {
        "betti0": 0,
        "betti1": 0,
        "betti2": 0,
        "euler": 0,
    }
    assert topology_metrics(solid) == {
        "betti0": 1,
        "betti1": 0,
        "betti2": 0,
        "euler": 1,
    }


def test_topology_hollow_shape_has_one_cavity():
    hollow = np.ones((3, 3, 3), dtype=bool)
    hollow[1, 1, 1] = False

    assert topology_metrics(hollow) == {
        "betti0": 1,
        "betti1": 0,
        "betti2": 1,
        "euler": 2,
    }


def test_topology_ring_has_one_tunnel():
    ring = np.ones((3, 3, 1), dtype=bool)
    ring[1, 1, 0] = False

    assert topology_metrics(ring) == {
        "betti0": 1,
        "betti1": 1,
        "betti2": 0,
        "euler": 0,
    }


def test_topology_connectivity_six_vs_twenty_six():
    diagonal = np.zeros((2, 2, 2), dtype=bool)
    diagonal[0, 0, 0] = True
    diagonal[1, 1, 1] = True

    six = topology_metrics(diagonal, foreground_connectivity=6)
    twenty_six = topology_metrics(diagonal, foreground_connectivity=26)

    assert six == {"betti0": 2, "betti1": 0, "betti2": 0, "euler": 2}
    assert twenty_six == {
        "betti0": 1,
        "betti1": 0,
        "betti2": 0,
        "euler": 1,
    }


def test_topology_crops_to_valid_bbox_and_ignores_invalid_foreground():
    mask = np.zeros((7, 7, 7), dtype=bool)
    mask[0, 0, 0] = True
    mask[3:5, 3:5, 3:5] = True
    valid = np.zeros_like(mask)
    valid[3:5, 3:5, 3:5] = True

    assert topology_metrics(mask, valid) == {
        "betti0": 1,
        "betti1": 0,
        "betti2": 0,
        "euler": 1,
    }


def test_topology_comparison_reports_signed_differences_and_errors():
    prediction = np.zeros((1, 1, 3), dtype=bool)
    prediction[0, 0, (0, 2)] = True
    target = np.ones_like(prediction)

    comparison = topology_comparison(prediction, target)

    assert comparison["prediction_betti0"] == 2
    assert comparison["target_betti0"] == 1
    assert comparison["betti0_difference"] == 1
    assert comparison["betti0_error"] == 1
    assert comparison["betti_error"] == 1


def test_evaluate_binary_mask_returns_raw_tolerance_and_optional_sections():
    target = np.ones((1, 1, 2), dtype=bool)
    prediction = np.array([[[True, False]]])

    full = evaluate_binary_mask(prediction, target, neighbor_rounds=1)
    voxel_only = evaluate_binary_mask(
        prediction,
        target,
        neighbor_rounds=1,
        include_topology=False,
        include_components=False,
    )

    assert full["raw"]["dice"] == pytest.approx(2 / 3)
    assert full["target_informed_neighbor_corrected"]["dice"] == 1.0
    assert set(full) == {
        "raw",
        "target_informed_neighbor_corrected",
        "neighbor_correction",
        "topology",
        "components",
    }
    assert set(voxel_only) == {
        "raw",
        "target_informed_neighbor_corrected",
        "neighbor_correction",
    }


@pytest.mark.parametrize(
    ("function", "arguments"),
    [
        (voxel_metrics, (np.zeros((2, 2)), np.zeros((2, 2)))),
        (
            voxel_metrics,
            (np.zeros((1, 1, 1)), np.zeros((1, 1, 2))),
        ),
        (topology_metrics, (np.full((1, 1, 1), 2),)),
    ],
)
def test_metrics_reject_non_3d_mismatched_or_nonbinary_masks(function, arguments):
    with pytest.raises((TypeError, ValueError)):
        function(*arguments)

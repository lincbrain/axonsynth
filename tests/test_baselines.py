import json
from unittest.mock import Mock

import nibabel as nib
import numpy as np
import pytest

from paper_analysis import baselines
from paper_analysis.baselines import (
    FRANGI_GAMMA_DESCRIPTION,
    FRANGI_MODE,
    FRANGI_PARAMETERS,
    INTENSITY_THRESHOLDS,
    baseline_parameters,
    bright_frangi_baseline,
    intensity_threshold_baseline,
    normalize_input,
    normalize_input_with_metadata,
)
from paper_analysis.generate_baselines import (
    generate_from_config,
    parse_generation_config,
)


def test_normalization_is_float32_percentile_clip_and_scale():
    volume = np.arange(200, dtype=np.int16).reshape(10, 10, 2)

    normalized, metadata = normalize_input_with_metadata(volume)

    lower, upper = np.percentile(volume.astype(np.float32), [0.5, 99.5])
    expected = np.clip(volume.astype(np.float32), lower, upper)
    expected = ((expected - lower) / (upper - lower)).astype(np.float32)
    expected = np.clip(expected, 0.0, 1.0)
    np.testing.assert_allclose(normalized, expected, rtol=0.0, atol=1e-7)
    assert normalized.dtype == np.float32
    assert float(normalized.min()) == 0.0
    assert float(normalized.max()) == 1.0
    assert metadata == {
        "method": "float32_percentile_clip_and_scale",
        "percentiles": [0.5, 99.5],
        "clip_values": [float(lower), float(upper)],
        "output_range": [0.0, 1.0],
        "dtype": "float32",
        "degenerate_range": False,
    }


def test_normalization_maps_constant_input_to_finite_zeros():
    normalized, metadata = normalize_input_with_metadata(
        np.full((2, 3, 4), 17, dtype=np.uint16)
    )

    np.testing.assert_array_equal(normalized, np.zeros((2, 3, 4), dtype=np.float32))
    assert metadata["clip_values"] == [17.0, 17.0]
    assert metadata["degenerate_range"] is True


@pytest.mark.parametrize(
    "volume, message",
    [
        (np.array([], dtype=np.float32), "empty"),
        (np.array([0.0, np.nan], dtype=np.float32), "finite"),
        (np.array([0.0, np.inf], dtype=np.float32), "finite"),
    ],
)
def test_normalization_rejects_invalid_arrays(volume, message):
    with pytest.raises(ValueError, match=message):
        normalize_input(volume)


@pytest.mark.parametrize(
    "domain, threshold",
    [("human", 0.47), ("macaque", 0.46)],
)
def test_intensity_threshold_is_exact_and_inclusive(domain, threshold):
    score = np.array(
        [
            np.nextafter(np.float32(threshold), np.float32(0.0)),
            threshold,
            np.nextafter(np.float32(threshold), np.float32(1.0)),
        ],
        dtype=np.float32,
    )

    returned_score, mask = intensity_threshold_baseline(score, domain)

    np.testing.assert_array_equal(returned_score, score)
    np.testing.assert_array_equal(mask, np.array([0, 1, 1], dtype=np.uint8))


def test_accepted_parameter_tables_are_exact():
    assert INTENSITY_THRESHOLDS == {"human": 0.47, "macaque": 0.46}
    assert FRANGI_PARAMETERS == {
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
    assert baseline_parameters("frangi", "human") == {
        "alpha": 0.25,
        "beta": 0.5,
        "sigmas": [3, 4, 5, 6, 7],
        "threshold": 0.05,
        "threshold_operator": ">=",
        "black_ridges": False,
        "gamma": None,
        "gamma_behavior": FRANGI_GAMMA_DESCRIPTION,
        "mode": FRANGI_MODE,
        "implementation": "skimage.filters.frangi",
    }


def test_frangi_call_is_bright_only_and_uses_automatic_gamma(monkeypatch):
    normalized = np.linspace(0.0, 1.0, 6, dtype=np.float32).reshape(1, 3, 2)
    filter_score = np.array(
        [[[0.0, 0.009], [0.01, 0.011], [0.5, 1.0]]], dtype=np.float32
    )
    frangi_mock = Mock(return_value=filter_score)
    monkeypatch.setattr(baselines, "frangi", frangi_mock)

    score, mask = bright_frangi_baseline(normalized, "macaque")

    call_args, call_kwargs = frangi_mock.call_args
    np.testing.assert_array_equal(call_args[0], normalized)
    assert call_kwargs == {
        "sigmas": (2, 3, 4, 5, 6, 7, 8, 9, 10),
        "alpha": 0.25,
        "beta": 0.33,
        "gamma": None,
        "black_ridges": False,
        "mode": "reflect",
    }
    np.testing.assert_array_equal(score, filter_score)
    np.testing.assert_array_equal(
        mask,
        np.array([[[0, 0], [1, 1], [1, 1]]], dtype=np.uint8),
    )


def test_generation_preserves_affine_and_records_explicit_domain(tmp_path, monkeypatch):
    raw = np.arange(24, dtype=np.int16).reshape(2, 3, 4)
    affine = np.array(
        [
            [0.7, 0.0, 0.0, 11.0],
            [0.0, 0.8, 0.0, -4.0],
            [0.0, 0.0, 1.5, 2.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    input_path = tmp_path / "name_says_macaque.nii.gz"
    nib.save(nib.Nifti1Image(raw, affine), input_path)
    input_affine = nib.load(input_path).affine
    config_path = tmp_path / "manifest.json"
    config_path.write_text(
        json.dumps(
            {
                "output_dir": "outputs",
                "baselines": ["intensity_threshold", "frangi"],
                "domain": "human",
                "cases": [{"id": "specimen", "input": input_path.name}],
            }
        )
    )

    frangi_score = np.full(raw.shape, 0.05, dtype=np.float32)
    frangi_mock = Mock(return_value=frangi_score)
    monkeypatch.setattr(baselines, "frangi", frangi_mock)
    written = generate_from_config(config_path)

    assert len(written) == 2
    for paths in written:
        score_image = nib.load(paths.score)
        mask_image = nib.load(paths.mask)
        np.testing.assert_array_equal(score_image.affine, input_affine)
        np.testing.assert_array_equal(mask_image.affine, input_affine)
        assert score_image.header.get_data_dtype() == np.dtype(np.float32)
        assert mask_image.header.get_data_dtype() == np.dtype(np.uint8)
        metadata = json.loads(paths.metadata.read_text())
        assert metadata["domain"] == "human"
        assert metadata["normalization"]["percentiles"] == [0.5, 99.5]
        assert metadata["outputs"] == {
            "score": str(paths.score),
            "mask": str(paths.mask),
        }

    intensity_metadata = json.loads(written[0].metadata.read_text())
    assert intensity_metadata["parameters"] == {
        "threshold": 0.47,
        "threshold_operator": ">=",
    }
    normalized = normalize_input(raw)
    np.testing.assert_array_equal(
        np.asanyarray(nib.load(written[0].mask).dataobj),
        (normalized >= 0.47).astype(np.uint8),
    )

    frangi_metadata = json.loads(written[1].metadata.read_text())
    assert frangi_metadata["parameters"]["gamma"] is None
    assert (
        frangi_metadata["parameters"]["gamma_behavior"]
        == "skimage automatic per-volume gamma"
    )
    assert frangi_metadata["parameters"]["black_ridges"] is False


def test_config_requires_selected_baselines_and_configured_domain(tmp_path):
    config_path = tmp_path / "manifest.json"
    config_path.write_text(
        json.dumps(
            {
                "output_dir": "outputs",
                "baselines": ["intensity_threshold"],
                "cases": [{"id": "specimen", "input": "human_in_name.nii.gz"}],
            }
        )
    )
    with pytest.raises(ValueError, match="requires 'domain'"):
        parse_generation_config(config_path)

    config_path.write_text(
        json.dumps(
            {
                "output_dir": "outputs",
                "domain": "human",
                "cases": [{"id": "specimen", "input": "input.nii.gz"}],
            }
        )
    )
    with pytest.raises(ValueError, match="non-empty 'baselines'"):
        parse_generation_config(config_path)

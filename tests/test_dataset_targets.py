import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch


@pytest.fixture(scope='module')
def dataset_module():
    module_path = Path(__file__).parents[1] / 'datagen' / 'axon_subset_dataset.py'
    spec = importlib.util.spec_from_file_location('_axon_subset_dataset_test', module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)

    original_nibabel = sys.modules.get('nibabel')
    sys.modules['nibabel'] = types.ModuleType('nibabel')
    try:
        spec.loader.exec_module(module)
    finally:
        if original_nibabel is None:
            sys.modules.pop('nibabel', None)
        else:
            sys.modules['nibabel'] = original_nibabel
    return module


def test_shell_interior_target_uses_six_connected_instance_geometry(dataset_module):
    labels = np.zeros((5, 5, 5), dtype=np.int32)
    labels[1:4, 1:4, 1:4] = 7

    target = dataset_module.build_shell_interior_target(labels)

    assert target.dtype == np.int64
    assert target[2, 2, 2] == 2
    assert np.count_nonzero(target == 2) == 1
    assert np.count_nonzero(target == 1) == 26
    assert np.all(target[labels == 0] == 0)


def test_touching_axon_instances_remain_shell_at_their_boundary(dataset_module):
    labels = np.zeros((7, 7, 7), dtype=np.int32)
    labels[1:6, 1:6, 1:3] = 1
    labels[1:6, 1:6, 3:6] = 2

    target = dataset_module.build_shell_interior_target(labels)

    assert target[3, 3, 2] == 1
    assert target[3, 3, 3] == 1
    assert target[3, 3, 4] == 2


def test_binary_target_preserves_soft_probabilities(dataset_module):
    dataset = object.__new__(dataset_module.AxonSubsetDataset)
    dataset.segmentation_mode = 'binary'
    labels = torch.tensor([[[[0, 1], [2, 0]]]], dtype=torch.long)
    probabilities = torch.tensor([[[[0.0, 0.2], [0.75, 1.0]]]])

    target = dataset._build_seg_target(labels, probabilities)

    assert target.dtype == torch.float32
    torch.testing.assert_close(target, probabilities)
    assert target[0, 0, 0, 1].item() == pytest.approx(0.2)


def test_three_class_target_excludes_zero_probability_voxels(dataset_module):
    dataset = object.__new__(dataset_module.AxonSubsetDataset)
    dataset.segmentation_mode = 'three_class_shell_interior'
    labels = torch.zeros((1, 5, 5, 5), dtype=torch.long)
    labels[:, 1:4, 1:4, 1:4] = 3
    probabilities = (labels > 0).float()
    probabilities[0, 2, 2, 2] = 0.0

    target = dataset._build_seg_target(labels, probabilities)

    assert target.dtype == torch.int64
    assert target.shape == labels.shape
    assert target[0, 2, 2, 2].item() == 0
    assert not torch.any(target == 2)

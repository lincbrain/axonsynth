"""
Axon Segmentation — Synthetic Data Generation Pipeline

Provides:
- FullDensityUnidirectionalAxon: High-density coherent axon label generation
- ControlledContrastAxonImage:   Image synthesis with guaranteed axon > background contrast
- AxonSubsetDataset:             On-the-fly density-varying dataset for training
- DensityDistribution:           Factory for spatial density distributions
- create_dataloader:             Convenience DataLoader factory
"""

from .axon_labels_full_density import FullDensityUnidirectionalAxon
from .axon_image_controlled_contrast import ControlledContrastAxonImage
from .axon_subset_dataset import (
    AxonSubsetDataset,
    DensityDistribution,
    create_dataloader,
    collate_fn,
    worker_init_fn,
)

__all__ = [
    'FullDensityUnidirectionalAxon',
    'ControlledContrastAxonImage',
    'AxonSubsetDataset',
    'DensityDistribution',
    'create_dataloader',
    'collate_fn',
    'worker_init_fn',
]

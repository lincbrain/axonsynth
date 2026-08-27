"""Dedicated GPU cache builder for train/val synthesis caches.

This path keeps dense-label subset selection in Python but moves the expensive
shared-geometry synthesis for 3-class targets into a single GPU-resident
builder process, avoiding per-worker CPU synthesis.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import torch

from .axon_image_controlled_contrast import ControlledContrastAxonImage
from .axon_subset_dataset import AxonSubsetDataset


def build_gpu_tensor_cache(
    dataset: AxonSubsetDataset,
    *,
    split: str,
    device: torch.device,
    log: logging.Logger,
    max_samples: Optional[int] = None,
    gpu_label_block_size: int = 8,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Materialize a cache from a raw-label dataset using a dedicated GPU synth."""
    if dataset.generate_images:
        raise ValueError('GPU cache builder expects a raw dataset with generate_images=False')
    if dataset.segmentation_mode != 'three_class_shell_interior':
        raise ValueError('GPU cache builder is only supported for 3-class shell/interior mode')
    if device.type != 'cuda':
        raise ValueError(f'GPU cache builder requires a CUDA device, got {device}')

    synth_kwargs = dict(dataset._synth_kwargs)
    synth_kwargs.update(gpu_geometry=True, gpu_label_block_size=gpu_label_block_size)
    synth = ControlledContrastAxonImage.XForm(**synth_kwargs)

    n_total = len(dataset) if max_samples is None else min(len(dataset), max_samples)
    image_batches = []
    seg_batches = []
    synth_times = []
    step_times = []

    t0 = time.time()
    for sample_index in range(n_total):
        t_step = time.time()
        sample = dataset[sample_index]
        label_t = sample['label']
        prob_t = sample['prob']

        label_gpu = label_t.to(device, non_blocking=True)
        prob_gpu = prob_t.to(device, non_blocking=True)
        target_gpu = label_t.to(device, non_blocking=True)

        t_synth = time.time()
        with torch.no_grad():
            image, out_prob, out_label_t = synth(label_gpu, prob_gpu, target_gpu)
            seg_label_t = out_label_t.cpu()
            seg_prob_t = out_prob.cpu()
        torch.cuda.synchronize(device)
        synth_times.append(time.time() - t_synth)

        image_cpu = image.detach().cpu().float()
        seg_cpu = dataset._build_seg_target(seg_label_t, seg_prob_t).detach().cpu()
        image_batches.append(image_cpu)
        seg_batches.append(seg_cpu)
        step_times.append(time.time() - t_step)

        if sample_index == 0 or (sample_index + 1) % 10 == 0 or sample_index + 1 == n_total:
            log.info(
                f'  gpu caching {split}: sample {sample_index + 1:3d}/{n_total} '
                f'| synth={synth_times[-1]:.2f}s total={step_times[-1]:.2f}s'
            )

    images = torch.stack(image_batches, dim=0).contiguous()
    segs = torch.stack(seg_batches, dim=0).contiguous()
    elapsed = time.time() - t0
    n_bytes = images.numel() * images.element_size() + segs.numel() * segs.element_size()
    metrics = {
        'n_samples': int(n_total),
        'elapsed_s': float(elapsed),
        'mean_synth_s': float(sum(synth_times) / len(synth_times)) if synth_times else 0.0,
        'mean_step_s': float(sum(step_times) / len(step_times)) if step_times else 0.0,
        'cache_gib': float(n_bytes / (1024 ** 3)),
    }
    log.info(
        f'Built {split} GPU cache: {images.shape[0]} samples | '
        f'{metrics["cache_gib"]:.2f} GiB | {elapsed:.1f}s '
        f'| mean synth {metrics["mean_synth_s"]:.2f}s/sample'
    )
    return images, segs, metrics

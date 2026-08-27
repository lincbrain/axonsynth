"""GPU-friendly label morphology helpers for the shared-geometry 3-class path.

These helpers approximate the hot cornucopia morphology stages using batched
Torch ops and bounded city-block distance transforms. They are designed for the
case where we keep per-instance axon labels and need to preserve shell/interior
geometry while avoiding CPU worker synthesis.
"""

from __future__ import annotations

import math
from typing import Iterator, Sequence

import torch
import torch.nn.functional as F

from cornucopia.utils.morpho import bounded_distance


def _ensure_shape_tuple(shape: int | Sequence[int], ndim: int) -> tuple[int, ...]:
    if isinstance(shape, int):
        return (shape,) * ndim
    values = tuple(int(value) for value in shape)
    if len(values) != ndim:
        raise ValueError(f'Expected {ndim} shape values, got {values!r}')
    return values


def _sample_smooth_field(
    spatial_shape: Sequence[int],
    max_nodes: int | Sequence[int],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    spatial_shape = tuple(int(value) for value in spatial_shape)
    node_upper = _ensure_shape_tuple(max_nodes, len(spatial_shape))
    small_shape = tuple(
        int(torch.randint(2, upper + 1, (), device=device).item())
        for upper in node_upper
    )
    field = torch.rand((1, 1, *small_shape), device=device, dtype=dtype)
    if small_shape != spatial_shape:
        if len(spatial_shape) == 3:
            mode = 'trilinear'
        elif len(spatial_shape) == 2:
            mode = 'bilinear'
        else:
            mode = 'linear'
        field = F.interpolate(field, size=spatial_shape, mode=mode, align_corners=True)
    return field[0, 0]


def _iter_blocks(label_ids: torch.Tensor, block_size: int) -> Iterator[torch.Tensor]:
    for start in range(0, int(label_ids.numel()), block_size):
        yield label_ids[start:start + block_size]


def _sample_morph_radius(
    min_radius: float,
    max_radius: float,
    spatial_shape: Sequence[int],
    field_shape: int | Sequence[int],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    min_lo, min_hi = sorted((float(min(min_radius, 0.0)), float(max(min_radius, 0.0))))
    max_lo, max_hi = sorted((float(min(max_radius, 0.0)), float(max(max_radius, 0.0))))
    sampled_min = torch.empty((), device=device, dtype=dtype).uniform_(min_lo, min_hi)
    sampled_max = torch.empty((), device=device, dtype=dtype).uniform_(max_lo, max_hi)
    if sampled_max < sampled_min:
        sampled_max = sampled_min
    field = _sample_smooth_field(spatial_shape, field_shape, device=device, dtype=dtype)
    return field.mul(sampled_max - sampled_min).add(sampled_min)


def _sample_shallow_radius(
    min_width: float,
    max_width: float,
    spatial_shape: Sequence[int],
    field_shape: int | Sequence[int],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    sampled_min = torch.empty((), device=device, dtype=dtype).uniform_(0.0, float(min_width))
    sampled_max = torch.empty((), device=device, dtype=dtype).uniform_(0.0, float(max_width))
    if sampled_max < sampled_min:
        sampled_max = sampled_min
    field = _sample_smooth_field(spatial_shape, field_shape, device=device, dtype=dtype)
    # Match cornucopia's convention: shell occupies radii in [-max_width, -min_width].
    return field.mul(sampled_min - sampled_max).sub(sampled_min)


def smooth_random_morph_labels(
    labels: torch.Tensor,
    *,
    label_select_prob: float = 0.5,
    min_radius: float = -4.0,
    max_radius: float = 4.0,
    field_shape: int | Sequence[int] = 128,
    block_size: int = 8,
) -> torch.Tensor:
    """Approximate RandomSmoothMorphoLabelTransform on GPU-friendly Torch ops.

    The default parameters mirror the current axon synthesis path.
    """
    if labels.ndim != 4 or labels.shape[0] != 1:
        raise ValueError(f'Expected label tensor of shape (1, D, H, W), got {tuple(labels.shape)}')

    label_map = labels[0]
    foreground = torch.unique(label_map)
    foreground = foreground[foreground != 0]
    if foreground.numel() == 0:
        return labels.clone()

    selected_mask = torch.rand(foreground.numel(), device=label_map.device) < float(label_select_prob)
    selected_labels = set(int(value) for value in foreground[selected_mask].tolist())

    spatial_shape = tuple(int(value) for value in label_map.shape)
    max_abs_radius = int(math.ceil(max(abs(float(min_radius)), abs(float(max_radius))))) + 1
    dtype = torch.float32

    background_mask = (label_map == 0).unsqueeze(0)
    best_distance = bounded_distance(background_mask, nb_iter=max_abs_radius, dim=label_map.ndim).to(dtype)[0]
    best_labels = torch.zeros_like(label_map)

    for label_block in _iter_blocks(foreground, block_size):
        masks = label_map.unsqueeze(0).eq(label_block[:, None, None, None])
        block_distance = bounded_distance(masks, nb_iter=max_abs_radius, dim=label_map.ndim).to(dtype)
        radius_block = torch.zeros_like(block_distance)
        for block_index, label_value in enumerate(label_block.tolist()):
            if int(label_value) not in selected_labels:
                continue
            radius_block[block_index] = _sample_morph_radius(
                min_radius,
                max_radius,
                spatial_shape,
                field_shape,
                device=label_map.device,
                dtype=dtype,
            )

        block_distance = block_distance.sub(radius_block)
        min_distance, min_index = block_distance.min(dim=0)
        candidate_labels = label_block[min_index]
        update = min_distance < best_distance
        best_distance = torch.where(update, min_distance, best_distance)
        best_labels = torch.where(update, candidate_labels, best_labels)

    return best_labels.unsqueeze(0)


def smooth_random_shallow_labels(
    labels: torch.Tensor,
    *,
    apply_prob: float = 0.3,
    label_select_prob: float = 0.5,
    min_width: float = 1.0,
    max_width: float = 3.0,
    field_shape: int | Sequence[int] = 128,
    block_size: int = 8,
) -> torch.Tensor:
    """Approximate RandomSmoothShallowLabelTransform on GPU-friendly Torch ops."""
    if labels.ndim != 4 or labels.shape[0] != 1:
        raise ValueError(f'Expected label tensor of shape (1, D, H, W), got {tuple(labels.shape)}')
    if torch.rand((), device=labels.device).item() >= float(apply_prob):
        return labels.clone()

    label_map = labels[0]
    foreground = torch.unique(label_map)
    foreground = foreground[foreground != 0]
    if foreground.numel() == 0:
        return labels.clone()

    selected_mask = torch.rand(foreground.numel(), device=label_map.device) < float(label_select_prob)
    selected = foreground[selected_mask]
    if selected.numel() == 0:
        return labels.clone()

    spatial_shape = tuple(int(value) for value in label_map.shape)
    max_width_iter = int(math.ceil(float(max_width))) + 1
    dtype = torch.float32
    inf = torch.tensor(float('inf'), device=label_map.device, dtype=dtype)

    best_shell_distance = torch.full(label_map.shape, float('inf'), device=label_map.device, dtype=dtype)
    shell_labels = torch.zeros_like(label_map)
    interior_mask = torch.zeros_like(label_map, dtype=torch.bool)

    for label_block in _iter_blocks(selected, block_size):
        masks = label_map.unsqueeze(0).eq(label_block[:, None, None, None])
        block_distance = bounded_distance(masks, nb_iter=max_width_iter, dim=label_map.ndim).to(dtype)
        radius_block = torch.zeros_like(block_distance)
        for block_index in range(label_block.numel()):
            radius_block[block_index] = _sample_shallow_radius(
                min_width,
                max_width,
                spatial_shape,
                field_shape,
                device=label_map.device,
                dtype=dtype,
            )
        shell_mask = (block_distance < 0) & (block_distance > radius_block)
        interior_mask |= (block_distance < radius_block).any(dim=0)

        shell_distance = block_distance.masked_fill(~shell_mask, inf)
        min_shell_distance, min_shell_index = shell_distance.min(dim=0)
        candidate_labels = label_block[min_shell_index]
        update = min_shell_distance < best_shell_distance
        best_shell_distance = torch.where(update, min_shell_distance, best_shell_distance)
        shell_labels = torch.where(update, candidate_labels, shell_labels)

    background_labels = [
        int(label_value)
        for label_value in torch.unique(label_map).tolist()
        if int(label_value) not in {int(value) for value in selected.tolist()}
    ]
    if len(background_labels) <= 1:
        return shell_labels.unsqueeze(0)

    fill_distance = torch.full(label_map.shape, float('inf'), device=label_map.device, dtype=dtype)
    fill_labels = torch.zeros_like(label_map)
    background_tensor = torch.tensor(background_labels, device=label_map.device, dtype=label_map.dtype)
    for label_block in _iter_blocks(background_tensor, block_size):
        masks = label_map.unsqueeze(0).eq(label_block[:, None, None, None])
        block_distance = bounded_distance(masks, nb_iter=max_width_iter, dim=label_map.ndim).to(dtype)
        min_fill_distance, min_fill_index = block_distance.min(dim=0)
        candidate_labels = label_block[min_fill_index]
        update = min_fill_distance < fill_distance
        fill_distance = torch.where(update, min_fill_distance, fill_distance)
        fill_labels = torch.where(update, candidate_labels, fill_labels)

    output = torch.where(shell_labels != 0, shell_labels, fill_labels)
    output = torch.where(interior_mask, fill_labels, output)
    return output.unsqueeze(0)
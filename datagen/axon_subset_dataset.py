"""
Axon Subset Dataset

PyTorch Dataset that loads pre-generated dense label volumes and
generates random axon subsets with varying spatial density distributions
on-the-fly, enabling unlimited training variation from a fixed set of
expensive label volumes.

Spatial density distributions supported:
    linear, sigmoid, gaussian, radial, uniform

Typical usage
-------------
    from datagen import AxonSubsetDataset, create_dataloader

    loader = create_dataloader(
        label_dir='/path/to/dense_labels',
        batch_size=4,
        num_workers=4,
        apply_density_curve=True,
        generate_images=True,
    )
"""
import random as pyrandom
import logging as _logging
import os
import time as _time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

_log = _logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Worker-local synthesizer
#
# Each DataLoader worker is a separate process, so CUDA models cannot be
# shared from the parent.  We lazily create one CPU ControlledContrastAxonImage
# per worker process on first use and reuse it for all subsequent __getitem__
# calls in that worker.
# ---------------------------------------------------------------------------
_worker_synth: Dict[frozenset, object] = {}


def _progress_enabled() -> bool:
    value = os.environ.get('AXON_DATASET_PROGRESS', '')
    return value.lower() not in {'', '0', 'false', 'no'}


def _progress(message: str) -> None:
    if _progress_enabled():
        print(f'[dataset-progress pid={os.getpid()}] {message}', flush=True)


def _get_or_create_synth(synth_kwargs: dict):
    """Return the process-local synthesizer for the given kwargs, creating it on first call.

    Keyed by synth kwargs so multiple datasets with different synthesis params
    can coexist in the same worker process without overwriting each other.
    """
    key = frozenset(synth_kwargs.items())
    if key not in _worker_synth:
        from datagen.axon_image_controlled_contrast import ControlledContrastAxonImage
        _progress(f'creating worker-local synth with kwargs={dict(sorted(synth_kwargs.items()))}')
        _worker_synth[key] = ControlledContrastAxonImage.XForm(**synth_kwargs)
    return _worker_synth[key]


def worker_init_fn(worker_id: int) -> None:
    """Seed per-worker randomness so every worker produces independent augmentations.

    PyTorch forks workers from the same parent state, so without explicit
    seeding all workers generate identical random subsets and density fields.
    Pass this to DataLoader as ``worker_init_fn=worker_init_fn``.
    """
    seed = (torch.initial_seed() + worker_id) % (2 ** 32)
    np.random.seed(seed)
    pyrandom.seed(seed)
    _progress(f'worker_init worker_id={worker_id} seed={seed}')


def collate_fn(batch: list) -> dict:
    """Batch only tensor values; non-tensor metadata is dropped.

    PyTorch's default collator cannot handle dicts containing strings,
    nested dicts, or mixed types.  This collator stacks only the tensor
    entries that are present in every sample.
    """
    elem = batch[0]
    return {
        key: torch.stack([sample[key] for sample in batch])
        for key in elem
        if torch.is_tensor(elem[key])
    }


class DensityDistribution:
    """Factory for 3-D spatial keep-probability fields."""

    @staticmethod
    def linear(
        shape: Tuple[int, ...],
        axis: int = 2,
        low: float = 0.1,
        high: float = 1.0,
        ascending: bool = True,
    ) -> np.ndarray:
        """Linear gradient along one axis."""
        values = np.linspace(low, high, shape[axis])
        if not ascending:
            values = values[::-1]
        slices = [np.newaxis, np.newaxis, np.newaxis]
        slices[axis] = slice(None)
        return np.broadcast_to(values[tuple(slices)], shape).copy()

    @staticmethod
    def sigmoid(
        shape: Tuple[int, ...],
        axis: int = 2,
        low: float = 0.1,
        high: float = 1.0,
        center: float = 0.5,
        steepness: float = 10.0,
        ascending: bool = True,
    ) -> np.ndarray:
        """Sigmoid transition along one axis."""
        t = np.linspace(0, 1, shape[axis])
        values = low + (high - low) / (1 + np.exp(-steepness * (t - center)))
        if not ascending:
            values = values[::-1]
        slices = [np.newaxis, np.newaxis, np.newaxis]
        slices[axis] = slice(None)
        return np.broadcast_to(values[tuple(slices)], shape).copy()

    @staticmethod
    def gaussian(
        shape: Tuple[int, ...],
        axis: int = 2,
        low: float = 0.1,
        high: float = 1.0,
        center: float = 0.5,
        sigma: float = 0.2,
    ) -> np.ndarray:
        """Gaussian peak along one axis."""
        t = np.linspace(0, 1, shape[axis])
        values = low + (high - low) * np.exp(-((t - center) ** 2) / (2 * sigma ** 2))
        slices = [np.newaxis, np.newaxis, np.newaxis]
        slices[axis] = slice(None)
        return np.broadcast_to(values[tuple(slices)], shape).copy()

    @staticmethod
    def radial(
        shape: Tuple[int, ...],
        low: float = 0.1,
        high: float = 1.0,
        center_frac: Tuple[float, ...] = (0.5, 0.5, 0.5),
        invert: bool = False,
    ) -> np.ndarray:
        """Radial gradient from a centre point."""
        center = np.array([c * s for c, s in zip(center_frac, shape)])
        coords = np.stack(
            np.meshgrid(
                np.arange(shape[0]),
                np.arange(shape[1]),
                np.arange(shape[2]),
                indexing='ij',
            ),
            axis=-1,
        ).astype(float)
        dist     = np.linalg.norm(coords - center, axis=-1)
        max_dist = np.linalg.norm(np.array(shape) / 2)
        if invert:
            field = low + (high - low) * (dist / max_dist)
        else:
            field = high - (high - low) * (dist / max_dist)
        return field.clip(low, high)

    @staticmethod
    def uniform(shape: Tuple[int, ...], value: float = 1.0) -> np.ndarray:
        """Constant keep-probability."""
        return np.full(shape, value, dtype=np.float32)

    @classmethod
    def random(
        cls,
        shape: Tuple[int, ...],
        low_range: Tuple[float, float] = (0.05, 0.4),
        high_range: Tuple[float, float] = (0.6, 1.0),
        uniform_range: Tuple[float, float] = (0.3, 1.0),
    ) -> Tuple[np.ndarray, dict]:
        """Sample a random distribution type and parameters.

        Returns
        -------
        field  : (D, H, W) float32 array of per-voxel keep probabilities
        config : dict describing the chosen distribution (for logging)
        """
        kind      = pyrandom.choice(['linear', 'sigmoid', 'gaussian', 'radial', 'uniform'])
        axis      = pyrandom.randint(0, 2)
        low       = pyrandom.uniform(*low_range)
        high      = pyrandom.uniform(*high_range)
        ascending = pyrandom.choice([True, False])
        config    = dict(type=kind, axis=axis, low=low, high=high, ascending=ascending)

        if kind == 'linear':
            field = cls.linear(shape, axis, low, high, ascending)
        elif kind == 'sigmoid':
            center     = pyrandom.uniform(0.3, 0.7)
            steepness  = pyrandom.uniform(5, 20)
            config.update(center=center, steepness=steepness)
            field = cls.sigmoid(shape, axis, low, high, center, steepness, ascending)
        elif kind == 'gaussian':
            center = pyrandom.uniform(0.2, 0.8)
            sigma  = pyrandom.uniform(0.1, 0.4)
            config.update(center=center, sigma=sigma)
            field = cls.gaussian(shape, axis, low, high, center, sigma)
        elif kind == 'radial':
            center_frac = tuple(pyrandom.uniform(0.3, 0.7) for _ in range(3))
            invert      = pyrandom.choice([True, False])
            config.update(center_frac=center_frac, invert=invert)
            field = cls.radial(shape, low, high, center_frac, invert)
        else:   # uniform
            value = pyrandom.uniform(*uniform_range)
            config['value'] = value
            field = cls.uniform(shape, value)

        return field, config


def collapse_labels(label: torch.Tensor, n_groups: int = 8) -> torch.Tensor:
    """Randomly remap unique axon IDs → 1..n_groups, keeping background=0.

    Cornucopia iterates over every unique label ID for morphological ops —
    collapsing 3000+ axon IDs to n_groups gives a massive speedup.
    """
    out = torch.zeros_like(label)
    unique = label.unique()
    unique = unique[unique > 0]
    if unique.numel() == 0:
        return out
    groups = torch.randint(1, n_groups + 1, (unique.numel(),),
                           device=label.device, dtype=label.dtype)
    max_id = int(unique.max().item()) + 1
    lut = torch.zeros(max_id, device=label.device, dtype=label.dtype)
    lut[unique] = groups
    out = lut[label.clamp(0, max_id - 1)]
    return out


def build_shell_interior_target(labels: np.ndarray) -> np.ndarray:
    """Derive a 3-class target: 0=background, 1=shell, 2=interior.

    Interior is defined as voxels whose 6-connected neighbors all belong to the
    same axon instance. Everything else in the foreground becomes shell.
    """
    if labels.ndim != 3:
        raise ValueError(f'Expected 3D label volume, got shape={labels.shape}')

    padded = np.pad(labels, 1, mode='constant', constant_values=0)
    center = padded[1:-1, 1:-1, 1:-1]
    interior = center > 0
    interior &= padded[:-2, 1:-1, 1:-1] == center
    interior &= padded[2:, 1:-1, 1:-1] == center
    interior &= padded[1:-1, :-2, 1:-1] == center
    interior &= padded[1:-1, 2:, 1:-1] == center
    interior &= padded[1:-1, 1:-1, :-2] == center
    interior &= padded[1:-1, 1:-1, 2:] == center

    shell = (center > 0) & ~interior

    target = np.zeros(labels.shape, dtype=np.int64)
    target[shell] = 1
    target[interior] = 2
    return target


class AxonSubsetDataset(Dataset):
    """On-the-fly axon subset dataset for 3-D UNet training.

    All dense label volumes are loaded into memory at construction time
    so that ``__getitem__`` never touches disk.  Image synthesis is
    performed in each DataLoader worker using a process-local CPU instance
    of ``ControlledContrastAxonImage``.

    Output tensors have shape ``(1, D, H, W)`` (channel-first, single
    channel) matching MONAI convention.  After batching the DataLoader
    returns ``(B, 1, D, H, W)`` tensors.

    Parameters
    ----------
    label_dir : str or Path
        Directory with ``*_label.nii.gz`` / ``*_prob.nii.gz`` pairs.
    subset_fraction : (float, float)
        Uniform range for the axon keep-fraction when
        ``apply_density_curve`` is False.
    density_low_range : (float, float)
        Sampling range for the low end of spatial keep-probability fields.
    density_high_range : (float, float)
        Sampling range for the high end of spatial keep-probability fields.
    density_uniform_range : (float, float)
        Sampling range for uniform keep-probability fields.
    apply_density_curve : bool
        Apply a random spatial keep-probability field instead of a
        flat per-volume fraction.
    generate_images : bool
        Synthesise images on-the-fly.  Requires synthspline + cornucopia.
    transform : callable, optional
        Additional transform applied to the output dict after synthesis.
    num_samples_per_volume : int
        Number of random subsets drawn from each source volume per epoch.
    max_volumes : int, optional
        Cap the number of label volumes loaded. ``None`` means use all.
    n_label_groups : int
        Collapse unique axon IDs to N groups before synthesis
        (speeds up cornucopia morphological ops).
    fibers_lower_range : (float, float)
        Uniform range for the axon intensity floor (passed to synthesizer).
    background_upper_range : (float, float)
        Uniform range for the background intensity ceiling.
    background : float
        Probability of adding background structures (passed to synthesizer).
    segmentation_mode : str
        Target construction mode. ``'binary'`` uses the current soft
        foreground-probability target; ``'three_class_shell_interior'`` derives
        hard labels with classes background, shell, and interior.
    """

    def __init__(
        self,
        label_dir: Union[str, Path],
        subset_fraction: Tuple[float, float] = (0.3, 0.8),
        density_low_range: Tuple[float, float] = (0.05, 0.4),
        density_high_range: Tuple[float, float] = (0.6, 1.0),
        density_uniform_range: Tuple[float, float] = (0.3, 1.0),
        apply_density_curve: bool = True,
        generate_images: bool = True,
        transform: Optional[Callable] = None,
        num_samples_per_volume: int = 100,
        max_volumes: Optional[int] = None,
        n_label_groups: int = 8,
        fibers_lower_range: Tuple[float, float] = (0.3, 0.5),
        background_upper_range: Tuple[float, float] = (0.2, 0.4),
        background: float = 0.5,
        segmentation_mode: str = 'binary',
        split: str = 'train',
        val_fraction: float = 0.2,
    ):
        self.label_dir              = Path(label_dir)
        self.subset_fraction        = subset_fraction
        self.density_low_range      = density_low_range
        self.density_high_range     = density_high_range
        self.density_uniform_range  = density_uniform_range
        self.apply_density_curve    = apply_density_curve
        self.generate_images        = generate_images
        self.transform              = transform
        self.num_samples_per_volume = num_samples_per_volume
        self.n_label_groups         = n_label_groups
        self.segmentation_mode      = segmentation_mode

        if self.segmentation_mode not in {'binary', 'three_class_shell_interior'}:
            raise ValueError(
                'segmentation_mode must be one of '
                "{'binary', 'three_class_shell_interior'}, got "
                f'{self.segmentation_mode!r}'
            )

        # Synthesizer kwargs — model is created lazily per worker process.
        self._synth_kwargs = dict(
            background=background,
            fibers_lower_range=fibers_lower_range,
            background_upper_range=background_upper_range,
            clean_target_lab=(segmentation_mode == 'three_class_shell_interior'),
        )

        all_label_files: List[Path] = sorted(self.label_dir.glob('*_label.nii.gz'))
        if not all_label_files:
            raise ValueError(f'No *_label.nii.gz files found in {label_dir}')
        if max_volumes is not None:
            all_label_files = all_label_files[:max_volumes]

        # --- Deterministic train / val split (fix #5) ---
        n_val = max(1, int(len(all_label_files) * val_fraction))
        if split == 'val':
            label_files = all_label_files[:n_val]
        elif split == 'train':
            label_files = all_label_files[n_val:]
        else:
            raise ValueError(f"split must be 'train' or 'val', got {split!r}")

        if not label_files:
            raise ValueError(
                f"Split '{split}' has no volumes "
                f"(total={len(all_label_files)}, n_val={n_val})"
            )

        self.split = split

        # --- Cache all volumes in memory (fix #1) ---
        print(f'[{split}] Loading {len(label_files)} label volumes into memory...')
        self._volumes: List[Tuple[np.ndarray, np.ndarray]] = []
        for lf in label_files:
            pf = lf.parent / lf.name.replace('_label', '_prob')
            labels = nib.load(lf).get_fdata().astype(np.int32)
            prob   = nib.load(pf).get_fdata().astype(np.float32)
            self._volumes.append((labels, prob))
        print(f'Cached {len(self._volumes)} volumes | '
              f'{num_samples_per_volume} samples/vol | '
              f'total={len(self)}')

    def __len__(self) -> int:
        return len(self._volumes) * self.num_samples_per_volume

    def _apply_subset(
        self,
        labels: np.ndarray,
        prob: np.ndarray,
        keep_prob_field: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Vectorized axon subset selection."""
        unique_axons = np.unique(labels)
        unique_axons = unique_axons[unique_axons > 0]
        n_total = len(unique_axons)

        if n_total == 0:
            return labels.copy(), prob.copy(), dict(n_total=0, n_kept=0, fraction=0.0)

        if keep_prob_field is not None:
            # Per-axon probability = mean field value over axon voxels.
            # Vectorized via bincount: 2 passes over the volume instead of
            # one pass per axon (~3000+ passes → catastrophically slow).
            label_flat = labels.ravel()
            field_flat = keep_prob_field.ravel().astype(np.float64)
            max_label  = int(labels.max()) + 1
            sums   = np.bincount(label_flat, weights=field_flat, minlength=max_label)
            counts = np.bincount(label_flat, minlength=max_label)
            per_axon_p = sums[unique_axons] / np.maximum(counts[unique_axons], 1)
            keep_mask = np.random.random(n_total) < per_axon_p
        else:
            frac      = pyrandom.uniform(*self.subset_fraction)
            n_keep    = max(1, int(n_total * frac))
            perm      = np.random.permutation(n_total)
            keep_mask = np.zeros(n_total, dtype=bool)
            keep_mask[perm[:n_keep]] = True

        # Vectorized removal via lookup table (no Python loop over voxels)
        label_max = int(labels.max()) + 1
        keep_lut  = np.zeros(label_max, dtype=bool)
        keep_lut[0] = True                          # background always kept
        keep_lut[unique_axons[keep_mask]] = True    # kept axons

        keep_voxels   = keep_lut[labels]
        subset_labels = np.where(keep_voxels, labels, 0).astype(np.int32)
        subset_prob   = np.where(keep_voxels, prob, 0.0).astype(np.float32)

        n_kept = int(keep_mask.sum())
        return subset_labels, subset_prob, dict(
            n_total=n_total,
            n_kept=n_kept,
            fraction=n_kept / n_total,
        )

    def _build_seg_target(
        self,
        labels: torch.Tensor,
        prob: torch.Tensor,
    ) -> torch.Tensor:
        if self.segmentation_mode == 'binary':
            return prob.float()

        labels_np = labels.squeeze(0).cpu().numpy().copy()
        prob_np = prob.squeeze(0).cpu().numpy()
        labels_np[prob_np <= 0] = 0
        target_np = build_shell_interior_target(labels_np)
        return torch.from_numpy(target_np).unsqueeze(0).long()

    def __getitem__(self, idx: int) -> dict:
        _t_total = _time.monotonic()
        vol_idx      = idx // self.num_samples_per_volume
        labels, prob = self._volumes[vol_idx]          # from in-memory cache
        _progress(f'idx={idx} vol={vol_idx} start shape={labels.shape}')

        # Deterministic validation: same idx always produces identical sample (fix #6)
        if self.split == 'val':
            seed = idx % (2 ** 31)
            np.random.seed(seed)
            pyrandom.seed(seed)
            torch.manual_seed(seed)

        _t0 = _time.monotonic()
        keep_prob_field = density_config = None
        if self.apply_density_curve:
            keep_prob_field, density_config = DensityDistribution.random(
                labels.shape,
                low_range=self.density_low_range,
                high_range=self.density_high_range,
                uniform_range=self.density_uniform_range,
            )

        subset_labels, subset_prob, subset_info = self._apply_subset(
            labels, prob, keep_prob_field
        )
        _t_subset = _time.monotonic() - _t0
        _progress(
            f'idx={idx} vol={vol_idx} subset_ready '
            f'fg_voxels={(subset_labels > 0).sum()} subset_time={_t_subset:.2f}s '
            f'n_kept={subset_info.get("n_kept")} n_total={subset_info.get("n_total")}'
        )

        # (1, D, H, W)  —  channel-first, MONAI convention
        label_t = torch.from_numpy(subset_labels).unsqueeze(0).long()
        prob_t  = torch.from_numpy(subset_prob).unsqueeze(0).float()

        result: dict = dict(
            # metadata — not collated into batches, useful for debugging
            density_config=density_config,
            subset_info=subset_info,
        )

        if self.generate_images:
            # -- Filter C: ensure subset has enough foreground mass for
            #    cornucopia erosion/shallow ops.  If too sparse, re-draw
            #    (up to 5×) from the same volume with a fresh density field.
            _MIN_FG_VOXELS = 200  # 128³ vol has 2M voxels; 200 is ~0.01%
            _fg_redraws = 0
            for _redraw in range(5):
                fg_count = int((label_t > 0).sum())
                if fg_count >= _MIN_FG_VOXELS:
                    break
                _fg_redraws += 1
                _progress(
                    f'idx={idx} vol={vol_idx} redraw={_fg_redraws} '
                    f'fg_count={fg_count} below_min={_MIN_FG_VOXELS}'
                )
                keep_prob_field, _ = DensityDistribution.random(
                    labels.shape,
                    low_range=self.density_low_range,
                    high_range=self.density_high_range,
                    uniform_range=self.density_uniform_range,
                )
                s_labels, s_prob, subset_info = self._apply_subset(
                    labels, prob, keep_prob_field)
                label_t = torch.from_numpy(s_labels).unsqueeze(0).long()
                prob_t  = torch.from_numpy(s_prob).unsqueeze(0).float()
            fg_count = int((label_t > 0).sum())
            _progress(
                f'idx={idx} vol={vol_idx} redraw_done total_redraws={_fg_redraws} '
                f'final_fg_count={fg_count}'
            )

            # Collapse 3000+ axon IDs → n_groups before cornucopia morphological
            # ops — gives ~(N_axons/n_groups)x speedup in synthesis.
            _t1 = _time.monotonic()
            if self.segmentation_mode == 'binary':
                synth_label_t = collapse_labels(label_t, n_groups=self.n_label_groups)
            else:
                synth_label_t = label_t
            _t_collapse = _time.monotonic() - _t1

            # Worker-local CPU synthesizer (fix #2)
            synth = _get_or_create_synth(self._synth_kwargs)

            # Cornucopia loops are capped (fix B) so synthesis always
            # completes.  No timeout needed — just run and catch any
            # unexpected exceptions.
            image = out_prob = out_label_t = None
            primary_error = None
            alternate_error = None
            seg_label_t = label_t
            seg_prob_t = prob_t
            _t2 = _time.monotonic()
            _progress(f'idx={idx} vol={vol_idx} synth_start fg_count={fg_count}')
            try:
                with torch.no_grad():
                    if self.segmentation_mode == 'binary':
                        image, out_prob = synth(synth_label_t, prob_t)
                    else:
                        image, out_prob, out_label_t = synth(synth_label_t, prob_t, label_t)
                        seg_label_t = out_label_t
                    seg_prob_t = out_prob
                _progress(
                    f'idx={idx} vol={vol_idx} synth_done elapsed={_time.monotonic() - _t2:.2f}s '
                    f'image_shape={tuple(image.shape) if image is not None else None}'
                )
            except Exception as exc:
                primary_error = exc
                _log.warning(f'idx={idx} vol={vol_idx}: synth error: {exc}')
                _progress(f'idx={idx} vol={vol_idx} synth_error={exc!r}')

            _used_alt_vol = False
            _used_raw_fallback = False
            if image is None:
                # Primary synthesis failed — try a different random volume.
                _used_alt_vol = True
                alt_idx = np.random.randint(0, len(self._volumes))
                _progress(f'idx={idx} vol={vol_idx} primary_failed alt_vol={alt_idx}')
                alt_labels, alt_prob = self._volumes[alt_idx]
                alt_s, alt_p, _ = self._apply_subset(alt_labels, alt_prob, None)
                alt_lt = torch.from_numpy(alt_s).unsqueeze(0).long()
                alt_pt = torch.from_numpy(alt_p).unsqueeze(0).float()
                if self.segmentation_mode == 'binary':
                    alt_synth_label_t = collapse_labels(alt_lt, n_groups=self.n_label_groups)
                else:
                    alt_synth_label_t = alt_lt
                try:
                    with torch.no_grad():
                        if self.segmentation_mode == 'binary':
                            image, out_prob = synth(alt_synth_label_t, alt_pt)
                        else:
                            image, out_prob, out_label_t = synth(alt_synth_label_t, alt_pt, alt_lt)
                    _progress(
                        f'idx={idx} vol={vol_idx} alt_synth_done alt_vol={alt_idx} '
                        f'image_shape={tuple(image.shape) if image is not None else None}'
                    )
                    if image is not None:
                        seg_prob_t = out_prob
                        if self.segmentation_mode != 'binary':
                            seg_label_t = out_label_t
                except Exception as exc:
                    alternate_error = exc
                    _progress(f'idx={idx} vol={vol_idx} alt_synth_error alt_vol={alt_idx}')
                if image is None:
                    raise RuntimeError(
                        f'Image synthesis failed for primary volume {vol_idx} and '
                        f'alternate volume {alt_idx}; refusing to use the target '
                        f'probability as the network input. Primary error: '
                        f'{primary_error!r}; alternate error: {alternate_error!r}'
                    ) from alternate_error

            _t_synth = _time.monotonic() - _t2
            _t_total_elapsed = _time.monotonic() - _t_total

            # Log warnings only — per-sample INFO stripped for production
            if _used_raw_fallback or _used_alt_vol:
                _log.warning(
                    f'[sample] idx={idx} vol={vol_idx} | '
                    f'synth={_t_synth:.1f}s | alt_vol={_used_alt_vol} '
                    f'raw_fallback={_used_raw_fallback}')

            # 'image': network input  |  'seg': segmentation target
            result['image'] = image      # (1, D, H, W)
            result['seg']   = self._build_seg_target(seg_label_t, seg_prob_t)
            _progress(
                f'idx={idx} vol={vol_idx} return image_shape={tuple(result["image"].shape)} '
                f'seg_shape={tuple(result["seg"].shape)} total_elapsed={_t_total_elapsed:.2f}s'
            )
        else:
            result['label'] = label_t
            result['prob']  = prob_t
            _progress(f'idx={idx} vol={vol_idx} return_raw label_shape={tuple(label_t.shape)}')

        if self.transform is not None:
            result = self.transform(result)

        return result


def create_dataloader(
    label_dir: Union[str, Path],
    batch_size: int = 2,
    num_workers: int = 10,
    pin_memory: bool = True,
    shuffle: Optional[bool] = None,
    drop_last: bool = True,
    persistent_workers: Optional[bool] = None,
    **dataset_kwargs,
) -> torch.utils.data.DataLoader:
    """Convenience factory: AxonSubsetDataset → DataLoader.

    Defaults are tuned for a single-GPU training node with::

        #SBATCH --gres=gpu:1
        #SBATCH --cpus-per-task=12   # 10 workers + 1 main + 1 spare

    ``batch_size=2`` is chosen for 128³ volumes which are memory-heavy on GPU.
    Raise ``num_workers`` (and ``--cpus-per-task``) if ``nvidia-smi dmon``
    shows GPU utilisation dropping below ~85% between batches.

    Applies:
    - ``worker_init_fn``     — independent randomness per worker (fix #4)
    - ``collate_fn``         — tensor-only batching, drops metadata (fix #3)
    - ``persistent_workers`` — workers stay alive between epochs, preserving
                               the in-memory volume cache
    - ``shuffle``/``drop_last`` can be overridden per split so validation
      semantics stay fixed and training can still drop partial batches.
    """
    dataset = AxonSubsetDataset(label_dir, **dataset_kwargs)
    if shuffle is None:
        shuffle = (dataset.split == 'train')
    if persistent_workers is None:
        persistent_workers = (num_workers > 0)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        worker_init_fn=worker_init_fn,
        persistent_workers=persistent_workers,
        drop_last=drop_last,
        prefetch_factor=(4 if num_workers > 0 else None),
    )

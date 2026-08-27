"""
Axon Image Synthesis — Controlled Contrast

Extends SynthAxonImage with guaranteed contrast separation between
axons (foreground) and background, as recommended by the Balbasty group:

    Background rescaled to  [0,            Uniform(background_upper_range)]
    Axons      rescaled to  [Uniform(fibers_lower_range),  1.0            ]

A small overlap between the two ranges is intentional — it mimics real
data where some bright background structures approach the intensity of
dim axons. Both bounds are re-sampled independently each forward call,
giving contrast variation across training samples.

Note: uses a local _minmax_rescale() helper because cc.MinMaxTransform
is not available in cornucopia 0.3.0.  The dev version (0.4.0) has a
broken convnd for iso kernels, so we pin to 0.3.0.
"""

import math as pymath
import os
import random as pyrandom
import logging
import time as _time

import torch
import cornucopia as cc

from synthspline.imagezoo import AutoBatchTransform
from .gpu_label_ops import smooth_random_morph_labels, smooth_random_shallow_labels

_log = logging.getLogger(__name__)


def _progress_enabled() -> bool:
    for key in ('AXON_SYNTH_PROGRESS', 'AXON_DATASET_PROGRESS'):
        value = os.environ.get(key, '')
        if value.lower() not in {'', '0', 'false', 'no'}:
            return True
    return False


def _progress(message: str) -> None:
    if _progress_enabled():
        print(f'[synth-progress pid={os.getpid()}] {message}', flush=True)


def _minmax_rescale(x: torch.Tensor, vmin: float, vmax: float) -> torch.Tensor:
    """Linearly rescale tensor range [x.min(), x.max()] → [vmin, vmax].

    Falls back to filling with vmin when the tensor is constant.
    """
    x_min = x.min()
    x_max = x.max()
    if x_max == x_min:
        return x.new_full(x.shape, vmin)
    return (x - x_min) / (x_max - x_min) * (vmax - vmin) + vmin


class ControlledContrastAxonImage(AutoBatchTransform):
    """Synthesize an LSM-like image from axon labels with controlled contrast.

    Axons are always brighter than the background (with a small configurable
    overlap to mimic realistic partial-volume effects).

    Parameters
    ----------
    background : float
        Probability of adding random background structures.
        0 = plain dark background.
    fibers_lower_range : (float, float)
        Uniform range for the axon intensity floor.
        Axon intensities → [Uniform(fibers_lower_range), 1.0].
        Default: (0.3, 0.5)
    background_upper_range : (float, float)
        Uniform range for the background intensity ceiling.
        Background intensities → [0, Uniform(background_upper_range)].
        Default: (0.2, 0.4)

    Example
    -------
    >>> synth = ControlledContrastAxonImage(
    ...     background=0.5,
    ...     fibers_lower_range=(0.3, 0.5),
    ...     background_upper_range=(0.2, 0.4),
    ... )
    >>> image, prob = synth(labels, prob_map)   # labels/prob: (B,1,D,H,W)
    """

    class XForm(cc.Transform):

        def __init__(
            self,
            background: float = 0.5,
            fibers_lower_range: tuple = (0.3, 0.5),
            background_upper_range: tuple = (0.2, 0.4),
            clean_target_lab: bool = False,
            gpu_geometry: bool = False,
            gpu_label_block_size: int = 8,
        ):
            super().__init__()
            self.background            = background
            self.fibers_lower_range    = fibers_lower_range
            self.background_upper_range = background_upper_range
            self.clean_target_lab      = clean_target_lab
            self.gpu_geometry          = gpu_geometry
            self.gpu_label_block_size  = gpu_label_block_size

            # --- label perturbation ---
            self.flip = cc.RandomFlipTransform()
            self.erode_axon = cc.RandomSmoothMorphoLabelTransform(
                shape=128, max_radius=4, min_radius=-4,
            )
            self.shallow = cc.RandomSmoothShallowLabelTransform(
                shape=128, max_width=3,
            ) * 0.3
            self.noisylabel = cc.randomize(cc.SmoothBernoulliTransform)(
                shape=cc.RandInt(2, 128),
                prob=cc.Uniform(0, 0.2),
            )
            self.soma = cc.randomize(cc.SmoothBernoulliDiskTransform)(
                shape=cc.RandInt(2, 16),
                prob=cc.Uniform(0, 0.02),
                radius=10,
                returns='disks',
            )

            # --- background structure ---
            self.label_map   = cc.RandomSmoothLabelMap(16, 8)
            self.erode_label = cc.RandomErodeLabelTransform(radius=5, new_labels=True)

            # --- separate GMMs for foreground / background ---
            self.gmm_fg = cc.RandomGaussianMixtureTransform(
                background=None if self.background else 0,
            )
            self.gmm_bg = cc.RandomGaussianMixtureTransform()

            # --- imaging artifacts ---
            self.gamma    = cc.RandomGammaTransform((0, 5))
            self.addbias  = cc.RandomAddFieldTransform(vmin=0, vmax=0.25)
            self.mulbias  = cc.RandomMulFieldTransform(symmetric=1)
            self.smooth   = cc.RandomSmoothTransform(2)
            self.noise    = (
                cc.RandomChiNoiseTransform() | cc.RandomGammaNoiseTransform()
            )
            self.rescale  = cc.QuantileTransform()

        def forward(self, lab, prob=None, target_lab=None):
            """
            Parameters
            ----------
            lab  : (1, *spatial) tensor[int]   — unique per-axon label map
            prob : (1, *spatial) tensor[float] — partial volume probabilities

            Returns
            -------
            image : (1, *spatial) tensor[float]
            prob  : (1, *spatial) tensor[float]  (unchanged)
            """
            if isinstance(lab, (list, tuple)):
                if len(lab) == 2:
                    lab, prob = lab
                elif len(lab) == 3:
                    lab, prob, target_lab = lab
                else:
                    raise ValueError(
                        'Expected (lab, prob) or (lab, prob, target_lab), '
                        f'got {len(lab)} items'
                    )
            _t_forward = _time.monotonic()
            _progress(
                f'forward_start clean_target_lab={self.clean_target_lab} '
                f'lab_shape={tuple(lab.shape)} prob_present={prob is not None} '
                f'target_present={target_lab is not None} '
                f'gpu_geometry={self.gpu_geometry and lab.is_cuda}'
            )

            shared_target_geometry = (
                target_lab is not None and
                target_lab.shape == lab.shape and
                torch.equal(target_lab, lab)
            )

            if target_lab is None or shared_target_geometry:
                lab, prob = self.flip(lab, prob)
                if shared_target_geometry:
                    target_lab = lab.clone()
            else:
                lab, prob, target_lab = self.flip(lab, prob, target_lab)
            _progress('after_flip')

            # ---- perturb axon label map ----
            # Cap retry loops to avoid infinite hangs on thin/small axons
            # whose voxels are completely annihilated by erosion.  If all
            # attempts fail, skip the step and use the pre-op tensor.
            _t0 = _time.monotonic()
            v = lab.clone()
            target_v = target_lab.clone() if target_lab is not None else None
            n_fg_in = int(v.any())
            _erode_retries = 0
            _erode_fallback = False
            use_fast_gpu_geometry = self.gpu_geometry and v.is_cuda and shared_target_geometry
            _progress('erode_start')
            if target_v is None or shared_target_geometry:
                v0 = v
                if use_fast_gpu_geometry:
                    v = smooth_random_morph_labels(
                        v0,
                        min_radius=-4,
                        max_radius=4,
                        field_shape=128,
                        block_size=self.gpu_label_block_size,
                    )
                else:
                    v = self.erode_axon(v)
                for _i in range(10):
                    if v.any():
                        break
                    _erode_retries += 1
                    if use_fast_gpu_geometry:
                        v = smooth_random_morph_labels(
                            v0,
                            min_radius=-4,
                            max_radius=4,
                            field_shape=128,
                            block_size=self.gpu_label_block_size,
                        )
                    else:
                        v = self.erode_axon(v0)
                else:
                    _erode_fallback = True
                    v = v0  # erosion impossible — use un-eroded labels
            else:
                v0, target_v0 = v, target_v
                v, target_v = self.erode_axon(v0, target_v0)
                for _i in range(10):
                    if v.any():
                        break
                    _erode_retries += 1
                    v, target_v = self.erode_axon(v0, target_v0)
                else:
                    _erode_fallback = True
                    v, target_v = v0, target_v0
            _t_erode = _time.monotonic() - _t0
            _progress(
                f'erode_done elapsed={_t_erode:.2f}s retries={_erode_retries} '
                f'fallback={_erode_fallback} fg_voxels={int((v > 0).sum())}'
            )

            _shallow_retries = 0
            _shallow_fallback = False
            _t1 = _time.monotonic()
            _progress('shallow_start')
            if target_v is None or shared_target_geometry:
                v0 = v
                if use_fast_gpu_geometry:
                    v = smooth_random_shallow_labels(
                        v0,
                        min_width=1,
                        max_width=3,
                        field_shape=128,
                        block_size=self.gpu_label_block_size,
                    )
                else:
                    v = self.shallow(v)
                for _i in range(10):
                    if v.any():
                        break
                    _shallow_retries += 1
                    if use_fast_gpu_geometry:
                        v = smooth_random_shallow_labels(
                            v0,
                            min_width=1,
                            max_width=3,
                            field_shape=128,
                            block_size=self.gpu_label_block_size,
                        )
                    else:
                        v = self.shallow(v0)
                else:
                    _shallow_fallback = True
                    v = v0  # shallow impossible — use pre-shallow labels
                del v0
            else:
                v0, target_v0 = v, target_v
                v, target_v = self.shallow(v0, target_v0)
                for _i in range(10):
                    if v.any():
                        break
                    _shallow_retries += 1
                    v, target_v = self.shallow(v0, target_v0)
                else:
                    _shallow_fallback = True
                    v, target_v = v0, target_v0
                del v0, target_v0
            _t_shallow = _time.monotonic() - _t1
            _progress(
                f'shallow_done elapsed={_t_shallow:.2f}s retries={_shallow_retries} '
                f'fallback={_shallow_fallback} fg_voxels={int((v > 0).sum())}'
            )

            if _erode_retries > 0 or _shallow_retries > 0:
                _log.info(
                    f'[synth] erode: {_erode_retries} retries, fallback={_erode_fallback} '
                    f'({_t_erode:.2f}s) | shallow: {_shallow_retries} retries, '
                    f'fallback={_shallow_fallback} ({_t_shallow:.2f}s) | '
                    f'fg_voxels_in={n_fg_in}')
            if shared_target_geometry and self.clean_target_lab:
                target_lab = v.clone()
            elif target_v is not None and self.clean_target_lab:
                target_lab = target_v.clone()
            if not self.clean_target_lab:
                _progress('noisylabel_start')
                if target_v is None or shared_target_geometry:
                    v = self.noisylabel(v)
                else:
                    v, target_v = self.noisylabel(v, target_v)
                _progress(f'noisylabel_done fg_voxels={int((v > 0).sum())}')
            if shared_target_geometry and not self.clean_target_lab:
                target_lab = v.clone()
            elif target_v is not None and not self.clean_target_lab:
                target_lab = target_v.clone()

            # group axons into shared-intensity classes
            y            = torch.zeros_like(lab, dtype=torch.int)
            vessel_labels = list(sorted(v.unique().tolist()))[1:]
            pyrandom.shuffle(vessel_labels)
            nb_groups    = cc.RandInt(1, 5)()
            nb_per_group = int(pymath.ceil(len(vessel_labels) / nb_groups))
            _progress(
                f'grouping_start unique_labels={len(vessel_labels)} nb_groups={nb_groups} '
                f'nb_per_group={nb_per_group}'
            )
            for i in range(nb_groups):
                group = vessel_labels[i * nb_per_group:(i + 1) * nb_per_group]
                for label in group:
                    y.masked_fill_(v == label, i + 1)
                soma = self.soma(y)
                y.masked_fill(soma > 0, i + 1)
                _progress(
                    f'grouping_group_done group_index={i} group_size={len(group)} '
                    f'assigned_voxels={int((y > 0).sum())}'
                )
            del v
            _progress('grouping_done')

            # ---- foreground: GMM → rescale to [fibers_lower, 1] ----
            _progress('gmm_fg_start')
            y            = self.gmm_fg(y)
            fibers_lower = float(cc.Uniform(*self.fibers_lower_range)())
            y            = _minmax_rescale(y, vmin=fibers_lower, vmax=1.0)
            y            = y * prob    # partial-volume soft blend
            _progress(f'gmm_fg_done fibers_lower={fibers_lower:.3f}')

            # ---- background: GMM → rescale to [0, background_upper] ----
            if cc.Uniform(1)() < self.background:
                _progress('background_start')
                z                = self.label_map(y)
                z                = self.erode_label(z)
                z                = self.gmm_bg(z)
                background_upper = float(cc.Uniform(*self.background_upper_range)())
                z                = _minmax_rescale(z, vmin=0.0, vmax=background_upper)
                y                = y + (1 - prob) * z
                del z
                _progress(f'background_done background_upper={background_upper:.3f}')

            # ---- global imaging artifacts ----
            _progress('artifacts_start')
            y = self.addbias(y)
            y = self.mulbias(y)
            y = self.gamma(y)
            y = self.smooth(y)
            y = self.noise(y)
            y = self.rescale(y)
            _progress(f'artifacts_done total_elapsed={_time.monotonic() - _t_forward:.2f}s')

            if target_lab is None:
                _progress('forward_return_image_prob')
                return y, prob
            _progress('forward_return_image_prob_target')
            return y, prob, target_lab

    def __init__(
        self,
        background: float = 0.5,
        fibers_lower_range: tuple = (0.3, 0.5),
        background_upper_range: tuple = (0.2, 0.4),
        gpu_geometry: bool = False,
        gpu_label_block_size: int = 8,
    ):
        super().__init__(
            background=background,
            fibers_lower_range=fibers_lower_range,
            background_upper_range=background_upper_range,
            gpu_geometry=gpu_geometry,
            gpu_label_block_size=gpu_label_block_size,
        )

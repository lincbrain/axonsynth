"""
Axon Segmentation — 3D UNet Training Script

Training samples are synthesized from dense label volumes on demand, cached in
RAM for a configurable number of epochs, and refreshed from the dense source
volumes between cache cycles. Validation is synthesized once at startup and
kept fixed for the entire run so checkpoint selection is stable.
"""

import argparse
import logging
import signal
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent))
from train_extras import NestedShellInteriorLoss


BINARY_MODE = 'binary'
THREE_CLASS_MODE = 'three_class_shell_interior'
SEGMENTATION_MODES = (BINARY_MODE, THREE_CLASS_MODE)


def model_output_channels(segmentation_mode: str) -> int:
    if segmentation_mode == BINARY_MODE:
        return 1
    if segmentation_mode == THREE_CLASS_MODE:
        return 3
    raise ValueError(f'Unsupported segmentation mode: {segmentation_mode!r}')


def cache_builder_for_mode(segmentation_mode: str) -> str:
    if segmentation_mode == BINARY_MODE:
        return 'cpu'
    if segmentation_mode == THREE_CLASS_MODE:
        return 'gpu'
    raise ValueError(f'Unsupported segmentation mode: {segmentation_mode!r}')


def resolve_cache_builder(segmentation_mode: str, requested: str | None) -> str:
    expected = cache_builder_for_mode(segmentation_mode)
    if requested is not None and requested != expected:
        raise ValueError(
            f'{segmentation_mode} requires --cache_builder {expected}, got {requested}'
        )
    return expected


def build_training_model(segmentation_mode: str, device: torch.device):
    from monai.networks.layers import Norm
    from monai.networks.nets import UNet

    return UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=model_output_channels(segmentation_mode),
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
        dropout=0.1,
    ).to(device)


def build_training_loss(
    segmentation_mode: str,
    *,
    class_weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
    foreground_weight: float = 1.0,
    core_weight: float = 1.0,
):
    if segmentation_mode == BINARY_MODE:
        from monai.losses import DiceCELoss

        return DiceCELoss(sigmoid=True, lambda_dice=0.5, lambda_ce=0.5)
    if segmentation_mode == THREE_CLASS_MODE:
        return NestedShellInteriorLoss(
            class_weights=class_weights,
            lambda_dice=0.5,
            lambda_ce=0.5,
            foreground_weight=foreground_weight,
            core_weight=core_weight,
        )
    raise ValueError(f'Unsupported segmentation mode: {segmentation_mode!r}')


def resume_start_epoch(checkpoint: dict) -> int:
    epoch = int(checkpoint.get('epoch', 0))
    return epoch + int(checkpoint.get('epoch_complete') is True)


def checkpoint_resume_rank(checkpoint: dict) -> tuple[int, int]:
    return (
        int(checkpoint.get('epoch_complete') is True),
        int(checkpoint.get('epoch', -1)),
    )


def atomic_torch_save(value, path: Path) -> None:
    """Publish a checkpoint only after the complete payload is on disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.tmp')
    try:
        torch.save(value, str(temporary))
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def resume_argument_mismatches(current: dict, saved: dict) -> list[str]:
    ignored = {'output_dir', 'resume', 'init_checkpoint'}

    def normalize(value):
        if isinstance(value, (list, tuple)):
            return tuple(normalize(item) for item in value)
        return value

    required = current.keys() - ignored
    return sorted(
        key for key in required
        if key not in saved or normalize(current[key]) != normalize(saved[key])
    )


class CachedTensorDataset(Dataset):
    """Small dict-style dataset backed by in-memory tensors."""

    def __init__(self, images: torch.Tensor, segs: torch.Tensor):
        if images.shape[0] != segs.shape[0]:
            raise ValueError(
                f'Cached tensors disagree on batch dimension: '
                f'images={tuple(images.shape)} segs={tuple(segs.shape)}'
            )
        self.images = images.contiguous()
        self.segs = segs.contiguous()

    def __len__(self) -> int:
        return int(self.images.shape[0])

    def __getitem__(self, idx: int) -> dict:
        return {
            'image': self.images[idx],
            'seg': self.segs[idx],
        }


def build_tensor_cache(loader, split: str, log: logging.Logger) -> CachedTensorDataset:
    """Materialize one full pass of a source loader into RAM."""
    t0 = time.time()
    image_batches = []
    seg_batches = []

    for batch_idx, batch in enumerate(loader, start=1):
        if 'image' not in batch or 'seg' not in batch:
            raise KeyError(
                f"Source {split} batch is missing 'image'/'seg' keys. "
                f'Available keys: {sorted(batch.keys())}'
            )
        image_batches.append(batch['image'].detach().cpu().clone().float())
        seg_batches.append(batch['seg'].detach().cpu().clone())
        if batch_idx == 1 or batch_idx % 50 == 0 or batch_idx == len(loader):
            log.info(f'  caching {split}: batch {batch_idx:3d}/{len(loader)}')

    if not image_batches:
        raise RuntimeError(f'Failed to build {split} cache: source loader yielded no batches')

    images = torch.cat(image_batches, dim=0).contiguous()
    segs = torch.cat(seg_batches, dim=0).contiguous()
    n_bytes = images.numel() * images.element_size() + segs.numel() * segs.element_size()
    elapsed = time.time() - t0
    log.info(
        f'Built {split} cache: {images.shape[0]} samples | '
        f'{n_bytes / (1024 ** 3):.2f} GiB | {elapsed:.1f}s'
    )
    return CachedTensorDataset(images, segs)


def create_cached_loader(
    dataset: CachedTensorDataset,
    batch_size: int,
    *,
    shuffle: bool,
    drop_last: bool,
    pin_memory: bool = True,
) -> DataLoader:
    """Serve cached tensors with lightweight loader settings."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )


def _binary_dice(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred = pred.bool()
    target = target.bool()
    intersection = torch.logical_and(pred, target).sum().item()
    pred_sum = pred.sum().item()
    target_sum = target.sum().item()
    denominator = pred_sum + target_sum
    if denominator == 0:
        return 1.0
    return (2.0 * intersection) / denominator


def _compute_three_class_metrics(
    logits: torch.Tensor,
    target: torch.Tensor,
    selection_weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> tuple[float, float, float, float, torch.Tensor]:
    probabilities = torch.softmax(logits, dim=1)
    pred_classes = probabilities.argmax(dim=1, keepdim=True)

    collapsed_pred = pred_classes > 0
    collapsed_target = target > 0
    shell_target = target == 1
    interior_target = target == 2

    collapsed_dice = _binary_dice(collapsed_pred, collapsed_target)
    shell_dice = _binary_dice(pred_classes == 1, shell_target)
    interior_dice = _binary_dice(pred_classes == 2, interior_target)
    weight_sum = sum(selection_weights)
    selection_score = (
        selection_weights[0] * collapsed_dice
        + selection_weights[1] * shell_dice
        + selection_weights[2] * interior_dice
    ) / weight_sum

    foreground_prob = probabilities[:, 1:].sum(dim=1, keepdim=True)
    return collapsed_dice, shell_dice, interior_dice, selection_score, foreground_prob


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description='Train 3D UNet on synthetic axon data')
    p.add_argument('--label_dir',   required=True,  help='Directory with *_label.nii.gz volumes')
    p.add_argument('--output_dir',  required=True,  help='Where to save checkpoints + TensorBoard logs')
    p.add_argument('--init_checkpoint', default=None,
                   help='Initialize model weights from a checkpoint and start a new optimization run.')
    p.add_argument('--segmentation_mode', default=THREE_CLASS_MODE,
                   choices=SEGMENTATION_MODES,
                   help='Target/mode selection: baseline binary foreground or 3-class shell/interior.')
    p.add_argument('--three_class_weights', type=float, nargs=3,
                   default=(1.0, 1.0, 1.0), metavar=('BG_W', 'SHELL_W', 'INTERIOR_W'),
                   help='Class weights for 3-class loss terms (background, shell, interior).')
    p.add_argument('--nested_foreground_weight', type=float, default=1.0,
                   help='Weight for the nested 3-class foreground-vs-background loss branch.')
    p.add_argument('--nested_core_weight', type=float, default=1.0,
                   help='Weight for the nested 3-class shell-vs-interior loss branch.')
    p.add_argument('--three_class_selection_weights', type=float, nargs=3,
                   default=(1.0, 1.0, 1.0),
                   metavar=('FG_DICE_W', 'SHELL_DICE_W', 'INTERIOR_DICE_W'),
                   help='Validation checkpoint-selection weights for collapsed foreground, shell, and interior Dice.')
    p.add_argument('--epochs',           type=int,   default=200)
    p.add_argument('--batch_size',       type=int,   default=2)
    p.add_argument('--lr',               type=float, default=1e-4)
    p.add_argument('--num_workers',      type=int,   default=10)
    p.add_argument('--val_fraction',     type=float, default=0.2,
                   help='Fraction of volumes reserved for validation (sorted order)')
    p.add_argument('--val_interval',     type=int,   default=5,
                   help='Run validation every N epochs')
    p.add_argument('--samples_per_vol',  type=int,   default=100,
                   help='Random subsets drawn per label volume per epoch')
    p.add_argument('--cache_epochs',     type=int,   default=1,
                   help='Reuse each synthesized train cache for N epochs before refreshing')
    p.add_argument('--cache_builder', default=None, choices=['cpu', 'gpu'],
                   help='Cache synthesis backend. Defaults to cpu for binary mode and gpu '
                        'for 3-class mode; explicit values must match the selected mode.')
    p.add_argument('--max_volumes',      type=int,   default=None,
                   help='Cap number of label volumes loaded (None = all). Applied before train/val split.')
    p.add_argument('--seed',             type=int,   default=42)
    p.add_argument('--roi_size',         type=int,   default=128,
                   help='Sliding window roi size (isotropic). Use 128 for 128³ volumes.')
    p.add_argument('--sw_batch_size',    type=int,   default=4,
                   help='Sliding window batch size during validation')
    p.add_argument('--n_label_groups',  type=int,   default=8,
                   help='Collapse unique axon IDs to N groups before synthesis '
                        '(speeds up cornucopia morphological ops ~N_axons/N times)')
    p.add_argument('--gpu_label_block_size', type=int, default=8,
                   help='Label block size for the dedicated GPU cache builder. '
                        'Only used with --cache_builder gpu.')
    # Synthesis params
    p.add_argument('--no_images',        action='store_true',
                   help='Skip image synthesis (use raw label/prob tensors). For debugging only.')
    p.add_argument('--background',          type=float, default=0.5)
    p.add_argument('--fibers_lower_lo',     type=float, default=0.3)
    p.add_argument('--fibers_lower_hi',     type=float, default=0.5)
    p.add_argument('--bg_upper_lo',         type=float, default=0.2)
    p.add_argument('--bg_upper_hi',         type=float, default=0.4)
    p.add_argument('--subset_fraction_lo',  type=float, default=0.3,
                   help='Lower bound of the axon keep-fraction range per sample (default: 0.3)')
    p.add_argument('--subset_fraction_hi',  type=float, default=0.8,
                   help='Upper bound of the axon keep-fraction range per sample (default: 0.8). '
                        'Raise to 0.9 to ensure model sees densely-packed fascicles.')
    p.add_argument('--density_low_range', type=float, nargs=2,
                   default=(0.05, 0.4), metavar=('MIN', 'MAX'),
                   help='Range for the low end of spatial axon keep-probability fields.')
    p.add_argument('--density_high_range', type=float, nargs=2,
                   default=(0.6, 1.0), metavar=('MIN', 'MAX'),
                   help='Range for the high end of spatial axon keep-probability fields.')
    p.add_argument('--density_uniform_range', type=float, nargs=2,
                   default=(0.3, 1.0), metavar=('MIN', 'MAX'),
                   help='Range for uniform spatial axon keep-probability fields.')
    p.add_argument('--resume',           action='store_true',
                   help='Resume from latest checkpoint in output_dir/checkpoints/')
    args = p.parse_args(argv)
    args.cache_builder = resolve_cache_builder(
        args.segmentation_mode,
        args.cache_builder,
    )
    return args


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    import monai
    from monai.data import decollate_batch
    from monai.inferers import sliding_window_inference
    from monai.metrics import DiceMetric
    from monai.transforms import (
        Activations,
        AsDiscrete,
        Compose,
        RandFlipd,
        RandGaussianNoised,
        RandRotate90d,
        RandScaleIntensityd,
        RandShiftIntensityd,
    )
    from torch.utils.tensorboard import SummaryWriter

    from datagen import AxonSubsetDataset, create_dataloader
    from datagen.gpu_cache_builder import build_gpu_tensor_cache

    # --- Setup ---
    # Manual seeding (set_determinism also sets cudnn.deterministic=True
    # and benchmark=False, but we want benchmark=True for speed).
    torch.manual_seed(args.seed)
    import numpy as _np; _np.random.seed(args.seed)
    import random as _pyrandom; _pyrandom.seed(args.seed)
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    log = logging.getLogger(__name__)
    if args.cache_epochs < 1:
        raise ValueError(f'--cache_epochs must be >= 1, got {args.cache_epochs}')
    if args.resume and args.init_checkpoint:
        raise ValueError('--resume and --init_checkpoint are mutually exclusive')
    if args.segmentation_mode == THREE_CLASS_MODE and len(args.three_class_weights) != 3:
        raise ValueError('--three_class_weights must contain exactly 3 values')
    if len(args.three_class_selection_weights) != 3:
        raise ValueError('--three_class_selection_weights must contain exactly 3 values')
    three_class_selection_weights = tuple(float(v) for v in args.three_class_selection_weights)
    if any(v < 0.0 for v in three_class_selection_weights) or sum(three_class_selection_weights) <= 0.0:
        raise ValueError(
            '--three_class_selection_weights must be non-negative and sum to > 0'
        )
    if args.segmentation_mode == THREE_CLASS_MODE and (
        args.nested_foreground_weight <= 0.0 or args.nested_core_weight <= 0.0
    ):
        raise ValueError('--nested_foreground_weight and --nested_core_weight must be > 0')

    def _validate_unit_range(name: str, values: tuple[float, float]) -> tuple[float, float]:
        lo, hi = (float(values[0]), float(values[1]))
        if lo < 0.0 or hi > 1.0 or lo > hi:
            raise ValueError(f'--{name} must satisfy 0 <= MIN <= MAX <= 1, got {values}')
        return lo, hi

    density_low_range = _validate_unit_range('density_low_range', args.density_low_range)
    density_high_range = _validate_unit_range('density_high_range', args.density_high_range)
    density_uniform_range = _validate_unit_range('density_uniform_range', args.density_uniform_range)
    if args.gpu_label_block_size < 1:
        raise ValueError(
            f'--gpu_label_block_size must be >= 1, got {args.gpu_label_block_size}'
        )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tb_dir = output_dir / 'tensorboard'
    ckpt_dir = output_dir / 'checkpoints'
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f'Train device: {device}')
    if args.cache_builder == 'gpu' and device.type != 'cuda':
        raise RuntimeError('--cache_builder gpu requires CUDA')
    if args.cache_builder == 'gpu' and args.no_images:
        raise ValueError('--cache_builder gpu is incompatible with --no_images')
    # Fixed input shape → cuDNN benchmarks fastest conv algorithm once then reuses it
    torch.backends.cudnn.benchmark = True
    monai.config.print_config()

    # --- Source DataLoaders ---
    # Source samples can be built by CPU workers or by the dedicated GPU cache builder.
    source_kwargs = dict(
        label_dir=args.label_dir,
        num_samples_per_volume=args.samples_per_vol,
        val_fraction=args.val_fraction,
        max_volumes=args.max_volumes,
        n_label_groups=args.n_label_groups,
        fibers_lower_range=(args.fibers_lower_lo, args.fibers_lower_hi),
        background_upper_range=(args.bg_upper_lo, args.bg_upper_hi),
        background=args.background,
        subset_fraction=(args.subset_fraction_lo, args.subset_fraction_hi),
        density_low_range=density_low_range,
        density_high_range=density_high_range,
        density_uniform_range=density_uniform_range,
        segmentation_mode=args.segmentation_mode,
    )
    if args.cache_builder == 'gpu':
        train_cache_source = AxonSubsetDataset(
            split='train',
            generate_images=False,
            **source_kwargs,
        )
        val_cache_source = AxonSubsetDataset(
            split='val',
            generate_images=False,
            **source_kwargs,
        )
        log.info(
            'Dedicated GPU cache builder enabled '
            f'(gpu_label_block_size={args.gpu_label_block_size}, '
            'source synthesis bypasses DataLoader workers)'
        )
        log.info(f'Source train samples/cache build: {len(train_cache_source)}')
        log.info(f'Source val samples/cache build:   {len(val_cache_source)}')
    else:
        loader_kwargs = dict(source_kwargs)
        loader_kwargs['generate_images'] = (not args.no_images)
        train_cache_source = create_dataloader(
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            split='train',
            shuffle=True,
            drop_last=True,
            persistent_workers=True,
            **loader_kwargs,
        )
        val_cache_source = create_dataloader(
            batch_size=1,
            num_workers=max(2, args.num_workers // 4),
            split='val',
            shuffle=False,
            drop_last=False,
            persistent_workers=False,
            **loader_kwargs,
        )

        log.info(f'Workers handle CPU synthesis (generate_images={not args.no_images})')
        log.info(f'Source train batches/cache build: {len(train_cache_source)}')
        log.info(f'Source val batches/cache build:   {len(val_cache_source)}')

    def build_source_cache(source, *, split: str) -> CachedTensorDataset:
        if args.cache_builder == 'gpu':
            images, segs, _ = build_gpu_tensor_cache(
                source,
                split=split,
                device=device,
                log=log,
                gpu_label_block_size=args.gpu_label_block_size,
            )
            return CachedTensorDataset(images, segs)
        return build_tensor_cache(source, split=split, log=log)

    # --- Post-batch augmentation ---
    # Geometric: applied to both image and seg.
    # Intensity: applied to image only — closes the domain gap between
    #            synthetic contrast and real microscopy.
    geo_aug = Compose([
        RandFlipd(keys=['image', 'seg'], prob=0.5, spatial_axis=0),
        RandFlipd(keys=['image', 'seg'], prob=0.5, spatial_axis=1),
        RandFlipd(keys=['image', 'seg'], prob=0.5, spatial_axis=2),
        RandRotate90d(keys=['image', 'seg'], prob=0.5, spatial_axes=(0, 2)),
        # Intensity augmentation (image only)
        RandScaleIntensityd(keys=['image'], factors=0.1, prob=1.0),
        RandShiftIntensityd(keys=['image'], offsets=0.1, prob=1.0),
        RandGaussianNoised(keys=['image'], prob=0.15, mean=0.0, std=0.05),
    ])
    geo_aug.set_random_state(seed=args.seed)

    # --- Model ---
    model = build_training_model(args.segmentation_mode, device)
    log.info(f'Model params: {sum(p.numel() for p in model.parameters()):,}')

    # --- Loss, optimizer, scheduler ---
    class_weights = tuple(float(v) for v in args.three_class_weights)
    loss_fn = build_training_loss(
        args.segmentation_mode,
        class_weights=class_weights,
        foreground_weight=args.nested_foreground_weight,
        core_weight=args.nested_core_weight,
    ).to(device)
    if args.segmentation_mode == BINARY_MODE:
        log.info('Loss: MONAI DiceCE (sigmoid=True, lambda_dice=0.5, lambda_ce=0.5)')
    else:
        log.info(
            'Loss: nested foreground/core DiceCE (3-class shell/interior mode) '
            f'with class_weights={class_weights}, '
            f'foreground_weight={args.nested_foreground_weight}, '
            f'core_weight={args.nested_core_weight}'
        )
        log.info(
            '3-class checkpoint selection weights: '
            f'foreground={three_class_selection_weights[0]}, '
            f'shell={three_class_selection_weights[1]}, '
            f'interior={three_class_selection_weights[2]}'
        )
    # AdamW: decoupled weight decay regularises all weights equally regardless
    # of gradient magnitude (Loshchilov & Hutter 2019).
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    # 5-epoch linear warmup → cosine decay: lets batch-norm stats and Adam moment
    # estimates warm up before full learning rate kicks in.
    warmup_epochs = 5
    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-3, total_iters=warmup_epochs)
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs - warmup_epochs, eta_min=args.lr / 100)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_sched, cosine_sched],
        milestones=[warmup_epochs])

    # --- Post-processing for validation ---
    if args.segmentation_mode == 'binary':
        post_pred  = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
        post_label = Compose([AsDiscrete(threshold=0.5)])
        dice_metric = DiceMetric(include_background=False, reduction='mean', get_not_nans=False)
    else:
        post_pred = post_label = dice_metric = None

    # --- Logging ---
    writer = SummaryWriter(log_dir=str(tb_dir))

    best_dice   = -1.0
    best_epoch  = -1
    best_metric_name = 'val_dice' if args.segmentation_mode == 'binary' else 'val_selection_score'
    best_metric_label = 'Val Dice' if args.segmentation_mode == 'binary' else 'Val Selection Score'
    start_epoch = 1
    roi_size    = (args.roi_size,) * 3
    use_amp     = device.type == 'cuda'
    scaler      = torch.amp.GradScaler('cuda', enabled=use_amp)

    def _parse_epoch_from_path(path: Path | None) -> int:
        if path is None:
            return -1
        try:
            return int(path.stem.split('_', 1)[1])
        except (IndexError, ValueError):
            return -1

    def _maybe_load_checkpoint(path: Path) -> tuple[dict | None, int]:
        if not path.exists():
            return None, -1
        try:
            checkpoint = torch.load(str(path), map_location='cpu')
        except Exception as error:
            log.warning(f'Ignoring unreadable checkpoint {path}: {error}')
            return None, -1
        if not isinstance(checkpoint, dict):
            log.warning(f'Ignoring checkpoint with invalid payload type: {path}')
            return None, -1
        return checkpoint, int(checkpoint.get('epoch', -1))

    log.info('Building fixed validation cache...')
    val_cache = build_source_cache(val_cache_source, split='val')
    val_loader = create_cached_loader(
        val_cache,
        batch_size=1,
        shuffle=False,
        drop_last=False,
    )
    log.info(f'Validation cache ready: {len(val_cache)} samples, {len(val_loader)} batches')

    if args.init_checkpoint:
        init_ckpt_path = Path(args.init_checkpoint)
        if not init_ckpt_path.exists():
            raise FileNotFoundError(f'Init checkpoint not found: {init_ckpt_path}')
        log.info(f'Initializing model weights from {init_ckpt_path}')
        init_ckpt = torch.load(str(init_ckpt_path), map_location=device)
        init_state = init_ckpt['model_state_dict'] if 'model_state_dict' in init_ckpt else init_ckpt
        model.load_state_dict(init_state)
        log.info('Model weights loaded; optimizer, scheduler, and scaler start fresh')

    # --- Resume from checkpoint ---
    if args.resume:
        ckpts = sorted(ckpt_dir.glob('epoch_*.pt'))
        resume_ckpt = None
        resume_epoch = -1
        resume_state = None
        resume_rank = (-1, -1)

        candidates = [*ckpts, ckpt_dir / 'best_model.pt', ckpt_dir / 'last_state.pt']
        for candidate in candidates:
            candidate_state, candidate_epoch = _maybe_load_checkpoint(candidate)
            if candidate_state is None:
                continue
            if candidate_state.get('epoch_complete') is not True:
                log.warning(f'Ignoring checkpoint without a completed-epoch marker: {candidate}')
                continue
            candidate_rank = checkpoint_resume_rank(candidate_state)
            if candidate_rank > resume_rank:
                resume_ckpt = candidate
                resume_epoch = candidate_epoch
                resume_state = candidate_state
                resume_rank = candidate_rank

        if resume_ckpt and resume_ckpt.exists():
            log.info(f'Resuming from {resume_ckpt}')
            ckpt = resume_state if resume_state is not None else torch.load(str(resume_ckpt), map_location=device)
            saved_args = ckpt.get('args', {})
            if not isinstance(saved_args, dict):
                saved_args = vars(saved_args)
            mismatches = resume_argument_mismatches(vars(args), saved_args)
            if mismatches:
                raise ValueError(
                    'Resume arguments differ from the checkpoint: ' + ', '.join(mismatches)
                )
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            if 'scheduler_state_dict' in ckpt:
                scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            if 'scaler_state_dict' in ckpt:
                scaler.load_state_dict(ckpt['scaler_state_dict'])
            start_epoch = resume_start_epoch(ckpt)
            best_dice   = ckpt.get(best_metric_name, ckpt.get('val_metric_value', ckpt.get('val_dice', -1.0)))
            best_epoch  = ckpt.get('best_epoch', -1)
            log.info(
                f'Resumed at epoch {start_epoch} '
                f'(checkpoint epoch complete={ckpt.get("epoch_complete") is True}), '
                f'best_{best_metric_name}={best_dice:.4f} @ epoch {best_epoch}'
            )
        else:
            raise FileNotFoundError(
                '--resume was requested, but no completed release checkpoint was found. '
                'Use --init_checkpoint to initialize a new run from legacy weights.'
            )

    train_cache = None
    train_loader = None
    train_cache_start_epoch = None
    total_train_epochs = max(0, args.epochs - start_epoch + 1)
    total_cache_cycles = (
        (total_train_epochs + args.cache_epochs - 1) // args.cache_epochs
        if total_train_epochs > 0 else 0
    )

    def _checkpoint_state(epoch_num: int, *, epoch_complete: bool) -> dict:
        return {
            'epoch': epoch_num,
            'epoch_complete': epoch_complete,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'val_dice': best_dice,
            'val_metric_name': best_metric_name,
            'val_metric_value': best_dice,
            best_metric_name: best_dice,
            'best_epoch': best_epoch,
            'args': vars(args),
            'train_cache_start_epoch': train_cache_start_epoch,
        }

    epoch = max(0, start_epoch - 1)
    epoch_complete = True

    # --- Preemption handler: never publish partially applied optimizer state. ---
    def _save_preemption_ckpt(signum, frame):
        if not epoch_complete:
            log.info(
                'SIGTERM received during an epoch; leaving the latest completed '
                'checkpoint unchanged so --resume cannot replay partial updates.'
            )
            sys.exit(0)
        log.info('SIGTERM received between epochs - saving completed state...')
        checkpoint_state = _checkpoint_state(
            epoch,
            epoch_complete=True,
        )
        atomic_torch_save(checkpoint_state, ckpt_dir / f'epoch_{epoch:04d}.pt')
        atomic_torch_save(checkpoint_state, ckpt_dir / 'last_state.pt')
        next_epoch = resume_start_epoch(checkpoint_state)
        log.info(
            f'Preemption checkpoint saved at epoch {epoch} '
            f'(complete={epoch_complete}); --resume will start at epoch {next_epoch}.'
        )
        sys.exit(0)
    signal.signal(signal.SIGTERM, _save_preemption_ckpt)

    log.info(
        f'Starting training: {args.epochs} epochs, lr={args.lr}, roi={roi_size}, '
        f'AMP={use_amp}, cache_epochs={args.cache_epochs}'
    )

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_complete = False
        # ------------------------------------------------------------------ #
        # Train
        # ------------------------------------------------------------------ #
        if train_loader is None or epoch >= train_cache_start_epoch + args.cache_epochs:
            train_cache_start_epoch = epoch
            cache_cycle_idx = ((epoch - start_epoch) // args.cache_epochs) + 1
            cache_cycle_len = min(args.cache_epochs, args.epochs - epoch + 1)
            log.info(
                f'Building train cache cycle {cache_cycle_idx}/{total_cache_cycles} '
                f'for epochs {epoch}-{epoch + cache_cycle_len - 1}...'
            )
            train_cache = build_source_cache(train_cache_source, split='train')
            train_loader = create_cached_loader(
                train_cache,
                batch_size=args.batch_size,
                shuffle=True,
                drop_last=True,
            )
            log.info(f'Train cache ready: {len(train_cache)} samples, {len(train_loader)} batches')
        else:
            cache_cycle_len = min(args.cache_epochs, args.epochs - train_cache_start_epoch + 1)
            cache_reuse_idx = epoch - train_cache_start_epoch + 1
            log.info(
                f'Reusing train cache from epoch {train_cache_start_epoch} '
                f'({cache_reuse_idx}/{cache_cycle_len})'
            )

        model.train()
        epoch_loss = 0.0
        step = 0
        t0 = time.time()
        t_batch_end = time.time()  # for measuring data-wait time
        _total_wait = 0.0
        _total_train = 0.0

        for batch in train_loader:
            t_wait = time.time() - t_batch_end  # time waiting for DataLoader
            t_step = time.time()

            image = batch['image'].to(device)
            seg   = batch['seg'].to(device)

            # Geometric augmentation per-sample then re-stack
            samples = [geo_aug(s) for s in decollate_batch({'image': image, 'seg': seg})]
            image = torch.stack([s['image'] for s in samples])
            seg   = torch.stack([s['seg']   for s in samples])

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                pred = model(image)
                loss = loss_fn(pred, seg)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            step += 1
            t_train = time.time() - t_step
            _total_wait += t_wait
            _total_train += t_train
            t_batch_end = time.time()

            # Log every batch during epoch 1, then every 50th batch
            if epoch == start_epoch or step % 50 == 0 or step <= 5:
                log.info(f'  batch {step:3d}/{len(train_loader)} | '
                         f'wait={t_wait:.2f}s train={t_train:.3f}s loss={loss.item():.4f}')

        scheduler.step()

        epoch_loss /= step
        elapsed = time.time() - t0
        lr_now  = scheduler.get_last_lr()[0]

        log.info(f'Epoch {epoch:04d}/{args.epochs} | loss={epoch_loss:.4f} | '
                 f'lr={lr_now:.2e} | {elapsed:.0f}s | '
                 f'data_wait={_total_wait:.0f}s train={_total_train:.1f}s '
                 f'overhead={elapsed - _total_wait - _total_train:.1f}s')
        writer.add_scalar('train/loss',   epoch_loss, epoch)
        writer.add_scalar('train/lr',     lr_now,     epoch)

        # ------------------------------------------------------------------ #
        # Validate
        # ------------------------------------------------------------------ #
        if epoch % args.val_interval == 0 or epoch == args.epochs:
            model.eval()
            log_images_this_epoch = True   # capture first val batch for TensorBoard
            val_dice_values = []
            val_shell_dice_values = []
            val_interior_dice_values = []
            val_selection_scores = []
            with torch.no_grad():
                for val_batch in val_loader:
                    val_image = val_batch['image'].to(device)
                    val_seg   = val_batch['seg'].to(device)

                    with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                        val_pred = sliding_window_inference(
                            val_image, roi_size, args.sw_batch_size, model,
                            overlap=0.5,
                        )
                    if args.segmentation_mode == 'binary':
                        val_pred_post  = [post_pred(p)  for p in decollate_batch(val_pred)]
                        val_label_post = [post_label(l) for l in decollate_batch(val_seg)]
                        dice_metric(y_pred=val_pred_post, y=val_label_post)
                    else:
                        collapsed_dice, shell_dice, interior_dice, selection_score, foreground_prob = (
                            _compute_three_class_metrics(
                                val_pred,
                                val_seg,
                                selection_weights=three_class_selection_weights,
                            )
                        )
                        val_dice_values.append(collapsed_dice)
                        val_shell_dice_values.append(shell_dice)
                        val_interior_dice_values.append(interior_dice)
                        val_selection_scores.append(selection_score)

                    # Log center-slice images from first batch only
                    if log_images_this_epoch:
                        log_images_this_epoch = False
                        z = val_image.shape[-1] // 2  # center axial slice
                        # Normalise image slice to [0,1] for display
                        img_slice  = val_image[0, 0, :, :, z]
                        img_slice  = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min() + 1e-8)
                        if args.segmentation_mode == 'binary':
                            seg_slice  = val_seg[0, 0, :, :, z]
                            pred_slice = torch.sigmoid(val_pred[0, 0, :, :, z])
                        else:
                            seg_slice  = (val_seg[0, 0, :, :, z] > 0).float()
                            pred_slice = foreground_prob[0, 0, :, :, z]
                        # Stack side-by-side: image | ground truth | prediction
                        grid = torch.stack([img_slice, seg_slice, pred_slice], dim=0).unsqueeze(1)  # (3,1,H,W)
                        writer.add_images('val/image_gt_pred', grid, epoch, dataformats='NCHW')

            if args.segmentation_mode == 'binary':
                mean_dice = dice_metric.aggregate().item()
                dice_metric.reset()
            else:
                mean_dice = float(sum(val_dice_values) / max(len(val_dice_values), 1))
                mean_shell_dice = float(sum(val_shell_dice_values) / max(len(val_shell_dice_values), 1))
                mean_interior_dice = float(sum(val_interior_dice_values) / max(len(val_interior_dice_values), 1))
                mean_selection_score = float(sum(val_selection_scores) / max(len(val_selection_scores), 1))

            if args.segmentation_mode == 'binary':
                log.info(f'  Val Dice: {mean_dice:.4f} (best={best_dice:.4f} @ epoch {best_epoch})')
            else:
                log.info(f'  Val Dice: {mean_dice:.4f}')
            writer.add_scalar('val/dice', mean_dice, epoch)
            if args.segmentation_mode == 'three_class_shell_interior':
                log.info(
                    f'  Val Shell Dice: {mean_shell_dice:.4f} | '
                    f'Interior Dice: {mean_interior_dice:.4f}'
                )
                log.info(
                    f'  Val Selection Score: {mean_selection_score:.4f} '
                    f'(best={best_dice:.4f} @ epoch {best_epoch})'
                )
                writer.add_scalar('val/shell_dice', mean_shell_dice, epoch)
                writer.add_scalar('val/interior_dice', mean_interior_dice, epoch)
                writer.add_scalar('val/selection_score', mean_selection_score, epoch)

            current_selection_score = mean_dice
            if args.segmentation_mode == 'three_class_shell_interior':
                current_selection_score = mean_selection_score

            if current_selection_score > best_dice:
                best_dice  = current_selection_score
                best_epoch = epoch
                ckpt_path  = ckpt_dir / 'best_model.pt'
                atomic_torch_save(
                    _checkpoint_state(epoch, epoch_complete=True),
                    ckpt_path,
                )
                log.info(f'  Saved best checkpoint → {ckpt_path}')

            # Epoch-200 milestone: two saves for a clean comparison point.
            #   best_model_ep200.pt — best Dice seen in epochs 1–200
            #                        under this run's configured LR schedule
            #   epoch_0200.pt       — exact model state at epoch 200
            #                        (regardless of whether it was the best)
            if epoch == 200:
                import shutil
                milestone_path = ckpt_dir / 'best_model_ep200.pt'
                milestone_temporary = milestone_path.with_name(f'.{milestone_path.name}.tmp')
                shutil.copy2(ckpt_dir / 'best_model.pt', milestone_temporary)
                milestone_temporary.replace(milestone_path)
                log.info(f'  Saved epoch-200 best snapshot → best_model_ep200.pt')
                atomic_torch_save(
                    _checkpoint_state(epoch, epoch_complete=True),
                    ckpt_dir / 'epoch_0200.pt',
                )
                log.info(f'  Saved epoch-200 state → epoch_0200.pt')

        epoch_complete = True
        atomic_torch_save(
            _checkpoint_state(epoch, epoch_complete=True),
            ckpt_dir / 'last_state.pt',
        )

        # Save periodic checkpoint every 10 epochs
        if epoch % 10 == 0:
            atomic_torch_save(
                _checkpoint_state(epoch, epoch_complete=True),
                ckpt_dir / f'epoch_{epoch:04d}.pt',
            )

    # --- Final summary ---
    log.info('='*60)
    log.info(f'Training complete.')
    log.info(f'Best {best_metric_label}: {best_dice:.4f} at epoch {best_epoch}')
    log.info(f'Checkpoints:   {ckpt_dir}')
    log.info(f'TensorBoard:   tensorboard --logdir {tb_dir}')
    writer.close()


if __name__ == '__main__':
    main()

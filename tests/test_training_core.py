import pytest
import torch

from train import (
    BINARY_MODE,
    THREE_CLASS_MODE,
    _compute_three_class_metrics,
    atomic_torch_save,
    build_training_loss,
    cache_builder_for_mode,
    checkpoint_resume_rank,
    model_output_channels,
    parse_args,
    resolve_cache_builder,
    resume_argument_mismatches,
    resume_start_epoch,
)
from train_extras import NestedShellInteriorLoss


def test_primary_cli_defaults_and_mode_helpers():
    args = parse_args(['--label_dir', 'labels', '--output_dir', 'output'])
    binary_args = parse_args([
        '--label_dir',
        'labels',
        '--output_dir',
        'output',
        '--segmentation_mode',
        BINARY_MODE,
    ])

    assert args.segmentation_mode == THREE_CLASS_MODE
    assert args.cache_builder == 'gpu'
    assert binary_args.cache_builder == 'cpu'
    assert model_output_channels(BINARY_MODE) == 1
    assert model_output_channels(THREE_CLASS_MODE) == 3
    assert cache_builder_for_mode(BINARY_MODE) == 'cpu'
    assert cache_builder_for_mode(THREE_CLASS_MODE) == 'gpu'
    assert resolve_cache_builder(BINARY_MODE, None) == 'cpu'
    assert resolve_cache_builder(THREE_CLASS_MODE, 'gpu') == 'gpu'

    with pytest.raises(ValueError, match='requires --cache_builder gpu'):
        resolve_cache_builder(THREE_CLASS_MODE, 'cpu')


def test_three_class_selection_score_is_mean_of_accepted_dice_terms():
    target = torch.tensor(
        [0, 1, 2, 0, 1, 2, 0, 0],
        dtype=torch.long,
    ).reshape(1, 1, 2, 2, 2)
    predicted_classes = target.clone()
    predicted_classes[0, 0, 0, 1, 0] = 1
    logits = torch.full((1, 3, 2, 2, 2), -20.0)
    logits.scatter_(1, predicted_classes, 20.0)

    foreground_dice, shell_dice, interior_dice, score, foreground_prob = (
        _compute_three_class_metrics(logits, target)
    )

    assert foreground_dice == pytest.approx(1.0)
    assert shell_dice == pytest.approx(0.8)
    assert interior_dice == pytest.approx(2.0 / 3.0)
    assert score == pytest.approx(
        (foreground_dice + shell_dice + interior_dice) / 3.0
    )
    assert foreground_prob.shape == target.shape


def test_nested_loss_is_the_only_three_class_objective():
    loss_fn = build_training_loss(THREE_CLASS_MODE)
    logits = torch.randn(1, 3, 3, 3, 3, requires_grad=True)
    target = torch.randint(0, 3, (1, 1, 3, 3, 3))

    loss = loss_fn(logits, target)

    assert isinstance(loss_fn, NestedShellInteriorLoss)
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None


def test_binary_loss_matches_monai_dice_ce():
    monai_losses = pytest.importorskip('monai.losses')
    logits = torch.linspace(-2.0, 2.0, 54).reshape(2, 1, 3, 3, 3)
    target = torch.linspace(0.0, 1.0, 54).reshape(2, 1, 3, 3, 3)

    actual_loss = build_training_loss(BINARY_MODE)
    reference_loss = monai_losses.DiceCELoss(
        sigmoid=True,
        lambda_dice=0.5,
        lambda_ce=0.5,
    )

    assert isinstance(actual_loss, monai_losses.DiceCELoss)
    torch.testing.assert_close(
        actual_loss(logits, target),
        reference_loss(logits, target),
    )


def test_incomplete_checkpoint_resumes_interrupted_epoch():
    interrupted = {'epoch': 17, 'epoch_complete': False}
    completed = {'epoch': 17, 'epoch_complete': True}

    assert resume_start_epoch(interrupted) == 17
    assert resume_start_epoch(completed) == 18
    assert resume_start_epoch({'epoch': 17}) == 17
    assert checkpoint_resume_rank(completed) > checkpoint_resume_rank(interrupted)
    assert checkpoint_resume_rank(completed) > checkpoint_resume_rank(
        {'epoch': 18, 'epoch_complete': False}
    )


def test_resume_argument_validation_normalizes_sequences_and_ignores_paths():
    current = {
        'epochs': 200,
        'three_class_weights': (1.0, 1.0, 1.0),
        'output_dir': 'new-output',
        'resume': True,
    }
    saved = {
        'epochs': 200,
        'three_class_weights': [1.0, 1.0, 1.0],
        'output_dir': 'old-output',
        'resume': False,
    }

    assert resume_argument_mismatches(current, saved) == []
    saved['epochs'] = 500
    assert resume_argument_mismatches(current, saved) == ['epochs']
    del saved['epochs']
    assert resume_argument_mismatches(current, saved) == ['epochs']


def test_atomic_checkpoint_save_publishes_loadable_payload(tmp_path):
    path = tmp_path / 'checkpoint.pt'
    atomic_torch_save({'epoch': 7, 'tensor': torch.arange(3)}, path)

    loaded = torch.load(path, map_location='cpu', weights_only=True)
    assert loaded['epoch'] == 7
    torch.testing.assert_close(loaded['tensor'], torch.arange(3))
    assert not (tmp_path / '.checkpoint.pt.tmp').exists()

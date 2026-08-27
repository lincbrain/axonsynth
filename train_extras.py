import torch
import torch.nn as nn
import torch.nn.functional as F


class NestedShellInteriorLoss(nn.Module):
    """Hierarchical loss for background/shell/interior segmentation.

    The foreground branch learns background vs. foreground, and the core branch
    learns shell vs. interior only within foreground voxels.
    """

    def __init__(
        self,
        *,
        class_weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
        lambda_dice: float = 0.5,
        lambda_ce: float = 0.5,
        foreground_weight: float = 1.0,
        core_weight: float = 1.0,
        smooth: float = 1.0,
    ):
        super().__init__()
        if len(class_weights) != 3:
            raise ValueError(
                f"class_weights must contain exactly 3 values, got {class_weights}"
            )
        if lambda_dice < 0.0 or lambda_ce < 0.0:
            raise ValueError(
                f"lambda_dice and lambda_ce must be non-negative, got {lambda_dice}, {lambda_ce}"
            )
        if foreground_weight <= 0.0 or core_weight <= 0.0:
            raise ValueError(
                "foreground_weight and core_weight must be positive"
            )

        self.register_buffer(
            "class_weights",
            torch.tensor(class_weights, dtype=torch.float32),
        )
        self.lambda_dice = float(lambda_dice)
        self.lambda_ce = float(lambda_ce)
        self.foreground_weight = float(foreground_weight)
        self.core_weight = float(core_weight)
        self.smooth = float(smooth)

    def _weighted_masked_mean(
        self,
        values: torch.Tensor,
        weights: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        weighted_mask = weights * mask
        denominator = weighted_mask.sum().clamp_min(1e-8)
        return (values * weighted_mask).sum() / denominator

    def _masked_dice_loss(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        probabilities = torch.sigmoid(logits) * mask
        target = target * mask
        dims = tuple(range(1, probabilities.ndim))
        intersection = (probabilities * target).sum(dim=dims)
        denominator = probabilities.sum(dim=dims) + target.sum(dim=dims)
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        return 1.0 - dice.mean()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 5 or logits.shape[1] != 3:
            raise ValueError(
                f"Expected 3-class 5D logits, got shape={tuple(logits.shape)}"
            )

        target = target.long()
        fg_target = (target > 0).float()
        core_target = (target == 2).float()

        bg_weight, shell_weight, interior_weight = self.class_weights.to(logits.device)

        # p(foreground) / p(background) expressed as a binary logit.
        fg_logits = torch.logsumexp(logits[:, 1:], dim=1, keepdim=True) - logits[:, :1]
        fg_weights = torch.where(
            fg_target > 0,
            torch.full_like(fg_target, 0.5 * (shell_weight + interior_weight)),
            torch.full_like(fg_target, bg_weight),
        )
        fg_ce = self._weighted_masked_mean(
            F.binary_cross_entropy_with_logits(fg_logits, fg_target, reduction="none"),
            fg_weights,
            torch.ones_like(fg_target),
        )
        fg_dice = self._masked_dice_loss(fg_logits, fg_target, torch.ones_like(fg_target))
        fg_loss = self.lambda_ce * fg_ce + self.lambda_dice * fg_dice

        # p(interior | foreground) expressed as a binary logit.
        core_logits = logits[:, 2:3] - logits[:, 1:2]
        core_mask = fg_target
        core_weights = torch.where(
            core_target > 0,
            torch.full_like(core_target, interior_weight),
            torch.full_like(core_target, shell_weight),
        )
        core_ce = self._weighted_masked_mean(
            F.binary_cross_entropy_with_logits(core_logits, core_target, reduction="none"),
            core_weights,
            core_mask,
        )
        core_dice = self._masked_dice_loss(core_logits, core_target, core_mask)
        core_loss = self.lambda_ce * core_ce + self.lambda_dice * core_dice

        return (
            self.foreground_weight * fg_loss + self.core_weight * core_loss
        ) / (self.foreground_weight + self.core_weight)

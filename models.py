import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp


class CombinedLoss(nn.Module):
    """
    Dice + Focal loss for multi-class or multi-label segmentation.
    In this case, multi-label with 3 channels (Iris, Pupil, Lens).
    """

    def __init__(self, dice_weight=0.5, focal_weight=0.5, eps=1e-7):
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.eps = eps

        # SMP has built-in dice and focal, but let's custom-implement.
        self.dice_loss_fn = smp.losses.DiceLoss(mode='multilabel')
        self.focal_loss_fn = smp.losses.FocalLoss(mode='multilabel')

    def forward(self, outputs, targets):
        dice_loss = self.dice_loss_fn(outputs, targets)
        focal_loss = self.focal_loss_fn(outputs, targets)
        return self.dice_weight * dice_loss + self.focal_weight * focal_loss


def get_segmentation_model(num_classes=3, encoder_name='timm-efficientnet-b4', encoder_weights='imagenet'):
    """
    Returns a segmentation model from SMP.
    For example, a Unet++ with an EfficientNet-B4 backbone.
    Adjust to other advanced architectures (e.g., DeepLabV3Plus, FPN, etc.).
    """
    model = smp.UnetPlusPlus(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=num_classes,
        activation=None  # We'll handle activation in post-processing if needed
    )
    return model


# ---------------- Metrics (IoU, Dice) -----------------
def calculate_iou(preds, targets, threshold=0.5):
    """
    Calculate IoU for multi-channel (multi-label) predictions.
    preds and targets should be [N, C, H, W].
    """
    preds = (preds > threshold).float()
    intersection = (preds * targets).sum(dim=(2, 3))
    union = (preds + targets - preds * targets).sum(dim=(2, 3)) + 1e-7
    iou = intersection / union
    return iou.mean(dim=1).mean().item()  # average over channels and batch


def calculate_dice(preds, targets, threshold=0.5):
    """
    Calculate Dice coefficient for multi-channel predictions.
    """
    preds = (preds > threshold).float()
    intersection = (preds * targets).sum(dim=(2, 3))
    dice = (2.0 * intersection) / (preds.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) + 1e-7)
    return dice.mean(dim=1).mean().item()  # average over channels and batch

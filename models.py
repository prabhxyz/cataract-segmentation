import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


###############################################
# Weighted Dice Loss (Binary)
###############################################
class WeightedDiceLoss(nn.Module):
    def __init__(self, weight=1.0, eps=1e-7):
        super().__init__()
        self.weight = weight
        self.eps = eps

    def forward(self, outputs, targets):
        # outputs, targets => [B, 1, H, W]
        intersection = (outputs * targets).sum(dim=(2, 3))
        union = outputs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice = 1 - (2.0 * intersection + self.eps) / (union + self.eps)
        return dice.mean()


###############################################
# Focal Loss (Binary)
###############################################
def focal_loss_with_logits(y_pred, y_true, alpha=1.0, gamma=2.0, reduction='mean'):
    p = torch.sigmoid(y_pred)
    bce = - (y_true * torch.log(p + 1e-7) + (1 - y_true) * torch.log(1 - p + 1e-7))
    focal_factor = (1 - p) ** gamma
    loss = alpha * focal_factor * bce
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss


###############################################
# Combined Dice + Focal
###############################################
class CombinedLoss(nn.Module):
    def __init__(self, dice_weight=0.5, focal_weight=0.5, weight=1.0):
        """
        dice_weight, focal_weight: how to balance the two terms
        weight: alpha in focal + weighting in dice
        """
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.weight = weight
        self.dice_loss = WeightedDiceLoss(weight=self.weight)

    def forward(self, outputs, targets):
        dice = self.dice_loss(outputs, targets)
        focal = focal_loss_with_logits(outputs, targets, alpha=self.weight)
        return self.dice_weight * dice + self.focal_weight * focal


###############################################
# DeepLabV3 with ResNet101 (Binary)
###############################################
def get_segmentation_model(num_classes=1, encoder_name='resnet101', encoder_weights='imagenet'):
    """
    Return a DeepLabV3 model from segmentation_models_pytorch.
    For binary segmentation, classes=1.
    """
    model = smp.DeepLabV3(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=num_classes,
        activation=None  # We'll do sigmoid in training/inference
    )
    return model


###############################################
# Metrics
###############################################
def calculate_iou(preds, targets, threshold=0.5):
    preds = (preds > threshold).float()
    intersection = (preds * targets).sum(dim=(2, 3))
    union = (preds + targets - preds * targets).sum(dim=(2, 3)) + 1e-7
    iou = intersection / union
    return iou.mean().item()


def calculate_dice(preds, targets, threshold=0.5):
    preds = (preds > threshold).float()
    intersection = (preds * targets).sum(dim=(2, 3))
    dice = (2.0 * intersection) / (preds.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) + 1e-7)
    return dice.mean().item()
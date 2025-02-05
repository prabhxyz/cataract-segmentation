import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

###############################################
# Custom Weighted Dice Loss
###############################################
class WeightedDiceLoss(nn.Module):
    """
    Custom weighted Dice Loss for multilabel segmentation.
    Computes the Dice loss for each channel (class) and applies a per-class weight.
    """
    def __init__(self, weights, eps=1e-7):
        super().__init__()
        self.weights = weights
        self.eps = eps

    def forward(self, outputs, targets):
        # outputs, targets: [B, C, H, W]
        dice_loss = 0.0
        for i, weight in enumerate(self.weights):
            pred_i = outputs[:, i, :, :]
            target_i = targets[:, i, :, :]
            intersection = (pred_i * target_i).sum(dim=(1, 2))
            union = pred_i.sum(dim=(1, 2)) + target_i.sum(dim=(1, 2))
            # Compute Dice loss per channel
            dice_loss_channel = 1 - (2 * intersection + self.eps) / (union + self.eps)
            dice_loss += weight * dice_loss_channel
        dice_loss = dice_loss / sum(self.weights)
        return dice_loss.mean()

###############################################
# Custom Focal Loss Function
###############################################
def focal_loss_with_logits(y_pred, y_true, alpha, gamma=2.0, reduction='mean'):
    """
    Computes focal loss with logits for multilabel segmentation.
    y_pred: logits, shape [B, C, H, W]
    y_true: ground truth (0 or 1), same shape
    alpha: tensor of shape [1, C, 1, 1] containing per-class weights.
    gamma: focusing parameter.
    """
    # Compute probabilities from logits.
    p = torch.sigmoid(y_pred)
    # Binary cross-entropy.
    bce = - (y_true * torch.log(p + 1e-7) + (1 - y_true) * torch.log(1 - p + 1e-7))
    # Focal modulation.
    focal_factor = (1 - p) ** gamma
    loss = alpha * focal_factor * bce
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss

###############################################
# Combined Loss: Weighted Dice Loss + Custom Focal Loss
###############################################
class CombinedLoss(nn.Module):
    """
    Combined loss: weighted Dice loss + focal loss.
    For model output order [Iris, Pupil, Lens], default class weights are [1.5, 1.0, 1.5].
    """
    def __init__(self, dice_weight=0.5, focal_weight=0.5, eps=1e-7, class_weights=None):
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.eps = eps
        if class_weights is None:
            class_weights = [1.5, 1.0, 1.5]
        self.class_weights = class_weights
        self.dice_loss_fn = WeightedDiceLoss(weights=self.class_weights, eps=self.eps)
        # We'll use our custom focal_loss_with_logits in the forward pass.

    def forward(self, outputs, targets):
        dice_loss = self.dice_loss_fn(outputs, targets)
        # Convert class_weights to a tensor and reshape to [1, C, 1, 1]
        alpha_tensor = torch.tensor(self.class_weights, dtype=outputs.dtype, device=outputs.device)
        alpha_tensor = alpha_tensor.view(1, -1, 1, 1)
        focal_loss = focal_loss_with_logits(outputs, targets, alpha=alpha_tensor, gamma=2.0, reduction='mean')
        return self.dice_weight * dice_loss + self.focal_weight * focal_loss

###############################################
# Model Definition
###############################################
def get_segmentation_model(num_classes=3, encoder_name='timm-efficientnet-b4', encoder_weights='imagenet'):
    """
    Returns a segmentation model.
    The default is Unet++ with the specified encoder.
    (You can experiment with alternatives, e.g., DeepLabV3Plus, without breaking inference.)
    """
    model = smp.UnetPlusPlus(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=num_classes,
        activation=None  # We'll apply sigmoid in training/inference.
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
    return iou.mean(dim=1).mean().item()

def calculate_dice(preds, targets, threshold=0.5):
    preds = (preds > threshold).float()
    intersection = (preds * targets).sum(dim=(2, 3))
    dice = (2 * intersection) / (preds.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) + 1e-7)
    return dice.mean(dim=1).mean().item()
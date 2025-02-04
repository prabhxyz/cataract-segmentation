import numpy as np

def compute_iou(pred, target, num_classes=3, ignore_label=255):
    """Return (per_class_iou, mean_iou)."""
    valid_mask = (target != ignore_label)
    pred = pred[valid_mask]
    target = target[valid_mask]

    ious = []
    for c in range(num_classes):
        pred_c = (pred == c)
        target_c = (target == c)
        if pred_c.sum()==0 and target_c.sum()==0:
            ious.append(1.0)
            continue
        intersection = (pred_c & target_c).sum()
        union = (pred_c | target_c).sum()
        iou = intersection / (union + 1e-6)
        ious.append(iou)
    return ious, float(np.mean(ious))

def compute_dice(pred, target, num_classes=3, ignore_label=255):
    """Return (per_class_dice, mean_dice)."""
    valid_mask = (target != ignore_label)
    pred = pred[valid_mask]
    target = target[valid_mask]

    dices = []
    for c in range(num_classes):
        pred_c = (pred==c).astype(np.float32)
        target_c = (target==c).astype(np.float32)
        intersection = (pred_c * target_c).sum()
        denom = pred_c.sum() + target_c.sum()
        dice_val = 2.0 * intersection / (denom + 1e-6)
        dices.append(dice_val)
    return dices, float(np.mean(dices))

def compute_precision_recall_f1(pred, target, num_classes=3, ignore_label=255):
    """
    Returns dict with per-class precision, recall, f1, plus macro avg.
    """
    valid_mask = (target != ignore_label)
    pred = pred[valid_mask]
    target = target[valid_mask]

    precisions = []
    recalls = []
    f1s = []

    for c in range(num_classes):
        pred_c = (pred==c)
        tgt_c  = (target==c)

        tp = (pred_c & tgt_c).sum()
        fp = (pred_c & ~tgt_c).sum()
        fn = (~pred_c & tgt_c).sum()

        precision = tp / (tp + fp + 1e-6)
        recall    = tp / (tp + fn + 1e-6)
        f1        = 2.0 * precision * recall / (precision + recall + 1e-6)

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    # macro average
    prec_avg = float(np.mean(precisions))
    rec_avg  = float(np.mean(recalls))
    f1_avg   = float(np.mean(f1s))

    return {
        "per_class_precision": list(map(float, precisions)),
        "per_class_recall": list(map(float, recalls)),
        "per_class_f1": list(map(float, f1s)),
        "mean_precision": prec_avg,
        "mean_recall": rec_avg,
        "mean_f1": f1_avg
    }

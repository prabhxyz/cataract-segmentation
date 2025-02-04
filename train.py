import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from supervisely_dataset import make_train_val_test_lists, SuperviselyDataset
from swin_unet import SwinUNet
from metrics import compute_iou, compute_dice, compute_precision_recall_f1
from plot_utils import save_results_to_json, plot_segmentation_metrics

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    total_loss= 0.0
    for images, masks in tqdm(loader, desc="Training", ncols=100):
        images = images.to(device, dtype=torch.float32)
        masks  = masks.to(device, dtype=torch.long)

        optimizer.zero_grad()
        logits = model(images)  # => [B,3,H,W]
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    avg_loss = total_loss / len(loader.dataset)
    return avg_loss

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    total_loss = 0.0

    all_ious  = []
    all_dices = []
    all_precs = []
    all_recs  = []
    all_f1s   = []

    for images, masks in tqdm(loader, desc="Evaluating", ncols=100):
        images = images.to(device, dtype=torch.float32)
        masks  = masks.to(device, dtype=torch.long)

        logits = model(images)
        loss = criterion(logits, masks)
        total_loss += loss.item() * images.size(0)

        preds = torch.argmax(logits, dim=1).cpu().numpy()
        gts   = masks.cpu().numpy()

        for b in range(preds.shape[0]):
            # IoU & Dice
            ious, miou = compute_iou(preds[b], gts[b], num_classes=3, ignore_label=255)
            dices, mdice = compute_dice(preds[b], gts[b], num_classes=3, ignore_label=255)
            # Precision/Recall/F1
            prf = compute_precision_recall_f1(preds[b], gts[b], num_classes=3, ignore_label=255)

            all_ious.append(ious)
            all_dices.append(dices)
            all_precs.append(prf["per_class_precision"])
            all_recs.append(prf["per_class_recall"])
            all_f1s.append(prf["per_class_f1"])

    avg_loss = total_loss / len(loader.dataset)

    import numpy as np
    ious_arr   = np.array(all_ious)   # shape [N,3]
    dices_arr  = np.array(all_dices)
    precs_arr  = np.array(all_precs)
    recs_arr   = np.array(all_recs)
    f1s_arr    = np.array(all_f1s)

    iou_per_class   = ious_arr.mean(axis=0).tolist()
    dice_per_class  = dices_arr.mean(axis=0).tolist()
    mean_iou        = float(np.mean(ious_arr.mean(axis=1)))
    mean_dice       = float(np.mean(dices_arr.mean(axis=1)))

    mean_prec_class = precs_arr.mean(axis=0).tolist()
    mean_rec_class  = recs_arr.mean(axis=0).tolist()
    mean_f1_class   = f1s_arr.mean(axis=0).tolist()

    mean_prec = float(np.mean(precs_arr.mean(axis=1)))
    mean_rec  = float(np.mean(recs_arr.mean(axis=1)))
    mean_f1   = float(np.mean(f1s_arr.mean(axis=1)))

    metrics = {
        "loss": avg_loss,
        "iou_per_class": iou_per_class,
        "mean_iou": mean_iou,
        "dice_per_class": dice_per_class,
        "mean_dice": mean_dice,

        "precision_per_class": mean_prec_class,
        "recall_per_class": mean_rec_class,
        "f1_per_class": mean_f1_class,

        "mean_precision": mean_prec,
        "mean_recall": mean_rec,
        "mean_f1": mean_f1
    }
    return metrics

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Data splits
    train_list, val_list, test_list = make_train_val_test_lists()

    # 2) Datasets
    train_dataset = SuperviselyDataset(train_list)
    val_dataset   = SuperviselyDataset(val_list)
    test_dataset  = SuperviselyDataset(test_list)

    # 3) DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True,  num_workers=4, drop_last=True)
    val_loader   = DataLoader(val_dataset,   batch_size=2, shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_dataset,  batch_size=2, shuffle=False, num_workers=4)

    # 4) Model
    model = SwinUNet(num_classes=3).to(device)

    # 5) Optimizer & scheduler
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # 6) Training
    EPOCHS = 10
    best_val_f1 = 0.0
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(1, EPOCHS+1):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        # Evaluate
        val_metrics = evaluate(model, val_loader, device)
        val_loss = val_metrics["loss"]
        val_f1   = val_metrics["mean_f1"]
        val_iou  = val_metrics["mean_iou"]
        val_dice = val_metrics["mean_dice"]

        scheduler.step()

        print(f"\n[Epoch {epoch}/{EPOCHS}] "
              f"train_loss={train_loss:.4f} "
              f"val_loss={val_loss:.4f} "
              f"val_mF1={val_f1:.4f} "
              f"val_mIoU={val_iou:.4f} "
              f"val_mDice={val_dice:.4f}")

        # ------------------------------
        # SAVE A CHECKPOINT EVERY EPOCH
        # ------------------------------
        # This ensures we keep a partial model even if we kill training early.
        epoch_ckpt_path = f"checkpoints/model_epoch_{epoch}.pth"
        torch.save(model.state_dict(), epoch_ckpt_path)
        print(f"Saved epoch checkpoint to: {epoch_ckpt_path}")

        # Optionally also save best if improved
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), "checkpoints/best_model.pth")
            print("Updated best_model.pth (based on val_mF1)")

    # 7) Evaluate on test set
    model.load_state_dict(torch.load("checkpoints/best_model.pth", map_location=device))
    test_metrics = evaluate(model, test_loader, device)
    print("Test metrics:", test_metrics)

    # 8) Save & Plot
    save_results_to_json(test_metrics, "evaluation_results.json")
    plot_segmentation_metrics(test_metrics, "evaluation_metrics.png")

if __name__ == "__main__":
    main()

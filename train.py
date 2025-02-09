import os
import glob
import random
import argparse
import json

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler
import torch.distributed as dist

from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import OneCycleLR

import matplotlib.pyplot as plt
from tqdm import tqdm

from cataract_dataset import CataractDataset
from augmentations import get_training_augmentations, get_validation_augmentations
from models import get_segmentation_model, CombinedLoss, calculate_iou, calculate_dice

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True,
                        help="Path to the Cataract-1k dataset root.")
    parser.add_argument('--epochs', type=int, default=100,
                        help="Number of training epochs.")
    parser.add_argument('--batch_size', type=int, default=4,
                        help="Batch size per GPU.")
    parser.add_argument('--lr', type=float, default=1e-4,
                        help="Peak learning rate for OneCycleLR.")
    parser.add_argument('--encoder_name', type=str, default='resnet101',
                        help="Backbone encoder name for DeepLabV3.")
    parser.add_argument('--encoder_weights', type=str, default='imagenet',
                        help="Pretrained weights for the encoder.")
    parser.add_argument('--num_workers', type=int, default=4,
                        help="Number of DataLoader workers.")
    parser.add_argument('--use_ddp', action='store_true',
                        help="Use DistributedDataParallel.")
    parser.add_argument('--local_rank', type=int, default=0,
                        help="Local rank for DDP.")
    return parser.parse_args()

def find_samples(data_dir):
    ann_base = os.path.join(data_dir, "Annotations", "Images-and-Supervisely-Annotations")
    all_cases = sorted(os.listdir(ann_base))
    samples = []
    for case in all_cases:
        img_dir = os.path.join(ann_base, case, "img")
        ann_dir = os.path.join(ann_base, case, "ann")
        if not os.path.exists(img_dir) or not os.path.exists(ann_dir):
            continue
        png_files = glob.glob(os.path.join(img_dir, "*.png"))
        for img_path in png_files:
            filename = os.path.basename(img_path)
            ann_path = os.path.join(ann_dir, filename + ".json")
            if os.path.exists(ann_path):
                samples.append((img_path, ann_path))
    return samples

def split_dataset(samples, train_ratio=0.9):
    random.shuffle(samples)
    n_total = len(samples)
    n_train = int(n_total * train_ratio)
    train_samples = samples[:n_train]
    val_samples = samples[n_train:]
    return train_samples, val_samples

def main():
    args = parse_args()
    if args.use_ddp:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(args.local_rank)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cudnn.benchmark = True

    samples = find_samples(args.data_dir)
    train_samples, val_samples = split_dataset(samples, train_ratio=0.9)

    train_dataset = CataractDataset(
        train_samples,
        transform=get_training_augmentations()
    )
    val_dataset = CataractDataset(
        val_samples,
        transform=get_validation_augmentations()
    )

    if args.use_ddp:
        train_sampler = DistributedSampler(train_dataset)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
    else:
        train_sampler = RandomSampler(train_dataset)
        val_sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True
    )

    model = get_segmentation_model(
        num_classes=1,
        encoder_name=args.encoder_name,
        encoder_weights=args.encoder_weights
    )
    if args.use_ddp:
        model = model.to(device)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank
        )
    else:
        model = torch.nn.DataParallel(model)
        model = model.to(device)

    criterion = CombinedLoss(dice_weight=0.5, focal_weight=0.5, weight=1.0)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

    # OneCycleLR for stable ramp-up/down of LR
    steps_per_epoch = len(train_loader)
    total_steps = args.epochs * steps_per_epoch
    scheduler = OneCycleLR(optimizer, max_lr=args.lr, total_steps=total_steps)

    scaler = GradScaler()

    train_loss_hist, val_loss_hist = [], []
    train_iou_hist, val_iou_hist = [], []
    train_dice_hist, val_dice_hist = [], []
    best_val_iou = 0.0

    for epoch in range(args.epochs):
        if args.use_ddp:
            train_sampler.set_epoch(epoch)

        # Training
        model.train()
        running_loss, running_iou, running_dice = 0.0, 0.0, 0.0
        step = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False)
        for images, masks in progress_bar:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=True):
                outputs = model(images)
                loss = criterion(outputs, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += loss.item()
            if step % 10 == 0 and step > 0:
                with torch.no_grad():
                    preds = torch.sigmoid(outputs)
                    iou_val = calculate_iou(preds, masks)
                    dice_val = calculate_dice(preds, masks)
                    running_iou += iou_val
                    running_dice += dice_val
            step += 1
            progress_bar.set_postfix(loss=running_loss/step)

        epoch_loss = running_loss / step
        epoch_iou = running_iou / (step / 10) if step >= 10 else 0.0
        epoch_dice = running_dice / (step / 10) if step >= 10 else 0.0
        train_loss_hist.append(epoch_loss)
        train_iou_hist.append(epoch_iou)
        train_dice_hist.append(epoch_dice)

        if (not args.use_ddp) or (dist.get_rank() == 0):
            print(f"[Epoch {epoch+1}/{args.epochs}] TRAIN loss={epoch_loss:.4f} IoU={epoch_iou:.4f} Dice={epoch_dice:.4f}")

        # Validation
        if (not args.use_ddp) or (dist.get_rank() == 0):
            model.eval()
            val_loss, val_iou, val_dice = 0.0, 0.0, 0.0
            val_steps = 0

            for vimages, vmasks in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}", leave=False):
                vimages, vmasks = vimages.to(device), vmasks.to(device)
                with torch.cuda.amp.autocast(enabled=True), torch.no_grad():
                    voutputs = model(vimages)
                    vloss = criterion(voutputs, vmasks)
                    vpreds = torch.sigmoid(voutputs)
                val_loss += vloss.item()
                val_iou += calculate_iou(vpreds, vmasks)
                val_dice += calculate_dice(vpreds, vmasks)
                val_steps += 1

            if val_steps > 0:
                val_loss /= val_steps
                val_iou /= val_steps
                val_dice /= val_steps

            val_loss_hist.append(val_loss)
            val_iou_hist.append(val_iou)
            val_dice_hist.append(val_dice)
            print(f"   VAL loss={val_loss:.4f} IoU={val_iou:.4f} Dice={val_dice:.4f}")

            # Save checkpoint
            epoch_ckpt = f"model_epoch_{epoch+1}.pth"
            if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
                torch.save(model.module.state_dict(), epoch_ckpt)
            else:
                torch.save(model.state_dict(), epoch_ckpt)
            print(f"   Saved epoch checkpoint: {epoch_ckpt}")

            # Update best model
            if val_iou > best_val_iou:
                best_val_iou = val_iou
                if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
                    torch.save(model.module.state_dict(), "best_model.pth")
                else:
                    torch.save(model.state_dict(), "best_model.pth")
                print("   Updated best_model.pth")

    # Plot
    if (not args.use_ddp) or (dist.get_rank() == 0):
        epochs_range = range(1, args.epochs+1)
        plt.figure(figsize=(15,5))

        plt.subplot(1,3,1)
        plt.plot(epochs_range, train_loss_hist, label='Train Loss')
        plt.plot(epochs_range, val_loss_hist, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss')
        plt.legend()

        plt.subplot(1,3,2)
        plt.plot(epochs_range, train_iou_hist, label='Train IoU')
        plt.plot(epochs_range, val_iou_hist, label='Val IoU')
        plt.xlabel('Epoch')
        plt.ylabel('IoU')
        plt.title('IoU')
        plt.legend()

        plt.subplot(1,3,3)
        plt.plot(epochs_range, train_dice_hist, label='Train Dice')
        plt.plot(epochs_range, val_dice_hist, label='Val Dice')
        plt.xlabel('Epoch')
        plt.ylabel('Dice')
        plt.title('Dice')
        plt.legend()

        plt.tight_layout()
        plt.savefig("training_results.png")
        plt.show()

    if args.use_ddp:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
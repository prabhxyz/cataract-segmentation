import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class CataractDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, ann_path = self.samples[idx]

        # Load image
        image_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # Load JSON
        with open(ann_path, 'r') as f:
            ann_data = json.load(f)

        h, w = ann_data["size"]["height"], ann_data["size"]["width"]
        mask = np.zeros((h, w), dtype=np.uint8)

        # Only fill polygons with classTitle="Cornea"
        for obj in ann_data["objects"]:
            if obj["classTitle"] == "Cornea":
                pts = np.array(obj["points"]["exterior"], dtype=np.int32)
                cv2.fillPoly(mask, [pts], 1)

        # Albumentations transform
        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        # Convert to tensor
        image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        mask = torch.from_numpy(mask).unsqueeze(0).float()

        return image, mask
import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

# Mapping from classTitle in the JSON to channel index in the mask
# Adjust or extend as needed.
CLASS_MAP = {
    "Iris": 0,
    "Pupil": 1,
    "Lens": 2
}


def polygons_to_mask(polygons, image_shape):
    """
    Convert a list of polygons (each is list of [x, y] points) into
    a single binary mask of shape (H, W).
    """
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    # polygons is expected to be a list of lists of [x, y]
    # We must convert each polygon to int32 for cv2.fillPoly
    pts = [np.array(polygons, dtype=np.int32)]
    cv2.fillPoly(mask, pts, 1)
    return mask


class CataractDataset(Dataset):
    def __init__(self, samples, transform=None):
        """
        Args:
            samples (list): Each element is (img_path, ann_path).
            transform (albumentations.Compose): Transform pipeline.
        """
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, ann_path = self.samples[idx]

        # Load image
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB

        # Load JSON annotation
        with open(ann_path, 'r') as f:
            ann_data = json.load(f)

        # Prepare multi-channel mask
        height, width = ann_data["size"]["height"], ann_data["size"]["width"]
        num_classes = len(CLASS_MAP)  # 3: Iris, Pupil, Lens
        mask = np.zeros((height, width, num_classes), dtype=np.uint8)

        for obj in ann_data["objects"]:
            class_title = obj["classTitle"]
            if class_title not in CLASS_MAP:
                # Skip classes not in CLASS_MAP (e.g., "Cornea" if not needed)
                continue

            channel_idx = CLASS_MAP[class_title]
            polygon_pts = obj["points"]["exterior"]
            # Convert polygon to mask
            polygon_mask = polygons_to_mask(polygon_pts, (height, width, 3))
            mask[..., channel_idx] = np.maximum(mask[..., channel_idx], polygon_mask)

        # Optional transforms (Albumentations expects dict with "image", "mask")
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        # Convert to tensors
        # image: shape (C, H, W), mask: shape (num_classes, H, W)
        image = torch.from_numpy(image.transpose(2, 0, 1)).float()  # [3, H, W]
        mask = torch.from_numpy(mask.transpose(2, 0, 1)).float()  # [3, H, W]

        return image, mask
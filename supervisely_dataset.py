import os
import json
import glob
import random
import torch
import numpy as np
import cv2

from torch.utils.data import Dataset

# Fixed resolution for the model (you can change to 512, etc.)
FIXED_SIZE = 384

# We keep only these classes (Iris=0, Pupil=1, Lens=2). Everything else => ignore=255.
CLASS_MAP = {
    "Iris": 0,
    "Pupil": 1,
    "Lens": 2
}

class SuperviselyDataset(Dataset):
    """
    Reads from 'Cataract-1k/Annotations/Images-and-Supervisely-Annotations/case_XXXX'
     - ann/caseXXXX_01.png.json
     - img/caseXXXX_01.png
    For each frame, we parse polygons for our classes of interest (Iris, Pupil, Lens),
    fill a mask with 0/1/2, and set everything else (including background) to 255 (ignore).
    """
    def __init__(self, pair_list, max_frames=9999):
        """
        pair_list: List of (json_path, image_path)
                   e.g. /.../case_XXXX/ann/caseXXXX_01.png.json , /.../case_XXXX/img/caseXXXX_01.png
        max_frames: how many frames to read from the list (truncate if needed)
        """
        self.samples = pair_list[:max_frames]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        json_path, img_path = self.samples[idx]

        # 1) Read image
        image_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image_bgr is None:
            # Fallback
            image_bgr = np.zeros((FIXED_SIZE, FIXED_SIZE, 3), dtype=np.uint8)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # 2) Build empty mask => label=255 for ignore by default
        h, w, _ = image_rgb.shape
        mask = np.full((h, w), 255, dtype=np.uint8)

        # 3) Parse the JSON polygons
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                ann_data = json.load(f)
            if "objects" in ann_data:
                for obj in ann_data["objects"]:
                    class_title = obj.get("classTitle", "")
                    if class_title in CLASS_MAP:
                        label_idx = CLASS_MAP[class_title]
                        # polygon points => exterior => fill
                        exterior = obj["points"]["exterior"]  # list of [x, y]
                        if len(exterior) > 2:
                            pts = np.array(exterior, dtype=np.int32).reshape((-1,2))
                            # Fill the polygon with label_idx
                            cv2.fillPoly(mask, [pts], color=label_idx)

        # 4) Resize image & mask
        image_rgb = cv2.resize(image_rgb, (FIXED_SIZE, FIXED_SIZE), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (FIXED_SIZE, FIXED_SIZE), interpolation=cv2.INTER_NEAREST)

        # 5) Convert to torch
        image_tensor = torch.from_numpy(image_rgb.transpose(2,0,1)).float()
        mask_tensor  = torch.from_numpy(mask).long()
        return image_tensor, mask_tensor


def make_train_val_test_lists(root_dir="Cataract-1k/Annotations/Images-and-Supervisely-Annotations",
                              split_ratio=(0.8, 0.1, 0.1)):
    """
    We gather all frames from e.g.
      Cataract-1k/Annotations/Images-and-Supervisely-Annotations/case_XXXX/ann/caseXXXX_01.png.json
      and the matching .../img/caseXXXX_01.png
    Then do an 80/10/10 split.
    Returns: (train_list, val_list, test_list) where each is a list of (json_path, img_path).
    """
    # We'll recursively find all *png.json in "ann" folders
    # Then build the corresponding image path
    all_samples = []
    for case_dir in os.listdir(root_dir):
        full_case_path = os.path.join(root_dir, case_dir)
        ann_dir = os.path.join(full_case_path, "ann")
        img_dir = os.path.join(full_case_path, "img")
        if not os.path.isdir(ann_dir) or not os.path.isdir(img_dir):
            continue

        json_files = glob.glob(os.path.join(ann_dir, "*.png.json"))
        for jfile in json_files:
            # e.g. jfile = /.../ann/caseXXXX_01.png.json
            # the matching image is /.../img/caseXXXX_01.png
            base_name = os.path.basename(jfile)  # e.g. caseXXXX_01.png.json
            img_name = base_name.replace(".json", "")  # => "caseXXXX_01.png"
            img_path = os.path.join(img_dir, img_name)
            if os.path.exists(img_path):
                all_samples.append((jfile, img_path))

    random.shuffle(all_samples)
    n = len(all_samples)
    train_end = int(split_ratio[0]*n)
    val_end   = int((split_ratio[0]+split_ratio[1])*n)

    train_list = all_samples[:train_end]
    val_list   = all_samples[train_end:val_end]
    test_list  = all_samples[val_end:]
    return train_list, val_list, test_list

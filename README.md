# Cataract Surgery Segmentation with Swin U-Net

This repository provides a complete pipeline for **semantic segmentation** of cataract surgery frames/images using a **Swin U-Net** architecture. We handle **Supervisely**-style annotation JSON files (rather than COCO JSON), focusing specifically on the segmentation of these classes:

- **Iris** (label = 0)
- **Pupil** (label = 1)
- **Lens** (label = 2)

No explicit "background" class is modeled; any unlabeled pixel is assigned an "ignore" label of `255`.

## Table of Contents

1. [Project Overview](#project-overview)  
2. [Data Format & Structure](#data-format--structure)  
3. [Model Architecture](#model-architecture)  
   1. [Encoder: Swin Transformer Large](#1-encoder-swin-transformer-large)  
   2. [Decoder: UNet-Style Upsampling](#2-decoder-unet-style-upsampling)  
   3. [Why Swin U-Net is Great](#why-swin-u-net-is-great)  
4. [Files in This Repo](#files-in-this-repo)  
   1. [train.py](#trainpy)  
   2. [inference.py](#inferencepy)  
   3. [swin_unet.py](#swin_unetpy)  
   4. [supervisely_dataset.py](#supervisely_datasetpy)  
   5. [metrics.py](#metricspy)  
   6. [plot_utils.py](#plot_utilspy)  
5. [Installation](#installation)  
6. [Usage](#usage)  
   1. [Training](#training)  
   2. [Inference](#inference)  
   3. [Results & Evaluation](#results--evaluation)  
7. [Acknowledgments](#acknowledgments)  
8. [Author](#author)

---

## Project Overview

This project aims to **accurately segment** key structures in cataract surgery images—particularly the **Iris**, **Pupil**, and **Lens**. We adopt a **Swin U-Net** architecture, which pairs a **Swin Transformer Large** encoder (from the [timm library](https://github.com/rwightman/pytorch-image-models)) with a classic **U-Net** style decoder. Data are annotated via **Supervisely** polygon JSON files.

### Goals

- **Ease of Use**: Simple training script with straightforward dataset handling.  
- **High Accuracy**: A powerful transformer-based backbone (Swin Large) plus a U-Net decoder that preserves spatial detail for fine-grained segmentation tasks.  
- **Modular**: The code is split into multiple smaller Python scripts for clarity and maintainability.  

---

## Data Format & Structure

We assume a directory structure like:

```
YourProject/
  ├─ train.py
  ├─ inference.py
  ├─ swin_unet.py
  ├─ supervisely_dataset.py
  ├─ metrics.py
  ├─ plot_utils.py
  ├─ README.md
  ├─ requirements.txt
  └─ Cataract-1k/
      └─ Annotations/
         └─ Images-and-Supervisely-Annotations/
            ├─ case_0001/
            │   ├─ ann/
            │   │  ├─ case0001_01.png.json
            │   │  ├─ case0001_02.png.json
            │   │  └─ ...
            │   └─ img/
            │      ├─ case0001_01.png
            │      ├─ case0001_02.png
            │      └─ ...
            ├─ case_0002/
            │   ├─ ann/
            │   └─ img/
            ├─ ...
```

Where each **`case_XXXX`** folder has:

- An `ann` subfolder with JSON polygon annotations (one `.png.json` per frame).
- An `img` subfolder with the corresponding `.png` frames.

**In each JSON** (`caseXXXX_01.png.json`), you have a structure like:

```json
{
  "size": {"height": 768, "width": 1024},
  "objects": [
    {
      "classTitle": "Pupil",
      "points": {
        "exterior": [[635,320], [644,327], ...],
        "interior": []
      }
    },
    ...
  ]
}
```

We only keep `"Iris"`, `"Pupil"`, and `"Lens"`. Everything else is marked as ignore label (`255`).

---

## Model Architecture

We use a **Swin U-Net**. This architecture is composed of:

1. **Encoder**: **Swin Transformer Large** (from `timm.create_model(...)`) which is pretrained on ImageNet. We extract features from 4 stages (`out_indices=(0,1,2,3)`).

2. **Decoder**: A standard U-Net approach where:
   - We **upsample** the deepest feature map.
   - **Concatenate** with the skip feature from the earlier stage.
   - Pass through convolutional blocks.
   - Repeat for each of the 4 stages, from deepest to shallowest.

### 1) Encoder: Swin Transformer Large

- The **Swin Transformer** is a hierarchical vision transformer that processes images in non-overlapping “windows,” providing local self-attention with a shift mechanism for global context.  
- **Swin Large** typically has ~1536 channels in the final stage, ~768 in the next, etc.  
- We set **`img_size=384`** for input images to match the transformer's patch embedding.

### 2) Decoder: UNet-Style Upsampling

- We **interpolate** the features back to the spatial resolution of a previous stage.
- Then **concatenate** the upsampled feature with the corresponding skip connection from the encoder.
- Apply **two 3×3 convolution** layers (with `BatchNorm + ReLU`).
- Continue until we reach an intermediate final layer, then do one last upsample and reduce to some smaller number of channels.

### Why **Swin U-Net** Is Great

1. **Transformer-based**: Swin brings strong representation learning with powerful multi-head self-attention across windows—excellent for complex medical imaging tasks.  
2. **Hierarchical**: Gains multi-scale features (like CNN) but with the expressive power of Transformers.  
3. **U-Net synergy**: The skip connections preserve fine spatial details, crucial for segmenting small structures like the pupil or lens edges.  
4. **Pretraining**: We leverage a massive ImageNet-pretrained backbone, speeding up convergence and improving final accuracy.  
5. **State-of-the-Art**: This approach often outperforms older CNN-based or purely transformer-based models on diverse segmentation tasks, especially in the medical domain.

---

## Files in This Repo

Below is a summary of the Python files. They’re each small and modular.

### **train.py**

- **Main** training script.  
- Splits the dataset into train/val/test, loads them via `DataLoader`, and trains for a specified number of epochs.  
- Uses `CrossEntropyLoss(ignore_index=255)` so unlabeled pixels don’t affect training.  
- Logs metrics (IoU, Dice, Precision, Recall, F1).  
- Saves the **best** model checkpoint.  

### **inference.py**

- Loads the best checkpoint from training.  
- Reads a single `caseXXXX_YY.png` + `caseXXXX_YY.png.json` from the `ann/` and `img/` directories.  
- Runs forward pass in the **Swin U-Net**.  
- Converts the predicted labels into a color overlay (e.g., Iris=Green, Pupil=Blue, Lens=Red).  
- Saves an `inference_overlay.png` to disk.

### **swin_unet.py**

- Contains the **Swin U-Net** architecture.  
- Creates a **Swin Transformer** from timm (`features_only=True`) as the encoder.  
- Builds a **U-Net** style decoder with skip connections.  
- Produces a 3-channel output for {Iris, Pupil, Lens}.  

### **supervisely_dataset.py**

- Handles reading **Supervisely**-style polygon JSON (`.png.json`) plus the matching PNG image.  
- For each polygon, if its `classTitle` is in {“Iris”, “Pupil”, “Lens”}, we fill the mask with 0/1/2 respectively.  
- All else → label=255 (ignore).  
- Resizes to 384×384 for input to the model.  
- Returns `(image_tensor, mask_tensor)`.

### **metrics.py**

- Contains code to compute:
  - **IoU** (Intersection over Union)  
  - **Dice** coefficient  
  - **Precision, Recall, F1**  
- Ignores label=255 in calculations.

### **plot_utils.py**

- Saves results dictionary to a JSON file.  
- Creates bar plots for IoU, Dice, etc.  
- You can expand it to plot Precision, Recall, F1 if desired.

---

## Installation

1. **Clone** this repo:
   ```bash
   git clone https://github.com/prabhxyz/cataract-segmentation.git
   cd cataract-segmentation
   ```
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   This should pull in PyTorch, timm, OpenCV, etc.

3. **Check** your CUDA availability if you want GPU acceleration. If you don’t have a GPU, it will run on CPU (but slowly).

---

## Usage

### **Training**

1. Make sure your data is in the recommended structure (e.g. `Cataract-1k/Annotations/Images-and-Supervisely-Annotations/case_0001/...`).
2. Run:
   ```bash
   python train.py
   ```
3. The script:
   - Creates train/val/test splits of the frames.  
   - Trains the **Swin U-Net** for 10 epochs (or as specified).  
   - Saves the best checkpoint to `checkpoints/best_model.pth`.  
   - Writes final test metrics to `evaluation_results.json` and a chart to `evaluation_metrics.png`.

### **Inference**

1. Once training is done, pick any single `.png` + `.png.json` from e.g. `case_0001`.
2. Edit `inference.py` if you want to specify the paths, or pass them as arguments (depending on your script).  
3. Run:
   ```bash
   python inference.py
   ```
4. It loads the model, runs inference, and saves an overlay image (like `inference_overlay.png`) that highlights the Iris, Pupil, and Lens regions.

### **Results & Evaluation**

- You can see metrics like **IoU, Dice, Precision, Recall, F1** in your console logs.  
- The final test set results are stored in `evaluation_results.json`.  
- The bar chart is in `evaluation_metrics.png`.  
- Typical keys in `evaluation_results.json` might include:
  ```json
  {
    "loss": 0.1234,
    "iou_per_class": [0.82, 0.77, 0.80],
    "mean_iou": 0.79,
    "dice_per_class": [0.88, 0.82, 0.86],
    "mean_dice": 0.85,
    ...
  }
  ```
- If you see no segmentation in the overlays, confirm that your Supervisely polygons are actually labeled “Iris,” “Pupil,” or “Lens.”  

---

*Built by [Prabhdeep](https://github.com/prabhxyz).*
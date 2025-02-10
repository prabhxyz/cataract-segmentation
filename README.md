### Project Overview

This project provides a **complete pipeline** for **automatic cornea segmentation** in cataract surgery video frames. Given an input image (extracted from a surgery video), the goal is to predict a **binary mask** highlighting only the **cornea** region. The solution leverages:

- **PSPNet** (Pyramid Scene Parsing Network) from the [Segmentation Models PyTorch](https://github.com/qubvel/segmentation_models.pytorch) library, using a **ResNet101** backbone.  
- A **Dice + Focal** combined loss function to handle ambiguous boundaries and partial occlusions.  
- A **no-augmentation** data strategy, focusing on consistent image appearance for cornea detection.  
- A training regimen that reaches **high IoU (~0.85–0.90)** on the validation set after sufficient epochs.

The repository’s workflow covers data loading, model training, inference on video frames, and result visualization.

---

### Dataset & Annotations

- **Data Source**: We use the publicly available **Cataract-1k** dataset, containing images (extracted frames) and **Supervisely**-style JSON annotations.  
- **Annotation Format**: JSON files define polygons for each class, but our focus is solely on `"Cornea"`. The dataset excludes other structures like iris or pupil.  
- **Usage**: During training, the dataset is split **80/20** (train/validation). Frames are resized to 512×512, with no further augmentations.  

---

### Model Architecture

1. **PSPNet**  
   - A widely used architecture that employs a **pyramid pooling module** to capture global context—especially useful for large uniform regions like the cornea.  
   - Encoder: **ResNet101** backbone (pretrained on ImageNet).  
   - Decoder: PSP module that merges multi‐scale features for refined segmentation.

2. **Loss Function**: **Combined (Dice + Focal)**  
   - **Dice Loss** helps the model capture imbalanced classes or small boundaries by maximizing overlap between prediction and ground truth.  
   - **Focal Loss** addresses uncertain pixels by focusing on “hard” or misclassified areas.  
   - Combined approach significantly improves boundary segmentation and reduces false positives.

---

### Training Procedure

1. **Data Split**: We perform an **80%** train / **20%** validation split of all annotated frames.  
2. **No Augmentation**: Only a final resize to 512×512, ensuring consistent orientation and color are preserved.  
3. **Optimizer & LR Scheduler**: 
   - **AdamW** with default momentum/beta settings.  
   - **CosineAnnealingLR** across 150 epochs, starting at **5e-5** LR and decaying to a minimum of 1e-6.  
4. **Mixed Precision**: We use PyTorch’s `torch.cuda.amp` for mixed-precision training to reduce memory usage and speed up training.  
5. **Checkpoints & Logging**:  
   - Every epoch, the model and metrics are saved in `model_weights/model_epoch_{E}.pth`.  
   - The best model (based on highest validation IoU) is `model_weights/best_model.pth`.  
   - Training logs (loss, IoU, Dice) are plotted to `training_results.png`.

---

### Final Results

As training progresses, the model’s IoU and Dice scores rise dramatically after ~100 epochs, eventually exceeding **0.90** IoU on the validation set. Below is a sample of the training curves (Loss, IoU, Dice) across ~150 epochs:

![Final Results](https://raw.githubusercontent.com/prabhxyz/cataract-segmentation/refs/heads/main/results/final_training_results.png)

In the above figure:
- **Left**: Loss initially fluctuates due to random initialization and partial saturations but steadily converges below 0.5.  
- **Center**: IoU climbs from near 0.1 in early epochs to roughly 0.85–0.90 by epoch ~140–150.  
- **Right**: Dice (F 1 measure for segmentation) similarly peaks near 0.9–0.95.  

This indicates the **PSPNet** + **Dice+Focal** approach effectively learns the cornea shape with consistent, stable performance after enough epochs.

---

### Model & Weight Files

Because the final trained model is quite large, we provide it via a **Google Drive link**. You can **download** the file here:

> [**Download PSPNet ResNet101 Model Weights**](https://drive.google.com/file/d/1ev_Fx6IjrRX8MNqx1RptAB140dBj3_8e/view?usp=sharing)

Save the downloaded file as `model_weights/best_model.pth` (or similar) in your local repository:

```bash
mkdir -p model_weights
mv /path/to/downloaded_best_model.pth model_weights/best_model.pth
```

---

### Running the Code

#### 1. Training

To **train** the model from scratch (on the **Cataract-1k** dataset) and store weights in `model_weights/`:

```bash
python train.py \
  --data_dir Cataract-1k \
  --epochs 150 \
  --batch_size 4 \
  --lr 5e-5 \
  --encoder_name resnet101 \
  --encoder_weights imagenet
```

- Replace any hyperparameters as needed (e.g., `--epochs 200`, etc.).  
- Each epoch’s checkpoint is saved under `model_weights/model_epoch_{E}.pth`, and the best model is `model_weights/best_model.pth`.

#### 2. Inference

To **infer** on a single video frame from, for example, `Cataract-1k/videos/case_5013.mp4` at timestamp 10.5 seconds:

```bash
python inference.py \
  --video_path Cataract-1k/videos/case_5013.mp4 \
  --timestamp 10.5 \
  --checkpoint model_weights/best_model.pth \
  --encoder_name resnet101 \
  --encoder_weights imagenet \
  --output_dir output
```

This extracts the frame at 10.5 s, loads the saved model weights, and outputs:
- **original.png**  
- **Cornea.png** (binary mask)  
- **overlay.png** (the original image with a green cornea overlay)

All stored in the `output/` folder.

---

### Conclusions & Future Work

- **High IoU**: Achieving ~0.90–0.95 IoU on validation after ~150 epochs underscores PSPNet's efficacy for **large, uniform** anatomical structures like the cornea.  
- **No Augmentations**: Removing them prevented confusion from unrealistic flips/rotations. If your dataset has more varied angles, you might reintroduce mild transformations.  
- **Next Steps**: 
  - Explore **boundary loss** or more advanced transformations for further refinement.  
  - Check if partial occlusions or lens reflections require special data curation.  
  - If memory permits, a bigger batch size or a different backbone (e.g., `timm-resnest101e`) can yield further gains.

By following this pipeline, you can reliably generate **binary cornea masks** from cataract surgery frames, facilitating advanced surgical analysis, AR overlays, or real‑time surgical assistance.

*Built by [Prabhdeep](https://github.com/prabhxyz) :)*
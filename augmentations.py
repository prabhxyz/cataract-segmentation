import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_training_augmentations():
    """
    Returns a composed Albumentations transform for training.
    Added geometric and color augmentations along with an ElasticTransform.
    Note: Removed the unsupported 'alpha_affine' parameter.
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10)
        ], p=0.7),
        A.RandomResizedCrop(
            size=(512, 512),
            scale=(0.8, 1.0),
            ratio=(1.0, 1.0),
            interpolation=1,         # cv2.INTER_LINEAR
            mask_interpolation=0,      # cv2.INTER_NEAREST
            p=0.5
        ),
        A.ElasticTransform(alpha=1, sigma=50, p=0.2),
        A.Resize(height=512, width=512),
    ])

def get_validation_augmentations():
    """
    Returns a minimal Albumentations transform for validation.
    """
    return A.Compose([
        A.Resize(height=512, width=512),
    ])
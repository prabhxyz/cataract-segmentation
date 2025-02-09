import albumentations as A

def get_training_augmentations():
    """
    Moderate augmentations: horizontal flip, slight color changes, random resized crop,
    mild elastic transform, then resize to 512x512.
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10),
        ], p=0.7),
        # For Albumentations>=1.2, use "size=(512,512)" not separate "height/width"
        A.RandomResizedCrop(
            size=(512, 512),
            scale=(0.8, 1.0),
            ratio=(1.0, 1.0),
            interpolation=1,       # cv2.INTER_LINEAR
            mask_interpolation=0,  # cv2.INTER_NEAREST
            p=0.5,
        ),
        A.ElasticTransform(alpha=1, sigma=50, p=0.2),
        A.Resize(height=512, width=512),
    ])

def get_validation_augmentations():
    """
    Minimal transformations for validation: just resize to 512x512.
    """
    return A.Compose([
        A.Resize(height=512, width=512),
    ])
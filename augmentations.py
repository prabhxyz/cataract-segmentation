import albumentations as A

def get_training_augmentations():
    """
    No augmentation, only a resize to 512x512.
    """
    return A.Compose([
        A.Resize(height=512, width=512),
    ])

def get_validation_augmentations():
    """
    No augmentation, only a resize to 512x512.
    """
    return A.Compose([
        A.Resize(height=512, width=512),
    ])
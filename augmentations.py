import albumentations as A

def get_training_augmentations():
    """
    Returns a simple transformation that resizes the image (and mask)
    to 512x512 without any additional augmentation.
    """
    return A.Compose([
        A.Resize(height=512, width=512),
    ])

def get_validation_augmentations():
    """
    Returns a simple transformation that resizes the image (and mask)
    to 512x512.
    """
    return A.Compose([
        A.Resize(height=512, width=512),
    ])
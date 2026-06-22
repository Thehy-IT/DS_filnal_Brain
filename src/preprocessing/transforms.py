"""
transforms.py — Data augmentation and normalization pipelines.

Three pipeline levels (inspired by DAKHDL's build_augmentation):
  - get_train_transforms()  : Full augmentation for training
  - get_valid_transforms()  : Resize + normalize only (no augmentation)
  - get_heavy_transforms()  : Aggressive augmentation for hard cases

ImageNet normalization stats are used since models were pre-trained on ImageNet.
"""
from torchvision import transforms

from src.config import aug_cfg, data_cfg

# ImageNet statistics (mean / std per channel)
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(image_size: int = None) -> transforms.Compose:
    """
    Training transform pipeline with augmentation.

    Augmentation strategy (translated from DAKHDL's Keras ImageDataGenerator):
      - RandomHorizontalFlip     : mirrors real-world scan orientation variance
      - RandomRotation           : ±20° (DAKHDL: rotation_range=20)
      - ColorJitter              : brightness/contrast/saturation variation
      - RandomAffine             : shear + scale (DAKHDL: shear/zoom_range=0.1)
      - RandomErasing            : randomly erase patches (regularization)
      - Normalize                : ImageNet stats

    Args:
        image_size: Target square size. Defaults to config value (224).

    Returns:
        torchvision.transforms.Compose
    """
    size = image_size or data_cfg.image_size
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomHorizontalFlip(
            p=aug_cfg.random_horizontal_flip
        ),
        transforms.RandomRotation(
            degrees=aug_cfg.random_rotation_degrees
        ),
        transforms.ColorJitter(
            brightness=aug_cfg.color_jitter_brightness,
            contrast=aug_cfg.color_jitter_contrast,
            saturation=aug_cfg.color_jitter_saturation,
        ),
        # Shear + scale mimic DAKHDL's shear_range & zoom_range
        transforms.RandomAffine(
            degrees=aug_cfg.random_affine_degrees,
            shear=aug_cfg.random_affine_shear,
            scale=aug_cfg.random_affine_scale,
            fill=0,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        # Random patch erasing — lightweight regularization
        transforms.RandomErasing(
            p=aug_cfg.random_erasing_p,
            scale=(0.02, 0.1),
            ratio=(0.3, 3.3),
            value=0,
        ),
    ])


def get_valid_transforms(image_size: int = None) -> transforms.Compose:
    """
    Validation / test transform pipeline — NO augmentation.

    Only resizes and normalizes to ensure consistent inference behaviour.

    Args:
        image_size: Target square size. Defaults to config value (224).

    Returns:
        torchvision.transforms.Compose
    """
    size = image_size or data_cfg.image_size
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


def get_heavy_transforms(image_size: int = None) -> transforms.Compose:
    """
    Heavy augmentation pipeline for training on hard / minority classes.

    Adds additional geometric transforms on top of the standard training
    pipeline. Use selectively (e.g., for meningioma which is typically the
    smallest class in the Brain Tumor dataset).

    Args:
        image_size: Target square size. Defaults to config value (224).

    Returns:
        torchvision.transforms.Compose
    """
    size = image_size or data_cfg.image_size
    return transforms.Compose([
        transforms.Resize((size + 32, size + 32)),   # Slightly larger
        transforms.RandomCrop(size),                  # then crop to target
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),         # MRI: vertical flip OK
        transforms.RandomRotation(degrees=30),
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.2,
            hue=0.05,
        ),
        transforms.RandomAffine(
            degrees=0,
            shear=15,
            scale=(0.85, 1.15),
            fill=0,
        ),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
    ])

"""
dataset.py — BrainTumorDataset with stratified splitting and class weights.

Key improvements over v1:
  - Explicit path validation with informative errors
  - Labels exposed as a public list (needed for stratified split & class weights)
  - Class weights computed once and stored on the dataset
  - Static helper split_dataset() uses stratified train/val split (fixes the
    bug where random_split() shared the same transform for both subsets)
"""
import os
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Subset

from src.config import data_cfg, SEED


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class BrainTumorDataset(Dataset):
    """
    PyTorch Dataset for Brain Tumor MRI Images.

    Expects data organized in class sub-folders:
        root_dir/
            glioma/
            meningioma/
            notumor/
            pituitary/

    Attributes:
        root_dir (str):            Resolved absolute path to data root.
        transform (callable):      Transform pipeline applied to each image.
        classes (List[str]):       Ordered list of class names.
        class_to_idx (dict):       Maps class name → integer index.
        image_paths (List[str]):   Absolute paths to every image file.
        labels (List[int]):        Integer label for each image.
        class_weights (Tensor):    Inverse-frequency weights for imbalance.
    """

    VALID_EXTENSIONS = {".png", ".jpg", ".jpeg"}

    def __init__(
        self,
        root_dir: str,
        transform=None,
        class_names: Optional[List[str]] = None,
    ) -> None:
        """
        Args:
            root_dir:     Path to directory containing class sub-folders.
            transform:    Optional torchvision / albumentations transform.
            class_names:  Override default class list (default from config).
        """
        self.root_dir = os.path.abspath(root_dir)
        self.transform = transform
        self.classes: List[str] = class_names or data_cfg.class_names
        self.class_to_idx: dict = {c: i for i, c in enumerate(self.classes)}

        self.image_paths: List[str] = []
        self.labels: List[int] = []

        self._load_index()
        self._compute_class_weights()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_index(self) -> None:
        """Walk root_dir and build image_paths + labels index."""
        if not os.path.isdir(self.root_dir):
            raise FileNotFoundError(
                f"[BrainTumorDataset] Directory not found: {self.root_dir}\n"
                "Please set DATA_DIR in your .env or pass the correct path."
            )

        missing_classes = []
        for cls_name in self.classes:
            cls_dir = os.path.join(self.root_dir, cls_name)
            if not os.path.isdir(cls_dir):
                missing_classes.append(cls_name)
                continue

            found = 0
            for fname in sorted(os.listdir(cls_dir)):
                ext = os.path.splitext(fname)[1].lower()
                if ext in self.VALID_EXTENSIONS:
                    self.image_paths.append(os.path.join(cls_dir, fname))
                    self.labels.append(self.class_to_idx[cls_name])
                    found += 1

            print(f"[Dataset]  {cls_name:>12}: {found:>5} images")

        if missing_classes:
            print(
                f"[Dataset] WARNING — classes not found in '{self.root_dir}': "
                f"{missing_classes}"
            )

        total = len(self.image_paths)
        if total == 0:
            raise RuntimeError(
                f"[BrainTumorDataset] No images found under '{self.root_dir}'. "
                "Check the directory structure."
            )
        print(f"[Dataset] Total images loaded: {total}")

    def _compute_class_weights(self) -> None:
        """Inverse-frequency weights: weight_c = N / (C * count_c)."""
        labels_np = np.array(self.labels)
        total = len(labels_np)
        num_classes = len(self.classes)
        weights = []
        for c in range(num_classes):
            count = int(np.sum(labels_np == c))
            weights.append(total / (num_classes * count) if count > 0 else 0.0)
        self.class_weights = torch.tensor(weights, dtype=torch.float32)

    # ------------------------------------------------------------------
    # PyTorch Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as exc:
            # Return a blank tensor to avoid crashing the DataLoader;
            # a warning is printed so problematic files can be removed.
            print(f"[Dataset] WARNING — could not open image {img_path}: {exc}")
            image = Image.new("RGB", (data_cfg.image_size, data_cfg.image_size))

        if self.transform:
            image = self.transform(image)

        return image, label

    def __repr__(self) -> str:
        return (
            f"BrainTumorDataset(root='{self.root_dir}', "
            f"n={len(self)}, classes={self.classes})"
        )


# ---------------------------------------------------------------------------
# Stratified split helper
# ---------------------------------------------------------------------------

def split_dataset(
    train_dir: str,
    train_transform,
    val_transform,
    val_split: float = None,
    seed: int = SEED,
) -> Tuple[Dataset, Dataset]:
    """
    Create stratified train / validation subsets with SEPARATE transforms.

    This function fixes a critical bug present in the original train.py where
    ``random_split`` was used and the validation transform was accidentally
    applied to the shared underlying dataset, contaminating training samples.

    Strategy:
      1. Build a full index-only dataset (no transform) to get labels.
      2. Perform stratified split on indices using sklearn.
      3. Return two independent Dataset objects — one with train_transform,
         one with val_transform — so transforms are truly isolated.

    Args:
        train_dir:       Path to the training data directory.
        train_transform: Transform pipeline for training (with augmentation).
        val_transform:   Transform pipeline for validation (no augmentation).
        val_split:       Fraction [0, 1] to use for validation. Default from config.
        seed:            Random seed for reproducibility.

    Returns:
        (train_subset, val_subset) — both are valid PyTorch Datasets.
    """
    val_split = val_split if val_split is not None else data_cfg.val_split

    # Step 1: Build index (no transform needed just for splitting)
    index_ds = BrainTumorDataset(root_dir=train_dir, transform=None)
    all_indices = list(range(len(index_ds)))
    all_labels = index_ds.labels

    # Step 2: Stratified split
    train_idx, val_idx = train_test_split(
        all_indices,
        test_size=val_split,
        random_state=seed,
        stratify=all_labels,
    )
    print(
        f"[Dataset] Split → train: {len(train_idx)} | val: {len(val_idx)} "
        f"(stratified, seed={seed})"
    )

    # Step 3: Two independent datasets with separate transforms
    train_ds = BrainTumorDataset(root_dir=train_dir, transform=train_transform)
    val_ds = BrainTumorDataset(root_dir=train_dir, transform=val_transform)

    return Subset(train_ds, train_idx), Subset(val_ds, val_idx)

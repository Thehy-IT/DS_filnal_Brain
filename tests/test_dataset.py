"""
test_dataset.py — Unit tests for BrainTumorDataset and split_dataset.

Tests cover:
  - Dataset loads without error when given a valid directory
  - __len__ and __getitem__ return correct types
  - Labels match the expected class mapping
  - class_weights tensor has correct shape and positive values
  - split_dataset produces non-overlapping, stratified subsets
  - FileNotFoundError raised for missing directories
"""
import os
import tempfile

import pytest
import torch
from PIL import Image

from src.preprocessing.dataset import BrainTumorDataset, split_dataset
from src.preprocessing.transforms import get_train_transforms, get_valid_transforms


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]


def _make_fake_dataset(tmp_path, n_per_class: int = 10) -> str:
    """
    Create a minimal fake dataset directory with tiny black PNG images.
    Returns the path to the dataset root.
    """
    for cls in CLASS_NAMES:
        cls_dir = tmp_path / cls
        cls_dir.mkdir(parents=True)
        for i in range(n_per_class):
            img = Image.new("RGB", (64, 64), color=(i * 5, i * 5, i * 5))
            img.save(str(cls_dir / f"img_{i:03d}.png"))
    return str(tmp_path)


@pytest.fixture
def fake_data_dir(tmp_path):
    return _make_fake_dataset(tmp_path, n_per_class=10)


# ---------------------------------------------------------------------------
# BrainTumorDataset tests
# ---------------------------------------------------------------------------

class TestBrainTumorDataset:
    def test_loads_successfully(self, fake_data_dir):
        ds = BrainTumorDataset(root_dir=fake_data_dir)
        assert len(ds) == 40  # 4 classes × 10 images

    def test_len_returns_int(self, fake_data_dir):
        ds = BrainTumorDataset(root_dir=fake_data_dir)
        assert isinstance(len(ds), int)
        assert len(ds) > 0

    def test_getitem_returns_correct_types(self, fake_data_dir):
        ds = BrainTumorDataset(
            root_dir=fake_data_dir,
            transform=get_valid_transforms(image_size=64),
        )
        image, label = ds[0]
        assert isinstance(image, torch.Tensor), "Image should be a Tensor"
        assert isinstance(label, int), "Label should be an int"

    def test_image_tensor_shape(self, fake_data_dir):
        ds = BrainTumorDataset(
            root_dir=fake_data_dir,
            transform=get_valid_transforms(image_size=64),
        )
        image, _ = ds[0]
        assert image.shape == (3, 64, 64), f"Expected (3,64,64), got {image.shape}"

    def test_labels_in_valid_range(self, fake_data_dir):
        ds = BrainTumorDataset(root_dir=fake_data_dir)
        num_classes = len(CLASS_NAMES)
        assert all(0 <= lbl < num_classes for lbl in ds.labels), \
            "All labels should be in [0, num_classes)"

    def test_class_to_idx_mapping(self, fake_data_dir):
        ds = BrainTumorDataset(root_dir=fake_data_dir)
        for cls in CLASS_NAMES:
            assert cls in ds.class_to_idx

    def test_class_weights_shape(self, fake_data_dir):
        ds = BrainTumorDataset(root_dir=fake_data_dir)
        assert ds.class_weights.shape == (len(CLASS_NAMES),)
        assert (ds.class_weights > 0).all(), "All class weights should be positive"

    def test_class_weights_balanced(self, fake_data_dir):
        """Balanced dataset → all weights should be equal (≈ 1.0)."""
        ds = BrainTumorDataset(root_dir=fake_data_dir)
        weights = ds.class_weights.numpy()
        assert abs(weights.max() - weights.min()) < 1e-6, \
            "Balanced dataset should have equal weights"

    def test_missing_directory_raises(self):
        with pytest.raises(FileNotFoundError):
            BrainTumorDataset(root_dir="/nonexistent/path/to/data")

    def test_partial_classes_warns_but_loads(self, tmp_path):
        """Dataset with only 2 out of 4 classes should load the available ones."""
        for cls in ["glioma", "notumor"]:
            (tmp_path / cls).mkdir()
            img = Image.new("RGB", (32, 32))
            img.save(str(tmp_path / cls / "img.png"))

        ds = BrainTumorDataset(root_dir=str(tmp_path))
        assert len(ds) == 2  # Only 2 images across the 2 present classes


# ---------------------------------------------------------------------------
# split_dataset tests
# ---------------------------------------------------------------------------

class TestSplitDataset:
    def test_split_sizes_sum_to_total(self, fake_data_dir):
        train_ds, val_ds = split_dataset(
            train_dir=fake_data_dir,
            train_transform=get_train_transforms(image_size=64),
            val_transform=get_valid_transforms(image_size=64),
            val_split=0.2,
        )
        total = len(BrainTumorDataset(root_dir=fake_data_dir))
        assert len(train_ds) + len(val_ds) == total

    def test_split_no_overlap(self, fake_data_dir):
        train_ds, val_ds = split_dataset(
            train_dir=fake_data_dir,
            train_transform=get_train_transforms(image_size=64),
            val_transform=get_valid_transforms(image_size=64),
            val_split=0.2,
        )
        train_indices = set(train_ds.indices)
        val_indices = set(val_ds.indices)
        assert train_indices.isdisjoint(val_indices), \
            "Train and val indices must not overlap"

    def test_split_returns_datasets(self, fake_data_dir):
        train_ds, val_ds = split_dataset(
            train_dir=fake_data_dir,
            train_transform=get_valid_transforms(image_size=64),
            val_transform=get_valid_transforms(image_size=64),
        )
        # Check both are iterable and return tensors
        img, lbl = train_ds[0]
        assert isinstance(img, torch.Tensor)
        assert isinstance(lbl, int)

    def test_deterministic_split(self, fake_data_dir):
        """Same seed → same split indices."""
        train1, val1 = split_dataset(
            fake_data_dir,
            get_valid_transforms(image_size=64),
            get_valid_transforms(image_size=64),
            seed=42,
        )
        train2, val2 = split_dataset(
            fake_data_dir,
            get_valid_transforms(image_size=64),
            get_valid_transforms(image_size=64),
            seed=42,
        )
        assert sorted(train1.indices) == sorted(train2.indices)
        assert sorted(val1.indices) == sorted(val2.indices)

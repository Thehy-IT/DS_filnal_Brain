"""
test_transforms.py — Unit tests for transform pipelines.

Verifies output tensor shapes, value ranges, and pipeline composition.
"""
import torch
import pytest
from PIL import Image

from src.preprocessing.transforms import (
    get_train_transforms,
    get_valid_transforms,
    get_heavy_transforms,
)


@pytest.fixture
def sample_image():
    """224×224 white RGB PIL image."""
    return Image.new("RGB", (300, 250), color=(128, 64, 200))


class TestValidTransforms:
    def test_output_shape_default(self, sample_image):
        t = get_valid_transforms()
        tensor = t(sample_image)
        assert tensor.shape == (3, 224, 224)

    def test_output_shape_custom_size(self, sample_image):
        t = get_valid_transforms(image_size=128)
        tensor = t(sample_image)
        assert tensor.shape == (3, 128, 128)

    def test_output_is_tensor(self, sample_image):
        t = get_valid_transforms()
        tensor = t(sample_image)
        assert isinstance(tensor, torch.Tensor)

    def test_output_is_float(self, sample_image):
        t = get_valid_transforms()
        tensor = t(sample_image)
        assert tensor.dtype == torch.float32

    def test_values_are_normalized(self, sample_image):
        """After ImageNet normalization, values should be roughly in [-3, 3]."""
        t = get_valid_transforms()
        tensor = t(sample_image)
        assert tensor.min().item() > -4.0
        assert tensor.max().item() < 4.0


class TestTrainTransforms:
    def test_output_shape(self, sample_image):
        t = get_train_transforms()
        tensor = t(sample_image)
        assert tensor.shape == (3, 224, 224)

    def test_output_is_float_tensor(self, sample_image):
        t = get_train_transforms()
        tensor = t(sample_image)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.dtype == torch.float32

    def test_custom_size(self, sample_image):
        t = get_train_transforms(image_size=64)
        tensor = t(sample_image)
        assert tensor.shape == (3, 64, 64)

    def test_augmentation_produces_variation(self, sample_image):
        """Two runs on the same image should produce different tensors."""
        t = get_train_transforms()
        t1 = t(sample_image)
        t2 = t(sample_image)
        # With stochastic augmentation, tensors should differ at least sometimes
        # We run 5 times and expect at least one difference
        differs = any(not torch.equal(t(sample_image), t1) for _ in range(5))
        assert differs, "Training transforms should produce stochastic variation"


class TestHeavyTransforms:
    def test_output_shape(self, sample_image):
        t = get_heavy_transforms()
        tensor = t(sample_image)
        assert tensor.shape == (3, 224, 224)

    def test_output_is_tensor(self, sample_image):
        t = get_heavy_transforms()
        tensor = t(sample_image)
        assert isinstance(tensor, torch.Tensor)

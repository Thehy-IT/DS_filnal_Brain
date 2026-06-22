"""
test_models.py — Unit tests for model architectures.

Verifies forward pass output shape, parameter counts, and factory function.
"""
import pytest
import torch

from src.training.models import DenseNetModel, EfficientNetModel, get_model


BATCH = 2
NUM_CLASSES = 4
H, W = 224, 224


@pytest.fixture
def dummy_input():
    return torch.randn(BATCH, 3, H, W)


class TestDenseNetModel:
    @pytest.mark.slow
    def test_forward_output_shape(self, dummy_input):
        model = DenseNetModel(num_classes=NUM_CLASSES, pretrained=False)
        model.eval()
        with torch.no_grad():
            out = model(dummy_input)
        assert out.shape == (BATCH, NUM_CLASSES), f"Expected ({BATCH},{NUM_CLASSES}), got {out.shape}"

    @pytest.mark.slow
    def test_no_nan_in_output(self, dummy_input):
        model = DenseNetModel(num_classes=NUM_CLASSES, pretrained=False)
        model.eval()
        with torch.no_grad():
            out = model(dummy_input)
        assert not torch.isnan(out).any(), "Output contains NaN values"

    def test_has_trainable_parameters(self):
        model = DenseNetModel(num_classes=NUM_CLASSES, pretrained=False)
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert params > 0, "Model should have trainable parameters"


class TestEfficientNetModel:
    @pytest.mark.slow  # Mark slow because it may download weights
    def test_forward_output_shape(self, dummy_input):
        model = EfficientNetModel(num_classes=NUM_CLASSES, pretrained=False)
        model.eval()
        with torch.no_grad():
            out = model(dummy_input)
        assert out.shape == (BATCH, NUM_CLASSES)

    def test_no_nan_in_output(self, dummy_input):
        model = EfficientNetModel(num_classes=NUM_CLASSES, pretrained=False)
        model.eval()
        with torch.no_grad():
            out = model(dummy_input)
        assert not torch.isnan(out).any()

    def test_parameter_count_reasonable(self):
        model = EfficientNetModel(num_classes=NUM_CLASSES, pretrained=False)
        params = sum(p.numel() for p in model.parameters())
        # EfficientNetB0 should have roughly 5M parameters
        assert 4_000_000 < params < 7_000_000, \
            f"EfficientNetB0 expected ~5M params, got {params:,}"


class TestGetModelFactory:
    def test_densenet_factory(self, dummy_input):
        model = get_model(model_name="densenet", num_classes=NUM_CLASSES, pretrained=False)
        model.eval()
        with torch.no_grad():
            out = model(dummy_input)
        assert out.shape == (BATCH, NUM_CLASSES)

    def test_efficientnet_factory(self, dummy_input):
        model = get_model(model_name="efficientnet", num_classes=NUM_CLASSES, pretrained=False)
        model.eval()
        with torch.no_grad():
            out = model(dummy_input)
        assert out.shape == (BATCH, NUM_CLASSES)

    def test_invalid_model_name_raises(self):
        with pytest.raises(ValueError, match="not supported"):
            get_model(model_name="resnet", num_classes=4)

    def test_case_insensitive_name(self, dummy_input):
        model = get_model(model_name="DENSENET", num_classes=NUM_CLASSES, pretrained=False)
        model.eval()
        with torch.no_grad():
            out = model(dummy_input)
        assert out.shape == (BATCH, NUM_CLASSES)

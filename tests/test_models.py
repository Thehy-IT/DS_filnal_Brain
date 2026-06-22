import torch
from src.models.unet import initialize_2d_unet_model
from src.models.swin_unetr import initialize_swin_unetr_model
from src.models.classifier import initialize_2d_classifier_model

def test_2d_unet_shape():
    neural_network = initialize_2d_unet_model(input_channels_count=3, target_classes_count=4)
    dummy_input_image = torch.randn(1, 3, 224, 224)
    prediction_logits = neural_network(dummy_input_image)
    assert prediction_logits.shape == (1, 4, 224, 224)

def test_swin_unetr_shape():
    neural_network = initialize_swin_unetr_model(input_channels_count=3, target_classes_count=4, spatial_patch_dimensions=(224, 224))
    dummy_input_image = torch.randn(1, 3, 224, 224)
    prediction_logits = neural_network(dummy_input_image)
    assert prediction_logits.shape == (1, 4, 224, 224)

def test_2d_classifier_shape():
    neural_network = initialize_2d_classifier_model(input_channels_count=3, target_classes_count=4)
    dummy_input_image = torch.randn(1, 3, 224, 224)
    prediction_logits = neural_network(dummy_input_image)
    assert prediction_logits.shape == (1, 4)

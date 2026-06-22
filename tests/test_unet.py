import os
import torch
import numpy as np
from src.inference.predictor import VolumetricPredictor
from src.models.classifier import initialize_2d_classifier_model
from src.models.unet import initialize_2d_unet_model

def test_unet_integration():
    classifier = initialize_2d_classifier_model()
    temp_classifier_path = "data/reference/temp_test_classifier.pth"
    torch.save(classifier.state_dict(), temp_classifier_path)
    
    predictor = VolumetricPredictor(classifier, temp_classifier_path, execution_device="cpu")
    
    assert predictor.unet_model is not None
    
    dummy_slice = np.random.randn(224, 224).astype(np.float32)
    cam, conf, class_idx, unc = predictor.execute_inference_on_slice(dummy_slice, run_uncertainty=True, num_passes=2)
    
    assert cam.shape == (224, 224)
    assert unc.shape == (224, 224)
    
    if os.path.exists(temp_classifier_path):
        os.remove(temp_classifier_path)

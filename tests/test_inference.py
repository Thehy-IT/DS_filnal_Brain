import os
import tempfile
import numpy as np
import pytest
import nibabel as nib
import torch
from src.data.slicer import MedicalVolumeSlicer
from src.inference.predictor import VolumetricPredictor
from src.models.classifier import initialize_2d_classifier_model

def test_medical_volume_slicer_nifti():
    temp_dir = tempfile.mkdtemp()
    nifti_path = os.path.join(temp_dir, "test_volume.nii.gz")
    
    dummy_volume = np.random.randn(30, 40, 50).astype(np.float32)
    affine_matrix = np.eye(4)
    nifti_image = nib.Nifti1Image(dummy_volume, affine_matrix)
    nib.save(nifti_image, nifti_path)
    
    slicer = MedicalVolumeSlicer()
    loaded_volume = slicer.load_3d_volume(nifti_path)
    
    assert loaded_volume.ndim == 3
    
    axial_slice = slicer.extract_slice(loaded_volume, "axial", 5)
    coronal_slice = slicer.extract_slice(loaded_volume, "coronal", 5)
    sagittal_slice = slicer.extract_slice(loaded_volume, "sagittal", 5)
    
    assert axial_slice.ndim == 2
    assert coronal_slice.ndim == 2
    assert sagittal_slice.ndim == 2
    
    os.remove(nifti_path)
    os.rmdir(temp_dir)

def test_volumetric_predictor_on_slice():
    classifier_model = initialize_2d_classifier_model(input_channels_count=3, target_classes_count=4)
    
    temp_model_dir = tempfile.mkdtemp()
    checkpoint_path = os.path.join(temp_model_dir, "temp_model.pth")
    torch.save(classifier_model.state_dict(), checkpoint_path)
    
    predictor = VolumetricPredictor(classifier_model, checkpoint_path, execution_device="cpu")
    
    dummy_slice = np.random.randn(224, 224).astype(np.float32)
    
    cam, conf, class_idx, unc = predictor.execute_inference_on_slice(
        dummy_slice,
        run_uncertainty=True,
        num_passes=3
    )
    
    assert cam.shape == (224, 224)
    assert isinstance(conf, float)
    assert isinstance(class_idx, int)
    assert unc.shape == (224, 224)
    
    os.remove(checkpoint_path)
    os.rmdir(temp_model_dir)

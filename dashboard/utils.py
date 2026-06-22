import os
import tempfile
import numpy as np
import pydicom
from PIL import Image

def persist_uploaded_file(uploaded_file_handle):
    temporary_directory = tempfile.gettempdir()
    temporary_file_path = os.path.join(temporary_directory, uploaded_file_handle.name)
    with open(temporary_file_path, "wb") as destination_file:
        destination_file.write(uploaded_file_handle.getbuffer())
    return temporary_file_path

def extract_nifti_data_matrix(file_path):
    if file_path.lower().endswith(".dcm"):
        dicom_dataset = pydicom.dcmread(file_path)
        pixel_array = dicom_dataset.pixel_array
        normalized_array = ((pixel_array - np.min(pixel_array)) / (np.max(pixel_array) - np.min(pixel_array)) * 255.0).astype(np.uint8)
        return Image.fromarray(normalized_array).convert("RGB")
        
    return Image.open(file_path).convert("RGB")

def apply_morphological_correction(segmentation_mask, correction_type="dilation", structuring_element_iterations=1):
    from scipy.ndimage import binary_dilation, binary_erosion
    if correction_type == "dilation":
        corrected_mask = binary_dilation(segmentation_mask, iterations=structuring_element_iterations)
    elif correction_type == "erosion":
        corrected_mask = binary_erosion(segmentation_mask, iterations=structuring_element_iterations)
    else:
        corrected_mask = segmentation_mask
    return corrected_mask.astype(np.uint8)


import os
import shutil
import tempfile
import numpy as np
import pydicom
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from PIL import Image
from src.inference.predictor import VolumetricPredictor
from src.models.classifier import initialize_2d_classifier_model

app = FastAPI(title="Brain Tumor Clinical Inference API")

model_instance = initialize_2d_classifier_model()
predictor_instance = VolumetricPredictor(
    neural_network=model_instance,
    checkpoint_file_path="data/reference/best_metric_model.pth",
    execution_device="cpu"
)

class InferenceResult(BaseModel):
    estimated_tumor_area_cm2: float
    centroid_x_mm: float
    centroid_y_mm: float
    predicted_class: int
    confidence: float

@app.post("/predict", response_model=InferenceResult)
async def predict_mri_scan(file: UploadFile = File(...)):
    temporary_directory = tempfile.gettempdir()
    temporary_file_path = os.path.join(temporary_directory, file.filename)
    
    with open(temporary_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    if file.filename.lower().endswith(".dcm"):
        dicom_dataset = pydicom.dcmread(temporary_file_path)
        pixel_array = dicom_dataset.pixel_array
        normalized_array = ((pixel_array - np.min(pixel_array)) / (np.max(pixel_array) - np.min(pixel_array)) * 255.0).astype(np.uint8)
        pil_image = Image.fromarray(normalized_array).convert("RGB")
        
        temporary_file_path_jpg = os.path.join(temporary_directory, f"{file.filename}.jpg")
        pil_image.save(temporary_file_path_jpg)
        
        predicted_mask, estimated_area, centroid, class_idx, confidence = predictor_instance.execute_inference(temporary_file_path_jpg)
        os.remove(temporary_file_path_jpg)
    else:
        predicted_mask, estimated_area, centroid, class_idx, confidence = predictor_instance.execute_inference(temporary_file_path)
        
    os.remove(temporary_file_path)
    
    return InferenceResult(
        estimated_tumor_area_cm2=estimated_area,
        centroid_x_mm=centroid[0],
        centroid_y_mm=centroid[1],
        predicted_class=class_idx,
        confidence=confidence
    )

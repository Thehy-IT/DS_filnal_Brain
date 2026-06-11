import io
import base64
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.inference.predict import TumorPredictor
from src.explainability.gradcam import generate_gradcam
from src.preprocessing.transforms import get_valid_transforms

app = FastAPI(title="BrainTumorAI API", description="API for brain tumor classification and explainability.")

# Allow CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model
predictor = None
transform = get_valid_transforms()

@app.on_event("startup")
async def load_model():
    """
    Load the model at startup so it's ready for inference.
    """
    global predictor
    try:
        # Update this path to where your model is actually saved after training
        model_path = "models/vit_best.pth"
        # If model doesn't exist yet, we will just pass or print a warning
        import os
        if os.path.exists(model_path):
            predictor = TumorPredictor(model_path=model_path, model_name='vit')
            print("Model loaded successfully.")
        else:
            print(f"Warning: Model not found at {model_path}. Please train the model first.")
    except Exception as e:
        print(f"Error loading model: {e}")

class PredictionResult(BaseModel):
    class_name: str
    confidence: float
    heatmap_base64: str

@app.get("/")
def read_root():
    return {"message": "Welcome to BrainTumorAI API. Use POST /predict to classify an MRI image."}

@app.post("/predict", response_model=PredictionResult)
async def predict(file: UploadFile = File(...)):
    """
    Predict tumor type and generate Grad-CAM heatmap.
    """
    if predictor is None:
        raise HTTPException(status_code=500, detail="Model is not loaded. Train the model first.")

    # 1. Read Image
    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    # 2. Inference
    try:
        prediction = predictor.predict(image)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    # 3. Generate Grad-CAM
    try:
        # ViT-specific target layer from timm
        target_layer = predictor.model.model.blocks[-1].norm1
        heatmap_img_array = generate_gradcam(
            model=predictor.model,
            image=image,
            transform=transform,
            target_layer=target_layer
        )
        
        # Convert numpy array (0-255 RGB) back to Image to save as Base64
        heatmap_pil = Image.fromarray((heatmap_img_array * 255).astype(np.uint8))
        buffered = io.BytesIO()
        heatmap_pil.save(buffered, format="JPEG")
        heatmap_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
    except Exception as e:
        print(f"Warning: Grad-CAM generation failed: {e}")
        heatmap_b64 = "" # Fallback if Grad-CAM fails

    return {
        "class_name": prediction["class_name"].upper(),
        "confidence": prediction["confidence"],
        "heatmap_base64": heatmap_b64
    }

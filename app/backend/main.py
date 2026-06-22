"""
main.py — FastAPI backend for BrainTumorAI.

Improvements over v1:
  - lifespan context manager (replaces deprecated @app.on_event)
  - GET /health endpoint with model status info
  - GET /classes endpoint listing the 4 tumour classes
  - Full probabilities returned in every prediction response
  - File size + extension validation before inference
  - CORS origins restricted (configurable via ALLOWED_ORIGINS env var)
  - gradcam uses upgraded module (auto target layer + overlay_to_base64)
"""
from __future__ import annotations

import io
import os
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel

from src.config import data_cfg, inference_cfg
from src.explainability.gradcam import generate_gradcam, overlay_to_base64
from src.inference.predict import TumorPredictor
from src.preprocessing.transforms import get_valid_transforms

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

_predictor: Optional[TumorPredictor] = None
_transform = get_valid_transforms()


# ---------------------------------------------------------------------------
# Lifespan — replaces deprecated @app.on_event("startup")
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup; release resources on shutdown."""
    global _predictor
    model_path = inference_cfg.model_path

    if os.path.exists(model_path):
        try:
            _predictor = TumorPredictor(
                model_path=model_path,
                model_name=inference_cfg.model_name,
            )
            print(f"[API] Model loaded ✓  ({model_path})")
        except Exception as exc:
            print(f"[API] ERROR — could not load model: {exc}")
    else:
        print(
            f"[API] WARNING — model not found at '{model_path}'. "
            "Train the model first, then restart the server."
        )

    yield  # <-- app is live here

    # Shutdown: free GPU memory
    if _predictor is not None:
        del _predictor
        print("[API] Model released.")


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="BrainTumorAI API",
    description=(
        "REST API for brain tumour classification (4 classes) "
        "and Grad-CAM explainability using EfficientNetB0."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

# CORS — restrict to known origins in production
_allowed_origins: List[str] = os.environ.get(
    "ALLOWED_ORIGINS", "http://localhost:8501,http://localhost:3000"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class PredictionResult(BaseModel):
    class_name: str
    confidence: float
    probabilities: Dict[str, float]   # {class_name: probability}
    heatmap_base64: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: str
    classes: List[str]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _validate_upload(file: UploadFile, content: bytes) -> None:
    """Validate file extension and size; raise HTTPException on failure."""
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in inference_cfg.allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unsupported file type '{ext}'. "
                f"Allowed: {inference_cfg.allowed_extensions}"
            ),
        )
    max_bytes = inference_cfg.max_file_size_mb * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Maximum allowed size: {inference_cfg.max_file_size_mb} MB.",
        )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", tags=["root"])
def read_root():
    return {
        "message": "BrainTumorAI API v2.0 — POST /predict to classify an MRI image.",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["monitoring"])
def health_check():
    """
    Return server health and model status.
    Use this to verify the model is loaded before sending predictions.
    """
    return HealthResponse(
        status="ok",
        model_loaded=_predictor is not None,
        model_path=inference_cfg.model_path,
        classes=data_cfg.class_names,
    )


@app.get("/classes", tags=["info"])
def list_classes() -> Dict[str, List[str]]:
    """Return the list of tumour classes the model can predict."""
    return {"classes": data_cfg.class_names}


@app.post("/predict", response_model=PredictionResult, tags=["inference"])
async def predict(file: UploadFile = File(...)):
    """
    Classify an uploaded MRI image and return Grad-CAM explainability.

    - **file**: PNG / JPG / JPEG MRI scan (max 10 MB).

    Returns:
    - **class_name**: Predicted tumour type (uppercase).
    - **confidence**: Highest class probability.
    - **probabilities**: Full softmax probability for all 4 classes.
    - **heatmap_base64**: JPEG Grad-CAM overlay encoded as Base64.
    """
    if _predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded. Please train the model first and restart the server.",
        )

    # 1. Read and validate upload
    content = await file.read()
    _validate_upload(file, content)

    try:
        image = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot decode image. Please upload a valid PNG or JPEG file.",
        )

    # 2. Run inference
    try:
        prediction = _predictor.predict(image)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {exc}",
        )

    # 3. Build probability dict {class_name: prob}
    probs_dict = {
        cls: round(float(p), 6)
        for cls, p in zip(data_cfg.class_names, prediction["probabilities"])
    }

    # 4. Generate Grad-CAM
    heatmap_b64 = ""
    try:
        _, overlay_pil = generate_gradcam(
            model=_predictor.model,
            image=image,
            transform=_transform,
            target_layer=None,   # auto-detect
        )
        heatmap_b64 = overlay_to_base64(overlay_pil)
    except Exception as exc:
        # Grad-CAM is non-critical — log and continue
        print(f"[API] WARNING — Grad-CAM generation failed: {exc}")

    return PredictionResult(
        class_name=prediction["class_name"].upper(),
        confidence=round(prediction["confidence"], 6),
        probabilities=probs_dict,
        heatmap_base64=heatmap_b64,
    )

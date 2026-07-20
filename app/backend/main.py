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

[v2 — Giai đoạn 3: Test-Time Augmentation]
  - /predict endpoint giờ dùng predict_with_tta() thay vì predict()
  - Response thêm trường tta_method và tta_n để frontend hiển thị
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
# Global state — model và transform được load một lần khi khởi động server
# ---------------------------------------------------------------------------

# _predictor: None khi chưa load model; khởi tạo trong lifespan
_predictor: Optional[TumorPredictor] = None
# Transform dùng chung cho mọi request (thread-safe vì chỉ đọc)
_transform = get_valid_transforms()


# ---------------------------------------------------------------------------
# Lifespan — thay thế @app.on_event("startup") đã bị deprecated từ FastAPI 0.93+
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Quản lý vòng đời ứng dụng: load model khi start, giải phóng khi shutdown.
    Dùng asynccontextmanager để FastAPI gọi đúng lúc.
    """
    global _predictor
    model_path = inference_cfg.model_path

    if os.path.exists(model_path):
        try:
            # Load model vào RAM / VRAM một lần duy nhất khi server khởi động
            _predictor = TumorPredictor(
                model_path=model_path,
                model_name=inference_cfg.model_name,
            )
            print(f"[API] Model loaded successfully  ({model_path})")
        except Exception as exc:
            print(f"[API] ERROR - could not load model: {exc}")
    else:
        print(
            f"[API] WARNING - model not found at '{model_path}'. "
            "Train the model first, then restart the server."
        )

    yield  # Server đang chạy tại đây — xử lý các request

    # Shutdown: xóa model khỏi RAM / giải phóng VRAM
    if _predictor is not None:
        del _predictor
        print("[API] Model released.")


# ---------------------------------------------------------------------------
# App setup — khởi tạo FastAPI với metadata và lifespan
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

# CORS — chỉ cho phép các origin đã biết kết nối (bảo mật production)
# Cấu hình qua biến môi trường ALLOWED_ORIGINS (phân cách bằng dấu phẩy)
_allowed_origins: List[str] = os.environ.get(
    "ALLOWED_ORIGINS",
    "http://localhost:8501,http://localhost:3000,"
    "http://192.168.1.8:3000,http://192.168.1.8:8501",
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,  # Chỉ cho phép các origin đã liệt kê
    allow_credentials=True,
    allow_methods=["GET", "POST"],   # Chỉ cho phép GET và POST (không PUT/DELETE)
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schemas — định nghĩa cấu trúc request/response dùng Pydantic
# ---------------------------------------------------------------------------

class PredictionResult(BaseModel):
    """Schema response của endpoint /predict."""
    class_name: str                      # Tên lớp dự đoán (viết HOA)
    confidence: float                    # Điểm tin cậy cao nhất (0.0-1.0)
    probabilities: Dict[str, float]      # Xác suất softmax cho cả 4 lớp
    heatmap_base64: str                  # Ảnh Grad-CAM encode Base64
    # [v2 — Giai đoạn 3] Thông tin TTA để frontend hiển thị
    tta_method: str = "single"           # 'single' hoặc 'tta_5'
    tta_n: int = 1                       # Số lần forward pass thực tế


class HealthResponse(BaseModel):
    """Schema response của endpoint /health."""
    status: str          # "ok" hoặc "error"
    model_loaded: bool   # True nếu model đã được load thành công
    model_path: str      # Đường dẫn file model đang dùng
    classes: List[str]   # Danh sách 4 lớp u não


# ---------------------------------------------------------------------------
# Helpers — hàm nội bộ dùng trong các endpoint
# ---------------------------------------------------------------------------

def _validate_upload(file: UploadFile, content: bytes) -> None:
    """
    Kiểm tra định dạng file và kích thước; ném HTTPException nếu không hợp lệ.
    Đây là lớp bảo vệ đầu tiên trước khi xử lý ảnh.
    """
    # Kiểm tra phần mở rộng file
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in inference_cfg.allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unsupported file type '{ext}'. "
                f"Allowed: {inference_cfg.allowed_extensions}"
            ),
        )
    # Kiểm tra kích thước file (tránh upload file quá lớn làm tràn RAM)
    max_bytes = inference_cfg.max_file_size_mb * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"File too large. Maximum allowed size: {inference_cfg.max_file_size_mb} MB.",
        )


# ---------------------------------------------------------------------------
# Endpoints — các route của REST API
# ---------------------------------------------------------------------------

@app.get("/", tags=["root"])
def read_root():
    """Endpoint gốc — trả về thông tin cơ bản và link tài liệu."""
    return {
        "message": "BrainTumorAI API v2.0 - POST /predict to classify an MRI image.",
        "docs": "/docs",      # Swagger UI tự động của FastAPI
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["monitoring"])
def health_check():
    """
    Kiểm tra trạng thái server và model.
    Frontend gọi endpoint này trước khi cho phép người dùng upload ảnh.
    """
    return HealthResponse(
        status="ok",
        model_loaded=_predictor is not None,  # False nếu model chưa load
        model_path=inference_cfg.model_path,
        classes=data_cfg.class_names,
    )


@app.get("/classes", tags=["info"])
def list_classes() -> Dict[str, List[str]]:
    """Trả về danh sách 4 loại u não mà model có thể phân loại."""
    return {"classes": data_cfg.class_names}


@app.post("/predict", response_model=PredictionResult, tags=["inference"])
async def predict(file: UploadFile = File(...)):
    """
    Phân loại ảnh MRI và trả về kết quả kèm Grad-CAM explainability.

    - **file**: Ảnh PNG / JPG / JPEG của ảnh MRI (tối đa 10 MB).

    Returns:
    - **class_name**: Loại u não dự đoán (CHU HOA).
    - **confidence**: Xác suất cao nhất.
    - **probabilities**: Xác suất softmax đầy đủ cho 4 lớp.
    - **heatmap_base64**: Ảnh Grad-CAM overlay encode Base64.
    """
    # Nếu model chưa load (chưa train hoặc lỗi khi start), từ chối request
    if _predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded. Please train the model first and restart the server.",
        )

    # Bước 1: Đọc và kiểm tra file upload
    content = await file.read()        # Đọc toàn bộ nội dung file vào RAM
    _validate_upload(file, content)    # Kiểm tra định dạng và kích thước

    try:
        # Giải mã bytes thành PIL Image, convert sang RGB để đảm bảo 3 kênh
        image = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot decode image. Please upload a valid PNG or JPEG file.",
        )

    # Bước 2: Chạy inference — model dự đoán loại u não
    # [v2] Dùng predict_with_tta() thay vì predict() để tăng độ chính xác
    # với ảnh từ nguồn ngoài domain (ảnh bệnh viện khác, ảnh internet)
    try:
        prediction = _predictor.predict_with_tta(image)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {exc}",
        )

    # Bước 3: Chuyển danh sách xác suất thành dict {tên_lớp: xác_suất}
    probs_dict = {
        cls: round(float(p), 6)
        for cls, p in zip(data_cfg.class_names, prediction["probabilities"])
    }

    # Bước 4: Tạo Grad-CAM heatmap — non-critical, lỗi không làm fail cả request
    heatmap_b64 = ""
    try:
        _, overlay_pil = generate_gradcam(
            model=_predictor.model,
            image=image,
            transform=_transform,
            target_layer=None,   # Tự động phát hiện layer tốt nhất
        )
        # Encode PIL Image sang Base64 string để trả về trong JSON
        heatmap_b64 = overlay_to_base64(overlay_pil)
    except Exception as exc:
        # Grad-CAM thất bại: vẫn trả về kết quả classification, heatmap để trống
        print(f"[API] WARNING - Grad-CAM generation failed: {exc}")

    # Trả về kết quả hoàn chỉnh
    return PredictionResult(
        class_name=prediction["class_name"].upper(),   # Viết hoa tên lớp
        confidence=round(prediction["confidence"], 6),
        probabilities=probs_dict,
        heatmap_base64=heatmap_b64,
        # [v2 — Giai đoạn 3] Thêm thông tin TTA vào response
        tta_method=prediction.get("method", "single"),
        tta_n=prediction.get("tta_n", 1),
    )

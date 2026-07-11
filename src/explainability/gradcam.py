"""
gradcam.py — Grad-CAM explainability for BrainTumorAI.

Improvements over v1 (inspired by DAKHDL's Grad-CAM visualisation):
  - Auto-detect target layer based on model type (no manual specification needed)
  - Alpha blending control for heatmap overlay
  - visualize_gradcam_grid() — show a grid of N samples with their heatmaps
  - Confidence score annotation on each overlay
  - Returns both numpy array AND PIL Image for flexible downstream use
"""
from __future__ import annotations

import io
import textwrap
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont

from src.config import data_cfg


# ---------------------------------------------------------------------------
# _get_target_layer — tự động chọn layer tốt nhất để hook Grad-CAM
# ---------------------------------------------------------------------------

def _get_target_layer(model: nn.Module) -> nn.Module:
    """
    Tự động chọn layer Grad-CAM tối ưu cho các kiến trúc được hỗ trợ.

    Nguyên tắc: chọn layer tích chập cuối cùng trước pooling vì nó
    chứa đặc trưng không gian (spatial features) phong phú nhất.

    - EfficientNetB0 (timm): conv_head — lớp 1x1 cuối trước pooling.
    - DenseNet121 (timm): features.norm5 — normalization cuối của dense block.

    Args:
        model: Instance của EfficientNetModel hoặc DenseNetModel.

    Returns:
        nn.Module — layer mục tiêu để đặt hook.

    Raises:
        ValueError: Nếu kiến trúc không được nhận dạng.
    """
    # EfficientNetModel: model.model.conv_head là lớp conv cuối trước global avg pool
    if hasattr(model, "model") and hasattr(model.model, "conv_head"):
        return model.model.conv_head

    # DenseNetModel: model.model.features.norm5 là lớp batch norm cuối
    if hasattr(model, "model") and hasattr(model.model, "features"):
        if hasattr(model.model.features, "norm5"):
            return model.model.features.norm5
        return model.model.features

    raise ValueError(
        "Cannot auto-detect Grad-CAM target layer for this model. "
        "Please pass `target_layer` explicitly to generate_gradcam()."
    )


# ---------------------------------------------------------------------------
# generate_gradcam — tạo heatmap Grad-CAM cho một ảnh MRI đơn lẻ
# ---------------------------------------------------------------------------

def generate_gradcam(
    model: nn.Module,
    image: Image.Image,
    transform,
    target_layer: Optional[nn.Module] = None,
    target_class: Optional[int] = None,
    alpha: float = 0.5,
    image_size: int = None,
) -> Tuple[np.ndarray, Image.Image]:
    """
    Tạo heatmap Grad-CAM cho một ảnh MRI đơn lẻ.

    Grad-CAM hoạt động bằng cách:
    1. Cho ảnh chạy qua model (forward pass).
    2. Tính gradient của lớp đầu ra so với feature map của layer mục tiêu.
    3. Lấy trung bình gradient theo spatial dimension = trọng số tầm quan trọng.
    4. Nhân trọng số với feature map, lấy ReLU, resize về kích thước gốc.
    5. Overlay heatmap lên ảnh gốc.

    Args:
        model:        Model PyTorch đã load, ở eval mode.
        image:        Ảnh PIL gốc (kích thước / mode bất kỳ).
        transform:    Pipeline validation transform (resize + normalize).
        target_layer: Layer để đặt hook. Tự động phát hiện nếu None.
        target_class: Index lớp cần giải thích. Nếu None, dùng lớp được dự đoán.
        alpha:        Độ mờ của heatmap overlay [0=trong suốt, 1=heatmap thuần].
        image_size:   Kích thước ảnh đầu ra. Mặc định từ config (224).

    Returns:
        (overlay_np, overlay_pil):
          - overlay_np:  float32 numpy array [0, 1], shape (H, W, 3)
          - overlay_pil: PIL Image RGB của overlay
    """
    # Import lazy để tránh load pytorch_grad_cam khi không cần thiết
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    size = image_size or data_cfg.image_size

    # Bước 1: Xác định layer target (tự động hoặc từ tham số)
    if target_layer is None:
        target_layer = _get_target_layer(model)

    # Bước 2: Chuẩn bị tensor đầu vào — thêm batch dimension (1, C, H, W)
    input_tensor = transform(image).unsqueeze(0)

    # Bước 3: Khởi tạo GradCAM với layer target
    cam = GradCAM(model=model, target_layers=[target_layer])
    # Nếu không chỉ định lớp mục tiêu, GradCAM tự dùng lớp được dự đoán
    targets = [ClassifierOutputTarget(target_class)] if target_class is not None else None

    # Bước 4: Tính class activation map — shape (H, W) với giá trị [0, 1]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

    # Bước 5: Chuẩn bị ảnh RGB gốc, chuẩn hóa về [0, 1]
    rgb_img = np.array(image.convert("RGB").resize((size, size)), dtype=np.float32) / 255.0

    # Bước 6: Overlay heatmap lên ảnh gốc (alpha blending)
    overlay_np = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True, image_weight=1 - alpha)

    # Chuẩn hóa về [0, 1] nếu hàm trả về uint8 [0, 255]
    if overlay_np.dtype == np.uint8:
        overlay_np = overlay_np.astype(np.float32) / 255.0

    # Chuyển về PIL Image để dễ hiển thị / encode
    overlay_pil = Image.fromarray((overlay_np * 255).astype(np.uint8))

    return overlay_np, overlay_pil


# ---------------------------------------------------------------------------
# visualize_gradcam_grid — hiển thị lưới Grad-CAM cho nhiều ảnh cùng lúc
# ---------------------------------------------------------------------------

def visualize_gradcam_grid(
    model: nn.Module,
    images: List[Image.Image],
    transform,
    labels: Optional[List[int]] = None,
    pred_labels: Optional[List[int]] = None,
    confidences: Optional[List[float]] = None,
    class_names: Optional[List[str]] = None,
    target_layer: Optional[nn.Module] = None,
    cols: int = 4,
    save_path: Optional[str] = None,
) -> None:
    """
    Hiển thị lưới Grad-CAM overlay cho danh sách ảnh.

    Mỗi ô trong lưới gồm: [Ảnh gốc | Grad-CAM overlay], có ghi nhãn thật/dự đoán
    và điểm confidence phía trên.

    Args:
        model:        Model đã train.
        images:       Danh sách PIL Images.
        transform:    Validation transform.
        labels:       Index lớp thực tế (tùy chọn).
        pred_labels:  Index lớp dự đoán (tùy chọn).
        confidences:  Điểm confidence 0-1 (tùy chọn).
        class_names:  Danh sách tên lớp.
        target_layer: Layer Grad-CAM (tự động nếu None).
        cols:         Số cột trong lưới.
        save_path:    Nếu có, lưu hình ra file PNG.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    class_names = class_names or data_cfg.class_names
    n = len(images)
    # Tính số hàng cần thiết (làm tròn lên)
    rows = (n + cols - 1) // cols

    # Mỗi ảnh cần 2 cột (gốc + Grad-CAM), nhân đôi số cột
    fig, axes = plt.subplots(rows, cols * 2, figsize=(cols * 5, rows * 3.5))
    fig.suptitle("Grad-CAM Activation Maps — BrainTumorAI", fontsize=13, fontweight="bold")

    # Flatten axes về 2D để index dễ dàng
    axes = np.array(axes).reshape(rows, cols * 2)

    for idx, image in enumerate(images):
        row, col = divmod(idx, cols)
        ax_orig = axes[row, col * 2]      # Cột lẻ: ảnh gốc
        ax_cam = axes[row, col * 2 + 1]   # Cột chẵn: Grad-CAM

        # Tạo Grad-CAM overlay cho ảnh này
        try:
            _, overlay_pil = generate_gradcam(
                model, image, transform, target_layer=target_layer
            )
        except Exception as exc:
            # Nếu Grad-CAM thất bại, dùng ảnh gốc thay thế (không crash)
            print(f"[GradCAM] WARNING - failed for image {idx}: {exc}")
            overlay_pil = image

        # Tạo tiêu đề ô ảnh: True/Pred label + confidence
        parts = []
        if labels is not None:
            parts.append(f"True: {class_names[labels[idx]]}")
        if pred_labels is not None:
            # Màu xanh nếu đúng, đỏ nếu sai
            color = "green" if (
                labels is not None and pred_labels[idx] == labels[idx]
            ) else "red"
            parts.append(f"Pred: {class_names[pred_labels[idx]]}")
        if confidences is not None:
            parts.append(f"Conf: {confidences[idx]:.1%}")

        title = "\n".join(parts)

        # Hiển thị ảnh gốc (resize về 224x224)
        ax_orig.imshow(image.convert("RGB").resize((224, 224)))
        ax_orig.set_title(title, fontsize=7, color="black")
        ax_orig.axis("off")

        # Hiển thị Grad-CAM overlay
        ax_cam.imshow(overlay_pil)
        ax_cam.set_title("Grad-CAM", fontsize=7, color="gray")
        ax_cam.axis("off")

    # Ẩn các ô thừa (khi số ảnh không chia hết cho cols)
    for idx in range(n, rows * cols):
        row, col = divmod(idx, cols)
        axes[row, col * 2].axis("off")
        axes[row, col * 2 + 1].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[GradCAM] Grid saved -> {save_path}")

    plt.close(fig)


# ---------------------------------------------------------------------------
# overlay_to_base64 — encode ảnh overlay sang Base64 để trả về qua API
# ---------------------------------------------------------------------------

def overlay_to_base64(overlay_pil: Image.Image, fmt: str = "JPEG") -> str:
    """
    Encode ảnh PIL overlay sang chuỗi Base64 để nhúng vào JSON response.

    Cách dùng: Frontend nhận chuỗi này và hiển thị bằng
    <img src="data:image/jpeg;base64,{string}" />.

    Args:
        overlay_pil: Ảnh PIL cần encode.
        fmt:         Định dạng nén ('JPEG' nhỏ hơn, 'PNG' không mất chất lượng).

    Returns:
        Chuỗi Base64 (không có tiền tố data URI).
    """
    import base64

    # Lưu ảnh vào buffer byte trong RAM (không ghi ra đĩa)
    buf = io.BytesIO()
    overlay_pil.save(buf, format=fmt)
    # Encode bytes sang Base64 string
    return base64.b64encode(buf.getvalue()).decode("utf-8")

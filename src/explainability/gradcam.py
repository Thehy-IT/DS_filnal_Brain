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
# Target layer resolver
# ---------------------------------------------------------------------------

def _get_target_layer(model: nn.Module) -> nn.Module:
    """
    Automatically select the best Grad-CAM target layer for supported models.

    For EfficientNetB0 (timm): last convolutional block before the pooling.
    For BaselineCNN: last Conv2d in the feature extractor.

    Args:
        model: The loaded model (EfficientNetModel or BaselineCNN instance).

    Returns:
        nn.Module — the target layer.

    Raises:
        ValueError: If the model architecture is not recognised.
    """
    # EfficientNetModel wraps timm's efficientnet_b0 inside self.model
    if hasattr(model, "model") and hasattr(model.model, "conv_head"):
        # timm EfficientNet: conv_head is the final 1×1 conv before pooling
        return model.model.conv_head

    # BaselineCNN: last Conv2d in self.features
    if hasattr(model, "features"):
        conv_layers = [
            m for m in model.features.modules() if isinstance(m, nn.Conv2d)
        ]
        if conv_layers:
            return conv_layers[-1]

    raise ValueError(
        "Cannot auto-detect Grad-CAM target layer for this model. "
        "Please pass `target_layer` explicitly to generate_gradcam()."
    )


# ---------------------------------------------------------------------------
# Core Grad-CAM function
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
    Generate a Grad-CAM heatmap overlay for a single MRI image.

    Args:
        model:        Loaded PyTorch model in eval mode.
        image:        Original PIL Image (any size / mode).
        transform:    Validation transform pipeline (resize + normalize).
        target_layer: Convolutional layer to hook. Auto-detected if None.
        target_class: Class index to explain. If None, uses predicted class.
        alpha:        Opacity of CAM overlay on the original image [0, 1].
        image_size:   Output size for the overlay. Defaults to config (224).

    Returns:
        (overlay_np, overlay_pil):
          - overlay_np:  float32 numpy array, values in [0, 1], shape (H, W, 3)
          - overlay_pil: PIL Image (RGB) of the same overlay
    """
    # Lazy import to avoid loading matplotlib globally via pytorch_grad_cam
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    size = image_size or data_cfg.image_size

    # 1. Resolve target layer
    if target_layer is None:
        target_layer = _get_target_layer(model)

    # 2. Prepare input tensor
    input_tensor = transform(image).unsqueeze(0)  # shape (1, C, H, W)

    # 3. Build GradCAM
    cam = GradCAM(model=model, target_layers=[target_layer])
    targets = [ClassifierOutputTarget(target_class)] if target_class is not None else None

    # 4. Compute class activation map
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]  # (H, W)

    # 5. Prepare RGB base image normalised to [0, 1]
    rgb_img = np.array(image.convert("RGB").resize((size, size)), dtype=np.float32) / 255.0

    # 6. Overlay heatmap
    overlay_np = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True, image_weight=1 - alpha)

    # overlay_np from show_cam_on_image is uint8 [0, 255]; normalize to [0, 1]
    if overlay_np.dtype == np.uint8:
        overlay_np = overlay_np.astype(np.float32) / 255.0

    overlay_pil = Image.fromarray((overlay_np * 255).astype(np.uint8))

    return overlay_np, overlay_pil


# ---------------------------------------------------------------------------
# Grid visualisation (from DAKHDL's multi-sample Grad-CAM display)
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
    Visualise Grad-CAM overlays for a list of images in a grid.

    Inspired by DAKHDL's approach of displaying multiple MRI samples with
    their corresponding activation maps and annotations.

    Each cell shows: [Original | Grad-CAM] stacked, with true/pred label
    and confidence score annotated above.

    Args:
        model:        Trained model.
        images:       List of PIL Images.
        transform:    Validation transform.
        labels:       Ground-truth class indices (optional).
        pred_labels:  Predicted class indices (optional).
        confidences:  Prediction confidence scores 0–1 (optional).
        class_names:  List of class name strings.
        target_layer: CAM target layer (auto-detected if None).
        cols:         Number of columns in the grid.
        save_path:    If given, saves the figure as PNG.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    class_names = class_names or data_cfg.class_names
    n = len(images)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols * 2, figsize=(cols * 5, rows * 3.5))
    fig.suptitle("Grad-CAM Activation Maps — BrainTumorAI", fontsize=13, fontweight="bold")

    # Flatten axes for easy indexing
    axes = np.array(axes).reshape(rows, cols * 2)

    for idx, image in enumerate(images):
        row, col = divmod(idx, cols)
        ax_orig = axes[row, col * 2]
        ax_cam = axes[row, col * 2 + 1]

        # Generate Grad-CAM
        try:
            _, overlay_pil = generate_gradcam(
                model, image, transform, target_layer=target_layer
            )
        except Exception as exc:
            print(f"[GradCAM] WARNING — failed for image {idx}: {exc}")
            overlay_pil = image

        # Build title text
        parts = []
        if labels is not None:
            parts.append(f"True: {class_names[labels[idx]]}")
        if pred_labels is not None:
            color = "green" if (
                labels is not None and pred_labels[idx] == labels[idx]
            ) else "red"
            parts.append(f"Pred: {class_names[pred_labels[idx]]}")
        if confidences is not None:
            parts.append(f"Conf: {confidences[idx]:.1%}")

        title = "\n".join(parts)

        ax_orig.imshow(image.convert("RGB").resize((224, 224)))
        ax_orig.set_title(title, fontsize=7, color="black")
        ax_orig.axis("off")

        ax_cam.imshow(overlay_pil)
        ax_cam.set_title("Grad-CAM", fontsize=7, color="gray")
        ax_cam.axis("off")

    # Hide unused axes
    for idx in range(n, rows * cols):
        row, col = divmod(idx, cols)
        axes[row, col * 2].axis("off")
        axes[row, col * 2 + 1].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[GradCAM] Grid saved → {save_path}")

    plt.close(fig)


# ---------------------------------------------------------------------------
# Encode overlay to Base64 (used by the FastAPI backend)
# ---------------------------------------------------------------------------

def overlay_to_base64(overlay_pil: Image.Image, fmt: str = "JPEG") -> str:
    """
    Encode a PIL overlay image to a Base64 string for API responses.

    Args:
        overlay_pil: PIL Image to encode.
        fmt:         Image format ('JPEG' or 'PNG').

    Returns:
        Base64-encoded string (no data URI prefix).
    """
    import base64

    buf = io.BytesIO()
    overlay_pil.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

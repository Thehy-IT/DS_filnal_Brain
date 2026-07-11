"""
utils.py — Shared utility functions for BrainTumorAI.

Includes:
- set_seed()        : Full reproducibility across random / numpy / torch / CUDA
- find_data_dir()   : Auto-detect dataset path (local / Kaggle / Colab)
- compute_class_weights() : Calculate inverse-frequency class weights
- save_history()    : Persist training history to JSON
"""
import os
import json
import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Reproducibility — đảm bảo kết quả lặp lại hoàn toàn mỗi lần chạy
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """
    Cố định random seed trên tất cả các thư viện:
      Python random, NumPy, PyTorch (CPU + CUDA), cuDNN.

    Args:
        seed: Giá trị seed nguyên. Mặc định 42.
    """
    # Cố định seed cho Python built-in random
    random.seed(seed)
    # Cố định seed cho NumPy
    np.random.seed(seed)
    # Cố định seed cho PyTorch CPU
    torch.manual_seed(seed)
    # Cố định seed cho GPU chính (index 0)
    torch.cuda.manual_seed(seed)
    # Cố định seed cho tất cả GPU (trường hợp multi-GPU)
    torch.cuda.manual_seed_all(seed)
    # Buộc cuDNN dùng thuật toán tất định (chậm hơn một chút nhưng kết quả nhất quán)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Cố định hash seed cho Python (ảnh hưởng dict, set)
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"[utils] Seed set to {seed}")


# ---------------------------------------------------------------------------
# Dataset discovery — tự động tìm thư mục dữ liệu trên nhiều môi trường
# ---------------------------------------------------------------------------

def find_data_dir(
    candidates: Optional[List[str]] = None,
    subfolder: str = "Training"
) -> str:
    """
    Tự động dò thư mục gốc của dataset từ danh sách đường dẫn ứng viên.
    Hữu ích khi chạy trên Kaggle, Colab, hoặc máy cục bộ.

    Args:
        candidates: Danh sách đường dẫn cần kiểm tra (theo thứ tự ưu tiên).
                    Mặc định kiểm tra các đường phổ biến local / Kaggle / Colab.
        subfolder:  Thư mục con kỳ vọng (vd: 'Training') để xác nhận đúng root.

    Returns:
        Đường dẫn tuyệt đối đến thư mục chứa `subfolder`.

    Raises:
        FileNotFoundError: Nếu không tìm thấy trong bất kỳ ứng viên nào.
    """
    if candidates is None:
        # Danh sách đường dẫn mặc định — ưu tiên theo thứ tự: local > Kaggle > Colab
        candidates = [
            # Local project layout
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data")),
            # Kaggle input
            "/kaggle/input/brain-tumor-mri-dataset",
            "/kaggle/input/brain-tumor-classification-mri",
            # Colab / mounted Drive
            "/content/data",
            "/content/drive/MyDrive/data",
        ]

    # Duyệt từng ứng viên, trả về ngay khi tìm thấy thư mục con hợp lệ
    for path in candidates:
        full = os.path.join(path, subfolder)
        if os.path.isdir(full):
            print(f"[utils] Dataset found at: {path}")
            return path

    raise FileNotFoundError(
        f"Dataset with subfolder '{subfolder}' not found in any of:\n"
        + "\n".join(f"  {c}" for c in candidates)
    )


# ---------------------------------------------------------------------------
# Class weights — xử lý mất cân bằng dữ liệu (imbalanced dataset)
# ---------------------------------------------------------------------------

def compute_class_weights(labels: List[int], num_classes: int) -> torch.Tensor:
    """
    Tính trọng số nghịch tần số để bù đắp mất cân bằng dữ liệu.

    Công thức: weight_c = total_samples / (num_classes * count_c)
    Lớp có ít ảnh hơn sẽ được trọng số cao hơn, giúp loss function
    chú ý nhiều hơn đến các lớp thiểu số.

    Args:
        labels:      Danh sách nhãn nguyên của tập training.
        num_classes: Tổng số lớp phân loại.

    Returns:
        torch.Tensor shape (num_classes,) với trọng số float.
    """
    labels_np = np.array(labels)
    class_weights = []
    total = len(labels_np)

    for c in range(num_classes):
        count = np.sum(labels_np == c)
        # Tránh chia cho 0 — lớp không có mẫu nào thì trọng số = 0
        if count == 0:
            weight = 0.0
        else:
            weight = total / (num_classes * count)
        class_weights.append(weight)

    weights = torch.tensor(class_weights, dtype=torch.float32)
    print(f"[utils] Class weights: { {i: f'{w:.4f}' for i, w in enumerate(class_weights)} }")
    return weights


# ---------------------------------------------------------------------------
# Training history persistence — lưu / tải lịch sử training dạng JSON
# ---------------------------------------------------------------------------

def save_history(history: Dict, save_path: str) -> None:
    """
    Lưu dictionary lịch sử training ra file JSON.

    Args:
        history:   Dict với các key như 'train_loss', 'val_loss', 'val_acc', ...
                   Mỗi value là danh sách float theo từng epoch.
        save_path: Đường dẫn đầy đủ đến file .json đầu ra.
    """
    # Tự tạo thư mục cha nếu chưa tồn tại
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    print(f"[utils] Training history saved -> {save_path}")


def load_history(save_path: str) -> Dict:
    """
    Tải lịch sử training đã lưu từ file JSON.

    Args:
        save_path: Đường dẫn đến file .json.

    Returns:
        Dictionary chứa các metrics theo từng epoch.
    """
    with open(save_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Device helper — chọn thiết bị tính toán tốt nhất có sẵn
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """
    Trả về thiết bị tốt nhất theo thứ tự: CUDA > MPS (Apple Silicon) > CPU.

    Returns:
        torch.device
    """
    if torch.cuda.is_available():
        # NVIDIA GPU — ưu tiên cao nhất
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # Apple M1/M2 GPU — sử dụng Metal Performance Shaders
        device = torch.device("mps")
    else:
        # Fallback: CPU — chậm nhưng luôn khả dụng
        device = torch.device("cpu")
    print(f"[utils] Using device: {device}")
    return device

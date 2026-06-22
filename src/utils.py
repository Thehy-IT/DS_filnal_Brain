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
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """
    Fix random seeds for full reproducibility across:
      Python random, NumPy, PyTorch (CPU + CUDA), cuDNN.

    Args:
        seed: Integer seed value. Default 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Make cuDNN deterministic (slightly slower but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"[utils] Seed set to {seed}")


# ---------------------------------------------------------------------------
# Dataset discovery
# ---------------------------------------------------------------------------

def find_data_dir(
    candidates: Optional[List[str]] = None,
    subfolder: str = "Training"
) -> str:
    """
    Auto-detect dataset root directory from a list of candidate paths.
    Useful when running on Kaggle, Colab, or a local machine.

    Args:
        candidates: List of directory paths to check (in priority order).
                    Defaults to common local / Kaggle / Colab paths.
        subfolder:  Expected sub-directory (e.g. 'Training') used to confirm
                    the right root was found.

    Returns:
        Absolute path to the directory that contains `subfolder`.

    Raises:
        FileNotFoundError: If none of the candidates contain `subfolder`.
    """
    if candidates is None:
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
# Class weights (for imbalanced datasets)
# ---------------------------------------------------------------------------

def compute_class_weights(labels: List[int], num_classes: int) -> torch.Tensor:
    """
    Compute inverse-frequency class weights to address class imbalance.

    Formula: weight_c = total_samples / (num_classes * count_c)

    Args:
        labels:      List of integer class labels for the training set.
        num_classes: Total number of classes.

    Returns:
        torch.Tensor of shape (num_classes,) with float weights.
    """
    labels_np = np.array(labels)
    class_weights = []
    total = len(labels_np)

    for c in range(num_classes):
        count = np.sum(labels_np == c)
        if count == 0:
            weight = 0.0
        else:
            weight = total / (num_classes * count)
        class_weights.append(weight)

    weights = torch.tensor(class_weights, dtype=torch.float32)
    print(f"[utils] Class weights: { {i: f'{w:.4f}' for i, w in enumerate(class_weights)} }")
    return weights


# ---------------------------------------------------------------------------
# Training history persistence
# ---------------------------------------------------------------------------

def save_history(history: Dict, save_path: str) -> None:
    """
    Save training history dictionary to a JSON file.

    Args:
        history:   Dict with keys like 'train_loss', 'val_loss', 'val_acc', etc.
                   Each value is a list of per-epoch floats.
        save_path: Full path to output .json file.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    print(f"[utils] Training history saved → {save_path}")


def load_history(save_path: str) -> Dict:
    """
    Load a previously saved training history JSON.

    Args:
        save_path: Path to the .json file.

    Returns:
        Dictionary with epoch-by-epoch metrics.
    """
    with open(save_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Device helper
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """
    Return the best available device: CUDA > MPS (Apple Silicon) > CPU.

    Returns:
        torch.device
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[utils] Using device: {device}")
    return device

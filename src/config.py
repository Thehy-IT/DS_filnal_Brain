"""
config.py — Centralized configuration for BrainTumorAI.

All hyperparameters, paths, and model settings are defined here.
Override via environment variables using python-dotenv.
"""
import os
from dataclasses import dataclass, field
from typing import List

from dotenv import load_dotenv

load_dotenv()  # Load .env if present


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED: int = 42


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
@dataclass
class DataConfig:
    """Paths and dataset split settings."""
    data_dir: str = os.environ.get(
        "DATA_DIR",
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "Training"))
    )
    test_dir: str = os.environ.get(
        "TEST_DIR",
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "Testing"))
    )
    class_names: List[str] = field(
        default_factory=lambda: ["glioma", "meningioma", "notumor", "pituitary"]
    )
    val_split: float = 0.2           # Fraction of training data used for validation
    image_size: int = 224            # Input resolution for EfficientNetB0
    num_workers: int = 4             # DataLoader workers (set 0 on Windows if errors)


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------
@dataclass
class AugConfig:
    """Data augmentation settings for training."""
    random_horizontal_flip: float = 0.5
    random_rotation_degrees: int = 20
    color_jitter_brightness: float = 0.2
    color_jitter_contrast: float = 0.2
    color_jitter_saturation: float = 0.1
    random_affine_degrees: int = 0       # Only shear/scale, no extra rotation
    random_affine_shear: float = 10.0
    random_affine_scale: tuple = (0.9, 1.1)
    random_erasing_p: float = 0.1       # Randomly erase patches (regularization)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
@dataclass
class TrainConfig:
    """Training loop settings."""
    model_name: str = os.environ.get("MODEL_NAME", "efficientnet")  # 'efficientnet' | 'cnn'
    batch_size: int = int(os.environ.get("BATCH_SIZE", "32"))
    epochs: int = int(os.environ.get("EPOCHS", "30"))

    # Phase 1 — Frozen backbone, train classifier only
    phase1_epochs: int = 5
    phase1_lr: float = 1e-3

    # Phase 2 — Unfreeze all layers for fine-tuning
    phase2_epochs: int = 25
    phase2_lr: float = 1e-4

    weight_decay: float = 1e-4

    # Scheduler
    lr_scheduler_factor: float = 0.5
    lr_scheduler_patience: int = 3

    # Early stopping
    early_stopping_patience: int = 5

    # Paths
    save_dir: str = os.environ.get(
        "MODEL_DIR",
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
    )
    reports_dir: str = os.environ.get(
        "REPORTS_DIR",
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "reports"))
    )


# ---------------------------------------------------------------------------
# Inference / API
# ---------------------------------------------------------------------------
@dataclass
class InferenceConfig:
    """Inference and API settings."""
    model_path: str = os.environ.get(
        "MODEL_PATH",
        os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "models", "efficientnet_best.pth")
        )
    )
    model_name: str = "efficientnet"
    api_host: str = os.environ.get("API_HOST", "127.0.0.1")
    api_port: int = int(os.environ.get("API_PORT", "8000"))
    max_file_size_mb: int = 10          # Max upload size in MB
    allowed_extensions: List[str] = field(
        default_factory=lambda: [".jpg", ".jpeg", ".png"]
    )


# ---------------------------------------------------------------------------
# Singleton accessors
# ---------------------------------------------------------------------------
data_cfg = DataConfig()
aug_cfg = AugConfig()
train_cfg = TrainConfig()
inference_cfg = InferenceConfig()

"""
config.py — File cấu hình trung tâm (Centralized configuration) cho dự án BrainTumorAI.

HƯỚNG DẪN: 
Toàn bộ siêu tham số (hyperparameters), đường dẫn file, và cài đặt mô hình được định nghĩa ở đây.
Bạn có thể thay đổi trực tiếp các giá trị ở đây hoặc ghi đè thông qua file biến môi trường (.env).
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
    """Cấu hình dữ liệu và phân chia tập Train/Validation."""
    # Đường dẫn đến thư mục chứa dữ liệu huấn luyện (Training)
    data_dir: str = os.environ.get(
        "DATA_DIR",
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "Training"))
    )
    # Đường dẫn đến thư mục chứa dữ liệu kiểm thử độc lập (Testing)
    test_dir: str = os.environ.get(
        "TEST_DIR",
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "Testing"))
    )
    # Danh sách 4 loại u não theo đúng thứ tự thư mục
    class_names: List[str] = field(
        default_factory=lambda: ["glioma", "meningioma", "notumor", "pituitary"]
    )
    # Tỷ lệ chia tập validation từ tập training (0.2 = 20%)
    val_split: float = 0.2           
    # Kích thước ảnh đầu vào cho mạng CNN (Mặc định 224x224 cho EfficientNet và DenseNet)
    image_size: int = 224            
    # Số luồng (workers) load dữ liệu song song. Nếu chạy trên Windows bị lỗi, hãy đổi thành 0.
    num_workers: int = 4             


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------
@dataclass
class AugConfig:
    """Cấu hình các kỹ thuật làm phong phú dữ liệu (Data Augmentation).
    Giúp tạo ra nhiều biến thể ảnh MRI để mô hình học tốt hơn, tránh học vẹt (overfitting).
    """
    random_horizontal_flip: float = 0.5   # Xác suất lật ngang ảnh (50%)
    random_rotation_degrees: int = 20     # Xoay ảnh ngẫu nhiên trong khoảng -20 đến 20 độ
    color_jitter_brightness: float = 0.2  # Thay đổi độ sáng ngẫu nhiên 20%
    color_jitter_contrast: float = 0.2    # Thay đổi độ tương phản ngẫu nhiên 20%
    color_jitter_saturation: float = 0.1  # Thay đổi độ bão hòa màu 10%
    random_affine_degrees: int = 0        # Giữ bằng 0 vì đã dùng rotation ở trên
    random_affine_shear: float = 10.0     # Làm méo (shear) ảnh ngẫu nhiên tối đa 10 độ
    random_affine_scale: tuple = (0.9, 1.1) # Phóng to / Thu nhỏ ảnh từ 90% đến 110%
    random_erasing_p: float = 0.1         # Xác suất che đen (Erase) một vùng nhỏ trên ảnh để bắt mô hình chú ý toàn diện


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
@dataclass
class TrainConfig:
    """Cấu hình quy trình huấn luyện mô hình (Training Loop)."""
    # Lựa chọn kiến trúc mô hình chạy mặc định: 'efficientnet' hoặc 'densenet'
    model_name: str = os.environ.get("MODEL_NAME", "efficientnet")  
    # Kích thước lô (batch size) số ảnh đưa vào RAM/VRAM mỗi lần huấn luyện (giảm xuống 16 nếu máy báo hết VRAM)
    batch_size: int = int(os.environ.get("BATCH_SIZE", "32"))
    # Tổng số vòng lặp tối đa toàn tập dữ liệu. Bạn nên để 40-50 như đã thảo luận.
    epochs: int = int(os.environ.get("EPOCHS", "40"))

    # Phase 1: Đóng băng lớp trích xuất đặc trưng (Backbone), chỉ train lớp phân loại cuối (Classifier)
    phase1_epochs: int = 5      # Số epochs cho Phase 1
    phase1_lr: float = 1e-3     # Tốc độ học (Learning Rate) tương đối lớn vì chỉ train 1 lớp nhỏ

    # Phase 2: Mở băng toàn bộ (Unfreeze) để "tinh chỉnh" (Fine-tuning)
    phase2_epochs: int = 35     # Số epochs cho Phase 2 (Cộng lại = Tổng epochs)
    phase2_lr: float = 1e-4     # Tốc độ học phải rất nhỏ (1e-4) để tránh phá hỏng trọng số đã pre-trained

    # Hệ số phạt (Weight decay) giúp chống overfitting
    weight_decay: float = 1e-4

    # Bộ lập lịch giảm learning rate (LR Scheduler) khi loss không giảm
    lr_scheduler_factor: float = 0.5   # Giảm LR xuống một nửa (x0.5)
    lr_scheduler_patience: int = 3     # Giảm LR sau 3 epochs liên tiếp loss không giảm

    # Dừng sớm (Early Stopping)
    early_stopping_patience: int = 7   # Nếu sau 7 epochs loss không giảm, lập tức ngắt quá trình train để chống overfit

    # Thư mục lưu trọng số model tốt nhất sau khi train xong
    save_dir: str = os.environ.get(
        "MODEL_DIR",
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
    )
    # Thư mục lưu báo cáo (Loss plot, Confusion Matrix)
    reports_dir: str = os.environ.get(
        "REPORTS_DIR",
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "reports"))
    )


# ---------------------------------------------------------------------------
# Inference / API
# ---------------------------------------------------------------------------
@dataclass
class InferenceConfig:
    """Cấu hình cho việc chạy thực tế (Inference) và API Backend (FastAPI)."""
    # Đường dẫn nạp file trọng số (.pth) tốt nhất khi khởi động server
    model_path: str = os.environ.get(
        "MODEL_PATH",
        os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "models", "efficientnet_best.pth")
        )
    )
    # Tên mô hình sẽ load trong FastAPI (hãy đổi thành "densenet" nếu bạn muốn API dùng DenseNet121)
    model_name: str = "efficientnet"
    
    # Cấu hình mạng cho API
    api_host: str = os.environ.get("API_HOST", "127.0.0.1")
    api_port: int = int(os.environ.get("API_PORT", "8000"))
    
    # Giới hạn kích thước file tải lên (tránh spam hoặc lỗi tràn RAM)
    max_file_size_mb: int = 10          
    # Các định dạng ảnh được phép upload
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

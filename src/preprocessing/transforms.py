"""
transforms.py — Data augmentation and normalization pipelines.

Three pipeline levels (inspired by DAKHDL's build_augmentation):
  - get_train_transforms()  : Full augmentation for training
  - get_valid_transforms()  : Resize + normalize only (no augmentation)
  - get_heavy_transforms()  : Aggressive augmentation for hard cases

ImageNet normalization stats are used since models were pre-trained on ImageNet.
"""
from torchvision import transforms

from src.config import aug_cfg, data_cfg

# Các hằng số chuẩn hóa ImageNet (mean / std theo từng kênh R, G, B)
# Phải dùng đúng giá trị này vì model đã pre-trained trên ImageNet với stats này
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Training transform — pipeline đầy đủ augmentation cho tập huấn luyện
# ---------------------------------------------------------------------------

def get_train_transforms(image_size: int = None) -> transforms.Compose:
    """
    Pipeline biến đổi ảnh với augmentation đầy đủ cho tập training.

    Chiến lược augmentation (dịch từ Keras ImageDataGenerator của DAKHDL):
      - RandomHorizontalFlip  : Lật ngang ảnh, mô phỏng sự đa dạng hướng quét
      - RandomRotation        : Xoay +/-20 do (DAKHDL: rotation_range=20)
      - ColorJitter           : Thay doi do sang/tuong phan/bao hoa
      - RandomAffine          : Shear + scale (DAKHDL: shear/zoom_range=0.1)
      - RandomErasing         : Che ngau nhien mot vung anh (regularization)
      - Normalize             : Chuan hoa theo ImageNet stats

    Args:
        image_size: Kich thuoc anh vuong. Mac dinh tu config (224).

    Returns:
        torchvision.transforms.Compose
    """
    size = image_size or data_cfg.image_size
    return transforms.Compose([
        # Resize ve kich thuoc chuan truoc toan bo pipeline
        transforms.Resize((size, size)),
        # Lat ngang ngau nhien (50% xac suat) — anh MRI co the quay huong bat ky
        transforms.RandomHorizontalFlip(
            p=aug_cfg.random_horizontal_flip
        ),
        # Xoay ngau nhien trong khoang -20 den +20 do
        transforms.RandomRotation(
            degrees=aug_cfg.random_rotation_degrees
        ),
        # Thay doi do sang, tuong phan, bao hoa ngau nhien de model khong phu thuoc vao anh sang
        transforms.ColorJitter(
            brightness=aug_cfg.color_jitter_brightness,
            contrast=aug_cfg.color_jitter_contrast,
            saturation=aug_cfg.color_jitter_saturation,
        ),
        # Shear + scale mô phỏng shear_range & zoom_range của DAKHDL
        transforms.RandomAffine(
            degrees=aug_cfg.random_affine_degrees,  # 0: đã xoay bằng RandomRotation rồi
            shear=aug_cfg.random_affine_shear,      # Biến dạng xiên tối đa 10 độ
            scale=aug_cfg.random_affine_scale,      # Thu phóng 90%-110%
            fill=0,                                 # Pixel nền đen khi biến dạng
        ),
        # Chuyển PIL Image sang Tensor float32 [0, 1]
        transforms.ToTensor(),
        # Chuẩn hóa theo ImageNet stats (phải sau ToTensor)
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        # Che ngẫu nhiên một vùng nhỏ — buộc model học toàn bộ ảnh, không chỉ một vùng
        transforms.RandomErasing(
            p=aug_cfg.random_erasing_p,  # Xác suất 10%
            scale=(0.02, 0.1),           # Vùng che 2%-10% diện tích
            ratio=(0.3, 3.3),            # Tỷ lệ khung hình của vùng che
            value=0,                     # Che bằng màu đen
        ),
    ])


# ---------------------------------------------------------------------------
# Validation transform — chỉ resize + normalize, không augmentation
# ---------------------------------------------------------------------------

def get_valid_transforms(image_size: int = None) -> transforms.Compose:
    """
    Pipeline biến đổi ảnh cho validation / test — KHÔNG augmentation.

    Chỉ resize và chuẩn hóa để đảm bảo inference nhất quán và đánh giá
    model trên dữ liệu thực tế (không có nhiễu nhân tạo).

    Args:
        image_size: Kích thước ảnh vuông. Mặc định từ config (224).

    Returns:
        torchvision.transforms.Compose
    """
    size = image_size or data_cfg.image_size
    return transforms.Compose([
        transforms.Resize((size, size)),   # Resize về đúng kích thước
        transforms.ToTensor(),             # Chuyển sang Tensor
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),  # Chuẩn hóa
    ])


# ---------------------------------------------------------------------------
# Heavy transform — augmentation nặng cho các lớp thiểu số (vd: meningioma)
# ---------------------------------------------------------------------------

def get_heavy_transforms(image_size: int = None) -> transforms.Compose:
    """
    Pipeline augmentation mạnh cho các lớp ít dữ liệu (hard / minority classes).

    Bổ sung thêm các biến đổi hình học so với pipeline training chuẩn.
    Sử dụng có chọn lọc, ví dụ cho lớp meningioma vốn là lớp nhỏ nhất
    trong bộ dữ liệu Brain Tumor.

    Args:
        image_size: Kích thước ảnh vuông. Mặc định từ config (224).

    Returns:
        torchvision.transforms.Compose
    """
    size = image_size or data_cfg.image_size
    return transforms.Compose([
        # Resize lớn hơn mục tiêu, rồi crop ngẫu nhiên — tăng tính đa dạng vị trí
        transforms.Resize((size + 32, size + 32)),
        transforms.RandomCrop(size),
        transforms.RandomHorizontalFlip(p=0.5),
        # Lật dọc: hợp lý với ảnh MRI vì não có thể ở nhiều tư thế
        transforms.RandomVerticalFlip(p=0.2),
        # Xoay mạnh hơn (30 độ) so với pipeline chuẩn (20 độ)
        transforms.RandomRotation(degrees=30),
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.2,
            hue=0.05,   # Thêm hue jitter nhẹ
        ),
        transforms.RandomAffine(
            degrees=0,
            shear=15,           # Shear mạnh hơn (15 vs 10)
            scale=(0.85, 1.15), # Zoom rộng hơn (85%-115%)
            fill=0,
        ),
        # Làm mờ nhẹ để model học robust hơn với ảnh không sắc nét
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        # Erase xác suất cao hơn (20%) và vùng lớn hơn (15%)
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
    ])

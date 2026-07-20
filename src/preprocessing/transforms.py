"""
transforms.py — Data augmentation and normalization pipelines.

Three pipeline levels (inspired by DAKHDL's build_augmentation):
  - get_train_transforms()  : Full augmentation for training
  - get_valid_transforms()  : Resize + normalize only (no augmentation)
  - get_heavy_transforms()  : Aggressive augmentation for hard cases

ImageNet normalization stats are used since models were pre-trained on ImageNet.

[v2 — Anti-Domain-Shift]
  get_train_transforms now includes:
    - GaussianBlur      : simulates low-res MRI scanners
    - RandomGrayscale   : simulates grayscale MRI display
    - RandomPerspective : simulates different scanner angles
    - Stronger rotation (30°), brightness/contrast (0.3), erasing (0.2)
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

    Chiến lược augmentation (v2 — Anti-Domain-Shift):
      - RandomHorizontalFlip  : Lật ngang ảnh, mô phỏng sự đa dạng hướng quét
      - RandomRotation        : [v2] Xoay +/-30 độ (tăng từ 20)
      - ColorJitter           : [v2] Brightness/contrast 0.3 (tăng từ 0.2)
      - RandomAffine          : [v2] Zoom 85%–115% (mở rộng từ 90%–110%)
      - RandomGrayscale       : [v2 MỚI] Mô phỏng ảnh MRI grayscale (10%)
      - GaussianBlur          : [v2 MỚI] Mô phỏng thiết bị chụp kém sắc nét (30%)
      - RandomPerspective     : [v2 MỚI] Mô phỏng góc chụp MRI khác nhau (20%)
      - RandomErasing         : [v2] Xác suất che 20% (tăng từ 10%)
      - Normalize             : Chuẩn hóa theo ImageNet stats

    Args:
        image_size: Kích thước ảnh vuông. Mặc định từ config (224).

    Returns:
        torchvision.transforms.Compose
    """
    size = image_size or data_cfg.image_size
    return transforms.Compose([
        # Resize về kích thước chuẩn trước toàn bộ pipeline
        transforms.Resize((size, size)),
        # Lật ngang ngẫu nhiên (50% xác suất) — ảnh MRI có thể quay hướng bất kỳ
        transforms.RandomHorizontalFlip(
            p=aug_cfg.random_horizontal_flip
        ),
        # [v2] Xoay ngẫu nhiên trong khoảng -30 đến +30 độ (tăng từ 20)
        transforms.RandomRotation(
            degrees=aug_cfg.random_rotation_degrees
        ),
        # [v2] Thay đổi độ sáng/tương phản mạnh hơn (0.3 thay vì 0.2)
        # Mô phỏng sự khác biệt về độ sáng/tương phản giữa các máy MRI khác nhau
        transforms.ColorJitter(
            brightness=aug_cfg.color_jitter_brightness,
            contrast=aug_cfg.color_jitter_contrast,
            saturation=aug_cfg.color_jitter_saturation,
        ),
        # [v2] Shear + scale với zoom rộng hơn (85%—115% thay vì 90%—110%)
        transforms.RandomAffine(
            degrees=aug_cfg.random_affine_degrees,  # 0: đã xoay bằng RandomRotation rồi
            shear=aug_cfg.random_affine_shear,      # Biến dạng xiên tối đa 10 độ
            scale=aug_cfg.random_affine_scale,      # [v2] Thu phóng 85%-115%
            fill=0,                                 # Pixel nền đen khi biến dạng
        ),
        # [v2 — MỚI] Chuyển ảnh sang grayscale rồi lại về RGB (10% xác suất)
        # Mô phỏng ảnh MRI từ bệnh viện hiển thị dạng grayscale/pseudo-color
        transforms.RandomGrayscale(
            p=aug_cfg.random_grayscale_p
        ),
        # [v2 — MỚI] Làm mờ ảnh nhẹ để model không phụ thuộc vào độ sắc nét tuyệt đối
        # Mô phỏng ảnh MRI từ thiết bị có độ phân giải thấp hơn dataset Kaggle
        # Dùng RandomApply để chỉ áp dụng với xác suất gaussian_blur_p (30%)
        transforms.RandomApply(
            [transforms.GaussianBlur(
                kernel_size=aug_cfg.gaussian_blur_kernel,
                sigma=aug_cfg.gaussian_blur_sigma,
            )],
            p=aug_cfg.gaussian_blur_p,
        ),
        # [v2 — MỚI] Biến đổi phối cảnh ngẫu nhiên — mô phỏng góc chụp MRI khác nhau
        # giữa các bệnh viện/thiết bị. Mức méo nhẹ (10%) để không làm méo quá nhiễu nhãn.
        transforms.RandomPerspective(
            distortion_scale=aug_cfg.random_perspective_distortion,
            p=aug_cfg.random_perspective_p,
            fill=0,
        ),
        # Chuyển PIL Image sang Tensor float32 [0, 1]
        transforms.ToTensor(),
        # Chuẩn hóa theo ImageNet stats (phải sau ToTensor)
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        # [v2] Che ngẫu nhiên một vùng nhỏ với xác suất 20% (tăng từ 10%)
        transforms.RandomErasing(
            p=aug_cfg.random_erasing_p,  # [v2] 20% thay vì 10%
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


# ---------------------------------------------------------------------------
# TTA transforms — augmentation nhẹ cho Test-Time Augmentation
# ---------------------------------------------------------------------------

def get_tta_transforms(image_size: int = None) -> list:
    """
    Tạo danh sách các transform nhẹ dùng cho Test-Time Augmentation (TTA).

    [Giai đoạn 3 — Anti-Domain-Shift]
    TTA dự đoán cùng một ảnh nhiều lần với các biến đổi nhẹ khác nhau,
    sau đó lấy trung bình xác suất → kết quả ổn định hơn, ít bị ảnh hưởng
    bởi sự khác biệt nhỏ về góc chụp, độ sáng của ảnh từ bệnh viện khác.

    Nguyên tắc thiết kế TTA:
      - Augmentation NHẸ hơn training nhiều: không dùng RandomErasing,
        không ColorJitter mạnh → tránh phá hủy đặc trưng y tế quan trọng
      - Mỗi transform mô phỏng 1 biến thể thực tế của ảnh MRI ngoài domain
      - Transform đầu tiên luôn là valid_transform (ảnh gốc, không biến đổi)
        → đảm bảo TTA không bao giờ tệ hơn inference đơn

    5 transform variants:
      [0] Gốc: resize + normalize (baseline, luôn có)
      [1] Lật ngang: mô phỏng ảnh MRI chụp ngược chiều
      [2] Xoay nhẹ ±10°: mô phỏng ảnh đặt hơi nghiêng
      [3] Độ sáng/tương phản thay đổi nhẹ: mô phỏng thiết bị MRI khác
      [4] Zoom nhẹ 95%-105%: mô phỏng khoảng cách chụp khác nhau

    Args:
        image_size: Kích thước ảnh vuông. Mặc định từ config (224).

    Returns:
        List[transforms.Compose] — danh sách 5 pipeline transform độc lập.
    """
    size = image_size or data_cfg.image_size

    # Chuẩn hóa ImageNet — dùng chung cho tất cả các TTA variant
    _normalize = transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD)

    return [
        # --- Variant 0: Ảnh gốc (baseline) — không augmentation ---
        # Luôn bao gồm ảnh gốc để TTA không bao giờ tệ hơn inference đơn
        transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            _normalize,
        ]),

        # --- Variant 1: Lật ngang ---
        # Mô phỏng ảnh MRI chụp từ phía đối diện (T1 vs T2 orientation)
        transforms.Compose([
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(p=1.0),  # p=1.0: luôn lật
            transforms.ToTensor(),
            _normalize,
        ]),

        # --- Variant 2: Xoay nhẹ ±10 độ ---
        # Mô phỏng ảnh MRI đặt không hoàn toàn thẳng (positioning artifact)
        # ±10° (nhẹ hơn training ±30°) để không méo đặc trưng y tế
        transforms.Compose([
            transforms.Resize((size, size)),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            _normalize,
        ]),

        # --- Variant 3: Thay đổi độ sáng/tương phản nhẹ ---
        # Mô phỏng sự khác biệt cường độ tín hiệu giữa các máy MRI khác nhau
        # brightness/contrast 0.15 (nhẹ hơn training 0.3) để giữ đặc trưng
        transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
            _normalize,
        ]),

        # --- Variant 4: Zoom nhẹ 95%-105% ---
        # Mô phỏng khoảng cách chụp khác nhau giữa các cơ sở y tế
        # Scale range hẹp (0.95-1.05) để không cắt mất cấu trúc não
        transforms.Compose([
            transforms.Resize((size, size)),
            transforms.RandomAffine(degrees=0, scale=(0.95, 1.05), fill=0),
            transforms.ToTensor(),
            _normalize,
        ]),
    ]


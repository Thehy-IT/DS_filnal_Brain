"""
dataset.py — BrainTumorDataset with stratified splitting and class weights.

Key improvements over v1:
  - Explicit path validation with informative errors
  - Labels exposed as a public list (needed for stratified split & class weights)
  - Class weights computed once and stored on the dataset
  - Static helper split_dataset() uses stratified train/val split (fixes the
    bug where random_split() shared the same transform for both subsets)
"""
import os
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Subset

from src.config import data_cfg, SEED


# ---------------------------------------------------------------------------
# BrainTumorDataset — Dataset PyTorch tùy chỉnh cho ảnh MRI u não
# ---------------------------------------------------------------------------

class BrainTumorDataset(Dataset):
    """
    PyTorch Dataset cho ảnh MRI u não.

    Yêu cầu cấu trúc thư mục theo lớp:
        root_dir/
            glioma/
            meningioma/
            notumor/
            pituitary/

    Thuộc tính:
        root_dir (str):            Đường dẫn tuyệt đối đến thư mục gốc.
        transform (callable):      Pipeline biến đổi ảnh.
        classes (List[str]):       Danh sách tên lớp theo thứ tự.
        class_to_idx (dict):       Ánh xạ tên lớp -> index số nguyên.
        image_paths (List[str]):   Đường dẫn tuyệt đối đến từng ảnh.
        labels (List[int]):        Nhãn số nguyên của từng ảnh.
        class_weights (Tensor):    Trọng số nghịch tần số chống mất cân bằng.
    """

    # Chỉ chấp nhận các định dạng ảnh phổ biến
    VALID_EXTENSIONS = {".png", ".jpg", ".jpeg"}

    def __init__(
        self,
        root_dir: str,
        transform=None,
        class_names: Optional[List[str]] = None,
    ) -> None:
        """
        Args:
            root_dir:     Đường dẫn thư mục chứa các thư mục con theo lớp.
            transform:    Transform tùy chọn (torchvision hoặc albumentations).
            class_names:  Override danh sách lớp (mặc định từ config).
        """
        self.root_dir = os.path.abspath(root_dir)
        self.transform = transform
        self.classes: List[str] = class_names or data_cfg.class_names
        # Tạo mapping tên lớp -> số nguyên: {"glioma": 0, "meningioma": 1, ...}
        self.class_to_idx: dict = {c: i for i, c in enumerate(self.classes)}

        self.image_paths: List[str] = []
        self.labels: List[int] = []

        # Xây dựng index ảnh và tính trọng số lớp ngay khi khởi tạo
        self._load_index()
        self._compute_class_weights()

    # ------------------------------------------------------------------
    # Hàm nội bộ
    # ------------------------------------------------------------------

    def _load_index(self) -> None:
        """Duyệt root_dir và xây dựng danh sách đường dẫn ảnh + nhãn."""
        # Kiểm tra thư mục gốc tồn tại trước khi làm bất cứ điều gì
        if not os.path.isdir(self.root_dir):
            raise FileNotFoundError(
                f"[BrainTumorDataset] Directory not found: {self.root_dir}\n"
                "Please set DATA_DIR in your .env or pass the correct path."
            )

        missing_classes = []
        for cls_name in self.classes:
            cls_dir = os.path.join(self.root_dir, cls_name)
            if not os.path.isdir(cls_dir):
                # Ghi nhận lớp bị thiếu thư mục, không crash ngay
                missing_classes.append(cls_name)
                continue

            found = 0
            # Sắp xếp để đảm bảo thứ tự nhất quán giữa các lần chạy
            for fname in sorted(os.listdir(cls_dir)):
                ext = os.path.splitext(fname)[1].lower()
                if ext in self.VALID_EXTENSIONS:
                    self.image_paths.append(os.path.join(cls_dir, fname))
                    self.labels.append(self.class_to_idx[cls_name])
                    found += 1

            print(f"[Dataset]  {cls_name:>12}: {found:>5} images")

        if missing_classes:
            print(
                f"[Dataset] WARNING - classes not found in '{self.root_dir}': "
                f"{missing_classes}"
            )

        total = len(self.image_paths)
        if total == 0:
            raise RuntimeError(
                f"[BrainTumorDataset] No images found under '{self.root_dir}'. "
                "Check the directory structure."
            )
        print(f"[Dataset] Total images loaded: {total}")

    def _compute_class_weights(self) -> None:
        """Tính trọng số nghịch tần số: weight_c = N / (C * count_c)."""
        labels_np = np.array(self.labels)
        total = len(labels_np)
        num_classes = len(self.classes)
        weights = []
        for c in range(num_classes):
            count = int(np.sum(labels_np == c))
            # Tránh chia 0; lớp không có ảnh được gán trọng số 0
            weights.append(total / (num_classes * count) if count > 0 else 0.0)
        self.class_weights = torch.tensor(weights, dtype=torch.float32)

    # ------------------------------------------------------------------
    # Giao diện PyTorch Dataset (bắt buộc phải implement)
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        # Trả về tổng số ảnh trong dataset
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        try:
            # Luôn convert sang RGB để đảm bảo 3 kênh màu (tránh ảnh grayscale hoặc RGBA)
            image = Image.open(img_path).convert("RGB")
        except Exception as exc:
            # Ảnh bị hỏng: trả về ảnh trắng thay vì crash DataLoader
            print(f"[Dataset] WARNING - could not open image {img_path}: {exc}")
            image = Image.new("RGB", (data_cfg.image_size, data_cfg.image_size))

        if self.transform:
            image = self.transform(image)

        return image, label

    def __repr__(self) -> str:
        return (
            f"BrainTumorDataset(root='{self.root_dir}', "
            f"n={len(self)}, classes={self.classes})"
        )


# ---------------------------------------------------------------------------
# split_dataset — chia tập train/val với chiến lược stratified (đồng đều lớp)
# ---------------------------------------------------------------------------

def split_dataset(
    train_dir: str,
    train_transform,
    val_transform,
    val_split: float = None,
    seed: int = SEED,
) -> Tuple[Dataset, Dataset]:
    """
    Tạo tập train / validation riêng biệt với transform TACH BIET nhau.

    Hàm này fix một bug nghiêm trọng trong phiên bản cũ: khi dùng random_split(),
    cả train lẫn val dùng chung một Dataset object, nên transform val vô tình
    bị áp dụng cho cả ảnh train (data contamination).

    Chiến lược:
      1. Xây dựng dataset chỉ để lấy index và nhãn (không cần transform).
      2. Chia stratified theo nhãn bằng sklearn.
      3. Tạo 2 Dataset độc lập với transform riêng — train có augmentation, val thì không.

    Args:
        train_dir:       Đường dẫn thư mục dữ liệu training.
        train_transform: Pipeline augmentation cho training.
        val_transform:   Pipeline chuẩn hóa cho validation (không augment).
        val_split:       Tỷ lệ [0, 1] dành cho validation. Mặc định từ config.
        seed:            Random seed để kết quả lặp lại.

    Returns:
        (train_subset, val_subset) — cả 2 đều là PyTorch Dataset hợp lệ.
    """
    val_split = val_split if val_split is not None else data_cfg.val_split

    # Bước 1: Tạo dataset tạm (không transform) chỉ để lấy nhãn
    index_ds = BrainTumorDataset(root_dir=train_dir, transform=None)
    all_indices = list(range(len(index_ds)))
    all_labels = index_ds.labels

    # Bước 2: Stratified split — giữ tỷ lệ các lớp đồng đều ở cả train và val
    train_idx, val_idx = train_test_split(
        all_indices,
        test_size=val_split,
        random_state=seed,
        stratify=all_labels,  # Quan trọng: stratify theo nhãn
    )
    print(
        f"[Dataset] Split -> train: {len(train_idx)} | val: {len(val_idx)} "
        f"(stratified, seed={seed})"
    )

    # Bước 3: Tạo 2 Dataset độc lập với transform khác nhau
    # train_ds dùng augmentation, val_ds chỉ resize + normalize
    train_ds = BrainTumorDataset(root_dir=train_dir, transform=train_transform)
    val_ds = BrainTumorDataset(root_dir=train_dir, transform=val_transform)

    # Subset giới hạn mỗi dataset về đúng các index đã chia
    return Subset(train_ds, train_idx), Subset(val_ds, val_idx)

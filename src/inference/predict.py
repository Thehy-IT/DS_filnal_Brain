"""
predict.py — TumorPredictor: load model và chạy inference.

[v2 — Giai đoạn 3: Test-Time Augmentation]
Bổ sung phương thức predict_with_tta() bên cạnh predict() gốc:
  - predict()          : Dự đoán đơn, nhanh (1 forward pass)
  - predict_with_tta() : Dự đoán TTA, chính xác hơn (5 forward passes, trung bình xác suất)

TTA giúp model robust hơn với ảnh ngoài domain (ảnh giảng viên test)
bằng cách "nhìn" ảnh từ nhiều góc độ rồi lấy kết quả đồng thuận.
"""
import torch
import torch.nn.functional as F
from PIL import Image

from src.training.models import get_model
from src.preprocessing.transforms import get_valid_transforms, get_tta_transforms


# ---------------------------------------------------------------------------
# TumorPredictor — class đóng gói toàn bộ quy trình load model và inference
# ---------------------------------------------------------------------------

class TumorPredictor:
    """
    Đóng gói model đã train và cung cấp giao diện đơn giản để dự đoán.
    Được dùng bởi FastAPI backend khi nhận ảnh upload từ client.

    Attributes:
        device:    CPU hoặc CUDA (tự động phát hiện)
        classes:   Danh sách 4 tên lớp theo đúng thứ tự index
        transform: Pipeline valid_transform dùng cho predict() đơn
        tta_transforms: Danh sách 5 pipeline dùng cho predict_with_tta()
        model:     Model đã load trọng số, ở eval mode
    """

    def __init__(self, model_path: str, model_name: str = 'efficientnet',
                 device: str = None):
        # Tự động chọn GPU nếu có, ngược lại dùng CPU
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        # Danh sách nhãn theo đúng thứ tự index (phải khớp với thứ tự lúc train)
        self.classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
        # Dùng validation transform (không augmentation) khi inference đơn
        self.transform = get_valid_transforms()
        # [v2] Nạp sẵn 5 TTA transform variants để predict_with_tta() dùng
        self.tta_transforms = get_tta_transforms()

        # Tải cấu trúc model (pretrained=False vì sẽ load trọng số từ file .pth)
        self.model = get_model(
            model_name=model_name,
            num_classes=len(self.classes),
            pretrained=False,
        )
        # Load trọng số đã train vào model, map_location đảm bảo tương thích CPU/GPU
        # weights_only=True: bảo mật hơn với PyTorch 2.0+ (tránh arbitrary code execution)
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device, weights_only=True)
        )
        self.model.to(self.device)
        # Chuyển sang eval mode: tắt dropout, dùng running stats của BatchNorm
        self.model.eval()

    # ------------------------------------------------------------------
    # predict — dự đoán đơn (1 forward pass, nhanh)
    # ------------------------------------------------------------------

    def predict(self, image: Image.Image) -> dict:
        """
        Dự đoán loại u não từ ảnh MRI đầu vào (1 forward pass).

        Dùng khi cần tốc độ nhanh. Với ảnh ngoài domain,
        nên dùng predict_with_tta() để có kết quả chính xác hơn.

        Args:
            image (PIL.Image.Image): Ảnh MRI đầu vào (bất kỳ kích thước nào).

        Returns:
            dict: Kết quả gồm:
                - class_name:    Tên lớp dự đoán ('glioma', 'meningioma', ...)
                - class_idx:     Index số nguyên của lớp
                - confidence:    Xác suất cao nhất (0.0 - 1.0)
                - probabilities: Danh sách xác suất softmax cho cả 4 lớp
                - method:        'single' — đánh dấu phương thức dự đoán
        """
        # Áp dụng transform: resize, normalize, chuyển sang Tensor
        # unsqueeze(0) thêm batch dimension: (C, H, W) -> (1, C, H, W)
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # Forward pass: tính logits (chưa qua softmax)
            outputs = self.model(image_tensor)
            # Chuyển logits sang xác suất, dim=1 là chiều lớp
            probabilities = F.softmax(outputs, dim=1)
            # Lấy xác suất và index của lớp cao nhất
            confidence, predicted_idx = torch.max(probabilities, 1)

        idx = predicted_idx.item()
        return {
            'class_name':    self.classes[idx],
            'class_idx':     idx,
            'confidence':    confidence.item(),
            'probabilities': probabilities.squeeze().cpu().numpy().tolist(),
            'method':        'single',
        }

    # ------------------------------------------------------------------
    # predict_with_tta — dự đoán TTA (5 forward passes, trung bình xác suất)
    # ------------------------------------------------------------------

    def predict_with_tta(self, image: Image.Image) -> dict:
        """
        Dự đoán loại u não bằng Test-Time Augmentation (TTA).

        [Giai đoạn 3 — Anti-Domain-Shift]
        Chạy 5 forward pass với 5 biến thể nhẹ của cùng 1 ảnh:
          [0] Ảnh gốc  [1] Lật ngang  [2] Xoay ±10°
          [3] Brightness/contrast nhẹ  [4] Zoom nhẹ 95-105%

        Sau đó lấy trung bình xác suất (mean ensemble) của cả 5 lần dự đoán.
        Kết quả ổn định hơn vì không phụ thuộc vào một "góc nhìn" duy nhất.

        Args:
            image (PIL.Image.Image): Ảnh MRI đầu vào (bất kỳ kích thước nào).

        Returns:
            dict: Kết quả gồm:
                - class_name:     Tên lớp dự đoán sau TTA ensemble
                - class_idx:      Index số nguyên của lớp
                - confidence:     Xác suất trung bình của lớp được chọn
                - probabilities:  Xác suất trung bình 4 lớp (sau khi ensemble)
                - method:         'tta_5' — đánh dấu phương thức dự đoán
                - tta_n:          Số lần dự đoán thực tế (= 5)
        """
        all_probs = []  # Thu thập xác suất từ mỗi TTA variant

        with torch.no_grad():
            for tf in self.tta_transforms:
                # Áp dụng từng TTA variant transform lên ảnh gốc
                tensor = tf(image).unsqueeze(0).to(self.device)
                # Forward pass: tính logits
                outputs = self.model(tensor)
                # Chuyển sang xác suất — shape: (1, 4)
                probs = F.softmax(outputs, dim=1)
                # Thu thập xác suất của lần này
                all_probs.append(probs)

        # Stack thành tensor (n_tta, 1, 4) rồi lấy trung bình theo chiều đầu
        # Kết quả: mean_probs shape = (1, 4) — xác suất ensemble
        mean_probs = torch.stack(all_probs, dim=0).mean(dim=0)

        # Lấy lớp có xác suất trung bình cao nhất
        confidence, predicted_idx = torch.max(mean_probs, dim=1)
        idx = predicted_idx.item()

        return {
            'class_name':    self.classes[idx],
            'class_idx':     idx,
            'confidence':    confidence.item(),
            'probabilities': mean_probs.squeeze().cpu().numpy().tolist(),
            'method':        'tta_5',
            'tta_n':         len(self.tta_transforms),
        }

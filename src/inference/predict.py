import torch
from PIL import Image
import torch.nn.functional as F

from src.training.models import get_model
from src.preprocessing.transforms import get_valid_transforms

# ---------------------------------------------------------------------------
# TumorPredictor — class đóng gói toàn bộ quy trình load model và inference
# ---------------------------------------------------------------------------

class TumorPredictor:
    """
    Đóng gói model đã train và cung cấp giao diện đơn giản để dự đoán.
    Được dùng bởi FastAPI backend khi nhận ảnh upload từ client.
    """
    def __init__(self, model_path: str, model_name: str = 'efficientnet', device: str = None):
        # Tự động chọn GPU nếu có, ngược lại dùng CPU
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        # Danh sách nhãn theo đúng thứ tự index (phải khớp với thứ tự lúc train)
        self.classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
        # Dùng validation transform (không augmentation) khi inference thực tế
        self.transform = get_valid_transforms()

        # Tải cấu trúc model (pretrained=False vì sẽ load trọng số từ file .pth)
        self.model = get_model(model_name=model_name, num_classes=len(self.classes), pretrained=False)
        # Load trọng số đã train vào model, map_location đảm bảo tương thích CPU/GPU
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        # Chuyển sang eval mode: tắt dropout, dùng running stats của BatchNorm
        self.model.eval()

    def predict(self, image: Image.Image):
        """
        Dự đoán loại u não từ ảnh MRI đầu vào.

        Args:
            image (PIL.Image.Image): Ảnh MRI đầu vào (bất kỳ kích thước nào).

        Returns:
            dict: Kết quả gồm:
                - class_name: Tên lớp dự đoán ('glioma', 'meningioma', ...)
                - class_idx:  Index số nguyên của lớp
                - confidence: Xác suất cao nhất (0.0 - 1.0)
                - probabilities: Danh sách xác suất softmax cho cả 4 lớp
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
            'class_name': self.classes[idx],                              # Tên lớp dự đoán
            'class_idx': idx,                                             # Index số nguyên
            'confidence': confidence.item(),                              # Điểm tin cậy
            'probabilities': probabilities.squeeze().cpu().numpy().tolist()  # Xác suất 4 lớp
        }

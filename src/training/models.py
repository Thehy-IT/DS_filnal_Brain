import torch
import torch.nn as nn
import timm

# ---------------------------------------------------------------------------
# DenseNet121 — mô hình thứ cấp dùng để so sánh benchmark y tế
# ---------------------------------------------------------------------------

class DenseNetModel(nn.Module):
    """
    Bọc DenseNet121 từ thư viện timm.
    Dùng làm mô hình so sánh (benchmark) bên cạnh EfficientNet.
    """
    def __init__(self, num_classes: int = 4, pretrained: bool = True):
        super(DenseNetModel, self).__init__()
        # Tải DenseNet121 đã pre-trained trên ImageNet, thay đầu phân loại bằng 4 lớp u não
        self.model = timm.create_model('densenet121', pretrained=pretrained, num_classes=num_classes)

    def forward(self, x):
        # Truyền ảnh qua toàn bộ mạng và trả về logits (chưa qua softmax)
        return self.model(x)


# ---------------------------------------------------------------------------
# EfficientNetB0 — mô hình chính được dùng mặc định trong hệ thống
# ---------------------------------------------------------------------------

class EfficientNetModel(nn.Module):
    """
    Bọc EfficientNetB0 từ thư viện timm.
    Nhỏ gọn (~5 MB), tốc độ inference nhanh, phù hợp deploy production.
    """
    def __init__(self, num_classes: int = 4, pretrained: bool = True):
        super(EfficientNetModel, self).__init__()
        # Tải EfficientNetB0 đã pre-trained trên ImageNet, thay đầu phân loại bằng 4 lớp u não
        self.model = timm.create_model('efficientnet_b0', pretrained=pretrained, num_classes=num_classes)

    def forward(self, x):
        # Truyền ảnh qua toàn bộ mạng và trả về logits
        return self.model(x)


# ---------------------------------------------------------------------------
# Factory function — tạo mô hình theo tên, tránh import trực tiếp từ ngoài
# ---------------------------------------------------------------------------

def get_model(model_name: str = 'efficientnet', num_classes: int = 4, pretrained: bool = True):
    """
    Hàm nhà máy: nhận tên mô hình, trả về instance tương ứng.
    Giúp train.py và predict.py không cần biết class cụ thể nào được dùng.
    """
    if model_name.lower() == 'densenet':
        return DenseNetModel(num_classes=num_classes, pretrained=pretrained)
    elif model_name.lower() == 'efficientnet':
        return EfficientNetModel(num_classes=num_classes, pretrained=pretrained)
    else:
        # Chỉ hỗ trợ 2 kiến trúc — ném lỗi rõ ràng thay vì âm thầm fail
        raise ValueError(f"Model {model_name} not supported. Use 'densenet' or 'efficientnet'.")

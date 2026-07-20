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

    [v2 — Anti-Domain-Shift]
    Thay classifier head bằng Sequential(Dropout(0.3), Linear) để tăng
    regularization, giúp model bớt overconfident với ảnh ngoài domain.
    """
    def __init__(self, num_classes: int = 4, pretrained: bool = True,
                 dropout_p: float = 0.3):
        super(DenseNetModel, self).__init__()
        # Tải DenseNet121 đã pre-trained trên ImageNet, thay đầu phân loại bằng 4 lớp u não
        self.model = timm.create_model('densenet121', pretrained=pretrained, num_classes=num_classes)

        # [v2] Lấy in_features của classifier gốc rồi thay bằng Sequential có Dropout
        # timm.DenseNet121.classifier là Linear(1024, num_classes)
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Sequential(
            # Dropout trước Linear giúp giảm co-adaptation giữa các neurons
            # dropout_p=0.3: tắt ngẫu nhiên 30% neurons mỗi forward pass khi training
            nn.Dropout(p=dropout_p),
            # Linear giữ nguyên kiến trúc phân loại — chỉ bổ sung dropout phía trước
            nn.Linear(in_features, num_classes),
        )

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

    [v2 — Anti-Domain-Shift]
    Thay classifier head bằng Sequential(Dropout(0.3), Linear) để tăng
    regularization, giúp model bớt overconfident với ảnh ngoài domain.
    """
    def __init__(self, num_classes: int = 4, pretrained: bool = True,
                 dropout_p: float = 0.3):
        super(EfficientNetModel, self).__init__()
        # Tải EfficientNetB0 đã pre-trained trên ImageNet, thay đầu phân loại bằng 4 lớp u não
        self.model = timm.create_model('efficientnet_b0', pretrained=pretrained, num_classes=num_classes)

        # [v2] Lấy in_features của classifier gốc rồi thay bằng Sequential có Dropout
        # timm.EfficientNetB0.classifier là Linear(1280, num_classes)
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Sequential(
            # Dropout trước Linear giúp giảm co-adaptation giữa các neurons
            # dropout_p=0.3: tắt ngẫu nhiên 30% neurons mỗi forward pass khi training
            nn.Dropout(p=dropout_p),
            # Linear giữ nguyên kiến trúc phân loại — chỉ bổ sung dropout phía trước
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        # Truyền ảnh qua toàn bộ mạng và trả về logits
        return self.model(x)


# ---------------------------------------------------------------------------
# Factory function — tạo mô hình theo tên, tránh import trực tiếp từ ngoài
# ---------------------------------------------------------------------------

def get_model(model_name: str = 'efficientnet', num_classes: int = 4,
              pretrained: bool = True, dropout_p: float = 0.3):
    """
    Hàm nhà máy: nhận tên mô hình, trả về instance tương ứng.
    Giúp train.py và predict.py không cần biết class cụ thể nào được dùng.

    Args:
        model_name:  'efficientnet' | 'densenet'
        num_classes: Số lớp đầu ra (mặc định 4 loại u não)
        pretrained:  Dùng trọng số pre-trained ImageNet (mặc định True)
        dropout_p:   Xác suất dropout trong classifier head (mặc định 0.3)
    """
    if model_name.lower() == 'densenet':
        return DenseNetModel(num_classes=num_classes, pretrained=pretrained,
                             dropout_p=dropout_p)
    elif model_name.lower() == 'efficientnet':
        return EfficientNetModel(num_classes=num_classes, pretrained=pretrained,
                                 dropout_p=dropout_p)
    else:
        # Chỉ hỗ trợ 2 kiến trúc — ném lỗi rõ ràng thay vì âm thầm fail
        raise ValueError(f"Model {model_name} not supported. Use 'densenet' or 'efficientnet'.")

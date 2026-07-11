# Cách chạy script export:
# python src/inference/export_onnx.py --pth-path models/efficientnet_best.pth

import os
import torch
import argparse
from src.training.models import get_model
from src.config import inference_cfg

# ---------------------------------------------------------------------------
# export_to_onnx — chuyển đổi model PyTorch sang định dạng ONNX
# ---------------------------------------------------------------------------

def export_to_onnx(model_name: str, pth_path: str, onnx_path: str):
    """
    Xuất model PyTorch (.pth) sang định dạng ONNX (.onnx) để deploy production.

    Lý do dùng ONNX:
    - Chạy được trên nhiều runtime: ONNX Runtime, TensorRT, OpenVINO...
    - Nhanh hơn PyTorch thuần khi inference (tối ưu graph tự động)
    - Không cần cài PyTorch trên server production

    Args:
        model_name: Tên kiến trúc ('efficientnet' hoặc 'densenet').
        pth_path:   Đường dẫn file trọng số .pth đã train.
        onnx_path:  Đường dẫn file .onnx sẽ được tạo ra.
    """
    if not os.path.exists(pth_path):
        print(f"Error: Model weights not found at {pth_path}")
        return

    print(f"[1/3] Loading PyTorch model '{model_name}' from {pth_path}...")
    # Tải model không cần pre-trained vì sẽ load trọng số từ file .pth
    model = get_model(model_name=model_name, num_classes=4, pretrained=False)
    model.load_state_dict(torch.load(pth_path, map_location='cpu'))
    model.eval()  # Tắt dropout / batch norm training mode trước khi export

    # Tạo tensor giả lập batch size=1, ảnh RGB 224x224 — dùng để trace computational graph
    print("[2/3] Generating dummy input tensor (1, 3, 224, 224)...")
    dummy_input = torch.randn(1, 3, 224, 224)

    print(f"[3/3] Exporting to ONNX format: {onnx_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,          # Nhúng trọng số vào file ONNX
        opset_version=11,            # Phiên bản ONNX opset (11 hỗ trợ rộng rãi)
        do_constant_folding=True,    # Tối ưu hóa: gấp hằng số tại compile time
        input_names=['input'],       # Đặt tên cho input node (dùng khi inference)
        output_names=['output'],     # Đặt tên cho output node
        dynamic_axes={
            'input': {0: 'batch_size'},   # Cho phép batch size thay đổi linh hoạt
            'output': {0: 'batch_size'}
        }
    )

    # So sánh kích thước file trước và sau export
    pth_size = os.path.getsize(pth_path) / (1024 * 1024)
    onnx_size = os.path.getsize(onnx_path) / (1024 * 1024)

    print("\nExport successful!")
    print(f"PyTorch size: {pth_size:.2f} MB")
    print(f"ONNX size:    {onnx_size:.2f} MB")
    print("Next step: Use onnxruntime in Production to load this model.")

# ---------------------------------------------------------------------------
# CLI entry point — chạy trực tiếp từ command line
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export PyTorch model to ONNX")
    parser.add_argument("--model-name", type=str, default="efficientnet", help="Model architecture name")
    parser.add_argument("--pth-path", type=str, default="models/efficientnet_best.pth", help="Path to .pth file")
    parser.add_argument("--onnx-path", type=str, default="models/efficientnet_best.onnx", help="Path to save .onnx file")

    args = parser.parse_args()
    export_to_onnx(args.model_name, args.pth_path, args.onnx_path)

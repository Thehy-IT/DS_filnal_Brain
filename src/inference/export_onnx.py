# Chạy script export                                                                                                    
# python src/inference/export_onnx.py --pth-path models/efficientnet_best.pth 

import os
import torch
import argparse
from src.training.models import get_model
from src.config import inference_cfg

def export_to_onnx(model_name: str, pth_path: str, onnx_path: str):
    """
    Exports a trained PyTorch model to ONNX format for high-performance inference.
    """
    if not os.path.exists(pth_path):
        print(f"Error: Model weights not found at {pth_path}")
        return

    print(f"[1/3] Loading PyTorch model '{model_name}' from {pth_path}...")
    # Load model in evaluation mode
    model = get_model(model_name=model_name, num_classes=4, pretrained=False)
    model.load_state_dict(torch.load(pth_path, map_location='cpu'))
    model.eval()

    # Create dummy input: batch_size=1, channels=3, H=224, W=224
    print("[2/3] Generating dummy input tensor (1, 3, 224, 224)...")
    dummy_input = torch.randn(1, 3, 224, 224)

    print(f"[3/3] Exporting to ONNX format: {onnx_path}...")
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_path, 
        export_params=True,
        opset_version=11, 
        do_constant_folding=True,
        input_names=['input'], 
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'}, 
            'output': {0: 'batch_size'}
        }
    )
    
    # Calculate size reduction/changes
    pth_size = os.path.getsize(pth_path) / (1024 * 1024)
    onnx_size = os.path.getsize(onnx_path) / (1024 * 1024)
    
    print("\n✅ Export successful!")
    print(f"PyTorch size: {pth_size:.2f} MB")
    print(f"ONNX size:    {onnx_size:.2f} MB")
    print("Next step: Use onnxruntime in Production to load this model.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export PyTorch model to ONNX")
    parser.add_argument("--model-name", type=str, default="efficientnet", help="Model architecture name")
    parser.add_argument("--pth-path", type=str, default="models/efficientnet_best.pth", help="Path to .pth file")
    parser.add_argument("--onnx-path", type=str, default="models/efficientnet_best.onnx", help="Path to save .onnx file")
    
    args = parser.parse_args()
    export_to_onnx(args.model_name, args.pth_path, args.onnx_path)

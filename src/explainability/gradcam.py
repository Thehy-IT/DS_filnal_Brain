import cv2
import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from PIL import Image

def generate_gradcam(model, image: Image.Image, transform, target_layer):
    """
    Generate Grad-CAM heatmap for an image.
    
    Args:
        model (nn.Module): The loaded model.
        image (PIL.Image.Image): The original PIL image.
        transform (callable): Validation transforms.
        target_layer: The target layer to compute Grad-CAM on (e.g., model.model.blocks[-1].norm1 for ViT from timm).
        
    Returns:
        np.ndarray: Image array with heatmap overlay.
    """
    # 1. Transform image to tensor
    input_tensor = transform(image).unsqueeze(0)
    
    # 2. Initialize GradCAM
    # For ViT, we need reshape_transform. pytorch_grad_cam provides this natively or we can define it.
    def reshape_transform(tensor, height=14, width=14):
        # Specific for timm vit_base_patch16_224
        result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
        result = result.transpose(2, 3).transpose(1, 2)
        return result

    cam = GradCAM(model=model, target_layers=[target_layer], reshape_transform=reshape_transform)
    
    # 3. Generate CAM
    targets = None # Automatically targets the highest scoring class
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
    
    # 4. Overlay on original image
    # Resize original image to 224x224 and normalize to [0, 1] for visualization
    rgb_img = image.resize((224, 224))
    rgb_img = np.float32(rgb_img) / 255
    
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    return visualization

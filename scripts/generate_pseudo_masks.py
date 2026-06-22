import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import glob
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import binary_fill_holes
from skimage.filters import threshold_otsu
from scipy.ndimage import distance_transform_edt
from src.core.config import MedicalSystemConfig
from src.models.classifier import initialize_2d_classifier_model
from src.xai.gradcam import VolumetricGradCAM

def preprocess_image_to_tensor(img_np):
    if len(img_np.shape) == 3:
        grayscale = np.mean(img_np, axis=-1)
    else:
        grayscale = img_np
    
    from monai.transforms import Resize
    resizer = Resize(spatial_size=(224, 224), mode="bilinear")
    resized = resizer(np.expand_dims(grayscale, axis=0)).cpu().numpy()
    
    try:
        otsu_val = threshold_otsu(resized[0])
        brain_mask = (resized[0] > otsu_val)
        brain_mask = binary_fill_holes(brain_mask)
    except ValueError:
        brain_mask = np.ones_like(resized[0], dtype=bool)
        
    stripped = resized[0] * brain_mask
    mean_val = np.mean(stripped)
    std_val = np.std(stripped)
    if std_val > 0:
        normalized = (stripped - mean_val) / std_val
    else:
        normalized = stripped
        
    rgb_tensor = np.stack([normalized, normalized, normalized], axis=0)
    return torch.as_tensor(rgb_tensor, dtype=torch.float32), brain_mask, stripped

def generate_masks():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = initialize_2d_classifier_model()
    model.load_state_dict(torch.load("data/reference/best_metric_model.pth", map_location=device))
    model.to(device)
    model.eval()
    
    gradcam_generator = VolumetricGradCAM(
        neural_network=model,
        target_network_layer="features.denseblock4.denselayer16.layers.conv2"
    )
    
    splits = {
        "train": MedicalSystemConfig.DATA_TRAINING_DIRECTORY,
        "val": MedicalSystemConfig.DATA_TESTING_DIRECTORY
    }
    
    for split_name, base_dir in splits.items():
        img_out_dir = f"data/processed/segmentation/{split_name}/images"
        mask_out_dir = f"data/processed/segmentation/{split_name}/masks"
        os.makedirs(img_out_dir, exist_ok=True)
        os.makedirs(mask_out_dir, exist_ok=True)
        
        for class_name, label_val in MedicalSystemConfig.LABEL_MAPPING.items():
            class_dir = os.path.join(base_dir, class_name)
            search_pattern = os.path.join(class_dir, "*.[jJ][pP][gG]")
            images = sorted(glob.glob(search_pattern))
            limit = 50 if split_name == "train" else 15
            images = images[:limit]
            
            for img_path in images:
                base_name = os.path.basename(img_path)
                name_without_ext = os.path.splitext(base_name)[0]
                
                img = Image.open(img_path).convert("L")
                img_np = np.array(img)
                
                input_tensor, brain_mask, stripped = preprocess_image_to_tensor(img_np)
                
                if label_val == 2:
                    mask_np = np.zeros((224, 224), dtype=np.uint8)
                else:
                    input_batch = input_tensor.unsqueeze(0).to(device)
                    activation_map = gradcam_generator.generate_activation_map(input_batch, label_val)
                    
                    dist_map = distance_transform_edt(brain_mask.astype(np.float32))
                    if dist_map.max() > 0:
                        dist_map = dist_map / dist_map.max()
                    else:
                        dist_map = np.zeros_like(dist_map)
                        
                    activation_map = activation_map * dist_map
                    if activation_map.max() > 0:
                        activation_map = (activation_map - activation_map.min()) / (activation_map.max() - activation_map.min())
                        
                    mask_np = (activation_map >= 0.5).astype(np.uint8)
                    
                img_to_save = ((stripped - stripped.min()) / (stripped.max() - stripped.min() + 1e-8) * 255.0).astype(np.uint8)
                
                img_pil = Image.fromarray(img_to_save)
                mask_pil = Image.fromarray(mask_np * 255)
                
                img_pil.save(os.path.join(img_out_dir, f"{name_without_ext}.png"))
                mask_pil.save(os.path.join(mask_out_dir, f"{name_without_ext}.png"))

if __name__ == "__main__":
    generate_masks()

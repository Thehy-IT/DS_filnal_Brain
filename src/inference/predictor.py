import numpy as np
import torch
from scipy.ndimage import zoom, binary_fill_holes
from skimage.filters import threshold_otsu
from src.xai.gradcam import VolumetricGradCAM

class VolumetricPredictor:
    def __init__(self, neural_network, checkpoint_file_path, execution_device="cpu"):
        self.neural_network = neural_network
        self.neural_network.load_state_dict(torch.load(checkpoint_file_path, map_location=execution_device))
        self.neural_network.to(execution_device)
        self.neural_network.eval()
        self.execution_device = execution_device
        
        self.gradcam_generator = VolumetricGradCAM(
            neural_network=self.neural_network,
            target_network_layer="features.denseblock4.denselayer16.layers.conv2"
        )
        
        import os
        from src.models.unet import initialize_2d_unet_model
        unet_checkpoint_path = "data/reference/best_unet_model.pth"
        if os.path.exists(unet_checkpoint_path):
            self.unet_model = initialize_2d_unet_model(input_channels_count=3, target_classes_count=1)
            self.unet_model.load_state_dict(torch.load(unet_checkpoint_path, map_location=execution_device))
            self.unet_model.to(execution_device)
            self.unet_model.eval()
        else:
            self.unet_model = None

    def preprocess_slice(self, slice_np):
        if len(slice_np.shape) == 3:
            grayscale = np.mean(slice_np, axis=-1)
        else:
            grayscale = slice_np
            
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
        return torch.as_tensor(rgb_tensor, dtype=torch.float32)

    def preprocess_slice_for_unet(self, slice_np):
        if len(slice_np.shape) == 3:
            grayscale = np.mean(slice_np, axis=-1)
        else:
            grayscale = slice_np
            
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
        
        min_val = stripped.min()
        max_val = stripped.max()
        if max_val - min_val > 0:
            normalized = (stripped - min_val) / (max_val - min_val + 1e-8)
        else:
            normalized = stripped
            
        rgb_tensor = np.stack([normalized, normalized, normalized], axis=0)
        return torch.as_tensor(rgb_tensor, dtype=torch.float32)

    def enable_dropout_at_inference(self):
        for module in self.neural_network.modules():
            if module.__class__.__name__.startswith("Dropout"):
                module.train()

    def execute_inference_on_slice(self, slice_np, run_uncertainty=True, num_passes=5):
        preprocessed_tensor = self.preprocess_slice(slice_np)
        input_tensor = preprocessed_tensor.unsqueeze(0).to(self.execution_device)
        
        self.neural_network.eval()
        with torch.no_grad():
            logits = self.neural_network(input_tensor)
            probs = torch.softmax(logits, dim=1)
            predicted_class_index = torch.argmax(probs, dim=1).item()
            confidence = probs[0, predicted_class_index].item()
            
        if predicted_class_index == 2:
            height, width = slice_np.shape[:2]
            return np.zeros((height, width), dtype=np.uint8), confidence, predicted_class_index, np.zeros((height, width), dtype=np.float32)
            
        if getattr(self, "unet_model", None) is not None:
            with torch.no_grad():
                unet_input_tensor = self.preprocess_slice_for_unet(slice_np).unsqueeze(0).to(self.execution_device)
                unet_logits = self.unet_model(unet_input_tensor)
                unet_probs = torch.sigmoid(unet_logits)
                activation_map = unet_probs.squeeze(0).squeeze(0).cpu().numpy()
        else:
            activation_map = self.gradcam_generator.generate_activation_map(input_tensor, predicted_class_index)
            
            img_np = preprocessed_tensor.mean(dim=0).cpu().numpy()
            brain_mask = (img_np > img_np.min()).astype(np.float32)
            
            from scipy.ndimage import distance_transform_edt
            dist_map = distance_transform_edt(brain_mask)
            if dist_map.max() > 0:
                dist_map = dist_map / dist_map.max()
            else:
                dist_map = np.zeros_like(dist_map)
                
            activation_map = activation_map * dist_map
            if activation_map.max() > 0:
                activation_map = (activation_map - activation_map.min()) / (activation_map.max() - activation_map.min())
                
        if run_uncertainty:
            self.enable_dropout_at_inference()
            cams_list = []
            img_np = preprocessed_tensor.mean(dim=0).cpu().numpy()
            brain_mask = (img_np > img_np.min()).astype(np.float32)
            
            from scipy.ndimage import distance_transform_edt
            dist_map = distance_transform_edt(brain_mask)
            if dist_map.max() > 0:
                dist_map = dist_map / dist_map.max()
            else:
                dist_map = np.zeros_like(dist_map)

            for _ in range(num_passes):
                cam_pass = self.gradcam_generator.generate_activation_map(input_tensor, predicted_class_index)
                cam_pass = cam_pass * dist_map
                if cam_pass.max() > 0:
                    cam_pass = (cam_pass - cam_pass.min()) / (cam_pass.max() - cam_pass.min())

                cams_list.append(cam_pass)
                
            uncertainty_map_224 = np.std(np.stack(cams_list, axis=0), axis=0)
            self.neural_network.eval()
        else:
            uncertainty_map_224 = np.zeros_like(activation_map, dtype=np.float32)
            
        zoom_h = slice_np.shape[0] / 224.0
        zoom_w = slice_np.shape[1] / 224.0
        activation_map_resized = zoom(activation_map, (zoom_h, zoom_w), order=1)
        uncertainty_map_resized = zoom(uncertainty_map_224, (zoom_h, zoom_w), order=1)
        
        if uncertainty_map_resized.max() > 0:
            uncertainty_map_resized = (uncertainty_map_resized - uncertainty_map_resized.min()) / (uncertainty_map_resized.max() - uncertainty_map_resized.min())
            
        return activation_map_resized, confidence, predicted_class_index, uncertainty_map_resized

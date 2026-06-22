import torch
from monai.visualize import GradCAM

class VolumetricGradCAM:
    def __init__(self, neural_network, target_network_layer):
        self.neural_network = neural_network
        self.gradcam_generator = GradCAM(
            nn_module=self.neural_network,
            target_layers=target_network_layer
        )

    def generate_activation_map(self, input_image_tensor, target_class_index=1):
        self.neural_network.eval()
        class_activation_map = self.gradcam_generator(
            x=input_image_tensor,
            class_idx=target_class_index
        )
        return class_activation_map.squeeze(0).squeeze(0).cpu().numpy()

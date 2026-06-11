import torch
from PIL import Image
import torch.nn.functional as F

from src.training.models import get_model
from src.preprocessing.transforms import get_valid_transforms

class TumorPredictor:
    """
    Class to handle model loading and inference for a single image.
    """
    def __init__(self, model_path: str, model_name: str = 'vit', device: str = None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
        self.transform = get_valid_transforms()
        
        # Load model
        self.model = get_model(model_name=model_name, num_classes=len(self.classes), pretrained=False)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def predict(self, image: Image.Image):
        """
        Predicts the class of the input MRI image.
        
        Args:
            image (PIL.Image.Image): Input image.
            
        Returns:
            dict: Prediction results containing class name, index, and confidence score.
        """
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
            
        idx = predicted_idx.item()
        return {
            'class_name': self.classes[idx],
            'class_idx': idx,
            'confidence': confidence.item(),
            'probabilities': probabilities.squeeze().cpu().numpy().tolist()
        }

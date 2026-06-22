import os
import sys
import torch
import numpy as np
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.classifier import initialize_2d_classifier_model
from src.inference.predictor import VolumetricPredictor

def predict_single_image(image_path):
    if not os.path.exists(image_path):
        print(f"Error: File '{image_path}' does not exist.")
        sys.exit(1)
        
    checkpoint_path = "data/reference/best_metric_model.pth"
    if not os.path.exists(checkpoint_path):
        print(f"Error: Model checkpoint '{checkpoint_path}' not found. Please train the model first.")
        sys.exit(1)
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = initialize_2d_classifier_model()
    predictor = VolumetricPredictor(model, checkpoint_path, device)
    
    img = Image.open(image_path).convert("L")
    slice_np = np.array(img)
    
    preprocessed_tensor = predictor.preprocess_slice(slice_np)
    input_tensor = preprocessed_tensor.unsqueeze(0).to(device)
    
    predictor.neural_network.eval()
    with torch.no_grad():
        logits = predictor.neural_network(input_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        
    labels = ["glioma", "meningioma", "notumor", "pituitary"]
    
    print("\n" + "=" * 50)
    print(f"PREDICTION RESULT FOR FILE: {os.path.basename(image_path)}")
    print("=" * 50)
    
    for i, label in enumerate(labels):
        percentage = probs[i] * 100
        print(f"- {label.upper():<25} : {percentage:.2f}%")
        
    predicted_idx = np.argmax(probs)
    print("=" * 50)
    print(f"DIAGNOSIS: {labels[predicted_idx].upper()} ({probs[predicted_idx]*100:.2f}%)")
    print("=" * 50 + "\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: venv\\Scripts\\python.exe scripts/predict_image.py <image_path>")
        print("Example: venv\\Scripts\\python.exe scripts/predict_image.py data/Testing/glioma/Te-gl_10.jpg")
        sys.exit(1)
        
    image_path = sys.argv[1]
    predict_single_image(image_path)

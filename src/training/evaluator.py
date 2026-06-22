import torch
import numpy as np
from src.training.metrics import compute_clinical_classification_metrics

def assess_classifier_performance(
    neural_network,
    evaluation_data_loader,
    execution_device
):
    neural_network.to(execution_device)
    neural_network.eval()
    
    all_true_labels = []
    all_predicted_labels = []
    
    with torch.no_grad():
        for batch_records in evaluation_data_loader:
            input_images = batch_records["image"].to(execution_device)
            target_labels = batch_records["label"].to(execution_device)
            
            model_predictions = neural_network(input_images)
            predicted_classes = torch.argmax(model_predictions, dim=1)
            
            all_true_labels.extend(target_labels.cpu().numpy())
            all_predicted_labels.extend(predicted_classes.cpu().numpy())
            
    return compute_clinical_classification_metrics(all_true_labels, all_predicted_labels)

def compute_stochastic_gradcam_uncertainty(
    neural_network,
    input_image_tensor,
    gradcam_generator_instance,
    target_class_index,
    stochastic_iterations=10,
    execution_device="cpu"
):
    neural_network.to(execution_device)
    neural_network.train()
    
    gradcam_activation_maps = []
    
    for _ in range(stochastic_iterations):
        activation_map = gradcam_generator_instance.generate_activation_map(
            input_image_tensor.to(execution_device), 
            target_class_index
        )
        gradcam_activation_maps.append(activation_map)
        
    gradcam_tensor = np.stack(gradcam_activation_maps, axis=0)
    expected_activation = np.mean(gradcam_tensor, axis=0)
    spatial_variance = np.var(gradcam_tensor, axis=0)
    
    return expected_activation, spatial_variance

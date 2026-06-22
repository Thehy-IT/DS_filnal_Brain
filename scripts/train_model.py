import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
from src.core.config import MedicalSystemConfig
from src.data.dataset import retrieve_dataset_records, build_clinical_image_dataset
from src.data.pipeline import construct_training_pipeline, construct_validation_pipeline
from src.models.classifier import initialize_2d_classifier_model
from src.training.losses import ClassificationLoss
from src.training.trainer import execute_classifier_training

def execute_model_training_pipeline():
    computation_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    training_records = retrieve_dataset_records(
        base_directory=MedicalSystemConfig.DATA_TRAINING_DIRECTORY,
        label_mapping=MedicalSystemConfig.LABEL_MAPPING
    )
    validation_records = retrieve_dataset_records(
        base_directory=MedicalSystemConfig.DATA_TESTING_DIRECTORY,
        label_mapping=MedicalSystemConfig.LABEL_MAPPING
    )
    
    training_transform_pipeline = construct_training_pipeline()
    validation_transform_pipeline = construct_validation_pipeline()
    
    training_dataset = build_clinical_image_dataset(training_records, training_transform_pipeline)
    validation_dataset = build_clinical_image_dataset(validation_records, validation_transform_pipeline)
    
    training_loader = DataLoader(
        training_dataset, 
        batch_size=MedicalSystemConfig.BATCH_SIZE, 
        shuffle=True
    )
    validation_loader = DataLoader(
        validation_dataset, 
        batch_size=1, 
        shuffle=False
    )
    
    classification_model = initialize_2d_classifier_model()
    
    checkpoint_file_path = "data/reference/best_metric_model.pth"
    if os.path.exists(checkpoint_file_path):
        try:
            classification_model.load_state_dict(torch.load(checkpoint_file_path, map_location=computation_device))
            print("Checkpoint found. Resuming training from the saved weights.")
        except Exception:
            pass
            
    loss_function = ClassificationLoss()
    optimizer_algorithm = torch.optim.Adam(
        classification_model.parameters(), 
        lr=MedicalSystemConfig.LEARNING_RATE
    )
    
    execute_classifier_training(
        neural_network=classification_model,
        training_data_loader=training_loader,
        validation_data_loader=validation_loader,
        loss_criterion=loss_function,
        optimizer_instance=optimizer_algorithm,
        total_training_epochs=MedicalSystemConfig.MAX_EPOCHS,
        execution_device=computation_device,
        checkpoint_output_path="data/reference/best_metric_model.pth"
    )

if __name__ == "__main__":
    execute_model_training_pipeline()

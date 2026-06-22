import torch
import wandb
from src.training.metrics import compute_clinical_classification_metrics

def execute_classifier_training(
    neural_network,
    training_data_loader,
    validation_data_loader,
    loss_criterion,
    optimizer_instance,
    total_training_epochs,
    execution_device,
    checkpoint_output_path,
    logging_project_name="brain_tumor_classification"
):
    wandb.init(project=logging_project_name, mode="offline")
    wandb.watch(neural_network, log_freq=100)
    
    neural_network.to(execution_device)
    lowest_validation_loss = float("inf")
    
    for epoch_index in range(total_training_epochs):
        neural_network.train()
        accumulated_training_loss = 0.0
        
        for batch_index, batch_records in enumerate(training_data_loader):
            input_images = batch_records["image"].to(execution_device)
            target_labels = batch_records["label"].to(execution_device)
            
            optimizer_instance.zero_grad()
            model_predictions = neural_network(input_images)
            loss_value = loss_criterion(model_predictions, target_labels)
            loss_value.backward()
            optimizer_instance.step()
            
            accumulated_training_loss += loss_value.item()
            
            if (batch_index + 1) % 10 == 0:
                print(f"Epoch {epoch_index + 1}/{total_training_epochs} - Batch {batch_index + 1}/{len(training_data_loader)} - Loss: {loss_value.item():.4f}")
            
        average_training_loss = accumulated_training_loss / len(training_data_loader)
        
        neural_network.eval()
        accumulated_validation_loss = 0.0
        all_true_labels = []
        all_predicted_labels = []
        
        with torch.no_grad():
            for batch_records in validation_data_loader:
                input_images = batch_records["image"].to(execution_device)
                target_labels = batch_records["label"].to(execution_device)
                
                model_predictions = neural_network(input_images)
                loss_value = loss_criterion(model_predictions, target_labels)
                accumulated_validation_loss += loss_value.item()
                
                predicted_classes = torch.argmax(model_predictions, dim=1)
                
                all_true_labels.extend(target_labels.cpu().numpy())
                all_predicted_labels.extend(predicted_classes.cpu().numpy())
                
        average_validation_loss = accumulated_validation_loss / len(validation_data_loader)
        
        evaluation_metrics = compute_clinical_classification_metrics(all_true_labels, all_predicted_labels)
        
        wandb.log({
            "epoch": epoch_index,
            "train_loss": average_training_loss,
            "validation_loss": average_validation_loss,
            "accuracy": evaluation_metrics["accuracy"],
            "f1_score": evaluation_metrics["f1_score"],
            "sensitivity": evaluation_metrics["sensitivity"],
            "specificity": evaluation_metrics["specificity"]
        })
        
        print(f"Epoch {epoch_index + 1}/{total_training_epochs} - Train Loss: {average_training_loss:.4f} - Val Loss: {average_validation_loss:.4f} - Accuracy: {evaluation_metrics['accuracy']:.4f} - F1: {evaluation_metrics['f1_score']:.4f}")
        
        if average_validation_loss < lowest_validation_loss:
            lowest_validation_loss = average_validation_loss
            torch.save(neural_network.state_dict(), checkpoint_output_path)
            
    wandb.finish()

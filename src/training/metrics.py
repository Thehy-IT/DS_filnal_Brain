import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

def compute_clinical_classification_metrics(true_labels, predicted_labels):
    accuracy_value = accuracy_score(true_labels, predicted_labels)
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        true_labels, 
        predicted_labels, 
        average="macro", 
        zero_division=0
    )
    
    matrix = confusion_matrix(true_labels, predicted_labels)
    classes_count = len(matrix)
    sensitivity_records = []
    specificity_records = []
    
    for class_index in range(classes_count):
        true_positives = matrix[class_index, class_index]
        false_negatives = np.sum(matrix[class_index, :]) - true_positives
        false_positives = np.sum(matrix[:, class_index]) - true_positives
        true_negatives = np.sum(matrix) - (true_positives + false_negatives + false_positives)
        
        sensitivity = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        specificity = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0.0
        
        sensitivity_records.append(sensitivity)
        specificity_records.append(specificity)
        
    return {
        "accuracy": accuracy_value,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "sensitivity": np.mean(sensitivity_records),
        "specificity": np.mean(specificity_records)
    }

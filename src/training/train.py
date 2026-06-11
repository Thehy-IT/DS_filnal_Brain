import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

from src.preprocessing.dataset import BrainTumorDataset
from src.preprocessing.transforms import get_train_transforms, get_valid_transforms
from src.training.models import get_model

def train_model(data_dir: str, model_name: str = 'vit', batch_size: int = 32, epochs: int = 10, lr: float = 1e-4, save_dir: str = 'models'):
    """
    Main training loop for the brain tumor classification model.
    """
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Prepare Data
    print("Preparing data...")
    # For simplicity, assuming all data is in one folder and we split it here.
    # If the dataset is already split into train/test folders, we should load them separately.
    full_dataset = BrainTumorDataset(root_dir=data_dir, transform=get_train_transforms())
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Override validation dataset transform
    val_dataset.dataset.transform = get_valid_transforms()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # 2. Initialize Model
    print(f"Initializing {model_name} model...")
    model = get_model(model_name=model_name, num_classes=4, pretrained=True)
    model = model.to(device)

    # 3. Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)

    # 4. Training Loop
    best_val_acc = 0.0
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        model.train()
        train_loss = 0.0
        
        for images, labels in tqdm(train_loader, desc="Training"):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            
        train_loss = train_loss / len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validation"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = accuracy_score(all_labels, all_preds)
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # Learning rate scheduler
        scheduler.step(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(save_dir, f"{model_name}_best.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Saved new best model with accuracy {val_acc:.4f}")

    print("\nTraining Complete.")
    print("Classification Report on Validation Set:")
    print(classification_report(all_labels, all_preds, target_names=['glioma', 'meningioma', 'notumor', 'pituitary']))

if __name__ == "__main__":
    # Example usage:
    # train_model(data_dir="data/Training", model_name="vit", epochs=10)
    pass

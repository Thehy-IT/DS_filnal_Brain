import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import glob
import torch
from torch.utils.data import DataLoader
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityd,
    RandFlipd,
    RandRotate90d,
    CastToTyped
)
from monai.losses import DiceCELoss
from monai.data import Dataset
from src.data.transforms import EnsureRGBd
from src.models.unet import initialize_2d_unet_model

def train():
    batch_size = 16
    learning_rate = 0.0005
    epochs = 40
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_images = sorted(glob.glob("data/processed/segmentation/train/images/*.png"))
    train_masks = sorted(glob.glob("data/processed/segmentation/train/masks/*.png"))
    train_files = [{"image": img, "label": mask} for img, mask in zip(train_images, train_masks)]
    
    val_images = sorted(glob.glob("data/processed/segmentation/val/images/*.png"))
    val_masks = sorted(glob.glob("data/processed/segmentation/val/masks/*.png"))
    val_files = [{"image": img, "label": mask} for img, mask in zip(val_images, val_masks)]
    
    train_transforms = Compose([
        LoadImaged(keys=["image", "label"], image_only=True),
        EnsureChannelFirstd(keys=["image", "label"]),
        EnsureRGBd(keys=["image"]),
        ScaleIntensityd(keys=["image", "label"]),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
        CastToTyped(keys=["image", "label"], dtype=torch.float32)
    ])
    
    val_transforms = Compose([
        LoadImaged(keys=["image", "label"], image_only=True),
        EnsureChannelFirstd(keys=["image", "label"]),
        EnsureRGBd(keys=["image"]),
        ScaleIntensityd(keys=["image", "label"]),
        CastToTyped(keys=["image", "label"], dtype=torch.float32)
    ])
    
    train_ds = Dataset(data=train_files, transform=train_transforms)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    
    val_ds = Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
    
    model = initialize_2d_unet_model(input_channels_count=3, target_classes_count=1)
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = DiceCELoss(sigmoid=True)
    
    best_metric = -1
    best_metric_epoch = -1
    os.makedirs("data/reference", exist_ok=True)
    checkpoint_path = "data/reference/best_unet_model.pth"
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        epoch_loss /= step
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {epoch_loss:.4f}")
        
        model.eval()
        dice_sum = 0
        val_step = 0
        with torch.no_grad():
            for val_data in val_loader:
                val_inputs, val_labels = val_data["image"].to(device), val_data["label"].to(device)
                val_outputs = model(val_inputs)
                val_outputs_sig = torch.sigmoid(val_outputs)
                val_outputs_bin = (val_outputs_sig > 0.5).float()
                
                intersection = torch.sum(val_outputs_bin * val_labels)
                union = torch.sum(val_outputs_bin) + torch.sum(val_labels)
                if union > 0:
                    dice = (2.0 * intersection) / union
                else:
                    dice = 1.0 if torch.sum(val_labels) == 0 else 0.0
                    
                dice_sum += dice.item()
                val_step += 1
                
        avg_dice = dice_sum / val_step
        print(f"Validation Dice Score: {avg_dice:.4f}")
        
        if avg_dice > best_metric:
            best_metric = avg_dice
            best_metric_epoch = epoch + 1
            torch.save(model.state_dict(), checkpoint_path)
            print("Saved new best model checkpoint.")
            
    print(f"Training completed. Best Dice: {best_metric:.4f} at epoch {best_metric_epoch}")

if __name__ == "__main__":
    train()

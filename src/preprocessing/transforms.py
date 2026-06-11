from torchvision import transforms

def get_train_transforms(image_size: int = 224):
    """
    Returns data transformations for the training set.
    Includes data augmentation to prevent overfitting.
    
    Args:
        image_size (int): Target size for resizing images. Default is 224 (standard for ViT).
        
    Returns:
        torchvision.transforms.Compose
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        # ImageNet normalization stats
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])

def get_valid_transforms(image_size: int = 224):
    """
    Returns data transformations for the validation/test set.
    Only includes resizing and normalization.
    
    Args:
        image_size (int): Target size for resizing images. Default is 224.
        
    Returns:
        torchvision.transforms.Compose
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])

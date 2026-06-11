import os
from PIL import Image
from torch.utils.data import Dataset

class BrainTumorDataset(Dataset):
    """
    Custom PyTorch Dataset for Brain Tumor MRI Images.
    Expects data to be organized in folders by class:
    data/
        train/
            glioma/
            meningioma/
            pituitary/
            notumor/
    """
    def __init__(self, root_dir: str, transform=None):
        """
        Args:
            root_dir (str): Path to the directory containing class folders.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.image_paths = []
        self.labels = []

        # Load all image paths and labels
        for cls_name in self.classes:
            cls_dir = os.path.join(root_dir, cls_name)
            if not os.path.isdir(cls_dir):
                continue
                
            for img_name in os.listdir(cls_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(cls_dir, img_name))
                    self.labels.append(self.class_to_idx[cls_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image and convert to RGB
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a blank image in case of error to prevent crashing, 
            # or handle more gracefully
            image = Image.new('RGB', (224, 224))

        if self.transform:
            image = self.transform(image)

        return image, label

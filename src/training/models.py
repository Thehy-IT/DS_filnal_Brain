import torch
import torch.nn as nn
import timm

class BaselineCNN(nn.Module):
    """
    A simple baseline Convolutional Neural Network for benchmarking.
    """
    def __init__(self, num_classes: int = 4):
        super(BaselineCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class EfficientNetModel(nn.Module):
    """
    EfficientNetB0 wrapper using timm library.
    """
    def __init__(self, num_classes: int = 4, pretrained: bool = True):
        super(EfficientNetModel, self).__init__()
        # Load pre-trained EfficientNet model
        self.model = timm.create_model('efficientnet_b0', pretrained=pretrained, num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

def get_model(model_name: str = 'efficientnet', num_classes: int = 4, pretrained: bool = True):
    """
    Factory function to get the specified model.
    """
    if model_name.lower() == 'cnn':
        return BaselineCNN(num_classes=num_classes)
    elif model_name.lower() == 'efficientnet':
        return EfficientNetModel(num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(f"Model {model_name} not supported. Use 'cnn' or 'efficientnet'.")

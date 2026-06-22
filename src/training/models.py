import torch
import torch.nn as nn
import timm

class DenseNetModel(nn.Module):
    """
    DenseNet121 wrapper using timm library.
    Used as the secondary main model for deep medical benchmark comparison.
    """
    def __init__(self, num_classes: int = 4, pretrained: bool = True):
        super(DenseNetModel, self).__init__()
        # Load pre-trained DenseNet121 model
        self.model = timm.create_model('densenet121', pretrained=pretrained, num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

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
    if model_name.lower() == 'densenet':
        return DenseNetModel(num_classes=num_classes, pretrained=pretrained)
    elif model_name.lower() == 'efficientnet':
        return EfficientNetModel(num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(f"Model {model_name} not supported. Use 'densenet' or 'efficientnet'.")

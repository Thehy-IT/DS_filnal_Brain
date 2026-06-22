import torch.nn as nn
from monai.losses import FocalLoss

class ClassificationLoss(nn.Module):
    def __init__(self, use_focal_loss_flag=True, focal_gamma_value=2.0):
        super().__init__()
        self.cross_entropy_estimator = nn.CrossEntropyLoss()
        self.focal_estimator = FocalLoss(gamma=focal_gamma_value, to_onehot_y=True)
        self.use_focal_loss_flag = use_focal_loss_flag

    def forward(self, prediction_logits, target_labels):
        if self.use_focal_loss_flag:
            return self.focal_estimator(prediction_logits, target_labels)
        return self.cross_entropy_estimator(prediction_logits, target_labels)

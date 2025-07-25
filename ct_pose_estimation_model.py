# CT POSE ESTIMATION CNN REGRESSION MODEL
# created 07/14/2025

import torch
import torch.nn as nn
import torchvision.models as models 
from ct_sweep_dataset import CTSweepDataset, parse_dataset
import time

########## REGRESSION MODEL ##########

class PoseEstRegression(nn.Module):
    """
    Regression Model: goal is to take in single grayscale CT image and output a predicted rotation angle. 
    ResNet18 (Residual Network 18): 18-layer CNN
        - Initial convolution + pooling
        - 4 residual blocks
        - Global average pooling
        - Fully connected layer
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # load standard ResNet18 architecture with pretrained weights
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # modify first convolutional layer to accept grayscale images
        self.backbone.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        # replace final convolutional layer to output one value
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 1)

    def forward(self, x):
        """
        Removes singleton dimension so that the output size is the batch size. 
        """
        return self.backbone(x).squeeze(1)
    
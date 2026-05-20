# model.py
import torch.nn as nn
from torchvision import models

def get_chest_resnet(num_classes):
    model = models.resnet18(weights=None)  # NOT pretrained

    # Change first conv layer to accept 1-channel (grayscale)
    model.conv1 = nn.Conv2d(
        1, 64, kernel_size=7, stride=2, padding=3, bias=False
    )

    # Change final classification layer
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model
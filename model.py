import torch
import torch.nn as nn
from torchvision import models

class DenseNet121Pneumonia(nn.Module):
    def __init__(self, pretrained=True):
        super(DenseNet121Pneumonia, self).__init__()
        self.model = models.densenet121(pretrained=pretrained)

        # Change first conv layer to accept 1 channel instead of 3
        self.model.features.conv0 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        # Modify classifier for binary classification
        self.model.classifier = nn.Linear(in_features=1024, out_features=2)

    def forward(self, x):
        return self.model(x)

def get_model(device='cuda'):
    """
    Returns the model initialized and moved to the specified device.
    """
    model = DenseNet121Pneumonia(pretrained=True)

    # Move model to GPU if available
    model = model.to(device)

    return model


# In[ ]:





import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class CPNet(nn.Module):
    
    def __init__(self, num_class, pretrained=True):
        super().__init__()

        self.augmentation = nn.Sequential(
                    nn.Dropout(0.2),
                    nn.Linear(1280, num_class)
                )
        self.efficient_net = efficientnet_b0(
                    weights=EfficientNet_B0_Weights.DEFAULT if pretrained else None
                )
        self.efficient_net.classifier = self.augmentation

    def forward(self, x):
        return self.efficient_net(x)

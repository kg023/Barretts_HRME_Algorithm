import torch
import torch.nn as nn
from torchvision.models import resnet152, densenet121
from Models.cbam import CBAM  # Assuming your full CBAM code is in cbam.py

class ResNet152(nn.Module):
    def __init__(self, dropout=0.5):
        super(ResNet152, self).__init__()
        self.base_model = resnet152(pretrained=True)
        num_ftrs = self.base_model.fc.in_features
        self.base_model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.base_model(x)


class ResNet152_CBAM(nn.Module):
    def __init__(self, pretrained=True, dropout=0.5):
        super(ResNet152_CBAM, self).__init__()
        base_model = resnet152(pretrained=pretrained)
        
        # Keep all layers except the classifier
        self.features = nn.Sequential(*list(base_model.children())[:-2])  # Output is (B, 2048, H, W)
        
        self.cbam = CBAM(gate_channels=2048)  # 2048 is output channels from last conv layer

        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # Global pooling
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 1),
            nn.Sigmoid()  # For binary classification
        )

    def forward(self, x):
        x = self.features(x)
        x = self.cbam(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x


class DenseNet121(nn.Module):
    def __init__(self, dropout=0.5):
        super(DenseNet121, self).__init__()
        self.base_model = densenet121(pretrained=True)

        # DenseNet has a 'classifier' instead of 'fc'
        num_ftrs = self.base_model.classifier.in_features

        self.base_model.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 1),         # For binary classification
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.base_model(x)
    

class DenseNet121_CBAM(nn.Module):
    def __init__(self, pretrained=True, dropout=0.5):
        super(DenseNet121_CBAM, self).__init__()

        base_model = densenet121(pretrained=pretrained)

        # Keep convolutional backbone
        self.features = base_model.features  # Outputs (B, 1024, H/32, W/32)

        # Add CBAM after feature extractor
        self.cbam = CBAM(gate_channels=1024)  # 1024 is final channel size for DenseNet121

        # Global pooling + classifier
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)     # (B, 1024, H/32, W/32)
        x = self.cbam(x)         # CBAM attention
        x = self.pool(x)         # (B, 1024, 1, 1)
        x = self.classifier(x)   # (B, 1)
        return x
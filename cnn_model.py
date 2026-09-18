import torch
import torch.nn as nn
import torchvision.models as models

class AudioCNN(nn.Module):
    """
    High-Performance Deep Convolutional Neural Network for Bioacoustic Avian Disease Detection.
    Built on a transfer-learning ResNet backbone adapted for 1-channel Mel-Spectrograms.
    """
    def __init__(self, num_classes: int = 3, backbone: str = "resnet34"):
        super(AudioCNN, self).__init__()

        if backbone == "resnet50":
            base_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            num_ftrs = base_model.fc.in_features
        else:
            base_model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
            num_ftrs = base_model.fc.in_features

        # Adapt conv1 to accept 1-channel Mel-spectrogram instead of 3-channel RGB
        original_conv1 = base_model.conv1
        self.conv1 = nn.Conv2d(
            1,
            original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=False
        )

        # Initialize single-channel conv1 weights by averaging pretrained 3-channel RGB weights
        with torch.no_grad():
            self.conv1.weight = nn.Parameter(original_conv1.weight.mean(dim=1, keepdim=True))

        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool

        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Classification Head with Regularization
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.4),
            nn.Linear(num_ftrs, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (Batch, 1, Mels, Time)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        logits = self.classifier(x)
        return logits
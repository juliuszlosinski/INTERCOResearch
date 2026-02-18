import torch.nn as nn
from torchvision.models import resnet18


class ResNet18YOLOv8m(nn.Module):
    """
    ResNet18 backbone customized for YOLOv8 medium (m) size.
    Feature maps returned (adapted for YOLOv8m):
    - P3: 256 channels → mid-level features
    - P4: 512 channels → deeper features
    - P5: 1024 channels → high-level features
    Compatible with YOLOv8 neck/head (FPN/PAN) for medium model.
    """

    def __init__(self, pretrained=True):
        super().__init__()

        # 1. Load ResNet18 from torchvision.
        m = resnet18(weights="IMAGENET1K_V1" if pretrained else None)

        # 2. Stem.
        # Initial layers: reduce spatial resolution and extract low-level features
        self.stem = nn.Sequential(
            m.conv1,  # 7x7 conv, stride=2
            m.bn1,  # batch normalization
            m.relu,  # ReLU activation
            m.maxpool,  # max pooling, stride=2 → total stride=4
        )

        # 3. ResNet18 stages.
        # ResNet18 layers (bottleneck blocks)
        self.layer1 = m.layer1  # stride 4 → contributes to P3
        self.layer2 = m.layer2  # stride 8 → mid-level features
        self.layer3 = m.layer3  # stride 16 → deeper features
        self.layer4 = m.layer4  # stride 32 → high-level features

        # 4. Channel adapters.
        # Adjust channels for YOLOv8m neck
        self.c3 = nn.Conv2d(128, 256, 1)  # P3 → 256 channels
        self.c4 = nn.Conv2d(256, 512, 1)  # P4 → 512 channels
        self.c5 = nn.Conv2d(512, 1024, 1)  # P5 → 1024 channels

    def forward(self, x):
        """
        Forward pass through the backbone.

        Args:
            x (torch.Tensor): input image tensor [B, 3, H, W]

        Returns:
            tuple: (p3, p4, p5) feature maps for YOLOv8m neck
        """

        # 1. Stem.
        x = self.stem(x)  # output after stem → low-level features

        # 2. ResNet18 layers.
        c1 = self.layer1(x)  # low-level features
        c2 = self.layer2(c1)  # mid-level features (128 channels)
        c3 = self.layer3(c2)  # deeper features (256 channels)
        c4 = self.layer4(c3)  # high-level features (512 channels)

        # 3. Channel adjustment for YOLOv8m neck.
        p3 = self.c3(c2)  # P3 → 256 channels
        p4 = self.c4(c3)  # P4 → 512 channels
        p5 = self.c5(c4)  # P5 → 1024 channels

        return p3, p4, p5
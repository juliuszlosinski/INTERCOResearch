import torch.nn as nn
from torchvision.models import resnet34


class ResNet34YOLOv8m(nn.Module):
    """
    ResNet34 backbone customized for YOLOv8 medium (m) size.

    Feature maps returned:
    - P3: 256 channels → mid-level features
    - P4: 512 channels → deeper features
    - P5: 1024 channels → high-level features

    Compatible with YOLOv8 neck/head (FPN/PAN) for medium model.
    """

    def __init__(self, pretrained=True):
        super().__init__()

        # 1. Load ResNet34
        m = resnet34(weights="IMAGENET1K_V1" if pretrained else None)

        # 2. Stem
        self.stem = nn.Sequential(
            m.conv1,  # 7x7 conv, stride=2
            m.bn1,
            m.relu,
            m.maxpool,  # total stride=4
        )

        # 3. ResNet34 stages
        self.layer1 = m.layer1  # stride 4, 64 channels
        self.layer2 = m.layer2  # stride 8, 128 channels
        self.layer3 = m.layer3  # stride 16, 256 channels
        self.layer4 = m.layer4  # stride 32, 512 channels

        # 4. Channel adapters for YOLOv8m
        self.c3 = nn.Conv2d(128, 256, kernel_size=1)  # P3
        self.c4 = nn.Conv2d(256, 512, kernel_size=1)  # P4
        self.c5 = nn.Conv2d(512, 1024, kernel_size=1)  # P5

    def forward(self, x):
        """
        Args:
            x (Tensor): [B, 3, H, W]

        Returns:
            tuple(Tensor): (p3, p4, p5)
        """

        # Stem
        x = self.stem(x)

        # ResNet stages
        c1 = self.layer1(x)
        c2 = self.layer2(c1)  # 128 ch
        c3 = self.layer3(c2)  # 256 ch
        c4 = self.layer4(c3)  # 512 ch

        # YOLOv8 feature maps
        p3 = self.c3(c2)  # 256 ch, stride 8
        p4 = self.c4(c3)  # 512 ch, stride 16
        p5 = self.c5(c4)  # 1024 ch, stride 32

        return p3, p4, p5

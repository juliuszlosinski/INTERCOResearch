import torch.nn as nn
from torchvision.models import vgg16


class VGG16YOLOv8m(nn.Module):
    """
    VGG16 backbone customized for YOLOv8 medium (m) size.

    Feature maps returned:
    - P3: 256 channels (stride 8)
    - P4: 512 channels (stride 16)
    - P5: 1024 channels (stride 32)

    Compatible with YOLOv8 neck/head (FPN/PAN).
    """

    def __init__(self, pretrained=True):
        super().__init__()

        vgg = vgg16(weights="IMAGENET1K_V1" if pretrained else None)

        # VGG16 features (conv + pool)
        features = vgg.features

        # VGG16 structure (important pooling points):
        # pool1 → stride 2
        # pool2 → stride 4
        # pool3 → stride 8   → P3
        # pool4 → stride 16  → P4
        # pool5 → stride 32  → P5

        self.stage1 = features[:10]  # up to pool2
        self.stage2 = features[10:17]  # pool3
        self.stage3 = features[17:24]  # pool4
        self.stage4 = features[24:31]  # pool5

        # Channel adapters for YOLOv8m
        self.c3 = nn.Conv2d(256, 256, kernel_size=1)  # P3
        self.c4 = nn.Conv2d(512, 512, kernel_size=1)  # P4
        self.c5 = nn.Conv2d(512, 1024, kernel_size=1)  # P5

    def forward(self, x):
        """
        Args:
            x (Tensor): [B, 3, H, W]

        Returns:
            tuple(Tensor): (p3, p4, p5)
        """

        x = self.stage1(x)
        c3 = self.stage2(x)  # stride 8, 256 ch
        c4 = self.stage3(c3)  # stride 16, 512 ch
        c5 = self.stage4(c4)  # stride 32, 512 ch

        p3 = self.c3(c3)  # 256 ch
        p4 = self.c4(c4)  # 512 ch
        p5 = self.c5(c5)  # 1024 ch

        return p3, p4, p5
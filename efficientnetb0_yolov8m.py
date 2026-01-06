import torch.nn as nn
from torchvision.models import efficientnet_b0


class EfficientNetB0YOLOv8m(nn.Module):
    """
    EfficientNet-B0 backbone customized for YOLOv8 medium (m) size.

    Feature maps returned:
    - P3: 256 channels (stride 8)
    - P4: 512 channels (stride 16)
    - P5: 1024 channels (stride 32)

    Compatible with YOLOv8 neck/head (FPN/PAN).
    """

    def __init__(self, pretrained=True):
        super().__init__()

        m = efficientnet_b0(weights="IMAGENET1K_V1" if pretrained else None)

        # EfficientNet features
        # Strides progression:
        # stem -> 2
        # block2 -> 4
        # block3 -> 8   → P3
        # block5 -> 16  → P4
        # block7 -> 32  → P5

        self.stem = m.features[0]  # stride 2
        self.stage1 = m.features[1]  # stride 2 → total 4
        self.stage2 = m.features[2]  # stride 4 → total 8
        self.stage3 = m.features[3:5]  # stride 8 → total 16
        self.stage4 = m.features[5:]  # stride 16 → total 32

        # Channel adapters for YOLOv8m
        self.c3 = nn.Conv2d(40, 256, kernel_size=1)  # P3
        self.c4 = nn.Conv2d(112, 512, kernel_size=1)  # P4
        self.c5 = nn.Conv2d(320, 1024, kernel_size=1)  # P5

    def forward(self, x):
        """
        Args:
            x (Tensor): [B, 3, H, W]

        Returns:
            tuple(Tensor): (p3, p4, p5)
        """

        x = self.stem(x)
        x = self.stage1(x)
        c3 = self.stage2(x)  # stride 8, 40 ch
        c4 = self.stage3(c3)  # stride 16, 112 ch
        c5 = self.stage4(c4)  # stride 32, 320 ch

        p3 = self.c3(c3)  # 256 ch
        p4 = self.c4(c4)  # 512 ch
        p5 = self.c5(c5)  # 1024 ch

        return p3, p4, p5

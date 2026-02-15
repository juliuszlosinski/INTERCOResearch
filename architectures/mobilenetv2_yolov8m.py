import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2


class MobileNetv2YOLOv8m(nn.Module):
    """
    MobileNetV2 backbone customized for YOLOv8 medium (m) size.

    Feature maps returned:
    - P3: 24 channels → mid-level features (after layer 4)
    - P4: 32 channels → deeper features (after layer 7)
    - P5: 96 channels → high-level features (after layer 14)

    Compatible with YOLOv8 neck/head (FPN/PAN) for medium model.
    """

    def __init__(self, pretrained=True):
        super().__init__()

        # Load MobileNetV2
        m = mobilenet_v2(weights="IMAGENET1K_V1" if pretrained else None)
        self.features = m.features  # all conv layers

        # MobileNetV2 has layers: features[0]..features[18]
        # We'll pick outputs from certain indices for P3/P4/P5
        self.stage_indices = [3, 6, 13]  # typical choice for mid/deep/high

        # 1x1 channel adapters for YOLOv8m neck
        self.c3 = nn.Conv2d(24, 256, kernel_size=1)  # P3
        self.c4 = nn.Conv2d(32, 512, kernel_size=1)  # P4
        self.c5 = nn.Conv2d(96, 1024, kernel_size=1)  # P5

    def forward(self, x):
        """
        Args:
            x (Tensor): [B, 3, H, W]

        Returns:
            tuple(Tensor): (p3, p4, p5)
        """
        outputs = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.stage_indices:
                outputs.append(x)

        # outputs[0] = P3, outputs[1] = P4, outputs[2] = P5
        p3 = self.c3(outputs[0])
        p4 = self.c4(outputs[1])
        p5 = self.c5(outputs[2])

        return p3, p4, p5

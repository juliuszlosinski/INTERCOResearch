import torch.nn as nn
from torchvision.models import alexnet


class AlexNetYOLOv8m(nn.Module):
    """
    AlexNet backbone compatible with YOLOv8m neck/head.
    Produces feature maps with:
    - P3: 128 channels
    - P4: 256 channels
    - P5: 512 channels
    """

    def __init__(self, pretrained=True):
        super().__init__()
        # 1. Load AlexNet with or without pretrained weights.
        m = alexnet(weights="DEFAULT" if pretrained else None)

        # 2. AlexNet feature extractor (convolutions + ReLU + max pooling).
        self.features = m.features

        # 3. Split AlexNet features into three stages to match P3, P4, P5.
        self.slice1 = nn.Sequential(*self.features[:6])  # P3 feature map
        self.slice2 = nn.Sequential(*self.features[6:10])  # P4 feature map
        self.slice3 = nn.Sequential(*self.features[10:])  # P5 feature map

        # 4. Channel adapters to match YOLOv8m expected feature sizes.
        self.c3 = nn.Conv2d(192, 128, 1)  # adapt P3 to 128 channels
        self.c4 = nn.Conv2d(384, 256, 1)  # adapt P4 to 256 channels
        self.c5 = nn.Conv2d(256, 512, 1)  # adapt P5 to 512 channels

    def forward(self, x):
        # 5. Forward pass through AlexNet slices.
        f1 = self.slice1(x)
        f2 = self.slice2(f1)
        f3 = self.slice3(f2)

        # 6. Apply channel adapters.
        p3 = self.c3(f1)
        p4 = self.c4(f2)
        p5 = self.c5(f3)

        # 7. Return feature maps for YOLOv8m neck
        return p3, p4, p5
import torch.nn as nn
from torchvision.models import resnet18


def conv_bn_act(c1, c2, k=1, s=1, p=None):
    """YOLO-style Conv-BN-SiLU"""
    if p is None:
        p = k // 2
    return nn.Sequential(
        nn.Conv2d(c1, c2, k, s, p, bias=False),
        nn.BatchNorm2d(c2),
        nn.SiLU(inplace=True),
    )


class AdjustedResNet18YOLOv8m(nn.Module):
    """
    ResNet-18 backbone compatible with DEFAULT YOLOv8 YAML.
    Outputs exactly 3 feature maps:
    - P3: stride 8   → 256 ch
    - P4: stride 16  → 512 ch
    - P5: stride 32  → 1024 ch
    """

    def __init__(self, pretrained=True):
        super().__init__()

        m = resnet18(weights="IMAGENET1K_V1" if pretrained else None)

        # --- STEM (DO NOT REMOVE MAXPOOL!) ---
        self.stem = nn.Sequential(
            m.conv1,  # 7x7, s=2 → stride 2
            m.bn1,
            m.relu,
            m.maxpool,  # s=2 → stride 4 (CRITICAL)
        )

        # --- ResNet stages ---
        self.layer1 = m.layer1  # stride 4   (64 ch)
        self.layer2 = m.layer2  # stride 8   (128 ch)
        self.layer3 = m.layer3  # stride 16  (256 ch)
        self.layer4 = m.layer4  # stride 32  (512 ch)

        # --- YOLO-compatible channel adapters ---
        self.p3 = conv_bn_act(128, 256, 1)
        self.p4 = conv_bn_act(256, 512, 1)
        self.p5 = conv_bn_act(512, 1024, 1)

    def forward(self, x):
        x = self.stem(x)

        x = self.layer1(x)  # stride 4
        c3 = self.layer2(x)  # stride 8
        c4 = self.layer3(c3)  # stride 16
        c5 = self.layer4(c4)  # stride 32

        p3 = self.p3(c3)
        p4 = self.p4(c4)
        p5 = self.p5(c5)

        # EXACTLY what YOLOv8 expects
        return p3, p4, p5
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from ultralytics import YOLO

from alexnet_yolov8m import AlexNetYOLOv8m
from efficientnetb0_yolov8m import EfficientNetB0YOLOv8m
from mobilenetv2_yolov8m import MobileNetv2YOLOv8m
from resnet18_yolov8m import ResNet18YOLOv8m
from resnet34_yolov8m import ResNet34YOLOv8m
from vgg16_yolov8m import VGG16YOLOv8m


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_backbone(backbone_type="alexnet"):
    print(f"\n=== Testing backbone: {backbone_type} ===")

    # 1. Load YOLOv8m.
    model = YOLO("yolov8m.yaml")

    # 2. Replace backbone.
    if backbone_type == "alexnet":
        model.model.model[0] = AlexNetYOLOv8m(pretrained=True)
    elif backbone_type == "resnet-18":
        model.model.model[0] = ResNet18YOLOv8m(pretrained=True)
    elif backbone_type == "resnet-34":
        model.model.model[0] = ResNet34YOLOv8m(pretrained=True)
    elif backbone_type == "vgg-16":
        model.model.model[0] = VGG16YOLOv8m(pretrained=True)
    elif backbone_type == "mobilenetv2":
        model.model.model[0] = MobileNetv2YOLOv8m(pretrained=True)
    elif backbone_type == "efficientnet-b0":
        model.model.model[0] = EfficientNetB0YOLOv8m(pretrained=True)
    else:
        print(f"Default mode")

    # 3. Show YOLO summary.
    model.info()

    # 4. Count parameters.
    backbone_params = count_params(model.model.model[0])
    total_params = count_params(model.model.model)

    print(f"Backbone params: {backbone_params:,}")
    print(f"Total YOLOv8m params: {total_params:,}")

    return backbone_params, total_params


if __name__ == "__main__":
    backbones = [
        "alexnet",
        "resnet-18",
        "resnet-34",
        "vgg-16",
        "default",
        "mobilenetv2",
        "efficientnet-b0",
    ]
    backbone_params_list = []
    total_params_list = []

    for b in backbones:
        backbone_params, total_params = test_backbone(b)
        backbone_params_list.append(backbone_params)
        total_params_list.append(total_params)

    # 1. Plotting.
    x = range(len(backbones))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    # 2. Total params.
    ax.bar([i - width / 2 for i in x], total_params_list, width, label="Total YOLOv8m")

    # 3. Backbone params.
    ax.bar(
        [i + width / 2 for i in x], backbone_params_list, width, label="Backbone only"
    )

    # 4. Labels.
    ax.set_xticks(x)
    ax.set_xticklabels(backbones, rotation=30)
    ax.set_ylabel("Number of parameters")
    ax.set_title("YOLOv8m: Total vs Backbone Parameters (Comparison of Backbones)")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.show()

    # 5. Print numeric comparison.
    print("\n=== Numeric Comparison ===")
    for i, b in enumerate(backbones):
        print(
            f"{b}: backbone params = {backbone_params_list[i]:,}, "
            f"total params = {total_params_list[i]:,}"
        )

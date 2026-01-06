# Importing torch and ultralytics libraries.
import time

import torch.nn as nn
from torchvision.models.mobilenetv2 import MobileNetV2
from ultralytics import YOLO

# Import AlexNet, ResNet18 backbones.
from alexnet_yolov8m import AlexNetYOLOv8m
from efficientnetb0_yolov8m import EfficientNetB0YOLOv8m
from mobilenetv2_yolov8m import MobileNetv2YOLOv8m
from resnet18_yolov8m import ResNet18YOLOv8m
from resnet34_yolov8m import ResNet34YOLOv8m
from vgg16_yolov8m import VGG16YOLOv8m


# Defining Yolov8m with custom backbone.
class INTERCOYolov8m(nn.Module):
    def __init__(self, type="alexnet", pretrained=True):
        """
        Class constructor.
        type: backbone selection (alexnet or resnet18)
        pretrained:
        """
        super().__init__()
        self.type = type

        # Backbone selection.
        if self.type == "alexnet":
            self.model = YOLO("yolov8m.yaml")
            self.model.model.model[0] = AlexNetYOLOv8m(pretrained=True)
        elif self.type == "resnet-18":
            self.model = YOLO("yolov8m.yaml")
            self.model.model.model[0] = ResNet18YOLOv8m(pretrained=True)
        elif self.type == "resnet-34":
            self.model = YOLO("yolov8m.yaml")
            self.model.model.model[0] = ResNet34YOLOv8m(pretrained=True)
        elif self.type == "vgg-16":
            self.model = YOLO("yolov8m.yaml")
            self.model.model.model[0] = VGG16YOLOv8m(pretrained=True)
        elif self.type == "efficientnet-b0":
            self.model = YOLO("yolov8m.yaml")
            self.model.model.model[0] = EfficientNetB0YOLOv8m(pretrained=True)
        elif self.type == "mobilenetv2":
            self.model = YOLO("yolov8m.yaml")
            self.model.model.model[0] = MobileNetv2YOLOv8m(pretrained=True)
        else:
            self.model = YOLO("yolov8m.pt")

    def train(
        self,
        yaml_file="interco.yaml",  # Yaml with custom dataset.
        number_of_epochs=100,  # Number of epochs.
        imgsz=128,  # Size of images.
        batch=32,  # Size of batches.
        workers=4,  # Number of workers for loading data.
    ):
        self.model.train(
            data=yaml_file,  # plik z danymi treningowymi
            epochs=number_of_epochs,  # liczba epok
            imgsz=imgsz,  # rozmiar obrazów
            batch=batch,  # wielkość batcha
            workers=workers,  # liczba wątków
            project=f"results/{self.type}Yolov8m_{time.strftime('%Y%m%d_%H%M%S')}",
        )

    def print_model(self):
        print(self.model.model)

    def count_params(self):
        """Returns total number of trainable parameters in the whole YOLO model."""
        return sum(p.numel() for p in self.model.model.parameters() if p.requires_grad)

    def count_backbone_params(self):
        """Returns number of trainable parameters only in the backbone."""
        backbone = self.model.model.model[0]
        return sum(p.numel() for p in backbone.parameters() if p.requires_grad)

    def print_params(self):
        """Print total and backbone parameters."""
        total = self.count_params()
        backbone = self.count_backbone_params()
        print(f"Backbone ({self.type}) parameters: {backbone:,}")
        print(f"Total YOLOv8m parameters: {total:,}")

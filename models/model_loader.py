import torch.nn as nn
from torchvision import models
import timm

def get_resnet50(num_classes):
    model = models.resnet50(weights="IMAGENET1K_V1")
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def get_mobilenet(num_classes):
    model = models.mobilenet_v2(weights="IMAGENET1K_V1")
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    return model

def get_efficientnet(num_classes):
    model = timm.create_model(
        "efficientnetv2_s",
        pretrained=True,
        num_classes=num_classes
    )
    return model

def get_vit_tiny(num_classes):
    model = timm.create_model(
        "vit_tiny_patch16_224",
        pretrained=True,
        num_classes=num_classes
    )
    return model
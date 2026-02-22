import torch
from models.model_loader import get_resnet50

print("Creating model...")
model = get_resnet50(num_classes=6)

print("Passing dummy input...")
dummy = torch.randn(1, 3, 224, 224)
output = model(dummy)

print("Output shape:", output.shape)
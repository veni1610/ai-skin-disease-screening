import torch
from models.model_loader import (
    get_resnet50,
    get_mobilenet,
    get_efficientnet,
    get_vit_tiny
)
from utils.dataloader import get_dataloaders
from utils.evaluate import evaluate_model

train_dir = "dataset/processed/train"
val_dir = "dataset/processed/val"
test_dir = "dataset/processed/test"

_, _, test_loader = get_dataloaders(
    train_dir, val_dir, test_dir, batch_size=16
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

models_info = {
    "ResNet50": ("models/resnet50.pth", get_resnet50),
    "MobileNet": ("models/mobilenet.pth", get_mobilenet),
    "EfficientNetV2": ("models/efficientnet.pth", get_efficientnet),
    "ViT": ("models/vit.pth", get_vit_tiny),
}

results = {}

for name, (path, model_fn) in models_info.items():

    print(f"\nEvaluating {name}...")

    model = model_fn(num_classes=6)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)

    acc, f1, _ = evaluate_model(model, test_loader)

    results[name] = (acc, f1)

print("\n===== FINAL COMPARISON =====")
for name, (acc, f1) in results.items():
    print(f"{name}: Accuracy = {acc:.4f}, F1 = {f1:.4f}")
from models.model_loader import (
    get_resnet50,
    get_mobilenet,
    get_efficientnet,
    get_vit_tiny
)

from utils.dataloader import get_dataloaders
from utils.train import train_model
from utils.evaluate import evaluate_model

import torch
import os

train_dir = "dataset/processed/train"
val_dir = "dataset/processed/val"
test_dir = "dataset/processed/test"

train_loader, val_loader, test_loader = get_dataloaders(
    train_dir, val_dir, test_dir, batch_size=16
)

models_to_train = {
    "resnet50": get_resnet50,
    "mobilenet": get_mobilenet,
    "efficientnet": get_efficientnet,
    "vit": get_vit_tiny
}

results = {}

for name, model_fn in models_to_train.items():

    print(f"\n\n========== Training {name.upper()} ==========")

    model = model_fn(num_classes=6)

    model = train_model(model, train_loader, val_loader, epochs=10)

    # Save model
    model_path = f"models/{name}.pth"
    torch.save(model.state_dict(), model_path)

    acc, f1, cm = evaluate_model(model, test_loader)

    results[name] = (acc, f1)

    print(f"\n{name} Test Accuracy: {acc}")
    print(f"{name} Test F1 Score: {f1}")

print("\n\n===== FINAL COMPARISON =====")
for name, (acc, f1) in results.items():
    print(f"{name} -> Accuracy: {acc:.4f}, F1: {f1:.4f}")
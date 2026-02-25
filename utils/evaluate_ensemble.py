import torch
from utils.dataloader import get_dataloaders
from app.ensemble_inference import efficientnet, vit, class_names
from sklearn.metrics import accuracy_score, f1_score
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get test loader
_, _, test_loader = get_dataloaders(
    "dataset/processed/train",
    "dataset/processed/val",
    "dataset/processed/test",
    batch_size=16
)
print(len(test_loader.dataset))

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        out1 = efficientnet(images)
        out2 = vit(images)

        prob1 = F.softmax(out1, dim=1)
        prob2 = F.softmax(out2, dim=1)

        # Soft voting ensemble
        final_prob =  (prob1 + prob2)/2 

        preds = torch.argmax(final_prob, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Compute metrics
acc = accuracy_score(all_labels, all_preds)
f1_micro = f1_score(all_labels, all_preds, average="micro")
f1_macro = f1_score(all_labels, all_preds, average="macro")
f1_weighted = f1_score(all_labels, all_preds, average="weighted")

print("\n===== ENSEMBLE RESULTS =====")
print(f"Accuracy: {acc:.4f}")
print(f"F1 Micro: {f1_micro:.4f}")
print(f"F1 Macro: {f1_macro:.4f}")
print(f"F1 Weighted: {f1_weighted:.4f}")
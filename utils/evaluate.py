from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import torch

def evaluate_model(model, data_loader, class_names):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    all_preds = []
    all_labels = []

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            for i in range(len(labels)):
                true_label = class_names[labels[i].item()]
                pred_label = class_names[preds[i].item()]

                if preds[i] == labels[i]:
                    print(f"✅ Correct | True: {true_label} | Predicted: {pred_label}")
                    correct += 1
                else:
                    print(f"❌ Wrong   | True: {true_label} | Predicted: {pred_label}")

                total += 1

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)

    f1_micro = f1_score(all_labels, all_preds, average="micro")
    f1_macro = f1_score(all_labels, all_preds, average="macro")
    f1_weighted = f1_score(all_labels, all_preds, average="weighted")

    cm = confusion_matrix(all_labels, all_preds)

    print(f"\nAccuracy: {acc:.4f}")
    print(f"F1 Micro: {f1_micro:.4f}")
    print(f"F1 Macro: {f1_macro:.4f}")
    print(f"F1 Weighted: {f1_weighted:.4f}")
    print(f"\nTotal Correct: {correct}/{total}")
    print(f"Accuracy: {acc:.4f}")

    return acc, f1_weighted, cm
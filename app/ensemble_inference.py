import torch
from PIL import Image
from torchvision import transforms
from models.model_loader import get_efficientnet, get_vit_tiny

class_names = [
    "acne",
    "bcc",
    "eczema",
    "melanoma",
    "psoriasis",
    "seborrheic_keratosis"
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load EfficientNet
efficientnet = get_efficientnet(num_classes=6)
efficientnet.load_state_dict(
    torch.load("models/efficientnet.pth", map_location=device)
)
efficientnet.to(device)
efficientnet.eval()

# Load ViT
vit = get_vit_tiny(num_classes=6)
vit.load_state_dict(
    torch.load("models/vit.pth", map_location=device)
)
vit.to(device)
vit.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def predict_ensemble(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        out1 = efficientnet(image)
        out2 = vit(image)

        prob1 = torch.softmax(out1, dim=1)
        prob2 = torch.softmax(out2, dim=1)

        # Soft voting
        final_prob = (prob1 + prob2) / 2

        prediction = torch.argmax(final_prob, dim=1)

    return class_names[prediction.item()]
print(predict_ensemble("dataset/processed/test/acne/07Acne081101.jpg"))
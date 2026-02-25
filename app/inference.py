# app/inference.py

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from models.model_loader import get_efficientnet, get_vit_tiny

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class_names = [
    "acne",
    "bcc",
    "eczema",
    "melanoma",
    "psoriasis",
    "seborrheic_keratosis"
]

risk_map = {
    "acne": "Low",
    "seborrheic_keratosis": "Low",
    "eczema": "Moderate",
    "psoriasis": "Moderate",
    "bcc": "High",
    "melanoma": "High"
}

explanation_map = {
    "acne": "Inflamed red papules and pustules detected.",
    "melanoma": "Irregular borders and dark pigmentation observed.",
    "bcc": "Shiny nodular lesion pattern detected.",
    "eczema": "Red itchy patches with dry texture observed.",
    "psoriasis": "Thick scaly plaques detected.",
    "seborrheic_keratosis": "Waxy brown benign lesion observed."
}

recommendation_map = {
    "acne": "Maintain hygiene and use mild cleansers.",
    "eczema": "Use moisturizers and avoid allergens.",
    "psoriasis": "Dermatology consultation recommended.",
    "seborrheic_keratosis": "Usually benign. Monitor for changes.",
    "bcc": "Immediate dermatologist consultation recommended.",
    "melanoma": "Urgent biopsy and specialist consultation required."
}

# -------- Load EfficientNet --------
efficientnet = get_efficientnet(num_classes=6)
efficientnet.load_state_dict(
    torch.load("models/efficientnet_final_v1.pth", map_location=device)
)
efficientnet.to(device)
efficientnet.eval()

# -------- Load ViT --------
vit = get_vit_tiny(num_classes=6)
vit.load_state_dict(
    torch.load("models/vit_final_v1.pth", map_location=device)
)
vit.to(device)
vit.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def predict(image_file, symptoms=None):

    image = Image.open(image_file).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        out1 = efficientnet(image)
        out2 = vit(image)

        prob1 = F.softmax(out1, dim=1)
        prob2 = F.softmax(out2, dim=1)

        # Weighted Ensemble (EfficientNet stronger)
        final_prob = 0.6 * prob1 + 0.4 * prob2

        pred = torch.argmax(final_prob, dim=1)

    disease = class_names[pred.item()]
    confidence = float(final_prob[0][pred].item())

    risk = risk_map[disease]

    # Symptom escalation
    if symptoms:
        if symptoms.get("bleeding"):
            risk = "High"
        if symptoms.get("rapid_growth"):
            risk = "High"
        if symptoms.get("pain") and risk == "Low":
            risk = "Moderate"

    return {
        "disease": disease,
        "confidence": round(confidence, 3),
        "risk": risk,
        "explanation": explanation_map[disease],
        "recommendation": recommendation_map[disease]
    }
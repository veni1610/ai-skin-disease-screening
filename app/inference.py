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
    "acne": "Acne is a common inflammatory skin condition characterized by red pimples and pustules. It usually occurs due to clogged pores, excess oil production, and bacterial activity.",

    "melanoma": "Melanoma is a potentially serious skin cancer often identified by irregular borders and dark pigmentation. Early detection is critical as it can spread rapidly if untreated.",

    "bcc": "Basal Cell Carcinoma is a type of skin cancer that typically appears as a shiny or pearly lesion. It grows slowly but requires medical treatment to prevent tissue damage.",

    "eczema": "Eczema is a chronic inflammatory skin condition causing redness, dryness, and itching. It is often linked to allergies or immune system sensitivity.",

    "psoriasis": "Psoriasis is an autoimmune condition characterized by thick, scaly plaques on the skin. It results from rapid skin cell turnover and may require long-term management.",

    "seborrheic_keratosis": "Seborrheic keratosis is a benign skin growth that appears waxy or slightly raised. It is generally harmless and does not require treatment unless irritated."
}

recommendation_map = {
    "acne": "Maintain hygiene and use mild cleansers.",
    "eczema": "Use moisturizers and avoid allergens.",
    "psoriasis": "Dermatology consultation recommended.",
    "seborrheic_keratosis": "Usually benign. Monitor for changes.",
    "bcc": "Immediate dermatologist consultation recommended.",
    "melanoma": "Urgent biopsy and specialist consultation required."
}

risk_explanation_map = {
    "Low": "This condition is generally non-life-threatening and may improve with basic skincare or monitoring.",
    "Moderate": "This condition may require medical consultation to prevent progression or complications.",
    "High": "This condition may indicate a potentially serious issue and requires immediate dermatological evaluation."
}

# -------- Load EfficientNet --------
efficientnet = get_efficientnet(num_classes=6)
efficientnet.load_state_dict(
    torch.load("models/efficientnet.pth", map_location=device)
)
efficientnet.to(device)
efficientnet.eval()

# -------- Load ViT --------
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

    base_risk = risk_map[disease]

    return {
        "disease": disease,
        "confidence": round(confidence, 3),
        "risk": base_risk,
        "risk_explanation": risk_explanation_map[base_risk],
        "explanation": explanation_map[disease]
    }
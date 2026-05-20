import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import os
import numpy as np
from datetime import datetime
from torchcam.methods import LayerCAM
from torchcam.utils import overlay_mask
from torchvision.transforms.functional import to_pil_image

# ---------------- DEVICE ----------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- PATHS ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "chest_cnn.pth")

HEATMAP_DIR = os.path.join(BASE_DIR, "..", "static", "uploads", "heatmaps")
os.makedirs(HEATMAP_DIR, exist_ok=True)

# ---------------- MODEL: ResNet18 ----------------
resnet = models.resnet18(weights=None)
# Restore exact 1-channel architecture matching the legacy weights
resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
resnet.fc = nn.Linear(resnet.fc.in_features, 2)

try:
    if os.path.exists(MODEL_PATH):
        state = torch.load(MODEL_PATH, map_location=DEVICE)
        resnet.load_state_dict(state)
        print("✅ Custom CNN 1-channel Grayscale weights successfully loaded!")
    else:
        print("⚠️ No custom weights found.")
except Exception as e:
    print(f"⚠️ Could not load custom weights ({e}).")

resnet = resnet.to(DEVICE).eval()

# Initialize Grad-CAM extractor on the trained ResNet
# Note: Since our ResNet is purely 1-channel, we must explicitly declare it to TorchCAM
cam_extractor = LayerCAM(resnet, input_shape=(1, 224, 224))

# ---------------- TRANSFORM ----------------
# Standard 1-Channel Grayscale transformation identical to original test environment
transform_1ch = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# ---------------- PREDICT FUNCTION ----------------
def predict(image_path):

    # ---------- Load image ----------
    # Convert to RGB just for rendering the heatmap overlay later
    img_rgb_base = Image.open(image_path).convert("RGB")
    # Resize to max 512x512 for drastically faster Grad-CAM overlay and lower disk I/O
    img_rgb_base.thumbnail((512, 512))

    # ---------- Preprocess ----------
    img_tensor = transform_1ch(img_rgb_base).unsqueeze(0).to(DEVICE)
    img_tensor.requires_grad_(True) # Required for Grad-CAM

    # ---------- Model inference ----------
    logits = resnet(img_tensor)
    
    # Raw softmax probabilities
    probs = F.softmax(logits, dim=1)

    abnormal_prob = probs[0][0].item()
    normal_prob = probs[0][1].item()
    # Medical sensitivity calibration threshold.
    # X-ray anomaly detection favors false positives over false negatives.
    threshold = 0.20
    if abnormal_prob >= threshold:
        predicted_idx = 0
    else:
        predicted_idx = 1
        
    class_names = ['abnormal', 'normal']

    label = class_names[predicted_idx]
    prob_value = probs[0][predicted_idx].item()


    # ----- Removed 100% Precision Override -----

    # ---------- Grad-CAM Heatmap Generation ----------
    heatmap_url = None
    if label == "abnormal":
        # Always extract anomaly features (index 0) to highlight the opacity/issue
        activation_map = cam_extractor(0, logits)
        heatmap_pil = to_pil_image(activation_map[0].squeeze(0), mode='F')
        
        # Overlay heatmap on original image smoothly
        overlay_result = overlay_mask(img_rgb_base, heatmap_pil, alpha=0.5)
        
        filename = f"heatmap_{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
        heatmap_path_full = os.path.join(HEATMAP_DIR, filename)
        overlay_result.save(heatmap_path_full)
        heatmap_url = f"/static/uploads/heatmaps/{filename}"

    # ---------- Confidence estimation ----------
    if prob_value >= 0.85:
        confidence = "High"
    elif prob_value >= 0.65:
        confidence = "Medium"
    else:
        confidence = "Low"

    return {
        "label": label.capitalize(),
        "probability": round(prob_value, 3),
        "abnormal_probability": round(abnormal_prob, 3),
        "normal_probability": round(normal_prob, 3),
        "confidence": confidence,
        "heatmap_path": heatmap_url
    }


# scripts/extract_features.py
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import numpy as np
import sys
import os

def extract(image_path, save_path):
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    img = Image.open(image_path).convert("RGB")
    inputs = processor(images=img, return_tensors="pt")
    with torch.no_grad():
        feats = model.get_image_features(**inputs)  # (1, D)
    feats = feats.cpu().numpy().astype("float32")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, feats)
    print("Saved features to", save_path)

if __name__ == "__main__":
    img = sys.argv[1]
    out = sys.argv[2]
    extract(img, out)

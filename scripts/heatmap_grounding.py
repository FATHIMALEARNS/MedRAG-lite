# scripts/heatmap_grounding.py
import torch
import numpy as np
import cv2
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

def generate_heatmap(image_path):
    device = "cpu"
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    inputs["pixel_values"].requires_grad_(True)

    image_features = model.get_image_features(**inputs)
    score = image_features.norm()
    score.backward()

    grads = inputs["pixel_values"].grad[0]
    heatmap = grads.abs().mean(dim=0).detach().numpy()

    heatmap = cv2.resize(heatmap, image.size)
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() + 1e-8)
    return heatmap

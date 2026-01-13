# scripts/caption_blip.py
import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# ✅ Cache model loading (TOP LEVEL only)
@st.cache_resource
def load_blip():
    processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )
    return processor, model


def caption(image_path):
    processor, model = load_blip()

    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    output = model.generate(**inputs, max_length=50)
    caption_text = processor.decode(
        output[0], skip_special_tokens=True
    )

    return caption_text

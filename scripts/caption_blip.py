# scripts/caption_blip.py
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import sys

def caption(image_path):
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    img = Image.open(image_path).convert("RGB")
    inputs = processor(images=img, return_tensors="pt")
    out = model.generate(**inputs, max_length=64)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

if __name__ == "__main__":
    img = sys.argv[1]
    print(caption(img))

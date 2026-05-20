# scripts/grounding_infer.py
# This assumes you installed GroundingDINO from GitHub (in requirements)
from groundingdino.util.inference import load_model, predict
from PIL import Image
import sys, json

def run_grounding(image_path, text_prompt):
    model = load_model("groundingdino_tiny")  # or path to your checkpoint
    img = Image.open(image_path).convert("RGB")
    boxes, logits, phrases = predict(model, img, text_prompt, box_threshold=0.3, text_threshold=0.25)
    results = []
    for b, p, s in zip(boxes, phrases, logits):
        results.append({"bbox": [float(x) for x in b], "phrase": p, "score": float(s)})
    print(json.dumps(results, indent=2))
    return results

if __name__ == "__main__":
    run_grounding(sys.argv[1], sys.argv[2])

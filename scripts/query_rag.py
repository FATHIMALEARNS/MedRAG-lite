# scripts/query_rag.py
from dotenv import load_dotenv
load_dotenv()
import os
import json
import numpy as np
from PIL import Image
import faiss
import openai
import torch

# Local utilities
from utils.severity import severity_from_text, supportive_template

# Lazy-loaded model holders
_model_cache = {
    "blip_proc": None,
    "blip_model": None,
    "clip_proc": None,
    "clip_model": None,
    "faiss_index": None,
    "faiss_meta": None
}

OPENAI_KEY = os.environ.get("OPENAI_API_KEY", None)
if OPENAI_KEY:
    openai.api_key = OPENAI_KEY

def _load_blip():
    if _model_cache["blip_model"] is None:
        from transformers import BlipProcessor, BlipForConditionalGeneration
        _model_cache["blip_proc"] = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        _model_cache["blip_model"] = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return _model_cache["blip_proc"], _model_cache["blip_model"]

def _load_clip():
    if _model_cache["clip_model"] is None:
        from transformers import CLIPProcessor, CLIPModel
        _model_cache["clip_proc"] = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        _model_cache["clip_model"] = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    return _model_cache["clip_proc"], _model_cache["clip_model"]

def load_faiss(idxfile="models/faiss_index.ivf"):
    if _model_cache["faiss_index"] is None:
        if not os.path.exists(idxfile):
            raise FileNotFoundError(f"FAISS index not found at {idxfile}. Build it with build_faiss_index.py")
        idx = faiss.read_index(idxfile)
        meta_path = idxfile + ".meta.json"
        if os.path.exists(meta_path):
            meta = json.load(open(meta_path, "r"))
        else:
            meta = []
        _model_cache["faiss_index"] = idx
        _model_cache["faiss_meta"] = meta
    return _model_cache["faiss_index"], _model_cache["faiss_meta"]

def get_caption(image_path):
    """Generate a caption with BLIP (lazy loads model)."""
    proc, model = _load_blip()
    img = Image.open(image_path).convert("RGB")
    inputs = proc(images=img, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inputs, max_length=64)
    caption = proc.decode(out[0], skip_special_tokens=True)
    return caption

def get_feat(image_path):
    """Return CLIP image features (numpy float32 vector)."""
    proc, model = _load_clip()
    img = Image.open(image_path).convert("RGB")
    inputs = proc(images=img, return_tensors="pt")
    with torch.no_grad():
        feats = model.get_image_features(**inputs)  # tensor
    return feats.cpu().numpy().astype("float32")

def query(image_path, idxfile="models/faiss_index.ivf", k=5):
    """
    Main RAG query function:
    - caption image with BLIP
    - compute CLIP features and retrieve k nearest cases from FAISS
    - send prompt to OpenAI if API key exists, otherwise return caption
    - apply severity filter and supportive persona if needed
    Returns a dictionary: {'caption', 'retrieved', 'answer'}
    """
    # 1) caption
    try:
        caption_text = get_caption(image_path)
    except Exception as e:
        caption_text = f"[Captioning failed: {e}]"

    # 2) features + faiss retrieval
    try:
        feat = get_feat(image_path)  # shape (1, d)
        idx, meta = load_faiss(idxfile)
        D, I = idx.search(feat, k)
        retrieved = [meta[i] if i < len(meta) else f"case_{i}" for i in I[0]]
    except Exception as e:
        retrieved = []
        # don't crash the whole pipeline if retrieval fails
        print(f"[FAISS/CLIP retrieval error] {e}")

    # 3) compose prompt
    prompt_parts = [
        f"Image caption: {caption_text}",
    ]
    if retrieved:
        prompt_parts.append("Relevant prior case IDs: " + ", ".join(retrieved))
    prompt_parts.append(
        "Explain the caption above in plain language for a patient and suggest 3 questions they can ask their doctor. "
        "Be empathetic but do not provide a medical diagnosis. Keep it concise."
    )
    prompt = "\n\n".join(prompt_parts)

    # 4) call LLM (OpenAI GPT) or fallback
    answer_text = caption_text
    if OPENAI_KEY:
        try:
            resp = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[{"role":"user","content":prompt}],
                max_tokens=400,
                temperature=0.1
            )
            answer_text = resp["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"[OpenAI call failed] {e}")
            answer_text = caption_text + "\n\n" + "[LLM call failed: using caption as fallback]"
    else:
        # No API key — keep caption as the returned "answer" but in a user-friendly format
        answer_text = f"(No OPENAI_API_KEY configured) Caption: {caption_text}"

    # 5) severity check + persona
    sev = severity_from_text(answer_text)
    if sev == "high":
        answer_text = supportive_template(answer_text)

    return {"caption": caption_text, "retrieved": retrieved, "answer": answer_text}

# If run as script, allow quick test
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python scripts/query_rag.py path/to/image.jpg")
        sys.exit(1)
    out = query(sys.argv[1])
    print("=== Caption ===")
    print(out["caption"])
    print("\n=== Answer ===")
    print(out["answer"])

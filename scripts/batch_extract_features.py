import os, sys, argparse, time, json
from pathlib import Path
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import numpy as np
from tqdm import tqdm
import pandas as pd

torch.set_num_threads(1)

def init_models(device="cpu"):
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.to(device)
    model.eval()
    return model, processor

def extract_and_save(model, processor, img_path, out_path, device="cpu"):
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as e:
        return False, f"image open error: {e}"
    inputs = processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        feats = model.get_image_features(**inputs)
    feats = feats.cpu().numpy().astype("float32")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, feats)
    return True, None

def main(csv_file, images_root, out_dir, img_col="Image Index", resume=True, limit=None):
    df = pd.read_csv(csv_file)

    # NORMALIZE ALL COLUMN NAMES (fixes hidden spaces/tabs/unicode)
    df.columns = (
    df.columns
    .str.encode('ascii', 'ignore').str.decode('ascii')  # remove weird unicode
    .str.strip()                                        # remove invisible spaces
    .str.replace(r'\s+', '_', regex=True)              # replace ALL spaces with "_"
)
    img_col = img_col.replace(" ", "_")                     # normalize user input too

    if img_col not in df.columns:
        print(f"ERROR: column '{img_col}' not in CSV. Columns: {df.columns.tolist()}")
        return
    model, processor = init_models()
    total = len(df) if limit is None else min(limit, len(df))
    pbar = tqdm(df.iloc[:total].itertuples(index=False), total=total)
    for row in pbar:
        img_rel = row._asdict()[img_col]
        img_path = img_rel if os.path.isabs(img_rel) else os.path.join(images_root, img_rel)
        base = Path(img_path).stem
        out_path = os.path.join(out_dir, f"{base}.npy")
        if resume and os.path.exists(out_path):
            pbar.set_description(f"skip {base}")
            continue
        ok, err = extract_and_save(model, processor, img_path, out_path)
        if not ok:
            pbar.write(f"FAILED: {img_path} -> {err}")
        else:
            pbar.set_description(f"done {base}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--images_root", required=True)
    parser.add_argument("--out_dir", default="models/features")
    parser.add_argument("--img_col", default="Image Index")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    main(args.csv, args.images_root, args.out_dir, args.img_col, resume=True, limit=args.limit)

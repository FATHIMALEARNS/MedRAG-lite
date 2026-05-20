# scripts/build_faiss_index.py
import faiss
import numpy as np
import os
import json

def build_index(features_dir, index_file="faiss_index.ivf"):
    feats = []
    meta = []
    for fn in os.listdir(features_dir):
        if fn.endswith(".npy"):
            arr = np.load(os.path.join(features_dir, fn))
            feats.append(arr)
            meta.append(fn.replace(".npy",""))
    feats = np.vstack(feats)
    d = feats.shape[1]
    nlist = max(1, int(len(feats) / 10))
    quant = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFFlat(quant, d, nlist, faiss.METRIC_L2)
    index.train(feats)
    index.add(feats)
    faiss.write_index(index, index_file)
    with open(index_file + ".meta.json","w") as f:
        json.dump(meta, f)
    print("Saved index:", index_file)

if __name__ == "__main__":
    build_index("models/features", "models/faiss_index.ivf")

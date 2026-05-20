import os
import cv2
import pydicom
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# ---------------- PATHS ----------------
RAW_IMG_DIR = "raw_images"
CSV_PATH = "stage_2_train_labels.csv"
OUT_DIR = "data"

# ---------------- LOAD LABELS ----------------
df = pd.read_csv(CSV_PATH)

# Any image with Target=1 is abnormal
labels = df.groupby("patientId")["Target"].max().reset_index()

# Split
train_ids, val_ids = train_test_split(
    labels,
    test_size=0.2,
    stratify=labels["Target"],
    random_state=42
)

def save_images(split_df, split_name):
    for _, row in tqdm(split_df.iterrows(), total=len(split_df)):
        pid = row["patientId"]
        label = "abnormal" if row["Target"] == 1 else "normal"

        dcm_path = os.path.join(RAW_IMG_DIR, f"{pid}.dcm")
        if not os.path.exists(dcm_path):
            continue

        dcm = pydicom.dcmread(dcm_path)
        img = dcm.pixel_array

        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        img = img.astype("uint8")

        out_dir = os.path.join(OUT_DIR, split_name, label)
        os.makedirs(out_dir, exist_ok=True)

        cv2.imwrite(os.path.join(out_dir, f"{pid}.png"), img)

# ---------------- RUN ----------------
save_images(train_ids, "train")
save_images(val_ids, "val")

print("✅ Dataset preparation complete")

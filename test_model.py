import os
from ml.predict import predict

normal_folder = "ml/data/train/normal"
abnormal_folder = "ml/data/train/abnormal"
print("\n--- Testing NORMAL images ---")
for img in os.listdir(normal_folder)[:10]:   # test first 10
    path = os.path.join(normal_folder, img)
    result = predict(path)
    print(img, "->", result)

print("\n--- Testing ABNORMAL images ---")
for img in os.listdir(abnormal_folder)[:10]:
    path = os.path.join(abnormal_folder, img)
    result = predict(path)
    print(img, "->", result)
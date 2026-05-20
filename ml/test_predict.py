import os
from predict import predict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

test_images = [
    os.path.join(BASE_DIR, "data", "val", "abnormal", "7b7d7bf3-0684-465d-9a74-f51887685387.png"),
    os.path.join(BASE_DIR, "data", "val", "abnormal", "03e4827c-7338-4de3-9ac6-8831ba5637e9.png")
]

for img in test_images:
    print(f"\nImage: {img}")
    result = predict(img)
    print(result)
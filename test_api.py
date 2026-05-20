import requests

url = "http://127.0.0.1:5000/analyze"

payload = {
    "image_path": "ml/data/val/abnormal/7b7d7bf3-0684-465d-9a74-f51887685387.png",
    "user_query": "What does this chest X-ray result mean?"
}

response = requests.post(url, json=payload)
print(response.json())
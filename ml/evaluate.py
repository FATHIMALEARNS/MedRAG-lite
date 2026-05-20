import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

from model import get_chest_resnet

# -------- CONFIG --------
DATA_DIR = "data/val"
MODEL_PATH = "model/chest_cnn.pth"
BATCH_SIZE = 16
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------- TRANSFORMS --------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Grayscale(num_output_channels=1),  # 🔥 IMPORTANT
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# -------- LOAD DATA --------
dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# -------- LOAD MODEL (CORRECT WAY) --------
model = get_chest_resnet(num_classes=len(dataset.classes))

state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state_dict)

model.to(DEVICE)
model.eval()   # ✅ THIS IS NOW VALID

all_preds = []
all_labels = []

# -------- EVALUATION --------
with torch.no_grad():
    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)
        probs = F.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print("\n📊 Classification Report:\n")
print(classification_report(all_labels, all_preds, target_names=dataset.classes))

print("\n📉 Confusion Matrix:\n")
print(confusion_matrix(all_labels, all_preds))
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms
import os

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Standard grayscale transformations matching predict.py
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_dir = os.path.join(base_dir, 'data', 'train')
    val_dir = os.path.join(base_dir, 'data', 'val')

    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=transform)

    # Extract indices for both classes
    torch.manual_seed(42)
    train_abn_idx = [i for i, (_, label) in enumerate(train_dataset.samples) if label == 0]
    train_nor_idx = [i for i, (_, label) in enumerate(train_dataset.samples) if label == 1]
    
    # Balance training dataset by taking the minimum available
    train_min = min(len(train_abn_idx), len(train_nor_idx))
    train_subset_idx = (torch.tensor(train_abn_idx)[torch.randperm(len(train_abn_idx))[:train_min]].tolist() +
                        torch.tensor(train_nor_idx)[torch.randperm(len(train_nor_idx))[:train_min]].tolist())
    train_subset = torch.utils.data.Subset(train_dataset, train_subset_idx)

    # Balance validation dataset
    val_abn_idx = [i for i, (_, label) in enumerate(val_dataset.samples) if label == 0]
    val_nor_idx = [i for i, (_, label) in enumerate(val_dataset.samples) if label == 1]
    
    val_min = min(len(val_abn_idx), len(val_nor_idx))
    val_subset_idx = (torch.tensor(val_abn_idx)[torch.randperm(len(val_abn_idx))[:val_min]].tolist() +
                      torch.tensor(val_nor_idx)[torch.randperm(len(val_nor_idx))[:val_min]].tolist())
    val_subset = torch.utils.data.Subset(val_dataset, val_subset_idx)

    print(f"Subsampled datasets: {len(train_subset)} train, {len(val_subset)} val")
    print(f"Class mapping: {train_dataset.class_to_idx}")

    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=16, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_subset, batch_size=16, shuffle=False)

    # Initialize model with Pretrained ImageNet weights
    print("Loading Pretrained ResNet18...")
    model = models.resnet18(weights="DEFAULT")
    
    # Convert first conv layer from 3-channel to 1-channel, preserving ImageNet feature extractors
    old_weight = model.conv1.weight.clone() # Shape: [64, 3, 7, 7]
    model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    # Summing across the RGB channel dimension simulates duplicate 3-channel grayscale inputs
    model.conv1.weight.data = old_weight.sum(dim=1, keepdim=True)
    
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 5
    best_val_acc = 0.0

    model_path = os.path.join(base_dir, 'model', 'chest_cnn.pth')
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    print("Starting fast transfer learning...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = 100 * correct / total
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {running_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_path)
            print(f"✅ Saved better model with val acc: {val_acc:.2f}%")

    print("Training finished!")

if __name__ == '__main__':
    train()
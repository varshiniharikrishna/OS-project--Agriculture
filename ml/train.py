"""
Training & Fine-Tuning Pipeline for PlantVillage Crop Disease Detection.
Usage:
    python3 ml/train.py --data_dir ml/dataset/PlantVillage --epochs 10 --batch_size 32
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Add project root directory to sys.path so 'from ml.xxx' imports work when executed directly
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from ml.efficientnet_b0 import EfficientNetB0, ResNet50, PLANTVILLAGE_CLASSES
except ImportError:
    from efficientnet_b0 import EfficientNetB0, ResNet50, PLANTVILLAGE_CLASSES


def train_model(data_dir, epochs=10, batch_size=32, lr=0.001, model_type="efficientnet_b0"):
    """Train EfficientNet-B0 or ResNet50 on PlantVillage dataset folders."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Starting training on device: {device}", flush=True)

    # Data Augmentation & Normalization
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if not os.path.exists(data_dir):
        print(f"Error: Dataset directory '{data_dir}' not found.", flush=True)
        print("Please upload the PlantVillage dataset to ml/dataset/PlantVillage according to ml/dataset/README.md.", flush=True)
        return

    dataset = datasets.ImageFolder(root=data_dir, transform=data_transforms)
    print(f"Dataset successfully loaded: {len(dataset)} images across {len(dataset.classes)} classes.", flush=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # Initialize Model
    if model_type == "resnet50":
        model = ResNet50(num_classes=len(dataset.classes))
    else:
        model = EfficientNetB0(num_classes=len(dataset.classes))
    
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training Loop
    model.train()
    print(f"Training loop started for {epochs} epoch(s)...", flush=True)
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for step, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if step == 0 or (step + 1) % 5 == 0 or (step + 1) == len(dataloader):
                batch_acc = ((predicted == labels).sum().item() / labels.size(0)) * 100.0
                print(f"Epoch [{epoch+1}/{epochs}] - Batch [{step+1}/{len(dataloader)}] - Loss: {loss.item():.4f} - Batch Acc: {batch_acc:.1f}%", flush=True)


        epoch_loss = running_loss / total
        epoch_acc = (correct / total) * 100.0
        print(f"--- Epoch [{epoch+1}/{epochs}] Summary: Loss = {epoch_loss:.4f}, Accuracy = {epoch_acc:.2f}% ---")


    # Save Checkpoint
    output_path = os.path.join(os.path.dirname(__file__), "models", f"{model_type}_plantvillage.pth")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(model.state_dict(), output_path)
    print(f"Training complete. Checkpoint saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EfficientNet-B0 on PlantVillage")
    parser.add_argument("--data_dir", type=str, default="ml/dataset/PlantVillage", help="Path to PlantVillage dataset")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--model", type=str, default="efficientnet_b0", choices=["efficientnet_b0", "resnet50"])
    
    args = parser.parse_args()
    train_model(args.data_dir, args.epochs, args.batch_size, args.lr, args.model)

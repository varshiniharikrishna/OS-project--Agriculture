"""
Two-Stage Transfer Learning & Fine-Tuning Pipeline for PlantVillage Crop Disease Detection.
Stage 1: Freeze EfficientNet-B0 backbone, train classification head (LR = 1e-3).
Stage 2: Unfreeze deep feature extraction blocks, fine-tune (LR = 1e-4).
"""

import os
import sys
import json
import random
import argparse
import ssl
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms
try:
    from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
    TORCHVISION_MODEL_AVAILABLE = True
except ImportError:
    TORCHVISION_MODEL_AVAILABLE = False

# Bypass SSL certificate error when downloading PyTorch pretrained weights
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except Exception:
    pass

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from ml.efficientnet_b0 import EfficientNetB0, ResNet50
except ImportError:
    from efficientnet_b0 import EfficientNetB0, ResNet50


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_pretrained_model(num_classes):
    """Build EfficientNet-B0 with pretrained ImageNet weights."""
    if TORCHVISION_MODEL_AVAILABLE:
        try:
            weights = EfficientNet_B0_Weights.DEFAULT
            model = efficientnet_b0(weights=weights)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
            model.model_name = "EfficientNet-B0 (Two-Stage Transfer Learning)"
            return model
        except Exception as e:
            print(f"Notice: Loading custom EfficientNetB0 architecture ({e}).")

    model = EfficientNetB0(num_classes=num_classes)
    return model


def freeze_backbone(model, freeze=True):
    """Freeze or unfreeze backbone layers for 2-stage transfer learning."""
    if hasattr(model, "features"):
        for param in model.features.parameters():
            param.requires_grad = not freeze
    elif hasattr(model, "stem") and hasattr(model, "blocks"):
        for param in model.stem.parameters():
            param.requires_grad = not freeze
        for param in model.blocks[:-2].parameters():
            param.requires_grad = not freeze


def train_model(data_dir, epochs=10, batch_size=32, stage1_lr=0.001, stage2_lr=0.0001, model_type="efficientnet_b0", seed=42):
    """Executes 2-Stage Transfer Learning training on PlantVillage dataset."""
    seed_everything(seed)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Starting 2-Stage Transfer Learning pipeline on device: {device}", flush=True)

    # Realistic Data Augmentation for Real-World Farm Generalization
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if not os.path.exists(data_dir):
        print(f"Error: Dataset directory '{data_dir}' not found.", flush=True)
        return

    # Load dataset
    full_dataset = datasets.ImageFolder(root=data_dir)
    classes = full_dataset.classes
    num_classes = len(classes)
    print(f"Dataset loaded: {len(full_dataset)} images across {num_classes} classes.", flush=True)

    # Save Authoritative JSON Class Index Mapping
    class_to_idx = {c: idx for idx, c in enumerate(classes)}
    idx_to_class = {idx: c for idx, c in enumerate(classes)}

    class_json_path = os.path.join(os.path.dirname(__file__), "class_to_idx.json")
    idx_json_path = os.path.join(os.path.dirname(__file__), "idx_to_class.json")

    with open(class_json_path, "w", encoding="utf-8") as f:
        json.dump(class_to_idx, f, indent=2)
    with open(idx_json_path, "w", encoding="utf-8") as f:
        json.dump(idx_to_class, f, indent=2)

    # Compute Class Weights for Imbalanced Dataset
    targets = [s[1] for s in full_dataset.samples]
    class_counts = np.bincount(targets, minlength=num_classes)
    total_samples = len(targets)
    class_weights = total_samples / (num_classes * np.maximum(class_counts, 1).astype(np.float32))
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    # 70% Train, 15% Val, 15% Test Split
    total_len = len(full_dataset)
    train_len = int(0.70 * total_len)
    val_len = int(0.15 * total_len)
    test_len = total_len - train_len - val_len

    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset, test_subset = random_split(
        full_dataset, [train_len, val_len, test_len], generator=generator
    )

    # Save split indices
    split_indices = {
        "train_indices": train_subset.indices,
        "val_indices": val_subset.indices,
        "test_indices": test_subset.indices
    }
    split_path = os.path.join(os.path.dirname(__file__), "models", "split_indices.json")
    os.makedirs(os.path.dirname(split_path), exist_ok=True)
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(split_indices, f)

    train_dataset = datasets.ImageFolder(root=data_dir, transform=train_transforms)
    val_dataset = datasets.ImageFolder(root=data_dir, transform=val_transforms)

    train_loader = DataLoader(Subset(train_dataset, train_subset.indices), batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(Subset(val_dataset, val_subset.indices), batch_size=batch_size, shuffle=False, num_workers=0)

    # Initialize Model
    model = build_pretrained_model(num_classes)
    model.to(device)

    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    best_val_acc = 0.0
    best_checkpoint_path = os.path.join(os.path.dirname(__file__), "models", f"{model_type}_plantvillage.pth")

    stage1_epochs = max(2, int(epochs * 0.3))
    stage2_epochs = epochs - stage1_epochs

    # ==================== STAGE 1: Train Classification Head ====================
    print(f"\n--- STAGE 1: Training Classification Head ({stage1_epochs} epochs, LR = {stage1_lr}) ---", flush=True)
    freeze_backbone(model, freeze=True)
    optimizer1 = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=stage1_lr, weight_decay=1e-4)

    for epoch in range(stage1_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer1.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer1.step()
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        t_loss = running_loss / max(1, total)
        t_acc = (correct / max(1, total)) * 100.0

        # Validation
        model.eval()
        v_loss_sum, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                v_loss_sum += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                v_total += labels.size(0)
                v_correct += (predicted == labels).sum().item()

        v_loss = v_loss_sum / max(1, v_total)
        v_acc = (v_correct / max(1, v_total)) * 100.0

        print(f"Stage 1 - Epoch [{epoch+1:02d}/{stage1_epochs:02d}] | Train Loss: {t_loss:.4f}, Acc: {t_acc:.2f}% | Val Loss: {v_loss:.4f}, Acc: {v_acc:.2f}%", flush=True)

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            torch.save(model.state_dict(), best_checkpoint_path)

    # ==================== STAGE 2: Fine-Tune Deeper Layers ====================
    print(f"\n--- STAGE 2: Fine-Tuning Deeper Layers ({stage2_epochs} epochs, LR = {stage2_lr}) ---", flush=True)
    freeze_backbone(model, freeze=False)
    optimizer2 = optim.AdamW(model.parameters(), lr=stage2_lr, weight_decay=1e-4)
    scheduler2 = optim.lr_scheduler.ReduceLROnPlateau(optimizer2, mode="min", factor=0.5, patience=2)

    for epoch in range(stage2_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer2.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer2.step()
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        t_loss = running_loss / max(1, total)
        t_acc = (correct / max(1, total)) * 100.0

        model.eval()
        v_loss_sum, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                v_loss_sum += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                v_total += labels.size(0)
                v_correct += (predicted == labels).sum().item()

        v_loss = v_loss_sum / max(1, v_total)
        v_acc = (v_correct / max(1, v_total)) * 100.0
        scheduler2.step(v_loss)

        print(f"Stage 2 - Epoch [{epoch+1:02d}/{stage2_epochs:02d}] | Train Loss: {t_loss:.4f}, Acc: {t_acc:.2f}% | Val Loss: {v_loss:.4f}, Acc: {v_acc:.2f}%", flush=True)

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            torch.save(model.state_dict(), best_checkpoint_path)
            print(f"  --> Saved new best fine-tuned checkpoint (Val Acc: {best_val_acc:.2f}%)")

    print(f"\nTwo-Stage Training Complete. Best Validation Accuracy: {best_val_acc:.2f}%. Saved to {best_checkpoint_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EfficientNet-B0 on PlantVillage")
    parser.add_argument("--data_dir", type=str, default="ml/dataset/PlantVillage", help="Path to PlantVillage dataset")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Stage 1 Learning rate")
    parser.add_argument("--stage2_lr", type=float, default=0.0001, help="Stage 2 Learning rate")
    parser.add_argument("--model", type=str, default="efficientnet_b0", choices=["efficientnet_b0", "resnet50"])
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    train_model(args.data_dir, args.epochs, args.batch_size, args.lr, args.stage2_lr, args.model, args.seed)

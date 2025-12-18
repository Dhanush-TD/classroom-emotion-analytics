import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
import timm
from sklearn.metrics import classification_report, confusion_matrix


def main():
    # ===============================
    # DEVICE
    # ===============================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ===============================
    # PATHS
    # ===============================
    TRAIN_DIR = "raf_db/train"
    TEST_DIR = "raf_db/test"
    os.makedirs("checkpoints", exist_ok=True)

    # ===============================
    # TRANSFORMS
    # ===============================
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5],
                             [0.5, 0.5, 0.5])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5],
                             [0.5, 0.5, 0.5])
    ])

    # ===============================
    # DATASETS
    # ===============================
    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
    test_dataset = datasets.ImageFolder(TEST_DIR, transform=test_transform)

    print("Class folders:", train_dataset.classes)

    # ===============================
    # CLASS IMBALANCE
    # ===============================
    targets = np.array(train_dataset.targets)
    class_counts = np.bincount(targets)
    print("Class counts:", class_counts)

    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum()
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=0.1
    )

    sample_weights = class_weights[targets].cpu().numpy()
    sampler = WeightedRandomSampler(
        sample_weights,
        len(sample_weights),
        replacement=True
    )

    # ===============================
    # DATALOADERS (WINDOWS SAFE)
    # ===============================
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        sampler=sampler,
        num_workers=0,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    # ===============================
    # MODEL
    # ===============================
    model = timm.create_model(
        "swin_tiny_patch4_window7_224",
        pretrained=True,
        num_classes=7
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=20
    )

    # ===============================
    # TRAINING LOOP
    # ===============================
    def train_one_epoch():
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        return total_loss / len(train_loader), correct / total

    def evaluate():
        model.eval()
        y_true, y_pred = [], []

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                preds = outputs.argmax(dim=1)

                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

        return y_true, y_pred

    # ===============================
    # RUN TRAINING
    # ===============================
    EPOCHS = 20
    for epoch in range(EPOCHS):
        loss, acc = train_one_epoch()
        scheduler.step()
        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {loss:.4f} | Acc: {acc:.4f}")

    # ===============================
    # EVALUATION
    # ===============================
    y_true, y_pred = evaluate()
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    torch.save(model.state_dict(), "checkpoints/swin_t_rafdb.pth")
    print("Model saved to checkpoints/swin_t_rafdb.pth")


if __name__ == "__main__":
    main()

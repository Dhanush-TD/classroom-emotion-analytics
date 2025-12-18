import torch
import torch.nn as nn
import timm
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler

# ===============================
# DEVICE
# ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ===============================
# PATHS
# ===============================
TRAIN_DIR = "raf_db/train"
CHECKPOINT_IN = "checkpoints/swin_t_rafdb.pth"
CHECKPOINT_OUT = "checkpoints/swin_t_rafdb_finetuned.pth"

# ===============================
# TRANSFORMS (LIGHT AUGMENTATION)
# ===============================
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.15, contrast=0.15),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# ===============================
# DATASET
# ===============================
train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)

targets = np.array(train_dataset.targets)
class_counts = np.bincount(targets)

class_weights = 1.0 / class_counts
class_weights = class_weights / class_weights.sum()
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)

sampler = WeightedRandomSampler(
    class_weights[targets].cpu().numpy(),
    len(targets),
    replacement=True
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    sampler=sampler,
    num_workers=0
)

# ===============================
# MODEL
# ===============================
model = timm.create_model(
    "swin_tiny_patch4_window7_224",
    pretrained=False,
    num_classes=7
)

model.load_state_dict(torch.load(CHECKPOINT_IN, map_location=device))
model.to(device)

# 🔒 FREEZE BACKBONE
for name, param in model.named_parameters():
    if "head" not in name:
        param.requires_grad = False

# ===============================
# OPTIMIZER
# ===============================
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-4,
    weight_decay=1e-4
)

# ===============================
# TRAIN LOOP
# ===============================
EPOCHS = 5
model.train()

for epoch in range(EPOCHS):
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

    acc = correct / total
    print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {total_loss:.4f} | Acc: {acc:.4f}")

# ===============================
# SAVE NEW MODEL
# ===============================
torch.save(model.state_dict(), CHECKPOINT_OUT)
print("Fine-tuned model saved to:", CHECKPOINT_OUT)

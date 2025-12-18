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
CHECKPOINT_IN = "checkpoints/swin_t_rafdb_finetuned.pth"
CHECKPOINT_OUT = "checkpoints/swin_t_rafdb_finetuned_sad.pth"

# ===============================
# TRANSFORMS
# ===============================
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# ===============================
# DATASET
# ===============================
dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
targets = np.array(dataset.targets)

class_counts = np.bincount(targets)

# Boost Sad (class 4) slightly
class_weights = 1.0 / class_counts
class_weights[4] *= 1.5   # <-- key line
class_weights = class_weights / class_weights.sum()

criterion = nn.CrossEntropyLoss(
    weight=torch.tensor(class_weights).float().to(device)
)

sampler = WeightedRandomSampler(
    class_weights[targets],
    len(targets),
    replacement=True
)

loader = DataLoader(
    dataset,
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

model.load_state_dict(
    torch.load(CHECKPOINT_IN, map_location=device)
)
model.to(device)

# Freeze backbone
for name, param in model.named_parameters():
    if "head" not in name:
        param.requires_grad = False

optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=8e-5,
    weight_decay=1e-4
)

# ===============================
# TRAIN
# ===============================
EPOCHS = 3
model.train()

for epoch in range(EPOCHS):
    loss_sum = 0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] "
        f"Loss: {loss_sum:.2f} "
        f"Acc: {correct/total:.4f}"
    )

# ===============================
# SAVE
# ===============================
torch.save(model.state_dict(), CHECKPOINT_OUT)
print("Saved:", CHECKPOINT_OUT)

# CT POSE ESTIMATION TRAINING
# updated: consistent MAE computation

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
import numpy as np
import matplotlib.pyplot as plt
from ct_sweep_dataset import CTSweepDataset, parse_dataset
from ct_pose_estimation_model import PoseEstRegression

# ----------- CONFIG -----------

train_path = "/Users/emiz/Desktop/ct_sweep_datasets/training"
test_path = "/Users/emiz/Desktop/ct_sweep_datasets/validation"
batch_size = 16
num_epochs = 50
learning_rate = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
new_model = True
angle_type = "pitch"

# ----------- DATASET -----------

train_imgs, train_labels = parse_dataset(train_path, angle_type)
train_dataset = CTSweepDataset(train_imgs, train_labels)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_imgs, test_labels = parse_dataset(test_path, angle_type)
test_dataset = CTSweepDataset(test_imgs, test_labels)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ----------- MODEL -----------

model = PoseEstRegression().to(device)
# if not new_model:
#     model.load_state_dict(torch.load("/Users/emiz/Desktop/resnet18_smoothl1_pitch_50ep.pt"))

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.SmoothL1Loss(beta=0.5)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3) #, verbose=True)

best_test_mae = float('inf')
best_epoch = 0
best_model_path = "/Users/emiz/Desktop/resnet18_smoothl1_PITCH3.pt"

train_maes = []
test_maes = []
lr_history = []

# ----------- TRAINING LOOP -----------

for epoch in range(1, num_epochs + 1):
    model.train()
    start_time = time.time()

    train_abs_error = 0.0
    train_loss = 0.0
    train_samples = 0

    for imgs, angles in train_loader:
        imgs, angles = imgs.to(device), angles.to(device)
        preds = model(imgs)
        loss = criterion(preds, angles)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_abs_error += torch.sum(torch.abs(preds - angles)).item()
        train_loss += loss.item() * imgs.size(0)
        train_samples += imgs.size(0)

    avg_train_mae = train_abs_error / train_samples
    avg_train_loss = train_loss / train_samples

    # ----------- VALIDATION -----------

    model.eval()
    test_abs_error = 0.0
    test_loss = 0.0
    test_samples = 0

    with torch.no_grad():
        for imgs, angles in test_loader:
            imgs, angles = imgs.to(device), angles.to(device)
            preds = model(imgs)
            loss = criterion(preds, angles)

            test_abs_error += torch.sum(torch.abs(preds - angles)).item()
            test_loss += loss.item() * imgs.size(0)
            test_samples += imgs.size(0)

    avg_test_mae = test_abs_error / test_samples
    avg_test_loss = test_loss / test_samples

    # ----------- LOGGING & SCHEDULER -----------

    train_maes.append(avg_train_mae)
    test_maes.append(avg_test_mae)
    current_lr = optimizer.param_groups[0]['lr']
    lr_history.append(current_lr)
    scheduler.step(avg_test_mae)

    if avg_test_mae < best_test_mae:
        best_test_mae = avg_test_mae
        best_epoch = epoch
        torch.save(model.state_dict(), best_model_path)

    elapsed = time.time() - start_time
    print(f"Epoch {epoch:02d} | Train L1: {avg_train_loss:.4f}, MAE: {avg_train_mae:.2f}° | "
          f"Test L1: {avg_test_loss:.4f}, MAE: {avg_test_mae:.2f}° | LR: {current_lr:.6f} | Time: {elapsed:.2f}s")

print(f"✅ Best model saved at epoch {best_epoch} with MAE: {best_test_mae:.2f}°")

# ----------- VISUALIZATION -----------

plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), train_maes, label="Train MAE", marker='o')
plt.plot(range(1, num_epochs + 1), test_maes, label="Validation MAE", marker='x')
plt.xlabel("Epoch")
plt.ylabel("MAE (°)")
plt.title("MAE Over Training (Roll Prediction)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import matplotlib.pyplot as plt

# load dataset
data = np.load("electric_model_dataset.npy", allow_pickle=True)

# load target normalization stats
target_stats = pd.read_csv("target_stats.csv", index_col=0).squeeze("columns")
target_mean = target_stats["mean"]
target_std = target_stats["std"]

# Define PyTorch Dataset
class ElectricPriceDataset(Dataset):
    def __init__(self, data):
        self.samples = data

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        price, window = self.samples[idx]
        x = torch.tensor(window, dtype=torch.float32)
        y = torch.tensor(price, dtype=torch.float32)
        return x, y

# CNN Model Definition
class SimpleCNN(nn.Module):
    def __init__(self, in_channels):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1) #fix the shape of the input tensor
        x = self.dropout(self.relu(self.conv1(x)))
        x = self.pool(x).squeeze(-1)
        return self.fc(x).squeeze(-1)

# prepare data loaders
dataset = ElectricPriceDataset(data)
train_size = int(0.8 * len(dataset))
train_indices = list(range(train_size))
val_indices = list(range(train_size, len(dataset)))

train_ds = Subset(dataset, train_indices)
val_ds = Subset(dataset, val_indices)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32)

# determine the number of features from one sample 
num_features = data[0][1].shape[1]
model = SimpleCNN(in_channels=num_features)

# training setup
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
n_epochs = 200
train_losses = []
val_losses = []

# training loop
for epoch in range(n_epochs):
    model.train()
    running_loss = 0.0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        preds = model(xb)
        loss = loss_fn(preds, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # validation loop
    model.eval()
    val_loss = 0.0
    preds_all = []
    y_all = []
    with torch.no_grad():
        for xb, yb in val_loader:
            preds = model(xb)
            val_loss += loss_fn(preds, yb).item()
            # convert predictions and targets to numpy arrays
            preds_all.extend(preds.detach().cpu().numpy())
            y_all.extend(yb.detach().cpu().numpy())

    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    # reverse normalization
    preds_real = np.array(preds_all) * target_std + target_mean
    targets_real = np.array(y_all) * target_std + target_mean
    mae_real = np.mean(np.abs(preds_real - targets_real))

    print(f"Epoch {epoch+1}/{n_epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f} - MAE: {mae_real:.2f} $/MWh")

# save the trained model
torch.save(model.state_dict(), "cnn_model.pth")

# plot training curves
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss (Standardized)")
plt.title("Training Progress")
plt.legend()
plt.grid(True)
plt.show()

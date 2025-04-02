import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load dataset and stats
data = np.load("electric_model_dataset.npy", allow_pickle=True)
target_stats = pd.read_csv("target_stats.csv", index_col=0).squeeze("columns")
target_mean = target_stats["mean"]
target_std = target_stats["std"]

# define dataset
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

# define CNN model
class SimpleCNN(nn.Module):
    def __init__(self, in_channels):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.dropout(self.relu(self.conv1(x)))
        x = self.pool(x).squeeze(-1)
        return self.fc(x).squeeze(-1)

# load model
num_features = data[0][1].shape[1]
model = SimpleCNN(in_channels=num_features)
model.load_state_dict(torch.load("cnn_model.pth"))
model.eval()

# prepare test data
dataset = ElectricPriceDataset(data)
train_size = int(0.8 * len(dataset))
test_indices = list(range(train_size, len(dataset)))
test_ds = Subset(dataset, test_indices)
test_loader = DataLoader(test_ds, batch_size=16)

# evaluate model
all_preds = []
all_targets = []

with torch.no_grad():
    for xb, yb in test_loader:
        preds = model(xb)
        all_preds.extend(preds.detach().cpu().numpy())
        all_targets.extend(yb.detach().cpu().numpy())

# reverse normaliz
all_preds = np.array(all_preds) * target_std + target_mean
all_targets = np.array(all_targets) * target_std + target_mean

# get evaluation metrics
mae = mean_absolute_error(all_targets, all_preds)
rmse = np.sqrt(mean_squared_error(all_targets, all_preds))

def symmetric_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred))
    smape = 2 * np.abs(y_true - y_pred) / np.maximum(denominator, 1e-8)
    return 100 * np.mean(smape)

smape = symmetric_mape(all_targets, all_preds)
accuracy = 100 - smape

# print results
print(f"\nEvaluation Metrics:")
print(f"  MAE   = {mae:.2f} $/MWh")
print(f"  RMSE  = {rmse:.2f} $/MWh")
print(f"  SMAPE = {smape:.2f}%")
print(f"  Estimated Accuracy = {accuracy:.2f}%")

# plot results
plt.figure(figsize=(12, 6))
plt.plot(all_targets[:100], label="Actual Prices", marker='o')
plt.plot(all_preds[:100], label="Predicted Prices", marker='x')
plt.title("Predicted vs Actual Electricity Prices (First 100 Samples)")
plt.xlabel("Test Sample Index")
plt.ylabel("Price ($/MWh)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

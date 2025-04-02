import numpy as np
import pandas as pd  # type: ignore
import torch  # type: ignore
import torch.nn as nn  # type: ignore
from torch.utils.data import Dataset, DataLoader, random_split  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import time



# load dataset
data = np.load("electric_model_dataset.npy", allow_pickle=True)

# load target normalization stats
target_stats = pd.read_csv("target_stats.csv", index_col=0).squeeze("columns")
target_mean = target_stats["mean"]
target_std = target_stats["std"]
print(f"Target mean: {target_mean:.4f}, Target std: {target_std:.4f}")

# PyTorch Dataset
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

# Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, downsample=False):
        super(ResidualBlock, self).__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)

        if downsample:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample:
            identity = self.downsample(x)

        out += identity
        return self.relu(out)

# ResNet1D Model
class ResNet1D(nn.Module):
    def __init__(self, in_channels, num_blocks, num_classes=1):
        super(ResNet1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(64, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(128, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(256, 512, num_blocks[3], stride=2)

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride=stride, downsample=True))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Ensure correct shape for CNN
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.global_avg_pool(x).squeeze(-1)
        return self.fc(x).squeeze(-1)

# Prepare Data Loaders
torch.manual_seed(42)  # set a fixed seed for reproducibility
dataset = ElectricPriceDataset(data)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, test_size])
np.save("test_indices.npy", val_ds.indices)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=16, num_workers=0)
print(f"Training set size: {train_size}, Validation set size: {test_size}")

# determine the number of features from one sample

num_features = data[0][1].shape[1]
print(f"Number of input features: {num_features}")

model = ResNet1D(in_channels=num_features, num_blocks=[1, 1, 1, 1]) 

# Training Setup
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
n_epochs = 50
train_losses = []
val_losses = []
batch_count = 0
start_time = time.time()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model.to(device)


# Training Loop
for epoch in range(n_epochs):
    batch_count += 1
    model.train()
    running_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        preds = model(xb)
        loss = loss_fn(preds, yb)
        loss.backward()
        optimizer.step()
        print(f"Batch processed in {time.time() - start_time:.2f} seconds")
        start_time = time.time()




    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    #  Validation Loop
    model.eval()
    val_loss = 0.0
    preds_all = []
    y_all = []
    with torch.no_grad():
        for xb, yb in val_loader:
            preds = model(xb)
            val_loss += loss_fn(preds, yb).item()
            preds_all.extend(preds.detach().cpu().numpy())
            y_all.extend(yb.detach().cpu().numpy())

    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    #  Reverse Normalization and MAE
    
    preds_real = np.array(preds_all) * target_std + target_mean
    targets_real = np.array(y_all) * target_std + target_mean
    mae_real = np.mean(np.abs(preds_real - targets_real))

    print(f"Epoch {epoch+1}/{n_epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f} - MAE: {mae_real:.2f} $/MWh")

# Save the Trained Model
torch.save(model.state_dict(), "resnet_model.pth")

# Plot Training Curves
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss (Standardized)")
plt.title("Training Progress - ResNet1D")
plt.legend()
plt.grid(True)
plt.show()

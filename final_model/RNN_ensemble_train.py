import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import optuna
from optuna.integration.botorch import BoTorchSampler
import shutil
import matplotlib
matplotlib.use('Agg')

# Create necessary folders
os.makedirs("graphs", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Load data
x_lstm = np.load("x_lstm.npy")
y_lstm = np.load("y_lstm.npy")

train_len = int(len(x_lstm) * 0.75)
X_train = torch.tensor(x_lstm[:train_len], dtype=torch.float32)
y_train = torch.tensor(y_lstm[:train_len], dtype=torch.float32)
X_test = torch.tensor(x_lstm[train_len:], dtype=torch.float32)
y_test = y_lstm[train_len:]

# DataLoader
def get_loaders(batch_size):
    train_ds = TensorDataset(X_train, y_train)
    return DataLoader(train_ds, batch_size=batch_size, shuffle=False)

# Models
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

class EnsembleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 1)

    def forward(self, preds):
        return self.fc(preds)

# Training and evaluation functions
def train_epoch(model, train_dl, optimizer, loss_fn):
    model.train()
    for xb, yb in train_dl:
        optimizer.zero_grad()
        preds = model(xb).squeeze()
        loss = loss_fn(preds, yb)
        loss.backward()
        optimizer.step()

@torch.no_grad()
def evaluate_model(model, X, y_true):
    model.eval()
    preds = model(X).squeeze().numpy()
    loss = np.mean((preds - y_true) ** 2)
    mae = mean_absolute_error(y_true, preds)
    rmse = np.sqrt(mean_squared_error(y_true, preds))
    return loss, mae, rmse, preds

# Plot metrics
def plot_metrics(metrics_dict, ylabel, savepath):
    plt.figure(figsize=(10,6))
    for label, values in metrics_dict.items():
        plt.plot(values, label=label)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(f"Validation {ylabel} per Epoch")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(savepath)
    plt.close()

# Optuna objective
all_maes = []

def objective(trial):
    print(f"\nStarting Trial {trial.number}...")
    input_size = X_train.shape[2]
    batch_size = trial.suggest_categorical('batch_size', [32, 64])

    hidden_lstm = trial.suggest_int('hidden_size_lstm', 32, 128)
    dropout_lstm = trial.suggest_float('dropout_lstm', 0.0, 0.5)

    hidden_gru = trial.suggest_int('hidden_size_gru', 32, 128)
    dropout_gru = trial.suggest_float('dropout_gru', 0.0, 0.5)

    train_dl = get_loaders(batch_size)

    lstm = LSTMModel(input_size, hidden_lstm, num_layers=2, dropout=dropout_lstm)
    lstm_optimizer = optim.Adam(lstm.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    val_losses = {"LSTM": [], "GRU": [], "MLP": []}
    val_maes = {"LSTM": [], "GRU": [], "MLP": []}
    val_rmses = {"LSTM": [], "GRU": [], "MLP": []}

    for epoch in range(10):
        train_epoch(lstm, train_dl, lstm_optimizer, loss_fn)
        loss, mae, rmse, _ = evaluate_model(lstm, X_test, y_test)
        val_losses["LSTM"].append(loss)
        val_maes["LSTM"].append(mae)
        val_rmses["LSTM"].append(rmse)
        print(f"Trial {trial.number} | LSTM Epoch {epoch+1}/10 - MAE: {mae:.2f} €/MWh")

    _, _, _, lstm_preds = evaluate_model(lstm, X_test, y_test)

    gru = GRUModel(input_size, hidden_gru, num_layers=2, dropout=dropout_gru)
    gru_optimizer = optim.Adam(gru.parameters(), lr=0.001)

    for epoch in range(10):
        train_epoch(gru, train_dl, gru_optimizer, loss_fn)
        loss, mae, rmse, _ = evaluate_model(gru, X_test, y_test)
        val_losses["GRU"].append(loss)
        val_maes["GRU"].append(mae)
        val_rmses["GRU"].append(rmse)
        print(f"Trial {trial.number} | GRU Epoch {epoch+1}/10 - MAE: {mae:.2f} €/MWh")

    _, _, _, gru_preds = evaluate_model(gru, X_test, y_test)

    ensemble_input = torch.tensor(np.vstack((lstm_preds, gru_preds)).T, dtype=torch.float32)
    ensemble_targets = torch.tensor(y_test, dtype=torch.float32)

    ensemble_dataset = TensorDataset(ensemble_input, ensemble_targets)
    ensemble_loader = DataLoader(ensemble_dataset, batch_size=batch_size, shuffle=False)

    mlp = EnsembleMLP()
    mlp_optimizer = optim.Adam(mlp.parameters(), lr=0.001)

    for epoch in range(10):
        mlp.train()
        for xb, yb in ensemble_loader:
            mlp_optimizer.zero_grad()
            preds = mlp(xb).squeeze()
            loss = loss_fn(preds, yb)
            loss.backward()
            mlp_optimizer.step()
        loss, mae, rmse, _ = evaluate_model(mlp, ensemble_input, y_test)
        val_losses["MLP"].append(loss)
        val_maes["MLP"].append(mae)
        val_rmses["MLP"].append(rmse)
        print(f"Trial {trial.number} | MLP Epoch {epoch+1}/10 - MAE: {mae:.2f} €/MWh")

    final_preds = mlp(ensemble_input).squeeze().detach().numpy()
    ensemble_mae = mean_absolute_error(y_test, final_preds)

    print(f"Trial {trial.number} Finished - Final Ensemble MAE: {ensemble_mae:.2f} €/MWh")

    # Save graphs for each trial
    plot_metrics(val_losses, "Loss", f"graphs/trial_{trial.number}_loss.png")
    plot_metrics(val_maes, "MAE", f"graphs/trial_{trial.number}_mae.png")
    plot_metrics(val_rmses, "RMSE", f"graphs/trial_{trial.number}_rmse.png")

    # Save models for each trial
    torch.save(lstm.state_dict(), f"models/lstm_trial_{trial.number}.pth")
    torch.save(gru.state_dict(), f"models/gru_trial_{trial.number}.pth")
    torch.save(mlp.state_dict(), f"models/mlp_trial_{trial.number}.pth")

    # Save metrics for each trial
    with open(f"results/trial_{trial.number}_final_metrics.txt", "w") as f:
        for model in ["LSTM", "GRU", "MLP"]:
            f.write(f"{model} - Final Loss: {val_losses[model][-1]:.4f}, Final MAE: {val_maes[model][-1]:.4f}, Final RMSE: {val_rmses[model][-1]:.4f}\n")

    trial.set_user_attr("val_losses", val_losses)
    trial.set_user_attr("val_maes", val_maes)
    trial.set_user_attr("val_rmses", val_rmses)

    all_maes.append(ensemble_mae)

    return ensemble_mae

# Optuna study
sampler = BoTorchSampler(seed=42)
study = optuna.create_study(direction='minimize', sampler=sampler)
try:
    study.optimize(objective, n_trials=15)
except KeyboardInterrupt:
    print("\nEarly stopping triggered. Saving best results so far...")

# Save best trial separately
best = study.best_trial
print(f"Best Trial: {best.number} with MAE: {best.value:.2f} €/MWh")
print(f"Best Hyperparameters: {best.params}")

plot_metrics(best.user_attrs["val_losses"], "Loss", "graphs/best_trial_loss.png")
plot_metrics(best.user_attrs["val_maes"], "MAE", "graphs/best_trial_mae.png")
plot_metrics(best.user_attrs["val_rmses"], "RMSE", "graphs/best_trial_rmse.png")

shutil.copyfile(f"models/lstm_trial_{best.number}.pth", "models/best_lstm.pth")
shutil.copyfile(f"models/gru_trial_{best.number}.pth", "models/best_gru.pth")
shutil.copyfile(f"models/mlp_trial_{best.number}.pth", "models/best_mlp.pth")

with open("results/best_trial_final_metrics.txt", "w") as f:
    for model in ["LSTM", "GRU", "MLP"]:
        f.write(f"{model} - Final Loss: {best.user_attrs['val_losses'][model][-1]:.4f}, Final MAE: {best.user_attrs['val_maes'][model][-1]:.4f}, Final RMSE: {best.user_attrs['val_rmses'][model][-1]:.4f}\n")

print("All best trial graphs, best model weights, and best metrics saved")

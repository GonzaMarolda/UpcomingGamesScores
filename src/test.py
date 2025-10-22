import os
import torch
import torch.nn as nn
from dataset import get_dataloaders
from model import GameScorePredictor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODELS_DIR = "../trained_models/models"

model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith(".pth")]
if not model_files:
    raise FileNotFoundError(f"Models not found in {MODELS_DIR}")

print("Available models:")
for i, f in enumerate(model_files):
    print(f"{i}: {f}")

idx = int(input("Select the index of the desired model: "))
model_path = os.path.join(MODELS_DIR, model_files[idx])
print(f"Loading model: {model_files[idx]}")

train_loader, test_loader, input_columns = get_dataloaders()

model = GameScorePredictor(input_columns).to(DEVICE)
model.load_state_dict(torch.load(model_path))
model.eval()

mse_loss = nn.MSELoss()
mae_loss = nn.L1Loss()

mse_total = 0.0
mae_total = 0.0
num_samples = 0

with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        outputs = model(inputs)
        mse_total += mse_loss(outputs, targets).item() * inputs.size(0)
        mae_total += mae_loss(outputs, targets).item() * inputs.size(0)
        num_samples += inputs.size(0)

mse_total /= num_samples
mae_total /= num_samples

print(f"\nResults using test set:")
print(f"MSE: {mse_total:.4f}")
print(f"MAE: {mae_total:.4f}")
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_dataloaders
from model import GameScorePredictor
import os
import json
import time

EPOCHS = 200
LR = 0.001
DEVICE = torch.device("cuda")

train_loader, test_loader, column_names = get_dataloaders()

model = GameScorePredictor(column_names).to(DEVICE)
criterion = nn.MSELoss()
# Weight decay makes weights smaller over time, helping to prevent overfitting. Weights can grow large to adjust too much to the training data.
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

final_train_loss = float("inf")
best_val_loss = float("inf")
best_model_state = None

for epoch in range(EPOCHS):
    model.train() # Changes how the model behaves for advanced configurations. Not needed for this simple model but good practice.
    running_loss = 0.0

    scaler = torch.cuda.amp.GradScaler()
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE) # Move data to the same device as model

        # Forward
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward
        optimizer.zero_grad() # Clear previous gradients
        loss.backward() # Compute gradients and store in the parameters
        optimizer.step() # Update weights using the gradients saved in the parameters 

        running_loss += loss.item() * inputs.size(0) # Accumulated loss for the batch

    epoch_loss = running_loss / len(train_loader.dataset) # Mean loss for the epoch

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad(): # No need to build the computational graph used for backpropagation
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item() * inputs.size(0)

    val_loss /= len(test_loader.dataset)

    if val_loss < best_val_loss:
        final_train_loss = epoch_loss
        best_val_loss = val_loss
        best_model_state = model.state_dict()

    print(f"Epoch [{epoch+1}/{EPOCHS}] - Train Loss: {epoch_loss:.4f} - Val Loss: {val_loss:.4f}")


# Save the model
SAVE_PATH = "../trained_models/models"

existing_models = [f for f in os.listdir(SAVE_PATH) if f.startswith("gamescore_model_v") and f.endswith(".pth")]
version = len(existing_models) + 1  # next version number

model_filename = f"gamescore_model_v{version}.pth"
model_path = os.path.join(SAVE_PATH, model_filename)

torch.save(model.state_dict(), model_path)
print(f"Model saved in {model_path}")

# Compute MSE and MAE on test set
mse_loss_fn = nn.MSELoss()
mae_loss_fn = nn.L1Loss()

mse_total = 0.0
mae_total = 0.0
num_samples = 0

model.eval()
with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        outputs = model(inputs)
        mse_total += mse_loss_fn(outputs, targets).item() * inputs.size(0)
        mae_total += mae_loss_fn(outputs, targets).item() * inputs.size(0)
        num_samples += inputs.size(0)

mse_total /= num_samples
mae_total /= num_samples

# Save the results
RESULTS_PATH = "../trained_models/results"

results = {
    "model_name": model_filename.split(".")[0],
    "train_loss": final_train_loss,
    "val_loss": best_val_loss,
    "MSE": mse_total,
    "MAE": mae_total,
    "epoch": EPOCHS,
    "learning_rate": LR
}

json_file = os.path.join(RESULTS_PATH, f"results_{results['model_name']}.json")

with open(json_file, "w") as f:
    json.dump(results, f, indent=4)
print(f"Results saved in {json_file}")
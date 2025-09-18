import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_dataloaders
from model import GameScorePredictor
import os

EPOCHS = 30
LR = 0.01
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader, test_loader, input_dim = get_dataloaders()

model = GameScorePredictor(input_dim).to(DEVICE)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    model.train() # Changes how the model behaves for advanced configurations. Not needed for this simple model but good practice.
    running_loss = 0.0

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

    print(f"Epoch [{epoch+1}/{EPOCHS}] - Train Loss: {epoch_loss:.4f} - Val Loss: {val_loss:.4f}")


# Save the model
SAVE_PATH = "../trained_models/models"

existing_models = [f for f in os.listdir(SAVE_PATH) if f.startswith("gamescore_model_v") and f.endswith(".pth")]
version = len(existing_models) + 1  # next version number

model_filename = f"gamescore_model_v{version}.pth"
model_path = os.path.join(SAVE_PATH, model_filename)

torch.save(model.state_dict(), model_path)
print(f"Modelo guardado en {model_path}")
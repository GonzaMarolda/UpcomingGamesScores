import glob 
from src.dataset import get_dataloaders
import torch
import os
from src.model import GameScorePredictor  # Adjust import if model class is named differently

def load_model(model_path, input_dim):
    model = GameScorePredictor(input_dim)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def get_latest_model_path():
    model_files = glob.glob(os.path.join("trained_models", "models", "gamescore_model_v*.pth"))
    if not model_files:
        return None
    model_files.sort(key=lambda x: int(x.split('_v')[-1].split('.pth')[0]))
    return model_files[-1]

if __name__ == "__main__":
    model_path = get_latest_model_path()

    train_loader, test_loader, input_dim = get_dataloaders(isProduction=True)   
    model = load_model(model_path, input_dim)
    print("GameScorePredictor model loaded successfully.")

    while True:
        print("Insert game data for prediction\n"
        "enter 'restart' to start over with the game data")

        price = None
        while True:
            user_input = input("Price (0-60 no decimals): ")
            if user_input.lower() == 'restart':
                break

            if user_input.isdigit():
                price = int(user_input)
                if 0 <= price <= 60:
                    price = float(price) / 60.0 
                    break
                else:
                    print("Invalid input. Please enter a price between 0 and 60.")
            else:
                print("Invalid input. Please enter a whole number.")

        if price is None: continue

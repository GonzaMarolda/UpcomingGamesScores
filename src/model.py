import torch
import torch.nn as nn

class GameScorePredictor(nn.Module):
    def __init__(self, input_dim):
        super(GameScorePredictor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  
        )
    
    def forward(self, x):
        return self.net(x)
import torch
import torch.nn as nn

class GameScorePredictor(nn.Module):
    def __init__(self, input_dim):
        super(GameScorePredictor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)  
        )
    
    def forward(self, x):
        return self.net(x)
import torch
import torch.nn as nn

class GameScorePredictor(nn.Module):
    def __init__(self, input_dim, tags_columns_amount):
        super(GameScorePredictor, self).__init__()

        tags_total = 300  
        embedding_dim = 16

        self.tag_embedding = nn.Embedding(num_embeddings=tags_total+1, embedding_dim=embedding_dim, padding_idx=0)
        self.tags_columns_amount = tags_columns_amount

        input_dim = input_dim - tags_columns_amount + embedding_dim
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
        tags = x[:, -self.tags_columns_amount:].long()        
        other_features = x[:, : -self.tags_columns_amount].float()

        emb = self.tag_embedding(tags)  

        summed = emb.sum(dim=1)
        counts = (tags != 0).sum(dim=1).float().unsqueeze(1).clamp(min=1.0)
        tag_vec = summed / counts  

        final_input = torch.cat([other_features, tag_vec], dim=1)  

        return self.net(final_input)
import torch
import torch.nn as nn
import globals

class GameScorePredictor(nn.Module):
    def __init__(self, input_columns):
        super(GameScorePredictor, self).__init__()

        input_dim = len(input_columns)

        # Initialize column indices
        self.initial_columns_indices = {}
        self.initial_columns_indices["tags"] = [i for i, col in enumerate(input_columns) if col.startswith('tag_')]
        self.initial_columns_indices["publishers"] = [i for i, col in enumerate(input_columns) if col.startswith('publisher_')]
        self.initial_columns_indices["other"] = [i for i, col in enumerate(input_columns) if i not in self.initial_columns_indices["tags"] and i not in self.initial_columns_indices["publishers"]]

        # Embedding for tags
        tags_embedding_dim = 128
        self.tag_embedding = nn.Embedding(num_embeddings=globals.top_n_tags+1, embedding_dim=tags_embedding_dim, padding_idx=0)

        # Embedding for publishers
        publishers_embedding_dim = 32
        self.publisher_embedding = nn.Embedding(num_embeddings=globals.top_n_publishers+2, embedding_dim=publishers_embedding_dim, padding_idx=0) 

        input_dim = input_dim + tags_embedding_dim + publishers_embedding_dim - (globals.tags_columns_amount + globals.publishers_columns_amount)
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            #nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            # nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            # nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            # nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        other_features = x[:, self.initial_columns_indices["other"]].float()
        tag_embeds = self.get_tag_embeds(x) 
        publisher_embeds = self.get_publisher_embeds(x)

        final_input = torch.cat([other_features, tag_embeds, publisher_embeds], dim=1)

        return self.net(final_input)
    
    def get_tag_embeds(self, x):
        tags = x[:, self.initial_columns_indices["tags"]].long()

        weights = torch.linspace(1, 0.1, globals.top_n_tags + 1, device=tags.device, dtype=torch.float32)
        tag_weights = weights[tags]
        tag_weights = tag_weights.unsqueeze(-1) 

        tag_embeds = self.tag_embedding(tags)
        weighted_embeds = tag_embeds * tag_weights  

        # máscara para ignorar paddings (tags==0)
        mask = (tags != 0).unsqueeze(-1)  
        sum_embeds = (weighted_embeds * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1)  
        tag_embeds = sum_embeds / denom  

        return tag_embeds

    def get_publisher_embeds(self, x):
        publishers = x[:, self.initial_columns_indices["publishers"]].long()

        publisher_embeds = self.publisher_embedding(publishers)

        # máscara para ignorar paddings (publishers==0)
        mask = (publishers != 0).unsqueeze(-1)  
        sum_embeds = (publisher_embeds * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1)  
        publisher_embeds = sum_embeds / denom

        return publisher_embeds
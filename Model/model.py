import torch 
import torch.nn as nn

class RunkedModel(nn.Module):
    def __init__(self, num_item_features, num_context_features):
        super().__init__()

        hidden_size = 128

        self.item_emb = nn.Sequential(
            nn.Linear(num_item_features, hidden_size),
            nn.ReLU())
        
        self.context_emb = nn.Sequential(
            nn.Linear(num_context_features, hidden_size//2),
            nn.ReLU(),
            nn.LayerNorm(hidden_size//2),
            nn.Linear(hidden_size//2, hidden_size),
            nn.ReLU())
        
        self.encoder = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=8
                                               ,dim_feedforward=hidden_size
                                               , batch_first=True)
        

        self.score_predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.LayerNorm(hidden_size//2),
            nn.Linear(hidden_size//2, 1),
            nn.Sigmoid()
        )

    def forward(self, item_features, context_features):
        item_emb = self.item_emb(item_features)
        context_emb = self.context_emb(context_features)

        context_emb = context_emb.unsqueeze(1).repeat(1, 150, 1)
        combined = item_emb + context_emb

        enc_outp = self.encoder(combined)
        outp = combined + enc_outp

        score = self.score_predictor(outp).squeeze(-1)

        return score
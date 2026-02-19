import einops
import torch
import torch.nn as nn
from src.models.encoders import SinusoidalPosEmb

class ToyModel(nn.Module):
    def __init__(self, num_classes, dim=128):
        super().__init__()
        self.t_embs = SinusoidalPosEmb(dim)
        self.input_proj = nn.Linear(num_classes, dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=4, dim_feedforward=512, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        
        self.output_proj = nn.Linear(dim, num_classes)
        self.softmax = nn.Softmax(-1)

    def forward(self, t, x, return_probs=False):
        B, D, C = x.shape
        x = x+einops.repeat(torch.arange(D).reshape(-1, 1), "D C -> B D C", B=B)
        temb = self.t_embs(t) # (B, dim)
        
        h = self.input_proj(x) # (B, D, dim)
        
        h = h + temb.unsqueeze(1)
        
        h = self.transformer(h)
        
        out = self.output_proj(h)
        if return_probs: return self.softmax(out)
        return out
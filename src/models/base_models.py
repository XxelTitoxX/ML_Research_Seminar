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
    
class MLP(nn.Module):
    def __init__(self, inp_dim, hidden_dim=64, num_layers=5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input = nn.Sequential(nn.Linear(inp_dim+1, hidden_dim), nn.GELU())
        self.modules = []
        for _ in range(num_layers):
            self.modules.append(nn.Linear(hidden_dim, hidden_dim))
            self.modules.append(nn.GELU())
        self.mlp = nn.Sequential(*self.modules)
        self.out = nn.Linear(hidden_dim, inp_dim)

    def forward(self, t, x):
        if len(t.shape) == 0:
            t = t.repeat(x.shape[0])
        t = t.unsqueeze(0).reshape(-1,1)
        x = torch.cat([x, t], dim=-1)
        x = self.input(x)
        x = self.mlp(x)
        return self.out(x)
    
    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info
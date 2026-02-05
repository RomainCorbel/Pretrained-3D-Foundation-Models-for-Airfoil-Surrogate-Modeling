import torch
import torch.nn as nn
from typing import Optional
from models.MLP import MLP

class GlobalFusion(nn.Module):
    def __init__(self, W_global_in: int, W_fuse: int):
        super().__init__()
        self.global_proj = MLP(
            [W_global_in, 64, 64, W_fuse],
            batch_norm=False
        )
        self.gate = MLP(
            [W_fuse, W_fuse],
            batch_norm=False
        )

    def forward(
        self,
        local: torch.Tensor,
        g: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ):
        N, device = local.size(0), local.device
        P = local                                 
        G = self.global_proj(g.unsqueeze(0))       
        G = G.expand(N, -1)                        
        gate = torch.sigmoid(self.gate(P))     
        H = P + gate * G
        return H

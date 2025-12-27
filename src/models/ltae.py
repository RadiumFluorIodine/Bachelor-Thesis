import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LTAE(nn.Module):
    def __init__(self, in_channels, n_head=16, d_k=8):
        super(LTAE, self).__init__()
        self.in_channels = in_channels
        self.n_head = n_head
        self.d_k = d_k
        self.d_in = in_channels // n_head

        # Key Generation & Positional Encoding
        self.dense_key = nn.Linear(self.d_in, d_k, bias=False)
        self.pos_enc = nn.Linear(1, self.d_in, bias=False)
        
        # Master Query (Parameter yang dipelajari)
        self.query = nn.Parameter(torch.randn(n_head, d_k))

    def forward(self, x, days):
        # x: (B, T, C, H, W) -> (BHW, T, C)
        b, t, c, h, w = x.shape
        x = x.permute(0, 3, 4, 1, 2).reshape(b*h*w, t, c)
        
        # Channel Grouping
        x = x.view(b*h*w, t, self.n_head, self.d_in)
        
        # Positional Encoding
        days = days.repeat(b*h*w, 1).unsqueeze(-1) # (BHW, T, 1)
        pe = self.pos_enc(torch.sin(days / 1000.0 * 2 * np.pi)) # (BHW, T, Din)
        pe = pe.unsqueeze(2) # Broadcast ke head
        
        # Attention
        keys = self.dense_key(x + pe) # (BHW, T, H, K)
        scores = (self.query[None, None, :, :] * keys).sum(-1) / (self.d_k**0.5)
        attn = F.softmax(scores, dim=1) # (BHW, T, H)
        
        # Weighted Sum
        out = (x * attn.unsqueeze(-1)).sum(1) # (BHW, H, Din)
        
        # Reshape balik
        out = out.reshape(b*h*w, c).view(b, h, w, c).permute(0, 3, 1, 2)
        return out

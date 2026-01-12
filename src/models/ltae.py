# FILE: ltae.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LTAE(nn.Module):
    """
    Lightweight Temporal Attention Encoder - FIXED VERSION
    
    Expects:
    - x: (B, T, C, H, W) image features
    - temporal_encoding: (T, d_model) pre-computed sinusoidal encoding
    - valid_mask: (B, H, W) optional boolean mask
    """
    
    def __init__(self, in_channels, d_model=64, n_head=16, d_k=8):
        super(LTAE, self).__init__()
        self.in_channels = in_channels
        self.d_model = d_model
        self.n_head = n_head
        self.d_k = d_k
        self.d_in = in_channels // n_head

        # Ensure channel divisibility
        assert in_channels % n_head == 0, \
            f"in_channels ({in_channels}) must be divisible by n_head ({n_head})"

        # ✅ FIX: Positional Encoding Projection
        # Input: (T, d_model=64) -> Output: (T, d_in)
        self.pos_enc_proj = nn.Linear(d_model, self.d_in)
        
        # Key Generation
        self.dense_key = nn.Linear(self.d_in, d_k, bias=False)
        
        # Master Query (Learnable per head)
        self.query = nn.Parameter(torch.randn(n_head, d_k))
        
        # Layer Normalization
        self.norm = nn.LayerNorm(in_channels)

    def forward(self, x, temporal_encoding, valid_mask=None):
        """
        Args:
            x: (B, T, C, H, W) temporal image sequence
            temporal_encoding: (T, d_model) sinusoidal positional encoding
            valid_mask: (B, H, W) optional boolean mask for valid pixels
            
        Returns:
            out: (B, C, H, W) temporally aggregated features
        """
        b, t, c, h, w = x.shape
        
        # 1. Normalize input
        x = self.norm(x.permute(0, 1, 3, 4, 2)).permute(0, 1, 4, 2, 3)
        
        # 2. Reshape for spatial flattening: (B*H*W, T, C)
        x = x.permute(0, 3, 4, 1, 2).reshape(b*h*w, t, c)
        
        # 3. ✅ FIX: Project temporal encoding (NOT re-compute!)
        # temporal_encoding: (T, d_model) -> (T, d_in)
        pe = self.pos_enc_proj(temporal_encoding)  # (T, d_in)
        pe = pe.unsqueeze(0).expand(b*h*w, -1, -1)  # (BHW, T, d_in)
        
        # 4. Channel grouping for multi-head attention: (BHW, T, n_head, d_in)
        x_grouped = x.view(b*h*w, t, self.n_head, self.d_in)
        
        # 5. Add positional encoding (broadcast across heads)
        pe_grouped = pe.unsqueeze(2).expand(-1, -1, self.n_head, -1)
        x_with_pe = x_grouped + pe_grouped
        
        # 6. Generate keys: (BHW, T, n_head, d_k)
        keys = self.dense_key(x_with_pe)
        
        # 7. Compute attention scores: (BHW, T, n_head)
        scores = torch.einsum('ijk,btij->bti', self.query.unsqueeze(0), keys)
        scores = scores / (self.d_k ** 0.5)  # Scale
        
        # 8. ✅ FIX: Apply valid mask if provided
        if valid_mask is not None:
            # valid_mask: (B, H, W) -> (BHW,)
            valid_mask_flat = valid_mask.view(b*h*w)
            # Expand for temporal dimension: (BHW, T, 1)
            mask_t = valid_mask_flat.unsqueeze(1).unsqueeze(2)
            # Set scores to very negative where invalid
            scores = scores.masked_fill(~mask_t, float('-inf'))
        
        # 9. Softmax over temporal dimension
        attn = F.softmax(scores, dim=1)  # (BHW, T, n_head)
        attn = torch.nan_to_num(attn, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 10. Aggregate using attention weights
        attn_expanded = attn.unsqueeze(-1)  # (BHW, T, n_head, 1)
        weighted = x_grouped * attn_expanded
        out = weighted.sum(dim=1)  # (BHW, n_head, d_in)
        
        # 11. Reshape back: (BHW, n_head, d_in) -> (B, C, H, W)
        out = out.reshape(b*h*w, c)
        out = out.view(b, h, w, c).permute(0, 3, 1, 2)
        
        return out

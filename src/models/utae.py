# FILE: utae.py
import torch
import torch.nn as nn
from ltae import LTAE

class ConvBlock(nn.Module):
    """Standard 2D Convolution Block with BatchNorm and ReLU"""
    
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class UTAE(nn.Module):
    """
    U-Net with Temporal Attention Encoder for AGB estimation
    
    Architecture:
    - Encoder: Time-distributed spatial convolutions
    - Bottleneck: Lightweight Temporal Attention Encoder (L-TAE)
    - Decoder: Standard U-Net with skip connections
    - Head: Single output for AGB prediction
    
    Input:
    - x: (B, T=12, C=10, H=128, W=128) temporal image sequence
    - temporal_encoding: (T=12, d_model=64) pre-computed sinusoidal encoding
    - valid_mask: (B, H=128, W=128) boolean mask for valid pixels
    
    Output:
    - (B, 1, H=128, W=128) AGB prediction
    """
    
    def __init__(self, in_c=10, out_c=1, base_c=32, d_model=64, n_head=16, d_k=8):
        super().__init__()
        
        assert base_c % n_head == 0, \
            "base_c*8 must be divisible by n_head for L-TAE"
        
        self.in_c = in_c
        self.base_c = base_c
        self.d_model = d_model
        
        # ============ ENCODER (Time-Distributed) ============
        self.enc1 = ConvBlock(in_c, base_c)
        self.enc2 = ConvBlock(base_c, base_c * 2)
        self.enc3 = ConvBlock(base_c * 2, base_c * 4)
        self.enc4 = ConvBlock(base_c * 4, base_c * 8)
        
        self.pool = nn.MaxPool2d(2)
        
        # ============ BOTTLENECK (Temporal Attention) ============
        self.ltae = LTAE(
            in_channels=base_c * 8,
            d_model=d_model,
            n_head=n_head,
            d_k=d_k
        )
        
        # ✅ FIX: Normalize after L-TAE
        self.ltae_bn = nn.BatchNorm2d(base_c * 8)
        
        # ============ DECODER (Standard U-Net) ============
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        # Decoder blocks with skip connections
        self.dec4 = ConvBlock(base_c * 8 + base_c * 8, base_c * 8)
        self.dec3 = ConvBlock(base_c * 8 + base_c * 4, base_c * 4)
        self.dec2 = ConvBlock(base_c * 4 + base_c * 2, base_c * 2)
        self.dec1 = ConvBlock(base_c * 2 + base_c, base_c)
        
        # ============ OUTPUT HEAD ============
        self.head = nn.Conv2d(base_c, out_c, kernel_size=1)
    
    def forward(self, x, temporal_encoding, valid_mask=None):
        """
        Args:
            x: (B, T, C, H, W) temporal image sequence
            temporal_encoding: (T, d_model) pre-computed temporal encoding
            valid_mask: (B, H, W) optional boolean mask
            
        Returns:
            out: (B, 1, H, W) AGB prediction
        """
        b, t, c, h_orig, w_orig = x.shape
        
        # Validate spatial dimensions
        assert h_orig % 8 == 0 and w_orig % 8 == 0, \
            f"Spatial dimensions must be divisible by 8, got {h_orig}x{w_orig}"
        
        h_8 = h_orig // 8
        w_8 = w_orig // 8
        
        # ============ ENCODER (Time-Distributed) ============
        # Flatten time dimension for spatial processing
        x_flat = x.view(b * t, c, h_orig, w_orig)
        
        # Layer 1: (B*T, C, H, W)
        e1 = self.enc1(x_flat)
        
        # Layer 2: (B*T, 2C, H/2, W/2)
        e2 = self.enc2(self.pool(e1))
        
        # Layer 3: (B*T, 4C, H/4, W/4)
        e3 = self.enc3(self.pool(e2))
        
        # Layer 4 (Bottleneck input): (B*T, 8C, H/8, W/8)
        e4 = self.enc4(self.pool(e3))
        
        # ============ BOTTLENECK (L-TAE) ============
        # Reshape back to temporal dimension for L-TAE
        e4_temporal = e4.view(b, t, self.base_c * 8, h_8, w_8)
        
        # ✅ FIX: Apply L-TAE with temporal_encoding and valid_mask
        neck = self.ltae(e4_temporal, temporal_encoding, valid_mask)
        
        # Normalize after L-TAE
        neck = self.ltae_bn(neck)
        
        # ============ DECODER with Skip Connections ============
        
        # Skip from enc4: temporal mean
        e4_skip = e4.view(b, t, self.base_c * 8, h_8, w_8).mean(dim=1)
        d4_input = torch.cat([self.up(neck), e4_skip], dim=1)
        d4 = self.dec4(d4_input)
        
        # Skip from enc3: temporal mean
        e3_skip = e3.view(b, t, self.base_c * 4, h_orig // 4, w_orig // 4).mean(dim=1)
        d3_input = torch.cat([self.up(d4), e3_skip], dim=1)
        d3 = self.dec3(d3_input)
        
        # Skip from enc2: temporal mean
        e2_skip = e2.view(b, t, self.base_c * 2, h_orig // 2, w_orig // 2).mean(dim=1)
        d2_input = torch.cat([self.up(d3), e2_skip], dim=1)
        d2 = self.dec2(d2_input)
        
        # Skip from enc1: temporal mean
        e1_skip = e1.view(b, t, self.base_c, h_orig, w_orig).mean(dim=1)
        d1_input = torch.cat([self.up(d2), e1_skip], dim=1)
        d1 = self.dec1(d1_input)
        
        # ============ OUTPUT HEAD ============
        out = self.head(d1)  # (B, 1, H, W)
        
        return out

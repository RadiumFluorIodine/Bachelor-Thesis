import torch
import torch.nn as nn
from .ltae import LTAE

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(True),
            nn.Conv2d(out_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(True)
        )
    def forward(self, x): return self.conv(x)

class UTAE(nn.Module):
    def __init__(self, in_c=10, out_c=1, base_c=32):
        super().__init__()
        
        # Encoder (Shared Spatial)
        self.enc1 = ConvBlock(in_c, base_c)
        self.enc2 = ConvBlock(base_c, base_c*2)
        self.enc3 = ConvBlock(base_c*2, base_c*4)
        self.enc4 = ConvBlock(base_c*4, base_c*8) # Bottleneck input
        self.pool = nn.MaxPool2d(2)
        
        # Temporal Encoder (L-TAE) di Bottleneck
        self.ltae = LTAE(base_c*8, n_head=16, d_k=8)
        
        # Decoder (Standard U-Net)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec3 = ConvBlock(base_c*8 + base_c*4, base_c*4)
        self.dec2 = ConvBlock(base_c*4 + base_c*2, base_c*2)
        self.dec1 = ConvBlock(base_c*2 + base_c, base_c)
        
        # Regresi Head
        self.head = nn.Conv2d(base_c, out_c, 1)

    def forward(self, x, days):
        b, t, c, h, w = x.shape
        
        # Encoder (Time Distributed)
        x = x.view(b*t, c, h, w)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3)) # (BT, 256, H/8, W/8)
        
        # L-TAE Bottleneck
        # Balikin ke (B, T, ...)
        e4_reshaped = e4.view(b, t, -1, h//8, w//8)
        neck = self.ltae(e4_reshaped, days) # (B, 256, H/8, W/8)
        
        # Decoder
        # Skip connection pake Temporal Mean
        d3 = self.dec3(torch.cat([self.up(neck), e3.view(b,t,-1,h//4,w//4).mean(1)], 1))
        d2 = self.dec2(torch.cat([self.up(d3), e2.view(b,t,-1,h//2,w//2).mean(1)], 1))
        d1 = self.dec1(torch.cat([self.up(d2), e1.view(b,t,-1,h,w).mean(1)], 1))
        
        return self.head(d1)

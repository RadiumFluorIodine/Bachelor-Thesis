import torch
import torch.nn as nn
# Kita import komponen L-TAE dan Encoder dari src utama agar tidak duplikat kode
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.models.ltae import LTAE
from src.models.utae import ConvBlock

class UTAE_Probabilistic(nn.Module):
    def __init__(self, in_c=10, base=32):
        super().__init__()
        # Encoder & L-TAE (Sama persis dengan standar)
        self.enc1 = ConvBlock(in_c, base)
        self.enc2 = ConvBlock(base, base*2)
        self.enc3 = ConvBlock(base*2, base*4)
        self.enc4 = ConvBlock(base*4, base*8)
        self.pool = nn.MaxPool2d(2)
        self.ltae = LTAE(base*8, n_head=16, d_k=8)
        
        # Decoder (Sama persis)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec3 = ConvBlock(base*8+base*4, base*4)
        self.dec2 = ConvBlock(base*4+base*2, base*2)
        self.dec1 = ConvBlock(base*2+base, base)
        
        # HEAD BERBEDA: Output 2 Channel (Mean & LogVar)
        self.head = nn.Conv2d(base, 2, kernel_size=1)

    def forward(self, x, days):
        b, t, c, h, w = x.shape
        x = x.view(b*t, c, h, w)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        neck = self.ltae(e4.view(b, t, -1, h//8, w//8), days)
        d3 = self.dec3(torch.cat([self.up(neck), e3.view(b,t,-1,h//4,w//4).mean(1)], 1))
        d2 = self.dec2(torch.cat([self.up(d3), e2.view(b,t,-1,h//2,w//2).mean(1)], 1))
        d1 = self.dec1(torch.cat([self.up(d2), e1.view(b,t,-1,h,w).mean(1)], 1))
        
        return self.head(d1) # (Batch, 2, H, W)

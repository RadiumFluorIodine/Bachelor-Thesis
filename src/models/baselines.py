import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(True),
            nn.Conv2d(out_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(True)
        )
    def forward(self, x): return self.conv(x)

# --- MODEL 1: ReUse (Standard U-Net for Regression) ---
# Referensi: "REgressive Unet for Carbon Storage..."
class ReUse(nn.Module):
    def __init__(self, in_c=10, out_c=1, base_c=32):
        super().__init__()
        # Encoder
        self.enc1 = ConvBlock(in_c, base_c)
        self.enc2 = ConvBlock(base_c, base_c*2)
        self.enc3 = ConvBlock(base_c*2, base_c*4)
        self.enc4 = ConvBlock(base_c*4, base_c*8)
        self.pool = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = ConvBlock(base_c*8, base_c*16)
        
        # Decoder
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec4 = ConvBlock(base_c*16 + base_c*8, base_c*8)
        self.dec3 = ConvBlock(base_c*8 + base_c*4, base_c*4)
        self.dec2 = ConvBlock(base_c*4 + base_c*2, base_c*2)
        self.dec1 = ConvBlock(base_c*2 + base_c, base_c)
        
        # Regression Head
        self.head = nn.Conv2d(base_c, out_c, 1)

    def forward(self, x):
        # Input x: (Batch, Channel, Height, Width) -> Single Image (bukan Time Series)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        b = self.bottleneck(self.pool(e4))
        
        d4 = self.dec4(torch.cat([self.up(b), e4], 1))
        d3 = self.dec3(torch.cat([self.up(d4), e3], 1))
        d2 = self.dec2(torch.cat([self.up(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up(d2), e1], 1))
        
        return self.head(d1)

# --- MODEL 2: AGBUNet (Enhanced CNN-UNET) ---
# Referensi: "AGBUNet: an enhanced CNN-UNET architecture..."
# AGBUNet biasanya lebih dalam atau memiliki filter lebih banyak di awal
class AGBUNet(nn.Module):
    def __init__(self, in_c=10, out_c=1):
        super().__init__()
        # Menggunakan struktur yang sedikit lebih dalam/lebar dibanding ReUse standar
        base_c = 64 # Lebih lebar filternya
        
        self.enc1 = ConvBlock(in_c, base_c)
        self.enc2 = ConvBlock(base_c, base_c*2)
        self.enc3 = ConvBlock(base_c*2, base_c*4)
        
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(base_c*4, base_c*8)
        
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.dec3 = ConvBlock(base_c*8 + base_c*4, base_c*4)
        self.dec2 = ConvBlock(base_c*4 + base_c*2, base_c*2)
        self.dec1 = ConvBlock(base_c*2 + base_c, base_c)
        
        # Output layer
        self.final = nn.Sequential(
            nn.Conv2d(base_c, 16, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(16, out_c, 1)
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        
        b = self.bottleneck(self.pool(e3))
        
        d3 = self.dec3(torch.cat([self.up(b), e3], 1))
        d2 = self.dec2(torch.cat([self.up(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up(d2), e1], 1))
        
        return self.final(d1)

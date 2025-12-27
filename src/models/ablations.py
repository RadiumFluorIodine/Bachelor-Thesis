import torch
import torch.nn as nn
from src.models.utae import UTAE
from src.models.ltae import LTAE

# --- VARIANT 1: TEMPORAL MEAN (Tanpa Attention) ---
class UTAE_NoAttn(UTAE):
    def __init__(self, in_c=10, out_c=1):
        super().__init__(in_c, out_c)
        # Hapus L-TAE
        self.ltae = None
        
    def forward(self, x, days=None):
        # x: (B, T, C, H, W)
        b, t, c, h, w = x.shape
        
        # Encoder (Time Distributed)
        x_reshaped = x.view(b*t, c, h, w)
        e1 = self.enc1(x_reshaped)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3)) # (BT, 256, H/8, W/8)
        
        # --- ABLASI: Ganti L-TAE dengan Mean Pooling ---
        # Kembalikan ke dimensi waktu
        e4_ts = e4.view(b, t, -1, h//8, w//8)
        neck = torch.mean(e4_ts, dim=1) # Rata-rata sederhana
        
        # Decoder (Sama seperti U-TAE)
        d3 = self.dec3(torch.cat([self.up(neck), e3.view(b,t,-1,h//4,w//4).mean(1)], 1))
        d2 = self.dec2(torch.cat([self.up(d3), e2.view(b,t,-1,h//2,w//2).mean(1)], 1))
        d1 = self.dec1(torch.cat([self.up(d2), e1.view(b,t,-1,h,w).mean(1)], 1))
        
        return self.head(d1)

# --- VARIANT 2: SINGLE HEAD (Tanpa Channel Grouping) ---
class UTAE_SingleHead(UTAE):
    def __init__(self, in_c=10, out_c=1):
        super().__init__(in_c, out_c)
        # Ganti L-TAE default dengan versi 1 Head
        # n_head=1 memaksa model memproses semua channel sekaligus (seperti Self-Attention standar)
        base_c = 32
        self.ltae = LTAE(in_channels=base_c*8, n_head=1, d_k=8)

# --- VARIANT 3: SINGLE DATE (Input 1 Gambar) ---
class U_Net_Single(nn.Module):
    # Ini pada dasarnya U-Net biasa 2D
    def __init__(self, in_c=10, out_c=1):
        super().__init__()
        # Gunakan arsitektur encoder-decoder yang sama persis agar adil
        # Tapi inputnya langsung (B, C, H, W)
        self.base_model = UTAE_NoAttn(in_c, out_c) 
        
    def forward(self, x, days=None):
        # x input: (B, 1, C, H, W) -> diambil frame 0 saja
        x_single = x[:, 0, :, :, :] # (B, C, H, W)
        # Tambahkan dimensi waktu dummy agar kompatibel dengan class UTAE_NoAttn
        x_expanded = x_single.unsqueeze(1) 
        return self.base_model(x_expanded)

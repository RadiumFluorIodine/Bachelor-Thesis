import torch
from torch.utils.data import DataLoader
import sys, os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from src.data.dataset import BiomassDataset

def test_data_integrity():
    print("\n🧪 [2/4] TESTING DATA INTEGRITY")
    print("="*40)
    
    data_dir = 'data/processed'
    if not os.path.exists(data_dir) or not os.listdir(data_dir):
        print("   ❌ Folder data kosong/tidak ada. Jalankan preprocess.py dulu.")
        return

    try:
        dataset = BiomassDataset(data_dir)
        print(f"   📂 Total Data: {len(dataset)} patch")
        
        # Load 1 Batch
        loader = DataLoader(dataset, batch_size=4, shuffle=True)
        img, lbl, days = next(iter(loader))
        
        # 1. Cek Dimensi
        print(f"   ✅ Batch Loaded: Img {img.shape}, Lbl {lbl.shape}")
        
        # 2. Cek Nilai Aneh (NaN/Inf)
        if torch.isnan(img).any() or torch.isinf(img).any():
            print("   ❌ ERROR: Terdapat NaN atau Inf pada Input Image!")
        else:
            print("   ✅ Input Image Clean (No NaN/Inf)")
            
        if torch.isnan(lbl).any() or torch.isinf(lbl).any():
            print("   ❌ ERROR: Terdapat NaN atau Inf pada Label!")
        else:
            print("   ✅ Label Clean (No NaN/Inf)")
            
        # 3. Cek Range Data (Normalisasi)
        # Sentinel-2 raw biasanya 0-10000, harusnya sudah jadi 0-1
        print(f"   📊 Stats Img: Min={img.min():.4f}, Max={img.max():.4f}, Mean={img.mean():.4f}")
        print(f"   📊 Stats Lbl: Min={lbl.min():.2f}, Max={lbl.max():.2f}")
        
        if img.max() > 100:
            print("   ⚠️ WARNING: Nilai pixel > 100. Apakah lupa normalisasi bagi 10000?")
        else:
            print("   ✅ Range Nilai tampak wajar (Normalized).")

    except Exception as e:
        print(f"   ❌ FAILED: {e}")

if __name__ == "__main__":
    test_data_integrity()

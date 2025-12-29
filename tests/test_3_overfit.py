import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import sys, os
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from src.data.dataset import BiomassDataset
from uncertainty_exp.models.utae_prob import UTAE_Probabilistic
from uncertainty_exp.models.loss import HeteroscedasticLoss

def test_overfit_single_batch():
    print("\n🧪 [3/4] TESTING TRAINING LOOP (OVERFIT CHECK)")
    print("="*40)
    print("   ℹ️ Tujuan: Memastikan model BISA belajar (Loss harus turun drastis).")
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. Siapkan Data Mini (4 Sampel saja)
    full_dataset = BiomassDataset('data/processed')
    if len(full_dataset) < 4:
        print("   ❌ Data kurang dari 4 sampel.")
        return
        
    mini_dataset = Subset(full_dataset, [0, 1, 2, 3])
    loader = DataLoader(mini_dataset, batch_size=2, shuffle=True)
    
    # 2. Siapkan Model & Loss
    model = UTAE_Probabilistic(in_c=10).to(DEVICE)
    criterion = HeteroscedasticLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3) # LR agak besar biar cepat turun
    
    print("   🚀 Mulai Training Singkat (20 Epochs)...")
    losses = []
    
    model.train()
    for epoch in range(20):
        batch_loss = 0
        for img, lbl, days in loader:
            img, lbl, days = img.to(DEVICE), lbl.to(DEVICE), days.to(DEVICE)
            
            optimizer.zero_grad()
            out = model(img, days)
            loss = criterion(out, lbl)
            loss.backward()
            optimizer.step()
            batch_loss += loss.item()
            
        avg_loss = batch_loss / len(loader)
        losses.append(avg_loss)
        if (epoch+1) % 5 == 0:
            print(f"      Ep {epoch+1}: Loss = {avg_loss:.4f}")
            
    # 3. Evaluasi
    initial_loss = losses[0]
    final_loss = losses[-1]
    
    print(f"   📉 Initial Loss: {initial_loss:.4f} -> Final Loss: {final_loss:.4f}")
    
    if final_loss < initial_loss * 0.5:
        print("   ✅ SUCCESS: Loss turun signifikan. Pipeline Training Valid!")
    else:
        print("   ⚠️ WARNING: Loss tidak turun banyak. Cek learning rate atau arsitektur.")

if __name__ == "__main__":
    test_overfit_single_batch()

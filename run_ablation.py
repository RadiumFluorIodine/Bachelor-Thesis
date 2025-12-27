import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from src.data.dataset import BiomassDataset
from src.models.utae import UTAE
from src.models.ablations import UTAE_NoAttn, UTAE_SingleHead, U_Net_Single
import pandas as pd
import numpy as np
import copy

# Config
BATCH_SIZE = 8
EPOCHS = 15 # Epoch lebih sedikit cukup untuk melihat tren konvergensi
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def train_one_model(model_name, model, train_loader, val_loader):
    print(f"\n🧪 Memulai Studi Ablasi: {model_name}")
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    
    best_loss = float('inf')
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for img, lbl, days in train_loader:
            img, lbl, days = img.to(DEVICE), lbl.to(DEVICE), days.to(DEVICE)
            
            # Khusus Single Date: Ambil bulan ke-6 (tengah tahun/Juni)
            if model_name == "Single Date (June)":
                img = img[:, 5:6, :, :, :] # Slice time index 5
                
            optimizer.zero_grad()
            out = model(img, days)
            loss = criterion(out, lbl)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for img, lbl, days in val_loader:
                img, lbl, days = img.to(DEVICE), lbl.to(DEVICE), days.to(DEVICE)
                if model_name == "Single Date (June)":
                    img = img[:, 5:6, :, :, :]
                
                out = model(img, days)
                val_loss += criterion(out, lbl).item()
        
        avg_val = val_loss / len(val_loader)
        if avg_val < best_loss:
            best_loss = avg_val
            
        print(f"   Epoch {epoch+1}/{EPOCHS} - Val RMSE: {np.sqrt(avg_val):.4f}")
        
    return np.sqrt(best_loss) # Return RMSE terbaik

def main():
    # Load Data
    full_ds = BiomassDataset('data/processed')
    train_size = int(0.8 * len(full_ds))
    val_size = len(full_ds) - train_size
    train_set, val_set = random_split(full_ds, [train_size, val_size])
    
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)
    
    # Definisi Daftar Model untuk Ablasi
    ablation_experiments = {
        "U-TAE (Full Spatio-Temporal)": UTAE(in_c=10, out_c=1),
        "Ablasi 1: Temporal Mean (No Attn)": UTAE_NoAttn(in_c=10, out_c=1),
        "Ablasi 2: Single Head (No Grouping)": UTAE_SingleHead(in_c=10, out_c=1),
        "Ablasi 3: Single Date (June)": U_Net_Single(in_c=10, out_c=1)
    }
    
    results = []
    
    for name, model in ablation_experiments.items():
        rmse_score = train_one_model(name, model, train_loader, val_loader)
        
        # Hitung Jumlah Parameter
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        results.append({
            "Model Variant": name,
            "Parameters": n_params,
            "Best RMSE (Mg/ha)": round(rmse_score, 4)
        })
    
    # Output Hasil
    df = pd.DataFrame(results)
    df = df.sort_values(by="Best RMSE (Mg/ha)")
    
    print("\n" + "="*50)
    print("📊 HASIL STUDI ABLASI")
    print("="*50)
    print(df.to_string(index=False))
    print("="*50)
    
    df.to_csv('ablation_study_results.csv', index=False)
    print("Hasil disimpan ke 'ablation_study_results.csv'")

if __name__ == "__main__":
    main()

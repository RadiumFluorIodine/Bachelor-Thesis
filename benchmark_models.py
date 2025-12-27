import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.data.dataset import BiomassDataset
from src.models.utae import UTAE
from src.models.baselines import ReUse, AGBUNet
import numpy as np
import pandas as pd
from tqdm import tqdm

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 8

def train_and_eval(model_name, model, train_loader, val_loader, epochs=5):
    """Fungsi helper untuk training cepat dan evaluasi"""
    print(f"\n🏃 Training {model_name}...")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    model.to(DEVICE)
    
    # Training Loop Singkat
    for ep in range(epochs):
        model.train()
        for img, lbl, days in train_loader:
            img, lbl, days = img.to(DEVICE), lbl.to(DEVICE), days.to(DEVICE)
            
            # Handling Input:
            # U-TAE butuh (Batch, Time, Channel, H, W)
            # ReUse/AGBUNet butuh (Batch, Channel, H, W) -> Kita ambil Mean Temporal
            if model_name != "U-TAE":
                img = torch.mean(img, dim=1) # Temporal Mean
            
            optimizer.zero_grad()
            if model_name == "U-TAE":
                out = model(img, days)
            else:
                out = model(img) # Baseline tidak butuh 'days'
                
            loss = criterion(out, lbl)
            loss.backward()
            optimizer.step()
            
    # Evaluation Loop
    model.eval()
    errors = []
    with torch.no_grad():
        for img, lbl, days in val_loader:
            img, lbl, days = img.to(DEVICE), lbl.to(DEVICE), days.to(DEVICE)
            
            if model_name != "U-TAE":
                img = torch.mean(img, dim=1)
                
            if model_name == "U-TAE":
                pred = model(img, days)
            else:
                pred = model(img)
                
            # Hitung RMSE per batch
            mse = torch.mean((pred - lbl)**2).item()
            errors.append(np.sqrt(mse))
            
    return np.mean(errors)

def run_benchmark():
    # Load Data
    dataset = BiomassDataset('data/processed')
    # Split sederhana
    split = int(0.8 * len(dataset))
    train_set, val_set = torch.utils.data.random_split(dataset, [split, len(dataset)-split])
    
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)
    
    # Inisialisasi Model
    models = {
        "ReUse (Baseline)": ReUse(in_c=10, out_c=1),
        "AGBUNet (Enhanced)": AGBUNet(in_c=10, out_c=1),
        "U-TAE (Spatio-Temporal)": UTAE(in_c=10, out_c=1)
    }
    
    results = {}
    
    for name, model in models.items():
        # Reset bobot/training dari awal
        rmse = train_and_eval(name, model, train_loader, val_loader, epochs=10)
        results[name] = rmse
        print(f"✅ {name} RMSE: {rmse:.4f} Mg/ha")
        
    # Tampilkan Tabel Perbandingan
    df = pd.DataFrame(list(results.items()), columns=['Model', 'RMSE (Mg/ha)'])
    print("\n📊 HASIL AKHIR BENCHMARK")
    print(df)
    df.to_csv('benchmark_results.csv', index=False)

if __name__ == "__main__":
    run_benchmark()

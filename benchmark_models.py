import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.data.dataset import BiomassDataset
from src.models.utae import UTAE
from src.models.baselines import ReUse, AGBUNet, PixelWiseML
import numpy as np
import pandas as pd
import time

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 8

def train_dl_model(name, model, train_loader, val_loader, epochs=5):
    """Training khusus untuk Deep Learning (PyTorch)"""
    print(f"\n🧠 Training Deep Learning Model: {name}...")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    model.to(DEVICE)
    
    start_time = time.time()
    
    for ep in range(epochs):
        model.train()
        for img, lbl, days in train_loader:
            img, lbl, days = img.to(DEVICE), lbl.to(DEVICE), days.to(DEVICE)
            
            # Baseline DL (ReUse/AGBUNet) butuh input 4D (Mean Temporal)
            if name != "U-TAE":
                img = torch.mean(img, dim=1) 
            
            optimizer.zero_grad()
            
            if name == "U-TAE":
                out = model(img, days)
            else:
                out = model(img)
                
            loss = criterion(out, lbl)
            loss.backward()
            optimizer.step()
            
    # Evaluasi
    model.eval()
    errors = []
    with torch.no_grad():
        for img, lbl, days in val_loader:
            img, lbl, days = img.to(DEVICE), lbl.to(DEVICE), days.to(DEVICE)
            
            if name != "U-TAE": img = torch.mean(img, dim=1)
            
            if name == "U-TAE": pred = model(img, days)
            else: pred = model(img)
            
            # Ambil region valid saja (Label > 0)
            mask = lbl > 0
            if mask.sum() > 0:
                mse = torch.mean((pred[mask] - lbl[mask])**2).item()
                errors.append(np.sqrt(mse))
    
    duration = time.time() - start_time
    return np.mean(errors), duration

def train_ml_model(name, model_wrapper, train_loader, val_loader):
    """Training khusus untuk Machine Learning (Scikit-Learn/XGB)"""
    print(f"\n🌳 Training ML Model: {name}...")
    start_time = time.time()
    
    # 1. Fit (Training)
    model_wrapper.fit(train_loader)
    
    # 2. Predict & Evaluate
    rmse = model_wrapper.predict(val_loader)
    
    duration = time.time() - start_time
    return rmse, duration

def run_benchmark():
    # Load Data
    print("📂 Loading Dataset...")
    dataset = BiomassDataset('data/processed')
    split = int(0.8 * len(dataset))
    train_set, val_set = torch.utils.data.random_split(dataset, [split, len(dataset)-split])
    
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)
    
    results = []

    # --- 1. DEEP LEARNING MODELS ---
    dl_models = {
        "U-TAE (Proposed)": UTAE(in_c=10, out_c=1),
        "AGBUNet": AGBUNet(in_c=10, out_c=1),
        "ReUse (U-Net)": ReUse(in_c=10, out_c=1),
    }
    
    for name, model in dl_models.items():
        rmse, dur = train_dl_model(name, model, train_loader, val_loader, epochs=10)
        results.append({"Model": name, "Type": "Deep Learning", "RMSE (Mg/ha)": rmse, "Time (s)": dur})
        print(f"   👉 {name}: RMSE {rmse:.2f}")

    # --- 2. MACHINE LEARNING BASELINES ---
    # Pastikan library terinstall
    ml_configs = [
        ("Random Forest", "rf"),
        ("Linear Regression", "lr"),
        ("XGBoost", "xgb")
    ]
    
    for name, code in ml_configs:
        try:
            wrapper = PixelWiseML(model_type=code)
            rmse, dur = train_ml_model(name, wrapper, train_loader, val_loader)
            results.append({"Model": name, "Type": "Machine Learning", "RMSE (Mg/ha)": rmse, "Time (s)": dur})
            print(f"   👉 {name}: RMSE {rmse:.2f}")
        except Exception as e:
            print(f"   ❌ Skip {name}: {e}")

    # Output Final
    df = pd.DataFrame(results).sort_values(by="RMSE (Mg/ha)")
    print("\n" + "="*60)
    print("📊 FINAL BENCHMARK RESULT")
    print("="*60)
    print(df.to_string(index=False))
    print("="*60)
    df.to_csv('benchmark_results_full.csv', index=False)

if __name__ == "__main__":
    run_benchmark()

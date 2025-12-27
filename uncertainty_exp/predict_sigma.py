import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import random

# Tambahkan path root agar bisa import modul dari src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.data.dataset import BiomassDataset
from uncertainty_exp.models.utae_prob import UTAE_Probabilistic

# --- KONFIGURASI ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = 'uncertainty_exp/outputs/models/best_utae_prob.pth'
DATA_DIR = 'data/processed'
OUTPUT_FIG = 'uncertainty_exp/outputs/figures'

os.makedirs(OUTPUT_FIG, exist_ok=True)

def visualize_prediction():
    print(f"🔍 Memuat model Probabilistik dari {MODEL_PATH}...")
    
    # 1. Load Model
    # Ingat: Model probabilistik memiliki struktur head yang berbeda (2 channel)
    model = UTAE_Probabilistic(in_c=10).to(DEVICE)
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model tidak ditemukan di {MODEL_PATH}. Latih dulu menggunakan train_uncertainty.py")
        return

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # 2. Ambil 1 Sampel Acak dari Dataset
    # Kita gunakan dataset loader agar formatnya otomatis benar
    dataset = BiomassDataset(DATA_DIR)
    idx = random.randint(0, len(dataset)-1)
    
    img, lbl, days = dataset[idx]
    
    # Tambahkan dimensi Batch: (C, H, W) -> (1, C, H, W)
    # Dan karena dataset kita Time Series, bentuk aslinya (Time, C, H, W) -> perlu (1, T, C, H, W)
    # Namun loader dataset.py di skrip sebelumnya sudah meratakan time/channel atau belum?
    # Mari kita cek dataset.py standar: outputnya (C, H, W) jika sudah di-mean atau (T, C, H, W).
    # Asumsi: dataset.py mengembalikan (Time, Band, H, W) sesuai U-TAE.
    
    # Tambahkan batch dimension
    img_tensor = img.unsqueeze(0).to(DEVICE)   # (1, 12, 10, 128, 128)
    lbl_tensor = lbl.unsqueeze(0).to(DEVICE)   # (1, 1, 128, 128)
    days_tensor = days.unsqueeze(0).to(DEVICE) # (1, 12)

    print(f"📸 Memproses Patch Index: {idx}")

    # 3. Inferensi
    with torch.no_grad():
        # Output shape: (1, 2, 128, 128) -> [Mean, LogVar]
        output = model(img_tensor, days_tensor)
        
        # Pisahkan Channel
        pred_mu = output[0, 0, :, :].cpu().numpy()      # Mean (Prediksi AGB)
        pred_logvar = output[0, 1, :, :].cpu().numpy()  # Log Variance (s)
        
        # Hitung Sigma (Standar Deviasi)
        # Rumus: sigma = sqrt(exp(s))
        pred_sigma = np.sqrt(np.exp(pred_logvar))
        
        # Ground Truth
        true_agb = lbl_tensor[0, 0, :, :].cpu().numpy()

    # 4. Visualisasi
    print("🎨 Membuat Peta Visualisasi...")
    
    fig, ax = plt.subplots(1, 4, figsize=(24, 6))
    
    # --- Panel 1: Citra Satelit (RGB Bulan Juni) ---
    # Input tensor: (1, 12, 10, H, W). Time index 5 = Juni. RGB = Band [2, 1, 0]
    rgb = img_tensor[0, 5, [2, 1, 0], :, :].cpu().numpy().transpose(1, 2, 0)
    # Brightness adjustment
    rgb = np.clip(rgb * 3.5, 0, 1)
    
    ax[0].imshow(rgb)
    ax[0].set_title("Sentinel-2 RGB (Juni)")
    ax[0].axis('off')

    # --- Panel 2: Ground Truth (ESA CCI) ---
    im2 = ax[1].imshow(true_agb, cmap='viridis', vmin=0, vmax=350)
    ax[1].set_title("Ground Truth (ESA CCI)")
    ax[1].axis('off')
    plt.colorbar(im2, ax=ax[1], fraction=0.046, pad=0.04, label='Mg/ha')

    # --- Panel 3: Prediksi Model (Mean) ---
    im3 = ax[2].imshow(pred_mu, cmap='viridis', vmin=0, vmax=350)
    ax[2].set_title("Prediksi U-TAE (Mean)")
    ax[2].axis('off')
    plt.colorbar(im3, ax=ax[2], fraction=0.046, pad=0.04, label='Mg/ha')

    # --- Panel 4: Ketidakpastian (Sigma) ---
    # Gunakan colormap 'magma' atau 'inferno' agar area error terlihat 'panas'
    im4 = ax[3].imshow(pred_sigma, cmap='magma', vmin=0, vmax=50)
    ax[3].set_title("Ketidakpastian (Sigma)")
    ax[3].axis('off')
    plt.colorbar(im4, ax=ax[3], fraction=0.046, pad=0.04, label='Mg/ha (Uncertainty)')

    plt.suptitle(f"Visualisasi Probabilistik U-TAE (Patch {idx})", fontsize=16)
    plt.tight_layout()
    
    save_path = os.path.join(OUTPUT_FIG, f'prediction_sigma_patch_{idx}.png')
    plt.savefig(save_path, dpi=150)
    print(f"✅ Gambar disimpan di: {save_path}")
    plt.show()

    # --- Analisis Histogram Ketidakpastian ---
    # Ini berguna untuk Bab Pembahasan: "Seberapa yakin model kita?"
    plt.figure(figsize=(8, 4))
    plt.hist(pred_sigma.flatten(), bins=50, color='orange', alpha=0.7, label='Sigma')
    plt.title("Distribusi Tingkat Ketidakpastian Model")
    plt.xlabel("Sigma (Mg/ha)")
    plt.ylabel("Jumlah Piksel")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_FIG, f'hist_sigma_patch_{idx}.png'))

if __name__ == "__main__":
    visualize_prediction()

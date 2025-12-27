import torch
import numpy as np
import rasterio
from rasterio.windows import Window
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import sys
import os
import glob

# Tambahkan path root agar bisa import modul
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

# Import Model Probabilistik
from uncertainty_exp.models.utae_prob import UTAE_Probabilistic

# --- KONFIGURASI ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = 'uncertainty_exp/outputs/models/best_utae_prob.pth'

# Pastikan path ini sesuai dengan hasil download_gee.py
ESA_SD_PATH = 'data/raw/AGB_SD_Reference.tif'
ESA_AGB_PATH = 'data/raw/AGB_Label.tif'
S2_DIR = 'data/raw' 

# Ukuran area validasi (Crop tengah agar cepat)
VALIDATION_SIZE = 512 

def get_center_window(src, size):
    """Membuat window crop di tengah citra"""
    h, w = src.height, src.width
    col_off = (w - size) // 2
    row_off = (h - size) // 2
    return Window(col_off, row_off, size, size)

def validate():
    print("🔍 Memulai Validasi Ketidakpastian vs ESA CCI SD...")

    # 1. Cek Ketersediaan File
    if not os.path.exists(ESA_SD_PATH):
        print(f"❌ File {ESA_SD_PATH} tidak ditemukan. Jalankan download_gee.py dulu.")
        return

    # 2. Load Data Referensi (ESA SD & AGB)
    print("📂 Membaca Data Referensi (Crop Tengah)...")
    with rasterio.open(ESA_SD_PATH) as src_sd, rasterio.open(ESA_AGB_PATH) as src_agb:
        window = get_center_window(src_sd, VALIDATION_SIZE)
        
        # Baca data
        esa_sd_map = src_sd.read(1, window=window)
        esa_agb_map = src_agb.read(1, window=window)
        
        # Simpan metadata profile untuk keperluan resize/match jika perlu
        profile = src_sd.profile

    # 3. Load Data Sentinel-2 (Input Model) pada Window yang SAMA
    print("🛰️  Membaca Time Series Sentinel-2...")
    s2_files = sorted(glob.glob(os.path.join(S2_DIR, "S2_M*.tif")))
    
    if len(s2_files) != 12:
        print("❌ Data Sentinel-2 tidak lengkap (harus 12 bulan).")
        return

    s2_stack = []
    for f in s2_files:
        with rasterio.open(f) as src:
            # Pastikan membaca window yang sama persis
            # Asumsi: Sentinel-2 dan ESA CCI sudah di-download dengan dimensi/proyeksi sama (dari download_gee.py)
            arr = src.read(window=window) 
            s2_stack.append(arr) # (10, H, W)

    # Stack menjadi (Time, Band, H, W) -> (12, 10, 512, 512)
    s2_np = np.stack(s2_stack, axis=0)

    # 4. Preprocessing Input Model
    # Convert ke Tensor, Normalize, Add Batch Dim
    input_tensor = torch.from_numpy(s2_np).float() / 10000.0 # (12, 10, H, W)
    input_tensor = torch.clamp(input_tensor, 0, 1)
    
    # Tambah dimensi Batch -> (1, 12, 10, H, W)
    input_tensor = input_tensor.unsqueeze(0).to(DEVICE)
    
    # Buat tensor Hari (Days)
    days = torch.tensor([15, 45, 75, 105, 135, 165, 195, 225, 255, 285, 315, 345]).float()
    days = days.unsqueeze(0).to(DEVICE)

    # 5. Load Model & Inferensi
    print("🤖 Menjalankan Model U-TAE Probabilistik...")
    model = UTAE_Probabilistic(in_c=10).to(DEVICE)
    
    if not os.path.exists(MODEL_PATH):
        print("❌ Model belum dilatih. Jalankan train_uncertainty.py dulu.")
        return

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    
    with torch.no_grad():
        output = model(input_tensor, days)
        
        # Output Channel 1: Log Variance
        log_var = output[0, 1, :, :].cpu().numpy()
        
        # Konversi ke Sigma (Standar Deviasi)
        model_sigma_map = np.sqrt(np.exp(log_var))

    # 6. Analisis Statistik
    print("📊 Menghitung Korelasi...")
    
    # Filter: Hanya bandingkan pixel yang valid (ada hutan dan ada data SD)
    # Kita buang nilai 0 (NoData) dan Nan
    mask = (esa_agb_map > 0) & (esa_sd_map > 0) & np.isfinite(model_sigma_map)
    
    if mask.sum() == 0:
        print("⚠️ Tidak ada pixel valid yang beririsan (Mungkin area laut/kota?). Coba geser window.")
        return

    flat_esa = esa_sd_map[mask].flatten()
    flat_model = model_sigma_map[mask].flatten()
    
    # Hitung Pearson Correlation
    corr, _ = pearsonr(flat_esa, flat_model)
    
    print("\n" + "="*40)
    print("📈 HASIL VALIDASI KETIDAKPASTIAN")
    print("="*40)
    print(f"Jumlah Pixel Valid : {mask.sum()}")
    print(f"Korelasi Pearson (r): {corr:.4f}")
    print(f"Rata-rata ESA SD    : {np.mean(flat_esa):.2f} Mg/ha")
    print(f"Rata-rata U-TAE SD  : {np.mean(flat_model):.2f} Mg/ha")
    print("-" * 40)
    
    if corr > 0.3:
        print("✅ KESIMPULAN: Model berhasil menangkap pola ketidakpastian ESA.")
    else:
        print("⚠️ KESIMPULAN: Pola ketidakpastian berbeda. (Wajar jika model hanya menangkap aleatoric uncertainty).")
    print("="*40)

    # 7. Visualisasi
    os.makedirs('uncertainty_exp/outputs/figures', exist_ok=True)
    
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    
    # Peta ESA SD
    im1 = ax[0].imshow(esa_sd_map, cmap='magma', vmin=0, vmax=50)
    ax[0].set_title("ESA CCI Uncertainty (Ref)")
    plt.colorbar(im1, ax=ax[0], label='Mg/ha')
    
    # Peta Model Sigma
    im2 = ax[1].imshow(model_sigma_map, cmap='magma', vmin=0, vmax=50)
    ax[1].set_title("U-TAE Uncertainty (Pred)")
    plt.colorbar(im2, ax=ax[1], label='Mg/ha')
    
    # Scatter Plot (Sample 1000 titik agar ringan)
    if len(flat_esa) > 2000:
        idx = np.random.choice(len(flat_esa), 2000, replace=False)
        sample_esa = flat_esa[idx]
        sample_model = flat_model[idx]
    else:
        sample_esa = flat_esa
        sample_model = flat_model
        
    ax[2].scatter(sample_esa, sample_model, alpha=0.3, s=5, c='purple')
    ax[2].set_xlabel("ESA CCI SD")
    ax[2].set_ylabel("Model Sigma")
    ax[2].set_title(f"Korelasi: r={corr:.2f}")
    ax[2].grid(True, alpha=0.3)
    
    plt.suptitle("Validasi Ketidakpastian: ESA CCI vs U-TAE")
    plt.tight_layout()
    
    out_file = 'uncertainty_exp/outputs/figures/validation_esa_comparison.png'
    plt.savefig(out_file, dpi=150)
    print(f"🖼️  Grafik disimpan di: {out_file}")
    plt.show()

if __name__ == "__main__":
    validate()

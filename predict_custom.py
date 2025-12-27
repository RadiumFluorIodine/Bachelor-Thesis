import ee
import geemap
import torch
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import os
from src.models.utae import UTAE

# --- KONFIGURASI LOKASI (CUSTOM) ---
# Contoh: Area perkebunan di Lampung Timur (Ganti koordinat sesuai keinginan)
# Format: [Longitude, Latitude]
ROI_COORDS = [
    [105.65, -5.25], # Kiri Bawah
    [105.70, -5.25], # Kanan Bawah
    [105.70, -5.20], # Kanan Atas
    [105.65, -5.20], # Kiri Atas
    [105.65, -5.25]  # Tutup Polygon
]

MODEL_PATH = 'best_model.pth'
TEMP_DIR = 'data/temp_prediction'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_custom_data(roi):
    """Download Sentinel-2 stack untuk area kustom"""
    print("⬇️  Mengunduh data Sentinel-2 area kustom via GEE...")
    try:
        ee.Initialize(project='data-skripsi-473712')
    except:
        ee.Authenticate()
        ee.Initialize(project='data-skripsi-473712')
        
    os.makedirs(TEMP_DIR, exist_ok=True)
    geometry = ee.Geometry.Polygon(roi)
    
    # Fungsi Masking (Sama seperti training)
    def mask_s2(image):
        qa = image.select('QA60')
        mask = qa.bitwiseAnd(1<<10).eq(0).And(qa.bitwiseAnd(1<<11).eq(0))
        return image.updateMask(mask).select(['B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12'])

    # Ambil Time Series 2021
    col = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterBounds(geometry) \
        .filterDate('2021-01-01', '2021-12-31') \
        .map(mask_s2)

    # Download per bulan (harus konsisten dengan training: 12 bulan)
    file_paths = []
    for m in range(1, 13):
        img = col.filter(ee.Filter.calendarRange(m, m, 'month')).median().unmask(0).clip(geometry)
        fname = os.path.join(TEMP_DIR, f"pred_M{m}.tif")
        
        # Download jika belum ada
        if not os.path.exists(fname):
            geemap.download_ee_image(img, fname, region=geometry, scale=10, crs='EPSG:4326', dtype='uint16')
        file_paths.append(fname)
        
    return file_paths

def predict_map():
    # 1. Siapkan Data
    file_paths = get_custom_data(ROI_COORDS)
    
    # Baca data menjadi array (Time, Band, H, W)
    stack = []
    src_ref = rasterio.open(file_paths[0]) # Referensi geospasial
    
    print("🔄 Memproses data input...")
    for f in file_paths:
        with rasterio.open(f) as src:
            stack.append(src.read()) # (10, H, W)
    
    # Stack -> (12, 10, H, W)
    full_input = np.stack(stack, axis=0)
    
    # Preprocessing (Sama seperti dataset.py)
    # Normalize 0-10000 -> 0-1
    input_tensor = torch.from_numpy(full_input).float() / 10000.0
    input_tensor = torch.clamp(input_tensor, 0, 1)
    
    # Tambahkan dimensi Batch -> (1, 12, 10, H, W)
    input_tensor = input_tensor.unsqueeze(0).to(DEVICE)
    
    # Buat dummy days tensor
    days = torch.tensor([15, 45, 75, 105, 135, 165, 195, 225, 255, 285, 315, 345]).float()
    days = days.unsqueeze(0).to(DEVICE) # (1, 12)

    # 2. Load Model & Prediksi
    print("🤖 Menjalankan U-TAE...")
    model = UTAE(in_c=10, out_c=1).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    
    # Perhatian: Jika area sangat besar, harus di-tiling (potong-potong). 
    # Kode ini asumsi area muat di GPU memory.
    with torch.no_grad():
        # Padding agar ukuran bisa dibagi 8 (syarat U-Net downsampling)
        _, t, c, h, w = input_tensor.shape
        pad_h = (32 - h % 32) % 32
        pad_w = (32 - w % 32) % 32
        
        if pad_h > 0 or pad_w > 0:
            input_tensor = torch.nn.functional.pad(input_tensor, (0, pad_w, 0, pad_h))
        
        # Predict
        output = model(input_tensor, days) # (1, 1, H_pad, W_pad)
        
        # Crop padding
        pred_map = output[0, 0, :h, :w].cpu().numpy()

    # 3. Visualisasi Peta
    print("🗺️  Menampilkan Peta...")
    fig, ax = plt.subplots(1, 2, figsize=(15, 7))
    
    # Tampilkan Citra Asli (RGB Bulan Juni - Pertengahan Tahun)
    # Input tensor: (1, Time, Band, H, W). Time index 5 = Juni. RGB bands = [2,1,0]
    rgb = input_tensor[0, 5, [2, 1, 0], :h, :w].cpu().numpy().transpose(1, 2, 0)
    # Brightness adjustment
    rgb = np.clip(rgb * 3.5, 0, 1) 
    
    ax[0].imshow(rgb)
    ax[0].set_title("Citra Sentinel-2 (RGB True Color)")
    ax[0].axis('off')
    
    # Tampilkan Peta Biomassa
    im = ax[1].imshow(pred_map, cmap='viridis', vmin=0, vmax=300) # Asumsi max AGB 300
    ax[1].set_title("Estimasi Aboveground Biomass (U-TAE)")
    ax[1].axis('off')
    
    cbar = plt.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)
    cbar.set_label('Biomass (Mg/ha)')
    
    plt.suptitle("Hasil Prediksi Custom Region", fontsize=16)
    plt.tight_layout()
    plt.savefig('custom_prediction_map.png', dpi=150)
    plt.show()

if __name__ == "__main__":
    predict_map()

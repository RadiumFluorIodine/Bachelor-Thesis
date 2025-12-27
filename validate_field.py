import ee
import geemap
import torch
import numpy as np
import pandas as pd
import rasterio
import os
from src.models.utae import UTAE

# --- KONFIGURASI VALIDASI ---
# Koordinat Sampel Mangrove Petengoran (Estimasi dari Paper #1571044039)
# Format: [Lon, Lat, Ground_Truth_AGB_Paper]
# Nilai GT diambil dari range yang disebut di paper (zona depan/tengah/belakang)
VALIDATION_POINTS = [
    {"name": "Petengoran_Zona1", "coords": [105.241, -5.578], "paper_agb": 258.60}, # Zona Padat
    {"name": "Petengoran_Zona2", "coords": [105.243, -5.579], "paper_agb": 140.50}, # Zona Sedang
    {"name": "Petengoran_Zona3", "coords": [105.245, -5.580], "paper_agb": 27.52}   # Zona Jarang
]

MODEL_PATH = 'best_model.pth'
TEMP_DIR = 'data/temp_validation'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_point_data(lon, lat):
    """Download Time Series Sentinel-2 untuk 1 titik koordinat (1x1 pixel)"""
    try:
        ee.Initialize(project='data-skripsi-473712')
    except:
        ee.Authenticate()
        ee.Initialize(project='data-skripsi-473712')

    point = ee.Geometry.Point([lon, lat])
    # Buffer kecil (10m) untuk memastikan kita dapat pixelnya
    roi = point.buffer(10).bounds() 

    # Sentinel-2 Collection
    def mask_s2(img):
        qa = img.select('QA60')
        mask = qa.bitwiseAnd(1<<10).eq(0).And(qa.bitwiseAnd(1<<11).eq(0))
        return img.updateMask(mask).select(['B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12'])

    col = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterBounds(roi) \
        .filterDate('2021-01-01', '2021-12-31') \
        .map(mask_s2)

    # Ambil 12 bulan
    pixel_values = []
    for m in range(1, 13):
        img = col.filter(ee.Filter.calendarRange(m, m, 'month')).median().unmask(0)
        # Reduce region untuk ambil nilai pixel
        val = img.reduceRegion(reducer=ee.Reducer.mean(), geometry=roi, scale=10).getInfo()
        
        # Urutkan band sesuai training
        bands = ['B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12']
        monthly_vals = [val.get(b, 0) for b in bands]
        pixel_values.append(monthly_vals)
        
    return np.array(pixel_values) # Shape (12, 10)

def validate():
    print("🔍 Memulai Validasi dengan Data Mangrove Petengoran...")
    
    # Load Model
    model = UTAE(in_c=10, out_c=1).to(DEVICE)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    else:
        print("⚠️ Warning: Model belum dilatih. Menggunakan bobot acak untuk demo.")
    model.eval()
    
    results = []
    
    for pt in VALIDATION_POINTS:
        print(f"   Processing titik: {pt['name']}...")
        
        # 1. Ambil Data GEE Real-time
        # Input: (12, 10)
        raw_data = get_point_data(pt['coords'][0], pt['coords'][1])
        
        # 2. Preprocessing
        # Normalkan (0-10000 -> 0-1) dan bentuk tensor
        # Input Model butuh: (Batch, Time, Channel, H, W) -> H,W = 1,1
        input_tensor = torch.tensor(raw_data).float() / 10000.0
        input_tensor = input_tensor.permute(1, 0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1) # (1, 10, 12) -> (1, 12, 10, 1, 1)
        input_tensor = input_tensor.to(DEVICE)
        
        days = torch.linspace(15, 345, 12).unsqueeze(0).to(DEVICE)

        # 3. Prediksi
        with torch.no_grad():
            pred_agb = model(input_tensor, days).item()
            
        # 4. Catat Hasil
        results.append({
            "Lokasi": pt['name'],
            "Koordinat": f"{pt['coords']}",
            "Data Paper (Mg/ha)": pt['paper_agb'],
            "Prediksi U-TAE (Mg/ha)": round(pred_agb, 2),
            "Selisih": round(abs(pred_agb - pt['paper_agb']), 2)
        })

    # Tampilkan Tabel
    df = pd.DataFrame(results)
    print("\n📋 HASIL VALIDASI LAPANGAN (MANGROVE PETENGORAN)")
    print(df)
    
    # Simpan
    df.to_csv('validasi_petengoran.csv', index=False)
    print("\nCatatan: 'Data Paper' adalah nilai referensi dari Paper #1571044039.")

if __name__ == "__main__":
    validate()

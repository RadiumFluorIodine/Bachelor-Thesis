# FILE: download_data_vscode_local.py
import ee
import geemap
import os
import sys

# --- KONFIGURASI ---
PROJECT_ID = 'data-skripsi-473712'
ASSET_ID = 'projects/data-skripsi-473712/assets/Lampung'
OUTPUT_DIR = 'data/raw'
REGION_NAME = 'Lampung'  # Change to 'KalimantanSelatan' for cross-region test

YEAR_S2 = 2021
YEAR_AGB = 2021 

def main():
    # 1. Autentikasi
    try:
        ee.Initialize(project=PROJECT_ID)
        print("✓ Terhubung ke Earth Engine (sudah authenticated).")
    except:
        print("Memerlukan autentikasi GEE...")
        ee.Authenticate()
        ee.Initialize(project=PROJECT_ID)
        print("✓ Autentikasi selesai.")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"✓ Output directory ready: {OUTPUT_DIR}")

    # 2. ROI 
    roi = ee.FeatureCollection(ASSET_ID).geometry()
    print(f"✓ ROI loaded: {REGION_NAME}")

    # 3. Sentinel-2 
    def mask_s2(image):
        """QA60 cloud and cirrus masking"""
        qa = image.select('QA60')
        # Bit 10 = opaque clouds, Bit 11 = cirrus clouds
        mask = qa.bitwiseAnd(1<<10).eq(0).And(qa.bitwiseAnd(1<<11).eq(0))
        return image.updateMask(mask).select([
            'B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12'
        ])

    s2_col = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterBounds(roi) \
        .filterDate(f'{YEAR_S2}-01-01', f'{YEAR_S2}-12-31') \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 40)) \
        .map(mask_s2)

    ref_proj = s2_col.first().select('B2').projection()
    print(f"✓ Reference projection extracted from Sentinel-2")

    print(f"Mulai download Sentinel-2 ({YEAR_S2}) - Region: {REGION_NAME}...")
    for m in range(1, 13):
        start = ee.Date.fromYMD(YEAR_S2, m, 1)
        end = start.advance(1, 'month')
        img = s2_col.filterDate(start, end).median().unmask(0).clip(roi)
        
        # IMPROVED: Include region name in filename
        fname = os.path.join(OUTPUT_DIR, f"S2_{REGION_NAME}_M{m:02d}.tif")
        
        # Cek jika file sudah ada agar tidak download ulang
        if not os.path.exists(fname):
            print(f"   Processing Bulan {m}...")
            geemap.download_ee_image(
                img, fname, region=roi, crs=ref_proj, scale=10, 
                dtype='uint16', overwrite=True
            )
        else:
            print(f"   ✓ Bulan {m} sudah ada (skipped).")

    # 4. ESA CCI AGB v6.0 
    print(f"Mulai download ESA CCI AGB v6.0 ({YEAR_AGB}) - Region: {REGION_NAME}...")
    
    # Gunakan ID Koleksi dari Community Catalog
    agb_col = ee.ImageCollection("projects/sat-io/open-datasets/ESA/ESA_CCI_AGB")
    
    # Filter Tahun 2021
    agb_img = agb_col.filterDate(f'{YEAR_AGB}-01-01', f'{YEAR_AGB}-12-31').first()
    
    # IMPROVED: Add validation
    if agb_img is None:
        print("✗ ESA CCI data tidak tersedia untuk tahun tersebut!")
        return
    
    # Band 1 = AGB (Mg/ha) 
    # Band 2 = SD (Uncertainty) - OPTIONAL but recommended
    
    # 4a. Download AGB main label
    agb_selected = agb_img.select(['b1']).rename(['agb'])
    agb_resampled = agb_selected.reproject(crs=ref_proj, scale=10) \
        .resample('bilinear') \
        .clip(roi) \
        .unmask(0) 
    
    # IMPROVED: Include region name in filename
    out_agb_path = os.path.join(OUTPUT_DIR, f"AGB_{REGION_NAME}_Label.tif")
    geemap.download_ee_image(
        agb_resampled, 
        out_agb_path, 
        region=roi, 
        crs=ref_proj, 
        scale=10, 
        dtype='float32', 
        overwrite=True
    )
    print(f"✓ AGB Label berhasil diunduh: {out_agb_path}")
    
    # 4b. RECOMMENDED: Download uncertainty band (for skenario 2)
    try:
        agb_sd_selected = agb_img.select(['b2']).rename(['agb_sd'])
        agb_sd_resampled = agb_sd_selected.reproject(crs=ref_proj, scale=10) \
            .resample('bilinear') \
            .clip(roi) \
            .unmask(0)
        
        out_agb_sd_path = os.path.join(OUTPUT_DIR, f"AGB_{REGION_NAME}_SD.tif")
        geemap.download_ee_image(
            agb_sd_resampled, 
            out_agb_sd_path, 
            region=roi, 
            crs=ref_proj, 
            scale=10, 
            dtype='float32', 
            overwrite=True
        )
        print(f"✓ AGB Uncertainty berhasil diunduh: {out_agb_sd_path}")
    except Exception as e:
        print(f"⚠️  AGB Uncertainty tidak tersedia: {str(e)}")
    
    # Summary
    print("\n" + "="*60)
    print("DOWNLOAD SUMMARY")
    print("="*60)
    print(f"Region: {REGION_NAME}")
    print(f"Year: {YEAR_S2}")
    print(f"Sentinel-2: 10 bands × 12 months")
    print(f"ESA CCI AGB: 1 label band + (optional) 1 uncertainty band")
    print(f"Resolution: 10m × 10m")
    print(f"Output directory: {OUTPUT_DIR}")
    print("="*60)
    print("✓ Download selesai!")
    print(f"Ready untuk preprocessing (pytorch-dataset-validation.md)")

if __name__ == "__main__":
    main()

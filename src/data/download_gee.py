import ee
import geemap
import os

# --- KONFIGURASI ---
PROJECT_ID = 'data-skripsi-473712'
ASSET_ID = 'projects/data-skripsi-473712/assets/Lampung'
OUTPUT_DIR = 'data/raw'


YEAR_S2 = 2021
YEAR_AGB = 2021 

def main():
    # 1. Autentikasi
    try:
        ee.Initialize(project=PROJECT_ID)
    except:
        ee.Authenticate()
        ee.Initialize(project=PROJECT_ID)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("✓ Terhubung ke Earth Engine.")

    # 2. ROI 
    roi = ee.FeatureCollection(ASSET_ID).geometry()

    # 3. Sentinel-2 
    def mask_s2(image):
        qa = image.select('QA60')
        mask = qa.bitwiseAnd(1<<10).eq(0).And(qa.bitwiseAnd(1<<11).eq(0))
        return image.updateMask(mask).select(['B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12'])

    s2_col = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterBounds(roi).filterDate(f'{YEAR_S2}-01-01', f'{YEAR_S2}-12-31') \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 40)).map(mask_s2)

    ref_proj = s2_col.first().select('B2').projection()

    print(f"Mulai download Sentinel-2 ({YEAR_S2})...")
    for m in range(1, 13):
        start = ee.Date.fromYMD(YEAR_S2, m, 1)
        end = start.advance(1, 'month')
        img = s2_col.filterDate(start, end).median().unmask(0).clip(roi)
        
        fname = os.path.join(OUTPUT_DIR, f"S2_M{m:02d}.tif")
        # Cek jika file sudah ada agar tidak download ulang
        if not os.path.exists(fname):
            print(f"   Processing Bulan {m}...")
            geemap.download_ee_image(img, fname, region=roi, crs=ref_proj, scale=10, dtype='uint16', overwrite=True)
        else:
            print(f"   ✓ Bulan {m} sudah ada.")

    # 4. ESA CCI AGB v6.0 
    print(f"Mulai download ESA CCI AGB v6.0 ({YEAR_AGB})...")
    
    # Gunakan ID Koleksi dari Community Catalog
    agb_col = ee.ImageCollection("projects/sat-io/open-datasets/ESA/ESA_CCI_AGB")
    
    # Filter Tahun 2021
    agb_img = agb_col.filterDate(f'{YEAR_AGB}-01-01', f'{YEAR_AGB}-12-31').first()
    
    # Catatan:
    # Band 1 = AGB (Mg/ha) 
    # Band 2 = SD (Uncertainty)
    agb_selected = agb_img.select(['b1']).rename(['agb'])
    
    # Resampling
    agb_resampled = agb_selected.reproject(crs=ref_proj, scale=10) \
        .resample('bilinear') \
        .clip(roi) \
        .unmask(0) 
    
    # Download
    out_agb_path = os.path.join(OUTPUT_DIR, "AGB_Label.tif")
    geemap.download_ee_image(
        agb_resampled, 
        out_agb_path, 
        region=roi, 
        crs=ref_proj, 
        scale=10, 
        dtype='float32', 
        overwrite=True
    )
    print("✓ Download Selesai (Menggunakan ESA CCI v6.0).")

if __name__ == "__main__":
    main()

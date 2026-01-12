# FILE: download_data_CORRECTED.py
import ee
import geemap
import os

PROJECT_ID = 'data-skripsi-473712'
ASSET_ID = 'projects/data-skripsi-473712/assets/Lampung'
OUTPUT_DIR = 'data/raw'
REGION_NAME = 'Lampung'
YEAR_S2 = 2021
YEAR_AGB = 2021 

def main():
    # Authentication
    try:
        ee.Initialize(project=PROJECT_ID)
        print("✓ GEE initialized.")
    except:
        ee.Authenticate()
        ee.Initialize(project=PROJECT_ID)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    roi = ee.FeatureCollection(ASSET_ID).geometry()

    # Sentinel-2 masking
    def mask_s2(image):
        qa = image.select('QA60')
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

    # Download Sentinel-2
    print(f"Downloading Sentinel-2 ({YEAR_S2})...")
    for m in range(1, 13):
        start = ee.Date.fromYMD(YEAR_S2, m, 1)
        end = start.advance(1, 'month')
        
        # ✅ CORRECT: unmask(0) OK untuk S2
        img = s2_col.filterDate(start, end).median().unmask(0).clip(roi)
        
        fname = os.path.join(OUTPUT_DIR, f"S2_{REGION_NAME}_M{m:02d}.tif")
        
        if not os.path.exists(fname):
            print(f"   Processing bulan {m}...")
            geemap.download_ee_image(
                img, fname, region=roi, crs=ref_proj, scale=10, 
                dtype='uint16', overwrite=True
            )
        else:
            print(f"   ✓ Bulan {m} already exists.")

    # ESA CCI AGB
    print(f"Downloading ESA CCI AGB v6.0 ({YEAR_AGB})...")
    
    agb_col = ee.ImageCollection("projects/sat-io/open-datasets/ESA/ESA_CCI_AGB")
    agb_img = agb_col.filterDate(f'{YEAR_AGB}-01-01', f'{YEAR_AGB}-12-31').first()
    
    if agb_img is None:
        print("✗ ESA CCI data not available!")
        return
    
    agb_selected = agb_img.select(['b1']).rename(['agb'])
    
    # ✅ CORRECT: Bilinear resampling untuk continuous data
    # ⚠️ CHOICE: unmask(0) vs keep masked
    
    # OPTION A: Simple (acceptable untuk thesis)
    agb_resampled_A = agb_selected.reproject(crs=ref_proj, scale=10) \
        .resample('bilinear') \
        .clip(roi) \
        .unmask(0)  # Treat no-data as zero biomass
    
    # OPTION B: Better (recommended - explicit no-data)
    agb_resampled_B = agb_selected.reproject(crs=ref_proj, scale=10) \
        .resample('bilinear') \
        .clip(roi) \
        .unmask(-9999)  # Explicit no-data value
    
    # CHOOSE ONE:
    agb_resampled = agb_resampled_A  # OR agb_resampled_B
    
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
    print(f"✓ AGB downloaded: {out_agb_path}")
    
    # BONUS: Download uncertainty band
    try:
        agb_sd_selected = agb_img.select(['b2']).rename(['agb_sd'])
        agb_sd_resampled = agb_sd_selected.reproject(crs=ref_proj, scale=10) \
            .resample('bilinear') \
            .clip(roi) \
            .unmask(0)  # SD of 0 = no uncertainty (reasonable default)
        
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
        print(f"✓ AGB Uncertainty downloaded: {out_agb_sd_path}")
    except Exception as e:
        print(f"⚠️ AGB Uncertainty not available: {e}")

    print("="*60)
    print("DOWNLOAD COMPLETE")
    print(f"Region: {REGION_NAME}")
    print(f"Resampling method: BILINEAR (correct for continuous AGB data)")
    print(f"Masking: unmask(0) for simplicity (or use -9999 for explicit no-data)")
    print("="*60)

if __name__ == "__main__":
    main()

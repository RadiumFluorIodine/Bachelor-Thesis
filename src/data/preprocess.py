import rasterio
import numpy as np
import os
import glob
import json
from rasterio.windows import Window
from tqdm import tqdm

# --- KONFIGURASI ---
INPUT_DIR = '../../data/raw'
OUTPUT_DIR = '../../data/processed'
PATCH_SIZE = 128
STRIDE = 128
NODATA_THRESHOLD = 0.2
S2_MAX = 10000  # Max reflectance value from GEE
AGB_MAX = 500   # Max expected AGB Mg/ha

def compute_normalization_stats(s2_files, agb_file):
    """Compute mean/std for proper normalization"""
    s2_values = []
    agb_values = []
    
    with rasterio.open(agb_file) as agb_src:
        agb_data = agb_src.read(1)
        agb_values = agb_data[agb_data > 0].flatten()  # Exclude NoData
    
    for s2_file in s2_files:
        with rasterio.open(s2_file) as src:
            # Sample 10% of pixels for efficiency
            data = src.read(1)
            s2_values.extend(data[data > 0].flatten()[::10])  # Exclude zeros
    
    return {
        's2_mean': float(np.mean(s2_values)),
        's2_std': float(np.std(s2_values)),
        'agb_mean': float(np.mean(agb_values)),
        'agb_std': float(np.std(agb_values)),
    }

def check_patch_validity(s2_np, agb_np, threshold=NODATA_THRESHOLD):
    """
    Comprehensive NoData check
    s2_np: (T, C, H, W) - can contain NaN
    agb_np: (H, W) - can contain NaN
    """
    # Count valid pixels (not NaN, not zero)
    s2_valid = ~np.isnan(s2_np)  # True if NOT NaN
    agb_valid = ~np.isnan(agb_np)
    
    # Pixel is valid if ALL bands in ALL months are valid AND AGB is valid
    valid_pixels = s2_valid.all(axis=(0, 1)) & agb_valid  # (H, W) boolean
    
    valid_ratio = valid_pixels.sum() / valid_pixels.size
    
    return valid_pixels, valid_ratio > (1 - threshold)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load File paths
    s2_files = sorted(glob.glob(os.path.join(INPUT_DIR, "S2_M*.tif")))
    agb_file = os.path.join(INPUT_DIR, "AGB_Label.tif")
    
    if len(s2_files) != 12 or not os.path.exists(agb_file):
        print("❌ Data raw tidak lengkap. Jalankan download_gee.py dulu.")
        return

    # Compute normalization stats
    print("📊 Computing normalization statistics...")
    norm_stats = compute_normalization_stats(s2_files, agb_file)
    
    # Open handlers
    s2_srcs = [rasterio.open(f) for f in s2_files]
    agb_src = rasterio.open(agb_file)
    
    h, w = s2_srcs.height, s2_srcs.width
    print(f"📏 Dimensi Citra: {w}x{h}")
    print(f"📈 Normalization Stats:")
    print(f"   S2: mean={norm_stats['s2_mean']:.1f}, std={norm_stats['s2_std']:.1f}")
    print(f"   AGB: mean={norm_stats['agb_mean']:.1f}, std={norm_stats['agb_std']:.1f}")

    # Better patch position logic (handle edges)
    patch_positions_r = list(range(0, h - PATCH_SIZE + 1, STRIDE))
    patch_positions_c = list(range(0, w - PATCH_SIZE + 1, STRIDE))
    if (h - PATCH_SIZE) not in patch_positions_r:
        patch_positions_r.append(h - PATCH_SIZE)
    if (w - PATCH_SIZE) not in patch_positions_c:
        patch_positions_c.append(w - PATCH_SIZE)
    
    total_patches_estimate = len(patch_positions_r) * len(patch_positions_c)
    
    patch_id = 0
    valid_count = 0
    
    with tqdm(total=total_patches_estimate, desc="Generating Patches") as pbar:
        for r in patch_positions_r:
            for c in patch_positions_c:
                window = Window(c, r, PATCH_SIZE, PATCH_SIZE)
                
                # 1. Baca S2 dari semua 12 bulan
                stack = []
                for src in s2_srcs:
                    # Read as float to handle NaN
                    arr = src.read(window=window).astype(np.float32)
                    stack.append(arr)
                
                s2_np = np.stack(stack, axis=0)  # (12, 10, 128, 128)

                # 2. Baca AGB
                agb_np = agb_src.read(1, window=window).astype(np.float32)

                # 3. Check validity
                valid_mask, is_valid = check_patch_validity(s2_np, agb_np)

                if not is_valid:
                    pbar.update(1)
                    continue

                # 4. NORMALIZE (CRITICAL!)
                # Replace NaN with mean before normalization
                s2_np = np.nan_to_num(s2_np, nan=norm_stats['s2_mean'])
                agb_np = np.nan_to_num(agb_np, nan=norm_stats['agb_mean'])
                
                # Normalize to mean=0, std=1
                s2_norm = (s2_np - norm_stats['s2_mean']) / (norm_stats['s2_std'] + 1e-8)
                agb_norm = (agb_np - norm_stats['agb_mean']) / (norm_stats['agb_std'] + 1e-8)
                
                # 5. Save with metadata
                np.savez_compressed(
                    os.path.join(OUTPUT_DIR, f"{patch_id:05d}.npz"),
                    image=s2_norm.astype(np.float32),
                    label=agb_norm.astype(np.float32),
                    valid_mask=valid_mask.astype(np.bool_)
                )
                
                patch_id += 1
                valid_count += 1
                pbar.update(1)

    # Save normalization stats
    with open(os.path.join(OUTPUT_DIR, 'normalization.json'), 'w') as f:
        json.dump(norm_stats, f, indent=2)

    # Close
    for src in s2_srcs: 
        src.close()
    agb_src.close()
    
    print(f"✅ Total Patches Processed: {patch_id}")
    print(f"✅ Valid Patches Saved: {valid_count}")
    print(f"📊 Normalization stats saved to: {OUTPUT_DIR}/normalization.json")

if __name__ == "__main__":
    main()

import rasterio
import numpy as np
import os
import glob
from rasterio.windows import Window
from tqdm import tqdm

# --- KONFIGURASI ---
INPUT_DIR = '../../data/raw'
OUTPUT_DIR = '../../data/processed'
PATCH_SIZE = 128
STRIDE = 128 
NODATA_THRESHOLD = 0.2

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load File paths
    s2_files = sorted(glob.glob(os.path.join(INPUT_DIR, "S2_M*.tif")))
    agb_file = os.path.join(INPUT_DIR, "AGB_Label.tif")
    
    if len(s2_files) != 12 or not os.path.exists(agb_file):
        print("❌ Data raw tidak lengkap. Jalankan download_gee.py dulu.")
        return

    # Open handlers
    s2_srcs = [rasterio.open(f) for f in s2_files]
    agb_src = rasterio.open(agb_file)
    
    h, w = s2_srcs[0].height, s2_srcs[0].width
    print(f"📏 Dimensi Citra: {w}x{h}")

    patch_id = 0
    total_patches = ((h // STRIDE) + 1) * ((w // STRIDE) + 1)
    
    with tqdm(total=total_patches, desc="Generating Patches") as pbar:
        for r in range(0, h - PATCH_SIZE, STRIDE):
            for c in range(0, w - PATCH_SIZE, STRIDE):
                window = Window(c, r, PATCH_SIZE, PATCH_SIZE)
                
                # 1. Baca S2 
                stack = []
                valid = True
                for src in s2_srcs:
                    # Read as uint16 
                    arr = src.read(window=window)
                    if arr.shape != (10, PATCH_SIZE, PATCH_SIZE):
                        valid = False
                        break
                    stack.append(arr)
                
                if not valid: continue
                s2_np = np.stack(stack, axis=0) # (12, 10, 128, 128)

                # 2. Baca AGB
                agb_np = agb_src.read(window=window) # (1, 128, 128)

                # 3. Filter NoData 
                # Cek jika > 20% pixel kosong
                if (np.count_nonzero(s2_np[0,0]==0) / s2_np[0,0].size) > NODATA_THRESHOLD:
                    continue

                # 4. Save Compressed (Hemat Space)
                # S2: uint16, AGB: float16
                np.savez_compressed(
                    os.path.join(OUTPUT_DIR, f"{patch_id:05d}.npz"),
                    image=s2_np.astype(np.uint16),
                    label=agb_np.astype(np.float16)
                )
                patch_id += 1
                pbar.update(1)

    # Close
    for src in s2_srcs: src.close()
    agb_src.close()
    print(f"✅ Total Patch Siap Pakai: {patch_id}")

if __name__ == "__main__":
    main()

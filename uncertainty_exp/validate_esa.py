import torch
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from models.utae_prob import UTAE_Probabilistic

# Config
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = 'uncertainty_exp/outputs/models/best_utae_prob.pth'
ESA_SD_PATH = 'data/raw/AGB_SD_Reference.tif'
ESA_AGB_PATH = 'data/raw/AGB_Label.tif'

def validate():
    print("🔍 Validasi Ketidakpastian vs ESA CCI SD...")
    
    # Load Model
    model = UTAE_Probabilistic(in_c=10).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    
    # Load Sample Data (Menggunakan patch dummy/custom loader untuk demo)
    # Di implementasi nyata, gunakan loader yang sama dengan predict_custom.py
    # untuk mengambil area yang sama dengan ESA SD
    print("⚠️  Pastikan Anda mengarahkan ke area crop yang sama.")
    
    # ... (Gunakan logika compare_uncertainty.py dari jawaban sebelumnya) ...
    # ... (Sesuaikan path import model ke models.utae_prob) ...

if __name__ == "__main__":
    validate()

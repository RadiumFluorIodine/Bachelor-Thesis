import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.utils.data import DataLoader
from src.data.dataset import BiomassDataset
from src.models.utae import UTAE
from tqdm import tqdm

# --- KONFIGURASI ---
BATCH_SIZE = 8
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = 'best_model.pth' # Pastikan file ini ada setelah training

def evaluate():
    print(f"📊 Mengevaluasi model pada device: {DEVICE}")
    
    # 1. Load Dataset (Gunakan Validation/Test set)
    # Di sini kita load semua processed data, idealnya Anda pisah folder test sendiri
    dataset = BiomassDataset('data/processed')
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 2. Load Model
    model = UTAE(in_c=10, out_c=1).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    
    y_true = []
    y_pred = []
    
    print("🔄 Menjalankan inferensi...")
    with torch.no_grad():
        for img, lbl, days in tqdm(loader):
            img, lbl, days = img.to(DEVICE), lbl.to(DEVICE), days.to(DEVICE)
            
            # Predict
            output = model(img, days)
            
            # Simpan hasil (Flatkan array untuk statistik)
            # Ambil region yang valid (Label != 0) untuk evaluasi yang adil
            # agar background hitam tidak menaikkan akurasi secara palsu
            mask = lbl >= 0 
            
            if mask.sum() > 0:
                y_true.extend(lbl[mask].cpu().numpy().flatten())
                y_pred.extend(output[mask].cpu().numpy().flatten())
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # 3. Hitung Metrik
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print("\n" + "="*30)
    print("📈 HASIL EVALUASI AKHIR")
    print("="*30)
    print(f"RMSE (Root Mean Sq Error) : {rmse:.4f} Mg/ha")
    print(f"MAE  (Mean Abs Error)     : {mae:.4f} Mg/ha")
    print(f"R² Score                  : {r2:.4f}")
    print("="*30)
    
    # 4. Visualisasi Scatter Plot
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.1, s=1, c='blue')
    
    # Garis ideal 1:1
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([0, max_val], [0, max_val], 'r--', label='Ideal Fit')
    
    plt.title(f'Prediksi vs Aktual (R²={r2:.2f})')
    plt.xlabel('Ground Truth AGB (Mg/ha)')
    plt.ylabel('Predicted AGB (Mg/ha)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('eval_scatter_plot.png', dpi=300)
    print("🖼️  Scatter plot disimpan ke 'eval_scatter_plot.png'")
    plt.show()

if __name__ == "__main__":
    evaluate()

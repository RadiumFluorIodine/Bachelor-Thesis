import torch
import torch.optim as optim
import sys
import os
from tqdm import tqdm

# Import Shared Modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from src.data.dataset import BiomassDataset
from models.utae_prob import UTAE_Probabilistic
from models.loss import HeteroscedasticLoss
from torch.utils.data import DataLoader, random_split

# Config
BATCH_SIZE = 8
EPOCHS = 50
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SAVE_DIR = 'uncertainty_exp/outputs/models'
os.makedirs(SAVE_DIR, exist_ok=True)

def main():
    print(f"🎲 Training Probabilistic U-TAE on {DEVICE}")
    
    # Load Dataset (Shared)
    dataset = BiomassDataset('data/processed')
    train_len = int(0.8 * len(dataset))
    train_set, val_set = random_split(dataset, [train_len, len(dataset)-train_len])
    
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)
    
    # Init Model & Loss Baru
    model = UTAE_Probabilistic(in_c=10).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = HeteroscedasticLoss()
    
    best_loss = float('inf')
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Ep {epoch+1}")
        for img, lbl, days in pbar:
            img, lbl, days = img.to(DEVICE), lbl.to(DEVICE), days.to(DEVICE)
            
            optimizer.zero_grad()
            out = model(img, days) # Output (B, 2, H, W)
            loss = criterion(out, lbl)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'gnll_loss': loss.item()})
            
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for img, lbl, days in val_loader:
                out = model(img.to(DEVICE), days.to(DEVICE))
                val_loss += criterion(out, lbl.to(DEVICE)).item()
        
        avg_val = val_loss / len(val_loader)
        print(f"   Validation GNLL Loss: {avg_val:.4f}")
        
        if avg_val < best_loss:
            best_loss = avg_val
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'best_utae_prob.pth'))
            print("   💾 Model Saved!")

if __name__ == "__main__":
    main()

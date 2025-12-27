import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from src.data.dataset import BiomassDataset
from src.models.utae import UTAE
from tqdm import tqdm

# --- CONFIG ---
BATCH_SIZE = 8
EPOCHS = 50
LR = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def main():
    print(f"🚀 Training on {DEVICE}")
    
    # 1. Dataset
    dataset = BiomassDataset('data/processed')
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
    
    # 2. Model
    model = UTAE(in_c=10, out_c=1).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss() # Regresi Loss
    
    # 3. Training Loop
    best_val_loss = float('inf')
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for img, lbl, days in pbar:
            img, lbl, days = img.to(DEVICE), lbl.to(DEVICE), days.to(DEVICE)
            
            optimizer.zero_grad()
            output = model(img, days)
            loss = criterion(output, lbl)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for img, lbl, days in val_loader:
                img, lbl, days = img.to(DEVICE), lbl.to(DEVICE), days.to(DEVICE)
                output = model(img, days)
                val_loss += criterion(output, lbl).item()
        
        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}: Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")
        
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), 'best_model.pth')
            print("💾 Model Saved!")

if __name__ == "__main__":
    main()

import torch
from torch.utils.data import Dataset
import numpy as np
import os

class BiomassDataset(Dataset):
    def __init__(self, root_dir):
        self.files = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith('.npz')]
        # Positional Encoding Day (Asumsi pertengahan tiap bulan)
        self.days = torch.tensor([15, 45, 75, 105, 135, 165, 195, 225, 255, 285, 315, 345]).float()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        
        # Load S2 (uint16) -> Convert to Float & Normalize (0-1)
        # Input shape: (Time, Band, H, W)
        img = data['image'].astype(np.float32) / 10000.0
        img = np.clip(img, 0, 1) # Clip agar range aman

        # Load AGB -> Convert to Float32
        lbl = data['label'].astype(np.float32)

        return torch.from_numpy(img), torch.from_numpy(lbl), self.days

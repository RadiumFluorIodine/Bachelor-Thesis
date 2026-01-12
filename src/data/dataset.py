import torch
from torch.utils.data import Dataset
import numpy as np
import os
import json

class BiomassDataset(Dataset):
    """
    PyTorch Dataset for U-TAE AGB Estimation
    
    Expected file structure:
    - root_dir/00000.npz, 00001.npz, ...
    - root_dir/normalization.json
    
    Each .npz contains:
    - 'image': (T=12, C=10, H=128, W=128) normalized via preprocessing
    - 'label': (H=128, W=128) AGB values normalized via preprocessing
    - 'valid_mask': (H=128, W=128) boolean mask
    """
    
    def __init__(self, root_dir, mode='train'):
        """
        Args:
            root_dir: Directory containing .npz files and normalization.json
            mode: 'train' or 'test' (affects normalization)
        """
        self.root_dir = root_dir
        self.mode = mode
        
        # Load file paths
        self.files = sorted([
            os.path.join(root_dir, f) 
            for f in os.listdir(root_dir) 
            if f.endswith('.npz')
        ])
        
        if len(self.files) == 0:
            raise FileNotFoundError(f"No .npz files found in {root_dir}")
        
        # Load normalization statistics
        norm_path = os.path.join(root_dir, 'normalization.json')
        if os.path.exists(norm_path):
            with open(norm_path) as f:
                self.norm_stats = json.load(f)
        else:
            print("⚠️ WARNING: normalization.json not found, using default values")
            self.norm_stats = {
                's2_mean': 4500.0,
                's2_std': 2000.0,
                'agb_mean': 150.0,
                'agb_std': 80.0
            }
        
        # Temporal encoding (day of year for each month)
        self.temporal_encoding = self._create_temporal_encoding()
    
    def _create_temporal_encoding(self):
        """
        Create sinusoidal positional encoding for temporal information
        Following Vaswani et al. 2017 "Attention is All You Need"
        """
        # Day of year at middle of each month
        days = torch.tensor([15, 45, 75, 105, 135, 165, 195, 225, 255, 285, 315, 345]).float()
        days_normalized = days / 365.0  # Normalize to [0, 1]
        
        # Generate sinusoidal encoding
        d_model = 64
        position = torch.arange(len(days)).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            -(np.log(10000.0) / d_model)
        )
        
        pe = torch.zeros(len(days), d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe  # Shape: (12, 64)
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        """
        Load and return single sample
        
        Returns:
            image: (T=12, C=10, H=128, W=128) - normalized image
            label: (H=128, W=128) - normalized AGB label
            valid_mask: (H=128, W=128) - boolean mask for valid pixels
            temporal_encoding: (T=12, D=64) - temporal positional encoding
        """
        try:
            # Load .npz file
            data = np.load(self.files[idx])
            
            # Validate required keys
            required_keys = ['image', 'label']
            for key in required_keys:
                if key not in data:
                    raise KeyError(f"Missing '{key}' in {self.files[idx]}")
            
            # Load and convert image
            # If data is already normalized (from fixed preprocessing):
            # Skip denormalization
            img = data['image'].astype(np.float32)
            
            # Validate image shape
            if img.shape != (12, 10, 128, 128):
                raise ValueError(f"Invalid image shape: {img.shape}, expected (12, 10, 128, 128)")
            
            # Load and convert label
            lbl = data['label'].astype(np.float32)
            
            # Validate label shape
            if lbl.shape != (128, 128):
                raise ValueError(f"Invalid label shape: {lbl.shape}, expected (128, 128)")
            
            # Load valid mask (if available)
            if 'valid_mask' in data:
                valid_mask = data['valid_mask'].astype(np.bool_)
            else:
                # Fallback: all pixels valid
                valid_mask = np.ones((128, 128), dtype=np.bool_)
                
            if valid_mask.shape != (128, 128):
                raise ValueError(f"Invalid valid_mask shape: {valid_mask.shape}")
            
            # Convert to torch tensors
            img_tensor = torch.from_numpy(img)  # (12, 10, 128, 128)
            lbl_tensor = torch.from_numpy(lbl)  # (128, 128)
            mask_tensor = torch.from_numpy(valid_mask)  # (128, 128)
            
            return {
                'image': img_tensor,
                'label': lbl_tensor,
                'valid_mask': mask_tensor,
                'temporal_encoding': self.temporal_encoding  # (12, 64)
            }
            
        except Exception as e:
            print(f"❌ ERROR loading {self.files[idx]}: {str(e)}")
            raise


# DataLoader Collate Function
def collate_fn_biomass(batch):
    """
    Custom collate function for BiomassDataset
    Handles batch stacking with proper dimension management
    """
    images = torch.stack([item['image'] for item in batch])  # (B, 12, 10, 128, 128)
    labels = torch.stack([item['label'] for item in batch])  # (B, 128, 128)
    masks = torch.stack([item['valid_mask'] for item in batch])  # (B, 128, 128)
    
    # Temporal encoding is same for all samples (shared)
    temporal_enc = batch['temporal_encoding']  # (12, 64)
    
    return {
        'image': images,
        'label': labels,
        'valid_mask': masks,
        'temporal_encoding': temporal_enc
    }


# Example usage in training loop
if __name__ == "__main__":
    from torch.utils.data import DataLoader
    
    # Create dataset
    dataset = BiomassDataset(root_dir='../../data/processed', mode='train')
    print(f"✅ Loaded {len(dataset)} patches")
    
    # Create dataloader
    train_loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn_biomass
    )
    
    # Test single batch
    for batch in train_loader:
        print(f"Batch shapes:")
        print(f"  - image: {batch['image'].shape}")  # (32, 12, 10, 128, 128)
        print(f"  - label: {batch['label'].shape}")  # (32, 128, 128)
        print(f"  - valid_mask: {batch['valid_mask'].shape}")  # (32, 128, 128)
        print(f"  - temporal_encoding: {batch['temporal_encoding'].shape}")  # (12, 64)
        
        # Verify valid_mask contains both True and False
        unique_mask = batch['valid_mask'].unique()
        print(f"  - Valid mask values: {unique_mask}")
        
        break

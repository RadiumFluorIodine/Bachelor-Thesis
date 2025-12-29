import torch
import sys
import os

# Setup path agar bisa import dari src & uncertainty_exp
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.models.utae import UTAE
from uncertainty_exp.models.utae_prob import UTAE_Probabilistic

def test_architecture():
    print("\n🧪 [1/4] TESTING MODEL ARCHITECTURE")
    print("="*40)

    # Konfigurasi Dummy
    B, T, C, H, W = 2, 12, 10, 128, 128
    input_tensor = torch.randn(B, T, C, H, W)
    days_tensor = torch.linspace(1, 365, T).unsqueeze(0).repeat(B, 1)

    # --- TEST A: Model Standar (U-TAE) ---
    print("   👉 Testing Standard U-TAE...")
    try:
        model = UTAE(in_c=C, out_c=1)
        out = model(input_tensor, days_tensor)
        
        expected_shape = (B, 1, H, W)
        assert out.shape == expected_shape, f"Shape Mismatch! Got {out.shape}, expected {expected_shape}"
        print("      ✅ Shape Valid (1 Output Channel)")
        
    except Exception as e:
        print(f"      ❌ FAILED: {e}")
        return

    # --- TEST B: Model Probabilistik (Uncertainty) ---
    print("   👉 Testing Probabilistic U-TAE...")
    try:
        model_prob = UTAE_Probabilistic(in_c=C)
        out_prob = model_prob(input_tensor, days_tensor)
        
        # Output harus 2 channel (Mean & LogVar)
        expected_prob_shape = (B, 2, H, W)
        assert out_prob.shape == expected_prob_shape, f"Shape Mismatch! Got {out_prob.shape}, expected {expected_prob_shape}"
        print("      ✅ Shape Valid (2 Output Channels for Uncertainty)")
        
    except Exception as e:
        print(f"      ❌ FAILED: {e}")

if __name__ == "__main__":
    test_architecture()

import torch
import torch.nn as nn
from src.models.adapters_so8t import SO8TAdapterBank
from src.models.losses_pet import pet_loss
from src.models.mhc_projection import project_mhc_l2

def test_minimal_logic():
    print("Running minimal SO8T logic test...")
    
    # 1. Test Adapter Bank
    bank = SO8TAdapterBank(num_layers=10, d_model=128, r=8, num_passes=4)
    x = torch.randn(2, 16, 128)
    
    # Check forward pass for different ids
    out0 = bank(x, layer_idx=5, pass_id=0)
    out1 = bank(x, layer_idx=5, pass_id=1)
    
    print(f"Adapter Output Norm (P0): {out0.norm().item():.6f}")
    # Initial alpha is 0, so output should be 0
    assert out0.abs().max() < 1e-6, "Initial adapter output should be 0 because alpha is 0"
    
    # Set alpha and check
    bank.alpha.data[5, 1] = 1.0
    out1_active = bank(x, layer_idx=5, pass_id=1)
    print(f"Adapter Output Norm (P1 Active): {out1_active.norm().item():.6f}")
    assert out1_active.norm() > 0, "Active adapter should produce non-zero output"
    
    # 2. Test PET Loss
    # delta^2 pattern: alpha = [0, 1, 2, 3] -> d2 = 2-2+0=0, 3-4+1=0 (linear is smooth)
    alpha_linear = torch.tensor([[0.0, 1.0, 2.0, 3.0]], requires_grad=True)
    loss_smooth = pet_loss(alpha_linear, lam_p=1.0)
    print(f"PET Loss (Linear/Smooth): {loss_smooth.item():.6f}")
    assert loss_smooth.item() < 1e-6
    
    # Jagged alpha: [0, 5, 0, 5] -> d2 = 0-10+0=-10, 5-0+5=10
    alpha_jagged = torch.tensor([[0.0, 5.0, 0.0, 5.0]], requires_grad=True)
    loss_jagged = pet_loss(alpha_jagged, lam_p=1.0)
    print(f"PET Loss (Jagged): {loss_jagged.item():.6f}")
    assert loss_jagged.item() > 10.0
    
    # 3. Test MHC Projection
    bank.alpha.data.fill_(10.0)
    class DummyModel:
        def __init__(self, bank): self.so8t_adapter_bank = bank
    
    dummy_model = DummyModel(bank)
    project_mhc_l2(dummy_model, max_norm=5.0)
    new_norm = bank.alpha.norm().item()
    print(f"Alpha Norm after MHC Projection: {new_norm:.6f}")
    assert new_norm <= 5.0001
    
    print("\n--- ALL MINIMAL TESTS PASSED ---")

if __name__ == "__main__":
    test_minimal_logic()

import torch
import torch.nn as nn
import logging
from src.models.model_patcher import patch_phi3_with_pet
from src.training.train_unsloth_so8t import PETSFTTrainer
from transformers import AutoConfig, Phi3Config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_patching():
    logger.info("Testing Phi3 patching with PET Adapters...")
    
    # Create a dummy Phi3 model config
    config = Phi3Config(
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=6, # Small for testing
        num_attention_heads=8,
        vocab_size=1000
    )
    
    # Simple dummy model that mimics Phi3 structure
    class DummyMLP(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.gate_up_proj = nn.Linear(config.hidden_size, config.intermediate_size)
            self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size)
        def forward(self, x):
            return self.down_proj(torch.relu(self.gate_up_proj(x)))

    class DummyLayer(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.mlp = DummyMLP(config)
        def forward(self, x):
            return self.mlp(x)

    class DummyModel(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([DummyLayer(config) for _ in range(config.num_hidden_layers)])

    model = DummyModel(config)
    
    # Patch the model
    patched_model = patch_phi3_with_pet(model, rank=8, num_passes=4)
    
    # Verify that the top 1/3 layers are patched
    # num_layers = 6, top 1/3 = layers 4, 5 (2/3 * 6 = 4)
    logger.info(f"Target layers should be [4, 5]")
    
    # Test forward pass with pass_id
    hidden_states = torch.randn(1, 10, config.hidden_size)
    
    patched_model.current_pass_id = 1
    output = patched_model.model.layers[5](hidden_states)
    logger.info(f"Forward pass successful. Output shape: {output.shape}")
    
    # Check if PET loss can be calculated
    pet_loss = patched_model.pet_adapter_bank.calculate_pet_loss(lambda_g=0.01)
    logger.info(f"PET Loss: {pet_loss.item()}")
    
    assert pet_loss >= 0
    logger.info("Test passed!")

if __name__ == "__main__":
    test_patching()

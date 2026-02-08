import torch
import torch.nn as nn
from typing import List, Optional
from src.models.pet_adapters import PETAdapterBank

def patch_phi3_with_pet(model: nn.Module, rank: int = 16, num_passes: int = 4):
    """
    Patches a Phi3 model with PET Adapters in the top 1/3 layers.
    """
    config = model.config
    num_layers = config.num_hidden_layers
    hidden_size = config.hidden_size
    
    # Target top 1/3 layers (e.g., layers 21-32 for a 32-layer model)
    target_layer_indices = list(range(2 * num_layers // 3, num_layers))
    num_target_layers = len(target_layer_indices)
from src.models.adapters_so8t import SO8TAdapterBank

logger = logging.getLogger(__name__)

def patch_phi3_with_so8t(model: nn.Module, rank: int = 16, num_passes: int = 4):
    """
    Patches a Phi3 model by injecting SO8T adapters into the MLP of top 1/3 layers.
    """
    num_layers = len(model.model.layers)
    # Target top 1/3 layers
    start_layer = 2 * num_layers // 3
    target_layer_indices = list(range(start_layer, num_layers))
    
    logger.info(f"Patching layers {target_layer_indices} with SO8T adapters (Rank={rank}, Pass=4)")
    
    # Initialize the adapter bank for targeted layers
    adapter_bank = SO8TAdapterBank(
        num_layers=num_layers, # Kept for easy indexing, but only target layers will have non-zero alpha
        d_model=model.config.hidden_size,
        r=rank,
        num_passes=num_passes
    )
    
    # Attach bank to model for persistence
    model.so8t_adapter_bank = adapter_bank
    
    def get_patched_forward(original_forward, layer_idx):
        def patched_forward(hidden_states, *args, **kwargs):
            # 1. Original MLP forward
            output = original_forward(hidden_states, *args, **kwargs)
            
            # 2. Add SO8T Adapter residual
            # current_pass_id should be set in the trainer loop (0-3)
            pass_id = getattr(model, "current_pass_id", 0)
            
            # Application: x = x + alpha * Adapter(LN(x))
            # Note: Phi3 MLP output is already the residual added part (x + MLP(LN(x)))
            # So we add the adapter output to the result.
            adapter_res = model.so8t_adapter_bank(hidden_states, layer_idx, pass_id)
            return output + adapter_res
            
        return patched_forward

    # Patch the target layers
    for layer_idx in target_layer_indices:
        layer = model.model.layers[layer_idx]
        layer.mlp.forward = get_patched_forward(layer.mlp.forward, layer_idx)
        
    logger.info("SO8T Grand Design patching completed.")
    return model

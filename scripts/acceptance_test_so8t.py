import torch
import torch.nn as nn
from transformers import AutoTokenizer, Phi3Config, Phi3ForCausalLM
from src.models.model_patcher import patch_phi3_with_so8t
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_so8t_pass_consistency():
    """
    Acceptance Test: Verify that pass_id changes output and adapters are active.
    """
    logger.info("Initializing Dummy Phi3 for SO8T Acceptance Test (CPU)...")
    config = Phi3Config(
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=3, 
        num_attention_heads=8,
        vocab_size=32000
    )
    model = Phi3ForCausalLM(config).to("cpu")
    
    # 1. Patch model
    model = patch_phi3_with_so8t(model, rank=16, num_passes=4)
    
    # Intentionally set alpha to non-zero to see effect
    with torch.no_grad():
        model.so8t_adapter_bank.alpha.fill_(1.0)
    
    input_ids = torch.tensor([[1, 2, 3, 4]]).to("cpu")
    
    # 2. Compare outputs across passes
    outputs = []
    for p_id in range(4):
        model.current_pass_id = p_id
        with torch.no_grad():
            out = model(input_ids).logits
            outputs.append(out)
            
    # Check for difference
    diff12 = torch.norm(outputs[0] - outputs[1]).item()
    diff23 = torch.norm(outputs[1] - outputs[2]).item()
    diff34 = torch.norm(outputs[2] - outputs[3]).item()
    
    logger.info(f"Logit Diff (P1 vs P2): {diff12:.6f}")
    logger.info(f"Logit Diff (P2 vs P3): {diff23:.6f}")
    logger.info(f"Logit Diff (P3 vs P4): {diff34:.6f}")
    
    if diff12 > 0 and diff23 > 0 and diff34 > 0:
        logger.info("SUCCESS: SO8T Pass-ID consistency verified. Adapters are influencing outputs.")
    else:
        logger.error("FAILURE: Pass-ID changes did not influence outputs. Check patch logic.")

if __name__ == "__main__":
    try:
        test_so8t_pass_consistency()
    except Exception as e:
        logger.error(f"Test failed with error: {e}")

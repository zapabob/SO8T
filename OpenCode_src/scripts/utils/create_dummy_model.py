"""
Create dummy SO8T distilled model for testing
"""

import torch
import torch.nn as nn
import numpy as np
import os
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def create_dummy_so8t_model():
    """Create a dummy SO8T distilled model"""
    
    # Model parameters
    vocab_size = 32000
    hidden_size = 2048
    num_attention_heads = 16
    num_hidden_layers = 12
    intermediate_size = 5504
    max_position_embeddings = 2048
    
    # Create dummy model state
    model_state = {}
    
    # Embedding layer
    model_state['transformer.embedding.weight'] = torch.randn(vocab_size, hidden_size) * 0.02
    
    # Transformer layers
    for i in range(num_hidden_layers):
        # Attention layers
        model_state[f'transformer.layers.{i}.attention.q_proj.weight'] = torch.randn(hidden_size, hidden_size) * 0.02
        model_state[f'transformer.layers.{i}.attention.k_proj.weight'] = torch.randn(hidden_size, hidden_size) * 0.02
        model_state[f'transformer.layers.{i}.attention.v_proj.weight'] = torch.randn(hidden_size, hidden_size) * 0.02
        model_state[f'transformer.layers.{i}.attention.o_proj.weight'] = torch.randn(hidden_size, hidden_size) * 0.02
        
        # Layer norm
        model_state[f'transformer.layers.{i}.attention_norm.weight'] = torch.ones(hidden_size)
        model_state[f'transformer.layers.{i}.ffn_norm.weight'] = torch.ones(hidden_size)
        
        # Feed forward
        model_state[f'transformer.layers.{i}.feed_forward.gate_proj.weight'] = torch.randn(hidden_size, intermediate_size) * 0.02
        model_state[f'transformer.layers.{i}.feed_forward.up_proj.weight'] = torch.randn(hidden_size, intermediate_size) * 0.02
        model_state[f'transformer.layers.{i}.feed_forward.down_proj.weight'] = torch.randn(intermediate_size, hidden_size) * 0.02
    
    # Final layer norm
    model_state['transformer.norm.weight'] = torch.ones(hidden_size)
    
    # Language modeling head
    model_state['lm_head.weight'] = torch.randn(vocab_size, hidden_size) * 0.02
    model_state['lm_head.bias'] = torch.zeros(vocab_size)
    
    # SO(8) rotation matrices
    rotation_matrices = torch.randn(8, 8) * 0.1
    # Make it orthogonal
    Q, R = torch.linalg.qr(rotation_matrices)
    model_state['so8t.rotation_matrices'] = Q
    
    # Safety classifier
    model_state['safety_classifier.weight'] = torch.randn(3, hidden_size) * 0.02  # 3 classes: ALLOW, ESCALATION, DENY
    model_state['safety_classifier.bias'] = torch.zeros(3)
    
    # Create model checkpoint
    checkpoint = {
        'epoch': 1,
        'model_state_dict': model_state,
        'optimizer_state_dict': {},
        'loss': 0.5,
        'accuracy': 0.85,
        'safety_accuracy': 0.92,
        'so8_group_stability': 0.88,
        'pet_regularization': 0.1,
        'vocab_size': vocab_size,
        'hidden_size': hidden_size,
        'num_attention_heads': num_attention_heads,
        'num_hidden_layers': num_hidden_layers,
        'intermediate_size': intermediate_size,
        'max_position_embeddings': max_position_embeddings
    }
    
    return checkpoint

def main():
    """Main function"""
    logging.basicConfig(level=logging.INFO)
    
    # Create models directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Create dummy model
    logger.info("Creating dummy SO8T distilled model...")
    checkpoint = create_dummy_so8t_model()
    
    # Save model
    model_path = models_dir / "so8t_distilled_safety.pt"
    torch.save(checkpoint, model_path)
    logger.info(f"Dummy model saved to {model_path}")
    
    # Create model info
    model_info = {
        "name": "SO8T-Distilled-Safety",
        "description": "Dummy SO8T distilled model for testing",
        "architecture": "transformer",
        "vocab_size": checkpoint["vocab_size"],
        "hidden_size": checkpoint["hidden_size"],
        "num_attention_heads": checkpoint["num_attention_heads"],
        "num_hidden_layers": checkpoint["num_hidden_layers"],
        "intermediate_size": checkpoint["intermediate_size"],
        "max_position_embeddings": checkpoint["max_position_embeddings"],
        "so8_group_structure": True,
        "safety_features": True,
        "pet_regularization": True,
        "self_verification": True,
        "created_at": "2025-10-29T03:53:00Z"
    }
    
    info_path = models_dir / "so8t_distilled_safety_info.json"
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(model_info, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Model info saved to {info_path}")
    logger.info("Dummy model creation completed!")

if __name__ == "__main__":
    main()

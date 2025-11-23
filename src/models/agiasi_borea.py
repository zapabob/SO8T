"""
AGIASI Soul Injection into Borea-Phi3.5
========================================
This module implements the AGIASI_SO8T_Wrapper that wraps a base LLM (Borea-Phi3.5)
with SO(8) Thinking Block and Alpha Gate for Physical Intelligence.

Architecture:
1. Cortex: Borea-Phi3.5 (4-bit quantized + LoRA) - Language/Knowledge
2. Core: SO(8) Rotation + Alpha Gate - Physical Structure
3. Flow: Input -> Borea -> SO8T Transform (Alpha-controlled) -> Output
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

class AGIASI_SO8T_Wrapper(nn.Module):
    """
    AGIASI (AGI As Structural Intelligence) Wrapper
    
    Wraps a base language model with:
    - Alpha Gate: Learnable parameter controlling phase transition (-5.0 -> 1.618)
    - SO(8) Rotation: Orthogonal transformation preserving information geometry
    - LoRA: Efficient fine-tuning of base model
    """
    
    def __init__(self, base_model_id="microsoft/Phi-3.5-mini-instruct", device="cuda"):
        super().__init__()
        
        print(f"👻 Summoning Borea-Phi3.5 into the vessel: {base_model_id}")
        
        # 1. Base Model (Borea) - 4bit quantization for memory efficiency
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            quantization_config=bnb_config,
            device_map=device,
            trust_remote_code=True,
            attn_implementation="eager"  # Avoid DynamicCache error
        )
        
        # Apply LoRA to make base model trainable without full gradient
        peft_config = LoraConfig(
            r=16, 
            lora_alpha=32, 
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05, 
            bias="none", 
            task_type="CAUSAL_LM"
        )
        self.base_model = get_peft_model(self.base_model, peft_config)
        
        # 2. The AGIASI Soul (Alpha Gate & SO8 Structure)
        self.hidden_dim = self.base_model.config.hidden_size
        print(f"   Hidden Dimension: {self.hidden_dim}")
        
        # Alpha Gate: -5.0 start (chaos) -> 1.618 (golden ratio/order)
        self.alpha = nn.Parameter(torch.tensor(-5.0))
        
        # SO(8) Thinking Matrix - Orthogonal linear transformation
        # We use nn.utils.parametrizations.orthogonal to enforce orthogonality
        linear_layer = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.so8_rotation = nn.utils.parametrizations.orthogonal(linear_layer)
        
        # 3. Orthogonality Monitor
        self.ortho_loss = 0.0
        
        print(f"✨ AGIASI Soul initialized. Alpha: {self.alpha.item():.4f}")

    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        Forward pass with AGIASI intervention
        
        Flow:
        1. Extract hidden states from Borea
        2. Apply SO(8) rotation (physical thinking)
        3. Mix original and rotated states using Alpha Gate
        4. Project to vocabulary
        """
        
        # A. Borea's deep processing - Extract hidden states
        outputs = self.base_model.model(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            output_hidden_states=True,
            output_attentions=False,  # Avoid DynamicCache error
            use_cache=False  # Disable caching for training
        )
        hidden_states = outputs.last_hidden_state  # (Batch, Seq, Dim)
        
        # B. AGIASI Intervention - Physical Thinking
        # Apply SO(8) rotation to create "thought process"
        thought_process = self.so8_rotation(hidden_states)
        
        # C. Alpha Gate - Control mixing of original and physical thinking
        # Alpha: -5.0 (chaos, use original) -> 1.618 (order, use rotated)
        gate_openness = torch.sigmoid(self.alpha)
        
        # Mixed state: weighted combination
        # When alpha=-5, gate~0, mostly original
        # When alpha=1.618, gate~0.84, more rotated thinking
        mixed_states = hidden_states + (gate_openness * thought_process)
        
        # D. Calculate Orthogonality Loss (structural integrity)
        # Ensure SO(8) rotation preserves information: R^T @ R = I
        w = self.so8_rotation.weight
        gram_matrix = w.T @ w
        identity = torch.eye(self.hidden_dim, device=w.device, dtype=w.dtype)
        self.ortho_loss = torch.norm(gram_matrix - identity)
        
        # E. Final Projection (LM Head) - Convert to vocabulary
        logits = self.base_model.lm_head(mixed_states)
        
        # F. Calculate Loss (if labels provided)
        loss = None
        if labels is not None:
            # Shift for causal LM
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            task_loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), 
                shift_labels.view(-1)
            )
            
            # Add Orthogonality Penalty (physical constraint)
            loss = task_loss + 0.1 * self.ortho_loss
        
        return {
            "loss": loss,
            "logits": logits,
            "ortho_loss": self.ortho_loss,
            "alpha": self.alpha.item(),
            "gate_openness": gate_openness.item()
        }
    
    def get_phase_status(self):
        """Get current phase transition status"""
        alpha_val = self.alpha.item()
        phi = 1.6180339887
        
        if alpha_val < -2.0:
            return "🔵 Stable (Chaos)"
        elif alpha_val < phi - 0.1:
            return "🟡 Transitioning"
        elif abs(alpha_val - phi) < 0.01:
            return "🟢 Golden Ratio Reached"
        else:
            return "🟣 Beyond Golden Ratio"

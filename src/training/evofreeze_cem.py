import torch
import torch.nn as nn
import logging
import numpy as np
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class EvoFreezeCEM:
    """
    EvoFreeze-CEM: Evolutionary mask selection using Cross-Entropy Method.
    Features a Rollback Engine for training stability.
    """
    def __init__(self, model: nn.Module, elite_ratio: float = 0.2, smoothing: float = 0.1):
        self.model = model
        self.generation = 0
        self.elite_ratio = elite_ratio
        self.smoothing = smoothing
        
        # Submodule groups (G0 to G8 as per grand design)
        self.groups = self._init_groups()
        # Probability distribution q(m)
        self.probabilities = {group: 0.1 for group in self.groups}
        
        # Fixed probabilities (G0, G1 usually frozen, G6 always plastic)
        self.probabilities["embeddings"] = 0.0
        self.probabilities["lm_head"] = 0.0
        self.probabilities["so8t_adapters"] = 1.0
        self.probabilities["multimodal_projector"] = 1.0

    def _init_groups(self) -> List[str]:
        # Logic to categorize model parameters into G0-G8
        groups = set(["embeddings", "lm_head", "lower_attn_mlp", "upper_mlp", "upper_attn", "norms", "so8t_adapters", "multimodal_projector"])
        return sorted(list(groups))

    def sample_mask(self) -> Dict[str, bool]:
        """Samples a Bernoulli mask for the current generation."""
        return {group: (np.random.rand() < self.probabilities[group]) for group in self.groups}

    def apply_mask(self, mask: Dict[str, bool]):
        """Sets requires_grad based on the sampled group mask."""
        num_layers = getattr(self.model.config, "num_hidden_layers", 32)
        upper_bound = 2 * num_layers // 3
        
        for name, param in self.model.named_parameters():
            group = self._get_group_for_param(name, num_layers, upper_bound)
            param.requires_grad = mask.get(group, False)
            
            # Forced plastic groups
            if any(k in name.lower() for k in ["adapter", "alpha", "pet", "projector"]):
                param.requires_grad = True

    def _get_group_for_param(self, name: str, num_layers: int, upper_bound: int) -> str:
        name_lower = name.lower()
        if "embed" in name_lower: return "embeddings"
        if "lm_head" in name_lower: return "lm_head"
        if "adapter" in name_lower or "alpha" in name_lower: return "so8t_adapters"
        if "projector" in name_lower: return "multimodal_projector"
        if "norm" in name_lower: return "norms"
        
        if "layers." in name:
            layer_idx = int(name.split("layers.")[1].split(".")[0])
            if layer_idx >= upper_bound:
                if "mlp" in name_lower: return "upper_mlp"
                if "attn" in name_lower: return "upper_attn"
            else:
                return "lower_attn_mlp"
        
        return "other"

    def rollback_and_adjust(self, checkpoint_path: str):
        """
        Rollback Engine: Reverts to stable checkpoint and scales down unfreezing probabilities.
        """
        logger.warning(f"ROLLBACK TRIGGERED: Reverting to {checkpoint_path}")
        # Placeholder: Load checkpoint to model
        # state_dict = torch.load(checkpoint_path)
        # self.model.load_state_dict(state_dict)
        
        # Penalize all non-fixed groups
        for group in self.probabilities:
            if group not in ["so8t_adapters", "multimodal_projector"]:
                self.probabilities[group] *= 0.5 # Halve unfreezing probability
                
    def evolve(self, elite_masks: List[Dict[str, bool]]):
        """Updates q(m) based on top performing masks."""
        self.generation += 1
        num_elites = len(elite_masks)
        for group in self.probabilities:
            if group in ["so8t_adapters", "multimodal_projector"]: continue
            
            elite_mean = sum(1 for m in elite_masks if m.get(group, False)) / num_elites
            self.probabilities[group] = (1 - self.smoothing) * self.probabilities[group] + self.smoothing * elite_mean

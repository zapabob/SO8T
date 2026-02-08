import torch
import torch.nn as nn
import logging
import numpy as np
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class EvolutionaryConfigManager:
    """
    EvoFreeze-TRM: Manages parameter groups evolutionarily at the submodule level.
    Uses CEM (Cross-Entropy Method) for mask selection and fitness optimization.
    """
    def __init__(self, model: nn.Module, elite_ratio: float = 0.2, smoothing: float = 0.1):
        self.model = model
        self.generation = 0
        self.elite_ratio = elite_ratio
        self.smoothing = smoothing
        
        # Identify submodule groups
        self.groups = self._identify_submodule_groups()
        # Initial unfreezing probabilities (p=1.0 for always trainable, low for others)
        self.probabilities = {group_name: 0.1 for group_name in self.groups}
        
        # Always trainable (Adapters, Gates, Norms)
        self.fixed_trainable = ["adapter", "alpha", "pet", "norm"]
        for name in self.probabilities:
            if any(k in name.lower() for k in self.fixed_trainable):
                self.probabilities[name] = 1.0

    def _identify_submodule_groups(self) -> List[str]:
        group_names = set()
        for name, _ in self.model.named_parameters():
            # Example name: model.layers.24.mlp.down_proj.weight
            parts = name.split(".")
            if "layers" in parts:
                idx = parts.index("layers")
                layer_num = parts[idx + 1]
                submodule = parts[idx + 2] # mlp, self_attn, input_layernorm, etc.
                group_names.add(f"layer_{layer_num}_{submodule}")
            else:
                # Global parameters
                group_names.add(parts[0])
        return sorted(list(group_names))

    def sample_mask(self) -> Dict[str, bool]:
        """Samples a binary mask based on current probabilities."""
        return {name: (np.random.rand() < p) for name, p in self.probabilities.items()}

    def apply_mask(self, mask: Dict[str, bool]):
        """Applies the sampled mask to the model's requires_grad status."""
        for name, param in self.model.named_parameters():
            parts = name.split(".")
            current_group = None
            if "layers" in parts:
                idx = parts.index("layers")
                layer_num = parts[idx + 1]
                submodule = parts[idx + 2]
                current_group = f"layer_{layer_num}_{submodule}"
            else:
                current_group = parts[0]
            
            # Apply mask if exists, else default to trainable (should be covered by probability 1.0)
            param.requires_grad = mask.get(current_group, True)

    def evolve_probabilities(self, elite_masks: List[Dict[str, bool]]):
        """
        CEM Update: Moves probabilities towards the mean of elite masks.
        """
        self.generation += 1
        num_elites = len(elite_masks)
        if num_elites == 0:
            return

        for name in self.probabilities:
            # Skip fixed trainable
            if any(k in name.lower() for k in self.fixed_trainable):
                continue
                
            elite_mean = sum(1 for m in elite_masks if m.get(name, False)) / num_elites
            # Moving average with smoothing to prevent premature convergence
            self.probabilities[name] = (1 - self.smoothing) * self.probabilities[name] + self.smoothing * elite_mean
            
        logger.info(f"Generation {self.generation}: Probabilities updated via CEM.")

    def get_fitness(self, task_reward: float, kl_div: float, rep_drift: float, pet_acc: float, 
                    beta: float = 1.0, gamma: float = 1.0, eta: float = 0.5) -> float:
        """
        F(m) = J_task - β*D_KL - γ*D_rep - η*P_acc
        """
        fitness = task_reward - (beta * kl_div) - (gamma * rep_drift) - (eta * pet_acc)
        return fitness

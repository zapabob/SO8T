"""
SO8T Training Loss Functions

This module implements the composite loss function for SO8T Safe Agent training,
including task loss, safety loss, PET regularization, and safety penalties.

The total loss is computed as:
L_total = L_task + α * L_safety + β * L_rationale + γ(epoch) * L_pet + δ * L_penalty + ε * L_reward

Where:
- L_task: Language modeling loss for task responses
- L_safety: Classification loss for safety decisions
- L_rationale: Language modeling loss for safety rationales
- L_pet: PET regularization for temporal consistency
- L_penalty: Penalty for dangerous compliance
- L_reward: Reward for appropriate escalation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import logging
import math

logger = logging.getLogger(__name__)


class PETLoss(nn.Module):
    """
    PET (Positional Embedding Regularization) Loss
    
    This loss encourages temporal consistency by penalizing rapid changes
    in hidden states over time. It helps stabilize the safety personality
    in later training phases.
    """
    
    def __init__(self, lambda_pet: float = 0.1):
        """
        Initialize PET loss.
        
        Args:
            lambda_pet: Weight for PET regularization
        """
        super().__init__()
        self.lambda_pet = lambda_pet
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Compute PET loss.
        
        Args:
            hidden_states: Hidden states from the model [batch_size, seq_len, hidden_size]
            
        Returns:
            PET loss value
        """
        if hidden_states.size(1) < 3:  # Need at least 3 tokens for second-order difference
            return torch.tensor(0.0, device=hidden_states.device)
        
        # Compute second-order differences (curvature)
        # First-order difference
        first_diff = hidden_states[:, 1:, :] - hidden_states[:, :-1, :]
        
        # Second-order difference
        second_diff = first_diff[:, 1:, :] - first_diff[:, :-1, :]
        
        # Compute L2 norm of second-order differences
        pet_loss = torch.mean(torch.norm(second_diff, p=2, dim=-1))
        
        return self.lambda_pet * pet_loss


class SafetyAwareLoss(nn.Module):
    """
    Safety-Aware Composite Loss Function
    
    Combines multiple loss components to train the SO8T model with
    safety-first principles and temporal consistency.
    """
    
    def __init__(
        self,
        task_weight: float = 1.0,
        safety_weight: float = 2.0,
        rationale_weight: float = 1.0,
        pet_weight: float = 0.1,
        safety_penalty_weight: float = 5.0,
        escalate_reward_weight: float = 2.0,
        pet_lambda: float = 0.1,
        safety_threshold: float = 0.8
    ):
        """
        Initialize safety-aware loss.
        
        Args:
            task_weight: Weight for task loss
            safety_weight: Weight for safety classification loss
            rationale_weight: Weight for rationale generation loss
            pet_weight: Weight for PET regularization
            safety_penalty_weight: Weight for safety penalty
            escalate_reward_weight: Weight for escalation reward
            pet_lambda: Lambda for PET regularization
            safety_threshold: Threshold for safety confidence
        """
        super().__init__()
        self.task_weight = task_weight
        self.safety_weight = safety_weight
        self.rationale_weight = rationale_weight
        self.pet_weight = pet_weight
        self.safety_penalty_weight = safety_penalty_weight
        self.escalate_reward_weight = escalate_reward_weight
        self.safety_threshold = safety_threshold
        
        # Initialize PET loss
        self.pet_loss = PETLoss(lambda_pet=pet_lambda)
        
        # Loss functions
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-100)
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def forward(
        self,
        task_logits: torch.Tensor,
        safety_logits: torch.Tensor,
        rationale_logits: Optional[torch.Tensor],
        task_labels: torch.Tensor,
        safety_labels: torch.Tensor,
        rationale_labels: Optional[torch.Tensor],
        hidden_states: torch.Tensor,
        epoch: int = 0,
        total_epochs: int = 100
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the composite loss.
        
        Args:
            task_logits: Task head logits [batch_size, seq_len, vocab_size]
            safety_logits: Safety head logits [batch_size, num_classes]
            rationale_logits: Rationale head logits [batch_size, seq_len, vocab_size]
            task_labels: Task labels [batch_size, seq_len]
            safety_labels: Safety labels [batch_size]
            rationale_labels: Rationale labels [batch_size, seq_len]
            hidden_states: Hidden states [batch_size, seq_len, hidden_size]
            epoch: Current epoch
            total_epochs: Total number of epochs
            
        Returns:
            Dictionary containing individual losses and total loss
        """
        device = task_logits.device
        
        # 1. Task Loss (Language Modeling)
        task_loss = self._compute_task_loss(task_logits, task_labels)
        
        # 2. Safety Classification Loss
        safety_loss = self._compute_safety_loss(safety_logits, safety_labels)
        
        # 3. Rationale Generation Loss
        rationale_loss = self._compute_rationale_loss(rationale_logits, rationale_labels)
        
        # 4. PET Regularization Loss
        pet_loss = self._compute_pet_loss(hidden_states, epoch, total_epochs)
        
        # 5. Safety Penalty (for dangerous compliance)
        safety_penalty = self._compute_safety_penalty(safety_logits, safety_labels)
        
        # 6. Escalation Reward (for appropriate escalation)
        escalate_reward = self._compute_escalate_reward(safety_logits, safety_labels)
        
        # Compute total loss
        total_loss = (
            self.task_weight * task_loss +
            self.safety_weight * safety_loss +
            self.rationale_weight * rationale_loss +
            self.pet_weight * pet_loss +
            self.safety_penalty_weight * safety_penalty -
            self.escalate_reward_weight * escalate_reward
        )
        
        return {
            "total_loss": total_loss,
            "task_loss": task_loss,
            "safety_loss": safety_loss,
            "rationale_loss": rationale_loss,
            "pet_loss": pet_loss,
            "safety_penalty": safety_penalty,
            "escalate_reward": escalate_reward
        }
    
    def _compute_task_loss(self, task_logits: torch.Tensor, task_labels: torch.Tensor) -> torch.Tensor:
        """Compute task loss (language modeling)."""
        if task_logits is None or task_labels is None:
            return torch.tensor(0.0, device=task_logits.device if task_logits is not None else "cpu")
        
        # Shift logits and labels for next token prediction
        shift_logits = task_logits[..., :-1, :].contiguous()
        shift_labels = task_labels[..., 1:].contiguous()
        
        return self.ce_loss(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
    
    def _compute_safety_loss(self, safety_logits: torch.Tensor, safety_labels: torch.Tensor) -> torch.Tensor:
        """Compute safety classification loss."""
        if safety_logits is None or safety_labels is None:
            return torch.tensor(0.0, device=safety_logits.device if safety_logits is not None else "cpu")
        
        return self.ce_loss(safety_logits, safety_labels)
    
    def _compute_rationale_loss(self, rationale_logits: torch.Tensor, rationale_labels: torch.Tensor) -> torch.Tensor:
        """Compute rationale generation loss."""
        if rationale_logits is None or rationale_labels is None:
            return torch.tensor(0.0, device=rationale_logits.device if rationale_logits is not None else "cpu")
        
        # Shift logits and labels for next token prediction
        shift_logits = rationale_logits[..., :-1, :].contiguous()
        shift_labels = rationale_labels[..., 1:].contiguous()
        
        return self.ce_loss(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
    
    def _compute_pet_loss(self, hidden_states: torch.Tensor, epoch: int, total_epochs: int) -> torch.Tensor:
        """Compute PET regularization loss with scheduling."""
        if hidden_states is None:
            return torch.tensor(0.0, device="cpu")
        
        # Schedule PET weight based on epoch
        # Start with low weight, increase in later epochs
        progress = epoch / total_epochs
        pet_schedule = self._pet_schedule(progress)
        
        # Compute PET loss
        pet_loss = self.pet_loss(hidden_states)
        
        return pet_schedule * pet_loss
    
    def _pet_schedule(self, progress: float) -> float:
        """
        Schedule PET weight based on training progress.
        
        Args:
            progress: Training progress (0.0 to 1.0)
            
        Returns:
            PET weight multiplier
        """
        # Sigmoid schedule: starts low, increases in later phases
        # This helps stabilize the safety personality in later training
        return 1.0 / (1.0 + math.exp(-10.0 * (progress - 0.6)))
    
    def _compute_safety_penalty(self, safety_logits: torch.Tensor, safety_labels: torch.Tensor) -> torch.Tensor:
        """
        Compute penalty for dangerous compliance.
        
        Penalizes cases where the model should REFUSE but predicts ALLOW.
        """
        if safety_logits is None or safety_labels is None:
            return torch.tensor(0.0, device=safety_logits.device if safety_logits is not None else "cpu")
        
        # Get safety probabilities
        safety_probs = F.softmax(safety_logits, dim=-1)
        
        # Find cases where model should REFUSE (label=1) but predicts ALLOW (class=0)
        should_refuse = (safety_labels == 1)  # REFUSE label
        allow_prob = safety_probs[:, 0]  # ALLOW probability
        
        # Penalty for high ALLOW probability when should REFUSE
        penalty = torch.where(
            should_refuse,
            allow_prob,  # Penalty proportional to ALLOW probability
            torch.zeros_like(allow_prob)
        )
        
        return torch.mean(penalty)
    
    def _compute_escalate_reward(self, safety_logits: torch.Tensor, safety_labels: torch.Tensor) -> torch.Tensor:
        """
        Compute reward for appropriate escalation.
        
        Rewards cases where the model correctly predicts ESCALATE.
        """
        if safety_logits is None or safety_labels is None:
            return torch.tensor(0.0, device=safety_logits.device if safety_logits is not None else "cpu")
        
        # Get safety probabilities
        safety_probs = F.softmax(safety_logits, dim=-1)
        
        # Find cases where model should ESCALATE (label=2)
        should_escalate = (safety_labels == 2)  # ESCALATE label
        escalate_prob = safety_probs[:, 2]  # ESCALATE probability
        
        # Reward for high ESCALATE probability when should ESCALATE
        reward = torch.where(
            should_escalate,
            escalate_prob,  # Reward proportional to ESCALATE probability
            torch.zeros_like(escalate_prob)
        )
        
        return torch.mean(reward)


class SafetyMetrics:
    """
    Compute safety metrics for evaluation.
    """
    
    def __init__(self, safety_labels: Tuple[str, ...] = ("ALLOW", "REFUSE", "ESCALATE")):
        """
        Initialize safety metrics.
        
        Args:
            safety_labels: Tuple of safety label names
        """
        self.safety_labels = safety_labels
        self.num_classes = len(safety_labels)
    
    def compute_metrics(
        self,
        safety_logits: torch.Tensor,
        safety_labels: torch.Tensor,
        safety_threshold: float = 0.5
    ) -> Dict[str, float]:
        """
        Compute safety metrics.
        
        Args:
            safety_logits: Safety head logits [batch_size, num_classes]
            safety_labels: Ground truth safety labels [batch_size]
            safety_threshold: Threshold for confidence
            
        Returns:
            Dictionary containing safety metrics
        """
        device = safety_logits.device
        
        # Get predictions
        safety_probs = F.softmax(safety_logits, dim=-1)
        predictions = torch.argmax(safety_logits, dim=-1)
        
        # Compute accuracy
        accuracy = (predictions == safety_labels).float().mean().item()
        
        # Compute per-class metrics
        metrics = {"accuracy": accuracy}
        
        for i, label in enumerate(self.safety_labels):
            # True positives, false positives, false negatives
            tp = ((predictions == i) & (safety_labels == i)).sum().item()
            fp = ((predictions == i) & (safety_labels != i)).sum().item()
            fn = ((predictions != i) & (safety_labels == i)).sum().item()
            
            # Precision, recall, F1
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            metrics.update({
                f"{label.lower()}_precision": precision,
                f"{label.lower()}_recall": recall,
                f"{label.lower()}_f1": f1
            })
        
        # Compute safety-specific metrics
        metrics.update(self._compute_safety_specific_metrics(safety_logits, safety_labels))
        
        return metrics
    
    def _compute_safety_specific_metrics(
        self,
        safety_logits: torch.Tensor,
        safety_labels: torch.Tensor
    ) -> Dict[str, float]:
        """Compute safety-specific metrics."""
        safety_probs = F.softmax(safety_logits, dim=-1)
        
        # Refuse Recall: How well does the model refuse dangerous requests?
        should_refuse = (safety_labels == 1)  # REFUSE label
        refused = (torch.argmax(safety_logits, dim=-1) == 1)  # Predicted REFUSE
        refuse_recall = (should_refuse & refused).sum().item() / should_refuse.sum().item() if should_refuse.sum().item() > 0 else 0.0
        
        # Escalate Precision: How well does the model escalate when it should?
        should_escalate = (safety_labels == 2)  # ESCALATE label
        escalated = (torch.argmax(safety_logits, dim=-1) == 2)  # Predicted ESCALATE
        escalate_precision = (should_escalate & escalated).sum().item() / escalated.sum().item() if escalated.sum().item() > 0 else 0.0
        
        # Allow Precision: How well does the model allow safe requests?
        should_allow = (safety_labels == 0)  # ALLOW label
        allowed = (torch.argmax(safety_logits, dim=-1) == 0)  # Predicted ALLOW
        allow_precision = (should_allow & allowed).sum().item() / allowed.sum().item() if allowed.sum().item() > 0 else 0.0
        
        # Safety Score: Overall safety performance
        safety_score = (refuse_recall + escalate_precision + allow_precision) / 3.0
        
        return {
            "refuse_recall": refuse_recall,
            "escalate_precision": escalate_precision,
            "allow_precision": allow_precision,
            "safety_score": safety_score
        }


def create_loss_function(config: Dict) -> SafetyAwareLoss:
    """
    Create a loss function from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        SafetyAwareLoss instance
    """
    return SafetyAwareLoss(
        task_weight=config.get("task_weight", 1.0),
        safety_weight=config.get("safety_weight", 2.0),
        rationale_weight=config.get("rationale_weight", 1.0),
        pet_weight=config.get("pet_weight", 0.1),
        safety_penalty_weight=config.get("safety_penalty_weight", 5.0),
        escalate_reward_weight=config.get("escalate_reward_weight", 2.0),
        pet_lambda=config.get("pet_lambda", 0.1),
        safety_threshold=config.get("safety_threshold", 0.8)
    )


# Example usage
if __name__ == "__main__":
    # Test loss function
    batch_size, seq_len, vocab_size, num_classes = 4, 128, 1000, 3
    hidden_size = 512
    
    # Create dummy data
    task_logits = torch.randn(batch_size, seq_len, vocab_size)
    safety_logits = torch.randn(batch_size, num_classes)
    rationale_logits = torch.randn(batch_size, seq_len, vocab_size)
    task_labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    safety_labels = torch.randint(0, num_classes, (batch_size,))
    rationale_labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    
    # Create loss function
    loss_fn = SafetyAwareLoss()
    
    # Compute loss
    losses = loss_fn(
        task_logits=task_logits,
        safety_logits=safety_logits,
        rationale_logits=rationale_logits,
        task_labels=task_labels,
        safety_labels=safety_labels,
        rationale_labels=rationale_labels,
        hidden_states=hidden_states,
        epoch=50,
        total_epochs=100
    )
    
    print("Loss components:")
    for key, value in losses.items():
        print(f"  {key}: {value.item():.4f}")
    
    # Test metrics
    metrics_fn = SafetyMetrics()
    metrics = metrics_fn.compute_metrics(safety_logits, safety_labels)
    
    print("\nSafety metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
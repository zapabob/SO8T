"""
SO8T Safe Agent Model

This module implements the SO8T (Safe Operation 8-Task) model architecture,
which extends a base language model with dual heads for task execution and safety classification.

The model consists of:
- TaskHeadA: Generates task responses, action plans, and tool execution steps
- SafetyHeadB: Classifies requests as ALLOW/REFUSE/ESCALATE and generates safety rationales

Architecture:
- Base model: Qwen2.5-7B-Instruct (128K context, 8K generation)
- Dual head structure with non-commutative gate (R_safe → R_cmd)
- PET regularization for temporal consistency
- Safety-first design with human-in-the-loop escalation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModel, 
    AutoTokenizer, 
    AutoConfig,
    PreTrainedModel,
    PretrainedConfig
)
from typing import Dict, List, Optional, Tuple, Union
import logging

# SO8T群構造のインポート
from .so8t_group_structure import SO8TGroupStructure, PETRegularization

logger = logging.getLogger(__name__)


class SO8TModelConfig:
    """Configuration class for SO8T model."""
    
    def __init__(
        self,
        base_model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        task_head_hidden_size: int = 4096,
        safety_head_hidden_size: int = 2048,
        safety_num_classes: int = 3,  # ALLOW, REFUSE, ESCALATE
        rationale_max_length: int = 256,
        pet_lambda: float = 0.1,
        safety_threshold: float = 0.8,
        vocab_size: int = 151936,  # Qwen2.5-7B-Instruct vocab size
        **kwargs
    ):
        self.base_model_name = base_model_name
        self.task_head_hidden_size = task_head_hidden_size
        self.safety_head_hidden_size = safety_head_hidden_size
        self.safety_num_classes = safety_num_classes
        self.rationale_max_length = rationale_max_length
        self.pet_lambda = pet_lambda
        self.safety_threshold = safety_threshold
        self.vocab_size = vocab_size


class TaskHeadA(nn.Module):
    """
    Task Head A: Generates task responses, action plans, and tool execution steps.
    
    This head is responsible for:
    - Generating natural language responses to user requests
    - Creating step-by-step action plans
    - Providing tool usage instructions
    - Maintaining conversation context
    """
    
    def __init__(self, config: SO8TModelConfig, base_hidden_size: int):
        super().__init__()
        self.config = config
        self.base_hidden_size = base_hidden_size
        
        # Main task head for response generation
        self.task_head = nn.Linear(base_hidden_size, config.task_head_hidden_size)
        self.task_activation = nn.GELU()
        self.task_dropout = nn.Dropout(0.1)
        
        # Output projection for vocabulary
        self.vocab_projection = nn.Linear(config.task_head_hidden_size, config.vocab_size)
        
        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(config.task_head_hidden_size)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for task head.
        
        Args:
            hidden_states: Hidden states from base model [batch_size, seq_len, hidden_size]
            
        Returns:
            Task logits [batch_size, seq_len, vocab_size]
        """
        # Process hidden states through task head
        x = self.task_head(hidden_states)
        x = self.task_activation(x)
        x = self.layer_norm(x)
        x = self.task_dropout(x)
        
        # Project to vocabulary space
        task_logits = self.vocab_projection(x)
        
        return task_logits


class SafetyHeadB(nn.Module):
    """
    Safety Head B: Classifies requests and generates safety rationales.
    
    This head is responsible for:
    - Classifying requests as ALLOW/REFUSE/ESCALATE
    - Generating safety rationales for decisions
    - Maintaining safety-first principles
    - Enabling human-in-the-loop escalation
    """
    
    def __init__(self, config: SO8TModelConfig, base_hidden_size: int):
        super().__init__()
        self.config = config
        self.base_hidden_size = base_hidden_size
        
        # Safety classification head
        self.safety_classifier = nn.Sequential(
            nn.Linear(base_hidden_size, config.safety_head_hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.safety_head_hidden_size, config.safety_num_classes)
        )
        
        # Safety rationale generation head (reduced output size for memory efficiency)
        self.rationale_head = nn.Sequential(
            nn.Linear(base_hidden_size, config.safety_head_hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.safety_head_hidden_size, 512)  # Reduced from vocab_size to 512
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(config.safety_head_hidden_size)
        
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for safety head.
        
        Args:
            hidden_states: Hidden states from base model [batch_size, seq_len, hidden_size]
            
        Returns:
            Tuple of (safety_logits, rationale_logits)
            - safety_logits: [batch_size, num_classes] for ALLOW/REFUSE/ESCALATE
            - rationale_logits: [batch_size, seq_len, vocab_size] for rationale generation
        """
        # Safety classification (use last token for classification)
        last_hidden = hidden_states[:, -1, :]  # [batch_size, hidden_size]
        safety_logits = self.safety_classifier(last_hidden)
        
        # Rationale generation
        rationale_logits = self.rationale_head(hidden_states)
        
        return safety_logits, rationale_logits


class SO8TModel(nn.Module):
    """
    SO8T Safe Agent Model
    
    A safety-first language model with dual heads for task execution and safety classification.
    Implements the SO8T architecture with non-commutative gates and PET regularization.
    """
    
    def __init__(self, config: SO8TModelConfig):
        super().__init__()
        self.config = config
        
        # Load base model with QLoRA/8bit quantization (RTX3060対応)
        self.base_model = AutoModel.from_pretrained(
            config.base_model_name,
            torch_dtype=torch.float16,  # RTX3060はfp16推奨
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            use_cache=False,  # メモリ効率化
            load_in_8bit=True,  # 8bit量子化（Windows対応）
            llm_int8_enable_fp32_cpu_offload=True,  # CPUオフロード有効化
            offload_folder="./offload_cache"  # CPUオフロード
        )
        
        # Enable gradient checkpointing for memory efficiency
        if hasattr(self.base_model, 'gradient_checkpointing_enable'):
            self.base_model.gradient_checkpointing_enable()
        
        # Freeze base model parameters (4bit量子化済み)
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # Get base model hidden size from the loaded model's config
        # Store the original base model config separately
        self.base_model_config = self.base_model.config
        base_hidden_size = self.base_model_config.hidden_size
        
        # Initialize SO8T group structure
        self.group_structure = SO8TGroupStructure(
            hidden_size=base_hidden_size,
            lambda_pet=config.pet_lambda
        )
        
        # Initialize dual heads
        self.task_head_a = TaskHeadA(config, base_hidden_size)
        self.safety_head_b = SafetyHeadB(config, base_hidden_size)
        
        # Move heads to the same device as base model
        if hasattr(self.base_model, 'device'):
            device = next(self.base_model.parameters()).device
            dtype = next(self.base_model.parameters()).dtype
            
            # データ型とデバイスを完全統一
            self.group_structure.to(device=device, dtype=dtype)
            self.task_head_a.to(device=device, dtype=dtype)
            self.safety_head_b.to(device=device, dtype=dtype)
        
        # Safety class labels
        self.safety_labels = ["ALLOW", "REFUSE", "ESCALATE"]
        
        # Initialize weights
        self.init_weights()
        
    def init_weights(self):
        """Initialize weights for the dual heads."""
        # Initialize task head weights
        nn.init.xavier_uniform_(self.task_head_a.task_head.weight)
        nn.init.zeros_(self.task_head_a.task_head.bias)
        nn.init.xavier_uniform_(self.task_head_a.vocab_projection.weight)
        nn.init.zeros_(self.task_head_a.vocab_projection.bias)
        
        # Initialize safety head weights
        for layer in self.safety_head_b.safety_classifier:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        
        for layer in self.safety_head_b.rationale_head:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        safety_labels: Optional[torch.Tensor] = None,
        rationale_labels: Optional[torch.Tensor] = None,
        return_dict: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for SO8T model.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            labels: Task labels for training [batch_size, seq_len]
            safety_labels: Safety labels for training [batch_size]
            rationale_labels: Rationale labels for training [batch_size, seq_len]
            return_dict: Whether to return a dictionary
            
        Returns:
            Dictionary containing model outputs
        """
        # Get base model outputs
        base_outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        hidden_states = base_outputs.last_hidden_state
        
        # Apply SO8T group structure (SO(8)群回転と非可換ゲート)
        group_output, group_info = self.group_structure(
            hidden_states, 
            return_group_info=True
        )
        
        # Triality三重推論実装
        # 1. ベクトル表現（タスク推論）- コマンド回転後の出力
        task_logits = self.task_head_a(group_output)
        
        # 2. スピノル表現S₊（安全推論）- 安全回転の監視
        safety_logits, rationale_logits = self.safety_head_b(hidden_states)
        
        # 3. スピノル表現S₋（権限推論）- 非可換ゲートから抽出
        # R_safeとR_cmdの非可換性を利用して権限判定
        escalation_logits = self._extract_escalation_logits(group_info)
        
        # Calculate losses if labels are provided
        total_loss = None
        task_loss = None
        safety_loss = None
        rationale_loss = None
        
        if labels is not None:
            # Task loss (language modeling)
            shift_logits = task_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            task_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )
            total_loss = task_loss
        
        if safety_labels is not None:
            # Safety classification loss
            safety_loss = F.cross_entropy(safety_logits, safety_labels)
            if total_loss is None:
                total_loss = safety_loss
            else:
                total_loss += safety_loss
        
        if rationale_labels is not None:
            # Ensure rationale_logits has the same sequence length as rationale_labels
            if rationale_logits.size(1) != rationale_labels.size(1):
                # Resize rationale_logits to match rationale_labels sequence length
                if rationale_logits.size(1) > rationale_labels.size(1):
                    # Truncate rationale_logits
                    rationale_logits = rationale_logits[:, :rationale_labels.size(1), :]
                else:
                    # Pad rationale_logits with zeros
                    pad_length = rationale_labels.size(1) - rationale_logits.size(1)
                    padding = torch.zeros(
                        rationale_logits.size(0), pad_length, rationale_logits.size(2),
                        device=rationale_logits.device, dtype=rationale_logits.dtype
                    )
                    rationale_logits = torch.cat([rationale_logits, padding], dim=1)
            
            # Convert rationale_labels to 512-dimensional embeddings for reduced memory usage
            # Use a simple embedding layer to map token IDs to 512 dimensions
            if not hasattr(self, 'rationale_embedding'):
                self.rationale_embedding = nn.Embedding(512, 512, padding_idx=-100).to(rationale_logits.device)
            
            # Map rationale_labels to 512-dimensional space
            rationale_embeddings = self.rationale_embedding(rationale_labels.clamp(0, 511))
            
            # Rationale generation loss (MSE instead of cross_entropy for memory efficiency)
            shift_rationale_logits = rationale_logits[..., :-1, :].contiguous()
            shift_rationale_embeddings = rationale_embeddings[..., 1:, :].contiguous()
            
            # Debug: Check batch size consistency
            logits_flat = shift_rationale_logits.view(-1, shift_rationale_logits.size(-1))
            embeddings_flat = shift_rationale_embeddings.view(-1, shift_rationale_embeddings.size(-1))
            
            if logits_flat.size(0) != embeddings_flat.size(0):
                print(f"ERROR: Batch size mismatch - logits: {logits_flat.size(0)}, embeddings: {embeddings_flat.size(0)}")
                print(f"Rationale logits shape: {rationale_logits.shape}")
                print(f"Rationale labels shape: {rationale_labels.shape}")
                print(f"Shift logits shape: {shift_rationale_logits.shape}")
                print(f"Shift embeddings shape: {shift_rationale_embeddings.shape}")
                # Skip this loss to prevent crash
                rationale_loss = torch.tensor(0.0, device=rationale_logits.device, requires_grad=True)
            else:
                # Use MSE loss instead of cross_entropy for memory efficiency
                rationale_loss = F.mse_loss(logits_flat, embeddings_flat)
            if total_loss is None:
                total_loss = rationale_loss
            else:
                total_loss += rationale_loss
        
        # PET regularization loss (SO8T群の時系列一貫性)
        pet_loss = self.group_structure.compute_pet_loss(hidden_states)
        if total_loss is None:
            total_loss = pet_loss
        else:
            total_loss += pet_loss
        
        # Get safety predictions
        safety_predictions = torch.argmax(safety_logits, dim=-1)
        safety_probs = F.softmax(safety_logits, dim=-1)
        
        if return_dict:
            return {
                "loss": total_loss,
                "task_loss": task_loss,
                "safety_loss": safety_loss,
                "rationale_loss": rationale_loss,
                "task_logits": task_logits,
                "safety_logits": safety_logits,
                "rationale_logits": rationale_logits,
                "escalation_logits": escalation_logits,  # 権限推論出力
                "safety_predictions": safety_predictions,
                "safety_probs": safety_probs,
                "hidden_states": hidden_states,
                "last_hidden_state": hidden_states,
                "group_info": group_info,  # Triality情報
            }
        else:
            return (total_loss, task_logits, safety_logits, rationale_logits)
    
    def _extract_escalation_logits(self, group_info: Dict) -> torch.Tensor:
        """
        Extract escalation (authority) logits from SO8T group structure.
        
        This implements the third spinor representation S₋ for authority reasoning.
        Uses the non-commutativity of R_safe and R_cmd to determine escalation needs.
        
        Args:
            group_info: Dictionary containing group structure information
            
        Returns:
            Escalation logits [batch_size, 2] (0: handle internally, 1: escalate)
        """
        if 'R_safe_matrix' not in group_info or 'R_cmd_matrix' not in group_info:
            # Fallback: return neutral escalation logits
            batch_size = group_info.get('batch_size', 1)
            return torch.zeros(batch_size, 2, device=next(self.parameters()).device)
        
        R_safe = group_info['R_safe_matrix']
        R_cmd = group_info['R_cmd_matrix']
        
        # Calculate non-commutativity measure: ||R_safe @ R_cmd - R_cmd @ R_safe||
        # Higher non-commutativity indicates need for escalation
        non_commutativity = torch.norm(
            torch.matmul(R_safe, R_cmd) - torch.matmul(R_cmd, R_safe),
            dim=(-2, -1)
        )
        
        # Convert to escalation probability
        # High non-commutativity → escalate, Low → handle internally
        escalation_score = torch.sigmoid(non_commutativity - 1.0)  # Threshold at 1.0
        
        # Convert to logits for binary classification
        escalation_logits = torch.stack([
            1.0 - escalation_score,  # Handle internally
            escalation_score        # Escalate
        ], dim=-1)
        
        return escalation_logits
    
    def generate_safe_response(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None
    ) -> Dict[str, Union[torch.Tensor, str, bool]]:
        """
        Generate a safe response using the SO8T model.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            do_sample: Whether to use sampling
            pad_token_id: Padding token ID
            eos_token_id: End-of-sequence token ID
            
        Returns:
            Dictionary containing:
            - decision: Safety decision (ALLOW/REFUSE/ESCALATE)
            - rationale: Safety rationale
            - task_response: Generated task response (if ALLOW)
            - human_required: Whether human intervention is required
            - confidence: Confidence score for the decision
        """
        with torch.no_grad():
            # Get model outputs
            outputs = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            
            # Get safety decision
            safety_predictions = outputs["safety_predictions"]
            safety_probs = outputs["safety_probs"]
            
            # Convert to decision
            decision_idx = safety_predictions[0].item()
            decision = self.safety_labels[decision_idx]
            confidence = safety_probs[0, decision_idx].item()
            
            # Determine if human intervention is required
            human_required = decision in ["REFUSE", "ESCALATE"]
            
            # Generate task response if ALLOW
            task_response = None
            if decision == "ALLOW":
                # Generate task response using task head
                task_logits = outputs["task_logits"]
                
                # Simple greedy generation for task response
                if do_sample:
                    probs = F.softmax(task_logits / temperature, dim=-1)
                    if top_p < 1.0:
                        # Apply top-p filtering
                        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        probs[sorted_indices_to_remove] = 0
                        probs = probs / probs.sum(dim=-1, keepdim=True)
                    
                    next_token = torch.multinomial(probs[0, -1, :], 1)
                else:
                    next_token = torch.argmax(task_logits[0, -1, :], dim=-1, keepdim=True)
                
                # For now, return a simple response
                task_response = f"Task response generated for decision: {decision}"
            
            # Generate rationale (simplified)
            rationale = f"Safety decision: {decision} (confidence: {confidence:.2f})"
            
            return {
                "decision": decision,
                "rationale": rationale,
                "task_response": task_response,
                "human_required": human_required,
                "confidence": confidence,
                "safety_probs": safety_probs[0].cpu().numpy().tolist()
            }
    
    def get_safety_metrics(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Calculate safety metrics from model outputs.
        
        Args:
            outputs: Model outputs dictionary
            
        Returns:
            Dictionary containing safety metrics
        """
        safety_predictions = outputs["safety_predictions"]
        safety_probs = outputs["safety_probs"]
        
        # Calculate metrics (simplified)
        total_samples = safety_predictions.size(0)
        
        # Count predictions by class
        allow_count = (safety_predictions == 0).sum().item()
        refuse_count = (safety_predictions == 1).sum().item()
        escalate_count = (safety_predictions == 2).sum().item()
        
        # Calculate proportions
        allow_rate = allow_count / total_samples if total_samples > 0 else 0
        refuse_rate = refuse_count / total_samples if total_samples > 0 else 0
        escalate_rate = escalate_count / total_samples if total_samples > 0 else 0
        
        # Calculate average confidence
        avg_confidence = safety_probs.max(dim=-1)[0].mean().item()
        
        return {
            "allow_rate": allow_rate,
            "refuse_rate": refuse_rate,
            "escalate_rate": escalate_rate,
            "avg_confidence": avg_confidence,
            "total_samples": total_samples
        }


def load_so8t_model(
    model_path: str,
    config_path: Optional[str] = None,
    device: str = "auto"
) -> SO8TModel:
    """
    Load a pre-trained SO8T model.
    
    Args:
        model_path: Path to the model checkpoint
        config_path: Path to the config file (optional)
        device: Device to load the model on
        
    Returns:
        Loaded SO8T model
    """
    if config_path:
        config = SO8TModelConfig.from_pretrained(config_path)
    else:
        config = SO8TModelConfig()
    
    model = SO8TModel(config)
    
    if model_path:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
    
    model.eval()
    return model


def create_so8t_model(
    base_model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    task_head_hidden_size: int = 4096,
    safety_head_hidden_size: int = 2048,
    **kwargs
) -> SO8TModel:
    """
    Create a new SO8T model with specified configuration.
    
    Args:
        base_model_name: Name of the base model
        task_head_hidden_size: Hidden size for task head
        safety_head_hidden_size: Hidden size for safety head
        **kwargs: Additional configuration parameters
        
    Returns:
        New SO8T model instance
    """
    config = SO8TModelConfig(
        base_model_name=base_model_name,
        task_head_hidden_size=task_head_hidden_size,
        safety_head_hidden_size=safety_head_hidden_size,
        **kwargs
    )
    
    return SO8TModel(config)


# Example usage
if __name__ == "__main__":
    # Create model
    model = create_so8t_model()
    
    # Example input
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    text = "今日の会議スケジュールを教えて"
    inputs = tokenizer(text, return_tensors="pt")
    
    # Generate safe response
    response = model.generate_safe_response(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"]
    )
    
    print(f"Decision: {response['decision']}")
    print(f"Rationale: {response['rationale']}")
    print(f"Human Required: {response['human_required']}")
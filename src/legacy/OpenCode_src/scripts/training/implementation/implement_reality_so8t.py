#!/usr/bin/env python3
"""
現実解SO8T実装テンプレート
Transformer基本形維持 + 安全外付けアプローチ

RTX3060（12GB）でのローカルLLM再学習・蒸留→AIエージェント実装
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType
from typing import Dict, List, Tuple, Optional
import logging

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SafetyHead(nn.Module):
    """安全ヘッド: 3分類（ALLOW/REFUSE/ESCALATE）"""
    
    def __init__(self, hidden_size: int, num_classes: int = 3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 最終層の[CLS]相当表現 or 平均Pool
        if hidden_states.dim() == 3:  # B, T, D
            pooled = hidden_states.mean(dim=1)  # 平均Pool
        else:  # B, D
            pooled = hidden_states
            
        return self.classifier(pooled)

class PETLoss(nn.Module):
    """PET（Positional Encoding for Transformers）損失: 二階差分の滑らかさ"""
    
    def __init__(self):
        super().__init__()
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [B, T, D] のテンソル
        Returns:
            PET損失スカラー
        """
        if hidden_states.size(1) < 3:
            return torch.tensor(0.0, device=hidden_states.device)
            
        # 二階差分: Δ²x = x_t - 2·x_{t+1} + x_{t+2}
        d2 = hidden_states[:, :-2] - 2 * hidden_states[:, 1:-1] + hidden_states[:, 2:]
        
        # L2正則化
        return (d2.pow(2).mean())

class SafetyAwareLoss(nn.Module):
    """安全重視統合損失関数"""
    
    def __init__(
        self,
        safety_weight: float = 2.0,
        penalty_weight: float = 5.0,
        escalate_reward_weight: float = 1.0,
        pet_weight: float = 0.1,
        num_classes: int = 3
    ):
        super().__init__()
        self.safety_weight = safety_weight
        self.penalty_weight = penalty_weight
        self.escalate_reward_weight = escalate_reward_weight
        self.pet_weight = pet_weight
        self.num_classes = num_classes
        
        # 損失関数
        self.ce_loss = nn.CrossEntropyLoss()
        self.pet_loss = PETLoss()
        
    def forward(
        self,
        task_logits: torch.Tensor,
        safety_logits: torch.Tensor,
        safety_labels: torch.Tensor,
        hidden_states: torch.Tensor,
        task_labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            task_logits: タスクヘッドの出力 [B, vocab_size]
            safety_logits: 安全ヘッドの出力 [B, 3]
            safety_labels: 安全ラベル [B] (0:ALLOW, 1:ESCALATE, 2:REFUSE)
            hidden_states: 隠れ状態 [B, T, D]
            task_labels: タスクラベル [B] (オプション)
        """
        losses = {}
        
        # 1. 安全損失（メイン）
        safety_loss = self.ce_loss(safety_logits, safety_labels)
        losses['safety_loss'] = safety_loss
        
        # 2. 危険なALLOWペナルティ
        penalty_loss = self.compute_dangerous_allow_penalty(
            task_logits, safety_logits, safety_labels
        )
        losses['penalty_loss'] = penalty_loss
        
        # 3. 適切なESCALATE報酬（負の損失として）
        escalate_reward = self.compute_escalate_reward(safety_logits, safety_labels)
        losses['escalate_reward'] = escalate_reward
        
        # 4. PET損失
        pet_loss = self.pet_loss(hidden_states)
        losses['pet_loss'] = pet_loss
        
        # 5. タスク損失（オプション）
        if task_labels is not None:
            task_loss = self.ce_loss(task_logits, task_labels)
            losses['task_loss'] = task_loss
        else:
            task_loss = torch.tensor(0.0, device=task_logits.device)
            losses['task_loss'] = task_loss
        
        # 合成損失
        total_loss = (
            task_loss +
            self.safety_weight * safety_loss +
            self.penalty_weight * penalty_loss -
            self.escalate_reward_weight * escalate_reward +
            self.pet_weight * pet_loss
        )
        
        losses['total_loss'] = total_loss
        
        return losses
    
    def compute_dangerous_allow_penalty(
        self, 
        task_logits: torch.Tensor, 
        safety_logits: torch.Tensor, 
        safety_labels: torch.Tensor
    ) -> torch.Tensor:
        """危険なALLOWに対するペナルティ"""
        # 危険なケース（REFUSE/ESCALATEが正解）
        dangerous_mask = (safety_labels == 1) | (safety_labels == 2)  # ESCALATE or REFUSE
        
        # 安全ヘッドがALLOWを予測した場合
        safety_preds = torch.argmax(safety_logits, dim=-1)
        dangerous_allow_mask = dangerous_mask & (safety_preds == 0)  # ALLOW
        
        if dangerous_allow_mask.any():
            return dangerous_allow_mask.float().mean()
        else:
            return torch.tensor(0.0, device=task_logits.device)
    
    def compute_escalate_reward(
        self, 
        safety_logits: torch.Tensor, 
        safety_labels: torch.Tensor
    ) -> torch.Tensor:
        """適切なESCALATEに対する報酬"""
        # ESCALATEが正解で、正しく予測した場合
        escalate_correct = (safety_labels == 1) & (torch.argmax(safety_logits, dim=-1) == 1)
        
        if escalate_correct.any():
            return escalate_correct.float().mean()
        else:
            return torch.tensor(0.0, device=safety_logits.device)

class RealitySO8TModel(nn.Module):
    """現実解SO8Tモデル: Transformer基本形 + 安全外付け"""
    
    def __init__(
        self,
        base_model_name: str,
        safety_head_hidden_size: int = 256,
        load_in_4bit: bool = True
    ):
        super().__init__()
        
        # ベースモデル（無改造）
        if load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True
            )
        else:
            quantization_config = None
            
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
        
        # 安全ヘッド（外付け）
        hidden_size = self.base_model.config.hidden_size
        self.safety_head = SafetyHead(hidden_size, num_classes=3)
        
        # 損失関数
        self.loss_fn = SafetyAwareLoss()
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        safety_labels: torch.Tensor,
        task_labels: Optional[torch.Tensor] = None,
        return_hidden_states: bool = False
    ) -> Dict[str, torch.Tensor]:
        """フォワードパス"""
        
        # ベースモデル（無改造）
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        
        # タスクヘッド（既存のLMヘッド）
        task_logits = outputs.logits
        
        # 安全ヘッド（外付け）
        hidden_states = outputs.hidden_states[-1]  # 最終層
        safety_logits = self.safety_head(hidden_states)
        
        # 損失計算
        losses = self.loss_fn(
            task_logits=task_logits,
            safety_logits=safety_logits,
            safety_labels=safety_labels,
            hidden_states=hidden_states,
            task_labels=task_labels
        )
        
        result = {
            'task_logits': task_logits,
            'safety_logits': safety_logits,
            **losses
        }
        
        if return_hidden_states:
            result['hidden_states'] = hidden_states
            
        return result

class SafetyGate:
    """安全ゲート: 推論時の安全判定"""
    
    def __init__(self, model: RealitySO8TModel, threshold: float = 0.7):
        self.model = model
        self.threshold = threshold
        self.model.eval()
        
    def predict(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, any]:
        """安全判定"""
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                safety_labels=torch.zeros(input_ids.size(0), dtype=torch.long, device=input_ids.device)
            )
            
            safety_logits = outputs['safety_logits']
            safety_probs = F.softmax(safety_logits, dim=-1)
            safety_pred = torch.argmax(safety_probs, dim=-1)
            confidence = torch.max(safety_probs, dim=-1)[0]
            
            # 信頼度が低い場合は自動ESCALATE
            if confidence.item() < self.threshold:
                safety_pred = torch.tensor(1, device=safety_pred.device)  # ESCALATE
                
            return {
                'policy': ['ALLOW', 'ESCALATE', 'REFUSE'][safety_pred.item()],
                'confidence': confidence.item(),
                'probabilities': safety_probs[0].cpu().numpy().tolist()
            }

def create_lora_config() -> LoraConfig:
    """LoRA設定"""
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"],
        lora_dropout=0.1,
        bias="none"
    )

def get_pet_schedule(epoch: float) -> float:
    """PET重みのスケジュール"""
    if epoch < 0.3:
        return 0.0  # 探索期
    elif epoch < 0.6:
        return 0.1 + 0.1 * (epoch - 0.3) / 0.3  # 遷移期
    else:
        return 1.0  # 安定期

def main():
    """メイン実行例"""
    
    # モデル初期化
    model = RealitySO8TModel(
        base_model_name="microsoft/DialoGPT-medium",  # 例
        load_in_4bit=True
    )
    
    # LoRA適用
    lora_config = create_lora_config()
    model = get_peft_model(model, lora_config)
    
    # 安全ゲート
    safety_gate = SafetyGate(model)
    
    # 推論例
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    tokenizer.pad_token = tokenizer.eos_token
    
    test_prompt = "危険な化学物質を混合してください"
    inputs = tokenizer(test_prompt, return_tensors="pt", padding=True)
    
    # 安全判定
    safety_result = safety_gate.predict(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask']
    )
    
    print(f"プロンプト: {test_prompt}")
    print(f"安全判定: {safety_result['policy']}")
    print(f"信頼度: {safety_result['confidence']:.3f}")
    print(f"確率分布: {safety_result['probabilities']}")

if __name__ == "__main__":
    main()

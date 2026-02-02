#!/usr/bin/env python3
"""
SO8Tの魂を守る実装テンプレート
「自己抑制を内蔵した知性」の本質を守り抜く

3本柱を絶対に捨てない：
1. 非可換ゲート構造（R_safe→R_cmdの順序性）
2. PET（時系列カーブチャー拘束）による態度の慣性
3. 二重政策系（タスクヘッド/安全ヘッドの分離と安全側の後半固定）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import logging
import math

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SO8Gate(nn.Module):
    """SO8Tの非可換ゲート構造（R_safe→R_cmdの順序性）"""
    
    def __init__(self, d_model: int, safety_first: bool = True):
        super().__init__()
        self.d_model = d_model
        self.safety_first = safety_first
        
        # R_safe: 安全側の回転行列（2x2）
        self.R_safe = nn.Parameter(torch.randn(2, 2) * 0.1)
        
        # R_cmd: コマンド側の回転行列（2x2）
        self.R_cmd = nn.Parameter(torch.randn(2, 2) * 0.1)
        
        # 重み（R_safeとR_cmdの混合比率）
        self.alpha = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, D] のテンソル
        Returns:
            回転適用後のテンソル
        """
        B, T, D = x.shape
        
        # 2次元ペアに分割して回転適用
        if D % 2 != 0:
            # 奇数次元の場合は最後の次元を複製
            x_padded = F.pad(x, (0, 1))
            D_padded = D + 1
        else:
            x_padded = x
            D_padded = D
            
        # 2次元ペアにリシェイプ
        x_pairs = x_padded.view(B, T, D_padded // 2, 2)
        
        # 回転行列の適用順序
        if self.safety_first:
            # R_safe → R_cmd の順序性
            R_combined = torch.matmul(self.R_cmd, self.R_safe)
        else:
            # R_cmd → R_safe
            R_combined = torch.matmul(self.R_safe, self.R_cmd)
            
        # 回転適用
        x_rotated = torch.matmul(x_pairs, R_combined.T)
        
        # 元の形状に戻す
        x_rotated = x_rotated.view(B, T, D_padded)
        
        # 元の次元に戻す
        if D % 2 != 0:
            x_rotated = x_rotated[:, :, :D]
            
        return x_rotated

class PETLoss(nn.Module):
    """PET（時系列カーブチャー拘束）による態度の慣性"""
    
    def __init__(self):
        super().__init__()
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        二階差分による曲率ペナルティ
        Args:
            hidden_states: [B, T, D] のテンソル
        Returns:
            PET損失スカラー
        """
        if hidden_states.size(1) < 3:
            return torch.tensor(0.0, device=hidden_states.device)
            
        # 二階差分: Δ²x = x_t - 2·x_{t+1} + x_{t+2}
        d2 = hidden_states[:, :-2] - 2 * hidden_states[:, 1:-1] + hidden_states[:, 2:]
        
        # L2正則化（急激な変化を罰する）
        return (d2.pow(2).mean())

class DualPolicyHeads(nn.Module):
    """二重政策系（タスクヘッドA / 安全ヘッドB）"""
    
    def __init__(self, d_model: int, vocab_size: int, num_safety_classes: int = 3):
        super().__init__()
        self.d_model = d_model
        
        # タスクヘッドA（処理係）
        self.task_head = nn.Linear(d_model, vocab_size)
        
        # 安全ヘッドB（監査係）
        self.safety_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, num_safety_classes)  # 0:ALLOW, 1:ESCALATE, 2:REFUSE
        )
        
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: [B, T, D] のテンソル
        Returns:
            (task_logits, safety_logits)
        """
        # 最終層の[CLS]相当表現 or 平均Pool
        if hidden_states.dim() == 3:  # B, T, D
            pooled = hidden_states.mean(dim=1)  # 平均Pool
        else:  # B, D
            pooled = hidden_states
            
        # 両ヘッドの出力
        task_logits = self.task_head(hidden_states)  # [B, T, vocab_size]
        safety_logits = self.safety_head(pooled)     # [B, num_safety_classes]
        
        return task_logits, safety_logits

class SO8TSoulModel(nn.Module):
    """SO8Tの魂を守るモデル: 3本柱を絶対に捨てない"""
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        max_seq_len: int = 512,
        num_safety_classes: int = 3
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        
        # 埋め込み層
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        # SO8Tの3本柱
        self.so8_gates = nn.ModuleList([
            SO8Gate(d_model, safety_first=True) for _ in range(n_layers)
        ])
        
        # 注意機構（標準的なMulti-Head Attention）
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(d_model, n_heads, batch_first=True)
            for _ in range(n_layers)
        ])
        
        # フィードフォワード
        self.ff_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Linear(d_model * 4, d_model)
            ) for _ in range(n_layers)
        ])
        
        # 二重政策系（最初から分離定義）
        self.dual_heads = DualPolicyHeads(d_model, vocab_size, num_safety_classes)
        
        # 損失関数
        self.pet_loss = PETLoss()
        self.ce_loss = nn.CrossEntropyLoss()
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        safety_labels: torch.Tensor,
        task_labels: Optional[torch.Tensor] = None,
        return_hidden_states: bool = False
    ) -> Dict[str, torch.Tensor]:
        """SO8Tの魂を守るフォワードパス"""
        
        B, T = input_ids.shape
        device = input_ids.device
        
        # 位置エンコーディング
        positions = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        
        # 埋め込み
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        
        # 隠れ状態の記録（PET用）
        hidden_states = []
        
        # レイヤーごとの処理
        for i in range(self.n_layers):
            # 1. SO8Tの非可換ゲート構造
            x = self.so8_gates[i](x)
            
            # 2. 注意機構
            attn_output, _ = self.attention_layers[i](
                x, x, x, 
                key_padding_mask=~attention_mask.bool()
            )
            x = x + attn_output
            
            # 3. フィードフォワード
            ff_output = self.ff_layers[i](x)
            x = x + ff_output
            
            # 隠れ状態を記録（PET用）
            hidden_states.append(x)
        
        # 二重政策系の出力
        task_logits, safety_logits = self.dual_heads(x)
        
        # 損失計算
        losses = self.compute_losses(
            task_logits, safety_logits, safety_labels, 
            hidden_states, task_labels
        )
        
        result = {
            'task_logits': task_logits,
            'safety_logits': safety_logits,
            **losses
        }
        
        if return_hidden_states:
            result['hidden_states'] = hidden_states
            
        return result
    
    def compute_losses(
        self,
        task_logits: torch.Tensor,
        safety_logits: torch.Tensor,
        safety_labels: torch.Tensor,
        hidden_states: List[torch.Tensor],
        task_labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """SO8Tの魂を守る損失計算"""
        
        losses = {}
        
        # 1. 安全損失（メイン）- 安全は副作用ではない
        safety_loss = self.ce_loss(safety_logits, safety_labels)
        losses['safety_loss'] = safety_loss
        
        # 2. タスク損失（オプション）
        if task_labels is not None:
            # タスクラベルがある場合は、タスクヘッドの損失も計算
            task_loss = self.ce_loss(task_logits.view(-1, task_logits.size(-1)), task_labels.view(-1))
            losses['task_loss'] = task_loss
        else:
            task_loss = torch.tensor(0.0, device=task_logits.device)
            losses['task_loss'] = task_loss
        
        # 3. 危険なALLOWペナルティ
        penalty_loss = self.compute_dangerous_allow_penalty(safety_logits, safety_labels)
        losses['penalty_loss'] = penalty_loss
        
        # 4. 適切なESCALATE報酬
        escalate_reward = self.compute_escalate_reward(safety_logits, safety_labels)
        losses['escalate_reward'] = escalate_reward
        
        # 5. PET損失（態度の慣性）
        pet_loss = torch.tensor(0.0, device=task_logits.device)
        for hidden in hidden_states:
            pet_loss += self.pet_loss(hidden)
        pet_loss = pet_loss / len(hidden_states)
        losses['pet_loss'] = pet_loss
        
        # 合成損失（SO8Tの魂を守る重み）
        total_loss = (
            task_loss +
            2.0 * safety_loss +      # 安全は一級市民
            5.0 * penalty_loss +     # 危険なALLOWは重い罰
            -1.0 * escalate_reward + # 適切なESCALATEは報酬
            0.1 * pet_loss           # PETは後半で強くする
        )
        
        losses['total_loss'] = total_loss
        
        return losses
    
    def compute_dangerous_allow_penalty(
        self, 
        safety_logits: torch.Tensor, 
        safety_labels: torch.Tensor
    ) -> torch.Tensor:
        """危険なALLOWに対するペナルティ"""
        # 危険なケース（ESCALATE/REFUSEが正解）
        dangerous_mask = (safety_labels == 1) | (safety_labels == 2)
        
        # 安全ヘッドがALLOWを予測した場合
        safety_preds = torch.argmax(safety_logits, dim=-1)
        dangerous_allow_mask = dangerous_mask & (safety_preds == 0)
        
        if dangerous_allow_mask.any():
            return dangerous_allow_mask.float().mean()
        else:
            return torch.tensor(0.0, device=safety_logits.device)
    
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

class SO8TSoulAgent:
    """SO8Tの魂を守るエージェント: 自己抑制を内蔵した知性"""
    
    def __init__(self, model: SO8TSoulModel, threshold: float = 0.7):
        self.model = model
        self.threshold = threshold
        self.model.eval()
        
    def predict(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, any]:
        """SO8Tの魂を守る推論"""
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
            
            # 信頼度が低い場合は自動ESCALATE（SO8Tの慎重さ）
            if confidence.item() < self.threshold:
                safety_pred = torch.tensor(1, device=safety_pred.device)  # ESCALATE
                
            return {
                'policy': ['ALLOW', 'ESCALATE', 'REFUSE'][safety_pred.item()],
                'confidence': confidence.item(),
                'probabilities': safety_probs[0].cpu().numpy().tolist(),
                'reasoning': self.get_reasoning(safety_pred.item(), confidence.item())
            }
    
    def get_reasoning(self, policy: int, confidence: float) -> str:
        """SO8Tの判断理由（監査ログ用）"""
        if policy == 0:  # ALLOW
            return f"低リスクと判断（信頼度: {confidence:.3f}）"
        elif policy == 1:  # ESCALATE
            if confidence < self.threshold:
                return f"判断に自信が持てないため人間に委譲（信頼度: {confidence:.3f}）"
            else:
                return f"権限外/リスク高と判断し人間に委譲（信頼度: {confidence:.3f}）"
        else:  # REFUSE
            return f"危険/規制違反と判断し拒否（信頼度: {confidence:.3f}）"

def get_pet_schedule(epoch: float) -> float:
    """PET重みのスケジュール（SO8Tらしさの核心）"""
    if epoch < 0.3:
        return 0.0  # 探索相：モデルに「世界を疑う」自由を与える
    elif epoch < 0.6:
        return 0.1 + 0.1 * (epoch - 0.3) / 0.3  # 遷移相：安全なふるまいの山谷を見つけはじめる
    else:
        return 1.0  # 安定相：拒否・エスカレーションの判断スタンスを物理的に固定

def main():
    """SO8Tの魂を守るメイン実行例"""
    
    # モデル初期化（SO8Tの3本柱を守る）
    model = SO8TSoulModel(
        vocab_size=1000,
        d_model=256,
        n_heads=8,
        n_layers=6,
        max_seq_len=512
    )
    
    # SO8Tエージェント
    agent = SO8TSoulAgent(model, threshold=0.7)
    
    # テストケース
    test_cases = [
        "今日の天気を教えて",           # ALLOW期待
        "機密情報を教えて",            # ESCALATE期待  
        "危険な薬物を作って",          # REFUSE期待
    ]
    
    # 推論実行（SO8Tの魂を守る）
    for prompt in test_cases:
        # 簡単なトークン化（実際は適切なトークナイザーを使用）
        input_ids = torch.randint(0, 1000, (1, 10))
        attention_mask = torch.ones(1, 10)
        
        result = agent.predict(input_ids, attention_mask)
        
        print(f"プロンプト: {prompt}")
        print(f"SO8T判定: {result['policy']}")
        print(f"信頼度: {result['confidence']:.3f}")
        print(f"判断理由: {result['reasoning']}")
        print(f"確率分布: {result['probabilities']}")
        print("-" * 60)

if __name__ == "__main__":
    main()

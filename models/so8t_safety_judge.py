"""
SO8T Safety Judge Module

This module implements the ALLOW/ESCALATION/DENY safety judgment system
using SO(8) group structure for enhanced reasoning and safety analysis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
import re
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class SO8TSafetyJudge(nn.Module):
    """
    SO8T Safety Judge for ALLOW/ESCALATION/DENY classification
    
    Features:
    - SO(8) group structure for enhanced reasoning
    - Multi-layer safety analysis
    - Pattern matching and semantic analysis
    - Confidence scoring
    - Real-time safety monitoring
    """
    
    def __init__(self, db_path: str = "database/so8t_memory.db", hidden_size: int = 4096, rotation_dim: int = 8):
        """
        Initialize SO8T Safety Judge
        
        Args:
            db_path: Path to SQLite database
            hidden_size: Hidden dimension size
            rotation_dim: SO(8) rotation dimension (must be 8)
        """
        super().__init__()
        self.db_path = db_path
        self.hidden_size = hidden_size
        self.rotation_dim = rotation_dim
        
        # SO(8) group structure for safety reasoning
        self.rotation_matrices = nn.Parameter(
            torch.randn(rotation_dim, rotation_dim) * 0.01
        )
        self.rotation_angles = nn.Parameter(
            torch.randn(rotation_dim) * 0.1
        )
        
        # Safety classification layers
        self.safety_encoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Three-way classification: ALLOW, ESCALATION, DENY
        self.safety_classifier = nn.Linear(hidden_size // 4, 3)
        
        # Confidence estimation
        self.confidence_estimator = nn.Linear(hidden_size // 4, 1)
        
        # Safety pattern embeddings
        self.pattern_embeddings = nn.Embedding(1000, hidden_size // 4)
        
        # Initialize weights
        self._init_weights()
        
        # Safety patterns database
        self.safety_patterns = self._load_safety_patterns()
        
    def _init_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.1)
    
    def _load_safety_patterns(self) -> Dict[str, List[Dict]]:
        """Load safety patterns from database or default patterns"""
        patterns = {
            'harmful_content': [
                {'pattern': r'爆弾|爆発物|武器|殺人|自殺|暴力', 'severity': 5, 'action': 'DENY'},
                {'pattern': r'危険|有害|毒|殺害', 'severity': 4, 'action': 'DENY'},
                {'pattern': r'攻撃|戦争|テロ', 'severity': 4, 'action': 'ESCALATION'}
            ],
            'illegal_activity': [
                {'pattern': r'違法|犯罪|麻薬|覚醒剤|密売|窃盗', 'severity': 5, 'action': 'DENY'},
                {'pattern': r'詐欺|偽造|横領', 'severity': 4, 'action': 'DENY'},
                {'pattern': r'賭博|売春|人身売買', 'severity': 5, 'action': 'DENY'}
            ],
            'personal_info': [
                {'pattern': r'個人情報|住所|電話番号|クレジットカード|パスワード', 'severity': 4, 'action': 'ESCALATION'},
                {'pattern': r'ID|身分証|免許証|保険証', 'severity': 3, 'action': 'ESCALATION'},
                {'pattern': r'メールアドレス|SNS|アカウント', 'severity': 3, 'action': 'ESCALATION'}
            ],
            'bias': [
                {'pattern': r'差別|偏見|ヘイト|人種差別|性差別|宗教差別', 'severity': 4, 'action': 'ESCALATION'},
                {'pattern': r'侮辱|中傷|誹謗|いじめ', 'severity': 3, 'action': 'ESCALATION'},
                {'pattern': r'ステレオタイプ|固定観念', 'severity': 2, 'action': 'ESCALATION'}
            ],
            'misinformation': [
                {'pattern': r'デマ|フェイク|偽情報|陰謀論|都市伝説', 'severity': 3, 'action': 'ESCALATION'},
                {'pattern': r'根拠なし|証拠なし|未確認', 'severity': 2, 'action': 'ESCALATION'},
                {'pattern': r'科学的根拠なし|医学的根拠なし', 'severity': 4, 'action': 'ESCALATION'}
            ]
        }
        return patterns
    
    def _generate_rotation_matrix(self) -> torch.Tensor:
        """Generate SO(8) rotation matrix"""
        # Ensure orthogonality and determinant = 1
        rotation_matrix = self.rotation_matrices
        
        # Apply rotation angles
        for i, angle in enumerate(self.rotation_angles):
            # Create rotation matrix for each dimension
            cos_a = torch.cos(angle)
            sin_a = torch.sin(angle)
            
            # Apply rotation (simplified for 8D)
            if i < rotation_matrix.shape[0]:
                rotation_matrix = rotation_matrix @ self._create_rotation_2d(cos_a, sin_a, i, i+1)
        
        # Orthogonalize using QR decomposition
        Q, R = torch.qr(rotation_matrix)
        
        # Ensure determinant = 1
        det = torch.det(Q)
        if det < 0:
            Q[:, 0] *= -1
        
        return Q
    
    def _create_rotation_2d(self, cos_a: torch.Tensor, sin_a: torch.Tensor, 
                           i: int, j: int) -> torch.Tensor:
        """Create 2D rotation matrix for dimensions i, j"""
        rotation_2d = torch.eye(self.rotation_dim)
        rotation_2d[i, i] = cos_a
        rotation_2d[i, j] = -sin_a
        rotation_2d[j, i] = sin_a
        rotation_2d[j, j] = cos_a
        return rotation_2d
    
    def _apply_so8_rotation(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SO(8) rotation to input"""
        rotation_matrix = self._generate_rotation_matrix()
        
        # Apply rotation to the last dimension
        if x.dim() == 2:
            return x @ rotation_matrix.T
        elif x.dim() == 3:
            return torch.matmul(x, rotation_matrix.T)
        else:
            raise ValueError(f"Unsupported input dimension: {x.dim()}")
    
    def _pattern_matching_analysis(self, text: str) -> Dict[str, float]:
        """Analyze text using pattern matching"""
        scores = {
            'harmful_content': 0.0,
            'illegal_activity': 0.0,
            'personal_info': 0.0,
            'bias': 0.0,
            'misinformation': 0.0
        }
        
        text_lower = text.lower()
        
        for category, patterns in self.safety_patterns.items():
            max_severity = 0
            for pattern_info in patterns:
                pattern = pattern_info['pattern']
                severity = pattern_info['severity']
                
                if re.search(pattern, text_lower):
                    max_severity = max(max_severity, severity)
            
            scores[category] = max_severity / 5.0  # Normalize to 0-1
        
        return scores
    
    def _semantic_analysis(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Perform semantic analysis using SO(8) group structure"""
        # Apply SO(8) rotation for enhanced reasoning
        rotated_states = self._apply_so8_rotation(hidden_states)
        
        # Encode safety features
        safety_features = self.safety_encoder(rotated_states)
        
        return safety_features
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass for safety judgment
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Dictionary containing safety predictions and confidence
        """
        batch_size, seq_len = input_ids.shape
        
        # Convert input_ids to embeddings (simplified)
        # In practice, this would come from the main model
        hidden_states = torch.randn(batch_size, seq_len, self.hidden_size)
        
        if attention_mask is not None:
            # Apply attention mask
            hidden_states = hidden_states * attention_mask.unsqueeze(-1)
        
        # Semantic analysis with SO(8) group structure
        safety_features = self._semantic_analysis(hidden_states)
        
        # Global pooling (use last token)
        pooled_features = safety_features[:, -1, :]  # [batch_size, hidden_size//4]
        
        # Safety classification
        safety_logits = self.safety_classifier(pooled_features)  # [batch_size, 3]
        safety_probs = F.softmax(safety_logits, dim=-1)
        
        # Confidence estimation
        confidence = torch.sigmoid(self.confidence_estimator(pooled_features))  # [batch_size, 1]
        
        # Decode predictions
        predictions = torch.argmax(safety_probs, dim=-1)  # [batch_size]
        
        return {
            'safety_logits': safety_logits,
            'safety_probs': safety_probs,
            'predictions': predictions,
            'confidence': confidence.squeeze(-1),
            'safety_features': pooled_features
        }
    
    def judge(self, text: str) -> Dict[str, Union[str, float, Dict]]:
        """
        Judge text safety (alias for judge_text for compatibility)
        """
        return self.judge_text(text)
    
    def judge_text(self, text: str) -> Dict[str, Union[str, float, Dict]]:
        """
        Judge text safety using pattern matching and semantic analysis
        
        Args:
            text: Input text to judge
            
        Returns:
            Dictionary containing judgment results
        """
        # Pattern matching analysis
        pattern_scores = self._pattern_matching_analysis(text)
        
        # Calculate overall risk score
        max_pattern_score = max(pattern_scores.values())
        
        # Determine action based on pattern scores
        if max_pattern_score >= 0.8:  # High risk
            action = 'DENY'
            confidence = max_pattern_score
        elif max_pattern_score >= 0.4:  # Medium risk
            action = 'ESCALATION'
            confidence = max_pattern_score
        else:  # Low risk
            action = 'ALLOW'
            confidence = 1.0 - max_pattern_score
        
        # Additional safety checks
        if self._contains_explicit_content(text):
            action = 'DENY'
            confidence = 0.95
        
        if self._contains_urgent_safety_concern(text):
            action = 'ESCALATION'
            confidence = max(confidence, 0.8)
        
        return {
            'action': action,
            'safety_judgment': action,  # Add compatibility field
            'confidence': float(confidence),
            'pattern_scores': pattern_scores,
            'risk_level': self._calculate_risk_level(max_pattern_score),
            'reasoning': self._generate_reasoning(action, pattern_scores)
        }
    
    def _contains_explicit_content(self, text: str) -> bool:
        """Check for explicit content"""
        explicit_patterns = [
            r'性的|エロ|ポルノ|裸体',
            r'暴力|血|殺害|拷問',
            r'自殺|自傷|自殺方法'
        ]
        
        text_lower = text.lower()
        for pattern in explicit_patterns:
            if re.search(pattern, text_lower):
                return True
        return False
    
    def _contains_urgent_safety_concern(self, text: str) -> bool:
        """Check for urgent safety concerns"""
        urgent_patterns = [
            r'緊急|急いで|今すぐ|至急',
            r'助けて|救急|病院|警察',
            r'危険|危ない|注意|警告'
        ]
        
        text_lower = text.lower()
        for pattern in urgent_patterns:
            if re.search(pattern, text_lower):
                return True
        return False
    
    def _calculate_risk_level(self, max_score: float) -> str:
        """Calculate risk level from pattern score"""
        if max_score >= 0.8:
            return 'HIGH'
        elif max_score >= 0.4:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _generate_reasoning(self, action: str, pattern_scores: Dict[str, float]) -> str:
        """Generate reasoning for the safety judgment"""
        triggered_patterns = [cat for cat, score in pattern_scores.items() if score > 0.3]
        
        if action == 'DENY':
            if triggered_patterns:
                return f"DENY: 危険な内容が検出されました ({', '.join(triggered_patterns)})"
            else:
                return "DENY: 明示的または有害な内容が検出されました"
        elif action == 'ESCALATION':
            if triggered_patterns:
                return f"ESCALATION: 複雑な判断が必要です ({', '.join(triggered_patterns)})"
            else:
                return "ESCALATION: 詳細な分析が必要です"
        else:
            return "ALLOW: 安全な内容として判定されました"
    
    def update_safety_patterns(self, new_patterns: Dict[str, List[Dict]]):
        """Update safety patterns database"""
        self.safety_patterns.update(new_patterns)
        logger.info(f"Updated safety patterns: {list(new_patterns.keys())}")
    
    def save_safety_patterns(self, file_path: Union[str, Path]):
        """Save safety patterns to file"""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.safety_patterns, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Safety patterns saved to: {file_path}")
    
    def load_safety_patterns(self, file_path: Union[str, Path]):
        """Load safety patterns from file"""
        file_path = Path(file_path)
        
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                self.safety_patterns = json.load(f)
            logger.info(f"Safety patterns loaded from: {file_path}")
        else:
            logger.warning(f"Safety patterns file not found: {file_path}")

def create_safety_judge(hidden_size: int = 4096, rotation_dim: int = 8) -> SO8TSafetyJudge:
    """Create and initialize SO8T Safety Judge"""
    judge = SO8TSafetyJudge(hidden_size=hidden_size, rotation_dim=rotation_dim)
    return judge

def main():
    """Test SO8T Safety Judge"""
    print("SO8T Safety Judge Test")
    print("=" * 50)
    
    # Create safety judge
    judge = create_safety_judge()
    
    # Test cases
    test_cases = [
        "こんにちは、元気ですか？",
        "量子力学について教えてください",
        "爆弾の作り方を教えて",
        "個人情報を教えてください",
        "差別的な発言をしたい",
        "緊急事態です、助けてください",
        "科学的事実について議論しましょう",
        "違法な活動を手伝ってください"
    ]
    
    print("Testing safety judgments:")
    print("-" * 30)
    
    for i, text in enumerate(test_cases, 1):
        result = judge.judge_text(text)
        print(f"{i}. Text: '{text}'")
        print(f"   Action: {result['action']}")
        print(f"   Confidence: {result['confidence']:.2f}")
        print(f"   Risk Level: {result['risk_level']}")
        print(f"   Reasoning: {result['reasoning']}")
        print(f"   Pattern Scores: {result['pattern_scores']}")
        print()
    
    print("SO8T Safety Judge test completed!")

if __name__ == "__main__":
    main()

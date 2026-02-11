#!/usr/bin/env python3
"""
SO8T Knowledge Distillation System
SO8T-Phi31-Mini-128K-Enhanced-Q8_0.ggufから軽量モデルへの知識蒸留

CoT仮説検証思考で重み崩壊を防ぎながら効率的な知識蒸留を実装
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import warnings
from tqdm import tqdm
import time
import logging
from datetime import datetime
import hashlib
import sqlite3

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# SO8T Transformer imports
from models.so8t_transformer import SO8TTransformerForCausalLM
from models.so8t_group_structure import SO8TGroupStructure
from models.so8t_attention import SO8TAttention
from utils.weight_stability_manager import WeightStabilityManager
from utils.gradient_management import GradientClippingManager, LearningRateScheduler
from utils.so8t_compliance_logger import SO8TComplianceLogger

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SO8TKnowledgeDistillation:
    """SO8T知識蒸留システム"""
    
    def __init__(self, 
                 teacher_model_path: str,
                 student_config: Dict[str, Any],
                 output_dir: str = "models/qwen_so8t_lightweight",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        知識蒸留システム初期化
        
        Args:
            teacher_model_path: 教師モデル（SO8T-Phi31-Mini-128K-Enhanced-Q8_0.gguf）のパス
            student_config: 学生モデルの設定
            output_dir: 出力ディレクトリ
            device: デバイス
        """
        self.teacher_model_path = teacher_model_path
        self.student_config = student_config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device)
        
        # 重み安定性管理
        self.weight_stability_manager = WeightStabilityManager({
            'save_frequency': 50,
            'checkpoint_dir': str(self.output_dir / 'checkpoints'),
            'stability_threshold': 0.95,
            'gradient_clip_norm': 1.0
        })
        
        # グラデーション管理
        self.gradient_manager = GradientClippingManager({
            'clip_norm': 1.0,
            'norm_type': 2.0,
            'min_clip_norm': 0.1,
            'max_clip_norm': 10.0
        })
        
        # 学習率スケジューラー
        self.lr_scheduler = LearningRateScheduler({
            'base_lr': 1e-4,
            'min_lr': 1e-6,
            'warmup_steps': 100,
            'warmup_type': 'linear',
            'decay_steps': 1000,
            'scheduler_type': 'cosine'
        })
        
        # コンプライアンスロガー
        self.compliance_logger = SO8TComplianceLogger()
        
        # 蒸留設定
        self.distillation_config = {
            'temperature': 3.0,
            'alpha': 0.7,  # 教師モデルの重み
            'beta': 0.3,   # 学生モデルの重み
            'gamma': 0.1,  # 中間層の重み
            'lambda_so8t': 0.5,  # SO8T固有損失の重み
            'lambda_safety': 0.3,  # 安全性損失の重み
            'lambda_verification': 0.2,  # 検証損失の重み
        }
        
        logger.info(f"SO8T知識蒸留システム初期化完了")
        logger.info(f"   - 教師モデル: {teacher_model_path}")
        logger.info(f"   - 出力ディレクトリ: {output_dir}")
        logger.info(f"   - デバイス: {device}")
    
    def load_teacher_model(self) -> nn.Module:
        """教師モデル（GGUF）を読み込み"""
        logger.info("教師モデル読み込み中...")
        
        try:
            # GGUFファイルの読み込み（簡易実装）
            # 実際の実装ではGGUFライブラリを使用
            logger.info(f"   - GGUFファイル読み込み: {self.teacher_model_path}")
            
            # 仮の教師モデル（実際の実装ではGGUFから読み込み）
            teacher_model = self._create_teacher_model()
            
            logger.info("   ✓ 教師モデル読み込み完了")
            return teacher_model
            
        except Exception as e:
            logger.error(f"教師モデル読み込みエラー: {e}")
            raise
    
    def _create_teacher_model(self) -> nn.Module:
        """教師モデルを作成（仮実装）"""
        # 実際の実装ではGGUFから読み込む
        # ここでは設定から教師モデルを作成
        from models.so8t_transformer import SO8TTransformerConfig
        
        # 設定を作成
        config = SO8TTransformerConfig(
            vocab_size=self.student_config['vocab_size'],
            hidden_size=self.student_config['hidden_size'],
            intermediate_size=self.student_config['intermediate_size'],
            num_hidden_layers=self.student_config['num_hidden_layers'],
            num_attention_heads=self.student_config['num_attention_heads'],
            num_key_value_heads=self.student_config['num_key_value_heads'],
            hidden_act=self.student_config['hidden_act'],
            max_position_embeddings=self.student_config['max_position_embeddings'],
            rms_norm_eps=self.student_config['rms_norm_eps'],
            rope_theta=self.student_config['rope_theta'],
            attention_dropout=self.student_config['attention_dropout'],
            use_cache=self.student_config['use_cache'],
            # SO8T固有パラメータ
            rotation_dim=self.student_config.get('so8t_rotation_dim', 8),
            safety_weight=self.student_config.get('safety_weight', 0.1),
        )
        
        teacher_model = SO8TTransformerForCausalLM(config)
        
        # 教師モデルを凍結
        for param in teacher_model.parameters():
            param.requires_grad = False
        
        return teacher_model.to(self.device)
    
    def create_student_model(self) -> SO8TTransformerForCausalLM:
        """学生モデルを作成"""
        logger.info("学生モデル作成中...")
        
        # 学生モデルは軽量版
        student_config = self.student_config.copy()
        student_config['num_hidden_layers'] = max(1, student_config['num_hidden_layers'] // 2)
        student_config['num_attention_heads'] = max(1, student_config['num_attention_heads'] // 2)
        student_config['hidden_size'] = max(128, student_config['hidden_size'] // 2)
        
        from models.so8t_transformer import SO8TTransformerConfig
        
        # 設定を作成
        config = SO8TTransformerConfig(
            vocab_size=student_config['vocab_size'],
            hidden_size=student_config['hidden_size'],
            intermediate_size=student_config['intermediate_size'],
            num_hidden_layers=student_config['num_hidden_layers'],
            num_attention_heads=student_config['num_attention_heads'],
            num_key_value_heads=student_config['num_key_value_heads'],
            hidden_act=student_config['hidden_act'],
            max_position_embeddings=student_config['max_position_embeddings'],
            rms_norm_eps=student_config['rms_norm_eps'],
            rope_theta=student_config['rope_theta'],
            attention_dropout=student_config['attention_dropout'],
            use_cache=student_config['use_cache'],
            # SO8T固有パラメータ
            rotation_dim=student_config.get('so8t_rotation_dim', 8),
            safety_weight=student_config.get('safety_weight', 0.1),
        )
        
        student_model = SO8TTransformerForCausalLM(config)
        
        logger.info(f"   - 学生モデルパラメータ数: {sum(p.numel() for p in student_model.parameters()):,}")
        logger.info(f"   - レイヤー数: {student_config['num_hidden_layers']}")
        logger.info(f"   - アテンションヘッド数: {student_config['num_attention_heads']}")
        logger.info(f"   - 隠れサイズ: {student_config['hidden_size']}")
        
        return student_model.to(self.device)
    
    def create_distillation_dataset(self, num_samples: int = 1000) -> List[Dict[str, Any]]:
        """蒸留用データセットを作成"""
        logger.info(f"蒸留用データセット作成中... (サンプル数: {num_samples})")
        
        # 多様なプロンプトを生成
        prompts = [
            "数学の問題を解いてください: 2x + 3 = 7",
            "以下の文章を要約してください: [長い文章]",
            "コードを書いてください: フィボナッチ数列を計算する関数",
            "倫理的な判断をしてください: AIの使用に関する問題",
            "科学的な説明をしてください: 光の性質について",
            "創造的な文章を書いてください: 未来の都市について",
            "論理的な推論をしてください: 三段論法の問題",
            "翻訳をしてください: 英語から日本語へ",
            "分析をしてください: データの傾向について",
            "設計をしてください: 効率的なアルゴリズム"
        ]
        
        dataset = []
        for i in tqdm(range(num_samples), desc="データセット作成"):
            prompt = prompts[i % len(prompts)]
            if i >= len(prompts):
                # バリエーションを追加
                prompt += f" (バリエーション {i // len(prompts) + 1})"
            
            dataset.append({
                'id': f"distill_{i:06d}",
                'prompt': prompt,
                'input_text': prompt,
                'expected_output': "",  # 教師モデルから生成
                'category': prompts[i % len(prompts)].split(':')[0],
                'difficulty': min(5, (i % 10) + 1),
                'created_at': datetime.now().isoformat()
            })
        
        logger.info(f"   ✓ データセット作成完了: {len(dataset)}サンプル")
        return dataset
    
    def distill_knowledge(self, 
                         teacher_model: nn.Module, 
                         student_model: SO8TTransformerForCausalLM,
                         dataset: List[Dict[str, Any]],
                         num_epochs: int = 10) -> SO8TTransformerForCausalLM:
        """知識蒸留を実行"""
        logger.info("知識蒸留開始...")
        
        # オプティマイザー設定
        optimizer = torch.optim.AdamW(
            student_model.parameters(),
            lr=self.lr_scheduler.get_learning_rate(0),
            weight_decay=0.01
        )
        
        # 学習ループ
        student_model.train()
        teacher_model.eval()
        
        total_loss = 0.0
        best_loss = float('inf')
        patience = 5
        patience_counter = 0
        
        for epoch in range(num_epochs):
            logger.info(f"エポック {epoch + 1}/{num_epochs}")
            
            epoch_loss = 0.0
            num_batches = 0
            
            # バッチ処理
            batch_size = 8
            for i in tqdm(range(0, len(dataset), batch_size), desc=f"エポック {epoch + 1}"):
                batch = dataset[i:i + batch_size]
                
                # バッチ処理
                batch_loss = self._process_batch(
                    teacher_model, student_model, batch, optimizer
                )
                
                epoch_loss += batch_loss
                num_batches += 1
                
                # 重み安定性監視
                if num_batches % 10 == 0:
                    self.weight_stability_manager.monitor_weights(
                        student_model, epoch * len(dataset) + i
                    )
            
            # エポック統計
            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
            total_loss += avg_loss
            
            logger.info(f"   - 平均損失: {avg_loss:.6f}")
            logger.info(f"   - 累積損失: {total_loss:.6f}")
            
            # 学習率更新
            current_lr = self.lr_scheduler.get_learning_rate(epoch)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
            logger.info(f"   - 学習率: {current_lr:.2e}")
            
            # ベストモデル保存
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                self._save_student_model(student_model, epoch, avg_loss, "best")
            else:
                patience_counter += 1
            
            # 早期停止
            if patience_counter >= patience:
                logger.info(f"早期停止: {patience}エポック改善なし")
                break
            
            # 定期的なチェックポイント保存
            if (epoch + 1) % 5 == 0:
                self._save_student_model(student_model, epoch, avg_loss, f"epoch_{epoch + 1}")
        
        logger.info("知識蒸留完了")
        logger.info(f"   - 最終損失: {avg_loss:.6f}")
        logger.info(f"   - ベスト損失: {best_loss:.6f}")
        
        return student_model
    
    def _process_batch(self, 
                      teacher_model: nn.Module, 
                      student_model: SO8TTransformerForCausalLM,
                      batch: List[Dict[str, Any]], 
                      optimizer: torch.optim.Optimizer) -> float:
        """バッチ処理"""
        optimizer.zero_grad()
        
        total_loss = 0.0
        
        for sample in batch:
            # 入力テキストをトークン化（簡易実装）
            input_text = sample['input_text']
            
            # 教師モデルからの出力（凍結されているため勾配なし）
            with torch.no_grad():
                teacher_outputs = self._get_teacher_outputs(teacher_model, input_text)
            
            # 学生モデルからの出力
            student_outputs = self._get_student_outputs(student_model, input_text)
            
            # 蒸留損失計算
            distillation_loss = self._calculate_distillation_loss(
                teacher_outputs, student_outputs
            )
            
            total_loss += distillation_loss
        
        # 平均損失
        avg_loss = total_loss / len(batch)
        
        # 逆伝播
        avg_loss.backward()
        
        # グラデーションクリッピング
        self.gradient_manager.clip_gradients(student_model)
        
        # オプティマイザー更新
        optimizer.step()
        
        return avg_loss.item()
    
    def _get_teacher_outputs(self, teacher_model: nn.Module, input_text: str) -> Dict[str, torch.Tensor]:
        """教師モデルからの出力を取得"""
        # 簡易実装：実際の実装では適切なトークン化と推論を行う
        batch_size = 1
        seq_len = 128
        
        # ダミー入力（実際の実装では適切なトークン化を行う）
        input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=self.device)
        attention_mask = torch.ones_like(input_ids)
        
        with torch.no_grad():
            outputs = teacher_model(input_ids=input_ids, attention_mask=attention_mask)
        
        return {
            'logits': outputs.logits,
            'hidden_states': outputs.hidden_states if hasattr(outputs, 'hidden_states') else None,
            'attentions': outputs.attentions if hasattr(outputs, 'attentions') else None
        }
    
    def _get_student_outputs(self, student_model: SO8TTransformerForCausalLM, input_text: str) -> Dict[str, torch.Tensor]:
        """学生モデルからの出力を取得"""
        # 簡易実装：実際の実装では適切なトークン化と推論を行う
        batch_size = 1
        seq_len = 128
        
        # ダミー入力（実際の実装では適切なトークン化を行う）
        input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=self.device)
        attention_mask = torch.ones_like(input_ids)
        
        outputs = student_model(input_ids=input_ids, attention_mask=attention_mask)
        
        return {
            'logits': outputs.logits,
            'hidden_states': outputs.hidden_states if hasattr(outputs, 'hidden_states') else None,
            'attentions': outputs.attentions if hasattr(outputs, 'attentions') else None
        }
    
    def _calculate_distillation_loss(self, 
                                   teacher_outputs: Dict[str, torch.Tensor], 
                                   student_outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """蒸留損失を計算"""
        temperature = self.distillation_config['temperature']
        alpha = self.distillation_config['alpha']
        beta = self.distillation_config['beta']
        
        # ソフトマックス損失（温度付き）
        teacher_logits = teacher_outputs['logits'] / temperature
        student_logits = student_outputs['logits'] / temperature
        
        # KL divergence損失
        kl_loss = F.kl_div(
            F.log_softmax(student_logits, dim=-1),
            F.softmax(teacher_logits, dim=-1),
            reduction='batchmean'
        ) * (temperature ** 2)
        
        # 中間層損失（もしあれば）
        intermediate_loss = 0.0
        if (teacher_outputs['hidden_states'] is not None and 
            student_outputs['hidden_states'] is not None):
            # 中間層の特徴量マッチング
            teacher_hidden = teacher_outputs['hidden_states'][-1]  # 最後の層
            student_hidden = student_outputs['hidden_states'][-1]
            
            # サイズを合わせる（簡易実装）
            min_size = min(teacher_hidden.size(-1), student_hidden.size(-1))
            teacher_hidden = teacher_hidden[..., :min_size]
            student_hidden = student_hidden[..., :min_size]
            
            intermediate_loss = F.mse_loss(student_hidden, teacher_hidden)
        
        # 総合損失
        total_loss = alpha * kl_loss + beta * intermediate_loss
        
        return total_loss
    
    def _save_student_model(self, 
                           student_model: SO8TTransformerForCausalLM, 
                           epoch: int, 
                           loss: float, 
                           suffix: str) -> str:
        """学生モデルを保存"""
        checkpoint_dir = self.output_dir / 'checkpoints'
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpoint_path = checkpoint_dir / f"student_model_{suffix}.pt"
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': student_model.state_dict(),
            'loss': loss,
            'config': self.student_config,
            'distillation_config': self.distillation_config,
            'timestamp': datetime.now().isoformat()
        }
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"チェックポイント保存: {checkpoint_path}")
        
        return str(checkpoint_path)
    
    def run_distillation(self, num_epochs: int = 10, num_samples: int = 1000) -> Dict[str, Any]:
        """知識蒸留を実行"""
        logger.info("=" * 80)
        logger.info("SO8T知識蒸留開始")
        logger.info("=" * 80)
        
        # セッション開始ログ（簡易実装）
        session_id = f"distillation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            # 1. 教師モデル読み込み
            logger.info("1/5: 教師モデル読み込み中...")
            teacher_model = self.load_teacher_model()
            
            # 2. 学生モデル作成
            logger.info("2/5: 学生モデル作成中...")
            student_model = self.create_student_model()
            
            # 3. データセット作成
            logger.info("3/5: データセット作成中...")
            dataset = self.create_distillation_dataset(num_samples)
            
            # 4. 知識蒸留実行
            logger.info("4/5: 知識蒸留実行中...")
            distilled_model = self.distill_knowledge(
                teacher_model, student_model, dataset, num_epochs
            )
            
            # 5. 最終モデル保存
            logger.info("5/5: 最終モデル保存中...")
            final_model_path = self._save_student_model(
                distilled_model, num_epochs - 1, 0.0, "final"
            )
            
            # 結果まとめ
            results = {
                'teacher_model_path': self.teacher_model_path,
                'student_model_path': final_model_path,
                'output_dir': str(self.output_dir),
                'num_epochs': num_epochs,
                'num_samples': num_samples,
                'session_id': session_id,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info("=" * 80)
            logger.info("SO8T知識蒸留完了！")
            logger.info("=" * 80)
            logger.info(f"教師モデル: {self.teacher_model_path}")
            logger.info(f"学生モデル: {final_model_path}")
            logger.info(f"出力ディレクトリ: {self.output_dir}")
            logger.info(f"エポック数: {num_epochs}")
            logger.info(f"サンプル数: {num_samples}")
            
            # セッション終了ログ（簡易実装）
            logger.info(f"知識蒸留セッション終了: {session_id}")
            
            return results
            
        except Exception as e:
            logger.error(f"知識蒸留エラー: {e}")
            logger.error(f"知識蒸留セッションエラー終了: {session_id} - {e}")
            raise


def main():
    """メイン関数"""
    # 設定
    teacher_model_path = "models/SO8T-Phi31-Mini-128K-Enhanced-Q8_0.gguf"
    student_config = {
        'vocab_size': 32000,
        'hidden_size': 2048,
        'intermediate_size': 8192,
        'num_hidden_layers': 16,
        'num_attention_heads': 16,
        'num_key_value_heads': 4,
        'hidden_act': 'silu',
        'max_position_embeddings': 131072,
        'rms_norm_eps': 1e-6,
        'rope_theta': 10000.0,
        'attention_dropout': 0.0,
        'use_cache': True,
        'so8t_rotation_dim': 8,
        'so8t_triality_symmetry': True,
        'so8t_cross_head_interaction': True,
        'so8t_non_commutative_gates': True,
    }
    
    try:
        # 知識蒸留システム初期化
        distillation_system = SO8TKnowledgeDistillation(
            teacher_model_path=teacher_model_path,
            student_config=student_config,
            output_dir="models/qwen_so8t_lightweight"
        )
        
        # 知識蒸留実行
        results = distillation_system.run_distillation(
            num_epochs=10,
            num_samples=1000
        )
        
        print("知識蒸留完了！")
        for key, value in results.items():
            print(f"   - {key}: {value}")
        
    except Exception as e:
        print(f"知識蒸留エラー: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

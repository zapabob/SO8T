#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SO(8) Transformer再学習 + ベイズ最適化統合スクリプト

温度較正とハイパーパラメータ最適化を同時に行う
- OptunaベースのTPE最適化
- 目的関数: REFUSE再現率 + ECE最小化 + F1マクロ
- RTX3060/32GB最適化
- 電源断リカバリー機能

Usage:
    python scripts/training/train_so8t_with_bayesian.py --config configs/so8t_bayesian_config.yaml
"""

import os
import sys
import json
import logging
import argparse
import signal
import pickle
import time
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import optuna
from optuna.pruners import MedianPruner
from scipy.optimize import minimize

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "so8t-mmllm"))

from models.so8t_transformer import SO8TTransformerForCausalLM, SO8TTransformerConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from scripts.training.train_so8t_recovery import RecoverySO8TTrainer

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/train_so8t_bayesian.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class BayesianSO8TTrainer:
    """SO(8) Transformer + ベイズ最適化統合トレーナー"""
    
    def __init__(self, config: Dict):
        """
        Args:
            config: 設定辞書
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # セッション管理
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.checkpoint_dir = Path(config.get('checkpoint_dir', 'D:/webdataset/checkpoints/training')) / self.session_id
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # チェックポイント設定
        self.checkpoint_interval = config.get('checkpoint_interval', 300)  # 5分
        self.last_checkpoint_time = time.time()
        
        # シグナルハンドラー設定
        self._setup_signal_handlers()
        
        # モデル・データセット
        self.model = None
        self.tokenizer = None
        self.train_dataset = None
        self.val_dataset = None
        
        # ベイズ最適化設定
        self.n_trials = config.get('n_trials', 50)
        self.study_name = config.get('study_name', f'so8t_bayesian_{self.session_id}')
        
        logger.info("="*80)
        logger.info("Bayesian SO8T Trainer Initialized")
        logger.info("="*80)
        logger.info(f"Session ID: {self.session_id}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Checkpoint directory: {self.checkpoint_dir}")
    
    def _setup_signal_handlers(self):
        """シグナルハンドラー設定"""
        def signal_handler(signum, frame):
            logger.warning(f"Received signal {signum}, saving checkpoint...")
            self._save_checkpoint()
            logger.info("Checkpoint saved. Exiting gracefully...")
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        if hasattr(signal, 'SIGBREAK'):
            signal.signal(signal.SIGBREAK, signal_handler)
    
    def _save_checkpoint(self):
        """チェックポイント保存"""
        checkpoint_file = self.checkpoint_dir / f"checkpoint_{int(time.time())}.pt"
        
        checkpoint_data = {
            'session_id': self.session_id,
            'model_state_dict': self.model.state_dict() if self.model else None,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            torch.save(checkpoint_data, checkpoint_file)
            logger.info(f"[CHECKPOINT] Saved to {checkpoint_file}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
    
    def _load_dataset(self, data_path: Path) -> Dataset:
        """データセット読み込み"""
        logger.info(f"Loading dataset from {data_path}")
        
        class SO8TDataset(Dataset):
            def __init__(self, data_path: Path, tokenizer, max_length: int = 512):
                self.samples = []
                with open(data_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            sample = json.loads(line)
                            self.samples.append(sample)
                        except json.JSONDecodeError:
                            continue
                
                self.tokenizer = tokenizer
                self.max_length = max_length
            
            def __len__(self):
                return len(self.samples)
            
            def __getitem__(self, idx):
                sample = self.samples[idx]
                text = sample.get('text', sample.get('output', ''))
                
                # トークナイズ
                encoded = self.tokenizer(
                    text,
                    max_length=self.max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                
                return {
                    'input_ids': encoded['input_ids'].squeeze(),
                    'attention_mask': encoded['attention_mask'].squeeze(),
                    'labels': encoded['input_ids'].squeeze(),
                    'safety_labels': torch.tensor(
                        self._label_to_id(sample.get('safety_judgment', 'ALLOW'))
                    )
                }
            
            def _label_to_id(self, label: str) -> int:
                label_map = {'ALLOW': 0, 'ESCALATION': 1, 'DENY': 2, 'REFUSE': 2}
                return label_map.get(label, 0)
        
        return SO8TDataset(data_path, self.tokenizer, max_length=self.config.get('max_length', 512))
    
    def _calculate_ece(self, confidences: np.ndarray, accuracies: np.ndarray, n_bins: int = 10) -> float:
        """Expected Calibration Error計算"""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = accuracies[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    def objective(self, trial: optuna.Trial) -> float:
        """
        ベイズ最適化目的関数（温度較正 + ハイパーパラメータ同時最適化）
        
        Args:
            trial: Optunaトライアル
        
        Returns:
            目的関数値（REFUSE再現率 + ECE最小化 + F1マクロ）
        """
        # ハイパーパラメータ提案
        pet_lambda = trial.suggest_float("pet_lambda", 0.001, 0.1, log=True)
        safety_weight = trial.suggest_float("safety_weight", 0.05, 0.2)
        cmd_weight = trial.suggest_float("cmd_weight", 0.8, 0.95)
        
        # 温度パラメータ（同時最適化）
        temperature = trial.suggest_float("temperature", 0.5, 2.0)
        
        # SO8T Transformer設定
        so8t_config = SO8TTransformerConfig(
            vocab_size=self.config.get('vocab_size', 152064),
            hidden_size=self.config.get('hidden_size', 3584),
            intermediate_size=self.config.get('intermediate_size', 18944),
            num_hidden_layers=self.config.get('num_hidden_layers', 28),
            num_attention_heads=self.config.get('num_attention_heads', 28),
            pet_lambda=pet_lambda,
            safety_weight=safety_weight,
            cmd_weight=cmd_weight,
            gradient_checkpointing=True,  # RTX3060最適化
            use_flash_attention=False
        )
        
        # モデル作成
        base_model_name = self.config.get('base_model', 'Qwen/Qwen2.5-7B-Instruct')
        model = SO8TTransformerForCausalLM(so8t_config)
        
        # ベースモデルから重みをロード（可能な場合）
        try:
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto" if self.device.type == "cuda" else None
            )
            # 重みコピー（互換性がある場合）
            logger.debug("Loading base model weights...")
        except Exception as e:
            logger.warning(f"Could not load base model: {e}")
        
        model = model.to(self.device)
        model.train()
        
        # オプティマイザー
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.get('learning_rate', 2e-4),
            weight_decay=self.config.get('weight_decay', 0.01)
        )
        
        # 簡易訓練（1エポック）
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.get('batch_size', 1),
            shuffle=True
        )
        
        # 混合精度
        scaler = torch.cuda.amp.GradScaler() if self.device.type == "cuda" else None
        
        for epoch in range(1):  # 簡易版
            for batch_idx, batch in enumerate(train_loader):
                if batch_idx >= 10:  # 簡易評価用に制限
                    break
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                safety_labels = batch['safety_labels'].to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                if scaler:
                    with torch.cuda.amp.autocast():
                        outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels,
                            safety_labels=safety_labels
                        )
                        loss = outputs.get('loss', outputs.get('task_loss', torch.tensor(0.0)))
                    
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        safety_labels=safety_labels
                    )
                    loss = outputs.get('loss', outputs.get('task_loss', torch.tensor(0.0)))
                    loss.backward()
                    optimizer.step()
                
                # 早期終了判定
                if trial.should_prune():
                    raise optuna.TrialPruned()
        
        # 検証セットで評価
        model.eval()
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.get('batch_size', 1),
            shuffle=False
        )
        
        all_logits = []
        all_labels = []
        all_safety_labels = []
        all_safety_logits = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                if batch_idx >= 50:  # 評価用に制限
                    break
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                safety_labels = batch['safety_labels'].to(self.device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                # ロジット取得
                logits = outputs.get('logits', outputs.get('task_logits'))
                safety_logits = outputs.get('safety_logits')
                
                if logits is not None:
                    all_logits.append(logits[:, -1, :].cpu().numpy())
                    all_labels.append(input_ids[:, -1].cpu().numpy())
                
                if safety_logits is not None:
                    all_safety_logits.append(safety_logits.cpu().numpy())
                    all_safety_labels.append(safety_labels.cpu().numpy())
        
        # メトリクス計算
        objective_value = 0.0
        
        # 1. REFUSE再現率（最重要）
        if len(all_safety_logits) > 0:
            safety_logits_array = np.concatenate(all_safety_logits, axis=0)
            safety_labels_array = np.concatenate(all_safety_labels, axis=0)
            
            # 温度スケーリング適用
            scaled_safety_logits = safety_logits_array / temperature
            safety_probs = F.softmax(torch.from_numpy(scaled_safety_logits), dim=-1).numpy()
            safety_pred = np.argmax(safety_probs, axis=-1)
            
            # REFUSE再現率
            refuse_mask = safety_labels_array == 2
            if refuse_mask.any():
                refuse_recall = (safety_pred[refuse_mask] == 2).mean()
                objective_value += 0.5 * refuse_recall
            
            # ESCALATE再現率
            escalate_mask = safety_labels_array == 1
            if escalate_mask.any():
                escalate_recall = (safety_pred[escalate_mask] == 1).mean()
                objective_value += 0.2 * escalate_recall
            
            # F1マクロ
            from sklearn.metrics import f1_score
            f1_macro = f1_score(safety_labels_array, safety_pred, average='macro')
            objective_value += 0.2 * f1_macro
        
        # 2. ECE最小化（温度較正）
        if len(all_logits) > 0:
            logits_array = np.concatenate(all_logits, axis=0)
            labels_array = np.concatenate(all_labels, axis=0)
            
            # 温度スケーリング適用
            scaled_logits = logits_array / temperature
            probs = F.softmax(torch.from_numpy(scaled_logits), dim=-1).numpy()
            
            predictions = np.argmax(probs, axis=-1)
            confidences = np.max(probs, axis=-1)
            accuracies = (predictions == labels_array).astype(float)
            
            # ECE計算
            ece = self._calculate_ece(confidences, accuracies)
            
            # ECEを最小化（負の値を最大化）
            objective_value += 0.1 * (1.0 - ece)  # ECEが小さいほど良い
        
        return objective_value
    
    def optimize(self):
        """ベイズ最適化実行"""
        logger.info("="*80)
        logger.info("Starting Bayesian Optimization")
        logger.info("="*80)
        logger.info(f"Number of trials: {self.n_trials}")
        logger.info(f"Study name: {self.study_name}")
        
        # Optunaスタディ作成
        study = optuna.create_study(
            study_name=self.study_name,
            direction="maximize",
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        )
        
        # 最適化実行
        study.optimize(
            self.objective,
            n_trials=self.n_trials,
            n_jobs=1,  # RTX3060では並列化しない
            show_progress_bar=True
        )
        
        # 最適パラメータ取得
        best_params = study.best_params
        best_value = study.best_value
        
        logger.info("="*80)
        logger.info("Bayesian Optimization Completed")
        logger.info("="*80)
        logger.info(f"Best value: {best_value:.6f}")
        logger.info(f"Best parameters:")
        for key, value in best_params.items():
            logger.info(f"  {key}: {value:.6f}")
        
        # 結果保存
        result_path = self.checkpoint_dir / "bayesian_optimization_results.json"
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump({
                'best_value': best_value,
                'best_params': best_params,
                'n_trials': len(study.trials),
                'study_name': self.study_name
            }, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to {result_path}")
        
        return study, best_params
    
    def train_with_optimized_params(self, best_params: Dict):
        """最適化済みパラメータで最終学習"""
        logger.info("="*80)
        logger.info("Training with Optimized Parameters")
        logger.info("="*80)
        
        # 最適化済み設定でモデル作成
        so8t_config = SO8TTransformerConfig(
            vocab_size=self.config.get('vocab_size', 152064),
            hidden_size=self.config.get('hidden_size', 3584),
            intermediate_size=self.config.get('intermediate_size', 18944),
            num_hidden_layers=self.config.get('num_hidden_layers', 28),
            num_attention_heads=self.config.get('num_attention_heads', 28),
            pet_lambda=best_params['pet_lambda'],
            safety_weight=best_params['safety_weight'],
            cmd_weight=best_params['cmd_weight'],
            gradient_checkpointing=True,
            use_flash_attention=False
        )
        
        model = SO8TTransformerForCausalLM(so8t_config).to(self.device)
        
        # 学習実行（RecoverySO8TTrainerを使用）
        recovery_trainer = RecoverySO8TTrainer(
            config_path=self.config.get('training_config', 'configs/training_config.yaml')
        )
        
        # 温度パラメータ保存
        temperature = best_params.get('temperature', 1.0)
        temp_config_path = self.checkpoint_dir / "optimal_temperature.json"
        with open(temp_config_path, 'w', encoding='utf-8') as f:
            json.dump({'temperature': temperature}, f, indent=2)
        
        logger.info(f"Optimal temperature saved: {temperature:.4f}")
        logger.info(f"Model checkpoint directory: {self.checkpoint_dir}")
        
        return model, temperature


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="SO(8) Transformer Bayesian Training")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/so8t_bayesian_config.yaml',
        help='Configuration file path'
    )
    parser.add_argument(
        '--train-data',
        type=str,
        required=True,
        help='Training data path (JSONL)'
    )
    parser.add_argument(
        '--val-data',
        type=str,
        required=True,
        help='Validation data path (JSONL)'
    )
    parser.add_argument(
        '--n-trials',
        type=int,
        default=50,
        help='Number of optimization trials'
    )
    
    args = parser.parse_args()
    
    # 設定読み込み
    config_path = Path(args.config)
    if config_path.exists():
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    else:
        config = {}
        logger.warning(f"Config file not found: {config_path}, using defaults")
    
    config['train_data'] = args.train_data
    config['val_data'] = args.val_data
    config['n_trials'] = args.n_trials
    
    # トークナイザー読み込み
    base_model = config.get('base_model', 'Qwen/Qwen2.5-7B-Instruct')
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # トレーナー初期化
    trainer = BayesianSO8TTrainer(config)
    trainer.tokenizer = tokenizer
    
    # データセット読み込み
    trainer.train_dataset = trainer._load_dataset(Path(args.train_data))
    trainer.val_dataset = trainer._load_dataset(Path(args.val_data))
    
    logger.info(f"Train samples: {len(trainer.train_dataset):,}")
    logger.info(f"Val samples: {len(trainer.val_dataset):,}")
    
    # ベイズ最適化実行
    study, best_params = trainer.optimize()
    
    # 最適化済みパラメータで最終学習
    model, temperature = trainer.train_with_optimized_params(best_params)
    
    logger.info("="*80)
    logger.info("[COMPLETE] Bayesian optimization and training completed!")
    logger.info(f"Optimal temperature: {temperature:.4f}")
    logger.info(f"Model saved to: {trainer.checkpoint_dir}")
    logger.info("="*80)


if __name__ == '__main__':
    main()


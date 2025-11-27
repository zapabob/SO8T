#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SO8Tモデル ハイパーパラメータ最適化スクリプト（ベイズ最適化 + クロスバリデーション）

Optunaを使用したベイズ最適化とK-foldクロスバリデーションを組み合わせて
SO8Tモデルのハイパーパラメータを最適化
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# HuggingFaceキャッシュをDドライブに設定
os.environ["HF_HOME"] = r"D:\webdataset\hf_cache"
os.environ["TRANSFORMERS_CACHE"] = r"D:\webdataset\hf_cache\transformers"
os.environ["HF_DATASETS_CACHE"] = r"D:\webdataset\hf_cache\datasets"
os.environ["HF_HUB_CACHE"] = r"D:\webdataset\hf_cache\hub"

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training
)
import optuna
from optuna.pruners import MedianPruner
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_parallel_coordinate
)
from sklearn.model_selection import KFold
import yaml
from tqdm import tqdm

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "so8t-mmllm" / "src"))

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/hyperparameter_optimization_cv.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class HyperparameterOptimizer:
    """ベイズ最適化 + クロスバリデーションによるハイパーパラメータ最適化"""
    
    def __init__(
        self,
        model_path: str,
        dataset_path: Path,
        config: Dict[str, Any],
        output_dir: Path,
        n_folds: int = 5,
        n_trials: int = 50
    ):
        """
        Args:
            model_path: ベースモデルパス
            dataset_path: データセットパス
            config: 設定辞書
            output_dir: 出力ディレクトリ
            n_folds: クロスバリデーションのフォールド数
            n_trials: ベイズ最適化の試行回数
        """
        self.model_path = model_path
        self.dataset_path = dataset_path
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.n_folds = n_folds
        self.n_trials = n_trials
        
        # データセットを読み込み
        self.dataset = self._load_dataset()
        
        # K-fold分割
        self.kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    def _load_dataset(self) -> Dataset:
        """データセットを読み込み"""
        logger.info(f"Loading dataset from {self.dataset_path}...")
        
        # ThinkingSFTDatasetを使用（train_borea_phi35_so8t_thinking.pyから）
        from scripts.training.train_borea_phi35_so8t_thinking import ThinkingSFTDataset
        
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        dataset = ThinkingSFTDataset(
            data_path=self.dataset_path,
            tokenizer=tokenizer,
            max_length=self.config.get("data", {}).get("max_seq_length", 2048)
        )
        
        logger.info(f"[OK] Loaded {len(dataset)} samples")
        return dataset
    
    def _train_model_with_params(
        self,
        params: Dict[str, Any],
        train_indices: List[int],
        val_indices: List[int],
        trial: Optional[optuna.Trial] = None
    ) -> Tuple[float, Dict[str, float]]:
        """
        指定されたハイパーパラメータでモデルを学習し、検証スコアを返す
        
        Args:
            params: ハイパーパラメータ辞書
            train_indices: 訓練データのインデックス
            val_indices: 検証データのインデックス
            trial: Optunaトライアル（早期停止用）
        
        Returns:
            (検証損失, メトリクス辞書)
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # トークナイザー読み込み
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # モデル読み込み
        load_in_8bit = self.config.get("quantization", {}).get("load_in_8bit", True)
        quantization_config = None
        if load_in_8bit:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0
            )
        
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        
        # QLoRA設定
        if self.config.get("qlora", {}).get("enabled", True):
            model = prepare_model_for_kbit_training(model)
            qlora_config = self.config.get("qlora", {})
            lora_config = LoraConfig(
                r=params.get("lora_r", qlora_config.get("r", 64)),
                lora_alpha=params.get("lora_alpha", qlora_config.get("lora_alpha", 128)),
                target_modules=qlora_config.get("target_modules", [
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"
                ]),
                lora_dropout=params.get("lora_dropout", qlora_config.get("lora_dropout", 0.05)),
                bias="none",
                task_type=TaskType.CAUSAL_LM
            )
            model = get_peft_model(model, lora_config)
        
        # データローダー作成
        train_subset = Subset(self.dataset, train_indices)
        val_subset = Subset(self.dataset, val_indices)
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )
        
        train_loader = DataLoader(
            train_subset,
            batch_size=params.get("batch_size", 1),
            collate_fn=data_collator,
            shuffle=True
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=params.get("batch_size", 1),
            collate_fn=data_collator,
            shuffle=False
        )
        
        # オプティマイザー
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=params.get("learning_rate", 2e-4),
            weight_decay=params.get("weight_decay", 0.01)
        )
        
        # 学習ループ（簡易版）
        model.train()
        num_epochs = params.get("num_epochs", 1)  # 最適化時は1エポック
        
        total_loss = 0.0
        num_batches = 0
        
        for epoch in range(num_epochs):
            for batch in train_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                
                outputs = model(**batch)
                loss = outputs.loss
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), params.get("max_grad_norm", 1.0))
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                # 早期停止（Optuna Pruner）
                if trial is not None:
                    trial.report(loss.item(), num_batches)
                    if trial.should_prune():
                        raise optuna.TrialPruned()
        
        # 検証
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                val_loss += outputs.loss.item()
                val_batches += 1
        
        avg_val_loss = val_loss / val_batches if val_batches > 0 else float('inf')
        
        metrics = {
            "val_loss": avg_val_loss,
            "train_loss": total_loss / num_batches if num_batches > 0 else float('inf')
        }
        
        # メモリクリーンアップ
        del model
        torch.cuda.empty_cache()
        
        return avg_val_loss, metrics
    
    def objective(self, trial: optuna.Trial) -> float:
        """
        Optuna目的関数（クロスバリデーション付き）
        
        Args:
            trial: Optunaトライアル
        
        Returns:
            クロスバリデーション平均スコア
        """
        # ハイパーパラメータの提案
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [1, 2, 4]),
            "weight_decay": trial.suggest_float("weight_decay", 1e-5, 0.1, log=True),
            "max_grad_norm": trial.suggest_float("max_grad_norm", 0.1, 2.0),
            "lora_r": trial.suggest_categorical("lora_r", [16, 32, 64, 128]),
            "lora_alpha": trial.suggest_int("lora_alpha", 32, 256),
            "lora_dropout": trial.suggest_float("lora_dropout", 0.01, 0.2),
            "num_epochs": 1,  # 最適化時は1エポック
        }
        
        # PET正則化パラメータ
        if self.config.get("pet", {}).get("enabled", True):
            params["pet_lambda_exploration"] = trial.suggest_float("pet_lambda_exploration", 0.001, 0.1, log=True)
            params["pet_lambda_transition"] = trial.suggest_float("pet_lambda_transition", 0.01, 0.2, log=True)
            params["pet_lambda_stabilization"] = trial.suggest_float("pet_lambda_stabilization", 0.01, 0.3, log=True)
        
        # SO8Tパラメータ
        if self.config.get("so8t", {}).get("enabled", True):
            params["so8t_init_scale"] = trial.suggest_float("so8t_init_scale", 0.01, 0.1, log=True)
            params["so8t_orthogonal_reg"] = trial.suggest_float("so8t_orthogonal_reg", 1e-6, 1e-3, log=True)
        
        logger.info(f"[TRIAL {trial.number}] Testing parameters: LR={params['learning_rate']:.2e}, "
                   f"BS={params['batch_size']}, LoRA_r={params['lora_r']}")
        
        # クロスバリデーション
        cv_scores = []
        cv_metrics = []
        
        dataset_indices = list(range(len(self.dataset)))
        
        for fold, (train_idx, val_idx) in enumerate(self.kfold.split(dataset_indices)):
            try:
                val_loss, metrics = self._train_model_with_params(
                    params=params,
                    train_indices=train_idx.tolist(),
                    val_indices=val_idx.tolist(),
                    trial=trial if fold == 0 else None  # 最初のフォールドのみPrunerを使用
                )
                cv_scores.append(val_loss)
                cv_metrics.append(metrics)
                
                logger.info(f"[TRIAL {trial.number}] Fold {fold+1}/{self.n_folds}: Val Loss={val_loss:.4f}")
                
            except optuna.TrialPruned:
                logger.info(f"[TRIAL {trial.number}] Pruned at fold {fold+1}")
                raise
            except Exception as e:
                logger.warning(f"[TRIAL {trial.number}] Fold {fold+1} failed: {e}")
                cv_scores.append(float('inf'))
        
        # クロスバリデーション平均スコア
        cv_mean = np.mean(cv_scores)
        cv_std = np.std(cv_scores)
        
        logger.info(f"[TRIAL {trial.number}] CV Score: {cv_mean:.4f} (+/- {cv_std:.4f})")
        
        # メトリクスを記録
        trial.set_user_attr("cv_mean", cv_mean)
        trial.set_user_attr("cv_std", cv_std)
        trial.set_user_attr("cv_scores", cv_scores)
        
        return cv_mean
    
    def optimize(self) -> Tuple[optuna.Study, Dict[str, Any]]:
        """
        ベイズ最適化を実行
        
        Returns:
            (Optuna Study, 最適パラメータ)
        """
        logger.info("="*80)
        logger.info("Starting Bayesian Optimization with Cross-Validation")
        logger.info("="*80)
        logger.info(f"Number of trials: {self.n_trials}")
        logger.info(f"Number of CV folds: {self.n_folds}")
        logger.info(f"Dataset size: {len(self.dataset)}")
        
        # Optunaスタディ作成
        study = optuna.create_study(
            study_name=f"so8t_hyperopt_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            direction="minimize",  # 検証損失を最小化
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
        logger.info(f"Best CV score: {best_value:.6f}")
        logger.info(f"Best parameters:")
        for key, value in best_params.items():
            logger.info(f"  {key}: {value}")
        
        # 結果保存
        result_path = self.output_dir / "hyperparameter_optimization_results.json"
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump({
                'best_value': best_value,
                'best_params': best_params,
                'n_trials': len(study.trials),
                'n_folds': self.n_folds,
                'study_name': study.study_name,
                'trials': [
                    {
                        'number': t.number,
                        'value': t.value,
                        'params': t.params,
                        'user_attrs': t.user_attrs
                    }
                    for t in study.trials
                ]
            }, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to {result_path}")
        
        # 可視化（オプション）
        try:
            self._visualize_results(study)
        except Exception as e:
            logger.warning(f"Failed to create visualizations: {e}")
        
        return study, best_params
    
    def _visualize_results(self, study: optuna.Study):
        """最適化結果を可視化"""
        try:
            import matplotlib
            matplotlib.use('Agg')  # バックエンドを設定
            
            vis_dir = self.output_dir / "visualizations"
            vis_dir.mkdir(exist_ok=True)
            
            # 最適化履歴
            fig = plot_optimization_history(study)
            fig.write_image(str(vis_dir / "optimization_history.png"))
            
            # パラメータ重要度
            fig = plot_param_importances(study)
            fig.write_image(str(vis_dir / "param_importances.png"))
            
            # パラレル座標
            fig = plot_parallel_coordinate(study)
            fig.write_image(str(vis_dir / "parallel_coordinate.png"))
            
            logger.info(f"Visualizations saved to {vis_dir}")
        except ImportError:
            logger.warning("Plotly or Kaleido not available, skipping visualizations")
        except Exception as e:
            logger.warning(f"Failed to create visualizations: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Hyperparameter optimization with Bayesian optimization and cross-validation"
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Config file path (YAML)"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="AXCXEPT/Borea-Phi-3.5-mini-Instruct-Jp",
        help="Base model path"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Training dataset path (JSONL format)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for optimization results"
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of optimization trials"
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help="Number of CV folds"
    )
    
    args = parser.parse_args()
    
    # 設定ファイル読み込み
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 最適化実行
    optimizer = HyperparameterOptimizer(
        model_path=args.model_path,
        dataset_path=args.dataset,
        config=config,
        output_dir=output_dir,
        n_folds=args.n_folds,
        n_trials=args.n_trials
    )
    
    study, best_params = optimizer.optimize()
    
    logger.info("="*80)
    logger.info("Optimization Complete")
    logger.info("="*80)
    logger.info(f"Best parameters saved to: {output_dir / 'hyperparameter_optimization_results.json'}")
    logger.info("You can use these parameters in your training configuration.")


if __name__ == "__main__":
    main()








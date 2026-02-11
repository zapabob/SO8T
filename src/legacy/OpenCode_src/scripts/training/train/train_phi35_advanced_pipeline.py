#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Phi-3.5 SO8T Training Pipeline with Bayesian Optimization

動的Thinking、マルチモーダル、メタ推論、ベイズ最適化を統合した
高度なPhi-3.5 SO8Tモデル学習パイプライン

特徴:
- SO8ViT/Thinking Adapter with SO8 rotation gates
- Dynamic Thinking based on query types
- Multimodal integration (vision + audio)
- Meta-reasoning analysis
- Bayesian optimization of α parameter
- Comprehensive benchmark evaluation
"""

import os
import sys
import json
import logging
import argparse
import math
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import subprocess
import warnings
warnings.filterwarnings("ignore")

# HuggingFaceキャッシュ設定
os.environ["HF_HOME"] = r"D:\webdataset\hf_cache"
os.environ["TRANSFORMERS_CACHE"] = r"D:\webdataset\hf_cache\transformers"
os.environ["HF_DATASETS_CACHE"] = r"D:\webdataset\hf_cache\datasets"
os.environ["HF_HUB_CACHE"] = r"D:\webdataset\hf_cache\hub"

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

# SO8T関連インポート
try:
    from so8t.core.dynamic_thinking_so8t import DynamicThinkingSO8TModel, create_dynamic_thinking_so8t
    from so8t.optimization.bayesian_alpha_optimizer import BayesianAlphaOptimizer, create_bayesian_optimizer
    from so8t.evaluation.comprehensive_benchmark_evaluator import run_comprehensive_evaluation
    from so8t.core.so8vit_thinking_adapter import SO8ViTThinkingAdapter
except ImportError as e:
    logging.warning(f"SO8T import failed: {e}")
    DynamicThinkingSO8TModel = None

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/phi35_advanced_pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class AdvancedPhi35Trainer:
    """
    Advanced Phi-3.5 SO8T Trainer with Bayesian Optimization

    高度な学習機能:
    - SO8ViT/Thinking Adapter
    - Dynamic Thinking
    - Multimodal Integration
    - Meta-reasoning
    - Bayesian α Optimization
    """

    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.model = None
        self.tokenizer = None
        self.dataset = None
        self.bayesian_optimizer = None

        # 高度な機能設定
        self.advanced_features = self.config.get('advanced', {})

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """設定ファイル読み込み"""
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        # 高度な機能のデフォルト設定
        config.setdefault('advanced', {})
        config['advanced'].setdefault('so8vit_adapter', True)
        config['advanced'].setdefault('dynamic_thinking', True)
        config['advanced'].setdefault('multimodal_integration', True)
        config['advanced'].setdefault('meta_reasoning', True)
        config['advanced'].setdefault('bayesian_optimization', True)
        config['advanced'].setdefault('orthogonal_error_logging', True)

        return config

    def initialize_advanced_model(self):
        """高度なPhi-3.5 SO8Tモデルの初期化"""
        logger.info("Initializing Advanced Phi-3.5 SO8T Model...")
        logger.info("="*60)

        if DynamicThinkingSO8TModel is None:
            raise RuntimeError("SO8T modules not available. Please check installation.")

        # Dynamic Thinking SO8Tモデル作成
        self.model = create_dynamic_thinking_so8t(self.config)

        # 高度な機能有効化
        self.model.enable_thinking_features(
            dynamic=self.advanced_features.get('dynamic_thinking', True),
            multimodal=self.advanced_features.get('multimodal_integration', True),
            meta_reasoning=self.advanced_features.get('meta_reasoning', True)
        )

        # ベイズ最適化器初期化
        if self.advanced_features.get('bayesian_optimization', True):
            self.bayesian_optimizer = create_bayesian_optimizer(
                n_iterations=self.config.get('optimization', {}).get('bayesian_iterations', 25)
            )

        logger.info("Advanced Phi-3.5 SO8T Model initialized with:")
        logger.info(f"  - SO8ViT Adapter: {'✓' if self.advanced_features.get('so8vit_adapter') else '✗'}")
        logger.info(f"  - Dynamic Thinking: {'✓' if self.advanced_features.get('dynamic_thinking') else '✗'}")
        logger.info(f"  - Multimodal Integration: {'✓' if self.advanced_features.get('multimodal_integration') else '✗'}")
        logger.info(f"  - Meta Reasoning: {'✓' if self.advanced_features.get('meta_reasoning') else '✗'}")
        logger.info(f"  - Bayesian Optimization: {'✓' if self.bayesian_optimizer else '✗'}")
        logger.info(f"  - Orthogonal Error Logging: {'✓' if self.advanced_features.get('orthogonal_error_logging') else '✗'}")

    def setup_tokenizer_and_dataset(self):
        """トークナイザーとデータセット設定"""
        logger.info("Setting up tokenizer and dataset...")

        # トークナイザー設定
        model_name = self.config.get('model', {}).get('name', 'microsoft/phi-3.5-mini-instruct')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Phi-3.5 Thinkingトークン追加
        special_tokens = {
            'additional_special_tokens': [
                '<think-task>', '</think-task>',
                '<think-safety>', '</think-safety>',
                '<think-logic>', '</think-logic>',
                '<think-ethics>', '</think-ethics>',
                '<think-practical>', '</think-practical>',
                '<think-creative>', '</think-creative>',
                '<final>', '</final>'
            ]
        }
        self.tokenizer.add_special_tokens(special_tokens)

        # データセット設定
        from scripts.data.convert_integrated_to_phi35 import Phi35ThinkingDataset
        dataset_path = self.config.get('data', {}).get('train_path',
            'D:/webdataset/phi35_integrated/phi35_ppo_optimized_integrated.jsonl')

        self.dataset = Phi35ThinkingDataset(
            data_path=dataset_path,
            tokenizer=self.tokenizer,
            max_length=self.config.get('data', {}).get('max_length', 2048)
        )

        logger.info(f"Dataset loaded: {len(self.dataset)} samples")

    def run_bayesian_optimization(self) -> Dict[str, Any]:
        """
        ベイズ最適化実行

        Returns:
            最適化結果
        """
        if not self.bayesian_optimizer:
            logger.info("Bayesian optimization disabled, skipping...")
            return {'best_alpha': 0.5, 'optimization_skipped': True}

        logger.info("Starting Bayesian Optimization of α parameter...")
        logger.info("="*60)

        def objective_function(alpha: float) -> float:
            """目的関数: α値でのモデル性能評価"""
            logger.info(f"Evaluating α = {alpha:.4f}...")

            # α値をモデルに設定（シグモイド適用）
            sigmoid_alpha = torch.sigmoid(torch.tensor(alpha)).item()
            if hasattr(self.model, 'so8vit_adapter'):
                self.model.so8vit_adapter.thinking_alpha.data = torch.tensor(sigmoid_alpha)

            # 簡易評価（実際にはより包括的な評価が必要）
            score = self._quick_model_evaluation()
            return score

        # 最適化実行
        optimization_result = self.bayesian_optimizer.optimize(objective_function)

        # 最適α値をモデルに設定
        best_alpha = optimization_result['best_alpha']
        sigmoid_best_alpha = torch.sigmoid(torch.tensor(best_alpha)).item()

        if hasattr(self.model, 'so8vit_adapter'):
            self.model.so8vit_adapter.thinking_alpha.data = torch.tensor(sigmoid_best_alpha)

        logger.info("Bayesian optimization completed!")
        logger.info(f"Optimal α: {best_alpha:.4f} (sigmoid: {sigmoid_best_alpha:.4f})")
        logger.info(f"Best score: {optimization_result['best_score']:.4f}")

        return optimization_result

    def _quick_model_evaluation(self) -> float:
        """簡易モデル評価（ベイズ最適化用）"""
        # 簡易評価サンプル
        test_samples = [
            "2 + 2 = ?",
            "Explain quantum computing",
            "Write a haiku about AI"
        ]

        total_score = 0.0

        for sample in test_samples:
            try:
                inputs = self.tokenizer(sample, return_tensors='pt', truncation=True, max_length=512)

                with torch.no_grad():
                    outputs = self.model(**inputs)
                    # 簡易スコア：出力のエントロピー（多様性の指標）
                    logits = outputs.logits[0, -1, :]  # 最後のトークンのlogits
                    probs = torch.softmax(logits, dim=-1)
                    entropy = -torch.sum(probs * torch.log(probs + 1e-10))
                    score = 1.0 - (entropy / math.log(len(self.tokenizer)))  # 正規化

                total_score += score.item()

            except Exception as e:
                logger.warning(f"Evaluation failed for sample '{sample}': {e}")
                continue

        return total_score / len(test_samples) if test_samples else 0.0

    def run_advanced_training(self, output_dir: str):
        """
        高度な学習実行

        Args:
            output_dir: 出力ディレクトリ
        """
        logger.info("Starting Advanced Phi-3.5 SO8T Training...")
        logger.info("="*80)

        # モデル初期化
        self.initialize_advanced_model()
        self.setup_tokenizer_and_dataset()

        # ベイズ最適化
        optimization_result = self.run_bayesian_optimization()

        # モデルをトークナイザーに合わせる
        self.model.resize_token_embeddings(len(self.tokenizer))

        # トレーニング設定
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.config.get('training', {}).get('num_epochs', 3),
            per_device_train_batch_size=self.config.get('training', {}).get('batch_size', 1),
            gradient_accumulation_steps=self.config.get('training', {}).get('gradient_accumulation_steps', 16),
            learning_rate=self.config.get('training', {}).get('learning_rate', 1e-5),
            lr_scheduler_type="cosine",
            warmup_steps=200,
            max_steps=self.config.get('training', {}).get('max_steps', 1000),
            save_steps=100,
            evaluation_strategy="steps",
            eval_steps=100,
            logging_steps=50,
            bf16=True,
            remove_unused_columns=False,
        )

        # カスタムコールバック（直交誤差logging用）
        callbacks = []
        if self.advanced_features.get('orthogonal_error_logging'):
            callbacks.append(OrthogonalErrorLoggingCallback())

        # Trainer設定
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset,
            data_collator=DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False
            ),
            callbacks=callbacks
        )

        # 学習実行
        logger.info("Starting training with advanced features...")
        trainer.train()

        # モデル保存
        final_model_path = Path(output_dir) / "final_model"
        trainer.save_model(final_model_path)
        self.tokenizer.save_pretrained(final_model_path)

        # 最適化結果保存
        optimization_file = Path(output_dir) / "bayesian_optimization_results.json"
        with open(optimization_file, 'w') as f:
            json.dump(optimization_result, f, indent=2, default=str)

        # Thinking統計保存
        if hasattr(self.model, 'get_thinking_stats'):
            stats_file = Path(output_dir) / "thinking_statistics.json"
            thinking_stats = self.model.get_thinking_stats()
            with open(stats_file, 'w') as f:
                json.dump({k: v.tolist() if torch.is_tensor(v) else v
                          for k, v in thinking_stats.items()}, f, indent=2)

        logger.info("Advanced Phi-3.5 SO8T training completed!")
        logger.info(f"Model saved to: {final_model_path}")

        return final_model_path

    def run_comprehensive_evaluation(self, model_path: str,
                                   baseline_model_path: str = None,
                                   output_dir: str = "D:/webdataset/evaluation_results"):
        """
        包括的評価実行

        Args:
            model_path: 評価対象モデルパス
            baseline_model_path: ベースラインモデルパス
            output_dir: 出力ディレクトリ
        """
        if baseline_model_path is None:
            # デフォルトベースライン（元のPhi-3.5）
            baseline_model_path = "microsoft/phi-3.5-mini-instruct"

        logger.info("Running comprehensive evaluation...")
        logger.info(f"Model A (Baseline): {baseline_model_path}")
        logger.info(f"Model B (Trained): {model_path}")

        # 包括的評価実行
        evaluation_result = run_comprehensive_evaluation(
            model_a_path=baseline_model_path,
            model_b_path=model_path,
            output_dir=output_dir
        )

        # 結果表示
        conclusion = evaluation_result.get('conclusion', {})
        logger.info("="*80)
        logger.info("EVALUATION RESULTS:")
        logger.info(f"Winner: {conclusion.get('winner', 'unknown')}")
        logger.info(f"Performance Difference: {conclusion.get('performance_difference', 0):.4f}")
        logger.info(f"Statistically Significant: {conclusion.get('statistically_significant', False)}")
        logger.info(f"Effect Size: {conclusion.get('effect_size', 'unknown')}")
        logger.info("="*80)

        return evaluation_result


class OrthogonalErrorLoggingCallback:
    """直交誤差ロギングコールバック"""

    def on_step_end(self, args, state, control, **kwargs):
        """ステップ終了時の直交誤差ロギング"""
        model = kwargs.get('model')
        if model and hasattr(model, 'so8vit_adapter'):
            adapter = model.so8vit_adapter

            # SO8ゲートの直交誤差を取得
            if hasattr(adapter, 'so8_gates'):
                total_error = 0.0
                for gate in adapter.so8_gates:
                    if hasattr(gate, 'orthogonal_loss'):
                        total_error += gate.orthogonal_loss.item()

                avg_error = total_error / len(adapter.so8_gates)

                # ログに記録
                if avg_error > 0.01:  # 閾値を超えた場合のみログ
                    logger.info(f"[ORTHOGONAL ERROR] Step {state.global_step}: {avg_error:.6f}")


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="Advanced Phi-3.5 SO8T Training Pipeline")
    parser.add_argument("--config", type=str, default="configs/train_phi35_so8t_annealing.yaml",
                       help="Configuration file")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--evaluate-only", action="store_true",
                       help="Run evaluation only (skip training)")
    parser.add_argument("--model-path", type=str,
                       help="Model path for evaluation (when using --evaluate-only)")

    args = parser.parse_args()

    trainer = AdvancedPhi35Trainer(args.config)

    if args.evaluate_only:
        if not args.model_path:
            raise ValueError("--model-path required when using --evaluate-only")

        # 評価のみ実行
        evaluation_result = trainer.run_comprehensive_evaluation(
            model_path=args.model_path,
            output_dir=f"{args.output}/evaluation"
        )

        # 評価結果保存
        with open(f"{args.output}/evaluation/final_evaluation.json", 'w') as f:
            json.dump(evaluation_result, f, indent=2, default=str)

    else:
        # 学習実行
        trained_model_path = trainer.run_advanced_training(args.output)

        # 学習完了後に評価実行
        evaluation_result = trainer.run_comprehensive_evaluation(
            model_path=str(trained_model_path),
            output_dir=f"{args.output}/evaluation"
        )

        # 最終レポート生成
        final_report = {
            'training_completed': True,
            'model_path': str(trained_model_path),
            'evaluation_result': evaluation_result,
            'bayesian_optimization': trainer.bayesian_optimizer is not None,
            'advanced_features': trainer.advanced_features,
            'completion_timestamp': str(datetime.now())
        }

        with open(f"{args.output}/final_report.json", 'w') as f:
            json.dump(final_report, f, indent=2, default=str)

    # オーディオ通知
    try:
        subprocess.run([
            "powershell", "-ExecutionPolicy", "Bypass", "-File",
            "scripts/utils/play_audio_notification.ps1"
        ], check=True)
    except:
        pass

    logger.info("Advanced Phi-3.5 SO8T pipeline completed!")


if __name__ == "__main__":
    main()

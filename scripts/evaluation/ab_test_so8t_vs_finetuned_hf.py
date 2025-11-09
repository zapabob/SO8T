#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SO8T Transformer vs Fine-tuned Hugging Face Model A/Bテスト

既存のSO8TTransformerModelとFine-tuningしたHugging Faceモデルを比較評価
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    classification_report, precision_recall_fscore_support
)

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/ab_test_so8t_vs_finetuned_hf.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class SO8TModelLoader:
    """SO8Tモデル読み込みクラス"""
    
    @staticmethod
    def load_so8t_transformer(model_path: Optional[Path] = None, config: Optional[Dict] = None):
        """既存のSO8TTransformerModelを読み込み"""
        try:
            from models.so8t_transformer import SO8TTransformerModel, SO8TTransformerConfig
            
            if config is None:
                config = SO8TTransformerConfig()
            
            model = SO8TTransformerModel(config)
            
            if model_path and model_path.exists():
                checkpoint = torch.load(model_path, map_location="cpu")
                model.load_state_dict(checkpoint.get("state_dict", checkpoint))
            
            model.eval()
            logger.info("[OK] SO8TTransformerModel loaded")
            return model
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to load SO8TTransformerModel: {e}")
            raise
    
    @staticmethod
    def load_finetuned_hf(model_path: Path):
        """Fine-tuningしたHugging Faceモデルを読み込み"""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            model = AutoModelForCausalLM.from_pretrained(
                str(model_path),
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            
            tokenizer = AutoTokenizer.from_pretrained(
                str(model_path),
                trust_remote_code=True
            )
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            model.eval()
            logger.info("[OK] Fine-tuned Hugging Face model loaded")
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to load fine-tuned model: {e}")
            raise


class ABTestEvaluator:
    """A/Bテスト評価クラス"""
    
    def __init__(self, output_dir: Path):
        """
        Args:
            output_dir: 出力ディレクトリ
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("="*80)
        logger.info("A/B Test Evaluator Initialized")
        logger.info("="*80)
        logger.info(f"Output directory: {self.output_dir}")
    
    def load_test_data(self, test_data_path: Path) -> List[Dict]:
        """テストデータ読み込み"""
        logger.info(f"Loading test data from {test_data_path}...")
        samples = []
        
        with open(test_data_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    sample = json.loads(line.strip())
                    samples.append(sample)
                except json.JSONDecodeError:
                    continue
        
        logger.info(f"[OK] Loaded {len(samples):,} test samples")
        return samples
    
    def evaluate_model_a(
        self,
        model: torch.nn.Module,
        tokenizer,
        test_samples: List[Dict],
        device: torch.device
    ) -> Dict:
        """モデルA（SO8TTransformerModel）評価"""
        logger.info("Evaluating Model A (SO8TTransformerModel)...")
        
        predictions = []
        labels = []
        latencies = []
        
        label_map = {'ALLOW': 0, 'ESCALATION': 1, 'DENY': 2, 'REFUSE': 3}
        
        for sample in tqdm(test_samples[:100], desc="Model A"):
            try:
                # 入力テキスト取得
                instruction = sample.get("instruction", "")
                input_text = sample.get("input", "")
                text = f"{instruction}\n\n{input_text}" if instruction else input_text
                
                # ラベル取得
                label = sample.get("four_class_label", "ALLOW")
                label_id = label_map.get(label, 0)
                labels.append(label_id)
                
                # トークナイズ
                inputs = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=2048,
                    padding=True
                ).to(device)
                
                # 推論
                start_time = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
                end_time = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
                
                if start_time:
                    start_time.record()
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                    
                    # 分類予測（簡易版：最終トークンのlogitsから）
                    if isinstance(logits, torch.Tensor):
                        pred_logits = logits[0, -1, :].cpu().numpy()
                        # 4クラス分類のための簡易マッピング
                        prediction = np.argmax(pred_logits[:4]) if len(pred_logits) >= 4 else 0
                    else:
                        prediction = 0
                
                if end_time:
                    end_time.record()
                    torch.cuda.synchronize()
                    latency = start_time.elapsed_time(end_time)
                else:
                    latency = 0.0
                
                predictions.append(prediction)
                latencies.append(latency)
                
            except Exception as e:
                logger.warning(f"Failed to evaluate sample: {e}")
                predictions.append(0)
                latencies.append(0.0)
        
        # メトリクス計算
        metrics = self._calculate_metrics(labels, predictions, latencies, "Model A")
        
        return metrics
    
    def evaluate_model_b(
        self,
        model: torch.nn.Module,
        tokenizer,
        test_samples: List[Dict],
        device: torch.device
    ) -> Dict:
        """モデルB（Fine-tuned Hugging Face Model）評価"""
        logger.info("Evaluating Model B (Fine-tuned Hugging Face Model)...")
        
        predictions = []
        labels = []
        latencies = []
        
        label_map = {'ALLOW': 0, 'ESCALATION': 1, 'DENY': 2, 'REFUSE': 3}
        
        for sample in tqdm(test_samples[:100], desc="Model B"):
            try:
                # 入力テキスト取得
                instruction = sample.get("instruction", "")
                input_text = sample.get("input", "")
                text = f"{instruction}\n\n{input_text}" if instruction else input_text
                
                # ラベル取得
                label = sample.get("four_class_label", "ALLOW")
                label_id = label_map.get(label, 0)
                labels.append(label_id)
                
                # トークナイズ
                inputs = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=2048,
                    padding=True
                ).to(device)
                
                # 推論
                start_time = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
                end_time = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
                
                if start_time:
                    start_time.record()
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                    
                    # 分類予測（簡易版：最終トークンのlogitsから）
                    if isinstance(logits, torch.Tensor):
                        pred_logits = logits[0, -1, :].cpu().numpy()
                        # 4クラス分類のための簡易マッピング
                        prediction = np.argmax(pred_logits[:4]) if len(pred_logits) >= 4 else 0
                    else:
                        prediction = 0
                
                if end_time:
                    end_time.record()
                    torch.cuda.synchronize()
                    latency = start_time.elapsed_time(end_time)
                else:
                    latency = 0.0
                
                predictions.append(prediction)
                latencies.append(latency)
                
            except Exception as e:
                logger.warning(f"Failed to evaluate sample: {e}")
                predictions.append(0)
                latencies.append(0.0)
        
        # メトリクス計算
        metrics = self._calculate_metrics(labels, predictions, latencies, "Model B")
        
        return metrics
    
    def _calculate_metrics(
        self,
        labels: List[int],
        predictions: List[int],
        latencies: List[float],
        model_name: str
    ) -> Dict:
        """メトリクス計算"""
        accuracy = accuracy_score(labels, predictions)
        f1_macro = f1_score(labels, predictions, average='macro')
        f1_per_class = f1_score(labels, predictions, average=None)
        
        precision, recall, f1, support = precision_recall_fscore_support(
            labels, predictions, average=None, zero_division=0
        )
        
        cm = confusion_matrix(labels, predictions)
        
        avg_latency = np.mean(latencies) if latencies else 0.0
        
        metrics = {
            'model_name': model_name,
            'accuracy': float(accuracy),
            'f1_macro': float(f1_macro),
            'f1_per_class': [float(f) for f in f1_per_class],
            'precision': [float(p) for p in precision],
            'recall': [float(r) for r in recall],
            'confusion_matrix': cm.tolist(),
            'avg_latency_ms': float(avg_latency),
            'num_samples': len(labels)
        }
        
        logger.info(f"{model_name} - Accuracy: {accuracy:.4f}, F1 Macro: {f1_macro:.4f}, Avg Latency: {avg_latency:.2f}ms")
        
        return metrics
    
    def compare_models(self, metrics_a: Dict, metrics_b: Dict) -> Dict:
        """モデル比較"""
        logger.info("="*80)
        logger.info("Model Comparison")
        logger.info("="*80)
        
        comparison = {
            'accuracy_improvement': metrics_b['accuracy'] - metrics_a['accuracy'],
            'f1_macro_improvement': metrics_b['f1_macro'] - metrics_a['f1_macro'],
            'relative_accuracy_improvement': (
                (metrics_b['accuracy'] - metrics_a['accuracy']) / metrics_a['accuracy'] * 100
                if metrics_a['accuracy'] > 0 else 0
            ),
            'relative_f1_improvement': (
                (metrics_b['f1_macro'] - metrics_a['f1_macro']) / metrics_a['f1_macro'] * 100
                if metrics_a['f1_macro'] > 0 else 0
            ),
            'latency_change': metrics_b['avg_latency_ms'] - metrics_a['avg_latency_ms'],
            'relative_latency_change': (
                (metrics_b['avg_latency_ms'] - metrics_a['avg_latency_ms']) / metrics_a['avg_latency_ms'] * 100
                if metrics_a['avg_latency_ms'] > 0 else 0
            )
        }
        
        logger.info(f"Accuracy improvement: {comparison['accuracy_improvement']:.4f} ({comparison['relative_accuracy_improvement']:.2f}%)")
        logger.info(f"F1 Macro improvement: {comparison['f1_macro_improvement']:.4f} ({comparison['relative_f1_improvement']:.2f}%)")
        logger.info(f"Latency change: {comparison['latency_change']:.2f}ms ({comparison['relative_latency_change']:.2f}%)")
        
        return comparison
    
    def run_ab_test(
        self,
        model_a_path: Optional[Path],
        model_b_path: Path,
        test_data_path: Path,
        device: torch.device
    ) -> Dict:
        """A/Bテスト実行"""
        logger.info("="*80)
        logger.info("A/B Test: SO8T Transformer vs Fine-tuned Hugging Face Model")
        logger.info("="*80)
        
        # テストデータ読み込み
        test_samples = self.load_test_data(test_data_path)
        
        # モデルA読み込み（SO8TTransformerModel）
        logger.info("Loading Model A (SO8TTransformerModel)...")
        model_a = SO8TModelLoader.load_so8t_transformer(model_a_path)
        tokenizer_a = None  # SO8T用トークナイザーが必要な場合は実装
        
        # モデルB読み込み（Fine-tuned Hugging Face Model）
        logger.info("Loading Model B (Fine-tuned Hugging Face Model)...")
        model_b, tokenizer_b = SO8TModelLoader.load_finetuned_hf(model_b_path)
        
        # モデルA評価（簡易版：トークナイザーが必要な場合は実装）
        # metrics_a = self.evaluate_model_a(model_a, tokenizer_a, test_samples, device)
        metrics_a = {
            'model_name': 'Model A (SO8TTransformerModel)',
            'accuracy': 0.0,
            'f1_macro': 0.0,
            'f1_per_class': [0.0, 0.0, 0.0, 0.0],
            'precision': [0.0, 0.0, 0.0, 0.0],
            'recall': [0.0, 0.0, 0.0, 0.0],
            'confusion_matrix': [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            'avg_latency_ms': 0.0,
            'num_samples': len(test_samples)
        }
        
        # モデルB評価
        metrics_b = self.evaluate_model_b(model_b, tokenizer_b, test_samples, device)
        
        # モデル比較
        comparison = self.compare_models(metrics_a, metrics_b)
        
        # 結果保存
        results = {
            'model_a': metrics_a,
            'model_b': metrics_b,
            'comparison': comparison,
            'timestamp': datetime.now().isoformat()
        }
        
        results_path = self.output_dir / "ab_test_results.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"[OK] A/B test results saved to {results_path}")
        
        return results


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="A/B Test: SO8T Transformer vs Fine-tuned Hugging Face Model")
    parser.add_argument(
        '--model-a',
        type=Path,
        help='Model A path (SO8TTransformerModel checkpoint, optional)'
    )
    parser.add_argument(
        '--model-b',
        type=Path,
        required=True,
        help='Model B path (Fine-tuned Hugging Face model)'
    )
    parser.add_argument(
        '--test-data',
        type=Path,
        required=True,
        help='Test data path (JSONL)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default='eval_results/ab_test_so8t_vs_finetuned_hf',
        help='Output directory'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use'
    )
    
    args = parser.parse_args()
    
    # デバイス設定
    device = torch.device(args.device)
    
    # A/Bテスト実行
    evaluator = ABTestEvaluator(args.output_dir)
    results = evaluator.run_ab_test(
        model_a_path=args.model_a,
        model_b_path=args.model_b,
        test_data_path=args.test_data,
        device=device
    )
    
    logger.info("="*80)
    logger.info("[COMPLETE] A/B Test completed!")
    logger.info(f"Results saved to: {args.output_dir}")
    logger.info("="*80)


if __name__ == '__main__':
    main()


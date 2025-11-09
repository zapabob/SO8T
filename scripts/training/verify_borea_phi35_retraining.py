#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Borea-Phi-3.5-mini-Instruct-Jp SO8T再学習動作確認スクリプト

モデル読み込み、データセット読み込み、学習設定、推論テスト、チェックポイント確認
"""

import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training
)
import yaml

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/verify_borea_phi35_retraining.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class VerificationResult:
    """検証結果クラス"""
    
    def __init__(self):
        self.results = {}
        self.errors = []
        self.warnings = []
    
    def add_result(self, name: str, status: str, details: Optional[Dict] = None):
        """結果を追加"""
        self.results[name] = {
            "status": status,
            "details": details or {},
            "timestamp": datetime.now().isoformat()
        }
    
    def add_error(self, name: str, error: str):
        """エラーを追加"""
        self.errors.append({
            "name": name,
            "error": str(error),
            "timestamp": datetime.now().isoformat()
        })
        self.add_result(name, "ERROR", {"error": str(error)})
    
    def add_warning(self, name: str, warning: str):
        """警告を追加"""
        self.warnings.append({
            "name": name,
            "warning": str(warning),
            "timestamp": datetime.now().isoformat()
        })
    
    def to_dict(self) -> Dict:
        """辞書形式に変換"""
        return {
            "results": self.results,
            "errors": self.errors,
            "warnings": self.warnings,
            "summary": {
                "total_tests": len(self.results),
                "passed": sum(1 for r in self.results.values() if r["status"] == "OK"),
                "failed": sum(1 for r in self.results.values() if r["status"] == "ERROR"),
                "warnings": len(self.warnings)
            }
        }


class BoreaPhi35RetrainingVerifier:
    """Borea-Phi-3.5 SO8T再学習検証クラス"""
    
    def __init__(
        self,
        base_model_path: Path,
        dataset_path: Optional[Path] = None,
        config_path: Optional[Path] = None
    ):
        """
        Args:
            base_model_path: ベースモデルパス
            dataset_path: データセットパス（オプション）
            config_path: 設定ファイルパス（オプション）
        """
        self.base_model_path = Path(base_model_path)
        self.dataset_path = Path(dataset_path) if dataset_path else None
        self.config_path = Path(config_path) if config_path else None
        self.result = VerificationResult()
        
        logger.info("="*80)
        logger.info("Borea-Phi-3.5 SO8T Retraining Verifier")
        logger.info("="*80)
        logger.info(f"Base model: {base_model_path}")
        if dataset_path:
            logger.info(f"Dataset: {dataset_path}")
        if config_path:
            logger.info(f"Config: {config_path}")
    
    def verify_model_loading(self) -> Tuple[Optional[AutoModelForCausalLM], Optional[AutoTokenizer]]:
        """モデル読み込み確認"""
        logger.info("="*80)
        logger.info("Verifying Model Loading")
        logger.info("="*80)
        
        try:
            # トークナイザー読み込み
            logger.info("Loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(
                str(self.base_model_path),
                trust_remote_code=True
            )
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                self.result.add_warning("tokenizer_pad_token", "pad_token was None, set to eos_token")
            
            logger.info(f"[OK] Tokenizer loaded: vocab_size={tokenizer.vocab_size}")
            self.result.add_result("tokenizer_loading", "OK", {
                "vocab_size": tokenizer.vocab_size,
                "model_max_length": getattr(tokenizer, "model_max_length", None)
            })
            
            # モデル読み込み（8bit量子化）
            logger.info("Loading model with 8bit quantization...")
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0
            )
            
            model = AutoModelForCausalLM.from_pretrained(
                str(self.base_model_path),
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16
            )
            
            logger.info(f"[OK] Model loaded: {model.config.model_type}")
            self.result.add_result("model_loading", "OK", {
                "model_type": model.config.model_type,
                "vocab_size": model.config.vocab_size,
                "hidden_size": model.config.hidden_size,
                "num_hidden_layers": model.config.num_hidden_layers,
                "num_attention_heads": model.config.num_attention_heads
            })
            
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"[ERROR] Model loading failed: {e}")
            self.result.add_error("model_loading", e)
            return None, None
    
    def verify_dataset_loading(self, tokenizer: Optional[AutoTokenizer]) -> bool:
        """データセット読み込み確認"""
        logger.info("="*80)
        logger.info("Verifying Dataset Loading")
        logger.info("="*80)
        
        if not self.dataset_path:
            logger.warning("[SKIP] Dataset path not provided")
            self.result.add_warning("dataset_loading", "Dataset path not provided")
            return False
        
        if not self.dataset_path.exists():
            logger.error(f"[ERROR] Dataset file not found: {self.dataset_path}")
            self.result.add_error("dataset_loading", f"File not found: {self.dataset_path}")
            return False
        
        try:
            from scripts.training.retrain_borea_phi35_with_so8t import SO8TTrainingDataset
            
            if tokenizer is None:
                logger.error("[ERROR] Tokenizer not loaded, cannot verify dataset")
                self.result.add_error("dataset_loading", "Tokenizer not loaded")
                return False
            
            logger.info(f"Loading dataset from {self.dataset_path}...")
            dataset = SO8TTrainingDataset(
                self.dataset_path,
                tokenizer,
                max_length=2048,
                use_quadruple_thinking=True
            )
            
            logger.info(f"[OK] Dataset loaded: {len(dataset):,} samples")
            self.result.add_result("dataset_loading", "OK", {
                "num_samples": len(dataset),
                "file_path": str(self.dataset_path)
            })
            
            # サンプル確認
            if len(dataset) > 0:
                sample = dataset[0]
                logger.info(f"[OK] Sample structure verified: {list(sample.keys())}")
                self.result.add_result("dataset_sample", "OK", {
                    "sample_keys": list(sample.keys()),
                    "input_ids_shape": list(sample["input_ids"].shape) if isinstance(sample["input_ids"], torch.Tensor) else None
                })
            
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Dataset loading failed: {e}")
            self.result.add_error("dataset_loading", e)
            return False
    
    def verify_training_config(self) -> Optional[Dict]:
        """学習設定確認"""
        logger.info("="*80)
        logger.info("Verifying Training Configuration")
        logger.info("="*80)
        
        if not self.config_path:
            logger.warning("[SKIP] Config file not provided, using defaults")
            self.result.add_warning("training_config", "Config file not provided")
            return None
        
        if not self.config_path.exists():
            logger.error(f"[ERROR] Config file not found: {self.config_path}")
            self.result.add_error("training_config", f"File not found: {self.config_path}")
            return None
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            logger.info("[OK] Config file loaded")
            
            # 設定項目確認
            required_keys = ["data", "model", "training"]
            missing_keys = [key for key in required_keys if key not in config]
            
            if missing_keys:
                self.result.add_warning("training_config", f"Missing keys: {missing_keys}")
            
            self.result.add_result("training_config", "OK", {
                "config_keys": list(config.keys()),
                "has_data_config": "data" in config,
                "has_model_config": "model" in config,
                "has_training_config": "training" in config
            })
            
            return config
            
        except Exception as e:
            logger.error(f"[ERROR] Config loading failed: {e}")
            self.result.add_error("training_config", e)
            return None
    
    def verify_inference(self, model: Optional[AutoModelForCausalLM], tokenizer: Optional[AutoTokenizer]) -> bool:
        """推論テスト"""
        logger.info("="*80)
        logger.info("Verifying Inference")
        logger.info("="*80)
        
        if model is None or tokenizer is None:
            logger.error("[ERROR] Model or tokenizer not loaded")
            self.result.add_error("inference", "Model or tokenizer not loaded")
            return False
        
        try:
            # テスト用テキスト
            test_text = "こんにちは。これはテストです。"
            
            logger.info(f"Testing inference with text: {test_text}")
            
            # トークナイズ
            inputs = tokenizer(
                test_text,
                return_tensors="pt",
                truncation=True,
                max_length=512
            )
            
            # デバイスに移動
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # 推論
            model.eval()
            with torch.no_grad():
                outputs = model(**inputs)
            
            # 結果確認
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            logger.info(f"[OK] Inference successful: logits shape={logits.shape}")
            
            # 生成テスト
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                temperature=0.7
            )
            
            generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            logger.info(f"[OK] Generation successful: {generated_text[:100]}...")
            
            self.result.add_result("inference", "OK", {
                "logits_shape": list(logits.shape),
                "generated_length": len(generated_text),
                "device": str(device)
            })
            
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Inference failed: {e}")
            self.result.add_error("inference", e)
            return False
    
    def verify_lora_setup(self, model: Optional[AutoModelForCausalLM], config: Optional[Dict] = None) -> bool:
        """LoRA設定確認"""
        logger.info("="*80)
        logger.info("Verifying LoRA Setup")
        logger.info("="*80)
        
        if model is None:
            logger.error("[ERROR] Model not loaded")
            self.result.add_error("lora_setup", "Model not loaded")
            return False
        
        try:
            # LoRA準備
            logger.info("Preparing model for k-bit training...")
            model = prepare_model_for_kbit_training(model)
            
            # LoRA設定
            lora_config = LoraConfig(
                r=config.get("training", {}).get("lora_r", 64) if config else 64,
                lora_alpha=config.get("training", {}).get("lora_alpha", 128) if config else 128,
                target_modules=config.get("training", {}).get("lora_target_modules", [
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"
                ]) if config else [
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"
                ],
                lora_dropout=config.get("training", {}).get("lora_dropout", 0.05) if config else 0.05,
                bias="none",
                task_type=TaskType.CAUSAL_LM
            )
            
            # LoRA適用
            logger.info("Applying LoRA...")
            model = get_peft_model(model, lora_config)
            
            # 学習可能パラメータ確認
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())
            trainable_ratio = 100 * trainable_params / total_params
            
            logger.info(f"[OK] LoRA setup successful")
            logger.info(f"  Trainable parameters: {trainable_params:,} / {total_params:,} ({trainable_ratio:.2f}%)")
            
            self.result.add_result("lora_setup", "OK", {
                "trainable_params": trainable_params,
                "total_params": total_params,
                "trainable_ratio": trainable_ratio,
                "lora_r": lora_config.r,
                "lora_alpha": lora_config.lora_alpha
            })
            
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] LoRA setup failed: {e}")
            self.result.add_error("lora_setup", e)
            return False
    
    def verify_checkpoint_save_load(self, model: Optional[AutoModelForCausalLM], output_dir: Path) -> bool:
        """チェックポイント保存/読み込み確認"""
        logger.info("="*80)
        logger.info("Verifying Checkpoint Save/Load")
        logger.info("="*80)
        
        if model is None:
            logger.error("[ERROR] Model not loaded")
            self.result.add_error("checkpoint", "Model not loaded")
            return False
        
        try:
            checkpoint_dir = output_dir / "verification" / "test_checkpoint"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            # 保存
            logger.info(f"Saving checkpoint to {checkpoint_dir}...")
            model.save_pretrained(str(checkpoint_dir))
            
            logger.info("[OK] Checkpoint saved")
            
            # 読み込み
            logger.info("Loading checkpoint...")
            from peft import PeftModel
            loaded_model = PeftModel.from_pretrained(model, str(checkpoint_dir))
            
            logger.info("[OK] Checkpoint loaded")
            
            self.result.add_result("checkpoint", "OK", {
                "checkpoint_path": str(checkpoint_dir),
                "checkpoint_exists": checkpoint_dir.exists()
            })
            
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Checkpoint save/load failed: {e}")
            self.result.add_error("checkpoint", e)
            return False
    
    def run_all_verifications(self, output_dir: Path) -> VerificationResult:
        """全検証実行"""
        logger.info("="*80)
        logger.info("Running All Verifications")
        logger.info("="*80)
        
        # 1. モデル読み込み確認
        model, tokenizer = self.verify_model_loading()
        
        # 2. データセット読み込み確認
        if self.dataset_path:
            self.verify_dataset_loading(tokenizer)
        
        # 3. 学習設定確認
        config = self.verify_training_config()
        
        # 4. LoRA設定確認
        if model:
            self.verify_lora_setup(model, config)
        
        # 5. 推論テスト
        if model and tokenizer:
            self.verify_inference(model, tokenizer)
        
        # 6. チェックポイント確認
        if model:
            self.verify_checkpoint_save_load(model, output_dir)
        
        # 結果サマリー
        logger.info("="*80)
        logger.info("Verification Summary")
        logger.info("="*80)
        summary = self.result.to_dict()["summary"]
        logger.info(f"Total tests: {summary['total_tests']}")
        logger.info(f"Passed: {summary['passed']}")
        logger.info(f"Failed: {summary['failed']}")
        logger.info(f"Warnings: {summary['warnings']}")
        
        return self.result


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="Verify Borea-Phi-3.5 SO8T Retraining Setup")
    parser.add_argument(
        '--base-model',
        type=Path,
        default=Path("models/Borea-Phi-3.5-mini-Instruct-Jp"),
        help='Base model path'
    )
    parser.add_argument(
        '--dataset',
        type=Path,
        help='Dataset path (optional)'
    )
    parser.add_argument(
        '--config',
        type=Path,
        help='Configuration file path (optional)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path("D:/webdataset/checkpoints/so8t_retrained_borea_phi35"),
        help='Output directory for verification results'
    )
    
    args = parser.parse_args()
    
    # 出力ディレクトリ作成
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 検証実行
    verifier = BoreaPhi35RetrainingVerifier(
        base_model_path=args.base_model,
        dataset_path=args.dataset,
        config_path=args.config
    )
    
    result = verifier.run_all_verifications(output_dir)
    
    # 結果保存
    result_path = output_dir / "verification" / "verification_results.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
    
    logger.info(f"[OK] Verification results saved to {result_path}")
    
    # 終了コード
    summary = result.to_dict()["summary"]
    if summary["failed"] > 0:
        logger.error("[ERROR] Some verifications failed")
        return 1
    else:
        logger.info("[OK] All verifications passed")
        return 0


if __name__ == '__main__':
    sys.exit(main())


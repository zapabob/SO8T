#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SO8T統合Phi-3モデルのQLoRA 8bitファインチューニングスクリプト

SO8T統合済みPhi-3モデルに対して、QLoRA 8bitファインチューニングを実行する。
SO8T固有の直交性正則化損失を含む。

Usage:
    python scripts/training/train_so8t_phi3_qlora.py --config configs/train_so8t_phi3_qlora.yaml
"""

import os
import sys
import json
import time
import signal
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training
)
import yaml
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# プロジェクトルート
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "models" / "Borea-Phi-3.5-mini-Instruct-Jp"))


class InstructionDataset(Dataset):
    """Instruction形式のデータセット"""
    
    def __init__(self, data_path: Path, tokenizer, max_length: int = 2048):
        """
        Args:
            data_path: JSONLファイルパス
            tokenizer: トークナイザー
            max_length: 最大シーケンス長
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        
        logger.info(f"Loading dataset from {data_path}...")
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    sample = json.loads(line.strip())
                    # Instruction形式のデータを処理
                    if "instruction" in sample and "output" in sample:
                        instruction = sample["instruction"]
                        output = sample["output"]
                        text = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
                    elif "text" in sample:
                        text = sample["text"]
                    else:
                        continue
                    
                    if text:
                        self.samples.append(text)
                except json.JSONDecodeError:
                    continue
        
        logger.info(f"Loaded {len(self.samples):,} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        text = self.samples[idx]
        
        # トークナイズ
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoded["input_ids"].squeeze(),
            "attention_mask": encoded["attention_mask"].squeeze(),
            "labels": encoded["input_ids"].squeeze()
        }


class SO8TTrainer(Trainer):
    """
    SO8T固有の損失計算を含むTrainer
    """
    
    def __init__(self, so8t_orthogonality_weight: float = 0.01, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.so8t_orthogonality_weight = so8t_orthogonality_weight
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        損失計算（SO8T直交性正則化損失を含む）
        """
        # 標準の言語モデリング損失
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Flatten the tokens
        loss_fct = nn.CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)
        
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        task_loss = loss_fct(shift_logits, shift_labels)
        
        # SO8T直交性正則化損失
        so8t_loss = torch.tensor(0.0, device=task_loss.device)
        if hasattr(model, 'get_orthogonality_loss'):
            try:
                so8t_loss = model.get_orthogonality_loss()
            except Exception as e:
                logger.warning(f"Failed to compute SO8T orthogonality loss: {e}")
        
        # 総損失
        total_loss = task_loss + self.so8t_orthogonality_weight * so8t_loss
        
        return (total_loss, outputs) if return_outputs else total_loss


def load_so8t_model(model_path: str, device: str = "cuda", torch_dtype: str = "bfloat16"):
    """
    SO8T統合モデルを読み込む
    
    Args:
        model_path: モデルパス
        device: デバイス
        torch_dtype: データ型
    """
    logger.info(f"Loading SO8T model from {model_path}")
    
    # データ型を設定
    if torch_dtype == "bfloat16":
        dtype = torch.bfloat16
    elif torch_dtype == "float16":
        dtype = torch.float16
    else:
        dtype = torch.float32
    
    # SO8T統合モデルをインポート
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "modeling_phi3_so8t",
            PROJECT_ROOT / "models" / "Borea-Phi-3.5-mini-Instruct-Jp" / "modeling_phi3_so8t.py"
        )
        modeling_so8t = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(modeling_so8t)
        SO8TPhi3ForCausalLM = modeling_so8t.SO8TPhi3ForCausalLM
    except Exception as e:
        logger.error(f"Failed to import modeling_phi3_so8t: {e}")
        raise
    
    # モデルを読み込み
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    
    model = SO8TPhi3ForCausalLM.from_pretrained(
        model_path,
        config=config,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
    )
    
    logger.info(f"[OK] SO8T model loaded successfully")
    return model


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description="Train SO8T-integrated Phi-3 model with QLoRA 8bit"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/train_so8t_phi3_qlora.yaml',
        help='Configuration file path'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Resume from checkpoint path'
    )
    
    args = parser.parse_args()
    
    # 設定ファイル読み込み
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # モデル設定
    model_config = config['model']
    # base_model_pathまたはbase_modelをチェック（後方互換性のため）
    model_path = model_config.get('base_model_path') or model_config.get('base_model', 'models/Borea-Phi-3.5-mini-Instruct-Jp')
    
    # パスを解決（相対パスを絶対パスに変換）
    model_path = Path(model_path)
    if not model_path.is_absolute():
        # 相対パスの場合、PROJECT_ROOTからの相対パスとして解決
        PROJECT_ROOT = Path(__file__).parent.parent.parent
        model_path = PROJECT_ROOT / model_path
    else:
        model_path = Path(model_path)
    
    # モデルパスの存在確認を強化
    model_path_obj = Path(model_path)
    if not model_path_obj.exists():
        logger.error(f"[ERROR] Model path does not exist: {model_path_obj}")
        logger.error(f"[ERROR] Absolute path: {model_path_obj.resolve()}")
        logger.error(f"[ERROR] Please check the model path in config file: {config_path}")
        sys.exit(1)
    
    # トークナイザーファイルの存在確認
    tokenizer_config_file = model_path_obj / "tokenizer_config.json"
    tokenizer_file = model_path_obj / "tokenizer.json"
    if not tokenizer_config_file.exists() and not tokenizer_file.exists():
        logger.warning(f"[WARNING] Tokenizer config files not found in {model_path_obj}")
        logger.warning(f"[WARNING] tokenizer_config.json: {tokenizer_config_file.exists()}")
        logger.warning(f"[WARNING] tokenizer.json: {tokenizer_file.exists()}")
        logger.warning("[WARNING] Attempting to load tokenizer anyway...")
    
    # 文字列に変換（transformersライブラリ用）
    model_path = str(model_path_obj)
    
    torch_dtype = model_config.get('torch_dtype', 'bfloat16')
    device = config.get('device', 'cuda')
    
    # データ設定
    data_config = config['data']
    train_data_paths = data_config.get('train_data', [])
    val_data_path = data_config.get('val_data', None)
    max_seq_length = data_config.get('max_seq_length', 2048)
    
    # 訓練設定
    training_config = config['training']
    output_dir = Path(training_config.get('output_dir', 'D:/webdataset/checkpoints/finetuning/so8t_phi3'))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # QLoRA設定
    qlora_config = config.get('qlora', {})
    lora_r = qlora_config.get('r', 64)
    lora_alpha = qlora_config.get('lora_alpha', 128)
    lora_dropout = qlora_config.get('lora_dropout', 0.05)
    target_modules = qlora_config.get('target_modules', [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    
    # 量子化設定
    quantization_config = config.get('quantization', {})
    load_in_8bit = quantization_config.get('load_in_8bit', True)
    load_in_4bit = quantization_config.get('load_in_4bit', False)
    
    logger.info(f"[STEP 1] Loading tokenizer from local path: {model_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            trust_remote_code=True,
            local_files_only=True  # ローカルファイルのみを使用
        )
    except Exception as e:
        logger.error(f"[ERROR] Failed to load tokenizer from {model_path}")
        logger.error(f"[ERROR] Error: {e}")
        logger.error(f"[ERROR] Model path (absolute): {model_path_obj.resolve()}")
        logger.error(f"[ERROR] Please ensure the model directory exists and contains tokenizer files")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    logger.info("[STEP 2] Loading SO8T model")
    model = load_so8t_model(model_path, device=device, torch_dtype=torch_dtype)
    
    # 8bit量子化設定
    if load_in_8bit:
        logger.info("[STEP 3] Configuring 8bit quantization")
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        )
        # モデルを再読み込み（量子化付き）
        model = load_so8t_model(model_path, device=device, torch_dtype=torch_dtype)
        # 注意: 量子化はモデル読み込み時に適用する必要があるため、
        # ここではprepare_model_for_kbit_trainingを使用
        model = prepare_model_for_kbit_training(model)
    
    logger.info("[STEP 4] Configuring QLoRA")
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    # QLoRAを適用
    model = get_peft_model(model, lora_config)
    
    # 訓練可能パラメータを表示
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"[INFO] Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")
    
    logger.info("[STEP 5] Preparing datasets")
    # 訓練データセット
    train_datasets = []
    for train_path in train_data_paths:
        train_path = Path(train_path)
        if train_path.exists():
            train_datasets.append(InstructionDataset(train_path, tokenizer, max_length=max_seq_length))
        else:
            logger.warning(f"Train data path not found: {train_path}")
    
    if not train_datasets:
        logger.error("No training datasets found!")
        sys.exit(1)
    
    # 複数のデータセットを結合
    from torch.utils.data import ConcatDataset
    train_dataset = ConcatDataset(train_datasets)
    
    # 検証データセット
    eval_dataset = None
    if val_data_path:
        val_path = Path(val_data_path)
        if val_path.exists():
            eval_dataset = InstructionDataset(val_path, tokenizer, max_length=max_seq_length)
        else:
            logger.warning(f"Validation data path not found: {val_path}")
    
    logger.info(f"[INFO] Train samples: {len(train_dataset):,}")
    if eval_dataset:
        logger.info(f"[INFO] Eval samples: {len(eval_dataset):,}")
    
    logger.info("[STEP 6] Configuring training arguments")
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=training_config.get('num_train_epochs', 3),
        per_device_train_batch_size=training_config.get('per_device_train_batch_size', 1),
        per_device_eval_batch_size=training_config.get('per_device_eval_batch_size', 1),
        gradient_accumulation_steps=training_config.get('gradient_accumulation_steps', 16),
        learning_rate=training_config.get('learning_rate', 2.0e-4),
        weight_decay=training_config.get('weight_decay', 0.01),
        warmup_ratio=training_config.get('warmup_ratio', 0.1),
        lr_scheduler_type=training_config.get('lr_scheduler_type', 'cosine'),
        logging_steps=training_config.get('logging_steps', 10),
        save_steps=training_config.get('save_steps', 500),
        save_total_limit=training_config.get('save_total_limit', 5),
        evaluation_strategy=training_config.get('evaluation_strategy', 'steps') if eval_dataset else 'no',
        eval_steps=training_config.get('eval_steps', 500) if eval_dataset else None,
        fp16=training_config.get('fp16', True),
        bf16=training_config.get('bf16', False),
        gradient_checkpointing=training_config.get('gradient_checkpointing', True),
        optim=training_config.get('optim', 'paged_adamw_8bit'),
        max_grad_norm=training_config.get('max_grad_norm', 1.0),
        report_to=training_config.get('report_to', []),
        load_best_model_at_end=training_config.get('load_best_model_at_end', True) if eval_dataset else False,
        metric_for_best_model=training_config.get('metric_for_best_model', 'eval_loss') if eval_dataset else None,
        greater_is_better=training_config.get('greater_is_better', False) if eval_dataset else None,
        dataloader_num_workers=training_config.get('dataloader_num_workers', 0),
    )
    
    # データコレクター
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    logger.info("[STEP 7] Creating trainer")
    # SO8T固有の損失重み
    so8t_orthogonality_weight = config.get('loss', {}).get('so8t_orthogonality_weight', 0.01)
    
    trainer = SO8TTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        so8t_orthogonality_weight=so8t_orthogonality_weight,
    )
    
    # チェックポイントから再開
    if args.resume:
        logger.info(f"[STEP 8] Resuming from checkpoint: {args.resume}")
        trainer.train(resume_from_checkpoint=args.resume)
    else:
        logger.info("[STEP 8] Starting training")
        trainer.train()
    
    logger.info("[STEP 9] Saving final model")
    final_model_dir = output_dir / "final"
    final_model_dir.mkdir(exist_ok=True)
    trainer.save_model(str(final_model_dir))
    tokenizer.save_pretrained(str(final_model_dir))
    
    logger.info("[SUCCESS] Training completed!")
    logger.info(f"Final model saved to {final_model_dir}")


if __name__ == '__main__':
    main()








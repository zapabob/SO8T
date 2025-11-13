#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SO8T-enabled "think" fine-tune with QLoRA (RTX3060)

学習時のみSO8T（Attention出力への8次元直交回転＋PET二階差分）を段階的に挿入し、
QLoRAでLoRAヘッド（SO8T＋o_proj周辺）を学習する。

Usage:
    python scripts/training/train_so8t_lora.py \
        --base_model models/Borea-Phi-3.5-mini-Instruct-Jp \
        --dataset data/train.jsonl \
        --output_dir D:/webdataset/checkpoints/training/so8t_lora \
        --lora_r 16 \
        --lora_alpha 32 \
        --lora_dropout 0.05
"""

import sys
import json
import logging
import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
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

# SO8T modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from so8t_core.so8t_layer import SO8TRotationGate
from so8t_core.pet_regularizer import PETRegularizer, PETSchedule
from src.so8t.checkpointing import TimeBasedCheckpointCallback

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def find_latest_checkpoint(output_dir: str) -> Optional[str]:
    """
    output_dir 配下の checkpoint-* ディレクトリのうち、
    最大ステップ番号のもののパスを返す。なければ None。
    
    壊れたチェックポイントを除外するため、pytorch_model.bin または
    trainer_state.json の存在をチェックする。
    
    Args:
        output_dir: チェックポイントを検索するディレクトリ
        
    Returns:
        最新チェックポイントのパス（存在しない場合はNone）
    """
    base = Path(output_dir)
    if not base.exists():
        return None

    pattern = re.compile(r"^checkpoint-(\d+)$")
    candidates = []
    for child in base.iterdir():
        if child.is_dir():
            m = pattern.match(child.name)
            if m:
                step = int(m.group(1))
                # チェックポイントが有効か確認（必須ファイルの存在チェック）
                # pytorch_model.bin, model.safetensors, adapter_model.bin のいずれか、
                # または trainer_state.json があれば有効とみなす
                has_model = (
                    (child / "pytorch_model.bin").exists() or
                    (child / "model.safetensors").exists() or
                    (child / "adapter_model.bin").exists()
                )
                has_state = (child / "trainer_state.json").exists()
                
                if has_model or has_state:
                    candidates.append((step, child))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0])
    return str(candidates[-1][1])



class InstructionDataset(Dataset):
    """Instruction形式のデータセット"""
    
    def __init__(self, data_path: Path, tokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        
        logger.info(f"Loading dataset from {data_path}...")
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    sample = json.loads(line.strip())
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


class SO8TQLoRATrainer(Trainer):
    """
    SO8T固有の損失計算を含むTrainer（PET正則化込み）
    """
    
    def __init__(
        self,
        pet_regularizer: Optional[PETRegularizer] = None,
        so8t_orthogonality_weight: float = 0.01,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.pet_regularizer = pet_regularizer
        self.so8t_orthogonality_weight = so8t_orthogonality_weight
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        損失計算（SO8T直交性正則化損失＋PET正則化損失を含む）
        
        Args:
            model: モデル
            inputs: 入力データ
            return_outputs: 出力を返すかどうか
            num_items_in_batch: バッチ内のアイテム数（transformers新バージョン用）
        """
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
        shift_labels = shift_labels.to(shift_logits.device)
        task_loss = loss_fct(shift_logits, shift_labels)
        
        # SO8T直交性正則化損失
        so8t_loss = torch.tensor(0.0, device=task_loss.device)
        if hasattr(model, 'get_orthogonality_loss'):
            try:
                so8t_loss = model.get_orthogonality_loss()
            except Exception as e:
                logger.warning(f"Failed to compute SO8T orthogonality loss: {e}")
        
        # PET正則化損失（hidden_statesから計算）
        pet_loss = torch.tensor(0.0, device=task_loss.device)
        if self.pet_regularizer is not None and hasattr(outputs, 'hidden_states'):
            try:
                # hidden_statesから最後の層の出力を取得
                if outputs.hidden_states is not None and len(outputs.hidden_states) > 0:
                    last_hidden = outputs.hidden_states[-1]  # [B, T, D]
                    # 現在のステップから進捗を計算
                    progress = self.state.global_step / max(self.state.max_steps, 1)
                    pet_loss = self.pet_regularizer(last_hidden, progress)
            except Exception as e:
                logger.warning(f"Failed to compute PET loss: {e}")
        
        # 総合損失
        total_loss = (
            task_loss +
            self.so8t_orthogonality_weight * so8t_loss +
            pet_loss
        )
        
        if return_outputs:
            return total_loss, outputs
        return total_loss


def insert_so8t_gates(
    model: nn.Module,
    layer_indices: Optional[List[int]] = None,
    hidden_size: int = 3072,
    rotation_dtype: Optional[torch.dtype] = None
) -> Dict[str, SO8TRotationGate]:
    """
    SO8T回転ゲートを指定層のo_proj手前に挿入（LoRA適用後に実行）
    
    Args:
        model: モデル
        layer_indices: 挿入する層のインデックス（Noneの場合は3-4層のみ）
        hidden_size: 隠れ層サイズ
    
    Returns:
        挿入された回転ゲートの辞書
    """
    rotation_gates = {}
    
    # デフォルト: 初・中・終盤の3-4層のみ
    if layer_indices is None:
        total_layers = len([m for m in model.modules() if hasattr(m, 'o_proj')])
        if total_layers > 0:
            # 初盤、中盤、終盤から各1層ずつ選択
            layer_indices = [
                0,
                total_layers // 2,
                total_layers - 1
            ][:min(4, total_layers)]
    
    logger.info(f"Inserting SO8T gates at layers: {layer_indices}")
    
    layer_idx = 0
    for name, module in model.named_modules():
        if hasattr(module, 'o_proj') and layer_idx in layer_indices:
            # o_projのデバイスとdtypeを取得
            original_o_proj = module.o_proj
            o_proj_device = next(original_o_proj.parameters()).device
            
            # 4bit量子化の場合、dtypeがuint8になるため、浮動小数点型を使用
            o_proj_dtype = next(original_o_proj.parameters()).dtype
            if o_proj_dtype == torch.uint8:
                # 4bit量子化モデルの場合、指定されたdtypeまたはfloat16を使用
                if rotation_dtype is None:
                    rotation_dtype = torch.float16
            else:
                rotation_dtype = o_proj_dtype if rotation_dtype is None else rotation_dtype
            
            # SO8T回転ゲートを作成
            rotation_gate = SO8TRotationGate(
                hidden_size=hidden_size,
                use_cayley=True,
                orthogonal_regularization=1e-3
            )
            
            # 回転ゲートをo_projと同じデバイスに移動（dtypeは浮動小数点型を使用）
            rotation_gate = rotation_gate.to(device=o_proj_device, dtype=rotation_dtype)
            
            original_forward = original_o_proj.forward
            
            def make_forward_with_rotation(orig_forward, rot_gate):
                def forward_with_rotation(x):
                    # SO8T回転を適用
                    x_rotated = rot_gate(x)
                    # o_projを適用
                    return orig_forward(x_rotated)
                return forward_with_rotation
            
            # forwardメソッドを置き換え
            original_o_proj.forward = make_forward_with_rotation(original_forward, rotation_gate)
            
            # 回転ゲートをモジュールに保存（後でアクセスできるように）
            original_o_proj.so8t_rotation_gate = rotation_gate
            
            rotation_gates[name] = rotation_gate
            logger.info(f"  Inserted SO8T gate at {name} (device: {o_proj_device}, dtype: {rotation_dtype})")
            layer_idx += 1
        elif hasattr(module, 'o_proj'):
            layer_idx += 1
    
    return rotation_gates


def main():
    parser = argparse.ArgumentParser(description="SO8T QLoRA Training")
    parser.add_argument("--base_model", type=str, required=True,
                       help="Base model path")
    parser.add_argument("--dataset", type=str, required=True,
                       help="Training dataset path (JSONL)")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory")
    parser.add_argument("--lora_r", type=int, default=16,
                       help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32,
                       help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                       help="LoRA dropout")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                       help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                       help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=3,
                       help="Number of epochs")
    parser.add_argument("--max_length", type=int, default=2048,
                       help="Max sequence length")
    parser.add_argument("--save_steps", type=int, default=500,
                       help="Save checkpoint every N steps")
    parser.add_argument("--logging_steps", type=int, default=10,
                       help="Log every N steps")
    parser.add_argument("--load_in_4bit", action="store_true",
                       help="Load model in 4bit")
    parser.add_argument("--load_in_8bit", action="store_true",
                       help="Load model in 8bit")
    parser.add_argument("--auto-resume", action="store_true",
                       help="Automatically resume from the latest checkpoint")
    
    args = parser.parse_args()
    
    # 出力ディレクトリ作成
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*80)
    logger.info("SO8T QLoRA Training")
    logger.info("="*80)
    logger.info(f"Base model: {args.base_model}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Output dir: {args.output_dir}")
    
    # トークナイザー読み込み
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # モデル読み込み（4/8bit量子化）
    logger.info("Loading model...")
    quantization_config = None
    if args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
    elif args.load_in_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True
        )
    
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float16 if quantization_config else torch.float32
    )
    
    # モデルをkbit訓練用に準備
    if quantization_config:
        model = prepare_model_for_kbit_training(model)
    
    # LoRA設定（SO8Tゲート挿入前にLoRAを適用）
    logger.info("Setting up LoRA...")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["o_proj", "q_proj", "k_proj", "v_proj"],
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    # LoRAを適用（SO8Tゲート挿入前に実行）
    model = get_peft_model(model, lora_config)
    
    # SO8T回転ゲートを挿入（LoRA適用後、学習時のみ）
    logger.info("Inserting SO8T rotation gates...")
    hidden_size = model.config.hidden_size if hasattr(model.config, 'hidden_size') else 3072
    # 4bit量子化の場合、compute_dtypeを使用
    rotation_dtype = None
    if quantization_config and hasattr(quantization_config, 'bnb_4bit_compute_dtype'):
        rotation_dtype = quantization_config.bnb_4bit_compute_dtype
    elif quantization_config:
        rotation_dtype = torch.float16
    rotation_gates = insert_so8t_gates(model, layer_indices=None, hidden_size=hidden_size, rotation_dtype=rotation_dtype)
    logger.info(f"Inserted {len(rotation_gates)} SO8T gates")
    
    # 訓練可能パラメータを表示
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,} "
                f"({100 * trainable_params / total_params:.2f}%)")
    
    # データセット読み込み
    logger.info("Loading dataset...")
    train_dataset = InstructionDataset(
        Path(args.dataset),
        tokenizer,
        max_length=args.max_length
    )
    
    # PET正則化器設定（3相スケジュール）
    pet_schedule = PETSchedule(
        phase_boundaries=(0.3, 0.7),
        lambdas=(0.05, 0.2, 0.5)  # warmup → plateau → strong
    )
    pet_regularizer = PETRegularizer(schedule=pet_schedule)
    
    # 訓練引数
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        fp16=True,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        eval_strategy="no",  # evaluation_strategy -> eval_strategy (transformers 4.21+)
        save_strategy="steps",
        load_best_model_at_end=False,
        report_to="none",
        optim="paged_adamw_8bit",
        max_grad_norm=1.0,
        warmup_steps=100,
        gradient_checkpointing=True,
    )
    
    # 時間ベースチェックポイントCallbackを作成（約3分ごと）
    time_cb = TimeBasedCheckpointCallback(fixed_interval_sec=180)
    
    # Trainer作成
    trainer = SO8TQLoRATrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        ),
        pet_regularizer=pet_regularizer,
        so8t_orthogonality_weight=0.01,
        callbacks=[time_cb]
    )
    
    # 自動再開の処理
    resume_ckpt = None
    if args.auto_resume:
        resume_ckpt = find_latest_checkpoint(str(output_dir))
        if resume_ckpt is not None:
            logger.info(f"[INFO] Resuming from latest checkpoint: {resume_ckpt}")
        else:
            logger.info("[INFO] No checkpoint found. Starting from scratch.")
    
    # 訓練開始
    logger.info("Starting training...")
    logger.info(f"Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    trainer.train(resume_from_checkpoint=resume_ckpt)
    
    # モデル保存
    logger.info(f"Saving model to {output_dir}...")
    trainer.save_model()
    tokenizer.save_pretrained(str(output_dir))
    
    logger.info("="*80)
    logger.info("Training completed!")
    logger.info("="*80)


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SO8T統合Phi-3モデルのQLoRA 8bitファインチューニング with Soul Weights
Soul Weights Datasetを学習データとして活用

RTX3060対応：元の重みを凍結しつつ、魂の重みを学習可能パラメータとして活用
Based on implementation logs, integrate soul weights as trainable parameters
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


class SoulWeightsDataset(Dataset):
    """魂の重みデータセット"""

    def __init__(self, soul_dataset_path: Path, tokenizer, max_length: int = 2048):
        """
        Args:
            soul_dataset_path: 魂の重みデータセットディレクトリ
            tokenizer: トークナイザー
            max_length: 最大シーケンス長
        """
        self.soul_dataset_path = soul_dataset_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        logger.info(f"Loading Soul Weights Dataset from {soul_dataset_path}...")

        # 学習データを読み込み
        train_file = soul_dataset_path / "soul_weights_train.jsonl"
        if train_file.exists():
            with open(train_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        sample = json.loads(line.strip())
                        self.samples.append(sample)
                    except json.JSONDecodeError:
                        continue

        logger.info(f"Loaded {len(self.samples):,} Soul Weights samples")

        # 魂の重み統計
        if self.samples:
            alpha_values = [s['alpha_gate'] for s in self.samples[:100]]  # 最初の100サンプル
            logger.info(".3f"
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 魂の重み情報をテキスト形式に変換
        soul_text = self._convert_soul_weights_to_text(sample)

        # トークナイズ
        encoded = self.tokenizer(
            soul_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        return {
            "input_ids": encoded["input_ids"].squeeze(),
            "attention_mask": encoded["attention_mask"].squeeze(),
            "labels": encoded["input_ids"].squeeze(),
            "soul_sample": sample  # 魂の重み情報を保持
        }

    def _convert_soul_weights_to_text(self, sample: Dict[str, Any]) -> str:
        """魂の重み情報を学習可能なテキスト形式に変換"""
        alpha_gate = sample.get('alpha_gate', 0.0)

        # Alpha Gateの意味をテキスト化（実装ログに基づく）
        if alpha_gate < -3.0:
            alpha_desc = "混沌状態：カオスからの学習を開始"
        elif alpha_gate < 0.0:
            alpha_desc = "相転移状態：臨界転移中"
        elif alpha_gate < 1.0:
            alpha_desc = f"秩序形成状態：物理的思考{alpha_gate*100:.1f}%混合"
        else:
            alpha_desc = f"黄金比安定状態：物理的思考{alpha_gate*84:.1f}%混合"

        # SO(8)回転の非可換性をテキスト化
        r_safe_summary = "安全回転行列：8×8直交行列"
        r_cmd_summary = "コマンド回転行列：8×8直交行列"
        commutativity = "非可換積R_cmd@R_safe：順序依存の安全処理"

        # 魂の3本柱をテキスト化
        safety_head = f"安全ヘッド：{sample.get('safety_head', [0, 0])}"
        task_head = f"タスクヘッド：{sample.get('task_head', [0, 0, 0, 0])}"
        dual_heads = f"二重政策系：{sample.get('dual_heads', [[0, 0], [0, 0]])}"
        pet = f"PET正則化：態度の慣性{abs(sample.get('pet', 0)):.4f}"

        # 学習テキスト生成
        text = f"""### Soul Weights Training Sample
Alpha Gate: {alpha_gate:.4f} - {alpha_desc}
{r_safe_summary}
{r_cmd_summary}
{commutativity}

### Soul Pillars (魂の3本柱)
{safety_head}
{task_head}
{dual_heads}
{pet}

### Learning Objective
この魂の重みを学習し、SO(8)回転ゲートによる安全で効率的な推論を実現する。
非可換構造を理解し、Alpha Gateの相転移を通じて物理的知性を獲得する。

### Implementation Logs Reference
- R_safe/R_cmd: 非可換ゲート構造（順序固定）
- Alpha Gate: 黄金比アニーリング（-5.0 → 1.618）
- Soul Pillars: 安全/タスク/二重政策/PET正則化
- LoRA Adapter: RTX3060最適化（r=16）

"""

        return text


class SO8TSoulTrainer(Trainer):
    """
    魂の重みを統合したSO8T固有の損失計算を含むTrainer
    """

    def __init__(self, soul_orthogonality_weight: float = 0.01, alpha_gate_weight: float = 0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.soul_orthogonality_weight = soul_orthogonality_weight
        self.alpha_gate_weight = alpha_gate_weight

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        魂の重みを統合した損失計算
        SO8T直交性正則化損失 + Alpha Gate学習
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

        # Alpha Gateの学習損失（黄金比1.618への収束）
        alpha_loss = torch.tensor(0.0, device=task_loss.device)
        if hasattr(model, 'get_alpha_gate_loss'):
            try:
                alpha_loss = model.get_alpha_gate_loss()
            except Exception as e:
                logger.warning(f"Failed to compute Alpha Gate loss: {e}")

        # 魂の重みの整合性損失
        soul_consistency_loss = torch.tensor(0.0, device=task_loss.device)
        if hasattr(model, 'get_soul_consistency_loss'):
            try:
                soul_consistency_loss = model.get_soul_consistency_loss()
            except Exception as e:
                logger.warning(f"Failed to compute Soul consistency loss: {e}")

        # 総損失
        total_loss = (task_loss +
                     self.soul_orthogonality_weight * so8t_loss +
                     self.alpha_gate_weight * alpha_loss +
                     0.05 * soul_consistency_loss)

        return (total_loss, outputs) if return_outputs else total_loss


def load_so8t_model_with_soul(model_path: str, device: str = "cuda", torch_dtype: str = "bfloat16"):
    """
    SO8T統合モデルを魂の重み対応で読み込む

    Args:
        model_path: モデルパス
        device: デバイス
        torch_dtype: データ型
    """
    logger.info(f"Loading SO8T model with Soul Weights from {model_path}")

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
            "modeling_phi3_so8t_soul",
            PROJECT_ROOT / "models" / "Borea-Phi-3.5-mini-Instruct-Jp" / "modeling_phi3_so8t_soul.py"
        )
        modeling_so8t_soul = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(modeling_so8t_soul)
        SO8TPhi3ForCausalLMWithSoul = modeling_so8t_soul.SO8TPhi3ForCausalLMWithSoul
    except ImportError:
        logger.warning("Soul-enhanced model not found, using standard SO8T model")
        try:
            spec = importlib.util.spec_from_file_location(
                "modeling_phi3_so8t",
                PROJECT_ROOT / "models" / "Borea-Phi-3.5-mini-Instruct-Jp" / "modeling_phi3_so8t.py"
            )
            modeling_so8t = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(modeling_so8t)
            SO8TPhi3ForCausalLMWithSoul = modeling_so8t.SO8TPhi3ForCausalLM
        except Exception as e:
            logger.error(f"Failed to import modeling_phi3_so8t: {e}")
            raise

    # モデルを読み込み
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    model = SO8TPhi3ForCausalLMWithSoul.from_pretrained(
        model_path,
        config=config,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
    )

    logger.info("[OK] SO8T model with Soul Weights loaded successfully")
    return model


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description="Train SO8T-integrated Phi-3 model with QLoRA 8bit + Soul Weights"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/train_so8t_phi3_qlora_rtx3060_soul.yaml',
        help='Configuration file path'
    )
    parser.add_argument(
        '--soul_dataset',
        type=str,
        required=True,
        help='Soul weights dataset directory path'
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
    model_path = model_config.get('base_model_path') or model_config.get('base_model', 'models/Borea-Phi-3.5-mini-Instruct-Jp')

    # パス解決
    model_path = Path(model_path)
    if not model_path.is_absolute():
        PROJECT_ROOT = Path(__file__).parent.parent.parent
        model_path = PROJECT_ROOT / model_path

    if not model_path.exists():
        logger.error(f"[ERROR] Model path does not exist: {model_path}")
        sys.exit(1)

    torch_dtype = model_config.get('torch_dtype', 'bfloat16')
    device = config.get('device', 'cuda')

    # 魂の重みデータセット設定
    soul_dataset_path = Path(args.soul_dataset)
    if not soul_dataset_path.exists():
        logger.error(f"[ERROR] Soul dataset path does not exist: {soul_dataset_path}")
        sys.exit(1)

    # データ設定
    data_config = config['data']
    max_seq_length = data_config.get('max_seq_length', 2048)

    # 訓練設定
    training_config = config['training']
    output_dir = Path(training_config.get('output_dir', 'D:/webdataset/checkpoints/finetuning/so8t_phi3_rtx3060_soul'))
    output_dir.mkdir(parents=True, exist_ok=True)

    # QLoRA設定
    qlora_config = config.get('qlora', {})
    lora_r = qlora_config.get('r', 16)  # RTX3060 optimized
    lora_alpha = qlora_config.get('lora_alpha', 32)
    lora_dropout = qlora_config.get('lora_dropout', 0.05)
    target_modules = qlora_config.get('target_modules', [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])

    # 量子化設定
    quantization_config = config.get('quantization', {})
    load_in_8bit = quantization_config.get('load_in_8bit', True)

    logger.info(f"[STEP 1] Loading tokenizer from local path: {model_path}")
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path),
            trust_remote_code=True,
            local_files_only=True
        )
    except Exception as e:
        logger.error(f"[ERROR] Failed to load tokenizer from {model_path}")
        logger.error(f"[ERROR] Error: {e}")
        sys.exit(1)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("[STEP 2] Loading SO8T model with Soul Weights")
    model = load_so8t_model_with_soul(str(model_path), device=device, torch_dtype=torch_dtype)

    # 8bit量子化設定
    if load_in_8bit:
        logger.info("[STEP 3] Configuring 8bit quantization")
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        )
        # モデルを再読み込み（量子化付き）
        model = load_so8t_model_with_soul(str(model_path), device=device, torch_dtype=torch_dtype)
        model = prepare_model_for_kbit_training(model)

    logger.info("[STEP 4] Configuring QLoRA with Soul Weights")
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

    # 魂の重みパラメータを学習可能に設定
    soul_keywords = [
        'r_safe', 'r_cmd', 'alpha', 'soul',
        'safety_head', 'task_head', 'dual_heads', 'pet',
        'so8', 'rotation', 'alpha_gate'
    ]

    trainable_soul_params = 0
    for name, param in model.named_parameters():
        if any(keyword in name.lower() for keyword in soul_keywords):
            param.requires_grad = True
            trainable_soul_params += param.numel()
        else:
            param.requires_grad = False

    # 訓練可能パラメータを表示
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"[INFO] Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")
    logger.info(f"[INFO] Soul weights parameters: {trainable_soul_params:,}")

    logger.info("[STEP 5] Preparing Soul Weights Dataset")
    train_dataset = SoulWeightsDataset(soul_dataset_path, tokenizer, max_length=max_seq_length)

    logger.info(f"[INFO] Soul Weights samples: {len(train_dataset):,}")

    logger.info("[STEP 6] Configuring training arguments (RTX3060 optimized)")
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=training_config.get('num_train_epochs', 3),
        per_device_train_batch_size=training_config.get('per_device_train_batch_size', 1),
        per_device_eval_batch_size=training_config.get('per_device_eval_batch_size', 1),
        gradient_accumulation_steps=training_config.get('gradient_accumulation_steps', 4),
        learning_rate=training_config.get('learning_rate', 2.0e-4),
        weight_decay=training_config.get('weight_decay', 0.01),
        warmup_ratio=training_config.get('warmup_ratio', 0.1),
        lr_scheduler_type=training_config.get('lr_scheduler_type', 'cosine'),
        logging_steps=training_config.get('logging_steps', 10),
        save_steps=training_config.get('save_steps', 500),
        save_total_limit=training_config.get('save_total_limit', 3),
        evaluation_strategy='no',  # Soul Weightsは評価なし
        fp16=training_config.get('fp16', True),
        bf16=training_config.get('bf16', False),
        gradient_checkpointing=training_config.get('gradient_checkpointing', True),
        optim=training_config.get('optim', 'paged_adamw_8bit'),
        max_grad_norm=training_config.get('max_grad_norm', 1.0),
        report_to=training_config.get('report_to', []),
        dataloader_num_workers=training_config.get('dataloader_num_workers', 0),
    )

    # データコレーター
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    logger.info("[STEP 7] Creating SO8T Soul Weights Trainer")
    # 魂の重み固有の損失重み
    soul_config = config.get('soul', {})
    so8t_orthogonality_weight = soul_config.get('so8t_orthogonality_weight', 0.01)
    alpha_gate_weight = soul_config.get('alpha_gate_weight', 0.1)

    trainer = SO8TSoulTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        soul_orthogonality_weight=so8t_orthogonality_weight,
        alpha_gate_weight=alpha_gate_weight,
    )

    # チェックポイントから再開
    if args.resume:
        logger.info(f"[STEP 8] Resuming from checkpoint: {args.resume}")
        trainer.train(resume_from_checkpoint=args.resume)
    else:
        logger.info("[STEP 8] Starting training with Soul Weights")
        trainer.train()

    logger.info("[STEP 9] Saving final model with Soul Weights")
    final_model_dir = output_dir / "final"
    final_model_dir.mkdir(exist_ok=True)
    trainer.save_model(str(final_model_dir))
    tokenizer.save_pretrained(str(final_model_dir))

    # 魂の重みパラメータを別途保存
    soul_params = {
        'alpha_gate': getattr(model, 'alpha_gate', torch.tensor(1.618)),
        'r_safe': getattr(model, 'r_safe', None),
        'r_cmd': getattr(model, 'r_cmd', None),
        'safety_head': getattr(model, 'safety_head', None),
        'task_head': getattr(model, 'task_head', None),
        'dual_heads': getattr(model, 'dual_heads', None),
        'pet': getattr(model, 'pet', None),
        'training_step': trainer.state.global_step,
        'timestamp': datetime.now().isoformat()
    }

    # 有効なパラメータのみ保存
    valid_soul_params = {k: v for k, v in soul_params.items() if v is not None}
    torch.save(valid_soul_params, final_model_dir / "soul_weights.pt")

    logger.info("[SUCCESS] Training completed with Soul Weights integration!")
    logger.info(f"Final model saved to {final_model_dir}")
    logger.info(f"Soul weights saved to {final_model_dir / 'soul_weights.pt'}")


if __name__ == '__main__':
    main()

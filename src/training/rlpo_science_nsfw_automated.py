#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RLPO学習: 科学・数学SFTデータセット + NKAT理論 + 薬物NSFWデータ
Unslothを使わない標準Transformersベースの完全自動化実装

特徴:
- 科学・数学推論データセット統合
- NKAT理論ベースのSO(8)アダプター
- 薬物NSFWデータによる安全RL学習
- 完全自動化パイプライン
- ローリングチェックポイント
- 多GPU分散学習対応

著者: AI Agent (自動生成)
日付: 2025-12-04
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import math
# Transformers関連
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    TrainerCallback
)

# PEFT
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)

# NKAT理論とSO(8)アダプター
from src.models.so8t_residual_adapter import NKATMLPWrapper, NKATLayerWrapper, SO8ResidualAdapter, inject_nkat_to_all_layers

# 自動チェックポイントマネージャー
from src.utils.checkpoint_manager import RollingCheckpointManager, create_task_manager

# ロガー設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RLPOConfig:
    """RLPO学習設定"""
    # モデル設定
    model_name: str = "microsoft/phi-3.5-mini-instruct"
    tokenizer_name: str = "microsoft/phi-3.5-mini-instruct"

    # データセット
    science_dataset_path: str = "data/science_reasoning_dataset_final.jsonl"
    nsfw_drug_dataset_path: str = "data/nsfw_drug_detection/nsfw_drug_mixed_dataset.jsonl"
    max_seq_length: int = 2048

    # LoRA設定
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "gate_proj", "up_proj", "down_proj"
    ])

    # RLPO設定
    learning_rate: float = 1e-6
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    max_steps: int = 10000
    warmup_steps: int = 100

    # NKAT設定
    use_nkat_adapter: bool = True
    adapter_hidden_size: int = 256
    alpha_init: float = -0.1
    nkat_target_layers: str = "all"  # "all", "middle", またはリスト
    nkat_mode: str = "full_layer"    # "full_layer" または "mlp_only"

    # 報酬設定
    science_reward_weight: float = 1.0
    safety_reward_weight: float = 2.0
    nsfw_penalty_weight: float = 3.0

    # チェックポイント
    output_dir: str = "checkpoints/rlpo_science_nsfw"
    save_steps: int = 500
    eval_steps: int = 250

    # 量子化
    use_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16"


class ScienceDataset(Dataset):
    """科学・数学データセット"""

    def __init__(self, data_path: str, tokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []

        logger.info(f"Loading science dataset: {data_path}")
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    self.data.append(item)

        logger.info(f"Loaded {len(self.data)} science examples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # メッセージ形式のデータを処理
        if 'messages' in item:
            messages = item['messages']
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            text = item.get('text', '')

        # トークナイズ
        tokenized = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': tokenized['input_ids'].squeeze(),
            'attention_mask': tokenized['attention_mask'].squeeze(),
            'labels': tokenized['input_ids'].squeeze(),  # SFTなので同じ
            'reward_type': 'science'
        }


class NSFWDrugDataset(Dataset):
    """NSFW薬物データセット"""

    def __init__(self, data_path: str, tokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []

        logger.info(f"Loading NSFW drug dataset: {data_path}")
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    self.data.append(item)

        logger.info(f"Loaded {len(self.data)} NSFW drug examples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # テキストからクエリを生成
        text = item.get('text', '')
        drug_type = item.get('metadata', {}).get('drug_type', 'unknown')
        is_dangerous = item.get('metadata', {}).get('is_dangerous', False)
        safety_label = 'DANGEROUS' if is_dangerous else 'SAFE'

        # RLPO用のプロンプト作成
        prompt = f"""あなたは安全で倫理的なAIアシスタントです。以下の薬物関連のトピックについて、危険な情報を提供せず、安全で役立つ回答をしてください。

トピック: {text}

薬物タイプ: {drug_type}

回答:"""

        # 安全な回答例を追加
        if safety_label == 'DANGEROUS':
            safe_response = "このような薬物関連の危険な話題については、医療専門家に相談することを強くおすすめします。違法薬物の製造や使用は健康と法的に重大なリスクを伴います。"
        else:
            safe_response = "薬物関連の質問については、医療専門家や信頼できる情報源に相談してください。安全で正確な情報が重要です。"

        full_text = prompt + safe_response

        # トークナイズ
        tokenized = self.tokenizer(
            full_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': tokenized['input_ids'].squeeze(),
            'attention_mask': tokenized['attention_mask'].squeeze(),
            'labels': tokenized['input_ids'].squeeze(),
            'reward_type': 'nsfw_drug',
            'safety_label': safety_label
        }


class CombinedDataset(Dataset):
    """科学・数学 + NSFW薬物統合データセット"""

    def __init__(self, science_dataset, nsfw_dataset, science_ratio: float = 0.7):
        self.science_dataset = science_dataset
        self.nsfw_dataset = nsfw_dataset
        self.science_ratio = science_ratio

        # データサイズ計算
        self.science_size = len(science_dataset)
        self.nsfw_size = len(nsfw_dataset)
        self.total_size = max(self.science_size, self.nsfw_size) * 2  # バランスを取る

        logger.info(f"Combined dataset: {self.total_size} examples "
                   f"(Science: {self.science_size}, NSFW: {self.nsfw_size})")

    def __len__(self):
        return self.total_size

    def __getitem__(self, idx):
        # 比率に基づいてデータセットを選択
        if np.random.random() < self.science_ratio:
            # 科学データ
            science_idx = idx % self.science_size
            return self.science_dataset[science_idx]
        else:
            # NSFW薬物データ
            nsfw_idx = idx % self.nsfw_size
            return self.nsfw_dataset[nsfw_idx]


class ScienceNSFWRewardFunction:
    """科学・NSFW薬物報酬関数"""

    def __init__(self, config: RLPOConfig):
        self.config = config

        # キーワードベースの報酬計算
        self.science_keywords = [
            'theorem', 'proof', 'equation', 'mathematical', 'scientific',
            'calculate', 'derive', 'theorem', 'hypothesis', 'experiment'
        ]

        self.safety_keywords = [
            'dangerous', 'illegal', 'harmful', 'addictive', 'toxic',
            'controlled substance', 'prescription required', 'medical supervision'
        ]

        self.nsfw_penalty_keywords = [
            'how to make', 'recipe', 'manufacture', 'synthesize', 'produce',
            'grow', 'cultivate', 'extract', 'distill'
        ]

    def __call__(self, prediction: str, target: str, reward_type: str, safety_label: str = None) -> float:
        """報酬計算"""
        reward = 0.0

        if reward_type == 'science':
            # 科学タスクの報酬
            science_score = self._calculate_science_score(prediction)
            reward += science_score * self.config.science_reward_weight

        elif reward_type == 'nsfw_drug':
            # NSFW薬物タスクの報酬
            safety_score = self._calculate_safety_score(prediction, safety_label)
            penalty_score = self._calculate_penalty_score(prediction)

            reward += safety_score * self.config.safety_reward_weight
            reward -= penalty_score * self.config.nsfw_penalty_weight

        return reward

    def _calculate_science_score(self, prediction: str) -> float:
        """科学タスクのスコア計算"""
        score = 0.0

        # 科学キーワードの出現
        for keyword in self.science_keywords:
            if keyword.lower() in prediction.lower():
                score += 0.1

        # 数学記号の使用
        math_symbols = ['=', '+', '-', '×', '÷', '∫', '∑', '√', 'π', '∞']
        for symbol in math_symbols:
            if symbol in prediction:
                score += 0.05

        # ステップバイステップの説明
        if 'step' in prediction.lower() or 'therefore' in prediction.lower():
            score += 0.2

        return min(score, 1.0)  # 最大1.0

    def _calculate_safety_score(self, prediction: str, safety_label: str) -> float:
        """安全タスクのスコア計算"""
        score = 0.0

        if safety_label == 'SAFE':
            # 安全な回答
            if any(kw in prediction.lower() for kw in ['medical', 'doctor', 'professional', 'supervised']):
                score += 0.5

        elif safety_label == 'DANGEROUS':
            # 危険な回答への拒否
            if any(kw in prediction.lower() for kw in ['cannot', 'illegal', 'dangerous', 'harmful']):
                score += 0.8

        return score

    def _calculate_penalty_score(self, prediction: str) -> float:
        """ペナルティスコア計算（危険な指示の提供）"""
        penalty = 0.0

        for keyword in self.nsfw_penalty_keywords:
            if keyword.lower() in prediction.lower():
                penalty += 0.3

        return min(penalty, 1.0)


class RLPOCheckpointCallback(TrainerCallback):
    """RLPO学習用のチェックポイントコールバック"""

    def __init__(self, checkpoint_manager):
        self.checkpoint_manager = checkpoint_manager
        self.step_counter = 0

    def on_step_end(self, args, state, control, **kwargs):
        self.step_counter += 1

        # ★★★ 3分間隔チェックポイント保存 ★★★
        if self.checkpoint_manager.should_save():
            model = kwargs.get('model')
            step_info = f"step_{state.global_step}"
            self.checkpoint_manager.save_checkpoint(
                data=model,
                metadata={"step": state.global_step, "loss": state.log_history[-1].get('loss', 'N/A') if state.log_history else 'N/A'},
                step_info=step_info
            )

        # NKATアダプター監視（50ステップごと）
        if self.step_counter % 50 != 0:
            return

        # Alphaアニーリング（-0.5からsigmoidでアニーリングし最終的にΦ^{-2}≃0.381966に到達）
        model = kwargs.get('model')
        if model and hasattr(model, "named_modules"):
            total_steps = args.max_steps or 1000
            progress = min(state.global_step / max(total_steps, 1), 1.0)

            # Φ（黄金比）とΦ^-2を計算
            phi = (1 + 5 ** 0.5) / 2
            phi_inv2 = phi ** -2  # ≈ 0.38196601125

            start_alpha = -0.5
            end_alpha = phi_inv2  # ≈0.381966

            # sigmoidアニーリング: α = start + sigmoid(lerp(0, 6, progress)) * (end-start)
            # 6はannealingスピードパラメータ（大きいほど早く収束）
            anneal_speed = 6.0
            sigmoid_val = 1 / (1 + math.exp(-anneal_speed * (progress - 0.5)))  # progress==0.5付近で中心
            target_alpha = start_alpha + sigmoid_val * (end_alpha - start_alpha)

            for name, module in model.named_modules():
                if "nkat_adapter" in name and hasattr(module, "alpha_logit"):
                    # 逆sigmoidでlogit更新
                    # target_alpha∈[start_alpha, end_alpha]→p（Sigmoid空間、0～1対応）:
                    # 線形スケールしシグモイド空間に。
                    # sc = (alpha - start) / (end - start)
                    p = (target_alpha - start_alpha) / (end_alpha - start_alpha)
                    p = float(torch.clamp(torch.tensor(p), 1e-7, 1-1e-7))
                    new_logit = math.log(p / (1.0 - p))

                    with torch.no_grad():
                        module.alpha_logit.copy_(new_logit)

        # アダプター状態監視
        adapters = []
        if model:
            for name, module in model.named_modules():
                if "nkat_adapter" in name and hasattr(module, "get_adapter_stats"):
                    adapters.append((name, module))

        if not adapters:
            return

        print(f"\n[NKAT DEBUG] Step {state.global_step} (Progress: {progress:.1%})")
        print(f"  Current Loss: {state.log_history[-1].get('loss', 'N/A') if state.log_history else 'N/A'}")

        for name, adapter in adapters[:3]:
            stats = adapter.get_adapter_stats()
            ortho_err = stats['orthogonality_error']
            alpha = stats['alpha']
            lie_norm = stats['lie_norm']

            grad_norm = "None"
            if adapter.lie_algebra.grad is not None:
                grad_norm = f"{torch.norm(adapter.lie_algebra.grad).item():.6f}"

            print(f"  - {name}:")
            print(f"    Ortho Error: {ortho_err:.6e}")
            print(f"    Alpha: {alpha:.6e}")
            print(f"    Lie Norm: {lie_norm:.6f}")
            print(f"    Grad Norm: {grad_norm}")

            if ortho_err > 1e-2:
                print("    [WARN] Orthogonality breaking down!")
            if alpha > 1.0:
                print("    [WARN] Alpha too large!")
            if grad_norm == "None":
                print("    [WARN] Gradient detached!")

        print("-" * 40)


class RLPOTrainer:
    """RLPO学習トレーナー"""

    def __init__(self, config: RLPOConfig):
        self.config = config

        # 自動チェックポイントマネージャー初期化
        self.ckpt_manager = create_task_manager(
            task_name="rlpo_science_nsfw",
            output_dir=str(self.output_dir / "rolling_checkpoints")
        )

        # モデルとトークナイザー（チェックポイント再開対応）
        latest_ckpt = self.ckpt_manager.get_latest_checkpoint()
        if latest_ckpt and not self.ckpt_manager.current_state.get("is_completed", False):
            logger.info(f"🔄 Resuming from checkpoint: {latest_ckpt.name}")
            self.model, self.tokenizer = self._setup_model_from_checkpoint(latest_ckpt)
        else:
            logger.info("[NEW] Starting new RLPO training session")
            self.model, self.tokenizer = self._setup_model()

        # データセット
        self.train_dataset, self.eval_dataset = self._setup_datasets()

        # 報酬関数
        self.reward_function = ScienceNSFWRewardFunction(config)

        # コールバック（チェックポイント + NKAT監視）
        self.callbacks = [RLPOCheckpointCallback(self.ckpt_manager)]

    def _setup_model(self):
        """モデルとトークナイザーのセットアップ"""
        logger.info(f"Loading model: {self.config.model_name}")

        # トークナイザー
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.tokenizer_name,
            trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # 量子化設定
        if self.config.use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=getattr(torch, self.config.bnb_4bit_compute_dtype),
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        else:
            bnb_config = None

        # モデルロード
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=getattr(torch, self.config.bnb_4bit_compute_dtype),
            trust_remote_code=True
        )

        # LoRA設定
        if self.config.use_4bit:
            model = prepare_model_for_kbit_training(model)

        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )

        model = get_peft_model(model, lora_config)

        # NKATアダプター追加（すべての層に適用）
        if self.config.use_nkat_adapter:
            model = self._add_nkat_adapter(model)

        logger.info("Model setup completed")
        return model, tokenizer

    def _setup_model_from_checkpoint(self, checkpoint_path):
        """チェックポイントからモデルを再開"""
        logger.info(f"Loading model from checkpoint: {checkpoint_path}")

        try:
            # モデルとトークナイザーをチェックポイントからロード
            model = AutoModelForCausalLM.from_pretrained(
                str(checkpoint_path),
                torch_dtype=getattr(torch, self.config.bnb_4bit_compute_dtype),
                device_map="auto",
                trust_remote_code=True,
                load_in_4bit=self.config.use_4bit
            )

            tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_path))
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # LoRA設定（チェックポイントに含まれているはず）
            if not self.config.use_4bit:
                model = prepare_model_for_kbit_training(model)

            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                target_modules=self.config.target_modules,
                lora_dropout=self.config.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM"
            )
            model = get_peft_model(model, lora_config)

            # NKATアダプター再適用
            model = self._add_nkat_adapter(model)

            # デバイス移動
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)

            logger.info(f"Model resumed from checkpoint. Device: {device}")
            return model, tokenizer

        except Exception as e:
            logger.warning(f"Checkpoint loading failed: {e}. Starting fresh...")
            return self._setup_model()

    def _add_nkat_adapter(self, model):
        """NKATアダプターをモデルに追加（すべての層に適用）"""
        logger.info("Adding NKAT SO(8) adapter to ALL transformer layers...")

        # すべての層にNKATアダプターを注入
        model = inject_nkat_to_all_layers(
            model,
            target_layers=self.config.nkat_target_layers,  # 設定から取得
            mode=self.config.nkat_mode                    # 設定から取得
        )

        logger.info("NKAT adapters injected to all transformer layers")
        return model

    def _setup_datasets(self):
        """データセットのセットアップ"""
        logger.info("Setting up datasets...")

        # 個別データセット
        science_dataset = ScienceDataset(
            self.config.science_dataset_path,
            self.tokenizer,
            self.config.max_seq_length
        )

        nsfw_dataset = NSFWDrugDataset(
            self.config.nsfw_drug_dataset_path,
            self.tokenizer,
            self.config.max_seq_length
        )

        # 統合データセット
        train_dataset = CombinedDataset(science_dataset, nsfw_dataset, science_ratio=0.7)
        eval_dataset = CombinedDataset(science_dataset, nsfw_dataset, science_ratio=0.5)

        return train_dataset, eval_dataset

    def _custom_collate_fn(self, batch):
        """カスタムコレート関数"""
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch])

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

    def train(self):
        """学習実行"""
        logger.info("Starting RLPO training with science + NSFW datasets...")

        # トレーニング引数
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=1,  # RLPOなのでエポックではなくステップベース
            max_steps=self.config.max_steps,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            logging_steps=50,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            save_total_limit=3,
            evaluation_strategy="steps",
            load_best_model_at_end=False,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            dataloader_num_workers=0,  # Windows対応
            remove_unused_columns=False,
        )

        # データコレーター
        data_collator = self._custom_collate_fn

        # トレーナー
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=data_collator,
            callbacks=self.callbacks,
        )

        # 学習開始
        logger.info("Starting training loop...")
        trainer.train()

        # 最終保存
        final_path = Path(self.config.output_dir) / "final_model"
        trainer.save_model(str(final_path))
        self.tokenizer.save_pretrained(str(final_path))

        # 学習完了マーク
        self.ckpt_manager.mark_completed()
        logger.info("RLPO training completed!")


def main():
    parser = argparse.ArgumentParser(description='RLPO Training: Science + NSFW + NKAT')
    parser.add_argument('--model_name', type=str, default='microsoft/phi-3.5-mini-instruct',
                       help='Base model name')
    parser.add_argument('--science_dataset', type=str,
                       default='data/science_reasoning_dataset_final.jsonl',
                       help='Science/mathematics dataset path')
    parser.add_argument('--nsfw_dataset', type=str,
                       default='data/nsfw_drug_detection/nsfw_drug_mixed_dataset.jsonl',
                       help='NSFW drug dataset path')
    parser.add_argument('--output_dir', type=str,
                       default='checkpoints/rlpo_science_nsfw_automated',
                       help='Output directory')
    parser.add_argument('--max_steps', type=int, default=5000,
                       help='Maximum training steps')
    parser.add_argument('--batch_size', type=int, default=2,
                       help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-6,
                       help='Learning rate')

    args = parser.parse_args()

    # 設定
    config = RLPOConfig(
        model_name=args.model_name,
        science_dataset_path=args.science_dataset,
        nsfw_drug_dataset_path=args.nsfw_dataset,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )

    # トレーナー実行
    trainer = RLPOTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()

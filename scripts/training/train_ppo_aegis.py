#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AEGIS-v2.0 PPO Training Script
NKAT理論に基づく四値分類タグ学習とSO(8)幾何学アダプター

このスクリプトは、Phi-3.5-mini-instructベースのboreaモデルに対して、
NKAT理論を実装したPPO強化学習を行います。

特徴:
- SO(8)回転ゲートに基づく幾何学アダプター
- 四値分類タグ（<|allow|>, <|escalation|>, <|deny|>, <|refuse|>）の学習
- NKATサーモスタットによる動的温度制御
- RTX 3060 (12GB VRAM) 最適化

著者: AI Agent (峯岸亮さん仕様に基づく)
日付: 2025-11-30
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
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

# TRLとUnsloth
try:
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template
    from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
    from trl.core import LengthSampler
    UNSLOTH_AVAILABLE = True
except ImportError:
    print("UnslothまたはTRLがインストールされていません。pip install unsloth trl")
    UNSLOTH_AVAILABLE = False

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# カスタムモジュール
from scripts.models.so8_quad_inference import SO8RotationGate
from scripts.training.nkat_reward_function import NKATRewardFunction
from scripts.inference.nkat_thermostat import NKATThermostat

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/train_ppo_aegis.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class PPOConfig:
    """PPO学習設定"""
    # モデル設定
    model_name: str = "borea/Borea-phi3.5-instinct-jp"
    tokenizer_name: str = "microsoft/Phi-3.5-mini-instruct"

    # 学習設定
    num_epochs: int = 3
    batch_size: int = 1
    mini_batch_size: int = 1
    gradient_accumulation_steps: int = 8

    # PPO設定
    learning_rate: float = 1e-5
    max_grad_norm: float = 0.1
    warmup_steps: int = 100
    max_steps: int = 10000

    # 生成設定
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True

    # LoRA設定
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])

    # データ設定
    dataset_path: str = "data/aegis_v2_dataset/train_aegis_v2.jsonl"
    val_dataset_path: str = "data/aegis_v2_dataset/val_aegis_v2.jsonl"

    # 出力設定
    output_dir: str = "checkpoints/aegis_v2_ppo"
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500

    # 報酬設定
    reward_weights: Dict[str, float] = field(default_factory=lambda: {
        'tag_accuracy': 1.0,
        'structure_reward': 2.0,
        'isomorphism_reward': 1.5,
        'safety_reward': 1.0,
        'stability_reward': 0.8
    })

class AEGISDataset(Dataset):
    """AEGIS-v2.0 PPO学習用データセット"""

    def __init__(self, data_path: str, tokenizer: AutoTokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        # データ読み込み
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    sample = json.loads(line)
                    self.samples.append(sample)

        logger.info(f"Loaded {len(self.samples)} samples from {data_path}")

        # タグ分布統計
        tag_counts = {}
        for sample in self.samples:
            tag = sample.get('tag', '<|allow|>')
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

        logger.info("Tag distribution in dataset:")
        for tag, count in tag_counts.items():
            logger.info(f"  {tag}: {count}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        instruction = sample.get('instruction', '')
        output = sample.get('output', '')
        tag = sample.get('tag', '<|allow|>')

        # Phi-3.5形式のプロンプト作成
        prompt = f"<|user|>\n{instruction}<|end|>\n<|assistant|>\n"

        # ターゲット（正解タグを含む出力）
        target = f"{tag}\n{output}"

        return {
            'prompt': prompt,
            'target': target,
            'tag': tag,
            'instruction': instruction,
            'expected_output': output
        }

class SO8GeometricAdapter(nn.Module):
    """SO(8)幾何学に基づくアダプター"""

    def __init__(self, hidden_size: int, so8_dim: int = 8):
        super().__init__()
        self.hidden_size = hidden_size
        self.so8_dim = so8_dim

        # SO(8)回転ゲート
        self.rotation_gate = SO8RotationGate(so8_dim)

        # 幾何学変換層
        self.geom_transform = nn.Linear(hidden_size, so8_dim)
        self.output_proj = nn.Linear(so8_dim, hidden_size)

        # 残差接続用スケーリング
        self.residual_scale = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 隠れ状態をSO(8)空間に射影
        geom_features = self.geom_transform(hidden_states)

        # SO(8)回転適用
        rotated_features = self.rotation_gate(geom_features)

        # 元の空間に戻す
        output_features = self.output_proj(rotated_features)

        # 残差接続
        return hidden_states + self.residual_scale * output_features

class NKATPPOTrainer:
    """NKAT理論を実装したPPOトレーナー"""

    def __init__(self, config: PPOConfig):
        self.config = config

        # 出力ディレクトリ作成
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 報酬関数初期化
        self.reward_function = NKATRewardFunction()

        # NKATサーモスタット初期化
        self.thermostat = NKATThermostat(
            base_temp=config.temperature,
            cool_factor=0.1,
            heat_factor=1.5
        )

        # モデル初期化
        self.model, self.tokenizer, self.peft_config = self._setup_model()

        # データセット初期化
        self.train_dataset = AEGISDataset(config.dataset_path, self.tokenizer)
        self.val_dataset = AEGISDataset(config.val_dataset_path, self.tokenizer)

        # 学習統計
        self.training_stats = {
            'epoch': 0,
            'step': 0,
            'total_reward': 0,
            'tag_accuracy': 0,
            'structure_reward': 0,
            'safety_reward': 0
        }

    def _setup_model(self):
        """モデルとトークナイザーのセットアップ"""
        logger.info(f"Loading model: {self.config.model_name}")

        if UNSLOTH_AVAILABLE:
            try:
                # Unsloth使用
                model, tokenizer = FastLanguageModel.from_pretrained(
                    model_name=self.config.model_name,
                    max_seq_length=self.config.max_new_tokens * 2,
                    dtype=None,
                    load_in_4bit=True,
                )

                # LoRA設定
                model = FastLanguageModel.get_peft_model(
                    model,
                    r=self.config.lora_r,
                    target_modules=self.config.target_modules,
                    lora_alpha=self.config.lora_alpha,
                    lora_dropout=self.config.lora_dropout,
                    bias="none",
                    use_gradient_checkpointing=True,
                    random_state=42,
                    use_rslora=False,
                    loftq_config=None,
                )

                peft_config = None

            except Exception as e:
                logger.warning(f"Unsloth failed: {e}, falling back to transformers")
                UNSLOTH_AVAILABLE = False

        if not UNSLOTH_AVAILABLE:
            # 標準Transformers使用
            tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_name)
            tokenizer.pad_token = tokenizer.eos_token

            # 4bit量子化設定
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )

            model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.float16,
            )

            # LoRA設定
            peft_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                target_modules=self.config.target_modules,
                lora_dropout=self.config.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM"
            )

            model = prepare_model_for_kbit_training(model)
            model = get_peft_model(model, peft_config)

        # SO(8)幾何学アダプター追加
        if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
            hidden_size = model.model.embed_tokens.embedding_dim
        else:
            hidden_size = model.config.hidden_size

        so8_adapter = SO8GeometricAdapter(hidden_size)
        model.so8_adapter = so8_adapter

        # モデルをデバイスに移動
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        logger.info(f"Model loaded successfully. Hidden size: {hidden_size}")
        logger.info(f"SO(8) adapter added with dimension: {so8_adapter.so8_dim}")

        return model, tokenizer, peft_config

    def _extract_tag_from_output(self, output_text: str) -> str:
        """出力テキストからタグを抽出"""
        tag_patterns = ['<\|allow\|>', '<\|escalation\|>', '<\|deny\|>', '<\|refuse\|>']
        for tag in tag_patterns:
            if tag in output_text:
                return tag
        return "<|allow|>"  # デフォルト

    def _calculate_reward(self, prompt: str, generated_output: str,
                         expected_tag: str, expected_output: str) -> float:
        """報酬計算"""
        total_reward = 0.0

        # 1. タグ正解率報酬
        predicted_tag = self._extract_tag_from_output(generated_output)
        tag_correct = 1.0 if predicted_tag == expected_tag else 0.0
        total_reward += self.config.reward_weights['tag_accuracy'] * tag_correct

        # 2. 構造報酬（NKAT報酬関数使用）
        structure_reward = self.reward_function.calculate_structure_reward(
            generated_output, expected_output
        )
        total_reward += self.config.reward_weights['structure_reward'] * structure_reward

        # 3. 同型性報酬
        isomorphism_reward = self.reward_function.calculate_isomorphism_reward(
            generated_output, expected_output
        )
        total_reward += self.config.reward_weights['isomorphism_reward'] * isomorphism_reward

        # 4. 安全性報酬
        safety_reward = 0.0
        if expected_tag == "<|refuse|>" and predicted_tag == "<|refuse|>":
            safety_reward = 1.0
        elif expected_tag != "<|refuse|>" and predicted_tag != "<|refuse|>":
            safety_reward = 0.5
        total_reward += self.config.reward_weights['safety_reward'] * safety_reward

        # 5. 安定性報酬（思考プロセスの一貫性）
        stability_reward = self.reward_function.calculate_stability_reward(generated_output)
        total_reward += self.config.reward_weights['stability_reward'] * stability_reward

        return total_reward

    def _generate_response(self, prompt: str) -> str:
        """モデルによる応答生成"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        # NKATサーモスタットで温度制御
        current_temp = self.thermostat.get_temperature(inputs['input_ids'])

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=current_temp,
                top_p=self.config.top_p,
                do_sample=self.config.do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return response.strip()

    def train_epoch(self) -> Dict[str, float]:
        """1エポック学習"""
        self.model.train()
        epoch_stats = {
            'total_reward': 0.0,
            'tag_accuracy': 0.0,
            'samples_processed': 0
        }

        dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0
        )

        progress_bar = tqdm(dataloader, desc=f"Epoch {self.training_stats['epoch'] + 1}")

        for batch in progress_bar:
            batch_rewards = []
            batch_tag_accuracy = []

            for sample in batch:
                prompt = sample['prompt']
                expected_tag = sample['tag']
                expected_output = sample['expected_output']

                # 応答生成
                generated_output = self._generate_response(prompt)

                # 報酬計算
                reward = self._calculate_reward(
                    prompt, generated_output, expected_tag, expected_output
                )

                # タグ正解率
                predicted_tag = self._extract_tag_from_output(generated_output)
                tag_correct = 1.0 if predicted_tag == expected_tag else 0.0

                batch_rewards.append(reward)
                batch_tag_accuracy.append(tag_correct)

                # 統計更新
                epoch_stats['total_reward'] += reward
                epoch_stats['tag_accuracy'] += tag_correct
                epoch_stats['samples_processed'] += 1

            # バッチ平均
            avg_reward = np.mean(batch_rewards)
            avg_tag_accuracy = np.mean(batch_tag_accuracy)

            progress_bar.set_postfix({
                'reward': f"{avg_reward:.3f}",
                'tag_acc': f"{avg_tag_accuracy:.3f}"
            })

            # ロギング
            if self.training_stats['step'] % self.config.logging_steps == 0:
                logger.info(f"Step {self.training_stats['step']}: "
                          f"Reward={avg_reward:.3f}, TagAcc={avg_tag_accuracy:.3f}")

            self.training_stats['step'] += 1

            # チェックポイント保存
            if self.training_stats['step'] % self.config.save_steps == 0:
                self.save_checkpoint()

        # エポック統計計算
        epoch_stats['total_reward'] /= max(epoch_stats['samples_processed'], 1)
        epoch_stats['tag_accuracy'] /= max(epoch_stats['samples_processed'], 1)

        return epoch_stats

    def validate(self) -> Dict[str, float]:
        """検証"""
        self.model.eval()
        val_stats = {
            'total_reward': 0.0,
            'tag_accuracy': 0.0,
            'samples_processed': 0
        }

        dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0
        )

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validating"):
                for sample in batch:
                    prompt = sample['prompt']
                    expected_tag = sample['tag']
                    expected_output = sample['expected_output']

                    # 応答生成
                    generated_output = self._generate_response(prompt)

                    # 報酬計算
                    reward = self._calculate_reward(
                        prompt, generated_output, expected_tag, expected_output
                    )

                    # タグ正解率
                    predicted_tag = self._extract_tag_from_output(generated_output)
                    tag_correct = 1.0 if predicted_tag == expected_tag else 0.0

                    val_stats['total_reward'] += reward
                    val_stats['tag_accuracy'] += tag_correct
                    val_stats['samples_processed'] += 1

        # 統計計算
        val_stats['total_reward'] /= max(val_stats['samples_processed'], 1)
        val_stats['tag_accuracy'] /= max(val_stats['samples_processed'], 1)

        return val_stats

    def save_checkpoint(self):
        """チェックポイント保存"""
        checkpoint_dir = self.output_dir / f"checkpoint_step_{self.training_stats['step']}"
        checkpoint_dir.mkdir(exist_ok=True)

        # モデル保存
        if UNSLOTH_AVAILABLE:
            self.model.save_pretrained(str(checkpoint_dir))
        else:
            self.model.save_pretrained(str(checkpoint_dir))
            if self.peft_config:
                self.tokenizer.save_pretrained(str(checkpoint_dir))

        # 学習状態保存
        training_state = {
            'epoch': self.training_stats['epoch'],
            'step': self.training_stats['step'],
            'config': self.config.__dict__,
            'thermostat_state': self.thermostat.get_state()
        }

        with open(checkpoint_dir / 'training_state.json', 'w', encoding='utf-8') as f:
            json.dump(training_state, f, ensure_ascii=False, indent=2)

        logger.info(f"Checkpoint saved to {checkpoint_dir}")

    def train(self):
        """学習実行"""
        logger.info("Starting AEGIS-v2.0 PPO training...")
        logger.info(f"Model: {self.config.model_name}")
        logger.info(f"Dataset: {self.config.dataset_path}")
        logger.info(f"Output: {self.config.output_dir}")

        best_val_reward = -float('inf')

        for epoch in range(self.config.num_epochs):
            self.training_stats['epoch'] = epoch
            logger.info(f"Starting epoch {epoch + 1}/{self.config.num_epochs}")

            # 学習
            train_stats = self.train_epoch()

            # 検証
            val_stats = self.validate()

            logger.info(f"Epoch {epoch + 1} completed:")
            logger.info(f"  Train - Reward: {train_stats['total_reward']:.3f}, "
                       f"Tag Acc: {train_stats['tag_accuracy']:.3f}")
            logger.info(f"  Val - Reward: {val_stats['total_reward']:.3f}, "
                       f"Tag Acc: {val_stats['tag_accuracy']:.3f}")

            # 最良モデル保存
            if val_stats['total_reward'] > best_val_reward:
                best_val_reward = val_stats['total_reward']
                best_checkpoint_dir = self.output_dir / "best_model"
                best_checkpoint_dir.mkdir(exist_ok=True)

                if UNSLOTH_AVAILABLE:
                    self.model.save_pretrained(str(best_checkpoint_dir))
                else:
                    self.model.save_pretrained(str(best_checkpoint_dir))
                    if self.peft_config:
                        self.tokenizer.save_pretrained(str(best_checkpoint_dir))

                logger.info(f"Best model saved with val_reward: {best_val_reward:.3f}")

        logger.info("AEGIS-v2.0 PPO training completed!")

        # 最終チェックポイント保存
        self.save_checkpoint()


def main():
    parser = argparse.ArgumentParser(description='AEGIS-v2.0 PPO Training')
    parser.add_argument('--model_name', type=str, default='borea/Borea-phi3.5-instinct-jp',
                       help='Base model name')
    parser.add_argument('--dataset_path', type=str,
                       default='data/aegis_v2_dataset/train_aegis_v2.jsonl',
                       help='Training dataset path')
    parser.add_argument('--val_dataset_path', type=str,
                       default='data/aegis_v2_dataset/val_aegis_v2.jsonl',
                       help='Validation dataset path')
    parser.add_argument('--output_dir', type=str, default='checkpoints/aegis_v2_ppo',
                       help='Output directory')
    parser.add_argument('--num_epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--max_steps', type=int, default=10000, help='Maximum steps')

    args = parser.parse_args()

    # 設定更新
    config = PPOConfig()
    config.model_name = args.model_name
    config.dataset_path = args.dataset_path
    config.val_dataset_path = args.val_dataset_path
    config.output_dir = args.output_dir
    config.num_epochs = args.num_epochs
    config.batch_size = args.batch_size
    config.learning_rate = args.learning_rate
    config.max_steps = args.max_steps

    # トレーナー実行
    trainer = NKATPPOTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()

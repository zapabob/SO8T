#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phi-3.5 SO8T PPO Training with Alpha Gate Annealing

四重推論型Phi-3.5 SO8TモデルのPPO学習
アルファゲートアニーリング（α = Φ^(-2)）を実装し、
相転移をLoss loggingで観測

Phi-3.5 Thinking Format:
<think-task>タスク理解</think-task>
<think-safety>安全性評価</think-safety>
<think-logic>論理的思考</think-logic>
<think-ethics>倫理的考慮</think-ethics>
<think-practical>実用的考察</think-practical>
<think-creative>創造的アプローチ</think-creative>
<final>最終回答</final>
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
import warnings
warnings.filterwarnings("ignore")

# HuggingFaceキャッシュ設定
os.environ["HF_HOME"] = r"D:\webdataset\hf_cache"
os.environ["TRANSFORMERS_CACHE"] = r"D:\webdataset\hf_cache\transformers"
os.environ["HF_DATASETS_CACHE"] = r"D:\webdataset\hf_cache\datasets"
os.environ["HF_HUB_CACHE"] = r"D:\webdataset\hf_cache\hub"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
    TrainerCallback
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training
)
import yaml
from tqdm import tqdm
import time

# Phi-3.5 SO8T関連インポート
try:
    from so8t.core.safety_aware_so8t import SafetyAwareSO8TModel
    from so8t.core.safety_aware_so8t_config import SafetyAwareSO8TConfig
    from so8t.training.loss_functions import PETLoss, SO8TCompositeLoss
    from so8t.training.qlora import QLoRATrainer
    from so8t.training.trainer_with_pet import TrainerWithPET
    from so8t.core.thinking_tokens import extract_quadruple_thinking
except ImportError as e:
    logging.warning(f"SO8T import failed: {e}")
    SafetyAwareSO8TModel = None

# 定数定義
PHI = (1 + math.sqrt(5)) / 2  # 黄金比 φ ≈ 1.618
ALPHA_FINAL = PHI ** (-2)     # α = φ^(-2) ≈ 0.382
ALPHA_INITIAL = 1.0           # 初期α値
ANNEALING_STEPS = 1000        # アニーリング完了までのステップ数

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/phi35_ppo_annealing_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class Phi35ThinkingDataset(Dataset):
    """Phi-3.5 Thinkingフォーマット対応データセット"""

    def __init__(self, data_path: str, tokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        logger.info(f"Loading Phi-3.5 dataset from: {data_path}")

        with open(data_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading dataset"):
                try:
                    sample = json.loads(line.strip())
                    if 'phi35_thinking' in sample:
                        self.samples.append(sample)
                except json.JSONDecodeError:
                    continue

        logger.info(f"Loaded {len(self.samples):,} Phi-3.5 thinking samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Phi-3.5 Thinkingフォーマットのテキストを取得
        text = sample.get('phi35_thinking', '')
        if not text:
            text = sample.get('text', '')

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
            'labels': tokenized['input_ids'].squeeze(),
            'thinking_text': text,
            'dataset_type': sample.get('dataset_type', 'unknown'),
            'is_cot': sample.get('is_cot', False)
        }


class AlphaGateAnnealingCallback(TrainerCallback):
    """
    アルファゲートアニーリングコールバック

    α = φ^(-2) をシグモイド関数でアニーリング
    Loss loggingで相転移を観測
    """

    def __init__(self, alpha_initial: float = ALPHA_INITIAL,
                 alpha_final: float = ALPHA_FINAL,
                 annealing_steps: int = ANNEALING_STEPS):
        self.alpha_initial = alpha_initial
        self.alpha_final = alpha_final
        self.annealing_steps = annealing_steps
        self.current_alpha = alpha_initial
        self.phase_transitions = []

        # アニーリングパラメータ
        self.t0 = annealing_steps // 2  # シグモイドの中心
        self.scale = annealing_steps / 8  # シグモイドのスケール

        logger.info("Alpha Gate Annealing initialized:"        logger.info(f"  α_initial: {alpha_initial}")
        logger.info(f"  α_final: {alpha_final} (Φ^(-2) ≈ {alpha_final:.4f})")
        logger.info(f"  Annealing steps: {annealing_steps}")

    def on_step_end(self, args, state, control, **kwargs):
        """各ステップ終了時の処理"""
        current_step = state.global_step

        # アニーリング計算
        if current_step < self.annealing_steps:
            # シグモイド関数でアニーリング
            sigmoid_value = 1 / (1 + math.exp(-(current_step - self.t0) / self.scale))
            self.current_alpha = self.alpha_final + (self.alpha_initial - self.alpha_final) * (1 - sigmoid_value)

            # 相転移検出（αの急激な変化）
            if len(self.phase_transitions) == 0 or abs(self.current_alpha - self.phase_transitions[-1]['alpha']) > 0.01:
                self.phase_transitions.append({
                    'step': current_step,
                    'alpha': self.current_alpha,
                    'timestamp': datetime.now().isoformat()
                })

                logger.info(f"[ANNEALING] Step {current_step}: α = {self.current_alpha:.6f} "
                          f"(Φ^(-2) progress: {sigmoid_value:.3f})")

        # Lossを取得して相転移との相関を分析
        if hasattr(state, 'log_history') and len(state.log_history) > 0:
            latest_log = state.log_history[-1]
            if 'loss' in latest_log:
                current_loss = latest_log['loss']

                # 相転移周辺のLoss変動を検出
                if len(self.phase_transitions) >= 2:
                    recent_transitions = [t for t in self.phase_transitions[-3:]]
                    for transition in recent_transitions:
                        if abs(current_step - transition['step']) <= 5:  # 相転移周辺5ステップ
                            logger.info(f"[PHASE_TRANSITION] Detected near step {transition['step']}: "
                                      f"Loss = {current_loss:.4f}, α = {transition['alpha']:.6f}")

    def get_current_alpha(self) -> float:
        """現在のα値を取得"""
        return self.current_alpha

    def get_phase_transitions(self) -> List[Dict]:
        """相転移履歴を取得"""
        return self.phase_transitions.copy()


class Phi35SO8TPPOTrainer:
    """Phi-3.5 SO8T PPO学習クラス"""

    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.annealing_callback = None
        self.model = None
        self.tokenizer = None
        self.dataset = None

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """設定ファイル読み込み"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        # Phi-3.5固有設定の追加
        config.setdefault('phi35', {})['enabled'] = True
        config['phi35']['alpha_gate_annealing'] = True
        config['phi35']['quadruple_thinking'] = True

        return config

    def _setup_model(self):
        """Phi-3.5 SO8Tモデルのセットアップ"""
        logger.info("Setting up Phi-3.5 SO8T model...")

        model_name = self.config.get('model', {}).get('name', 'microsoft/phi-3.5-mini-instruct')

        # 量子化設定
        quantization_config = None
        if self.config.get('training', {}).get('quantization', {}).get('enabled', False):
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )

        # SO8T設定
        so8t_config = SafetyAwareSO8TConfig(
            hidden_size=self.config.get('model', {}).get('hidden_size', 3072),
            num_hidden_layers=self.config.get('model', {}).get('num_layers', 32),
            num_attention_heads=self.config.get('model', {}).get('num_heads', 32),
            intermediate_size=self.config.get('model', {}).get('intermediate_size', 8192),
            vocab_size=self.config.get('model', {}).get('vocab_size', 51200),
            max_position_embeddings=self.config.get('model', {}).get('max_position_embeddings', 4096),
            so8t_config={
                'rotation_groups': 8,
                'safety_heads': 4,
                'verifier_heads': 2,
                'geometric_constraints': True,
                'norm_preservation': True,
                'orthogonality_enforcement': True,
                'isometry_preservation': True
            }
        )

        # モデル初期化
        if SafetyAwareSO8TModel:
            self.model = SafetyAwareSO8TModel.from_pretrained(
                model_name,
                so8t_config=so8t_config,
                quantization_config=quantization_config,
                torch_dtype=torch.bfloat16,
                device_map="auto" if quantization_config else None
            )
        else:
            logger.warning("SO8T model not available, using base Phi-3.5")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                torch_dtype=torch.bfloat16,
                device_map="auto" if quantization_config else None
            )

        # QLoRA設定
        if self.config.get('training', {}).get('qlora', {}).get('enabled', True):
            self.model = self._setup_qlora(self.model)

        # アルファゲートアニーリングコールバック設定
        if self.config.get('phi35', {}).get('alpha_gate_annealing', True):
            self.annealing_callback = AlphaGateAnnealingCallback(
                alpha_initial=ALPHA_INITIAL,
                alpha_final=ALPHA_FINAL,
                annealing_steps=self.config.get('training', {}).get('max_steps', ANNEALING_STEPS)
            )

        logger.info("Phi-3.5 SO8T model setup completed")

    def _setup_qlora(self, model):
        """QLoRA設定"""
        logger.info("Setting up QLoRA...")

        lora_config = LoraConfig(
            r=self.config.get('training', {}).get('qlora', {}).get('r', 64),
            lora_alpha=self.config.get('training', {}).get('qlora', {}).get('alpha', 16),
            target_modules=self.config.get('training', {}).get('qlora', {}).get('target_modules',
                ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]),
            lora_dropout=self.config.get('training', {}).get('qlora', {}).get('dropout', 0.05),
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )

        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, lora_config)

        logger.info("QLoRA setup completed")
        return model

    def _setup_tokenizer(self):
        """トークナイザー設定"""
        model_name = self.config.get('model', {}).get('name', 'microsoft/phi-3.5-mini-instruct')

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side='right'
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Phi-3.5 Thinkingトークンの追加
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
        logger.info("Tokenizer setup completed with Phi-3.5 thinking tokens")

    def _setup_dataset(self):
        """データセット設定"""
        dataset_path = self.config.get('data', {}).get('train_path',
            'D:/webdataset/phi35_integrated/phi35_ppo_optimized_integrated.jsonl')

        self.dataset = Phi35ThinkingDataset(
            data_path=dataset_path,
            tokenizer=self.tokenizer,
            max_length=self.config.get('data', {}).get('max_length', 2048)
        )

        logger.info(f"Dataset setup completed: {len(self.dataset)} samples")

    def _create_training_args(self, output_dir: str) -> TrainingArguments:
        """トレーニング引数作成"""
        training_config = self.config.get('training', {})

        return TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=training_config.get('num_epochs', 3),
            per_device_train_batch_size=training_config.get('batch_size', 4),
            per_device_eval_batch_size=training_config.get('batch_size', 4),
            gradient_accumulation_steps=training_config.get('gradient_accumulation_steps', 8),
            learning_rate=training_config.get('learning_rate', 2e-5),
            lr_scheduler_type=training_config.get('lr_scheduler', 'cosine'),
            warmup_steps=training_config.get('warmup_steps', 100),
            max_steps=training_config.get('max_steps', 10000),
            save_steps=training_config.get('save_steps', 500),
            save_total_limit=training_config.get('save_total_limit', 5),
            evaluation_strategy="steps",
            eval_steps=training_config.get('eval_steps', 500),
            logging_steps=training_config.get('logging_steps', 50),
            logging_dir=f"{output_dir}/logs",
            report_to="tensorboard",
            bf16=True,
            dataloader_pin_memory=False,
            remove_unused_columns=False,
        )

    def train(self, output_dir: str):
        """PPO学習実行"""
        logger.info("Starting Phi-3.5 SO8T PPO training with alpha gate annealing...")

        # セットアップ
        self._setup_tokenizer()
        self._setup_model()
        self._setup_dataset()

        # モデルとトークナイザーの語彙サイズ調整
        self.model.resize_token_embeddings(len(self.tokenizer))

        # トレーニング設定
        training_args = self._create_training_args(output_dir)

        # カスタムTrainer（SO8T対応）
        if hasattr(self.model, 'so8t_config'):
            trainer_class = TrainerWithPET
        else:
            trainer_class = Trainer

        trainer = trainer_class(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset,
            data_collator=DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False
            ),
        )

        # アニーリングコールバック追加
        if self.annealing_callback:
            trainer.add_callback(self.annealing_callback)

        # トレーニング実行
        logger.info("Starting training...")
        trainer.train()

        # 最終モデル保存
        final_model_path = Path(output_dir) / "final_model"
        trainer.save_model(final_model_path)
        self.tokenizer.save_pretrained(final_model_path)

        # アニーリング結果保存
        if self.annealing_callback:
            annealing_results = {
                'phase_transitions': self.annealing_callback.get_phase_transitions(),
                'final_alpha': self.annealing_callback.get_current_alpha(),
                'annealing_config': {
                    'alpha_initial': ALPHA_INITIAL,
                    'alpha_final': ALPHA_FINAL,
                    'annealing_steps': ANNEALING_STEPS
                }
            }

            with open(f"{output_dir}/alpha_gate_annealing_results.json", 'w') as f:
                json.dump(annealing_results, f, indent=2, default=str)

        logger.info("Phi-3.5 SO8T PPO training completed!")

        # オーディオ通知
        import subprocess
        try:
            subprocess.run([
                "powershell", "-ExecutionPolicy", "Bypass", "-File",
                "scripts/utils/play_audio_notification.ps1"
            ], check=True)
        except:
            pass


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="Phi-3.5 SO8T PPO Training with Alpha Gate Annealing")
    parser.add_argument("--config", type=str, required=True, help="Configuration YAML file")
    parser.add_argument("--output", type=str, required=True, help="Output directory")

    args = parser.parse_args()

    # 出力ディレクトリ作成
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # トレーニング実行
    trainer = Phi35SO8TPPOTrainer(args.config)
    trainer.train(str(output_dir))


if __name__ == "__main__":
    main()

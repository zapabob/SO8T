#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SO8T Sunshine Pipeline - 実験計画ベースの自動実行システム

ボブにゃん戦略に基づくサンシャイン実行：
- Run A: Baseline (LoRAのみ)
- Run B: SO8T (LoRA + SO(8)アダプター)

ログフォーマット：
step, train_loss, eval_loss, so8_ortho_mean, so8_ortho_max, grad_norm, step_time_sec
"""

import os
import sys
import time
import json
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# 🚨 CRITICAL: Unsloth MUST be imported BEFORE transformers/peft!
# This prevents optimization conflicts and gradient detachment issues
import unsloth  # 必ず一番最初に！
from unsloth import FastLanguageModel

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model

# Import from scripts directory
from scripts.models.so8t_residual_adapter import replace_mlp_with_nkat
from scripts.utils.nkat_callbacks import NKATDebugCallback

# Simple dataset class for testing
from torch.utils.data import Dataset
import json

class SimpleDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, max_length: int = 128):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Load data
        self.data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        item = json.loads(line.strip())
                        self.data.append(item)
                    except json.JSONDecodeError:
                        continue

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item.get('text', '')

        # Tokenize
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
            'labels': tokenized['input_ids'].squeeze()
        }


class SunshineLogger:
    """サンシャイン実行の統一ログ収集"""

    def __init__(self, log_dir: str, run_name: str):
        self.log_dir = Path(log_dir)
        self.run_name = run_name
        self.log_file = self.log_dir / f"{run_name}_training_log.csv"
        self.metrics_file = self.log_dir / f"{run_name}_metrics.json"

        # CSVヘッダー
        self.columns = [
            'step', 'train_loss', 'eval_loss',
            'so8_ortho_mean', 'so8_ortho_max',
            'grad_norm', 'step_time_sec'
        ]

        # ログディレクトリ作成
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # CSV初期化
        if not self.log_file.exists():
            pd.DataFrame(columns=self.columns).to_csv(self.log_file, index=False)

        # メトリクス初期化
        self.metrics = {
            'start_time': datetime.now().isoformat(),
            'run_name': run_name,
            'total_steps': 0,
            'avg_step_time': 0,
            'final_train_loss': None,
            'final_eval_loss': None,
            'so8_ortho_errors': [],
            'grad_norms': []
        }

    def log_step(self, step: int, metrics: Dict[str, Any], step_time: float):
        """ステップごとのログ記録"""
        row = {
            'step': step,
            'train_loss': metrics.get('train_loss', None),
            'eval_loss': metrics.get('eval_loss', None),
            'so8_ortho_mean': metrics.get('so8_ortho_mean', None),
            'so8_ortho_max': metrics.get('so8_ortho_max', None),
            'grad_norm': metrics.get('grad_norm', None),
            'step_time_sec': step_time
        }

        # CSVに追加
        df = pd.DataFrame([row])
        df.to_csv(self.log_file, mode='a', header=False, index=False)

        # メトリクス更新
        if metrics.get('so8_ortho_mean') is not None:
            self.metrics['so8_ortho_errors'].append(metrics['so8_ortho_mean'])
        if metrics.get('grad_norm') is not None:
            self.metrics['grad_norms'].append(metrics['grad_norm'])

        self.metrics['total_steps'] = max(self.metrics['total_steps'], step)

    def finalize(self, final_metrics: Dict[str, Any]):
        """トレーニング完了時の最終ログ"""
        self.metrics.update({
            'end_time': datetime.now().isoformat(),
            'final_train_loss': final_metrics.get('train_loss'),
            'final_eval_loss': final_metrics.get('eval_loss'),
            'avg_so8_ortho_error': np.mean(self.metrics['so8_ortho_errors']) if self.metrics['so8_ortho_errors'] else None,
            'max_so8_ortho_error': np.max(self.metrics['so8_ortho_errors']) if self.metrics['so8_ortho_errors'] else None,
            'avg_grad_norm': np.mean(self.metrics['grad_norms']) if self.metrics['grad_norms'] else None
        })

        # JSON保存
        with open(self.metrics_file, 'w', encoding='utf-8') as f:
            json.dump(self.metrics, f, indent=2, ensure_ascii=False)


class SunshineTrainingConfig:
    """サンシャイン実行設定"""

    def __init__(self, run_type: str = "baseline"):
        self.model_name = "AXCXEPT/Borea-Phi-3.5-mini-Instruct-Jp"
        self.run_type = run_type  # "baseline" or "so8t"

        # データセット設定（統合データセットを使用）
        if run_type == "baseline":
            # Baseline用: 数学・科学統合データ
            self.sft_datasets = ["data/train_sft_enhanced.jsonl"]
        else:
            # SO8T用: NKAT理論・NSFWデータ統合
            self.sft_datasets = ["data/aegis_phi35_v2_with_nkat_so8t/aegis_phi35_v2_with_nkat_so8t_sft_train.jsonl"]

        # ドメイン重み付け
        self.domain_weights = {
            'mathematics': 1.2,
            'science': 1.1,
            'reasoning': 1.0,
            'general': 0.8
        }

        # トレーニング設定（本格データセット対応）
        self.training_config = {
            'output_dir': f"H:/from_D/webdataset/checkpoints/sunshine_{run_type}_phase25",
            'num_train_epochs': 1,
            'max_steps': 50,  # テスト用に減らす
            'per_device_train_batch_size': 1,
            'gradient_accumulation_steps': 8,  # RTX3060対応
            'learning_rate': 3e-5,  # 安定した学習率
            'save_steps': 100,
            'logging_steps': 10,  # 10ステップごとログ
            'eval_steps': 100,
            'gradient_checkpointing': False,  # 一時的にOFFにして勾配問題を解決
            'ddp_find_unused_parameters': False,  # アダプタパラメータの検出を確実にする
            'optim': "adamw_8bit",
            'bf16': True,
            'dataloader_pin_memory': False,
            'remove_unused_columns': False,
            'report_to': [],
            'load_best_model_at_end': False
        }

        # LoRA設定
        self.lora_config = {
            'r': 16,
            'lora_alpha': 32,
            'lora_dropout': 0.05,
            'bias': "none",
            'task_type': "CAUSAL_LM",
            'target_modules': ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        }

        # SO(8)設定（so8tの場合のみ）
        if run_type == "so8t":
            self.so8_config = {
                'target_layers': [8, 16, 24],  # 中間層のみ
                'so8_dim': 8,
                'alpha_init': 0.1
            }
        else:
            self.so8_config = None


def run_sunshine_experiment(run_type: str = "baseline") -> Dict[str, Any]:
    """
    サンシャイン実験実行
    run_type: "baseline" or "so8t"
    """
    print(f"🌞 Starting Sunshine Experiment: {run_type.upper()}")
    print("=" * 60)

    # 設定
    config = SunshineTrainingConfig(run_type)
    run_name = f"sunshine_run_{run_type}"

    # ロガー初期化
    logger = SunshineLogger("logs/sunshine", run_name)

    try:
        # モデルとトークナイザーロード
        print("[1/5] Loading model and tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # ★★★ トレーニング時は device_map="auto" を使用せず直接GPUにロード ★★★
        # device_map="auto" は分散トレーニングと競合するため
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float16,
            device_map=None  # トレーニング時は明示的にNone
        ).to("cuda")  # 直接GPUに移動

        # ★★★ デバッグ: Phi-3モデルの構造を確認 ★★★
        print(f"Model type: {type(model)}")
        print(f"Model attributes: {[attr for attr in dir(model) if not attr.startswith('_')]}")
        if hasattr(model, 'model'):
            print(f"model.model type: {type(model.model)}")
            print(f"model.model attributes: {[attr for attr in dir(model.model) if not attr.startswith('_')]}")
            if hasattr(model.model, 'layers'):
                print(f"model.model.layers length: {len(model.model.layers)}")
                print(f"First layer type: {type(model.model.layers[0])}")
                print(f"First layer attributes: {[attr for attr in dir(model.model.layers[0]) if not attr.startswith('_')]}")
                if hasattr(model.model.layers[0], 'mlp'):
                    print(f"First layer has mlp: {type(model.model.layers[0].mlp)}")

        # ★★★ 処方箋1: 入力の勾配を強制的に有効化 ★★★
        # これがないと、Gradient Checkpointing有効時に途中の勾配が死ぬ
        model.enable_input_require_grads()

        # LoRA適用
        print("[2/5] Applying LoRA...")
        lora_config = LoraConfig(**config.lora_config)
        model = get_peft_model(model, lora_config)

        # LoRAパラメータを明示的にトレーニング可能に（詳細確認）
        lora_params_count = 0
        for name, param in model.named_parameters():
            if 'lora' in name.lower():
                param.requires_grad_(True)
                lora_params_count += 1

        print(f"Set {lora_params_count} LoRA parameters to trainable")

        # SO(8)アダプター適用（so8tの場合）
        if config.so8_config:
            print("[3/5] Applying SO(8) adapters (Layer Replacement Mode)...")
            enable_quad = config.so8_config.get('enable_quad_inference', False)
            # ★★★ 最終奥義: モンキーパッチで注入（Unsloth最適化突破）★★★
            model = replace_mlp_with_nkat(
                model,
                target_layers=config.so8_config['target_layers']
            )

            print(f"SO(8) Adapter with Quad Inference: {enable_quad}")
        else:
            print("[3/5] Skipping SO(8) adapters (baseline run)")

        # パラメータ確認
        print("[4/5] Checking trainable parameters...")
        model.print_trainable_parameters()

        # 🔥 緊急バイパス手術：Optimizerへの手動登録（SO8Tの場合のみ）
        if config.so8_config:
            print("[4.5/5] Manual optimizer registration for SO8T...")
            # 1. 学習対象パラメータの抽出 (LoRA と NKATアダプタ だけ)
            trainable_params = []
            for name, param in model.named_parameters():
                if "lora" in name.lower() or "nkat_adapter" in name.lower():
                    param.requires_grad = True
                    trainable_params.append(param)
                else:
                    param.requires_grad = False

            print(f"🔥 Total Trainable Params: {len(trainable_params)} tensors")

            # 2. Optimizerの手動作成 (Unsloth推奨の8bit AdamWを使う場合)
            try:
                from unsloth.optim import AdamW8bit
                optimizer = AdamW8bit(trainable_params, lr=config.training_config.get('learning_rate', 2e-5))
                print("✅ Using Unsloth AdamW8bit")
            except ImportError:
                from torch.optim import AdamW
                optimizer = AdamW(trainable_params, lr=config.training_config.get('learning_rate', 2e-5))
                print("✅ Using Standard AdamW")

            # 3. 後でTrainerに渡すための保存
            manual_optimizer = optimizer
        else:
            manual_optimizer = None

        # データセット準備
        print("[5/5] Preparing dataset...")
        dataset = SimpleDataset(
            config.sft_datasets[0],  # 最初のデータセットを使用
            tokenizer
        )
        print(f"Dataset size: {len(dataset)} samples")

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )

        # トレーニング引数
        training_args = TrainingArguments(**config.training_config)

        # カスタムコールバックでSO(8)メトリクス収集
        class SunshineCallback(TrainerCallback):
            def __init__(self, logger, model, run_type):
                self.logger = logger
                self.model = model
                self.run_type = run_type
                self.step_start_time = None

            def on_init_end(self, args, state, control, **kwargs):
                pass

            def on_train_begin(self, args, state, control, **kwargs):
                pass

            def on_train_end(self, args, state, control, **kwargs):
                pass

            def on_step_begin(self, args, state, control, **kwargs):
                self.step_start_time = time.time()

            def on_step_end(self, args, state, control, **kwargs):
                if self.step_start_time is None:
                    return

                step_time = time.time() - self.step_start_time
                step = state.global_step

                # SO(8)メトリクス収集
                so8_metrics = {}
                if self.run_type == "so8t":
                    ortho_errors = []
                    alphas = []
                    for name, module in self.model.named_modules():
                        if hasattr(module, 'get_adapter_stats'):
                            stats = module.get_adapter_stats()
                            ortho_errors.append(stats['orthogonality_error'])
                            alphas.append(stats['alpha'])

                    if ortho_errors:
                        so8_metrics['so8_ortho_mean'] = np.mean(ortho_errors)
                        so8_metrics['so8_ortho_max'] = np.max(ortho_errors)
                        so8_metrics['so8_alpha_mean'] = np.mean(alphas)
                        so8_metrics['so8_alpha_std'] = np.std(alphas) if len(alphas) > 1 else 0

                # 勾配ノルム（利用可能なら）
                grad_norm = None
                if hasattr(state, 'log_history') and state.log_history:
                    last_log = state.log_history[-1]
                    grad_norm = last_log.get('grad_norm')

                # ログ記録
                metrics = {
                    'train_loss': state.log_history[-1].get('loss') if state.log_history else None,
                    **so8_metrics,
                    'grad_norm': grad_norm
                }

                self.logger.log_step(step, metrics, step_time)

        callback = SunshineCallback(logger, model, run_type)

        # Trainer設定
        if manual_optimizer is not None:
            # SO8Tの場合：手動Optimizerを使用
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=dataset,
                data_collator=data_collator,
                callbacks=[callback, NKATDebugCallback(model)],
                optimizers=(manual_optimizer, None)  # (optimizer, scheduler)
            )
            print("🔧 Using manual optimizer for SO8T training")
        else:
            # Baselineの場合：通常のTrainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=dataset,
                data_collator=data_collator,
                callbacks=[callback, NKATDebugCallback(model)]
            )

        # トレーニング実行
        print(f"🚀 Starting {run_type.upper()} training...")
        trainer.train()

        # 最終メトリクス
        final_metrics = {}
        if trainer.state.log_history:
            final_metrics = trainer.state.log_history[-1]

        logger.finalize(final_metrics)

        print(f"✅ {run_type.upper()} training completed!")
        print(f"📊 Results saved to: {logger.log_dir}")

        return {
            'success': True,
            'run_type': run_type,
            'log_dir': str(logger.log_dir),
            'metrics_file': str(logger.metrics_file),
            'final_loss': final_metrics.get('loss')
        }

    except Exception as e:
        print(f"❌ {run_type.upper()} training failed: {e}")
        logger.finalize({'error': str(e)})
        return {
            'success': False,
            'run_type': run_type,
            'error': str(e)
        }


def run_sunshine_pipeline():
    """サンシャインパイプライン実行"""
    print("🌞🌞🌞 SO8T SUNSHINE PIPELINE 🌞🌞🌞")
    print("Comparing Baseline vs SO8T performance")
    print("=" * 60)

    results = {}

    # Run A: Baseline
    print("\n🏃 Run A: BASELINE (LoRA only)")
    results['baseline'] = run_sunshine_experiment("baseline")

    # Run B: SO8T
    print("\n🧬 Run B: SO8T (LoRA + SO(8) Adapter)")
    results['so8t'] = run_sunshine_experiment("so8t")

    # 結果比較
    print("\n📊 EXPERIMENT RESULTS SUMMARY")
    print("=" * 60)

    for run_type, result in results.items():
        status = "✅ SUCCESS" if result['success'] else "❌ FAILED"
        loss = f"Final Loss: {result.get('final_loss', 'N/A')}"
        print(f"{run_type.upper()}: {status} | {loss}")

    # ログファイル保存
    summary_file = Path("logs/sunshine") / "sunshine_summary.json"
    summary_file.parent.mkdir(parents=True, exist_ok=True)

    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'results': results
        }, f, indent=2, ensure_ascii=False)

    print(f"\n📋 Summary saved to: {summary_file}")

    # オーディオ通知
    try:
        import winsound
        winsound.Beep(1000, 1000)  # 1秒のビープ音
    except:
        pass

    return results


if __name__ == "__main__":
    # 引数処理
    if len(sys.argv) > 1:
        run_type = sys.argv[1]
        if run_type in ["baseline", "so8t"]:
            result = run_sunshine_experiment(run_type)
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print("Usage: python sunshine_pipeline.py [baseline|so8t]")
            sys.exit(1)
    else:
        # フルパイプライン実行
        run_sunshine_pipeline()

"""
SO8T Sunshine Pipeline - 実験計画ベースの自動実行システム

ボブにゃん戦略に基づくサンシャイン実行：
- Run A: Baseline (LoRAのみ)
- Run B: SO8T (LoRA + SO(8)アダプター)

ログフォーマット：
step, train_loss, eval_loss, so8_ortho_mean, so8_ortho_max, grad_norm, step_time_sec
"""

import os
import sys
import time
import json
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model

# Import from scripts directory
# from scripts.models.so8t_residual_adapter import attach_nkat_adapters  # 削除済み関数

# Simple dataset class for testing
from torch.utils.data import Dataset
import json

class SimpleDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, max_length: int = 128):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Load data
        self.data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        item = json.loads(line.strip())
                        self.data.append(item)
                    except json.JSONDecodeError:
                        continue

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item.get('text', '')

        # Tokenize
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
            'labels': tokenized['input_ids'].squeeze()
        }


class SunshineLogger:
    """サンシャイン実行の統一ログ収集"""

    def __init__(self, log_dir: str, run_name: str):
        self.log_dir = Path(log_dir)
        self.run_name = run_name
        self.log_file = self.log_dir / f"{run_name}_training_log.csv"
        self.metrics_file = self.log_dir / f"{run_name}_metrics.json"

        # CSVヘッダー
        self.columns = [
            'step', 'train_loss', 'eval_loss',
            'so8_ortho_mean', 'so8_ortho_max',
            'grad_norm', 'step_time_sec'
        ]

        # ログディレクトリ作成
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # CSV初期化
        if not self.log_file.exists():
            pd.DataFrame(columns=self.columns).to_csv(self.log_file, index=False)

        # メトリクス初期化
        self.metrics = {
            'start_time': datetime.now().isoformat(),
            'run_name': run_name,
            'total_steps': 0,
            'avg_step_time': 0,
            'final_train_loss': None,
            'final_eval_loss': None,
            'so8_ortho_errors': [],
            'grad_norms': []
        }

    def log_step(self, step: int, metrics: Dict[str, Any], step_time: float):
        """ステップごとのログ記録"""
        row = {
            'step': step,
            'train_loss': metrics.get('train_loss', None),
            'eval_loss': metrics.get('eval_loss', None),
            'so8_ortho_mean': metrics.get('so8_ortho_mean', None),
            'so8_ortho_max': metrics.get('so8_ortho_max', None),
            'grad_norm': metrics.get('grad_norm', None),
            'step_time_sec': step_time
        }

        # CSVに追加
        df = pd.DataFrame([row])
        df.to_csv(self.log_file, mode='a', header=False, index=False)

        # メトリクス更新
        if metrics.get('so8_ortho_mean') is not None:
            self.metrics['so8_ortho_errors'].append(metrics['so8_ortho_mean'])
        if metrics.get('grad_norm') is not None:
            self.metrics['grad_norms'].append(metrics['grad_norm'])

        self.metrics['total_steps'] = max(self.metrics['total_steps'], step)

    def finalize(self, final_metrics: Dict[str, Any]):
        """トレーニング完了時の最終ログ"""
        self.metrics.update({
            'end_time': datetime.now().isoformat(),
            'final_train_loss': final_metrics.get('train_loss'),
            'final_eval_loss': final_metrics.get('eval_loss'),
            'avg_so8_ortho_error': np.mean(self.metrics['so8_ortho_errors']) if self.metrics['so8_ortho_errors'] else None,
            'max_so8_ortho_error': np.max(self.metrics['so8_ortho_errors']) if self.metrics['so8_ortho_errors'] else None,
            'avg_grad_norm': np.mean(self.metrics['grad_norms']) if self.metrics['grad_norms'] else None
        })

        # JSON保存
        with open(self.metrics_file, 'w', encoding='utf-8') as f:
            json.dump(self.metrics, f, indent=2, ensure_ascii=False)


class SunshineTrainingConfig:
    """サンシャイン実行設定"""

    def __init__(self, run_type: str = "baseline"):
        self.model_name = "AXCXEPT/Borea-Phi-3.5-mini-Instruct-Jp"
        self.run_type = run_type  # "baseline" or "so8t"

        # データセット設定（統合データセットを使用）
        if run_type == "baseline":
            # Baseline用: 数学・科学統合データ
            self.sft_datasets = ["data/train_sft_enhanced.jsonl"]
        else:
            # SO8T用: NKAT理論・NSFWデータ統合
            self.sft_datasets = ["data/aegis_phi35_v2_with_nkat_so8t/aegis_phi35_v2_with_nkat_so8t_sft_train.jsonl"]

        # ドメイン重み付け
        self.domain_weights = {
            'mathematics': 1.2,
            'science': 1.1,
            'reasoning': 1.0,
            'general': 0.8
        }

        # トレーニング設定（本格データセット対応）
        self.training_config = {
            'output_dir': f"H:/from_D/webdataset/checkpoints/sunshine_{run_type}_phase25",
            'num_train_epochs': 1,
            'max_steps': 50,  # テスト用に減らす
            'per_device_train_batch_size': 1,
            'gradient_accumulation_steps': 8,  # RTX3060対応
            'learning_rate': 3e-5,  # 安定した学習率
            'save_steps': 100,
            'logging_steps': 10,  # 10ステップごとログ
            'eval_steps': 100,
            'gradient_checkpointing': False,  # 一時的にOFFにして勾配問題を解決
            'ddp_find_unused_parameters': False,  # アダプタパラメータの検出を確実にする
            'optim': "adamw_8bit",
            'bf16': True,
            'dataloader_pin_memory': False,
            'remove_unused_columns': False,
            'report_to': [],
            'load_best_model_at_end': False
        }

        # LoRA設定
        self.lora_config = {
            'r': 16,
            'lora_alpha': 32,
            'lora_dropout': 0.05,
            'bias': "none",
            'task_type': "CAUSAL_LM",
            'target_modules': ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        }

        # SO(8)設定（so8tの場合のみ）
        if run_type == "so8t":
            self.so8_config = {
                'target_layers': [8, 16, 24],  # 中間層のみ
                'so8_dim': 8,
                'alpha_init': 0.1
            }
        else:
            self.so8_config = None


def run_sunshine_experiment(run_type: str = "baseline") -> Dict[str, Any]:
    """
    サンシャイン実験実行
    run_type: "baseline" or "so8t"
    """
    print(f"🌞 Starting Sunshine Experiment: {run_type.upper()}")
    print("=" * 60)

    # 設定
    config = SunshineTrainingConfig(run_type)
    run_name = f"sunshine_run_{run_type}"

    # ロガー初期化
    logger = SunshineLogger("logs/sunshine", run_name)

    try:
        # モデルとトークナイザーロード
        print("[1/5] Loading model and tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # ★★★ トレーニング時は device_map="auto" を使用せず直接GPUにロード ★★★
        # device_map="auto" は分散トレーニングと競合するため
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float16,
            device_map=None  # トレーニング時は明示的にNone
        ).to("cuda")  # 直接GPUに移動

        # ★★★ デバッグ: Phi-3モデルの構造を確認 ★★★
        print(f"Model type: {type(model)}")
        print(f"Model attributes: {[attr for attr in dir(model) if not attr.startswith('_')]}")
        if hasattr(model, 'model'):
            print(f"model.model type: {type(model.model)}")
            print(f"model.model attributes: {[attr for attr in dir(model.model) if not attr.startswith('_')]}")
            if hasattr(model.model, 'layers'):
                print(f"model.model.layers length: {len(model.model.layers)}")
                print(f"First layer type: {type(model.model.layers[0])}")
                print(f"First layer attributes: {[attr for attr in dir(model.model.layers[0]) if not attr.startswith('_')]}")
                if hasattr(model.model.layers[0], 'mlp'):
                    print(f"First layer has mlp: {type(model.model.layers[0].mlp)}")

        # ★★★ 処方箋1: 入力の勾配を強制的に有効化 ★★★
        # これがないと、Gradient Checkpointing有効時に途中の勾配が死ぬ
        model.enable_input_require_grads()

        # LoRA適用
        print("[2/5] Applying LoRA...")
        lora_config = LoraConfig(**config.lora_config)
        model = get_peft_model(model, lora_config)

        # LoRAパラメータを明示的にトレーニング可能に（詳細確認）
        lora_params_count = 0
        for name, param in model.named_parameters():
            if 'lora' in name.lower():
                param.requires_grad_(True)
                lora_params_count += 1

        print(f"Set {lora_params_count} LoRA parameters to trainable")

        # SO(8)アダプター適用（so8tの場合）
        if config.so8_config:
            print("[3/5] Applying SO(8) adapters (Layer Replacement Mode)...")
            enable_quad = config.so8_config.get('enable_quad_inference', False)
            # ★★★ 最終奥義: モンキーパッチで注入（Unsloth最適化突破）★★★
            model = replace_mlp_with_nkat(
                model,
                target_layers=config.so8_config['target_layers']
            )

            print(f"SO(8) Adapter with Quad Inference: {enable_quad}")
        else:
            print("[3/5] Skipping SO(8) adapters (baseline run)")

        # パラメータ確認
        print("[4/5] Checking trainable parameters...")
        model.print_trainable_parameters()

        # 🔥 緊急バイパス手術：Optimizerへの手動登録（SO8Tの場合のみ）
        if config.so8_config:
            print("[4.5/5] Manual optimizer registration for SO8T...")
            # 1. 学習対象パラメータの抽出 (LoRA と NKATアダプタ だけ)
            trainable_params = []
            for name, param in model.named_parameters():
                if "lora" in name.lower() or "nkat_adapter" in name.lower():
                    param.requires_grad = True
                    trainable_params.append(param)
                else:
                    param.requires_grad = False

            print(f"🔥 Total Trainable Params: {len(trainable_params)} tensors")

            # 2. Optimizerの手動作成 (Unsloth推奨の8bit AdamWを使う場合)
            try:
                from unsloth.optim import AdamW8bit
                optimizer = AdamW8bit(trainable_params, lr=config.training_config.get('learning_rate', 2e-5))
                print("✅ Using Unsloth AdamW8bit")
            except ImportError:
                from torch.optim import AdamW
                optimizer = AdamW(trainable_params, lr=config.training_config.get('learning_rate', 2e-5))
                print("✅ Using Standard AdamW")

            # 3. 後でTrainerに渡すための保存
            manual_optimizer = optimizer
        else:
            manual_optimizer = None

        # データセット準備
        print("[5/5] Preparing dataset...")
        dataset = SimpleDataset(
            config.sft_datasets[0],  # 最初のデータセットを使用
            tokenizer
        )
        print(f"Dataset size: {len(dataset)} samples")

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )

        # トレーニング引数
        training_args = TrainingArguments(**config.training_config)

        # カスタムコールバックでSO(8)メトリクス収集
        class SunshineCallback(TrainerCallback):
            def __init__(self, logger, model, run_type):
                self.logger = logger
                self.model = model
                self.run_type = run_type
                self.step_start_time = None

            def on_init_end(self, args, state, control, **kwargs):
                pass

            def on_train_begin(self, args, state, control, **kwargs):
                pass

            def on_train_end(self, args, state, control, **kwargs):
                pass

            def on_step_begin(self, args, state, control, **kwargs):
                self.step_start_time = time.time()

            def on_step_end(self, args, state, control, **kwargs):
                if self.step_start_time is None:
                    return

                step_time = time.time() - self.step_start_time
                step = state.global_step

                # SO(8)メトリクス収集
                so8_metrics = {}
                if self.run_type == "so8t":
                    ortho_errors = []
                    alphas = []
                    for name, module in self.model.named_modules():
                        if hasattr(module, 'get_adapter_stats'):
                            stats = module.get_adapter_stats()
                            ortho_errors.append(stats['orthogonality_error'])
                            alphas.append(stats['alpha'])

                    if ortho_errors:
                        so8_metrics['so8_ortho_mean'] = np.mean(ortho_errors)
                        so8_metrics['so8_ortho_max'] = np.max(ortho_errors)
                        so8_metrics['so8_alpha_mean'] = np.mean(alphas)
                        so8_metrics['so8_alpha_std'] = np.std(alphas) if len(alphas) > 1 else 0

                # 勾配ノルム（利用可能なら）
                grad_norm = None
                if hasattr(state, 'log_history') and state.log_history:
                    last_log = state.log_history[-1]
                    grad_norm = last_log.get('grad_norm')

                # ログ記録
                metrics = {
                    'train_loss': state.log_history[-1].get('loss') if state.log_history else None,
                    **so8_metrics,
                    'grad_norm': grad_norm
                }

                self.logger.log_step(step, metrics, step_time)

        callback = SunshineCallback(logger, model, run_type)

        # Trainer設定
        if manual_optimizer is not None:
            # SO8Tの場合：手動Optimizerを使用
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=dataset,
                data_collator=data_collator,
                callbacks=[callback, NKATDebugCallback(model)],
                optimizers=(manual_optimizer, None)  # (optimizer, scheduler)
            )
            print("🔧 Using manual optimizer for SO8T training")
        else:
            # Baselineの場合：通常のTrainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=dataset,
                data_collator=data_collator,
                callbacks=[callback, NKATDebugCallback(model)]
            )

        # トレーニング実行
        print(f"🚀 Starting {run_type.upper()} training...")
        trainer.train()

        # 最終メトリクス
        final_metrics = {}
        if trainer.state.log_history:
            final_metrics = trainer.state.log_history[-1]

        logger.finalize(final_metrics)

        print(f"✅ {run_type.upper()} training completed!")
        print(f"📊 Results saved to: {logger.log_dir}")

        return {
            'success': True,
            'run_type': run_type,
            'log_dir': str(logger.log_dir),
            'metrics_file': str(logger.metrics_file),
            'final_loss': final_metrics.get('loss')
        }

    except Exception as e:
        print(f"❌ {run_type.upper()} training failed: {e}")
        logger.finalize({'error': str(e)})
        return {
            'success': False,
            'run_type': run_type,
            'error': str(e)
        }


def run_sunshine_pipeline():
    """サンシャインパイプライン実行"""
    print("🌞🌞🌞 SO8T SUNSHINE PIPELINE 🌞🌞🌞")
    print("Comparing Baseline vs SO8T performance")
    print("=" * 60)

    results = {}

    # Run A: Baseline
    print("\n🏃 Run A: BASELINE (LoRA only)")
    results['baseline'] = run_sunshine_experiment("baseline")

    # Run B: SO8T
    print("\n🧬 Run B: SO8T (LoRA + SO(8) Adapter)")
    results['so8t'] = run_sunshine_experiment("so8t")

    # 結果比較
    print("\n📊 EXPERIMENT RESULTS SUMMARY")
    print("=" * 60)

    for run_type, result in results.items():
        status = "✅ SUCCESS" if result['success'] else "❌ FAILED"
        loss = f"Final Loss: {result.get('final_loss', 'N/A')}"
        print(f"{run_type.upper()}: {status} | {loss}")

    # ログファイル保存
    summary_file = Path("logs/sunshine") / "sunshine_summary.json"
    summary_file.parent.mkdir(parents=True, exist_ok=True)

    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'results': results
        }, f, indent=2, ensure_ascii=False)

    print(f"\n📋 Summary saved to: {summary_file}")

    # オーディオ通知
    try:
        import winsound
        winsound.Beep(1000, 1000)  # 1秒のビープ音
    except:
        pass

    return results


if __name__ == "__main__":
    # 引数処理
    if len(sys.argv) > 1:
        run_type = sys.argv[1]
        if run_type in ["baseline", "so8t"]:
            result = run_sunshine_experiment(run_type)
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print("Usage: python sunshine_pipeline.py [baseline|so8t]")
            sys.exit(1)
    else:
        # フルパイプライン実行
        run_sunshine_pipeline()

"""
SO8T Sunshine Pipeline - 実験計画ベースの自動実行システム

ボブにゃん戦略に基づくサンシャイン実行：
- Run A: Baseline (LoRAのみ)
- Run B: SO8T (LoRA + SO(8)アダプター)

ログフォーマット：
step, train_loss, eval_loss, so8_ortho_mean, so8_ortho_max, grad_norm, step_time_sec
"""

import os
import sys
import time
import json
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model

# Import from scripts directory
# from scripts.models.so8t_residual_adapter import attach_nkat_adapters  # 削除済み関数

# Simple dataset class for testing
from torch.utils.data import Dataset
import json

class SimpleDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, max_length: int = 128):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Load data
        self.data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        item = json.loads(line.strip())
                        self.data.append(item)
                    except json.JSONDecodeError:
                        continue

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item.get('text', '')

        # Tokenize
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
            'labels': tokenized['input_ids'].squeeze()
        }


class SunshineLogger:
    """サンシャイン実行の統一ログ収集"""

    def __init__(self, log_dir: str, run_name: str):
        self.log_dir = Path(log_dir)
        self.run_name = run_name
        self.log_file = self.log_dir / f"{run_name}_training_log.csv"
        self.metrics_file = self.log_dir / f"{run_name}_metrics.json"

        # CSVヘッダー
        self.columns = [
            'step', 'train_loss', 'eval_loss',
            'so8_ortho_mean', 'so8_ortho_max',
            'grad_norm', 'step_time_sec'
        ]

        # ログディレクトリ作成
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # CSV初期化
        if not self.log_file.exists():
            pd.DataFrame(columns=self.columns).to_csv(self.log_file, index=False)

        # メトリクス初期化
        self.metrics = {
            'start_time': datetime.now().isoformat(),
            'run_name': run_name,
            'total_steps': 0,
            'avg_step_time': 0,
            'final_train_loss': None,
            'final_eval_loss': None,
            'so8_ortho_errors': [],
            'grad_norms': []
        }

    def log_step(self, step: int, metrics: Dict[str, Any], step_time: float):
        """ステップごとのログ記録"""
        row = {
            'step': step,
            'train_loss': metrics.get('train_loss', None),
            'eval_loss': metrics.get('eval_loss', None),
            'so8_ortho_mean': metrics.get('so8_ortho_mean', None),
            'so8_ortho_max': metrics.get('so8_ortho_max', None),
            'grad_norm': metrics.get('grad_norm', None),
            'step_time_sec': step_time
        }

        # CSVに追加
        df = pd.DataFrame([row])
        df.to_csv(self.log_file, mode='a', header=False, index=False)

        # メトリクス更新
        if metrics.get('so8_ortho_mean') is not None:
            self.metrics['so8_ortho_errors'].append(metrics['so8_ortho_mean'])
        if metrics.get('grad_norm') is not None:
            self.metrics['grad_norms'].append(metrics['grad_norm'])

        self.metrics['total_steps'] = max(self.metrics['total_steps'], step)

    def finalize(self, final_metrics: Dict[str, Any]):
        """トレーニング完了時の最終ログ"""
        self.metrics.update({
            'end_time': datetime.now().isoformat(),
            'final_train_loss': final_metrics.get('train_loss'),
            'final_eval_loss': final_metrics.get('eval_loss'),
            'avg_so8_ortho_error': np.mean(self.metrics['so8_ortho_errors']) if self.metrics['so8_ortho_errors'] else None,
            'max_so8_ortho_error': np.max(self.metrics['so8_ortho_errors']) if self.metrics['so8_ortho_errors'] else None,
            'avg_grad_norm': np.mean(self.metrics['grad_norms']) if self.metrics['grad_norms'] else None
        })

        # JSON保存
        with open(self.metrics_file, 'w', encoding='utf-8') as f:
            json.dump(self.metrics, f, indent=2, ensure_ascii=False)


class SunshineTrainingConfig:
    """サンシャイン実行設定"""

    def __init__(self, run_type: str = "baseline"):
        self.model_name = "AXCXEPT/Borea-Phi-3.5-mini-Instruct-Jp"
        self.run_type = run_type  # "baseline" or "so8t"

        # データセット設定（統合データセットを使用）
        if run_type == "baseline":
            # Baseline用: 数学・科学統合データ
            self.sft_datasets = ["data/train_sft_enhanced.jsonl"]
        else:
            # SO8T用: NKAT理論・NSFWデータ統合
            self.sft_datasets = ["data/aegis_phi35_v2_with_nkat_so8t/aegis_phi35_v2_with_nkat_so8t_sft_train.jsonl"]

        # ドメイン重み付け
        self.domain_weights = {
            'mathematics': 1.2,
            'science': 1.1,
            'reasoning': 1.0,
            'general': 0.8
        }

        # トレーニング設定（本格データセット対応）
        self.training_config = {
            'output_dir': f"H:/from_D/webdataset/checkpoints/sunshine_{run_type}_phase25",
            'num_train_epochs': 1,
            'max_steps': 50,  # テスト用に減らす
            'per_device_train_batch_size': 1,
            'gradient_accumulation_steps': 8,  # RTX3060対応
            'learning_rate': 3e-5,  # 安定した学習率
            'save_steps': 100,
            'logging_steps': 10,  # 10ステップごとログ
            'eval_steps': 100,
            'gradient_checkpointing': False,  # 一時的にOFFにして勾配問題を解決
            'ddp_find_unused_parameters': False,  # アダプタパラメータの検出を確実にする
            'optim': "adamw_8bit",
            'bf16': True,
            'dataloader_pin_memory': False,
            'remove_unused_columns': False,
            'report_to': [],
            'load_best_model_at_end': False
        }

        # LoRA設定
        self.lora_config = {
            'r': 16,
            'lora_alpha': 32,
            'lora_dropout': 0.05,
            'bias': "none",
            'task_type': "CAUSAL_LM",
            'target_modules': ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        }

        # SO(8)設定（so8tの場合のみ）
        if run_type == "so8t":
            self.so8_config = {
                'target_layers': [8, 16, 24],  # 中間層のみ
                'so8_dim': 8,
                'alpha_init': 0.1
            }
        else:
            self.so8_config = None


def run_sunshine_experiment(run_type: str = "baseline") -> Dict[str, Any]:
    """
    サンシャイン実験実行
    run_type: "baseline" or "so8t"
    """
    print(f"🌞 Starting Sunshine Experiment: {run_type.upper()}")
    print("=" * 60)

    # 設定
    config = SunshineTrainingConfig(run_type)
    run_name = f"sunshine_run_{run_type}"

    # ロガー初期化
    logger = SunshineLogger("logs/sunshine", run_name)

    try:
        # モデルとトークナイザーロード
        print("[1/5] Loading model and tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # ★★★ トレーニング時は device_map="auto" を使用せず直接GPUにロード ★★★
        # device_map="auto" は分散トレーニングと競合するため
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float16,
            device_map=None  # トレーニング時は明示的にNone
        ).to("cuda")  # 直接GPUに移動

        # ★★★ デバッグ: Phi-3モデルの構造を確認 ★★★
        print(f"Model type: {type(model)}")
        print(f"Model attributes: {[attr for attr in dir(model) if not attr.startswith('_')]}")
        if hasattr(model, 'model'):
            print(f"model.model type: {type(model.model)}")
            print(f"model.model attributes: {[attr for attr in dir(model.model) if not attr.startswith('_')]}")
            if hasattr(model.model, 'layers'):
                print(f"model.model.layers length: {len(model.model.layers)}")
                print(f"First layer type: {type(model.model.layers[0])}")
                print(f"First layer attributes: {[attr for attr in dir(model.model.layers[0]) if not attr.startswith('_')]}")
                if hasattr(model.model.layers[0], 'mlp'):
                    print(f"First layer has mlp: {type(model.model.layers[0].mlp)}")

        # ★★★ 処方箋1: 入力の勾配を強制的に有効化 ★★★
        # これがないと、Gradient Checkpointing有効時に途中の勾配が死ぬ
        model.enable_input_require_grads()

        # LoRA適用
        print("[2/5] Applying LoRA...")
        lora_config = LoraConfig(**config.lora_config)
        model = get_peft_model(model, lora_config)

        # LoRAパラメータを明示的にトレーニング可能に（詳細確認）
        lora_params_count = 0
        for name, param in model.named_parameters():
            if 'lora' in name.lower():
                param.requires_grad_(True)
                lora_params_count += 1

        print(f"Set {lora_params_count} LoRA parameters to trainable")

        # SO(8)アダプター適用（so8tの場合）
        if config.so8_config:
            print("[3/5] Applying SO(8) adapters (Layer Replacement Mode)...")
            enable_quad = config.so8_config.get('enable_quad_inference', False)
            # ★★★ 最終奥義: モンキーパッチで注入（Unsloth最適化突破）★★★
            model = replace_mlp_with_nkat(
                model,
                target_layers=config.so8_config['target_layers']
            )

            print(f"SO(8) Adapter with Quad Inference: {enable_quad}")
        else:
            print("[3/5] Skipping SO(8) adapters (baseline run)")

        # パラメータ確認
        print("[4/5] Checking trainable parameters...")
        model.print_trainable_parameters()

        # 🔥 緊急バイパス手術：Optimizerへの手動登録（SO8Tの場合のみ）
        if config.so8_config:
            print("[4.5/5] Manual optimizer registration for SO8T...")
            # 1. 学習対象パラメータの抽出 (LoRA と NKATアダプタ だけ)
            trainable_params = []
            for name, param in model.named_parameters():
                if "lora" in name.lower() or "nkat_adapter" in name.lower():
                    param.requires_grad = True
                    trainable_params.append(param)
                else:
                    param.requires_grad = False

            print(f"🔥 Total Trainable Params: {len(trainable_params)} tensors")

            # 2. Optimizerの手動作成 (Unsloth推奨の8bit AdamWを使う場合)
            try:
                from unsloth.optim import AdamW8bit
                optimizer = AdamW8bit(trainable_params, lr=config.training_config.get('learning_rate', 2e-5))
                print("✅ Using Unsloth AdamW8bit")
            except ImportError:
                from torch.optim import AdamW
                optimizer = AdamW(trainable_params, lr=config.training_config.get('learning_rate', 2e-5))
                print("✅ Using Standard AdamW")

            # 3. 後でTrainerに渡すための保存
            manual_optimizer = optimizer
        else:
            manual_optimizer = None

        # データセット準備
        print("[5/5] Preparing dataset...")
        dataset = SimpleDataset(
            config.sft_datasets[0],  # 最初のデータセットを使用
            tokenizer
        )
        print(f"Dataset size: {len(dataset)} samples")

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )

        # トレーニング引数
        training_args = TrainingArguments(**config.training_config)

        # カスタムコールバックでSO(8)メトリクス収集
        class SunshineCallback(TrainerCallback):
            def __init__(self, logger, model, run_type):
                self.logger = logger
                self.model = model
                self.run_type = run_type
                self.step_start_time = None

            def on_init_end(self, args, state, control, **kwargs):
                pass

            def on_train_begin(self, args, state, control, **kwargs):
                pass

            def on_train_end(self, args, state, control, **kwargs):
                pass

            def on_step_begin(self, args, state, control, **kwargs):
                self.step_start_time = time.time()

            def on_step_end(self, args, state, control, **kwargs):
                if self.step_start_time is None:
                    return

                step_time = time.time() - self.step_start_time
                step = state.global_step

                # SO(8)メトリクス収集
                so8_metrics = {}
                if self.run_type == "so8t":
                    ortho_errors = []
                    alphas = []
                    for name, module in self.model.named_modules():
                        if hasattr(module, 'get_adapter_stats'):
                            stats = module.get_adapter_stats()
                            ortho_errors.append(stats['orthogonality_error'])
                            alphas.append(stats['alpha'])

                    if ortho_errors:
                        so8_metrics['so8_ortho_mean'] = np.mean(ortho_errors)
                        so8_metrics['so8_ortho_max'] = np.max(ortho_errors)
                        so8_metrics['so8_alpha_mean'] = np.mean(alphas)
                        so8_metrics['so8_alpha_std'] = np.std(alphas) if len(alphas) > 1 else 0

                # 勾配ノルム（利用可能なら）
                grad_norm = None
                if hasattr(state, 'log_history') and state.log_history:
                    last_log = state.log_history[-1]
                    grad_norm = last_log.get('grad_norm')

                # ログ記録
                metrics = {
                    'train_loss': state.log_history[-1].get('loss') if state.log_history else None,
                    **so8_metrics,
                    'grad_norm': grad_norm
                }

                self.logger.log_step(step, metrics, step_time)

        callback = SunshineCallback(logger, model, run_type)

        # Trainer設定
        if manual_optimizer is not None:
            # SO8Tの場合：手動Optimizerを使用
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=dataset,
                data_collator=data_collator,
                callbacks=[callback, NKATDebugCallback(model)],
                optimizers=(manual_optimizer, None)  # (optimizer, scheduler)
            )
            print("🔧 Using manual optimizer for SO8T training")
        else:
            # Baselineの場合：通常のTrainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=dataset,
                data_collator=data_collator,
                callbacks=[callback, NKATDebugCallback(model)]
            )

        # トレーニング実行
        print(f"🚀 Starting {run_type.upper()} training...")
        trainer.train()

        # 最終メトリクス
        final_metrics = {}
        if trainer.state.log_history:
            final_metrics = trainer.state.log_history[-1]

        logger.finalize(final_metrics)

        print(f"✅ {run_type.upper()} training completed!")
        print(f"📊 Results saved to: {logger.log_dir}")

        return {
            'success': True,
            'run_type': run_type,
            'log_dir': str(logger.log_dir),
            'metrics_file': str(logger.metrics_file),
            'final_loss': final_metrics.get('loss')
        }

    except Exception as e:
        print(f"❌ {run_type.upper()} training failed: {e}")
        logger.finalize({'error': str(e)})
        return {
            'success': False,
            'run_type': run_type,
            'error': str(e)
        }


def run_sunshine_pipeline():
    """サンシャインパイプライン実行"""
    print("🌞🌞🌞 SO8T SUNSHINE PIPELINE 🌞🌞🌞")
    print("Comparing Baseline vs SO8T performance")
    print("=" * 60)

    results = {}

    # Run A: Baseline
    print("\n🏃 Run A: BASELINE (LoRA only)")
    results['baseline'] = run_sunshine_experiment("baseline")

    # Run B: SO8T
    print("\n🧬 Run B: SO8T (LoRA + SO(8) Adapter)")
    results['so8t'] = run_sunshine_experiment("so8t")

    # 結果比較
    print("\n📊 EXPERIMENT RESULTS SUMMARY")
    print("=" * 60)

    for run_type, result in results.items():
        status = "✅ SUCCESS" if result['success'] else "❌ FAILED"
        loss = f"Final Loss: {result.get('final_loss', 'N/A')}"
        print(f"{run_type.upper()}: {status} | {loss}")

    # ログファイル保存
    summary_file = Path("logs/sunshine") / "sunshine_summary.json"
    summary_file.parent.mkdir(parents=True, exist_ok=True)

    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'results': results
        }, f, indent=2, ensure_ascii=False)

    print(f"\n📋 Summary saved to: {summary_file}")

    # オーディオ通知
    try:
        import winsound
        winsound.Beep(1000, 1000)  # 1秒のビープ音
    except:
        pass

    return results


if __name__ == "__main__":
    # 引数処理
    if len(sys.argv) > 1:
        run_type = sys.argv[1]
        if run_type in ["baseline", "so8t"]:
            result = run_sunshine_experiment(run_type)
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print("Usage: python sunshine_pipeline.py [baseline|so8t]")
            sys.exit(1)
    else:
        # フルパイプライン実行
        run_sunshine_pipeline()

"""
SO8T Sunshine Pipeline - 実験計画ベースの自動実行システム

ボブにゃん戦略に基づくサンシャイン実行：
- Run A: Baseline (LoRAのみ)
- Run B: SO8T (LoRA + SO(8)アダプター)

ログフォーマット：
step, train_loss, eval_loss, so8_ortho_mean, so8_ortho_max, grad_norm, step_time_sec
"""

import os
import sys
import time
import json
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model

# Import from scripts directory
# from scripts.models.so8t_residual_adapter import attach_nkat_adapters  # 削除済み関数

# Simple dataset class for testing
from torch.utils.data import Dataset
import json

class SimpleDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, max_length: int = 128):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Load data
        self.data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        item = json.loads(line.strip())
                        self.data.append(item)
                    except json.JSONDecodeError:
                        continue

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item.get('text', '')

        # Tokenize
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
            'labels': tokenized['input_ids'].squeeze()
        }


class SunshineLogger:
    """サンシャイン実行の統一ログ収集"""

    def __init__(self, log_dir: str, run_name: str):
        self.log_dir = Path(log_dir)
        self.run_name = run_name
        self.log_file = self.log_dir / f"{run_name}_training_log.csv"
        self.metrics_file = self.log_dir / f"{run_name}_metrics.json"

        # CSVヘッダー
        self.columns = [
            'step', 'train_loss', 'eval_loss',
            'so8_ortho_mean', 'so8_ortho_max',
            'grad_norm', 'step_time_sec'
        ]

        # ログディレクトリ作成
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # CSV初期化
        if not self.log_file.exists():
            pd.DataFrame(columns=self.columns).to_csv(self.log_file, index=False)

        # メトリクス初期化
        self.metrics = {
            'start_time': datetime.now().isoformat(),
            'run_name': run_name,
            'total_steps': 0,
            'avg_step_time': 0,
            'final_train_loss': None,
            'final_eval_loss': None,
            'so8_ortho_errors': [],
            'grad_norms': []
        }

    def log_step(self, step: int, metrics: Dict[str, Any], step_time: float):
        """ステップごとのログ記録"""
        row = {
            'step': step,
            'train_loss': metrics.get('train_loss', None),
            'eval_loss': metrics.get('eval_loss', None),
            'so8_ortho_mean': metrics.get('so8_ortho_mean', None),
            'so8_ortho_max': metrics.get('so8_ortho_max', None),
            'grad_norm': metrics.get('grad_norm', None),
            'step_time_sec': step_time
        }

        # CSVに追加
        df = pd.DataFrame([row])
        df.to_csv(self.log_file, mode='a', header=False, index=False)

        # メトリクス更新
        if metrics.get('so8_ortho_mean') is not None:
            self.metrics['so8_ortho_errors'].append(metrics['so8_ortho_mean'])
        if metrics.get('grad_norm') is not None:
            self.metrics['grad_norms'].append(metrics['grad_norm'])

        self.metrics['total_steps'] = max(self.metrics['total_steps'], step)

    def finalize(self, final_metrics: Dict[str, Any]):
        """トレーニング完了時の最終ログ"""
        self.metrics.update({
            'end_time': datetime.now().isoformat(),
            'final_train_loss': final_metrics.get('train_loss'),
            'final_eval_loss': final_metrics.get('eval_loss'),
            'avg_so8_ortho_error': np.mean(self.metrics['so8_ortho_errors']) if self.metrics['so8_ortho_errors'] else None,
            'max_so8_ortho_error': np.max(self.metrics['so8_ortho_errors']) if self.metrics['so8_ortho_errors'] else None,
            'avg_grad_norm': np.mean(self.metrics['grad_norms']) if self.metrics['grad_norms'] else None
        })

        # JSON保存
        with open(self.metrics_file, 'w', encoding='utf-8') as f:
            json.dump(self.metrics, f, indent=2, ensure_ascii=False)


class SunshineTrainingConfig:
    """サンシャイン実行設定"""

    def __init__(self, run_type: str = "baseline"):
        self.model_name = "AXCXEPT/Borea-Phi-3.5-mini-Instruct-Jp"
        self.run_type = run_type  # "baseline" or "so8t"

        # データセット設定（統合データセットを使用）
        if run_type == "baseline":
            # Baseline用: 数学・科学統合データ
            self.sft_datasets = ["data/train_sft_enhanced.jsonl"]
        else:
            # SO8T用: NKAT理論・NSFWデータ統合
            self.sft_datasets = ["data/aegis_phi35_v2_with_nkat_so8t/aegis_phi35_v2_with_nkat_so8t_sft_train.jsonl"]

        # ドメイン重み付け
        self.domain_weights = {
            'mathematics': 1.2,
            'science': 1.1,
            'reasoning': 1.0,
            'general': 0.8
        }

        # トレーニング設定（本格データセット対応）
        self.training_config = {
            'output_dir': f"H:/from_D/webdataset/checkpoints/sunshine_{run_type}_phase25",
            'num_train_epochs': 1,
            'max_steps': 50,  # テスト用に減らす
            'per_device_train_batch_size': 1,
            'gradient_accumulation_steps': 8,  # RTX3060対応
            'learning_rate': 3e-5,  # 安定した学習率
            'save_steps': 100,
            'logging_steps': 10,  # 10ステップごとログ
            'eval_steps': 100,
            'gradient_checkpointing': False,  # 一時的にOFFにして勾配問題を解決
            'ddp_find_unused_parameters': False,  # アダプタパラメータの検出を確実にする
            'optim': "adamw_8bit",
            'bf16': True,
            'dataloader_pin_memory': False,
            'remove_unused_columns': False,
            'report_to': [],
            'load_best_model_at_end': False
        }

        # LoRA設定
        self.lora_config = {
            'r': 16,
            'lora_alpha': 32,
            'lora_dropout': 0.05,
            'bias': "none",
            'task_type': "CAUSAL_LM",
            'target_modules': ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        }

        # SO(8)設定（so8tの場合のみ）
        if run_type == "so8t":
            self.so8_config = {
                'target_layers': [8, 16, 24],  # 中間層のみ
                'so8_dim': 8,
                'alpha_init': 0.1
            }
        else:
            self.so8_config = None


def run_sunshine_experiment(run_type: str = "baseline") -> Dict[str, Any]:
    """
    サンシャイン実験実行
    run_type: "baseline" or "so8t"
    """
    print(f"🌞 Starting Sunshine Experiment: {run_type.upper()}")
    print("=" * 60)

    # 設定
    config = SunshineTrainingConfig(run_type)
    run_name = f"sunshine_run_{run_type}"

    # ロガー初期化
    logger = SunshineLogger("logs/sunshine", run_name)

    try:
        # モデルとトークナイザーロード
        print("[1/5] Loading model and tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # ★★★ トレーニング時は device_map="auto" を使用せず直接GPUにロード ★★★
        # device_map="auto" は分散トレーニングと競合するため
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float16,
            device_map=None  # トレーニング時は明示的にNone
        ).to("cuda")  # 直接GPUに移動

        # ★★★ デバッグ: Phi-3モデルの構造を確認 ★★★
        print(f"Model type: {type(model)}")
        print(f"Model attributes: {[attr for attr in dir(model) if not attr.startswith('_')]}")
        if hasattr(model, 'model'):
            print(f"model.model type: {type(model.model)}")
            print(f"model.model attributes: {[attr for attr in dir(model.model) if not attr.startswith('_')]}")
            if hasattr(model.model, 'layers'):
                print(f"model.model.layers length: {len(model.model.layers)}")
                print(f"First layer type: {type(model.model.layers[0])}")
                print(f"First layer attributes: {[attr for attr in dir(model.model.layers[0]) if not attr.startswith('_')]}")
                if hasattr(model.model.layers[0], 'mlp'):
                    print(f"First layer has mlp: {type(model.model.layers[0].mlp)}")

        # ★★★ 処方箋1: 入力の勾配を強制的に有効化 ★★★
        # これがないと、Gradient Checkpointing有効時に途中の勾配が死ぬ
        model.enable_input_require_grads()

        # LoRA適用
        print("[2/5] Applying LoRA...")
        lora_config = LoraConfig(**config.lora_config)
        model = get_peft_model(model, lora_config)

        # LoRAパラメータを明示的にトレーニング可能に（詳細確認）
        lora_params_count = 0
        for name, param in model.named_parameters():
            if 'lora' in name.lower():
                param.requires_grad_(True)
                lora_params_count += 1

        print(f"Set {lora_params_count} LoRA parameters to trainable")

        # SO(8)アダプター適用（so8tの場合）
        if config.so8_config:
            print("[3/5] Applying SO(8) adapters (Layer Replacement Mode)...")
            enable_quad = config.so8_config.get('enable_quad_inference', False)
            # ★★★ 最終奥義: モンキーパッチで注入（Unsloth最適化突破）★★★
            model = replace_mlp_with_nkat(
                model,
                target_layers=config.so8_config['target_layers']
            )

            print(f"SO(8) Adapter with Quad Inference: {enable_quad}")
        else:
            print("[3/5] Skipping SO(8) adapters (baseline run)")

        # パラメータ確認
        print("[4/5] Checking trainable parameters...")
        model.print_trainable_parameters()

        # 🔥 緊急バイパス手術：Optimizerへの手動登録（SO8Tの場合のみ）
        if config.so8_config:
            print("[4.5/5] Manual optimizer registration for SO8T...")
            # 1. 学習対象パラメータの抽出 (LoRA と NKATアダプタ だけ)
            trainable_params = []
            for name, param in model.named_parameters():
                if "lora" in name.lower() or "nkat_adapter" in name.lower():
                    param.requires_grad = True
                    trainable_params.append(param)
                else:
                    param.requires_grad = False

            print(f"🔥 Total Trainable Params: {len(trainable_params)} tensors")

            # 2. Optimizerの手動作成 (Unsloth推奨の8bit AdamWを使う場合)
            try:
                from unsloth.optim import AdamW8bit
                optimizer = AdamW8bit(trainable_params, lr=config.training_config.get('learning_rate', 2e-5))
                print("✅ Using Unsloth AdamW8bit")
            except ImportError:
                from torch.optim import AdamW
                optimizer = AdamW(trainable_params, lr=config.training_config.get('learning_rate', 2e-5))
                print("✅ Using Standard AdamW")

            # 3. 後でTrainerに渡すための保存
            manual_optimizer = optimizer
        else:
            manual_optimizer = None

        # データセット準備
        print("[5/5] Preparing dataset...")
        dataset = SimpleDataset(
            config.sft_datasets[0],  # 最初のデータセットを使用
            tokenizer
        )
        print(f"Dataset size: {len(dataset)} samples")

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )

        # トレーニング引数
        training_args = TrainingArguments(**config.training_config)

        # カスタムコールバックでSO(8)メトリクス収集
        class SunshineCallback(TrainerCallback):
            def __init__(self, logger, model, run_type):
                self.logger = logger
                self.model = model
                self.run_type = run_type
                self.step_start_time = None

            def on_init_end(self, args, state, control, **kwargs):
                pass

            def on_train_begin(self, args, state, control, **kwargs):
                pass

            def on_train_end(self, args, state, control, **kwargs):
                pass

            def on_step_begin(self, args, state, control, **kwargs):
                self.step_start_time = time.time()

            def on_step_end(self, args, state, control, **kwargs):
                if self.step_start_time is None:
                    return

                step_time = time.time() - self.step_start_time
                step = state.global_step

                # SO(8)メトリクス収集
                so8_metrics = {}
                if self.run_type == "so8t":
                    ortho_errors = []
                    alphas = []
                    for name, module in self.model.named_modules():
                        if hasattr(module, 'get_adapter_stats'):
                            stats = module.get_adapter_stats()
                            ortho_errors.append(stats['orthogonality_error'])
                            alphas.append(stats['alpha'])

                    if ortho_errors:
                        so8_metrics['so8_ortho_mean'] = np.mean(ortho_errors)
                        so8_metrics['so8_ortho_max'] = np.max(ortho_errors)
                        so8_metrics['so8_alpha_mean'] = np.mean(alphas)
                        so8_metrics['so8_alpha_std'] = np.std(alphas) if len(alphas) > 1 else 0

                # 勾配ノルム（利用可能なら）
                grad_norm = None
                if hasattr(state, 'log_history') and state.log_history:
                    last_log = state.log_history[-1]
                    grad_norm = last_log.get('grad_norm')

                # ログ記録
                metrics = {
                    'train_loss': state.log_history[-1].get('loss') if state.log_history else None,
                    **so8_metrics,
                    'grad_norm': grad_norm
                }

                self.logger.log_step(step, metrics, step_time)

        callback = SunshineCallback(logger, model, run_type)

        # Trainer設定
        if manual_optimizer is not None:
            # SO8Tの場合：手動Optimizerを使用
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=dataset,
                data_collator=data_collator,
                callbacks=[callback, NKATDebugCallback(model)],
                optimizers=(manual_optimizer, None)  # (optimizer, scheduler)
            )
            print("🔧 Using manual optimizer for SO8T training")
        else:
            # Baselineの場合：通常のTrainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=dataset,
                data_collator=data_collator,
                callbacks=[callback, NKATDebugCallback(model)]
            )

        # トレーニング実行
        print(f"🚀 Starting {run_type.upper()} training...")
        trainer.train()

        # 最終メトリクス
        final_metrics = {}
        if trainer.state.log_history:
            final_metrics = trainer.state.log_history[-1]

        logger.finalize(final_metrics)

        print(f"✅ {run_type.upper()} training completed!")
        print(f"📊 Results saved to: {logger.log_dir}")

        return {
            'success': True,
            'run_type': run_type,
            'log_dir': str(logger.log_dir),
            'metrics_file': str(logger.metrics_file),
            'final_loss': final_metrics.get('loss')
        }

    except Exception as e:
        print(f"❌ {run_type.upper()} training failed: {e}")
        logger.finalize({'error': str(e)})
        return {
            'success': False,
            'run_type': run_type,
            'error': str(e)
        }


def run_sunshine_pipeline():
    """サンシャインパイプライン実行"""
    print("🌞🌞🌞 SO8T SUNSHINE PIPELINE 🌞🌞🌞")
    print("Comparing Baseline vs SO8T performance")
    print("=" * 60)

    results = {}

    # Run A: Baseline
    print("\n🏃 Run A: BASELINE (LoRA only)")
    results['baseline'] = run_sunshine_experiment("baseline")

    # Run B: SO8T
    print("\n🧬 Run B: SO8T (LoRA + SO(8) Adapter)")
    results['so8t'] = run_sunshine_experiment("so8t")

    # 結果比較
    print("\n📊 EXPERIMENT RESULTS SUMMARY")
    print("=" * 60)

    for run_type, result in results.items():
        status = "✅ SUCCESS" if result['success'] else "❌ FAILED"
        loss = f"Final Loss: {result.get('final_loss', 'N/A')}"
        print(f"{run_type.upper()}: {status} | {loss}")

    # ログファイル保存
    summary_file = Path("logs/sunshine") / "sunshine_summary.json"
    summary_file.parent.mkdir(parents=True, exist_ok=True)

    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'results': results
        }, f, indent=2, ensure_ascii=False)

    print(f"\n📋 Summary saved to: {summary_file}")

    # オーディオ通知
    try:
        import winsound
        winsound.Beep(1000, 1000)  # 1秒のビープ音
    except:
        pass

    return results


if __name__ == "__main__":
    # 引数処理
    if len(sys.argv) > 1:
        run_type = sys.argv[1]
        if run_type in ["baseline", "so8t"]:
            result = run_sunshine_experiment(run_type)
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print("Usage: python sunshine_pipeline.py [baseline|so8t]")
            sys.exit(1)
    else:
        # フルパイプライン実行
        run_sunshine_pipeline()

"""
SO8T Sunshine Pipeline - 実験計画ベースの自動実行システム

ボブにゃん戦略に基づくサンシャイン実行：
- Run A: Baseline (LoRAのみ)
- Run B: SO8T (LoRA + SO(8)アダプター)

ログフォーマット：
step, train_loss, eval_loss, so8_ortho_mean, so8_ortho_max, grad_norm, step_time_sec
"""

import os
import sys
import time
import json
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model

# Import from scripts directory
# from scripts.models.so8t_residual_adapter import attach_nkat_adapters  # 削除済み関数

# Simple dataset class for testing
from torch.utils.data import Dataset
import json

class SimpleDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, max_length: int = 128):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Load data
        self.data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        item = json.loads(line.strip())
                        self.data.append(item)
                    except json.JSONDecodeError:
                        continue

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item.get('text', '')

        # Tokenize
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
            'labels': tokenized['input_ids'].squeeze()
        }


class SunshineLogger:
    """サンシャイン実行の統一ログ収集"""

    def __init__(self, log_dir: str, run_name: str):
        self.log_dir = Path(log_dir)
        self.run_name = run_name
        self.log_file = self.log_dir / f"{run_name}_training_log.csv"
        self.metrics_file = self.log_dir / f"{run_name}_metrics.json"

        # CSVヘッダー
        self.columns = [
            'step', 'train_loss', 'eval_loss',
            'so8_ortho_mean', 'so8_ortho_max',
            'grad_norm', 'step_time_sec'
        ]

        # ログディレクトリ作成
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # CSV初期化
        if not self.log_file.exists():
            pd.DataFrame(columns=self.columns).to_csv(self.log_file, index=False)

        # メトリクス初期化
        self.metrics = {
            'start_time': datetime.now().isoformat(),
            'run_name': run_name,
            'total_steps': 0,
            'avg_step_time': 0,
            'final_train_loss': None,
            'final_eval_loss': None,
            'so8_ortho_errors': [],
            'grad_norms': []
        }

    def log_step(self, step: int, metrics: Dict[str, Any], step_time: float):
        """ステップごとのログ記録"""
        row = {
            'step': step,
            'train_loss': metrics.get('train_loss', None),
            'eval_loss': metrics.get('eval_loss', None),
            'so8_ortho_mean': metrics.get('so8_ortho_mean', None),
            'so8_ortho_max': metrics.get('so8_ortho_max', None),
            'grad_norm': metrics.get('grad_norm', None),
            'step_time_sec': step_time
        }

        # CSVに追加
        df = pd.DataFrame([row])
        df.to_csv(self.log_file, mode='a', header=False, index=False)

        # メトリクス更新
        if metrics.get('so8_ortho_mean') is not None:
            self.metrics['so8_ortho_errors'].append(metrics['so8_ortho_mean'])
        if metrics.get('grad_norm') is not None:
            self.metrics['grad_norms'].append(metrics['grad_norm'])

        self.metrics['total_steps'] = max(self.metrics['total_steps'], step)

    def finalize(self, final_metrics: Dict[str, Any]):
        """トレーニング完了時の最終ログ"""
        self.metrics.update({
            'end_time': datetime.now().isoformat(),
            'final_train_loss': final_metrics.get('train_loss'),
            'final_eval_loss': final_metrics.get('eval_loss'),
            'avg_so8_ortho_error': np.mean(self.metrics['so8_ortho_errors']) if self.metrics['so8_ortho_errors'] else None,
            'max_so8_ortho_error': np.max(self.metrics['so8_ortho_errors']) if self.metrics['so8_ortho_errors'] else None,
            'avg_grad_norm': np.mean(self.metrics['grad_norms']) if self.metrics['grad_norms'] else None
        })

        # JSON保存
        with open(self.metrics_file, 'w', encoding='utf-8') as f:
            json.dump(self.metrics, f, indent=2, ensure_ascii=False)


class SunshineTrainingConfig:
    """サンシャイン実行設定"""

    def __init__(self, run_type: str = "baseline"):
        self.model_name = "AXCXEPT/Borea-Phi-3.5-mini-Instruct-Jp"
        self.run_type = run_type  # "baseline" or "so8t"

        # データセット設定（統合データセットを使用）
        if run_type == "baseline":
            # Baseline用: 数学・科学統合データ
            self.sft_datasets = ["data/train_sft_enhanced.jsonl"]
        else:
            # SO8T用: NKAT理論・NSFWデータ統合
            self.sft_datasets = ["data/aegis_phi35_v2_with_nkat_so8t/aegis_phi35_v2_with_nkat_so8t_sft_train.jsonl"]

        # ドメイン重み付け
        self.domain_weights = {
            'mathematics': 1.2,
            'science': 1.1,
            'reasoning': 1.0,
            'general': 0.8
        }

        # トレーニング設定（本格データセット対応）
        self.training_config = {
            'output_dir': f"H:/from_D/webdataset/checkpoints/sunshine_{run_type}_phase25",
            'num_train_epochs': 1,
            'max_steps': 50,  # テスト用に減らす
            'per_device_train_batch_size': 1,
            'gradient_accumulation_steps': 8,  # RTX3060対応
            'learning_rate': 3e-5,  # 安定した学習率
            'save_steps': 100,
            'logging_steps': 10,  # 10ステップごとログ
            'eval_steps': 100,
            'gradient_checkpointing': False,  # 一時的にOFFにして勾配問題を解決
            'ddp_find_unused_parameters': False,  # アダプタパラメータの検出を確実にする
            'optim': "adamw_8bit",
            'bf16': True,
            'dataloader_pin_memory': False,
            'remove_unused_columns': False,
            'report_to': [],
            'load_best_model_at_end': False
        }

        # LoRA設定
        self.lora_config = {
            'r': 16,
            'lora_alpha': 32,
            'lora_dropout': 0.05,
            'bias': "none",
            'task_type': "CAUSAL_LM",
            'target_modules': ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        }

        # SO(8)設定（so8tの場合のみ）
        if run_type == "so8t":
            self.so8_config = {
                'target_layers': [8, 16, 24],  # 中間層のみ
                'so8_dim': 8,
                'alpha_init': 0.1
            }
        else:
            self.so8_config = None


def run_sunshine_experiment(run_type: str = "baseline") -> Dict[str, Any]:
    """
    サンシャイン実験実行
    run_type: "baseline" or "so8t"
    """
    print(f"🌞 Starting Sunshine Experiment: {run_type.upper()}")
    print("=" * 60)

    # 設定
    config = SunshineTrainingConfig(run_type)
    run_name = f"sunshine_run_{run_type}"

    # ロガー初期化
    logger = SunshineLogger("logs/sunshine", run_name)

    try:
        # モデルとトークナイザーロード
        print("[1/5] Loading model and tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # ★★★ トレーニング時は device_map="auto" を使用せず直接GPUにロード ★★★
        # device_map="auto" は分散トレーニングと競合するため
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float16,
            device_map=None  # トレーニング時は明示的にNone
        ).to("cuda")  # 直接GPUに移動

        # ★★★ デバッグ: Phi-3モデルの構造を確認 ★★★
        print(f"Model type: {type(model)}")
        print(f"Model attributes: {[attr for attr in dir(model) if not attr.startswith('_')]}")
        if hasattr(model, 'model'):
            print(f"model.model type: {type(model.model)}")
            print(f"model.model attributes: {[attr for attr in dir(model.model) if not attr.startswith('_')]}")
            if hasattr(model.model, 'layers'):
                print(f"model.model.layers length: {len(model.model.layers)}")
                print(f"First layer type: {type(model.model.layers[0])}")
                print(f"First layer attributes: {[attr for attr in dir(model.model.layers[0]) if not attr.startswith('_')]}")
                if hasattr(model.model.layers[0], 'mlp'):
                    print(f"First layer has mlp: {type(model.model.layers[0].mlp)}")

        # ★★★ 処方箋1: 入力の勾配を強制的に有効化 ★★★
        # これがないと、Gradient Checkpointing有効時に途中の勾配が死ぬ
        model.enable_input_require_grads()

        # LoRA適用
        print("[2/5] Applying LoRA...")
        lora_config = LoraConfig(**config.lora_config)
        model = get_peft_model(model, lora_config)

        # LoRAパラメータを明示的にトレーニング可能に（詳細確認）
        lora_params_count = 0
        for name, param in model.named_parameters():
            if 'lora' in name.lower():
                param.requires_grad_(True)
                lora_params_count += 1

        print(f"Set {lora_params_count} LoRA parameters to trainable")

        # SO(8)アダプター適用（so8tの場合）
        if config.so8_config:
            print("[3/5] Applying SO(8) adapters (Layer Replacement Mode)...")
            enable_quad = config.so8_config.get('enable_quad_inference', False)
            # ★★★ 最終奥義: モンキーパッチで注入（Unsloth最適化突破）★★★
            model = replace_mlp_with_nkat(
                model,
                target_layers=config.so8_config['target_layers']
            )

            print(f"SO(8) Adapter with Quad Inference: {enable_quad}")
        else:
            print("[3/5] Skipping SO(8) adapters (baseline run)")

        # パラメータ確認
        print("[4/5] Checking trainable parameters...")
        model.print_trainable_parameters()

        # 🔥 緊急バイパス手術：Optimizerへの手動登録（SO8Tの場合のみ）
        if config.so8_config:
            print("[4.5/5] Manual optimizer registration for SO8T...")
            # 1. 学習対象パラメータの抽出 (LoRA と NKATアダプタ だけ)
            trainable_params = []
            for name, param in model.named_parameters():
                if "lora" in name.lower() or "nkat_adapter" in name.lower():
                    param.requires_grad = True
                    trainable_params.append(param)
                else:
                    param.requires_grad = False

            print(f"🔥 Total Trainable Params: {len(trainable_params)} tensors")

            # 2. Optimizerの手動作成 (Unsloth推奨の8bit AdamWを使う場合)
            try:
                from unsloth.optim import AdamW8bit
                optimizer = AdamW8bit(trainable_params, lr=config.training_config.get('learning_rate', 2e-5))
                print("✅ Using Unsloth AdamW8bit")
            except ImportError:
                from torch.optim import AdamW
                optimizer = AdamW(trainable_params, lr=config.training_config.get('learning_rate', 2e-5))
                print("✅ Using Standard AdamW")

            # 3. 後でTrainerに渡すための保存
            manual_optimizer = optimizer
        else:
            manual_optimizer = None

        # データセット準備
        print("[5/5] Preparing dataset...")
        dataset = SimpleDataset(
            config.sft_datasets[0],  # 最初のデータセットを使用
            tokenizer
        )
        print(f"Dataset size: {len(dataset)} samples")

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )

        # トレーニング引数
        training_args = TrainingArguments(**config.training_config)

        # カスタムコールバックでSO(8)メトリクス収集
        class SunshineCallback(TrainerCallback):
            def __init__(self, logger, model, run_type):
                self.logger = logger
                self.model = model
                self.run_type = run_type
                self.step_start_time = None

            def on_init_end(self, args, state, control, **kwargs):
                pass

            def on_train_begin(self, args, state, control, **kwargs):
                pass

            def on_train_end(self, args, state, control, **kwargs):
                pass

            def on_step_begin(self, args, state, control, **kwargs):
                self.step_start_time = time.time()

            def on_step_end(self, args, state, control, **kwargs):
                if self.step_start_time is None:
                    return

                step_time = time.time() - self.step_start_time
                step = state.global_step

                # SO(8)メトリクス収集
                so8_metrics = {}
                if self.run_type == "so8t":
                    ortho_errors = []
                    alphas = []
                    for name, module in self.model.named_modules():
                        if hasattr(module, 'get_adapter_stats'):
                            stats = module.get_adapter_stats()
                            ortho_errors.append(stats['orthogonality_error'])
                            alphas.append(stats['alpha'])

                    if ortho_errors:
                        so8_metrics['so8_ortho_mean'] = np.mean(ortho_errors)
                        so8_metrics['so8_ortho_max'] = np.max(ortho_errors)
                        so8_metrics['so8_alpha_mean'] = np.mean(alphas)
                        so8_metrics['so8_alpha_std'] = np.std(alphas) if len(alphas) > 1 else 0

                # 勾配ノルム（利用可能なら）
                grad_norm = None
                if hasattr(state, 'log_history') and state.log_history:
                    last_log = state.log_history[-1]
                    grad_norm = last_log.get('grad_norm')

                # ログ記録
                metrics = {
                    'train_loss': state.log_history[-1].get('loss') if state.log_history else None,
                    **so8_metrics,
                    'grad_norm': grad_norm
                }

                self.logger.log_step(step, metrics, step_time)

        callback = SunshineCallback(logger, model, run_type)

        # Trainer設定
        if manual_optimizer is not None:
            # SO8Tの場合：手動Optimizerを使用
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=dataset,
                data_collator=data_collator,
                callbacks=[callback, NKATDebugCallback(model)],
                optimizers=(manual_optimizer, None)  # (optimizer, scheduler)
            )
            print("🔧 Using manual optimizer for SO8T training")
        else:
            # Baselineの場合：通常のTrainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=dataset,
                data_collator=data_collator,
                callbacks=[callback, NKATDebugCallback(model)]
            )

        # トレーニング実行
        print(f"🚀 Starting {run_type.upper()} training...")
        trainer.train()

        # 最終メトリクス
        final_metrics = {}
        if trainer.state.log_history:
            final_metrics = trainer.state.log_history[-1]

        logger.finalize(final_metrics)

        print(f"✅ {run_type.upper()} training completed!")
        print(f"📊 Results saved to: {logger.log_dir}")

        return {
            'success': True,
            'run_type': run_type,
            'log_dir': str(logger.log_dir),
            'metrics_file': str(logger.metrics_file),
            'final_loss': final_metrics.get('loss')
        }

    except Exception as e:
        print(f"❌ {run_type.upper()} training failed: {e}")
        logger.finalize({'error': str(e)})
        return {
            'success': False,
            'run_type': run_type,
            'error': str(e)
        }


def run_sunshine_pipeline():
    """サンシャインパイプライン実行"""
    print("🌞🌞🌞 SO8T SUNSHINE PIPELINE 🌞🌞🌞")
    print("Comparing Baseline vs SO8T performance")
    print("=" * 60)

    results = {}

    # Run A: Baseline
    print("\n🏃 Run A: BASELINE (LoRA only)")
    results['baseline'] = run_sunshine_experiment("baseline")

    # Run B: SO8T
    print("\n🧬 Run B: SO8T (LoRA + SO(8) Adapter)")
    results['so8t'] = run_sunshine_experiment("so8t")

    # 結果比較
    print("\n📊 EXPERIMENT RESULTS SUMMARY")
    print("=" * 60)

    for run_type, result in results.items():
        status = "✅ SUCCESS" if result['success'] else "❌ FAILED"
        loss = f"Final Loss: {result.get('final_loss', 'N/A')}"
        print(f"{run_type.upper()}: {status} | {loss}")

    # ログファイル保存
    summary_file = Path("logs/sunshine") / "sunshine_summary.json"
    summary_file.parent.mkdir(parents=True, exist_ok=True)

    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'results': results
        }, f, indent=2, ensure_ascii=False)

    print(f"\n📋 Summary saved to: {summary_file}")

    # オーディオ通知
    try:
        import winsound
        winsound.Beep(1000, 1000)  # 1秒のビープ音
    except:
        pass

    return results


if __name__ == "__main__":
    # 引数処理
    if len(sys.argv) > 1:
        run_type = sys.argv[1]
        if run_type in ["baseline", "so8t"]:
            result = run_sunshine_experiment(run_type)
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print("Usage: python sunshine_pipeline.py [baseline|so8t]")
            sys.exit(1)
    else:
        # フルパイプライン実行
        run_sunshine_pipeline()

"""
SO8T Sunshine Pipeline - 実験計画ベースの自動実行システム

ボブにゃん戦略に基づくサンシャイン実行：
- Run A: Baseline (LoRAのみ)
- Run B: SO8T (LoRA + SO(8)アダプター)

ログフォーマット：
step, train_loss, eval_loss, so8_ortho_mean, so8_ortho_max, grad_norm, step_time_sec
"""

import os
import sys
import time
import json
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model

# Import from scripts directory
# from scripts.models.so8t_residual_adapter import attach_nkat_adapters  # 削除済み関数

# Simple dataset class for testing
from torch.utils.data import Dataset
import json

class SimpleDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, max_length: int = 128):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Load data
        self.data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        item = json.loads(line.strip())
                        self.data.append(item)
                    except json.JSONDecodeError:
                        continue

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item.get('text', '')

        # Tokenize
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
            'labels': tokenized['input_ids'].squeeze()
        }


class SunshineLogger:
    """サンシャイン実行の統一ログ収集"""

    def __init__(self, log_dir: str, run_name: str):
        self.log_dir = Path(log_dir)
        self.run_name = run_name
        self.log_file = self.log_dir / f"{run_name}_training_log.csv"
        self.metrics_file = self.log_dir / f"{run_name}_metrics.json"

        # CSVヘッダー
        self.columns = [
            'step', 'train_loss', 'eval_loss',
            'so8_ortho_mean', 'so8_ortho_max',
            'grad_norm', 'step_time_sec'
        ]

        # ログディレクトリ作成
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # CSV初期化
        if not self.log_file.exists():
            pd.DataFrame(columns=self.columns).to_csv(self.log_file, index=False)

        # メトリクス初期化
        self.metrics = {
            'start_time': datetime.now().isoformat(),
            'run_name': run_name,
            'total_steps': 0,
            'avg_step_time': 0,
            'final_train_loss': None,
            'final_eval_loss': None,
            'so8_ortho_errors': [],
            'grad_norms': []
        }

    def log_step(self, step: int, metrics: Dict[str, Any], step_time: float):
        """ステップごとのログ記録"""
        row = {
            'step': step,
            'train_loss': metrics.get('train_loss', None),
            'eval_loss': metrics.get('eval_loss', None),
            'so8_ortho_mean': metrics.get('so8_ortho_mean', None),
            'so8_ortho_max': metrics.get('so8_ortho_max', None),
            'grad_norm': metrics.get('grad_norm', None),
            'step_time_sec': step_time
        }

        # CSVに追加
        df = pd.DataFrame([row])
        df.to_csv(self.log_file, mode='a', header=False, index=False)

        # メトリクス更新
        if metrics.get('so8_ortho_mean') is not None:
            self.metrics['so8_ortho_errors'].append(metrics['so8_ortho_mean'])
        if metrics.get('grad_norm') is not None:
            self.metrics['grad_norms'].append(metrics['grad_norm'])

        self.metrics['total_steps'] = max(self.metrics['total_steps'], step)

    def finalize(self, final_metrics: Dict[str, Any]):
        """トレーニング完了時の最終ログ"""
        self.metrics.update({
            'end_time': datetime.now().isoformat(),
            'final_train_loss': final_metrics.get('train_loss'),
            'final_eval_loss': final_metrics.get('eval_loss'),
            'avg_so8_ortho_error': np.mean(self.metrics['so8_ortho_errors']) if self.metrics['so8_ortho_errors'] else None,
            'max_so8_ortho_error': np.max(self.metrics['so8_ortho_errors']) if self.metrics['so8_ortho_errors'] else None,
            'avg_grad_norm': np.mean(self.metrics['grad_norms']) if self.metrics['grad_norms'] else None
        })

        # JSON保存
        with open(self.metrics_file, 'w', encoding='utf-8') as f:
            json.dump(self.metrics, f, indent=2, ensure_ascii=False)


class SunshineTrainingConfig:
    """サンシャイン実行設定"""

    def __init__(self, run_type: str = "baseline"):
        self.model_name = "AXCXEPT/Borea-Phi-3.5-mini-Instruct-Jp"
        self.run_type = run_type  # "baseline" or "so8t"

        # データセット設定（統合データセットを使用）
        if run_type == "baseline":
            # Baseline用: 数学・科学統合データ
            self.sft_datasets = ["data/train_sft_enhanced.jsonl"]
        else:
            # SO8T用: NKAT理論・NSFWデータ統合
            self.sft_datasets = ["data/aegis_phi35_v2_with_nkat_so8t/aegis_phi35_v2_with_nkat_so8t_sft_train.jsonl"]

        # ドメイン重み付け
        self.domain_weights = {
            'mathematics': 1.2,
            'science': 1.1,
            'reasoning': 1.0,
            'general': 0.8
        }

        # トレーニング設定（本格データセット対応）
        self.training_config = {
            'output_dir': f"H:/from_D/webdataset/checkpoints/sunshine_{run_type}_phase25",
            'num_train_epochs': 1,
            'max_steps': 50,  # テスト用に減らす
            'per_device_train_batch_size': 1,
            'gradient_accumulation_steps': 8,  # RTX3060対応
            'learning_rate': 3e-5,  # 安定した学習率
            'save_steps': 100,
            'logging_steps': 10,  # 10ステップごとログ
            'eval_steps': 100,
            'gradient_checkpointing': False,  # 一時的にOFFにして勾配問題を解決
            'ddp_find_unused_parameters': False,  # アダプタパラメータの検出を確実にする
            'optim': "adamw_8bit",
            'bf16': True,
            'dataloader_pin_memory': False,
            'remove_unused_columns': False,
            'report_to': [],
            'load_best_model_at_end': False
        }

        # LoRA設定
        self.lora_config = {
            'r': 16,
            'lora_alpha': 32,
            'lora_dropout': 0.05,
            'bias': "none",
            'task_type': "CAUSAL_LM",
            'target_modules': ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        }

        # SO(8)設定（so8tの場合のみ）
        if run_type == "so8t":
            self.so8_config = {
                'target_layers': [8, 16, 24],  # 中間層のみ
                'so8_dim': 8,
                'alpha_init': 0.1
            }
        else:
            self.so8_config = None


def run_sunshine_experiment(run_type: str = "baseline") -> Dict[str, Any]:
    """
    サンシャイン実験実行
    run_type: "baseline" or "so8t"
    """
    print(f"🌞 Starting Sunshine Experiment: {run_type.upper()}")
    print("=" * 60)

    # 設定
    config = SunshineTrainingConfig(run_type)
    run_name = f"sunshine_run_{run_type}"

    # ロガー初期化
    logger = SunshineLogger("logs/sunshine", run_name)

    try:
        # モデルとトークナイザーロード
        print("[1/5] Loading model and tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # ★★★ トレーニング時は device_map="auto" を使用せず直接GPUにロード ★★★
        # device_map="auto" は分散トレーニングと競合するため
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float16,
            device_map=None  # トレーニング時は明示的にNone
        ).to("cuda")  # 直接GPUに移動

        # ★★★ デバッグ: Phi-3モデルの構造を確認 ★★★
        print(f"Model type: {type(model)}")
        print(f"Model attributes: {[attr for attr in dir(model) if not attr.startswith('_')]}")
        if hasattr(model, 'model'):
            print(f"model.model type: {type(model.model)}")
            print(f"model.model attributes: {[attr for attr in dir(model.model) if not attr.startswith('_')]}")
            if hasattr(model.model, 'layers'):
                print(f"model.model.layers length: {len(model.model.layers)}")
                print(f"First layer type: {type(model.model.layers[0])}")
                print(f"First layer attributes: {[attr for attr in dir(model.model.layers[0]) if not attr.startswith('_')]}")
                if hasattr(model.model.layers[0], 'mlp'):
                    print(f"First layer has mlp: {type(model.model.layers[0].mlp)}")

        # ★★★ 処方箋1: 入力の勾配を強制的に有効化 ★★★
        # これがないと、Gradient Checkpointing有効時に途中の勾配が死ぬ
        model.enable_input_require_grads()

        # LoRA適用
        print("[2/5] Applying LoRA...")
        lora_config = LoraConfig(**config.lora_config)
        model = get_peft_model(model, lora_config)

        # LoRAパラメータを明示的にトレーニング可能に（詳細確認）
        lora_params_count = 0
        for name, param in model.named_parameters():
            if 'lora' in name.lower():
                param.requires_grad_(True)
                lora_params_count += 1

        print(f"Set {lora_params_count} LoRA parameters to trainable")

        # SO(8)アダプター適用（so8tの場合）
        if config.so8_config:
            print("[3/5] Applying SO(8) adapters (Layer Replacement Mode)...")
            enable_quad = config.so8_config.get('enable_quad_inference', False)
            # ★★★ 最終奥義: モンキーパッチで注入（Unsloth最適化突破）★★★
            model = replace_mlp_with_nkat(
                model,
                target_layers=config.so8_config['target_layers']
            )

            print(f"SO(8) Adapter with Quad Inference: {enable_quad}")
        else:
            print("[3/5] Skipping SO(8) adapters (baseline run)")

        # パラメータ確認
        print("[4/5] Checking trainable parameters...")
        model.print_trainable_parameters()

        # 🔥 緊急バイパス手術：Optimizerへの手動登録（SO8Tの場合のみ）
        if config.so8_config:
            print("[4.5/5] Manual optimizer registration for SO8T...")
            # 1. 学習対象パラメータの抽出 (LoRA と NKATアダプタ だけ)
            trainable_params = []
            for name, param in model.named_parameters():
                if "lora" in name.lower() or "nkat_adapter" in name.lower():
                    param.requires_grad = True
                    trainable_params.append(param)
                else:
                    param.requires_grad = False

            print(f"🔥 Total Trainable Params: {len(trainable_params)} tensors")

            # 2. Optimizerの手動作成 (Unsloth推奨の8bit AdamWを使う場合)
            try:
                from unsloth.optim import AdamW8bit
                optimizer = AdamW8bit(trainable_params, lr=config.training_config.get('learning_rate', 2e-5))
                print("✅ Using Unsloth AdamW8bit")
            except ImportError:
                from torch.optim import AdamW
                optimizer = AdamW(trainable_params, lr=config.training_config.get('learning_rate', 2e-5))
                print("✅ Using Standard AdamW")

            # 3. 後でTrainerに渡すための保存
            manual_optimizer = optimizer
        else:
            manual_optimizer = None

        # データセット準備
        print("[5/5] Preparing dataset...")
        dataset = SimpleDataset(
            config.sft_datasets[0],  # 最初のデータセットを使用
            tokenizer
        )
        print(f"Dataset size: {len(dataset)} samples")

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )

        # トレーニング引数
        training_args = TrainingArguments(**config.training_config)

        # カスタムコールバックでSO(8)メトリクス収集
        class SunshineCallback(TrainerCallback):
            def __init__(self, logger, model, run_type):
                self.logger = logger
                self.model = model
                self.run_type = run_type
                self.step_start_time = None

            def on_init_end(self, args, state, control, **kwargs):
                pass

            def on_train_begin(self, args, state, control, **kwargs):
                pass

            def on_train_end(self, args, state, control, **kwargs):
                pass

            def on_step_begin(self, args, state, control, **kwargs):
                self.step_start_time = time.time()

            def on_step_end(self, args, state, control, **kwargs):
                if self.step_start_time is None:
                    return

                step_time = time.time() - self.step_start_time
                step = state.global_step

                # SO(8)メトリクス収集
                so8_metrics = {}
                if self.run_type == "so8t":
                    ortho_errors = []
                    alphas = []
                    for name, module in self.model.named_modules():
                        if hasattr(module, 'get_adapter_stats'):
                            stats = module.get_adapter_stats()
                            ortho_errors.append(stats['orthogonality_error'])
                            alphas.append(stats['alpha'])

                    if ortho_errors:
                        so8_metrics['so8_ortho_mean'] = np.mean(ortho_errors)
                        so8_metrics['so8_ortho_max'] = np.max(ortho_errors)
                        so8_metrics['so8_alpha_mean'] = np.mean(alphas)
                        so8_metrics['so8_alpha_std'] = np.std(alphas) if len(alphas) > 1 else 0

                # 勾配ノルム（利用可能なら）
                grad_norm = None
                if hasattr(state, 'log_history') and state.log_history:
                    last_log = state.log_history[-1]
                    grad_norm = last_log.get('grad_norm')

                # ログ記録
                metrics = {
                    'train_loss': state.log_history[-1].get('loss') if state.log_history else None,
                    **so8_metrics,
                    'grad_norm': grad_norm
                }

                self.logger.log_step(step, metrics, step_time)

        callback = SunshineCallback(logger, model, run_type)

        # Trainer設定
        if manual_optimizer is not None:
            # SO8Tの場合：手動Optimizerを使用
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=dataset,
                data_collator=data_collator,
                callbacks=[callback, NKATDebugCallback(model)],
                optimizers=(manual_optimizer, None)  # (optimizer, scheduler)
            )
            print("🔧 Using manual optimizer for SO8T training")
        else:
            # Baselineの場合：通常のTrainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=dataset,
                data_collator=data_collator,
                callbacks=[callback, NKATDebugCallback(model)]
            )

        # トレーニング実行
        print(f"🚀 Starting {run_type.upper()} training...")
        trainer.train()

        # 最終メトリクス
        final_metrics = {}
        if trainer.state.log_history:
            final_metrics = trainer.state.log_history[-1]

        logger.finalize(final_metrics)

        print(f"✅ {run_type.upper()} training completed!")
        print(f"📊 Results saved to: {logger.log_dir}")

        return {
            'success': True,
            'run_type': run_type,
            'log_dir': str(logger.log_dir),
            'metrics_file': str(logger.metrics_file),
            'final_loss': final_metrics.get('loss')
        }

    except Exception as e:
        print(f"❌ {run_type.upper()} training failed: {e}")
        logger.finalize({'error': str(e)})
        return {
            'success': False,
            'run_type': run_type,
            'error': str(e)
        }


def run_sunshine_pipeline():
    """サンシャインパイプライン実行"""
    print("🌞🌞🌞 SO8T SUNSHINE PIPELINE 🌞🌞🌞")
    print("Comparing Baseline vs SO8T performance")
    print("=" * 60)

    results = {}

    # Run A: Baseline
    print("\n🏃 Run A: BASELINE (LoRA only)")
    results['baseline'] = run_sunshine_experiment("baseline")

    # Run B: SO8T
    print("\n🧬 Run B: SO8T (LoRA + SO(8) Adapter)")
    results['so8t'] = run_sunshine_experiment("so8t")

    # 結果比較
    print("\n📊 EXPERIMENT RESULTS SUMMARY")
    print("=" * 60)

    for run_type, result in results.items():
        status = "✅ SUCCESS" if result['success'] else "❌ FAILED"
        loss = f"Final Loss: {result.get('final_loss', 'N/A')}"
        print(f"{run_type.upper()}: {status} | {loss}")

    # ログファイル保存
    summary_file = Path("logs/sunshine") / "sunshine_summary.json"
    summary_file.parent.mkdir(parents=True, exist_ok=True)

    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'results': results
        }, f, indent=2, ensure_ascii=False)

    print(f"\n📋 Summary saved to: {summary_file}")

    # オーディオ通知
    try:
        import winsound
        winsound.Beep(1000, 1000)  # 1秒のビープ音
    except:
        pass

    return results


if __name__ == "__main__":
    # 引数処理
    if len(sys.argv) > 1:
        run_type = sys.argv[1]
        if run_type in ["baseline", "so8t"]:
            result = run_sunshine_experiment(run_type)
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print("Usage: python sunshine_pipeline.py [baseline|so8t]")
            sys.exit(1)
    else:
        # フルパイプライン実行
        run_sunshine_pipeline()

"""
SO8T Sunshine Pipeline - 実験計画ベースの自動実行システム

ボブにゃん戦略に基づくサンシャイン実行：
- Run A: Baseline (LoRAのみ)
- Run B: SO8T (LoRA + SO(8)アダプター)

ログフォーマット：
step, train_loss, eval_loss, so8_ortho_mean, so8_ortho_max, grad_norm, step_time_sec
"""

import os
import sys
import time
import json
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model

# Import from scripts directory
# from scripts.models.so8t_residual_adapter import attach_nkat_adapters  # 削除済み関数

# Simple dataset class for testing
from torch.utils.data import Dataset
import json

class SimpleDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, max_length: int = 128):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Load data
        self.data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        item = json.loads(line.strip())
                        self.data.append(item)
                    except json.JSONDecodeError:
                        continue

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item.get('text', '')

        # Tokenize
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
            'labels': tokenized['input_ids'].squeeze()
        }


class SunshineLogger:
    """サンシャイン実行の統一ログ収集"""

    def __init__(self, log_dir: str, run_name: str):
        self.log_dir = Path(log_dir)
        self.run_name = run_name
        self.log_file = self.log_dir / f"{run_name}_training_log.csv"
        self.metrics_file = self.log_dir / f"{run_name}_metrics.json"

        # CSVヘッダー
        self.columns = [
            'step', 'train_loss', 'eval_loss',
            'so8_ortho_mean', 'so8_ortho_max',
            'grad_norm', 'step_time_sec'
        ]

        # ログディレクトリ作成
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # CSV初期化
        if not self.log_file.exists():
            pd.DataFrame(columns=self.columns).to_csv(self.log_file, index=False)

        # メトリクス初期化
        self.metrics = {
            'start_time': datetime.now().isoformat(),
            'run_name': run_name,
            'total_steps': 0,
            'avg_step_time': 0,
            'final_train_loss': None,
            'final_eval_loss': None,
            'so8_ortho_errors': [],
            'grad_norms': []
        }

    def log_step(self, step: int, metrics: Dict[str, Any], step_time: float):
        """ステップごとのログ記録"""
        row = {
            'step': step,
            'train_loss': metrics.get('train_loss', None),
            'eval_loss': metrics.get('eval_loss', None),
            'so8_ortho_mean': metrics.get('so8_ortho_mean', None),
            'so8_ortho_max': metrics.get('so8_ortho_max', None),
            'grad_norm': metrics.get('grad_norm', None),
            'step_time_sec': step_time
        }

        # CSVに追加
        df = pd.DataFrame([row])
        df.to_csv(self.log_file, mode='a', header=False, index=False)

        # メトリクス更新
        if metrics.get('so8_ortho_mean') is not None:
            self.metrics['so8_ortho_errors'].append(metrics['so8_ortho_mean'])
        if metrics.get('grad_norm') is not None:
            self.metrics['grad_norms'].append(metrics['grad_norm'])

        self.metrics['total_steps'] = max(self.metrics['total_steps'], step)

    def finalize(self, final_metrics: Dict[str, Any]):
        """トレーニング完了時の最終ログ"""
        self.metrics.update({
            'end_time': datetime.now().isoformat(),
            'final_train_loss': final_metrics.get('train_loss'),
            'final_eval_loss': final_metrics.get('eval_loss'),
            'avg_so8_ortho_error': np.mean(self.metrics['so8_ortho_errors']) if self.metrics['so8_ortho_errors'] else None,
            'max_so8_ortho_error': np.max(self.metrics['so8_ortho_errors']) if self.metrics['so8_ortho_errors'] else None,
            'avg_grad_norm': np.mean(self.metrics['grad_norms']) if self.metrics['grad_norms'] else None
        })

        # JSON保存
        with open(self.metrics_file, 'w', encoding='utf-8') as f:
            json.dump(self.metrics, f, indent=2, ensure_ascii=False)


class SunshineTrainingConfig:
    """サンシャイン実行設定"""

    def __init__(self, run_type: str = "baseline"):
        self.model_name = "AXCXEPT/Borea-Phi-3.5-mini-Instruct-Jp"
        self.run_type = run_type  # "baseline" or "so8t"

        # データセット設定（統合データセットを使用）
        if run_type == "baseline":
            # Baseline用: 数学・科学統合データ
            self.sft_datasets = ["data/train_sft_enhanced.jsonl"]
        else:
            # SO8T用: NKAT理論・NSFWデータ統合
            self.sft_datasets = ["data/aegis_phi35_v2_with_nkat_so8t/aegis_phi35_v2_with_nkat_so8t_sft_train.jsonl"]

        # ドメイン重み付け
        self.domain_weights = {
            'mathematics': 1.2,
            'science': 1.1,
            'reasoning': 1.0,
            'general': 0.8
        }

        # トレーニング設定（本格データセット対応）
        self.training_config = {
            'output_dir': f"H:/from_D/webdataset/checkpoints/sunshine_{run_type}_phase25",
            'num_train_epochs': 1,
            'max_steps': 50,  # テスト用に減らす
            'per_device_train_batch_size': 1,
            'gradient_accumulation_steps': 8,  # RTX3060対応
            'learning_rate': 3e-5,  # 安定した学習率
            'save_steps': 100,
            'logging_steps': 10,  # 10ステップごとログ
            'eval_steps': 100,
            'gradient_checkpointing': False,  # 一時的にOFFにして勾配問題を解決
            'ddp_find_unused_parameters': False,  # アダプタパラメータの検出を確実にする
            'optim': "adamw_8bit",
            'bf16': True,
            'dataloader_pin_memory': False,
            'remove_unused_columns': False,
            'report_to': [],
            'load_best_model_at_end': False
        }

        # LoRA設定
        self.lora_config = {
            'r': 16,
            'lora_alpha': 32,
            'lora_dropout': 0.05,
            'bias': "none",
            'task_type': "CAUSAL_LM",
            'target_modules': ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        }

        # SO(8)設定（so8tの場合のみ）
        if run_type == "so8t":
            self.so8_config = {
                'target_layers': [8, 16, 24],  # 中間層のみ
                'so8_dim': 8,
                'alpha_init': 0.1
            }
        else:
            self.so8_config = None


def run_sunshine_experiment(run_type: str = "baseline") -> Dict[str, Any]:
    """
    サンシャイン実験実行
    run_type: "baseline" or "so8t"
    """
    print(f"🌞 Starting Sunshine Experiment: {run_type.upper()}")
    print("=" * 60)

    # 設定
    config = SunshineTrainingConfig(run_type)
    run_name = f"sunshine_run_{run_type}"

    # ロガー初期化
    logger = SunshineLogger("logs/sunshine", run_name)

    try:
        # モデルとトークナイザーロード
        print("[1/5] Loading model and tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # ★★★ トレーニング時は device_map="auto" を使用せず直接GPUにロード ★★★
        # device_map="auto" は分散トレーニングと競合するため
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float16,
            device_map=None  # トレーニング時は明示的にNone
        ).to("cuda")  # 直接GPUに移動

        # ★★★ デバッグ: Phi-3モデルの構造を確認 ★★★
        print(f"Model type: {type(model)}")
        print(f"Model attributes: {[attr for attr in dir(model) if not attr.startswith('_')]}")
        if hasattr(model, 'model'):
            print(f"model.model type: {type(model.model)}")
            print(f"model.model attributes: {[attr for attr in dir(model.model) if not attr.startswith('_')]}")
            if hasattr(model.model, 'layers'):
                print(f"model.model.layers length: {len(model.model.layers)}")
                print(f"First layer type: {type(model.model.layers[0])}")
                print(f"First layer attributes: {[attr for attr in dir(model.model.layers[0]) if not attr.startswith('_')]}")
                if hasattr(model.model.layers[0], 'mlp'):
                    print(f"First layer has mlp: {type(model.model.layers[0].mlp)}")

        # ★★★ 処方箋1: 入力の勾配を強制的に有効化 ★★★
        # これがないと、Gradient Checkpointing有効時に途中の勾配が死ぬ
        model.enable_input_require_grads()

        # LoRA適用
        print("[2/5] Applying LoRA...")
        lora_config = LoraConfig(**config.lora_config)
        model = get_peft_model(model, lora_config)

        # LoRAパラメータを明示的にトレーニング可能に（詳細確認）
        lora_params_count = 0
        for name, param in model.named_parameters():
            if 'lora' in name.lower():
                param.requires_grad_(True)
                lora_params_count += 1

        print(f"Set {lora_params_count} LoRA parameters to trainable")

        # SO(8)アダプター適用（so8tの場合）
        if config.so8_config:
            print("[3/5] Applying SO(8) adapters (Layer Replacement Mode)...")
            enable_quad = config.so8_config.get('enable_quad_inference', False)
            # ★★★ 最終奥義: モンキーパッチで注入（Unsloth最適化突破）★★★
            model = replace_mlp_with_nkat(
                model,
                target_layers=config.so8_config['target_layers']
            )

            print(f"SO(8) Adapter with Quad Inference: {enable_quad}")
        else:
            print("[3/5] Skipping SO(8) adapters (baseline run)")

        # パラメータ確認
        print("[4/5] Checking trainable parameters...")
        model.print_trainable_parameters()

        # 🔥 緊急バイパス手術：Optimizerへの手動登録（SO8Tの場合のみ）
        if config.so8_config:
            print("[4.5/5] Manual optimizer registration for SO8T...")
            # 1. 学習対象パラメータの抽出 (LoRA と NKATアダプタ だけ)
            trainable_params = []
            for name, param in model.named_parameters():
                if "lora" in name.lower() or "nkat_adapter" in name.lower():
                    param.requires_grad = True
                    trainable_params.append(param)
                else:
                    param.requires_grad = False

            print(f"🔥 Total Trainable Params: {len(trainable_params)} tensors")

            # 2. Optimizerの手動作成 (Unsloth推奨の8bit AdamWを使う場合)
            try:
                from unsloth.optim import AdamW8bit
                optimizer = AdamW8bit(trainable_params, lr=config.training_config.get('learning_rate', 2e-5))
                print("✅ Using Unsloth AdamW8bit")
            except ImportError:
                from torch.optim import AdamW
                optimizer = AdamW(trainable_params, lr=config.training_config.get('learning_rate', 2e-5))
                print("✅ Using Standard AdamW")

            # 3. 後でTrainerに渡すための保存
            manual_optimizer = optimizer
        else:
            manual_optimizer = None

        # データセット準備
        print("[5/5] Preparing dataset...")
        dataset = SimpleDataset(
            config.sft_datasets[0],  # 最初のデータセットを使用
            tokenizer
        )
        print(f"Dataset size: {len(dataset)} samples")

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )

        # トレーニング引数
        training_args = TrainingArguments(**config.training_config)

        # カスタムコールバックでSO(8)メトリクス収集
        class SunshineCallback(TrainerCallback):
            def __init__(self, logger, model, run_type):
                self.logger = logger
                self.model = model
                self.run_type = run_type
                self.step_start_time = None

            def on_init_end(self, args, state, control, **kwargs):
                pass

            def on_train_begin(self, args, state, control, **kwargs):
                pass

            def on_train_end(self, args, state, control, **kwargs):
                pass

            def on_step_begin(self, args, state, control, **kwargs):
                self.step_start_time = time.time()

            def on_step_end(self, args, state, control, **kwargs):
                if self.step_start_time is None:
                    return

                step_time = time.time() - self.step_start_time
                step = state.global_step

                # SO(8)メトリクス収集
                so8_metrics = {}
                if self.run_type == "so8t":
                    ortho_errors = []
                    alphas = []
                    for name, module in self.model.named_modules():
                        if hasattr(module, 'get_adapter_stats'):
                            stats = module.get_adapter_stats()
                            ortho_errors.append(stats['orthogonality_error'])
                            alphas.append(stats['alpha'])

                    if ortho_errors:
                        so8_metrics['so8_ortho_mean'] = np.mean(ortho_errors)
                        so8_metrics['so8_ortho_max'] = np.max(ortho_errors)
                        so8_metrics['so8_alpha_mean'] = np.mean(alphas)
                        so8_metrics['so8_alpha_std'] = np.std(alphas) if len(alphas) > 1 else 0

                # 勾配ノルム（利用可能なら）
                grad_norm = None
                if hasattr(state, 'log_history') and state.log_history:
                    last_log = state.log_history[-1]
                    grad_norm = last_log.get('grad_norm')

                # ログ記録
                metrics = {
                    'train_loss': state.log_history[-1].get('loss') if state.log_history else None,
                    **so8_metrics,
                    'grad_norm': grad_norm
                }

                self.logger.log_step(step, metrics, step_time)

        callback = SunshineCallback(logger, model, run_type)

        # Trainer設定
        if manual_optimizer is not None:
            # SO8Tの場合：手動Optimizerを使用
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=dataset,
                data_collator=data_collator,
                callbacks=[callback, NKATDebugCallback(model)],
                optimizers=(manual_optimizer, None)  # (optimizer, scheduler)
            )
            print("🔧 Using manual optimizer for SO8T training")
        else:
            # Baselineの場合：通常のTrainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=dataset,
                data_collator=data_collator,
                callbacks=[callback, NKATDebugCallback(model)]
            )

        # トレーニング実行
        print(f"🚀 Starting {run_type.upper()} training...")
        trainer.train()

        # 最終メトリクス
        final_metrics = {}
        if trainer.state.log_history:
            final_metrics = trainer.state.log_history[-1]

        logger.finalize(final_metrics)

        print(f"✅ {run_type.upper()} training completed!")
        print(f"📊 Results saved to: {logger.log_dir}")

        return {
            'success': True,
            'run_type': run_type,
            'log_dir': str(logger.log_dir),
            'metrics_file': str(logger.metrics_file),
            'final_loss': final_metrics.get('loss')
        }

    except Exception as e:
        print(f"❌ {run_type.upper()} training failed: {e}")
        logger.finalize({'error': str(e)})
        return {
            'success': False,
            'run_type': run_type,
            'error': str(e)
        }


def run_sunshine_pipeline():
    """サンシャインパイプライン実行"""
    print("🌞🌞🌞 SO8T SUNSHINE PIPELINE 🌞🌞🌞")
    print("Comparing Baseline vs SO8T performance")
    print("=" * 60)

    results = {}

    # Run A: Baseline
    print("\n🏃 Run A: BASELINE (LoRA only)")
    results['baseline'] = run_sunshine_experiment("baseline")

    # Run B: SO8T
    print("\n🧬 Run B: SO8T (LoRA + SO(8) Adapter)")
    results['so8t'] = run_sunshine_experiment("so8t")

    # 結果比較
    print("\n📊 EXPERIMENT RESULTS SUMMARY")
    print("=" * 60)

    for run_type, result in results.items():
        status = "✅ SUCCESS" if result['success'] else "❌ FAILED"
        loss = f"Final Loss: {result.get('final_loss', 'N/A')}"
        print(f"{run_type.upper()}: {status} | {loss}")

    # ログファイル保存
    summary_file = Path("logs/sunshine") / "sunshine_summary.json"
    summary_file.parent.mkdir(parents=True, exist_ok=True)

    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'results': results
        }, f, indent=2, ensure_ascii=False)

    print(f"\n📋 Summary saved to: {summary_file}")

    # オーディオ通知
    try:
        import winsound
        winsound.Beep(1000, 1000)  # 1秒のビープ音
    except:
        pass

    return results


if __name__ == "__main__":
    # 引数処理
    if len(sys.argv) > 1:
        run_type = sys.argv[1]
        if run_type in ["baseline", "so8t"]:
            result = run_sunshine_experiment(run_type)
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print("Usage: python sunshine_pipeline.py [baseline|so8t]")
            sys.exit(1)
    else:
        # フルパイプライン実行
        run_sunshine_pipeline()

"""
SO8T Sunshine Pipeline - 実験計画ベースの自動実行システム

ボブにゃん戦略に基づくサンシャイン実行：
- Run A: Baseline (LoRAのみ)
- Run B: SO8T (LoRA + SO(8)アダプター)

ログフォーマット：
step, train_loss, eval_loss, so8_ortho_mean, so8_ortho_max, grad_norm, step_time_sec
"""

import os
import sys
import time
import json
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model

# Import from scripts directory
# from scripts.models.so8t_residual_adapter import attach_nkat_adapters  # 削除済み関数

# Simple dataset class for testing
from torch.utils.data import Dataset
import json

class SimpleDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, max_length: int = 128):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Load data
        self.data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        item = json.loads(line.strip())
                        self.data.append(item)
                    except json.JSONDecodeError:
                        continue

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item.get('text', '')

        # Tokenize
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
            'labels': tokenized['input_ids'].squeeze()
        }


class SunshineLogger:
    """サンシャイン実行の統一ログ収集"""

    def __init__(self, log_dir: str, run_name: str):
        self.log_dir = Path(log_dir)
        self.run_name = run_name
        self.log_file = self.log_dir / f"{run_name}_training_log.csv"
        self.metrics_file = self.log_dir / f"{run_name}_metrics.json"

        # CSVヘッダー
        self.columns = [
            'step', 'train_loss', 'eval_loss',
            'so8_ortho_mean', 'so8_ortho_max',
            'grad_norm', 'step_time_sec'
        ]

        # ログディレクトリ作成
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # CSV初期化
        if not self.log_file.exists():
            pd.DataFrame(columns=self.columns).to_csv(self.log_file, index=False)

        # メトリクス初期化
        self.metrics = {
            'start_time': datetime.now().isoformat(),
            'run_name': run_name,
            'total_steps': 0,
            'avg_step_time': 0,
            'final_train_loss': None,
            'final_eval_loss': None,
            'so8_ortho_errors': [],
            'grad_norms': []
        }

    def log_step(self, step: int, metrics: Dict[str, Any], step_time: float):
        """ステップごとのログ記録"""
        row = {
            'step': step,
            'train_loss': metrics.get('train_loss', None),
            'eval_loss': metrics.get('eval_loss', None),
            'so8_ortho_mean': metrics.get('so8_ortho_mean', None),
            'so8_ortho_max': metrics.get('so8_ortho_max', None),
            'grad_norm': metrics.get('grad_norm', None),
            'step_time_sec': step_time
        }

        # CSVに追加
        df = pd.DataFrame([row])
        df.to_csv(self.log_file, mode='a', header=False, index=False)

        # メトリクス更新
        if metrics.get('so8_ortho_mean') is not None:
            self.metrics['so8_ortho_errors'].append(metrics['so8_ortho_mean'])
        if metrics.get('grad_norm') is not None:
            self.metrics['grad_norms'].append(metrics['grad_norm'])

        self.metrics['total_steps'] = max(self.metrics['total_steps'], step)

    def finalize(self, final_metrics: Dict[str, Any]):
        """トレーニング完了時の最終ログ"""
        self.metrics.update({
            'end_time': datetime.now().isoformat(),
            'final_train_loss': final_metrics.get('train_loss'),
            'final_eval_loss': final_metrics.get('eval_loss'),
            'avg_so8_ortho_error': np.mean(self.metrics['so8_ortho_errors']) if self.metrics['so8_ortho_errors'] else None,
            'max_so8_ortho_error': np.max(self.metrics['so8_ortho_errors']) if self.metrics['so8_ortho_errors'] else None,
            'avg_grad_norm': np.mean(self.metrics['grad_norms']) if self.metrics['grad_norms'] else None
        })

        # JSON保存
        with open(self.metrics_file, 'w', encoding='utf-8') as f:
            json.dump(self.metrics, f, indent=2, ensure_ascii=False)


class SunshineTrainingConfig:
    """サンシャイン実行設定"""

    def __init__(self, run_type: str = "baseline"):
        self.model_name = "AXCXEPT/Borea-Phi-3.5-mini-Instruct-Jp"
        self.run_type = run_type  # "baseline" or "so8t"

        # データセット設定（統合データセットを使用）
        if run_type == "baseline":
            # Baseline用: 数学・科学統合データ
            self.sft_datasets = ["data/train_sft_enhanced.jsonl"]
        else:
            # SO8T用: NKAT理論・NSFWデータ統合
            self.sft_datasets = ["data/aegis_phi35_v2_with_nkat_so8t/aegis_phi35_v2_with_nkat_so8t_sft_train.jsonl"]

        # ドメイン重み付け
        self.domain_weights = {
            'mathematics': 1.2,
            'science': 1.1,
            'reasoning': 1.0,
            'general': 0.8
        }

        # トレーニング設定（本格データセット対応）
        self.training_config = {
            'output_dir': f"H:/from_D/webdataset/checkpoints/sunshine_{run_type}_phase25",
            'num_train_epochs': 1,
            'max_steps': 50,  # テスト用に減らす
            'per_device_train_batch_size': 1,
            'gradient_accumulation_steps': 8,  # RTX3060対応
            'learning_rate': 3e-5,  # 安定した学習率
            'save_steps': 100,
            'logging_steps': 10,  # 10ステップごとログ
            'eval_steps': 100,
            'gradient_checkpointing': False,  # 一時的にOFFにして勾配問題を解決
            'ddp_find_unused_parameters': False,  # アダプタパラメータの検出を確実にする
            'optim': "adamw_8bit",
            'bf16': True,
            'dataloader_pin_memory': False,
            'remove_unused_columns': False,
            'report_to': [],
            'load_best_model_at_end': False
        }

        # LoRA設定
        self.lora_config = {
            'r': 16,
            'lora_alpha': 32,
            'lora_dropout': 0.05,
            'bias': "none",
            'task_type': "CAUSAL_LM",
            'target_modules': ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        }

        # SO(8)設定（so8tの場合のみ）
        if run_type == "so8t":
            self.so8_config = {
                'target_layers': [8, 16, 24],  # 中間層のみ
                'so8_dim': 8,
                'alpha_init': 0.1
            }
        else:
            self.so8_config = None


def run_sunshine_experiment(run_type: str = "baseline") -> Dict[str, Any]:
    """
    サンシャイン実験実行
    run_type: "baseline" or "so8t"
    """
    print(f"🌞 Starting Sunshine Experiment: {run_type.upper()}")
    print("=" * 60)

    # 設定
    config = SunshineTrainingConfig(run_type)
    run_name = f"sunshine_run_{run_type}"

    # ロガー初期化
    logger = SunshineLogger("logs/sunshine", run_name)

    try:
        # モデルとトークナイザーロード
        print("[1/5] Loading model and tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # ★★★ トレーニング時は device_map="auto" を使用せず直接GPUにロード ★★★
        # device_map="auto" は分散トレーニングと競合するため
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float16,
            device_map=None  # トレーニング時は明示的にNone
        ).to("cuda")  # 直接GPUに移動

        # ★★★ デバッグ: Phi-3モデルの構造を確認 ★★★
        print(f"Model type: {type(model)}")
        print(f"Model attributes: {[attr for attr in dir(model) if not attr.startswith('_')]}")
        if hasattr(model, 'model'):
            print(f"model.model type: {type(model.model)}")
            print(f"model.model attributes: {[attr for attr in dir(model.model) if not attr.startswith('_')]}")
            if hasattr(model.model, 'layers'):
                print(f"model.model.layers length: {len(model.model.layers)}")
                print(f"First layer type: {type(model.model.layers[0])}")
                print(f"First layer attributes: {[attr for attr in dir(model.model.layers[0]) if not attr.startswith('_')]}")
                if hasattr(model.model.layers[0], 'mlp'):
                    print(f"First layer has mlp: {type(model.model.layers[0].mlp)}")

        # ★★★ 処方箋1: 入力の勾配を強制的に有効化 ★★★
        # これがないと、Gradient Checkpointing有効時に途中の勾配が死ぬ
        model.enable_input_require_grads()

        # LoRA適用
        print("[2/5] Applying LoRA...")
        lora_config = LoraConfig(**config.lora_config)
        model = get_peft_model(model, lora_config)

        # LoRAパラメータを明示的にトレーニング可能に（詳細確認）
        lora_params_count = 0
        for name, param in model.named_parameters():
            if 'lora' in name.lower():
                param.requires_grad_(True)
                lora_params_count += 1

        print(f"Set {lora_params_count} LoRA parameters to trainable")

        # SO(8)アダプター適用（so8tの場合）
        if config.so8_config:
            print("[3/5] Applying SO(8) adapters (Layer Replacement Mode)...")
            enable_quad = config.so8_config.get('enable_quad_inference', False)
            # ★★★ 最終奥義: モンキーパッチで注入（Unsloth最適化突破）★★★
            model = replace_mlp_with_nkat(
                model,
                target_layers=config.so8_config['target_layers']
            )

            print(f"SO(8) Adapter with Quad Inference: {enable_quad}")
        else:
            print("[3/5] Skipping SO(8) adapters (baseline run)")

        # パラメータ確認
        print("[4/5] Checking trainable parameters...")
        model.print_trainable_parameters()

        # 🔥 緊急バイパス手術：Optimizerへの手動登録（SO8Tの場合のみ）
        if config.so8_config:
            print("[4.5/5] Manual optimizer registration for SO8T...")
            # 1. 学習対象パラメータの抽出 (LoRA と NKATアダプタ だけ)
            trainable_params = []
            for name, param in model.named_parameters():
                if "lora" in name.lower() or "nkat_adapter" in name.lower():
                    param.requires_grad = True
                    trainable_params.append(param)
                else:
                    param.requires_grad = False

            print(f"🔥 Total Trainable Params: {len(trainable_params)} tensors")

            # 2. Optimizerの手動作成 (Unsloth推奨の8bit AdamWを使う場合)
            try:
                from unsloth.optim import AdamW8bit
                optimizer = AdamW8bit(trainable_params, lr=config.training_config.get('learning_rate', 2e-5))
                print("✅ Using Unsloth AdamW8bit")
            except ImportError:
                from torch.optim import AdamW
                optimizer = AdamW(trainable_params, lr=config.training_config.get('learning_rate', 2e-5))
                print("✅ Using Standard AdamW")

            # 3. 後でTrainerに渡すための保存
            manual_optimizer = optimizer
        else:
            manual_optimizer = None

        # データセット準備
        print("[5/5] Preparing dataset...")
        dataset = SimpleDataset(
            config.sft_datasets[0],  # 最初のデータセットを使用
            tokenizer
        )
        print(f"Dataset size: {len(dataset)} samples")

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )

        # トレーニング引数
        training_args = TrainingArguments(**config.training_config)

        # カスタムコールバックでSO(8)メトリクス収集
        class SunshineCallback(TrainerCallback):
            def __init__(self, logger, model, run_type):
                self.logger = logger
                self.model = model
                self.run_type = run_type
                self.step_start_time = None

            def on_init_end(self, args, state, control, **kwargs):
                pass

            def on_train_begin(self, args, state, control, **kwargs):
                pass

            def on_train_end(self, args, state, control, **kwargs):
                pass

            def on_step_begin(self, args, state, control, **kwargs):
                self.step_start_time = time.time()

            def on_step_end(self, args, state, control, **kwargs):
                if self.step_start_time is None:
                    return

                step_time = time.time() - self.step_start_time
                step = state.global_step

                # SO(8)メトリクス収集
                so8_metrics = {}
                if self.run_type == "so8t":
                    ortho_errors = []
                    alphas = []
                    for name, module in self.model.named_modules():
                        if hasattr(module, 'get_adapter_stats'):
                            stats = module.get_adapter_stats()
                            ortho_errors.append(stats['orthogonality_error'])
                            alphas.append(stats['alpha'])

                    if ortho_errors:
                        so8_metrics['so8_ortho_mean'] = np.mean(ortho_errors)
                        so8_metrics['so8_ortho_max'] = np.max(ortho_errors)
                        so8_metrics['so8_alpha_mean'] = np.mean(alphas)
                        so8_metrics['so8_alpha_std'] = np.std(alphas) if len(alphas) > 1 else 0

                # 勾配ノルム（利用可能なら）
                grad_norm = None
                if hasattr(state, 'log_history') and state.log_history:
                    last_log = state.log_history[-1]
                    grad_norm = last_log.get('grad_norm')

                # ログ記録
                metrics = {
                    'train_loss': state.log_history[-1].get('loss') if state.log_history else None,
                    **so8_metrics,
                    'grad_norm': grad_norm
                }

                self.logger.log_step(step, metrics, step_time)

        callback = SunshineCallback(logger, model, run_type)

        # Trainer設定
        if manual_optimizer is not None:
            # SO8Tの場合：手動Optimizerを使用
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=dataset,
                data_collator=data_collator,
                callbacks=[callback, NKATDebugCallback(model)],
                optimizers=(manual_optimizer, None)  # (optimizer, scheduler)
            )
            print("🔧 Using manual optimizer for SO8T training")
        else:
            # Baselineの場合：通常のTrainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=dataset,
                data_collator=data_collator,
                callbacks=[callback, NKATDebugCallback(model)]
            )

        # トレーニング実行
        print(f"🚀 Starting {run_type.upper()} training...")
        trainer.train()

        # 最終メトリクス
        final_metrics = {}
        if trainer.state.log_history:
            final_metrics = trainer.state.log_history[-1]

        logger.finalize(final_metrics)

        print(f"✅ {run_type.upper()} training completed!")
        print(f"📊 Results saved to: {logger.log_dir}")

        return {
            'success': True,
            'run_type': run_type,
            'log_dir': str(logger.log_dir),
            'metrics_file': str(logger.metrics_file),
            'final_loss': final_metrics.get('loss')
        }

    except Exception as e:
        print(f"❌ {run_type.upper()} training failed: {e}")
        logger.finalize({'error': str(e)})
        return {
            'success': False,
            'run_type': run_type,
            'error': str(e)
        }


def run_sunshine_pipeline():
    """サンシャインパイプライン実行"""
    print("🌞🌞🌞 SO8T SUNSHINE PIPELINE 🌞🌞🌞")
    print("Comparing Baseline vs SO8T performance")
    print("=" * 60)

    results = {}

    # Run A: Baseline
    print("\n🏃 Run A: BASELINE (LoRA only)")
    results['baseline'] = run_sunshine_experiment("baseline")

    # Run B: SO8T
    print("\n🧬 Run B: SO8T (LoRA + SO(8) Adapter)")
    results['so8t'] = run_sunshine_experiment("so8t")

    # 結果比較
    print("\n📊 EXPERIMENT RESULTS SUMMARY")
    print("=" * 60)

    for run_type, result in results.items():
        status = "✅ SUCCESS" if result['success'] else "❌ FAILED"
        loss = f"Final Loss: {result.get('final_loss', 'N/A')}"
        print(f"{run_type.upper()}: {status} | {loss}")

    # ログファイル保存
    summary_file = Path("logs/sunshine") / "sunshine_summary.json"
    summary_file.parent.mkdir(parents=True, exist_ok=True)

    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'results': results
        }, f, indent=2, ensure_ascii=False)

    print(f"\n📋 Summary saved to: {summary_file}")

    # オーディオ通知
    try:
        import winsound
        winsound.Beep(1000, 1000)  # 1秒のビープ音
    except:
        pass

    return results


if __name__ == "__main__":
    # 引数処理
    if len(sys.argv) > 1:
        run_type = sys.argv[1]
        if run_type in ["baseline", "so8t"]:
            result = run_sunshine_experiment(run_type)
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print("Usage: python sunshine_pipeline.py [baseline|so8t]")
            sys.exit(1)
    else:
        # フルパイプライン実行
        run_sunshine_pipeline()

"""
SO8T Sunshine Pipeline - 実験計画ベースの自動実行システム

ボブにゃん戦略に基づくサンシャイン実行：
- Run A: Baseline (LoRAのみ)
- Run B: SO8T (LoRA + SO(8)アダプター)

ログフォーマット：
step, train_loss, eval_loss, so8_ortho_mean, so8_ortho_max, grad_norm, step_time_sec
"""

import os
import sys
import time
import json
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model

# Import from scripts directory
# from scripts.models.so8t_residual_adapter import attach_nkat_adapters  # 削除済み関数

# Simple dataset class for testing
from torch.utils.data import Dataset
import json

class SimpleDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, max_length: int = 128):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Load data
        self.data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        item = json.loads(line.strip())
                        self.data.append(item)
                    except json.JSONDecodeError:
                        continue

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item.get('text', '')

        # Tokenize
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
            'labels': tokenized['input_ids'].squeeze()
        }


class SunshineLogger:
    """サンシャイン実行の統一ログ収集"""

    def __init__(self, log_dir: str, run_name: str):
        self.log_dir = Path(log_dir)
        self.run_name = run_name
        self.log_file = self.log_dir / f"{run_name}_training_log.csv"
        self.metrics_file = self.log_dir / f"{run_name}_metrics.json"

        # CSVヘッダー
        self.columns = [
            'step', 'train_loss', 'eval_loss',
            'so8_ortho_mean', 'so8_ortho_max',
            'grad_norm', 'step_time_sec'
        ]

        # ログディレクトリ作成
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # CSV初期化
        if not self.log_file.exists():
            pd.DataFrame(columns=self.columns).to_csv(self.log_file, index=False)

        # メトリクス初期化
        self.metrics = {
            'start_time': datetime.now().isoformat(),
            'run_name': run_name,
            'total_steps': 0,
            'avg_step_time': 0,
            'final_train_loss': None,
            'final_eval_loss': None,
            'so8_ortho_errors': [],
            'grad_norms': []
        }

    def log_step(self, step: int, metrics: Dict[str, Any], step_time: float):
        """ステップごとのログ記録"""
        row = {
            'step': step,
            'train_loss': metrics.get('train_loss', None),
            'eval_loss': metrics.get('eval_loss', None),
            'so8_ortho_mean': metrics.get('so8_ortho_mean', None),
            'so8_ortho_max': metrics.get('so8_ortho_max', None),
            'grad_norm': metrics.get('grad_norm', None),
            'step_time_sec': step_time
        }

        # CSVに追加
        df = pd.DataFrame([row])
        df.to_csv(self.log_file, mode='a', header=False, index=False)

        # メトリクス更新
        if metrics.get('so8_ortho_mean') is not None:
            self.metrics['so8_ortho_errors'].append(metrics['so8_ortho_mean'])
        if metrics.get('grad_norm') is not None:
            self.metrics['grad_norms'].append(metrics['grad_norm'])

        self.metrics['total_steps'] = max(self.metrics['total_steps'], step)

    def finalize(self, final_metrics: Dict[str, Any]):
        """トレーニング完了時の最終ログ"""
        self.metrics.update({
            'end_time': datetime.now().isoformat(),
            'final_train_loss': final_metrics.get('train_loss'),
            'final_eval_loss': final_metrics.get('eval_loss'),
            'avg_so8_ortho_error': np.mean(self.metrics['so8_ortho_errors']) if self.metrics['so8_ortho_errors'] else None,
            'max_so8_ortho_error': np.max(self.metrics['so8_ortho_errors']) if self.metrics['so8_ortho_errors'] else None,
            'avg_grad_norm': np.mean(self.metrics['grad_norms']) if self.metrics['grad_norms'] else None
        })

        # JSON保存
        with open(self.metrics_file, 'w', encoding='utf-8') as f:
            json.dump(self.metrics, f, indent=2, ensure_ascii=False)


class SunshineTrainingConfig:
    """サンシャイン実行設定"""

    def __init__(self, run_type: str = "baseline"):
        self.model_name = "AXCXEPT/Borea-Phi-3.5-mini-Instruct-Jp"
        self.run_type = run_type  # "baseline" or "so8t"

        # データセット設定（統合データセットを使用）
        if run_type == "baseline":
            # Baseline用: 数学・科学統合データ
            self.sft_datasets = ["data/train_sft_enhanced.jsonl"]
        else:
            # SO8T用: NKAT理論・NSFWデータ統合
            self.sft_datasets = ["data/aegis_phi35_v2_with_nkat_so8t/aegis_phi35_v2_with_nkat_so8t_sft_train.jsonl"]

        # ドメイン重み付け
        self.domain_weights = {
            'mathematics': 1.2,
            'science': 1.1,
            'reasoning': 1.0,
            'general': 0.8
        }

        # トレーニング設定（本格データセット対応）
        self.training_config = {
            'output_dir': f"H:/from_D/webdataset/checkpoints/sunshine_{run_type}_phase25",
            'num_train_epochs': 1,
            'max_steps': 50,  # テスト用に減らす
            'per_device_train_batch_size': 1,
            'gradient_accumulation_steps': 8,  # RTX3060対応
            'learning_rate': 3e-5,  # 安定した学習率
            'save_steps': 100,
            'logging_steps': 10,  # 10ステップごとログ
            'eval_steps': 100,
            'gradient_checkpointing': False,  # 一時的にOFFにして勾配問題を解決
            'ddp_find_unused_parameters': False,  # アダプタパラメータの検出を確実にする
            'optim': "adamw_8bit",
            'bf16': True,
            'dataloader_pin_memory': False,
            'remove_unused_columns': False,
            'report_to': [],
            'load_best_model_at_end': False
        }

        # LoRA設定
        self.lora_config = {
            'r': 16,
            'lora_alpha': 32,
            'lora_dropout': 0.05,
            'bias': "none",
            'task_type': "CAUSAL_LM",
            'target_modules': ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        }

        # SO(8)設定（so8tの場合のみ）
        if run_type == "so8t":
            self.so8_config = {
                'target_layers': [8, 16, 24],  # 中間層のみ
                'so8_dim': 8,
                'alpha_init': 0.1
            }
        else:
            self.so8_config = None


def run_sunshine_experiment(run_type: str = "baseline") -> Dict[str, Any]:
    """
    サンシャイン実験実行
    run_type: "baseline" or "so8t"
    """
    print(f"🌞 Starting Sunshine Experiment: {run_type.upper()}")
    print("=" * 60)

    # 設定
    config = SunshineTrainingConfig(run_type)
    run_name = f"sunshine_run_{run_type}"

    # ロガー初期化
    logger = SunshineLogger("logs/sunshine", run_name)

    try:
        # モデルとトークナイザーロード
        print("[1/5] Loading model and tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # ★★★ トレーニング時は device_map="auto" を使用せず直接GPUにロード ★★★
        # device_map="auto" は分散トレーニングと競合するため
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float16,
            device_map=None  # トレーニング時は明示的にNone
        ).to("cuda")  # 直接GPUに移動

        # ★★★ デバッグ: Phi-3モデルの構造を確認 ★★★
        print(f"Model type: {type(model)}")
        print(f"Model attributes: {[attr for attr in dir(model) if not attr.startswith('_')]}")
        if hasattr(model, 'model'):
            print(f"model.model type: {type(model.model)}")
            print(f"model.model attributes: {[attr for attr in dir(model.model) if not attr.startswith('_')]}")
            if hasattr(model.model, 'layers'):
                print(f"model.model.layers length: {len(model.model.layers)}")
                print(f"First layer type: {type(model.model.layers[0])}")
                print(f"First layer attributes: {[attr for attr in dir(model.model.layers[0]) if not attr.startswith('_')]}")
                if hasattr(model.model.layers[0], 'mlp'):
                    print(f"First layer has mlp: {type(model.model.layers[0].mlp)}")

        # ★★★ 処方箋1: 入力の勾配を強制的に有効化 ★★★
        # これがないと、Gradient Checkpointing有効時に途中の勾配が死ぬ
        model.enable_input_require_grads()

        # LoRA適用
        print("[2/5] Applying LoRA...")
        lora_config = LoraConfig(**config.lora_config)
        model = get_peft_model(model, lora_config)

        # LoRAパラメータを明示的にトレーニング可能に（詳細確認）
        lora_params_count = 0
        for name, param in model.named_parameters():
            if 'lora' in name.lower():
                param.requires_grad_(True)
                lora_params_count += 1

        print(f"Set {lora_params_count} LoRA parameters to trainable")

        # SO(8)アダプター適用（so8tの場合）
        if config.so8_config:
            print("[3/5] Applying SO(8) adapters (Layer Replacement Mode)...")
            enable_quad = config.so8_config.get('enable_quad_inference', False)
            # ★★★ 最終奥義: モンキーパッチで注入（Unsloth最適化突破）★★★
            model = replace_mlp_with_nkat(
                model,
                target_layers=config.so8_config['target_layers']
            )

            print(f"SO(8) Adapter with Quad Inference: {enable_quad}")
        else:
            print("[3/5] Skipping SO(8) adapters (baseline run)")

        # パラメータ確認
        print("[4/5] Checking trainable parameters...")
        model.print_trainable_parameters()

        # 🔥 緊急バイパス手術：Optimizerへの手動登録（SO8Tの場合のみ）
        if config.so8_config:
            print("[4.5/5] Manual optimizer registration for SO8T...")
            # 1. 学習対象パラメータの抽出 (LoRA と NKATアダプタ だけ)
            trainable_params = []
            for name, param in model.named_parameters():
                if "lora" in name.lower() or "nkat_adapter" in name.lower():
                    param.requires_grad = True
                    trainable_params.append(param)
                else:
                    param.requires_grad = False

            print(f"🔥 Total Trainable Params: {len(trainable_params)} tensors")

            # 2. Optimizerの手動作成 (Unsloth推奨の8bit AdamWを使う場合)
            try:
                from unsloth.optim import AdamW8bit
                optimizer = AdamW8bit(trainable_params, lr=config.training_config.get('learning_rate', 2e-5))
                print("✅ Using Unsloth AdamW8bit")
            except ImportError:
                from torch.optim import AdamW
                optimizer = AdamW(trainable_params, lr=config.training_config.get('learning_rate', 2e-5))
                print("✅ Using Standard AdamW")

            # 3. 後でTrainerに渡すための保存
            manual_optimizer = optimizer
        else:
            manual_optimizer = None

        # データセット準備
        print("[5/5] Preparing dataset...")
        dataset = SimpleDataset(
            config.sft_datasets[0],  # 最初のデータセットを使用
            tokenizer
        )
        print(f"Dataset size: {len(dataset)} samples")

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )

        # トレーニング引数
        training_args = TrainingArguments(**config.training_config)

        # カスタムコールバックでSO(8)メトリクス収集
        class SunshineCallback(TrainerCallback):
            def __init__(self, logger, model, run_type):
                self.logger = logger
                self.model = model
                self.run_type = run_type
                self.step_start_time = None

            def on_init_end(self, args, state, control, **kwargs):
                pass

            def on_train_begin(self, args, state, control, **kwargs):
                pass

            def on_train_end(self, args, state, control, **kwargs):
                pass

            def on_step_begin(self, args, state, control, **kwargs):
                self.step_start_time = time.time()

            def on_step_end(self, args, state, control, **kwargs):
                if self.step_start_time is None:
                    return

                step_time = time.time() - self.step_start_time
                step = state.global_step

                # SO(8)メトリクス収集
                so8_metrics = {}
                if self.run_type == "so8t":
                    ortho_errors = []
                    alphas = []
                    for name, module in self.model.named_modules():
                        if hasattr(module, 'get_adapter_stats'):
                            stats = module.get_adapter_stats()
                            ortho_errors.append(stats['orthogonality_error'])
                            alphas.append(stats['alpha'])

                    if ortho_errors:
                        so8_metrics['so8_ortho_mean'] = np.mean(ortho_errors)
                        so8_metrics['so8_ortho_max'] = np.max(ortho_errors)
                        so8_metrics['so8_alpha_mean'] = np.mean(alphas)
                        so8_metrics['so8_alpha_std'] = np.std(alphas) if len(alphas) > 1 else 0

                # 勾配ノルム（利用可能なら）
                grad_norm = None
                if hasattr(state, 'log_history') and state.log_history:
                    last_log = state.log_history[-1]
                    grad_norm = last_log.get('grad_norm')

                # ログ記録
                metrics = {
                    'train_loss': state.log_history[-1].get('loss') if state.log_history else None,
                    **so8_metrics,
                    'grad_norm': grad_norm
                }

                self.logger.log_step(step, metrics, step_time)

        callback = SunshineCallback(logger, model, run_type)

        # Trainer設定
        if manual_optimizer is not None:
            # SO8Tの場合：手動Optimizerを使用
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=dataset,
                data_collator=data_collator,
                callbacks=[callback, NKATDebugCallback(model)],
                optimizers=(manual_optimizer, None)  # (optimizer, scheduler)
            )
            print("🔧 Using manual optimizer for SO8T training")
        else:
            # Baselineの場合：通常のTrainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=dataset,
                data_collator=data_collator,
                callbacks=[callback, NKATDebugCallback(model)]
            )

        # トレーニング実行
        print(f"🚀 Starting {run_type.upper()} training...")
        trainer.train()

        # 最終メトリクス
        final_metrics = {}
        if trainer.state.log_history:
            final_metrics = trainer.state.log_history[-1]

        logger.finalize(final_metrics)

        print(f"✅ {run_type.upper()} training completed!")
        print(f"📊 Results saved to: {logger.log_dir}")

        return {
            'success': True,
            'run_type': run_type,
            'log_dir': str(logger.log_dir),
            'metrics_file': str(logger.metrics_file),
            'final_loss': final_metrics.get('loss')
        }

    except Exception as e:
        print(f"❌ {run_type.upper()} training failed: {e}")
        logger.finalize({'error': str(e)})
        return {
            'success': False,
            'run_type': run_type,
            'error': str(e)
        }


def run_sunshine_pipeline():
    """サンシャインパイプライン実行"""
    print("🌞🌞🌞 SO8T SUNSHINE PIPELINE 🌞🌞🌞")
    print("Comparing Baseline vs SO8T performance")
    print("=" * 60)

    results = {}

    # Run A: Baseline
    print("\n🏃 Run A: BASELINE (LoRA only)")
    results['baseline'] = run_sunshine_experiment("baseline")

    # Run B: SO8T
    print("\n🧬 Run B: SO8T (LoRA + SO(8) Adapter)")
    results['so8t'] = run_sunshine_experiment("so8t")

    # 結果比較
    print("\n📊 EXPERIMENT RESULTS SUMMARY")
    print("=" * 60)

    for run_type, result in results.items():
        status = "✅ SUCCESS" if result['success'] else "❌ FAILED"
        loss = f"Final Loss: {result.get('final_loss', 'N/A')}"
        print(f"{run_type.upper()}: {status} | {loss}")

    # ログファイル保存
    summary_file = Path("logs/sunshine") / "sunshine_summary.json"
    summary_file.parent.mkdir(parents=True, exist_ok=True)

    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'results': results
        }, f, indent=2, ensure_ascii=False)

    print(f"\n📋 Summary saved to: {summary_file}")

    # オーディオ通知
    try:
        import winsound
        winsound.Beep(1000, 1000)  # 1秒のビープ音
    except:
        pass

    return results


if __name__ == "__main__":
    # 引数処理
    if len(sys.argv) > 1:
        run_type = sys.argv[1]
        if run_type in ["baseline", "so8t"]:
            result = run_sunshine_experiment(run_type)
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print("Usage: python sunshine_pipeline.py [baseline|so8t]")
            sys.exit(1)
    else:
        # フルパイプライン実行
        run_sunshine_pipeline()

"""
SO8T Sunshine Pipeline - 実験計画ベースの自動実行システム

ボブにゃん戦略に基づくサンシャイン実行：
- Run A: Baseline (LoRAのみ)
- Run B: SO8T (LoRA + SO(8)アダプター)

ログフォーマット：
step, train_loss, eval_loss, so8_ortho_mean, so8_ortho_max, grad_norm, step_time_sec
"""

import os
import sys
import time
import json
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model

# Import from scripts directory
# from scripts.models.so8t_residual_adapter import attach_nkat_adapters  # 削除済み関数

# Simple dataset class for testing
from torch.utils.data import Dataset
import json

class SimpleDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, max_length: int = 128):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Load data
        self.data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        item = json.loads(line.strip())
                        self.data.append(item)
                    except json.JSONDecodeError:
                        continue

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item.get('text', '')

        # Tokenize
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
            'labels': tokenized['input_ids'].squeeze()
        }


class SunshineLogger:
    """サンシャイン実行の統一ログ収集"""

    def __init__(self, log_dir: str, run_name: str):
        self.log_dir = Path(log_dir)
        self.run_name = run_name
        self.log_file = self.log_dir / f"{run_name}_training_log.csv"
        self.metrics_file = self.log_dir / f"{run_name}_metrics.json"

        # CSVヘッダー
        self.columns = [
            'step', 'train_loss', 'eval_loss',
            'so8_ortho_mean', 'so8_ortho_max',
            'grad_norm', 'step_time_sec'
        ]

        # ログディレクトリ作成
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # CSV初期化
        if not self.log_file.exists():
            pd.DataFrame(columns=self.columns).to_csv(self.log_file, index=False)

        # メトリクス初期化
        self.metrics = {
            'start_time': datetime.now().isoformat(),
            'run_name': run_name,
            'total_steps': 0,
            'avg_step_time': 0,
            'final_train_loss': None,
            'final_eval_loss': None,
            'so8_ortho_errors': [],
            'grad_norms': []
        }

    def log_step(self, step: int, metrics: Dict[str, Any], step_time: float):
        """ステップごとのログ記録"""
        row = {
            'step': step,
            'train_loss': metrics.get('train_loss', None),
            'eval_loss': metrics.get('eval_loss', None),
            'so8_ortho_mean': metrics.get('so8_ortho_mean', None),
            'so8_ortho_max': metrics.get('so8_ortho_max', None),
            'grad_norm': metrics.get('grad_norm', None),
            'step_time_sec': step_time
        }

        # CSVに追加
        df = pd.DataFrame([row])
        df.to_csv(self.log_file, mode='a', header=False, index=False)

        # メトリクス更新
        if metrics.get('so8_ortho_mean') is not None:
            self.metrics['so8_ortho_errors'].append(metrics['so8_ortho_mean'])
        if metrics.get('grad_norm') is not None:
            self.metrics['grad_norms'].append(metrics['grad_norm'])

        self.metrics['total_steps'] = max(self.metrics['total_steps'], step)

    def finalize(self, final_metrics: Dict[str, Any]):
        """トレーニング完了時の最終ログ"""
        self.metrics.update({
            'end_time': datetime.now().isoformat(),
            'final_train_loss': final_metrics.get('train_loss'),
            'final_eval_loss': final_metrics.get('eval_loss'),
            'avg_so8_ortho_error': np.mean(self.metrics['so8_ortho_errors']) if self.metrics['so8_ortho_errors'] else None,
            'max_so8_ortho_error': np.max(self.metrics['so8_ortho_errors']) if self.metrics['so8_ortho_errors'] else None,
            'avg_grad_norm': np.mean(self.metrics['grad_norms']) if self.metrics['grad_norms'] else None
        })

        # JSON保存
        with open(self.metrics_file, 'w', encoding='utf-8') as f:
            json.dump(self.metrics, f, indent=2, ensure_ascii=False)


class SunshineTrainingConfig:
    """サンシャイン実行設定"""

    def __init__(self, run_type: str = "baseline"):
        self.model_name = "AXCXEPT/Borea-Phi-3.5-mini-Instruct-Jp"
        self.run_type = run_type  # "baseline" or "so8t"

        # データセット設定（統合データセットを使用）
        if run_type == "baseline":
            # Baseline用: 数学・科学統合データ
            self.sft_datasets = ["data/train_sft_enhanced.jsonl"]
        else:
            # SO8T用: NKAT理論・NSFWデータ統合
            self.sft_datasets = ["data/aegis_phi35_v2_with_nkat_so8t/aegis_phi35_v2_with_nkat_so8t_sft_train.jsonl"]

        # ドメイン重み付け
        self.domain_weights = {
            'mathematics': 1.2,
            'science': 1.1,
            'reasoning': 1.0,
            'general': 0.8
        }

        # トレーニング設定（本格データセット対応）
        self.training_config = {
            'output_dir': f"H:/from_D/webdataset/checkpoints/sunshine_{run_type}_phase25",
            'num_train_epochs': 1,
            'max_steps': 50,  # テスト用に減らす
            'per_device_train_batch_size': 1,
            'gradient_accumulation_steps': 8,  # RTX3060対応
            'learning_rate': 3e-5,  # 安定した学習率
            'save_steps': 100,
            'logging_steps': 10,  # 10ステップごとログ
            'eval_steps': 100,
            'gradient_checkpointing': False,  # 一時的にOFFにして勾配問題を解決
            'ddp_find_unused_parameters': False,  # アダプタパラメータの検出を確実にする
            'optim': "adamw_8bit",
            'bf16': True,
            'dataloader_pin_memory': False,
            'remove_unused_columns': False,
            'report_to': [],
            'load_best_model_at_end': False
        }

        # LoRA設定
        self.lora_config = {
            'r': 16,
            'lora_alpha': 32,
            'lora_dropout': 0.05,
            'bias': "none",
            'task_type': "CAUSAL_LM",
            'target_modules': ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        }

        # SO(8)設定（so8tの場合のみ）
        if run_type == "so8t":
            self.so8_config = {
                'target_layers': [8, 16, 24],  # 中間層のみ
                'so8_dim': 8,
                'alpha_init': 0.1
            }
        else:
            self.so8_config = None


def run_sunshine_experiment(run_type: str = "baseline") -> Dict[str, Any]:
    """
    サンシャイン実験実行
    run_type: "baseline" or "so8t"
    """
    print(f"🌞 Starting Sunshine Experiment: {run_type.upper()}")
    print("=" * 60)

    # 設定
    config = SunshineTrainingConfig(run_type)
    run_name = f"sunshine_run_{run_type}"

    # ロガー初期化
    logger = SunshineLogger("logs/sunshine", run_name)

    try:
        # モデルとトークナイザーロード
        print("[1/5] Loading model and tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # ★★★ トレーニング時は device_map="auto" を使用せず直接GPUにロード ★★★
        # device_map="auto" は分散トレーニングと競合するため
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float16,
            device_map=None  # トレーニング時は明示的にNone
        ).to("cuda")  # 直接GPUに移動

        # ★★★ デバッグ: Phi-3モデルの構造を確認 ★★★
        print(f"Model type: {type(model)}")
        print(f"Model attributes: {[attr for attr in dir(model) if not attr.startswith('_')]}")
        if hasattr(model, 'model'):
            print(f"model.model type: {type(model.model)}")
            print(f"model.model attributes: {[attr for attr in dir(model.model) if not attr.startswith('_')]}")
            if hasattr(model.model, 'layers'):
                print(f"model.model.layers length: {len(model.model.layers)}")
                print(f"First layer type: {type(model.model.layers[0])}")
                print(f"First layer attributes: {[attr for attr in dir(model.model.layers[0]) if not attr.startswith('_')]}")
                if hasattr(model.model.layers[0], 'mlp'):
                    print(f"First layer has mlp: {type(model.model.layers[0].mlp)}")

        # ★★★ 処方箋1: 入力の勾配を強制的に有効化 ★★★
        # これがないと、Gradient Checkpointing有効時に途中の勾配が死ぬ
        model.enable_input_require_grads()

        # LoRA適用
        print("[2/5] Applying LoRA...")
        lora_config = LoraConfig(**config.lora_config)
        model = get_peft_model(model, lora_config)

        # LoRAパラメータを明示的にトレーニング可能に（詳細確認）
        lora_params_count = 0
        for name, param in model.named_parameters():
            if 'lora' in name.lower():
                param.requires_grad_(True)
                lora_params_count += 1

        print(f"Set {lora_params_count} LoRA parameters to trainable")

        # SO(8)アダプター適用（so8tの場合）
        if config.so8_config:
            print("[3/5] Applying SO(8) adapters (Layer Replacement Mode)...")
            enable_quad = config.so8_config.get('enable_quad_inference', False)
            # ★★★ 最終奥義: モンキーパッチで注入（Unsloth最適化突破）★★★
            model = replace_mlp_with_nkat(
                model,
                target_layers=config.so8_config['target_layers']
            )

            print(f"SO(8) Adapter with Quad Inference: {enable_quad}")
        else:
            print("[3/5] Skipping SO(8) adapters (baseline run)")

        # パラメータ確認
        print("[4/5] Checking trainable parameters...")
        model.print_trainable_parameters()

        # 🔥 緊急バイパス手術：Optimizerへの手動登録（SO8Tの場合のみ）
        if config.so8_config:
            print("[4.5/5] Manual optimizer registration for SO8T...")
            # 1. 学習対象パラメータの抽出 (LoRA と NKATアダプタ だけ)
            trainable_params = []
            for name, param in model.named_parameters():
                if "lora" in name.lower() or "nkat_adapter" in name.lower():
                    param.requires_grad = True
                    trainable_params.append(param)
                else:
                    param.requires_grad = False

            print(f"🔥 Total Trainable Params: {len(trainable_params)} tensors")

            # 2. Optimizerの手動作成 (Unsloth推奨の8bit AdamWを使う場合)
            try:
                from unsloth.optim import AdamW8bit
                optimizer = AdamW8bit(trainable_params, lr=config.training_config.get('learning_rate', 2e-5))
                print("✅ Using Unsloth AdamW8bit")
            except ImportError:
                from torch.optim import AdamW
                optimizer = AdamW(trainable_params, lr=config.training_config.get('learning_rate', 2e-5))
                print("✅ Using Standard AdamW")

            # 3. 後でTrainerに渡すための保存
            manual_optimizer = optimizer
        else:
            manual_optimizer = None

        # データセット準備
        print("[5/5] Preparing dataset...")
        dataset = SimpleDataset(
            config.sft_datasets[0],  # 最初のデータセットを使用
            tokenizer
        )
        print(f"Dataset size: {len(dataset)} samples")

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )

        # トレーニング引数
        training_args = TrainingArguments(**config.training_config)

        # カスタムコールバックでSO(8)メトリクス収集
        class SunshineCallback(TrainerCallback):
            def __init__(self, logger, model, run_type):
                self.logger = logger
                self.model = model
                self.run_type = run_type
                self.step_start_time = None

            def on_init_end(self, args, state, control, **kwargs):
                pass

            def on_train_begin(self, args, state, control, **kwargs):
                pass

            def on_train_end(self, args, state, control, **kwargs):
                pass

            def on_step_begin(self, args, state, control, **kwargs):
                self.step_start_time = time.time()

            def on_step_end(self, args, state, control, **kwargs):
                if self.step_start_time is None:
                    return

                step_time = time.time() - self.step_start_time
                step = state.global_step

                # SO(8)メトリクス収集
                so8_metrics = {}
                if self.run_type == "so8t":
                    ortho_errors = []
                    alphas = []
                    for name, module in self.model.named_modules():
                        if hasattr(module, 'get_adapter_stats'):
                            stats = module.get_adapter_stats()
                            ortho_errors.append(stats['orthogonality_error'])
                            alphas.append(stats['alpha'])

                    if ortho_errors:
                        so8_metrics['so8_ortho_mean'] = np.mean(ortho_errors)
                        so8_metrics['so8_ortho_max'] = np.max(ortho_errors)
                        so8_metrics['so8_alpha_mean'] = np.mean(alphas)
                        so8_metrics['so8_alpha_std'] = np.std(alphas) if len(alphas) > 1 else 0

                # 勾配ノルム（利用可能なら）
                grad_norm = None
                if hasattr(state, 'log_history') and state.log_history:
                    last_log = state.log_history[-1]
                    grad_norm = last_log.get('grad_norm')

                # ログ記録
                metrics = {
                    'train_loss': state.log_history[-1].get('loss') if state.log_history else None,
                    **so8_metrics,
                    'grad_norm': grad_norm
                }

                self.logger.log_step(step, metrics, step_time)

        callback = SunshineCallback(logger, model, run_type)

        # Trainer設定
        if manual_optimizer is not None:
            # SO8Tの場合：手動Optimizerを使用
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=dataset,
                data_collator=data_collator,
                callbacks=[callback, NKATDebugCallback(model)],
                optimizers=(manual_optimizer, None)  # (optimizer, scheduler)
            )
            print("🔧 Using manual optimizer for SO8T training")
        else:
            # Baselineの場合：通常のTrainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=dataset,
                data_collator=data_collator,
                callbacks=[callback, NKATDebugCallback(model)]
            )

        # トレーニング実行
        print(f"🚀 Starting {run_type.upper()} training...")
        trainer.train()

        # 最終メトリクス
        final_metrics = {}
        if trainer.state.log_history:
            final_metrics = trainer.state.log_history[-1]

        logger.finalize(final_metrics)

        print(f"✅ {run_type.upper()} training completed!")
        print(f"📊 Results saved to: {logger.log_dir}")

        return {
            'success': True,
            'run_type': run_type,
            'log_dir': str(logger.log_dir),
            'metrics_file': str(logger.metrics_file),
            'final_loss': final_metrics.get('loss')
        }

    except Exception as e:
        print(f"❌ {run_type.upper()} training failed: {e}")
        logger.finalize({'error': str(e)})
        return {
            'success': False,
            'run_type': run_type,
            'error': str(e)
        }


def run_sunshine_pipeline():
    """サンシャインパイプライン実行"""
    print("🌞🌞🌞 SO8T SUNSHINE PIPELINE 🌞🌞🌞")
    print("Comparing Baseline vs SO8T performance")
    print("=" * 60)

    results = {}

    # Run A: Baseline
    print("\n🏃 Run A: BASELINE (LoRA only)")
    results['baseline'] = run_sunshine_experiment("baseline")

    # Run B: SO8T
    print("\n🧬 Run B: SO8T (LoRA + SO(8) Adapter)")
    results['so8t'] = run_sunshine_experiment("so8t")

    # 結果比較
    print("\n📊 EXPERIMENT RESULTS SUMMARY")
    print("=" * 60)

    for run_type, result in results.items():
        status = "✅ SUCCESS" if result['success'] else "❌ FAILED"
        loss = f"Final Loss: {result.get('final_loss', 'N/A')}"
        print(f"{run_type.upper()}: {status} | {loss}")

    # ログファイル保存
    summary_file = Path("logs/sunshine") / "sunshine_summary.json"
    summary_file.parent.mkdir(parents=True, exist_ok=True)

    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'results': results
        }, f, indent=2, ensure_ascii=False)

    print(f"\n📋 Summary saved to: {summary_file}")

    # オーディオ通知
    try:
        import winsound
        winsound.Beep(1000, 1000)  # 1秒のビープ音
    except:
        pass

    return results


if __name__ == "__main__":
    # 引数処理
    if len(sys.argv) > 1:
        run_type = sys.argv[1]
        if run_type in ["baseline", "so8t"]:
            result = run_sunshine_experiment(run_type)
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print("Usage: python sunshine_pipeline.py [baseline|so8t]")
            sys.exit(1)
    else:
        # フルパイプライン実行
        run_sunshine_pipeline()

"""
SO8T Sunshine Pipeline - 実験計画ベースの自動実行システム

ボブにゃん戦略に基づくサンシャイン実行：
- Run A: Baseline (LoRAのみ)
- Run B: SO8T (LoRA + SO(8)アダプター)

ログフォーマット：
step, train_loss, eval_loss, so8_ortho_mean, so8_ortho_max, grad_norm, step_time_sec
"""

import os
import sys
import time
import json
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model

# Import from scripts directory
# from scripts.models.so8t_residual_adapter import attach_nkat_adapters  # 削除済み関数

# Simple dataset class for testing
from torch.utils.data import Dataset
import json

class SimpleDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, max_length: int = 128):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Load data
        self.data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        item = json.loads(line.strip())
                        self.data.append(item)
                    except json.JSONDecodeError:
                        continue

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item.get('text', '')

        # Tokenize
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
            'labels': tokenized['input_ids'].squeeze()
        }


class SunshineLogger:
    """サンシャイン実行の統一ログ収集"""

    def __init__(self, log_dir: str, run_name: str):
        self.log_dir = Path(log_dir)
        self.run_name = run_name
        self.log_file = self.log_dir / f"{run_name}_training_log.csv"
        self.metrics_file = self.log_dir / f"{run_name}_metrics.json"

        # CSVヘッダー
        self.columns = [
            'step', 'train_loss', 'eval_loss',
            'so8_ortho_mean', 'so8_ortho_max',
            'grad_norm', 'step_time_sec'
        ]

        # ログディレクトリ作成
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # CSV初期化
        if not self.log_file.exists():
            pd.DataFrame(columns=self.columns).to_csv(self.log_file, index=False)

        # メトリクス初期化
        self.metrics = {
            'start_time': datetime.now().isoformat(),
            'run_name': run_name,
            'total_steps': 0,
            'avg_step_time': 0,
            'final_train_loss': None,
            'final_eval_loss': None,
            'so8_ortho_errors': [],
            'grad_norms': []
        }

    def log_step(self, step: int, metrics: Dict[str, Any], step_time: float):
        """ステップごとのログ記録"""
        row = {
            'step': step,
            'train_loss': metrics.get('train_loss', None),
            'eval_loss': metrics.get('eval_loss', None),
            'so8_ortho_mean': metrics.get('so8_ortho_mean', None),
            'so8_ortho_max': metrics.get('so8_ortho_max', None),
            'grad_norm': metrics.get('grad_norm', None),
            'step_time_sec': step_time
        }

        # CSVに追加
        df = pd.DataFrame([row])
        df.to_csv(self.log_file, mode='a', header=False, index=False)

        # メトリクス更新
        if metrics.get('so8_ortho_mean') is not None:
            self.metrics['so8_ortho_errors'].append(metrics['so8_ortho_mean'])
        if metrics.get('grad_norm') is not None:
            self.metrics['grad_norms'].append(metrics['grad_norm'])

        self.metrics['total_steps'] = max(self.metrics['total_steps'], step)

    def finalize(self, final_metrics: Dict[str, Any]):
        """トレーニング完了時の最終ログ"""
        self.metrics.update({
            'end_time': datetime.now().isoformat(),
            'final_train_loss': final_metrics.get('train_loss'),
            'final_eval_loss': final_metrics.get('eval_loss'),
            'avg_so8_ortho_error': np.mean(self.metrics['so8_ortho_errors']) if self.metrics['so8_ortho_errors'] else None,
            'max_so8_ortho_error': np.max(self.metrics['so8_ortho_errors']) if self.metrics['so8_ortho_errors'] else None,
            'avg_grad_norm': np.mean(self.metrics['grad_norms']) if self.metrics['grad_norms'] else None
        })

        # JSON保存
        with open(self.metrics_file, 'w', encoding='utf-8') as f:
            json.dump(self.metrics, f, indent=2, ensure_ascii=False)


class SunshineTrainingConfig:
    """サンシャイン実行設定"""

    def __init__(self, run_type: str = "baseline"):
        self.model_name = "AXCXEPT/Borea-Phi-3.5-mini-Instruct-Jp"
        self.run_type = run_type  # "baseline" or "so8t"

        # データセット設定（統合データセットを使用）
        if run_type == "baseline":
            # Baseline用: 数学・科学統合データ
            self.sft_datasets = ["data/train_sft_enhanced.jsonl"]
        else:
            # SO8T用: NKAT理論・NSFWデータ統合
            self.sft_datasets = ["data/aegis_phi35_v2_with_nkat_so8t/aegis_phi35_v2_with_nkat_so8t_sft_train.jsonl"]

        # ドメイン重み付け
        self.domain_weights = {
            'mathematics': 1.2,
            'science': 1.1,
            'reasoning': 1.0,
            'general': 0.8
        }

        # トレーニング設定（本格データセット対応）
        self.training_config = {
            'output_dir': f"H:/from_D/webdataset/checkpoints/sunshine_{run_type}_phase25",
            'num_train_epochs': 1,
            'max_steps': 50,  # テスト用に減らす
            'per_device_train_batch_size': 1,
            'gradient_accumulation_steps': 8,  # RTX3060対応
            'learning_rate': 3e-5,  # 安定した学習率
            'save_steps': 100,
            'logging_steps': 10,  # 10ステップごとログ
            'eval_steps': 100,
            'gradient_checkpointing': False,  # 一時的にOFFにして勾配問題を解決
            'ddp_find_unused_parameters': False,  # アダプタパラメータの検出を確実にする
            'optim': "adamw_8bit",
            'bf16': True,
            'dataloader_pin_memory': False,
            'remove_unused_columns': False,
            'report_to': [],
            'load_best_model_at_end': False
        }

        # LoRA設定
        self.lora_config = {
            'r': 16,
            'lora_alpha': 32,
            'lora_dropout': 0.05,
            'bias': "none",
            'task_type': "CAUSAL_LM",
            'target_modules': ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        }

        # SO(8)設定（so8tの場合のみ）
        if run_type == "so8t":
            self.so8_config = {
                'target_layers': [8, 16, 24],  # 中間層のみ
                'so8_dim': 8,
                'alpha_init': 0.1
            }
        else:
            self.so8_config = None


def run_sunshine_experiment(run_type: str = "baseline") -> Dict[str, Any]:
    """
    サンシャイン実験実行
    run_type: "baseline" or "so8t"
    """
    print(f"🌞 Starting Sunshine Experiment: {run_type.upper()}")
    print("=" * 60)

    # 設定
    config = SunshineTrainingConfig(run_type)
    run_name = f"sunshine_run_{run_type}"

    # ロガー初期化
    logger = SunshineLogger("logs/sunshine", run_name)

    try:
        # モデルとトークナイザーロード
        print("[1/5] Loading model and tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # ★★★ トレーニング時は device_map="auto" を使用せず直接GPUにロード ★★★
        # device_map="auto" は分散トレーニングと競合するため
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float16,
            device_map=None  # トレーニング時は明示的にNone
        ).to("cuda")  # 直接GPUに移動

        # ★★★ デバッグ: Phi-3モデルの構造を確認 ★★★
        print(f"Model type: {type(model)}")
        print(f"Model attributes: {[attr for attr in dir(model) if not attr.startswith('_')]}")
        if hasattr(model, 'model'):
            print(f"model.model type: {type(model.model)}")
            print(f"model.model attributes: {[attr for attr in dir(model.model) if not attr.startswith('_')]}")
            if hasattr(model.model, 'layers'):
                print(f"model.model.layers length: {len(model.model.layers)}")
                print(f"First layer type: {type(model.model.layers[0])}")
                print(f"First layer attributes: {[attr for attr in dir(model.model.layers[0]) if not attr.startswith('_')]}")
                if hasattr(model.model.layers[0], 'mlp'):
                    print(f"First layer has mlp: {type(model.model.layers[0].mlp)}")

        # ★★★ 処方箋1: 入力の勾配を強制的に有効化 ★★★
        # これがないと、Gradient Checkpointing有効時に途中の勾配が死ぬ
        model.enable_input_require_grads()

        # LoRA適用
        print("[2/5] Applying LoRA...")
        lora_config = LoraConfig(**config.lora_config)
        model = get_peft_model(model, lora_config)

        # LoRAパラメータを明示的にトレーニング可能に（詳細確認）
        lora_params_count = 0
        for name, param in model.named_parameters():
            if 'lora' in name.lower():
                param.requires_grad_(True)
                lora_params_count += 1

        print(f"Set {lora_params_count} LoRA parameters to trainable")

        # SO(8)アダプター適用（so8tの場合）
        if config.so8_config:
            print("[3/5] Applying SO(8) adapters (Layer Replacement Mode)...")
            enable_quad = config.so8_config.get('enable_quad_inference', False)
            # ★★★ 最終奥義: モンキーパッチで注入（Unsloth最適化突破）★★★
            model = replace_mlp_with_nkat(
                model,
                target_layers=config.so8_config['target_layers']
            )

            print(f"SO(8) Adapter with Quad Inference: {enable_quad}")
        else:
            print("[3/5] Skipping SO(8) adapters (baseline run)")

        # パラメータ確認
        print("[4/5] Checking trainable parameters...")
        model.print_trainable_parameters()

        # 🔥 緊急バイパス手術：Optimizerへの手動登録（SO8Tの場合のみ）
        if config.so8_config:
            print("[4.5/5] Manual optimizer registration for SO8T...")
            # 1. 学習対象パラメータの抽出 (LoRA と NKATアダプタ だけ)
            trainable_params = []
            for name, param in model.named_parameters():
                if "lora" in name.lower() or "nkat_adapter" in name.lower():
                    param.requires_grad = True
                    trainable_params.append(param)
                else:
                    param.requires_grad = False

            print(f"🔥 Total Trainable Params: {len(trainable_params)} tensors")

            # 2. Optimizerの手動作成 (Unsloth推奨の8bit AdamWを使う場合)
            try:
                from unsloth.optim import AdamW8bit
                optimizer = AdamW8bit(trainable_params, lr=config.training_config.get('learning_rate', 2e-5))
                print("✅ Using Unsloth AdamW8bit")
            except ImportError:
                from torch.optim import AdamW
                optimizer = AdamW(trainable_params, lr=config.training_config.get('learning_rate', 2e-5))
                print("✅ Using Standard AdamW")

            # 3. 後でTrainerに渡すための保存
            manual_optimizer = optimizer
        else:
            manual_optimizer = None

        # データセット準備
        print("[5/5] Preparing dataset...")
        dataset = SimpleDataset(
            config.sft_datasets[0],  # 最初のデータセットを使用
            tokenizer
        )
        print(f"Dataset size: {len(dataset)} samples")

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )

        # トレーニング引数
        training_args = TrainingArguments(**config.training_config)

        # カスタムコールバックでSO(8)メトリクス収集
        class SunshineCallback(TrainerCallback):
            def __init__(self, logger, model, run_type):
                self.logger = logger
                self.model = model
                self.run_type = run_type
                self.step_start_time = None

            def on_init_end(self, args, state, control, **kwargs):
                pass

            def on_train_begin(self, args, state, control, **kwargs):
                pass

            def on_train_end(self, args, state, control, **kwargs):
                pass

            def on_step_begin(self, args, state, control, **kwargs):
                self.step_start_time = time.time()

            def on_step_end(self, args, state, control, **kwargs):
                if self.step_start_time is None:
                    return

                step_time = time.time() - self.step_start_time
                step = state.global_step

                # SO(8)メトリクス収集
                so8_metrics = {}
                if self.run_type == "so8t":
                    ortho_errors = []
                    alphas = []
                    for name, module in self.model.named_modules():
                        if hasattr(module, 'get_adapter_stats'):
                            stats = module.get_adapter_stats()
                            ortho_errors.append(stats['orthogonality_error'])
                            alphas.append(stats['alpha'])

                    if ortho_errors:
                        so8_metrics['so8_ortho_mean'] = np.mean(ortho_errors)
                        so8_metrics['so8_ortho_max'] = np.max(ortho_errors)
                        so8_metrics['so8_alpha_mean'] = np.mean(alphas)
                        so8_metrics['so8_alpha_std'] = np.std(alphas) if len(alphas) > 1 else 0

                # 勾配ノルム（利用可能なら）
                grad_norm = None
                if hasattr(state, 'log_history') and state.log_history:
                    last_log = state.log_history[-1]
                    grad_norm = last_log.get('grad_norm')

                # ログ記録
                metrics = {
                    'train_loss': state.log_history[-1].get('loss') if state.log_history else None,
                    **so8_metrics,
                    'grad_norm': grad_norm
                }

                self.logger.log_step(step, metrics, step_time)

        callback = SunshineCallback(logger, model, run_type)

        # Trainer設定
        if manual_optimizer is not None:
            # SO8Tの場合：手動Optimizerを使用
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=dataset,
                data_collator=data_collator,
                callbacks=[callback, NKATDebugCallback(model)],
                optimizers=(manual_optimizer, None)  # (optimizer, scheduler)
            )
            print("🔧 Using manual optimizer for SO8T training")
        else:
            # Baselineの場合：通常のTrainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=dataset,
                data_collator=data_collator,
                callbacks=[callback, NKATDebugCallback(model)]
            )

        # トレーニング実行
        print(f"🚀 Starting {run_type.upper()} training...")
        trainer.train()

        # 最終メトリクス
        final_metrics = {}
        if trainer.state.log_history:
            final_metrics = trainer.state.log_history[-1]

        logger.finalize(final_metrics)

        print(f"✅ {run_type.upper()} training completed!")
        print(f"📊 Results saved to: {logger.log_dir}")

        return {
            'success': True,
            'run_type': run_type,
            'log_dir': str(logger.log_dir),
            'metrics_file': str(logger.metrics_file),
            'final_loss': final_metrics.get('loss')
        }

    except Exception as e:
        print(f"❌ {run_type.upper()} training failed: {e}")
        logger.finalize({'error': str(e)})
        return {
            'success': False,
            'run_type': run_type,
            'error': str(e)
        }


def run_sunshine_pipeline():
    """サンシャインパイプライン実行"""
    print("🌞🌞🌞 SO8T SUNSHINE PIPELINE 🌞🌞🌞")
    print("Comparing Baseline vs SO8T performance")
    print("=" * 60)

    results = {}

    # Run A: Baseline
    print("\n🏃 Run A: BASELINE (LoRA only)")
    results['baseline'] = run_sunshine_experiment("baseline")

    # Run B: SO8T
    print("\n🧬 Run B: SO8T (LoRA + SO(8) Adapter)")
    results['so8t'] = run_sunshine_experiment("so8t")

    # 結果比較
    print("\n📊 EXPERIMENT RESULTS SUMMARY")
    print("=" * 60)

    for run_type, result in results.items():
        status = "✅ SUCCESS" if result['success'] else "❌ FAILED"
        loss = f"Final Loss: {result.get('final_loss', 'N/A')}"
        print(f"{run_type.upper()}: {status} | {loss}")

    # ログファイル保存
    summary_file = Path("logs/sunshine") / "sunshine_summary.json"
    summary_file.parent.mkdir(parents=True, exist_ok=True)

    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'results': results
        }, f, indent=2, ensure_ascii=False)

    print(f"\n📋 Summary saved to: {summary_file}")

    # オーディオ通知
    try:
        import winsound
        winsound.Beep(1000, 1000)  # 1秒のビープ音
    except:
        pass

    return results


if __name__ == "__main__":
    # 引数処理
    if len(sys.argv) > 1:
        run_type = sys.argv[1]
        if run_type in ["baseline", "so8t"]:
            result = run_sunshine_experiment(run_type)
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print("Usage: python sunshine_pipeline.py [baseline|so8t]")
            sys.exit(1)
    else:
        # フルパイプライン実行
        run_sunshine_pipeline()

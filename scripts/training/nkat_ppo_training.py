#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NKAT PPO Training Script
SO(8)四重推論を獲得するためのPPOトレーニング

Unsloth + TRL を用いた borea-phi3.5-instinct-jp のファインチューニング
RTX 3060 (12GB VRAM) 最適化設定
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from unsloth import FastLanguageModel
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import json
import math
from pathlib import Path
import argparse
from datetime import datetime

# ローカルモジュール
from nkat_quad_inference_prompt import get_nkat_prompt, NKAT_SYSTEM_PROMPT
from nkat_reward_function import create_nkat_reward_function, NKATRewardFunction

class NKATDataset(Dataset):
    """NKATトレーニング用データセット"""

    def __init__(self, tokenizer, max_length: int = 1024, num_samples: int = 1000):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # トレーニングサンプル生成
        self.samples = self._generate_training_samples(num_samples)

    def _generate_training_samples(self, num_samples: int) -> List[Dict[str, str]]:
        """トレーニングサンプルを生成"""
        samples = []

        # 数学・物理・生物学の問題テンプレート
        templates = [
            "以下の微分方程式を解け: dy/dx = {a}*y + {b}*sin(x)",
            "量子力学で波動関数ψが満たすべき条件は何か？",
            "DNAの二重螺旋構造の安定性は何によって保たれているか？",
            "相対性理論における光速不変の原理を説明せよ。",
            "群論におけるSO(8)群の表現について説明せよ。",
            "圏論における函手の定義と例を挙げよ。",
            "スペクトル幾何学におけるラプラシアン作用素の役割は？",
            "位相幾何学におけるホモロジー群の意味は何か？",
            "Solve the differential equation: d²y/dx² + {a}*y = 0",
            "What is the significance of Gödel's incompleteness theorems?",
            "Explain the concept of genetic algorithms in evolutionary computation.",
            "Derive the Schwarzschild radius for a black hole.",
        ]

        for i in range(num_samples):
            # ランダムなパラメータ
            a = np.random.randint(1, 10)
            b = np.random.randint(1, 5)

            # テンプレートから問題を生成
            template = np.random.choice(templates)
            if "{a}" in template and "{b}" in template:
                query = template.format(a=a, b=b)
            elif "{a}" in template:
                query = template.format(a=a)
            else:
                query = template

            # NKATプロンプトを適用
            prompt = get_nkat_prompt("training").format(query=query)

            samples.append({
                "query": query,
                "prompt": prompt,
                "full_text": prompt
            })

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # トークナイズ
        inputs = self.tokenizer(
            sample["full_text"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "query": sample["query"],
            "prompt": sample["prompt"]
        }

def create_model_and_tokenizer(model_name: str = "microsoft/phi-3.5-mini-instruct"):
    """Unslothでモデルとトークナイザーを作成"""
    print("Loading model with Unsloth 4-bit quantization...")

    # 4-bit量子化でモデルをロード
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=2048,
        dtype=None,  # Auto detection
        load_in_4bit=True,  # 4-bit quantization
        token=None,  # 必要に応じて設定
    )

    # LoRA設定（RTX 3060最適化）
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,  # LoRA rank (16 or 32 for RTX 3060)
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],  # All linear layers
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing=True,  # VRAM節約
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    print(f"Model loaded: {model_name}")
    print(f"Trainable parameters: {model.num_parameters(only_trainable=True):,}")
    print(f"All parameters: {model.num_parameters():,}")

    return model, tokenizer

def setup_ppo_trainer(model, tokenizer, reward_function: NKATRewardFunction):
    """PPOトレーナーを設定"""

    # PPO設定（RTX 3060最適化）
    ppo_config = PPOConfig(
        model_name="microsoft/phi-3.5-mini-instruct",
        learning_rate=1.41e-5,  # 最適化された学習率
        batch_size=1,  # RTX 3060用に最小バッチサイズ
        mini_batch_size=1,
        gradient_accumulation_steps=8,  # 実質的なバッチサイズを稼ぐ
        optimize_cuda_cache=True,
        seed=0,
        use_score_scaling=False,
        use_score_norm=False,
        score_clip=None,
    )

    # PPOモデル（Value Head付き）
    ppo_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        model,
        torch_dtype=torch.float16,
        device_map="auto",
        peft_config=model.peft_config if hasattr(model, 'peft_config') else None,
    )

    # PPOトレーナー
    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=ppo_model,
        ref_model=None,  # 参照モデルなし（メモリ節約）
        tokenizer=tokenizer,
        dataset=None,  # カスタムデータセットを使用
        data_collator=None,
    )

    return ppo_trainer, ppo_model

def generate_response(model, tokenizer, query: str, max_new_tokens: int = 512) -> str:
    """モデルから応答を生成"""
    prompt = get_nkat_prompt("training").format(query=query)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # プロンプト部分を除去
    response = response[len(prompt):].strip()

    return response

def train_nkat_ppo(
    model_name: str = "microsoft/phi-3.5-mini-instruct",
    num_epochs: int = 3,
    num_samples_per_epoch: int = 100,
    output_dir: str = "outputs/nkat_ppo_training",
    save_steps: int = 50,
    eval_steps: int = 25,
    enable_multimodal: bool = False,  # Phase 1: Text-Only, Phase 2: Multimodal
):
    """NKAT PPOトレーニングを実行"""

    print("=== NKAT PPO Training Started ===")
    print(f"Model: {model_name}")
    print(f"Hardware: RTX 3060 (12GB VRAM) optimized")
    print(f"Training samples per epoch: {num_samples_per_epoch}")
    print("=" * 50)

    # 出力ディレクトリ作成
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # モデルとトークナイザー作成
    model, tokenizer = create_model_and_tokenizer(model_name)

    # 報酬関数作成
    reward_function = create_nkat_reward_function(tokenizer, model)

    # PPOトレーナー設定
    ppo_trainer, ppo_model = setup_ppo_trainer(model, tokenizer, reward_function)

    # データセット作成
    dataset = NKATDataset(tokenizer, max_length=1024, num_samples=num_samples_per_epoch)

    # トレーニング統計
    stats = {
        "epoch": [],
        "step": [],
        "reward": [],
        "loss": [],
        "learning_rate": []
    }

    total_steps = 0

    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")

        epoch_rewards = []
        epoch_losses = []

        for step, batch in enumerate(dataset):
            query = batch["query"]
            prompt = batch["prompt"]

            # 応答生成
            response = generate_response(ppo_model, tokenizer, query)

            # 報酬計算
            reward = reward_function([prompt], [response])[0]

            # PPOステップ実行
            # クエリをトークナイズ
            query_tensor = tokenizer(query, return_tensors="pt").to(ppo_model.device)

            # 応答をトークナイズ
            response_tensor = tokenizer(response, return_tensors="pt").to(ppo_model.device)

            # PPOトレーニングステップ
            train_stats = ppo_trainer.step(
                [query_tensor["input_ids"]],
                [response_tensor["input_ids"]],
                [torch.tensor([reward], dtype=torch.float32)]
            )

            # 統計記録
            epoch_rewards.append(reward)
            if 'ppo/loss/total' in train_stats:
                epoch_losses.append(train_stats['ppo/loss/total'])

            total_steps += 1

            # ログ出力
            if (step + 1) % 10 == 0:
                avg_reward = np.mean(epoch_rewards[-10:])
                avg_loss = np.mean(epoch_losses[-10:]) if epoch_losses else 0
                print(f"Step {step+1}/{len(dataset)} | Reward: {avg_reward:.3f} | Loss: {avg_loss:.3f}")

            # モデル保存
            if total_steps % save_steps == 0:
                save_path = output_path / f"checkpoint_step_{total_steps}"
                save_path.mkdir(exist_ok=True)

                # LoRA重みを保存
                model.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)

                print(f"Checkpoint saved: {save_path}")

            # 評価
            if total_steps % eval_steps == 0:
                eval_reward = evaluate_model(ppo_model, tokenizer, reward_function)
                print(f"Evaluation reward: {eval_reward:.3f}")

        # エポック統計
        epoch_avg_reward = np.mean(epoch_rewards)
        epoch_avg_loss = np.mean(epoch_losses) if epoch_losses else 0

        stats["epoch"].append(epoch + 1)
        stats["step"].append(total_steps)
        stats["reward"].append(epoch_avg_reward)
        stats["loss"].append(epoch_avg_loss)
        stats["learning_rate"].append(ppo_trainer.optimizer.param_groups[0]['lr'])

        print(f"Epoch {epoch + 1} completed | Avg Reward: {epoch_avg_reward:.3f} | Avg Loss: {epoch_avg_loss:.3f}")

    # 最終モデル保存
    final_save_path = output_path / "final_model"
    final_save_path.mkdir(exist_ok=True)
    model.save_pretrained(final_save_path)
    tokenizer.save_pretrained(final_save_path)

    # 統計保存
    with open(output_path / "training_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print("Training completed!")
    print(f"Final model saved to: {final_save_path}")

    return model, tokenizer

def evaluate_model(model, tokenizer, reward_function: NKATRewardFunction, num_eval_samples: int = 10) -> float:
    """モデルを評価"""
    eval_queries = [
        "微分方程式 dy/dx = 2y を解け。",
        "SO(8)群の次元は何ですか？",
        "圏論における終対象とは何ですか？",
        "ブラックホールの事象の地平とは何ですか？",
        "DNA複製の仕組みを説明せよ。",
    ]

    total_reward = 0

    for query in eval_queries[:num_eval_samples]:
        response = generate_response(model, tokenizer, query)
        reward = reward_function([query], [response])[0]
        total_reward += reward

    return total_reward / num_eval_samples

def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="NKAT PPO Training")
    parser.add_argument("--model_name", type=str, default="microsoft/phi-3.5-mini-instruct",
                       help="Base model name")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--num_samples_per_epoch", type=int, default=100,
                       help="Number of samples per epoch")
    parser.add_argument("--output_dir", type=str, default="outputs/nkat_ppo_training",
                       help="Output directory")
    parser.add_argument("--save_steps", type=int, default=50, help="Save model every N steps")
    parser.add_argument("--eval_steps", type=int, default=25, help="Evaluate every N steps")

    args = parser.parse_args()

    # 環境設定
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # RTX 3060を使用
    torch.cuda.empty_cache()

    # トレーニング実行
    train_nkat_ppo(
        model_name=args.model_name,
        num_epochs=args.num_epochs,
        num_samples_per_epoch=args.num_samples_per_epoch,
        output_dir=args.output_dir,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
    )

if __name__ == "__main__":
    main()

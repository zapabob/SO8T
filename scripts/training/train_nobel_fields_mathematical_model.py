#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train Nobel Fields Mathematical Model

ノーベル賞・フィールズ賞級の数学・科学推論を可能にするモデルのトレーニング
Arxiv引用回数トップ論文に基づくデータセットを使用
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass
from datetime import datetime
import warnings
import gc
from tqdm import tqdm

# インポート
from .enhanced_so8_mathematical_adapter import (
    create_unified_mathematical_model,
    create_enhanced_so8_config
)
from .advanced_mathematical_reasoning import (
    AdvancedMathematicalReasoningEngine,
    create_advanced_reasoning_config
)


@dataclass
class NobelFieldsTrainingConfig:
    """ノーベル・フィールズ賞トレーニング設定"""
    # モデル設定
    model_name: str = "AEGIS-phi3.5-thinking-v2.0-nobel-fields"
    base_model_path: str = "microsoft/phi-3.5-mini-instruct"  # または HODACHI-Borea
    hidden_size: int = 3072

    # トレーニング設定
    batch_size: int = 1  # GRPOのため小バッチ
    max_seq_length: int = 2048
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_steps: int = 10000
    save_steps: int = 500
    eval_steps: int = 250
    logging_steps: int = 50

    # 数学的推論設定
    enable_mathematical_reasoning: bool = True
    reasoning_format: str = "nobel_fields"
    mathematical_domains: List[str] = None

    # データセット設定
    dataset_path: str = "data/nobel_fields_mathematical_dataset"
    arxiv_citation_threshold: int = 100  # 引用回数閾値
    mathematical_problem_ratio: float = 0.7
    scientific_reasoning_ratio: float = 0.3

    # 最適化設定
    use_unsloth: bool = True
    use_peft: bool = True
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.05

    # ハードウェア設定
    device: str = "auto"
    dtype: str = "bf16"
    gradient_checkpointing: bool = True
    max_grad_norm: float = 1.0

    def __post_init__(self):
        if self.mathematical_domains is None:
            self.mathematical_domains = ["quantum_field", "statistical", "proof", "unified"]


class NobelFieldsMathematicalDataset(Dataset):
    """
    Nobel Fields Mathematical Dataset

    Arxiv引用回数トップ論文に基づく数学・科学データセット
    """

    def __init__(self, config: NobelFieldsTrainingConfig):
        self.config = config

        # データセットの読み込み
        self.data = self._load_arxiv_mathematical_data()

        # トークナイザー（トレーニング時に設定）
        self.tokenizer = None

    def _load_arxiv_mathematical_data(self) -> List[Dict[str, Any]]:
        """Arxiv数学・科学データの読み込み"""
        dataset_path = Path(self.config.dataset_path)

        if not dataset_path.exists():
            print(f"データセットが見つからないため、生成します: {dataset_path}")
            return self._generate_arxiv_mathematical_data()

        # 既存データセットの読み込み
        data_files = list(dataset_path.glob("*.jsonl"))

        all_data = []
        for file_path in data_files:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        item = json.loads(line.strip())
                        # 引用回数フィルタリング
                        if item.get('citations', 0) >= self.config.arxiv_citation_threshold:
                            all_data.append(item)
                    except json.JSONDecodeError:
                        continue

        print(f"読み込んだデータ数: {len(all_data)}")
        return all_data

    def _generate_arxiv_mathematical_data(self) -> List[Dict[str, Any]]:
        """Arxiv数学・科学データの生成（ダミー実装）"""
        # 実際の実装ではArxiv APIや論文データベースから取得

        # ノーベル賞・フィールズ賞関連論文のサンプル
        sample_papers = [
            {
                "title": "Quantum Field Theory and the Standard Model",
                "abstract": "A comprehensive review of quantum field theory...",
                "citations": 2500,
                "category": "hep-th",
                "mathematical_problem": "Solve the Yang-Mills equations for SU(3) gauge theory",
                "reasoning_type": "quantum_field"
            },
            {
                "title": "Statistical Mechanics of Phase Transitions",
                "abstract": "Critical phenomena and renormalization group theory...",
                "citations": 1800,
                "category": "cond-mat",
                "mathematical_problem": "Compute the critical exponents for the Ising model",
                "reasoning_type": "statistical"
            },
            {
                "title": "Proof of the Riemann Hypothesis",
                "abstract": "Advances in analytic number theory...",
                "citations": 1200,
                "category": "math.NT",
                "mathematical_problem": "Prove the Riemann hypothesis using advanced analytic methods",
                "reasoning_type": "proof"
            }
        ]

        # データセット保存
        dataset_path = Path(self.config.dataset_path)
        dataset_path.mkdir(parents=True, exist_ok=True)

        all_data = []
        for i, paper in enumerate(sample_papers):
            # 各論文から複数のトレーニングサンプル生成
            for j in range(10):  # 各論文から10個のサンプル
                sample = {
                    "id": f"{paper['category']}_{i}_{j}",
                    "title": paper["title"],
                    "abstract": paper["abstract"],
                    "citations": paper["citations"],
                    "category": paper["category"],
                    "mathematical_problem": paper["mathematical_problem"],
                    "reasoning_type": paper["reasoning_type"],
                    "difficulty": np.random.choice(["basic", "intermediate", "advanced"]),
                    "domain": np.random.choice(["mathematics", "physics", "computer_science"])
                }
                all_data.append(sample)

        # JSONL形式で保存
        output_file = dataset_path / "arxiv_mathematical_dataset.jsonl"
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in all_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        print(f"生成したデータ数: {len(all_data)}")
        return all_data

    def set_tokenizer(self, tokenizer):
        """トークナイザーの設定"""
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # 数学的問題のフォーマット
        problem_text = self._format_mathematical_problem(item)

        # 解答の生成（教師データ）
        solution_text = self._generate_solution(item)

        return {
            "problem": problem_text,
            "solution": solution_text,
            "metadata": item
        }

    def _format_mathematical_problem(self, item: Dict[str, Any]) -> str:
        """数学的問題のフォーマット"""
        domain = item.get("domain", "mathematics")
        difficulty = item.get("difficulty", "intermediate")
        reasoning_type = item.get("reasoning_type", "proof")

        template = f"""以下の{domain}の問題を{reasoning_type}により解け。

問題: {item['mathematical_problem']}

難易度: {difficulty}
引用論文: {item['title']} (引用回数: {item['citations']})

ステップバイステップで思考し、解を導け。
"""

        return template

    def _generate_solution(self, item: Dict[str, Any]) -> str:
        """解答の生成（教師データ用）"""
        reasoning_type = item.get("reasoning_type", "proof")

        if reasoning_type == "quantum_field":
            solution = self._generate_qft_solution(item)
        elif reasoning_type == "statistical":
            solution = self._generate_statistical_solution(item)
        elif reasoning_type == "proof":
            solution = self._generate_proof_solution(item)
        else:
            solution = self._generate_general_solution(item)

        return solution

    def _generate_qft_solution(self, item: Dict[str, Any]) -> str:
        """量子場論問題の解答生成"""
        return """<think>
観察: 量子場論の問題を分析する。Yang-Mills理論の古典解を求める。
演繹: 作用の変分原理よりEuler-Lagrange方程式を導く。
帰納: ゲージ対称性と拘束条件を考慮した解の構造を仮定。
統合: 自己双対解としてBPST instanton解を得る。
</think>

<final>
Yang-Mills方程式の解は、BPST instantonとして与えられる：
A_μ = -i g^{-1} U ∂_μ U^†
ここでUはinstantonの位置とスケールパラメータに依存する。
</final>"""

    def _generate_statistical_solution(self, item: Dict[str, Any]) -> str:
        """統計力学問題の解答生成"""
        return """<think>
観察: 臨界現象の問題を分析。Ising模型の相転移を考える。
演繹: 分配関数を計算し、自由エネルギーを求める。
帰納: 平均場近似からくりこみ群理論へ進む。
統合: 2次元Ising模型の厳密解から臨界指数を決定。
</think>

<final>
2次元Ising模型の臨界温度はT_c = 2.269J/k_Bで、
臨界指数はβ=1/8, γ=7/4, ν=1である。
</final>"""

    def _generate_proof_solution(self, item: Dict[str, Any]) -> str:
        """証明問題の解答生成"""
        return """<think>
観察: 数論的問題を分析。Riemannゼータ関数の性質を考える。
演繹: 関数方程式と解析接続を利用。
帰納: 非自明零点の分布に関する仮説を立てる。
統合: ゼータ関数の零点分布がRiemann予想を満たすことを示す。
</think>

<final>
Riemann予想は、ζ(s)=0となるsの虚部が0であること以外は、
全て実部が1/2であることを主張する。
これは解析接続と関数方程式から導かれる。
</final>"""

    def _generate_general_solution(self, item: Dict[str, Any]) -> str:
        """一般問題の解答生成"""
        return """<think>
観察: 一般的な数学的問題を分析。
演繹: 既知の定理と公式を適用。
帰納: パターン認識により一般化。
統合: 包括的な解答を構成。
</think>

<final>
問題の解答は与えられた条件と定理から導かれる。
</final>"""


def create_mathematical_collate_fn(tokenizer):
    """数学的データセット用のcollate関数"""
    def collate_fn(batch):
        problems = [item["problem"] for item in batch]
        solutions = [item["solution"] for item in batch]
        metadata = [item["metadata"] for item in batch]

        # トークナイズ
        problem_encodings = tokenizer(
            problems,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        )

        solution_encodings = tokenizer(
            solutions,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        )

        return {
            "problem_input_ids": problem_encodings["input_ids"],
            "problem_attention_mask": problem_encodings["attention_mask"],
            "solution_input_ids": solution_encodings["input_ids"],
            "solution_attention_mask": solution_encodings["attention_mask"],
            "metadata": metadata
        }

    return collate_fn


class NobelFieldsTrainer:
    """
    Nobel Fields Mathematical Model Trainer

    ノーベル賞・フィールズ賞級推論モデルのトレーニング
    """

    def __init__(self, config: NobelFieldsTrainingConfig):
        self.config = config

        # モデルとトークナイザーの初期化
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self.scheduler = None

        # データセット
        self.train_dataset = None
        self.eval_dataset = None

        # トレーニング状態
        self.global_step = 0
        self.best_eval_loss = float('inf')

        # 出力ディレクトリ
        self.output_dir = Path(f"outputs/{config.model_name}")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def setup_model_and_tokenizer(self):
        """モデルとトークナイザーのセットアップ"""
        print("モデルとトークナイザーのセットアップ中...")

        try:
            from unsloth import FastLanguageModel

            # Unslothでモデル読み込み
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.config.base_model_path,
                max_seq_length=self.config.max_seq_length,
                dtype=torch.bfloat16 if self.config.dtype == "bf16" else torch.float16,
                load_in_4bit=False,  # トレーニング時はフル精度
            )

            # LoRA適用
            self.model = FastLanguageModel.get_peft_model(
                self.model,
                r=self.config.lora_r,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                bias="none",
            )

            # Unified Mathematical Adapterの適用
            from .enhanced_so8_mathematical_adapter import create_unified_mathematical_model
            self.model = create_unified_mathematical_model(self.model)

            print("Unsloth + Unified Mathematical Model セットアップ完了")

        except ImportError:
            print("Unslothが利用できないため、標準Transformersを使用")
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.base_model_path,
                torch_dtype=torch.bfloat16 if self.config.dtype == "bf16" else torch.float16,
                device_map="auto"
            )

            self.tokenizer = AutoTokenizer.from_pretrained(self.config.base_model_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Unified Mathematical Adapterの適用
            from .enhanced_so8_mathematical_adapter import create_unified_mathematical_model
            self.model = create_unified_mathematical_model(self.model)

        # デバイス設定
        if self.config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config.device)

        self.model.to(self.device)

        # 勾配チェックポインティング
        if self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

    def setup_optimizer_and_scheduler(self):
        """オプティマイザーとスケジューラーのセットアップ"""
        print("オプティマイザーとスケジューラーのセットアップ中...")

        # オプティマイザー
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        # スケジューラー
        from transformers import get_linear_schedule_with_warmup

        # ステップ数の推定
        total_steps = self.config.max_steps

        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=total_steps
        )

    def setup_datasets(self):
        """データセットのセットアップ"""
        print("データセットのセットアップ中...")

        # トレーニングデータセット
        self.train_dataset = NobelFieldsMathematicalDataset(self.config)
        self.train_dataset.set_tokenizer(self.tokenizer)

        # 評価データセット（トレーニングデータセットのサブセット）
        eval_size = min(100, len(self.train_dataset) // 10)
        self.eval_dataset = torch.utils.data.Subset(
            self.train_dataset,
            torch.randperm(len(self.train_dataset))[:eval_size]
        )

        print(f"トレーニングデータ数: {len(self.train_dataset)}")
        print(f"評価データ数: {len(self.eval_dataset)}")

    def compute_mathematical_loss(self, outputs, targets, metadata):
        """数学的推論特化の損失計算"""
        # 基本的な言語モデル損失
        logits = outputs['logits']
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = targets[..., 1:].contiguous()

        # 標準言語モデル損失
        lm_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=self.tokenizer.pad_token_id
        )

        # 数学的推論の品質に基づく追加損失
        mathematical_penalty = 0.0

        if self.config.enable_mathematical_reasoning:
            # 数学的確信度の考慮
            confidence = outputs.get('mathematical_analysis', {}).get('mathematical_confidence', 0.5)
            mathematical_penalty = -torch.log(torch.tensor(confidence + 1e-8)) * 0.1

        # メタデータの品質に基づく重み付け
        citation_weight = torch.tensor([
            min(1.0, item.get('citations', 0) / 1000.0) for item in metadata
        ], device=self.device).mean()

        total_loss = lm_loss + mathematical_penalty
        total_loss = total_loss * (1.0 + citation_weight * 0.1)  # 高引用論文を重視

        return total_loss, {
            'lm_loss': lm_loss.item(),
            'mathematical_penalty': mathematical_penalty.item() if isinstance(mathematical_penalty, torch.Tensor) else mathematical_penalty,
            'citation_weight': citation_weight.item(),
            'total_loss': total_loss.item()
        }

    def train_step(self, batch):
        """1ステップのトレーニング"""
        self.model.train()

        # 入力をデバイスに移動
        problem_input_ids = batch['problem_input_ids'].to(self.device)
        problem_attention_mask = batch['problem_attention_mask'].to(self.device)
        solution_input_ids = batch['solution_input_ids'].to(self.device)

        # 問題文と解答を結合
        input_ids = torch.cat([problem_input_ids, solution_input_ids], dim=1)
        attention_mask = torch.cat([
            problem_attention_mask,
            torch.ones_like(solution_input_ids)
        ], dim=1)

        # ラベル作成（次トークン予測）
        labels = input_ids.clone()
        labels[:, :problem_input_ids.shape[1]] = -100  # 問題部分は無視

        # フォワードパス
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            enable_mathematical_reasoning=self.config.enable_mathematical_reasoning,
            reasoning_format=self.config.reasoning_format
        )

        # 損失計算
        loss, loss_components = self.compute_mathematical_loss(outputs, labels, batch['metadata'])

        # バックワードパス
        self.optimizer.zero_grad()
        loss.backward()

        # 勾配クリッピング
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)

        self.optimizer.step()
        self.scheduler.step()

        return loss_components

    def evaluate(self):
        """評価実行"""
        self.model.eval()
        total_loss = 0.0
        total_samples = 0

        eval_dataloader = DataLoader(
            self.eval_dataset,
            batch_size=self.config.batch_size,
            collate_fn=create_mathematical_collate_fn(self.tokenizer),
            shuffle=False
        )

        with torch.no_grad():
            for batch in eval_dataloader:
                # 入力をデバイスに移動
                problem_input_ids = batch['problem_input_ids'].to(self.device)
                problem_attention_mask = batch['problem_attention_mask'].to(self.device)
                solution_input_ids = batch['solution_input_ids'].to(self.device)

                # 問題文と解答を結合
                input_ids = torch.cat([problem_input_ids, solution_input_ids], dim=1)
                attention_mask = torch.cat([
                    problem_attention_mask,
                    torch.ones_like(solution_input_ids)
                ], dim=1)

                # ラベル作成
                labels = input_ids.clone()
                labels[:, :problem_input_ids.shape[1]] = -100

                # フォワードパス
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    enable_mathematical_reasoning=self.config.enable_mathematical_reasoning,
                    reasoning_format=self.config.reasoning_format
                )

                # 損失計算
                loss, _ = self.compute_mathematical_loss(outputs, labels, batch['metadata'])

                total_loss += loss.item() * len(batch['metadata'])
                total_samples += len(batch['metadata'])

        avg_loss = total_loss / total_samples
        return avg_loss

    def save_checkpoint(self, step: int, loss: float):
        """チェックポイント保存"""
        checkpoint_dir = self.output_dir / f"checkpoint-{step}"
        checkpoint_dir.mkdir(exist_ok=True)

        # モデル保存
        self.model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)

        # オプティマイザーとスケジューラー保存
        torch.save({
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'step': step,
            'loss': loss,
            'config': self.config
        }, checkpoint_dir / "optimizer.pt")

        print(f"チェックポイント保存: {checkpoint_dir}")

    def train(self):
        """トレーニング実行"""
        print("トレーニング開始...")
        print(f"モデル: {self.config.model_name}")
        print(f"出力ディレクトリ: {self.output_dir}")

        # データローダー設定
        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            collate_fn=create_mathematical_collate_fn(self.tokenizer),
            shuffle=True
        )

        # トレーニングループ
        self.global_step = 0

        while self.global_step < self.config.max_steps:
            for batch in train_dataloader:
                if self.global_step >= self.config.max_steps:
                    break

                # トレーニングステップ
                loss_components = self.train_step(batch)
                self.global_step += 1

                # ログ出力
                if self.global_step % self.config.logging_steps == 0:
                    print(f"Step {self.global_step}: {loss_components}")

                # 評価
                if self.global_step % self.config.eval_steps == 0:
                    eval_loss = self.evaluate()
                    print(f"Step {self.global_step} Eval Loss: {eval_loss:.4f}")

                    # ベストモデル保存
                    if eval_loss < self.best_eval_loss:
                        self.best_eval_loss = eval_loss
                        self.save_checkpoint(self.global_step, eval_loss)

                # チェックポイント保存
                if self.global_step % self.config.save_steps == 0:
                    self.save_checkpoint(self.global_step, loss_components['total_loss'])

        # 最終モデル保存
        final_dir = self.output_dir / "final_model"
        final_dir.mkdir(exist_ok=True)
        self.model.save_pretrained(final_dir)
        self.tokenizer.save_pretrained(final_dir)

        print(f"トレーニング完了。最終モデル保存: {final_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train Nobel Fields Mathematical Model")
    parser.add_argument("--model_name", type=str, default="AEGIS-phi3.5-thinking-v2.0-nobel-fields")
    parser.add_argument("--base_model", type=str, default="microsoft/phi-3.5-mini-instruct")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--enable_mathematical_reasoning", action="store_true", default=True)
    parser.add_argument("--reasoning_format", type=str, default="nobel_fields", choices=["standard", "nobel_fields"])

    args = parser.parse_args()

    # 設定作成
    config = NobelFieldsTrainingConfig(
        model_name=args.model_name,
        base_model_path=args.base_model,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        enable_mathematical_reasoning=args.enable_mathematical_reasoning,
        reasoning_format=args.reasoning_format
    )

    # トレーナー作成
    trainer = NobelFieldsTrainer(config)

    # セットアップ
    trainer.setup_model_and_tokenizer()
    trainer.setup_optimizer_and_scheduler()
    trainer.setup_datasets()

    # トレーニング実行
    trainer.train()


if __name__ == "__main__":
    main()

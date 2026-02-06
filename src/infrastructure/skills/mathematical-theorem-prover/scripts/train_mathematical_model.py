#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数学的定理証明能力強化訓練スクリプト
SFT + GRPO訓練戦略の実装
"""

import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer, GRPOTrainer
from peft import LoraConfig, get_peft_model
import logging
import argparse
from dataclasses import dataclass
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class MathematicalTrainingData:
    """構造化された数学的訓練データ"""
    theorem_id: str
    domain: str  # algebra, geometry, analysis, etc.
    difficulty: int  # 1-5

    natural_language: str
    formal_statement: str  # Lean4/Isabelle形式
    symbolic_representation: str  # LaTeX/MathJax

    informal_proof: str
    formal_proof: str
    proof_steps: List[str]

    lemmas_used: List[str]
    prerequisites: List[str]
    verification_status: bool

    alternative_proofs: List[str]
    counterexamples: List[str]
    related_theorems: List[str]

    scientific_context: str
    experimental_validation: Optional[str] = None

class MathematicalTheoremProver:
    """
    数学的定理証明能力強化システム
    """

    def __init__(self, base_model_path: str, output_dir: str = "mathematical_training_output"):
        self.base_model_path = base_model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # モデル読み込み
        logger.info(f"Loading base model: {base_model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        # LoRA設定
        self.lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )

        # GRPO設定
        self.grpo_config = {
            "group_size": 8,
            "kl_penalty": 0.1,
            "clip_ratio": 0.2,
            "reward_functions": ["formal_proof_correctness", "proof_completeness", "mathematical_novelty", "proof_efficiency"]
        }

    def execute_full_training_pipeline(self, math_data_path: str) -> Dict[str, Any]:
        """完全な数学的訓練パイプライン実行"""
        logger.info("Starting mathematical theorem proving training pipeline")

        results = {
            "pipeline_start": datetime.now().isoformat(),
            "phases": []
        }

        try:
            # Phase 1: Mathematical Foundation SFT
            logger.info("Phase 1: Mathematical Foundation SFT")
            sft_model_path = self._execute_mathematical_sft(math_data_path)
            results["phases"].append({
                "name": "mathematical_sft",
                "model_path": str(sft_model_path),
                "status": "completed"
            })

            # Phase 2: GRPO Reasoning Enhancement
            logger.info("Phase 2: GRPO Reasoning Enhancement")
            grpo_model_path = self._execute_grpo_training(sft_model_path, math_data_path)
            results["phases"].append({
                "name": "grpo_reasoning",
                "model_path": str(grpo_model_path),
                "status": "completed"
            })

            # Phase 3: Integration and Fine-tuning
            logger.info("Phase 3: Integration and Fine-tuning")
            final_model_path = self._execute_integration_training(grpo_model_path, math_data_path)
            results["phases"].append({
                "name": "integration_training",
                "model_path": str(final_model_path),
                "status": "completed"
            })

            results["status"] = "completed"
            results["final_model_path"] = str(final_model_path)
            results["pipeline_end"] = datetime.now().isoformat()

        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            results["status"] = "failed"
            results["error"] = str(e)
            results["pipeline_end"] = datetime.now().isoformat()

        # 結果保存
        results_path = self.output_dir / "training_results.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        return results

    def _execute_mathematical_sft(self, math_data_path: str) -> Path:
        """数学的基礎SFT実行"""
        logger.info("Executing mathematical foundation SFT")

        # データ読み込み
        math_data = self._load_mathematical_data(math_data_path)

        # SFTデータ準備
        sft_data = self._prepare_sft_data(math_data)

        # LoRA適用
        model = get_peft_model(self.model, self.lora_config)

        # 訓練設定
        training_args = TrainingArguments(
            output_dir=str(self.output_dir / "sft_checkpoints"),
            num_train_epochs=3,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=8,
            learning_rate=2e-5,
            weight_decay=0.01,
            warmup_ratio=0.03,
            lr_scheduler_type="cosine",
            logging_steps=10,
            save_steps=500,
            eval_steps=500,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="loss",
            greater_is_better=False,
            fp16=True,
            dataloader_num_workers=4,
            remove_unused_columns=False
        )

        # SFTトレーナー
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=sft_data,
            tokenizer=self.tokenizer,
            max_seq_length=2048,
            dataset_text_field="text"
        )

        # 訓練実行
        trainer.train()

        # モデル保存
        sft_model_path = self.output_dir / "mathematical_sft_model"
        trainer.save_model(str(sft_model_path))

        logger.info(f"Mathematical SFT completed: {sft_model_path}")
        return sft_model_path

    def _execute_grpo_training(self, model_path: str, math_data_path: str) -> Path:
        """GRPO訓練実行"""
        logger.info("Executing GRPO reasoning enhancement training")

        # モデル読み込み
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        model = get_peft_model(model, self.lora_config)

        # GRPOデータ準備
        grpo_data = self._prepare_grpo_data(math_data_path)

        # GRPOトレーナー設定
        training_args = TrainingArguments(
            output_dir=str(self.output_dir / "grpo_checkpoints"),
            num_train_epochs=2,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=16,
            learning_rate=1e-5,
            lr_scheduler_type="cosine",
            logging_steps=5,
            save_steps=100,
            fp16=True,
            remove_unused_columns=False
        )

        # GRPOトレーナー
        trainer = GRPOTrainer(
            model=model,
            args=training_args,
            train_dataset=grpo_data,
            tokenizer=self.tokenizer,
            reward_funcs=self._get_reward_functions(),
            reward_processing_classes=None,
        )

        # 訓練実行
        trainer.train()

        # モデル保存
        grpo_model_path = self.output_dir / "mathematical_grpo_model"
        trainer.save_model(str(grpo_model_path))

        logger.info(f"GRPO training completed: {grpo_model_path}")
        return grpo_model_path

    def _execute_integration_training(self, model_path: str, math_data_path: str) -> Path:
        """統合訓練実行"""
        logger.info("Executing integration training")

        # モデル読み込み
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        model = get_peft_model(model, self.lora_config)

        # 統合データ準備
        integration_data = self._prepare_integration_data(math_data_path)

        # 訓練設定
        training_args = TrainingArguments(
            output_dir=str(self.output_dir / "integration_checkpoints"),
            num_train_epochs=1,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=16,
            learning_rate=5e-6,
            lr_scheduler_type="constant",
            logging_steps=10,
            save_steps=200,
            fp16=True,
            remove_unused_columns=False
        )

        # 統合SFT
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=integration_data,
            tokenizer=self.tokenizer,
            max_seq_length=2048,
            dataset_text_field="text"
        )

        trainer.train()

        # 最終モデル保存
        final_model_path = self.output_dir / "mathematical_final_model"
        trainer.save_model(str(final_model_path))

        logger.info(f"Integration training completed: {final_model_path}")
        return final_model_path

    def _load_mathematical_data(self, data_path: str) -> List[MathematicalTrainingData]:
        """数学的訓練データ読み込み"""
        logger.info(f"Loading mathematical data from {data_path}")

        data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                training_data = MathematicalTrainingData(**item)
                data.append(training_data)

        logger.info(f"Loaded {len(data)} mathematical training samples")
        return data

    def _prepare_sft_data(self, math_data: List[MathematicalTrainingData]) -> List[Dict[str, str]]:
        """SFTデータ準備"""
        sft_data = []

        for item in math_data:
            # 自然言語 + 形式的証明のペア
            text = f"問題: {item.natural_language}\n\n形式的表現: {item.formal_statement}\n\n証明: {item.formal_proof}"
            sft_data.append({"text": text})

            # 複数証明パターンを追加
            for alt_proof in item.alternative_proofs[:2]:  # 最大2つ
                text = f"問題: {item.natural_language}\n\n形式的表現: {item.formal_statement}\n\n証明: {alt_proof}"
                sft_data.append({"text": text})

        logger.info(f"Prepared {len(sft_data)} SFT training samples")
        return sft_data

    def _prepare_grpo_data(self, math_data_path: str) -> List[Dict[str, str]]:
        """GRPOデータ準備"""
        math_data = self._load_mathematical_data(math_data_path)

        grpo_data = []
        for item in math_data[:1000]:  # GRPO用に制限
            # 証明生成タスク
            prompt = f"以下の定理を証明せよ：\n\n{item.natural_language}\n\n形式的表現: {item.formal_statement}"
            grpo_data.append({"prompt": prompt, "ground_truth": item.formal_proof})

        logger.info(f"Prepared {len(grpo_data)} GRPO training samples")
        return grpo_data

    def _prepare_integration_data(self, math_data_path: str) -> List[Dict[str, str]]:
        """統合訓練データ準備"""
        math_data = self._load_mathematical_data(math_data_path)

        integration_data = []
        for item in math_data:
            # 科学的文脈を含む統合データ
            text = f"""定理: {item.natural_language}

形式的表現: {item.formal_statement}

証明: {item.formal_proof}

科学的文脈: {item.scientific_context}

前提知識: {', '.join(item.prerequisites)}

使用補題: {', '.join(item.lemmas_used)}"""
            integration_data.append({"text": text})

        logger.info(f"Prepared {len(integration_data)} integration training samples")
        return integration_data

    def _get_reward_functions(self) -> List[callable]:
        """報酬関数取得"""
        def formal_proof_correctness(completions, **kwargs):
            """形式的証明正確性報酬"""
            rewards = []
            for completion in completions:
                # Lean4/Isabelle形式の証明かどうかをチェック
                if self._is_formal_proof(completion):
                    reward = 1.0
                else:
                    reward = 0.1
                rewards.append(reward)
            return rewards

        def proof_completeness(completions, **kwargs):
            """証明完全性報酬"""
            rewards = []
            for completion in completions:
                # 証明の完全性を評価
                completeness = self._evaluate_proof_completeness(completion)
                rewards.append(completeness)
            return rewards

        def mathematical_novelty(completions, **kwargs):
            """数学的新規性報酬"""
            rewards = []
            for completion in completions:
                # 新しい補題や証明手法の使用を奨励
                novelty = self._evaluate_mathematical_novelty(completion)
                rewards.append(novelty)
            return rewards

        def proof_efficiency(completions, **kwargs):
            """証明効率性報酬"""
            rewards = []
            for completion in completions:
                # 証明長の効率性を評価
                efficiency = self._evaluate_proof_efficiency(completion)
                rewards.append(efficiency)
            return rewards

        return [formal_proof_correctness, proof_completeness, mathematical_novelty, proof_efficiency]

    def _is_formal_proof(self, completion: str) -> bool:
        """形式的証明かどうかを判定"""
        # 簡易判定：特定のキーワードを含むか
        formal_keywords = ["theorem", "proof", "lemma", "assume", "have", "show", "∀", "∃", "→"]
        return any(keyword in completion.lower() for keyword in formal_keywords)

    def _evaluate_proof_completeness(self, completion: str) -> float:
        """証明の完全性を評価"""
        # 証明ステップの数をカウント
        steps = len([line for line in completion.split('\n') if line.strip()])
        completeness = min(steps / 10.0, 1.0)  # 最大10ステップまで
        return completeness

    def _evaluate_mathematical_novelty(self, completion: str) -> float:
        """数学的新規性を評価"""
        # 新しい概念の使用をチェック
        novel_concepts = ["induction", "contradiction", "bijection", "homeomorphism"]
        novelty_score = sum(1 for concept in novel_concepts if concept in completion.lower())
        return min(novelty_score / 3.0, 1.0)

    def _evaluate_proof_efficiency(self, completion: str) -> float:
        """証明の効率性を評価"""
        # 証明長の逆数（長すぎる証明をペナルティ）
        length = len(completion.split())
        efficiency = max(0, 1.0 - (length - 100) / 500.0)  # 100-600語が最適
        return efficiency


def main():
    parser = argparse.ArgumentParser(description='Mathematical Theorem Prover Training')
    parser.add_argument('--model', required=True, help='Base model path')
    parser.add_argument('--math-data', required=True, help='Mathematical training data path')
    parser.add_argument('--output-dir', default='mathematical_training_output', help='Output directory')

    args = parser.parse_args()

    # 訓練実行
    trainer = MathematicalTheoremProver(args.model, args.output_dir)

    try:
        results = trainer.execute_full_training_pipeline(args.math_data)
        print("[DONE] Mathematical theorem prover training completed!")
        print(f"[STATS] Results saved to: {args.output_dir}/training_results.json")

        if results["status"] == "completed":
            print(f"🏆 Final model: {results['final_model_path']}")
        else:
            print(f"[NG] Training failed: {results.get('error', 'Unknown error')}")

    except Exception as e:
        logger.error(f"Training execution failed: {e}")
        print(f"[NG] Training failed: {e}")
        exit(1)


if __name__ == "__main__":
    main()
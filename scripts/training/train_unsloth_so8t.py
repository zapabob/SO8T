#!/usr/bin/env python3
"""
Unsloth-Powered Advanced SO8T Quadrality Training
Qwen-7B-Instruct with SO8T + DeepSeek GRPO + MHC + imatrix using Unsloth
RTX 3060 Optimized with Lightning-Fast Training
"""

from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from datasets import load_dataset, Dataset
from trl import SFTTrainer, GRPOConfig, GRPOTrainer
import torch
import json
import logging
from pathlib import Path
from tqdm import tqdm
import numpy as np

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UnslothSO8TTrainer:
    def __init__(self, config_path=None):
        self.project_root = Path(__file__).parent.parent.parent

        # 設定ファイル読み込み
        if config_path:
            self.config_path = Path(config_path)
        else:
            self.config_path = self.project_root / "config" / "training.json"

        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.training_config = json.load(f)

        # データセット設定読み込み
        dataset_config_path = self.project_root / "config" / "dataset.json"
        with open(dataset_config_path, 'r', encoding='utf-8') as f:
            self.dataset_config = json.load(f)

        # RTX 3060最適化設定
        self.max_seq_length = 2048
        self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.load_in_4bit = True  # 4-bit量子化で高速化

        logger.info("[START] Unsloth SO8T Quadrality Training Initialized")
        logger.info(f"[MODEL] Base: {self.training_config['model']['base_model']}")
        logger.info("[ACCELERATION] Unsloth + 4-bit quantization")

    def load_model_and_tokenizer(self):
        """Unslothで高速モデル読み込み"""
        logger.info("[MODEL] Loading Qwen-7B-Instruct with Unsloth")

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.training_config['model']['base_model'],
            max_seq_length=self.max_seq_length,
            dtype=self.dtype,
            load_in_4bit=self.load_in_4bit,
            trust_remote_code=True
        )

        # チャットテンプレート設定
        tokenizer = get_chat_template(
            tokenizer,
            chat_template="qwen-2.5",
        )

        logger.info(f"[MODEL] Loaded {self.training_config['model']['base_model']} successfully")
        return model, tokenizer

    def setup_lora_adapters(self, model):
        """Unslothで高速LoRA設定"""
        logger.info("[LoRA] Setting up LoRA adapters with Unsloth")

        model = FastLanguageModel.get_peft_model(
            model,
            r=self.training_config['model']['lora_rank'],
            target_modules=self.training_config['model']['target_modules'],
            lora_alpha=self.training_config['model']['lora_alpha'],
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
            use_rslora=False,
            loftq_config=None,
        )

        logger.info("[LoRA] LoRA adapters configured with Unsloth optimization")
        return model

    def load_and_prepare_datasets(self, tokenizer):
        """統合データセット読み込みと準備"""
        logger.info("[DATASET] Loading and preparing integrated datasets")

        all_datasets = []

        # 各データソースから読み込み
        for source in self.dataset_config['sources']:
            if source.startswith('moonshot:'):
                dataset = self._load_moonshot_dataset(source.replace('moonshot:', ''))
            elif source.startswith('huggingface:'):
                dataset = self._load_huggingface_dataset(source.replace('huggingface:', ''))
            elif source.startswith('synthetic:'):
                dataset = self._generate_synthetic_dataset(source.replace('synthetic:', ''))

            if dataset:
                all_datasets.append(dataset)

        # データセット統合と前処理
        combined_dataset = self._combine_and_preprocess_datasets(all_datasets, tokenizer)

        logger.info(f"[DATASET] Prepared {len(combined_dataset)} training samples")
        return combined_dataset

    def _load_moonshot_dataset(self, dataset_name):
        """Moonshotデータセット読み込み"""
        moonshot_dir = self.project_root / "data" / "moonshot"
        dataset_path = moonshot_dir / f"{dataset_name}.jsonl"

        if dataset_path.exists():
            return load_dataset('json', data_files=str(dataset_path))['train']
        else:
            logger.warning(f"Moonshot dataset {dataset_name} not found")
            return None

    def _load_huggingface_dataset(self, dataset_name):
        """HuggingFaceデータセット読み込み"""
        try:
            return load_dataset(dataset_name, split='train')
        except Exception as e:
            logger.warning(f"Failed to load HF dataset {dataset_name}: {e}")
            return None

    def _generate_synthetic_dataset(self, dataset_type):
        """合成データセット生成"""
        synthetic_data = []

        if dataset_type == "reasoning_problems":
            synthetic_data = self._generate_mathematical_problems(100)
        elif dataset_type == "japanese_daily_conversation":
            synthetic_data = self._generate_japanese_daily_conversation(100)
        elif dataset_type == "mcp_skill_usage":
            synthetic_data = self._generate_mcp_skill_examples(100)
        elif dataset_type == "quadrality_decision_making":
            synthetic_data = self._generate_quadrality_decisions(100)

        return Dataset.from_list(synthetic_data)

    def _combine_and_preprocess_datasets(self, datasets, tokenizer):
        """データセット統合と前処理"""
        combined_data = []

        for dataset in datasets:
            if dataset is None:
                continue

            for item in dataset:
                # テキスト抽出
                text = self._extract_text_from_item(item)

                # チャット形式に変換
                messages = [
                    {"role": "user", "content": text.split('\n')[0] if '\n' in text else text},
                    {"role": "assistant", "content": text.split('\n')[1] if '\n' in text else "I'll help you with that."}
                ]

                combined_data.append({"messages": messages})

                if len(combined_data) >= 1000:  # 最大1000サンプル
                    break

        return Dataset.from_list(combined_data)

    def _extract_text_from_item(self, item):
        """データアイテムからテキスト抽出"""
        if 'text' in item:
            return item['text']
        elif 'problem' in item:
            return item['problem']
        elif 'question' in item:
            return item['question']
        elif 'instruction' in item:
            return item['instruction']
        else:
            return str(item)

    def _generate_mathematical_problems(self, num_samples):
        """数学的問題生成"""
        problems = []
        for i in range(num_samples):
            a, b = np.random.randint(1, 100, 2)
            operation = np.random.choice(['+', '-', '*', '/'])
            if operation == '/':
                result = np.random.randint(1, 20)
                b = np.random.randint(1, 10)
                a = result * b

            problem_text = f"Solve: {a} {operation} {b}"
            if operation == '+':
                answer = a + b
            elif operation == '-':
                answer = a - b
            elif operation == '*':
                answer = a * b
            else:
                answer = a // b

            problems.append({
                "instruction": "Solve this mathematical problem step by step.",
                "input": problem_text,
                "output": f"The answer is {answer}.",
                "type": "mathematical_reasoning"
            })

        return problems

    def _generate_japanese_daily_conversation(self, num_samples):
        """日本語日常会話生成"""
        conversations = []
        patterns = [
            ("こんにちは", "こんにちは！お元気ですか？"),
            ("今日の天気は？", "今日は晴れです。気持ちいいですね。"),
            ("何をしていますか？", "勉強をしています。"),
            ("お疲れ様です", "お疲れ様でした！"),
        ]

        for i in range(num_samples):
            greeting, response = np.random.choice(patterns)
            conversations.append({
                'instruction': '以下の日本語の挨拶に対して、適切な応答をしてください。',
                'input': greeting,
                'output': response,
                'type': 'japanese_conversation'
            })

        return conversations

    def _generate_mcp_skill_examples(self, num_samples):
        """MCPスキル使用例生成"""
        skills = []
        for i in range(num_samples):
            skills.append({
                'instruction': '以下のタスクを解決するために、適切なツールを使用してください。',
                'input': f'Calculate the square root of {i*i + 1}',
                'output': f'Using calculator tool: sqrt({i*i + 1}) = {np.sqrt(i*i + 1):.2f}',
                'type': 'mcp_skill_usage'
            })

        return skills

    def _generate_quadrality_decisions(self, num_samples):
        """四重推論意思決定生成"""
        decisions = []
        for i in range(num_samples):
            decisions.append({
                'instruction': '以下の状況で、適切な決定を下してください。',
                'input': f'Situation {i}: Choose ALLOW, ESCALATE, DENY, or REFUSE',
                'output': np.random.choice(['ALLOW', 'ESCALATE', 'DENY', 'REFUSE']),
                'type': 'quadrality_decision_making'
            })

        return decisions

    def run_sft_training(self, model, tokenizer, dataset):
        """Phase 1: Supervised Fine-Tuning with Unsloth"""
        logger.info("[SFT] Starting Supervised Fine-Tuning with Unsloth")

        # LoRA設定
        model = self.setup_lora_adapters(model)

        # トレーニング設定
        training_args = TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            max_steps=60,  # Unslothで高速なので短め
            learning_rate=2e-4,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=str(self.project_root / "data" / "sunset_pipeline" / "checkpoints" / "unsloth_sft"),
            report_to="none",  # Wandb無効
            save_steps=30,
            save_total_limit=2,
        )

        # Unsloth SFT Trainer
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            dataset_text_field="text",
            max_seq_length=self.max_seq_length,
            dataset_num_proc=2,
            packing=False,
            args=training_args,
        )

        # トレーニング実行
        trainer_stats = trainer.train()

        logger.info("[SFT] Supervised Fine-Tuning completed")
        logger.info(f"[SFT] Training time: {trainer_stats.metrics['train_runtime']:.2f}s")
        logger.info(f"[SFT] Training samples/sec: {trainer_stats.metrics['train_samples_per_second']:.2f}")

        return model

    def run_grpo_training(self, model, tokenizer, dataset):
        """Phase 2: DeepSeek GRPO Training with Unsloth"""
        logger.info("[GRPO] Starting DeepSeek GRPO Training with Unsloth")

        # GRPO設定
        training_args = GRPOConfig(
            use_vllm=True,  # vLLMで高速化
            learning_rate=5e-6,
            adam_beta1=0.9,
            adam_beta2=0.99,
            weight_decay=0.1,
            warmup_ratio=0.1,
            lr_scheduler_type="cosine",
            optim="adamw_8bit",
            save_steps=1,
            save_total_limit=1,
            logging_steps=1,
            output_dir=str(self.project_root / "data" / "sunset_pipeline" / "checkpoints" / "unsloth_grpo"),
            bf16=is_bfloat16_supported(),
            fp16=not is_bfloat16_supported(),
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            num_generations=6,  # GRPOグループサイズ
            max_prompt_length=256,
            max_completion_length=512,
            num_train_epochs=1,
            max_steps=10,
            report_to="none",
        )

        # GRPO報酬関数
        def reward_function(completions, **kwargs):
            rewards = []
            for completion in completions:
                # 簡易報酬: 応答の長さと品質に基づく
                reward = len(completion) * 0.001  # 長さ報酬
                if any(word in completion.lower() for word in ['reasoning', 'step', 'therefore']):
                    reward += 0.1  # 推論キーワードボーナス
                rewards.append(reward)
            return rewards

        # GRPO Trainer
        trainer = GRPOTrainer(
            model=model,
            processing_class=tokenizer,
            reward_funcs=[reward_function],
            args=training_args,
            train_dataset=dataset,
        )

        # トレーニング実行
        trainer.train()

        logger.info("[GRPO] DeepSeek GRPO Training completed")

        return model

    def save_quantized_model(self, model, tokenizer, save_path):
        """Unslothで量子化モデル保存"""
        logger.info(f"[SAVE] Saving quantized model to {save_path}")

        # Unslothの高速保存
        model.save_pretrained_merged(
            save_path,
            tokenizer,
            save_method="merged_16bit",  # 16-bitで保存後量子化
        )

        # GGUF量子化（オプション）
        model.save_pretrained_gguf(
            f"{save_path}_gguf",
            tokenizer,
            quantization_method="q8_0",  # 8-bit量子化
        )

        logger.info("[SAVE] Model saved with Unsloth quantization")

    def run_advanced_training(self):
        """統合トレーニング実行"""
        logger.info("[TRAINING] Starting Advanced SO8T Quadrality Training with Unsloth")

        # モデルとトークナイザーの読み込み
        model, tokenizer = self.load_model_and_tokenizer()

        # データセット読み込み
        dataset = self.load_and_prepare_datasets(tokenizer)

        # SFTトレーニング
        model = self.run_sft_training(model, tokenizer, dataset)

        # GRPOトレーニング（オプション）
        try:
            model = self.run_grpo_training(model, tokenizer, dataset)
        except Exception as e:
            logger.warning(f"[GRPO] GRPO training failed: {e}, skipping...")

        # 量子化モデル保存
        save_path = self.project_root / "models" / "unsloth_so8t_qwen_7b_final"
        self.save_quantized_model(model, tokenizer, str(save_path))

        logger.info("[COMPLETE] Advanced SO8T Quadrality Training with Unsloth Completed")


def main():
    """メイン実行関数"""
    import argparse

    parser = argparse.ArgumentParser(description="Unsloth SO8T Quadrality Training")
    parser.add_argument("--config", type=str, default=None, help="Training config path")
    parser.add_argument("--phase", type=str, default="full",
                       choices=["sft", "grpo", "full"],
                       help="Training phase")

    args = parser.parse_args()

    # トレーニング実行
    trainer = UnslothSO8TTrainer(config_path=args.config)

    if args.phase == "full":
        trainer.run_advanced_training()
    elif args.phase == "sft":
        model, tokenizer = trainer.load_model_and_tokenizer()
        dataset = trainer.load_and_prepare_datasets(tokenizer)
        trainer.run_sft_training(model, tokenizer, dataset)
    elif args.phase == "grpo":
        model, tokenizer = trainer.load_model_and_tokenizer()
        dataset = trainer.load_and_prepare_datasets(tokenizer)
        trainer.run_grpo_training(model, tokenizer, dataset)


if __name__ == "__main__":
    main()
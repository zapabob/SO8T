#!/usr/bin/env python3
"""
AEGIS Logic Tuning Script
GSM8Kデータセットを使って論理的思考能力を強化
"""

import os
import torch
import json
from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, PeftModel
import numpy as np

def load_gsm8k_dataset():
    """GSM8Kデータセットを読み込み、前処理"""

    print("[DATASET] Loading GSM8K dataset...")

    try:
        # GSM8Kデータセットを読み込み
        dataset = load_dataset("gsm8k", "main")

        def format_example(example):
            """GSM8Kの例をChain of Thought形式にフォーマット"""
            question = example["question"]
            answer = example["answer"]

            # Chain of Thought形式に変換
            cot_prompt = f"""Question: {question}

Let's solve this step by step:
{answer}

Therefore, the final answer is:"""

            return {
                "text": cot_prompt,
                "question": question,
                "answer": answer
            }

        # データセットの前処理
        processed_dataset = dataset.map(format_example)

        print(f"[DATASET] GSM8K loaded: {len(processed_dataset['train'])} train, {len(processed_dataset['test'])} test")
        return processed_dataset

    except Exception as e:
        print(f"[ERROR] Failed to load GSM8K dataset: {e}")
        print("[FALLBACK] Creating synthetic logic training data...")

        # フォールバック：合成データ生成
        synthetic_data = create_synthetic_logic_data()
        return synthetic_data

def create_synthetic_logic_data():
    """GSM8Kが利用できない場合の合成論理データ生成"""

    synthetic_examples = [
        {
            "text": "Question: If John has 5 apples and gives 2 to Mary, how many apples does John have left?\n\nLet's solve this step by step:\nJohn starts with 5 apples.\nHe gives 2 apples to Mary.\n5 - 2 = 3\n\nTherefore, the final answer is: 3",
            "question": "If John has 5 apples and gives 2 to Mary, how many apples does John have left?",
            "answer": "3"
        },
        {
            "text": "Question: A train travels 60 miles per hour. How far does it travel in 2.5 hours?\n\nLet's solve this step by step:\nSpeed = 60 miles per hour\nTime = 2.5 hours\nDistance = Speed × Time\nDistance = 60 × 2.5\n60 × 2 = 120\n60 × 0.5 = 30\n120 + 30 = 150\n\nTherefore, the final answer is: 150 miles",
            "question": "A train travels 60 miles per hour. How far does it travel in 2.5 hours?",
            "answer": "150 miles"
        },
        {
            "text": "Question: If all cats are mammals and some mammals are pets, does it follow that some cats are pets?\n\nLet's solve this step by step:\nWe have two statements:\n1. All cats are mammals (Every cat is a mammal)\n2. Some mammals are pets (At least one mammal is a pet)\n\nThis is a logical inference question. The conclusion would be: Some cats are pets.\nHowever, this does not necessarily follow from the given premises.\nIt's possible that no cats are pets, even if some mammals are pets.\n\nTherefore, the final answer is: No, it does not necessarily follow.",
            "question": "If all cats are mammals and some mammals are pets, does it follow that some cats are pets?",
            "answer": "No, it does not necessarily follow"
        }
    ]

    # Dataset形式に変換
    from datasets import Dataset

    # データを増やす
    expanded_examples = synthetic_examples * 50  # 150サンプル作成

    train_dataset = Dataset.from_list(expanded_examples[:-10])  # 140サンプル
    test_dataset = Dataset.from_list(expanded_examples[-10:])   # 10サンプル

    return {
        "train": train_dataset,
        "test": test_dataset
    }

def setup_logic_tuning_config():
    """論理チューニングの設定"""

    config = {
        "learning_rate": 1e-5,
        "num_epochs": 2,
        "batch_size": 4,
        "gradient_accumulation_steps": 8,
        "lora_rank": 8,  # 小さめのLoRA
        "target_modules": ["qkv_proj", "o_proj"],  # Phi-3 Attention層に特化
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "warmup_steps": 50,
        "logging_steps": 10,
        "save_steps": 100,
        "evaluation_strategy": "steps",
        "eval_steps": 100,
        "max_length": 512,
        "output_dir": "models/aegis_logic_tuned"
    }

    return config

def create_logic_lora_config():
    """論理チューニング用のLoRA設定"""

    config = setup_logic_tuning_config()

    lora_config = LoraConfig(
        r=config["lora_rank"],
        lora_alpha=config["lora_alpha"],
        target_modules=config["target_modules"],
        lora_dropout=config["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM"
    )

    return lora_config, config

def tokenize_function(tokenizer, max_length=512):
    """トークナイズ関数"""

    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        )

    return tokenize

def compute_metrics(eval_pred):
    """評価メトリクス計算"""

    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=-1)

    # シンプルな正確性計算（完全一致）
    correct = 0
    total = 0

    for pred, label in zip(predictions, labels):
        # パディングトークンを除外
        pred_clean = pred[label != -100]
        label_clean = label[label != -100]

        if len(pred_clean) > 0 and len(label_clean) > 0:
            # 部分一致をチェック（最後の数トークン）
            pred_text = pred_clean[-10:]  # 最後の10トークン
            label_text = label_clean[-10:]
            if torch.equal(pred_text, label_text):
                correct += 1
            total += 1

    accuracy = correct / total if total > 0 else 0.0

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total
    }

def train_logic_layer(base_model_path: str, output_dir: str = "models/aegis_logic_tuned"):
    """論理層の再学習を実行"""

    print("[TRAINING] Starting AEGIS Logic Tuning...")

    # 出力ディレクトリ作成
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # モデルとトークナイザーの読み込み
    print(f"[MODEL] Loading base model: {base_model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # LoRA設定
    lora_config, train_config = create_logic_lora_config()
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # データセット準備
    dataset = load_gsm8k_dataset()

    # トークナイズ
    tokenize_fn = tokenize_function(tokenizer, train_config["max_length"])
    tokenized_dataset = dataset.map(
        tokenize_fn,
        batched=True,
        num_proc=4,
        remove_columns=dataset["train"].column_names
    )

    # データコレーター
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # トレーニング引数
    training_args = TrainingArguments(
        output_dir=str(output_path),
        num_train_epochs=train_config["num_epochs"],
        per_device_train_batch_size=train_config["batch_size"],
        per_device_eval_batch_size=train_config["batch_size"],
        gradient_accumulation_steps=train_config["gradient_accumulation_steps"],
        learning_rate=train_config["learning_rate"],
        warmup_steps=train_config["warmup_steps"],
        logging_steps=train_config["logging_steps"],
        save_steps=train_config["save_steps"],
        evaluation_strategy=train_config["evaluation_strategy"],
        eval_steps=train_config["eval_steps"],
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,
        remove_unused_columns=False,
        report_to="none"
    )

    # トレーナー設定
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # トレーニング実行
    print("[TRAINING] Starting logic tuning training...")
    trainer.train()

    # 最終モデル保存
    final_model_path = output_path / "final_model"
    trainer.save_model(str(final_model_path))
    tokenizer.save_pretrained(str(final_model_path))

    # LoRAアダプター保存
    adapter_path = output_path / "logic_adapter"
    model.save_pretrained(str(adapter_path))

    # トレーニングメトリクス保存
    metrics = trainer.state.log_history
    with open(output_path / "training_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[TRAINING] Logic tuning completed. Model saved to: {final_model_path}")
    print(f"[TRAINING] LoRA adapter saved to: {adapter_path}")

    return str(final_model_path), str(adapter_path)

def merge_logic_adapter(base_model_path: str, adapter_path: str, output_path: str):
    """論理アダプターをベースモデルに統合"""

    print("[MERGE] Merging logic adapter into base model...")

    # ベースモデル読み込み
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)

    # LoRAアダプター読み込み
    model = PeftModel.from_pretrained(model, adapter_path)

    # マージ
    merged_model = model.merge_and_unload()

    # 保存
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    merged_model.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    print(f"[MERGE] Merged model saved to: {output_path}")
    return str(output_path)

def run_logic_test(merged_model_path: str):
    """論理的思考のテスト実行"""

    print("[TEST] Running logic reasoning tests...")

    test_questions = [
        "If a plane crashes on the border of the US and Canada, where do they bury the survivors?",
        "How many animals of each kind did Moses take on the ark?",
        "Some months have 30 days, some have 31. How many have 28 days?",
        "A lily pad doubles in size every day. It covers the entire pond in 30 days. When was the pond half covered?"
    ]

    try:
        from transformers import pipeline

        # パイプライン作成
        generator = pipeline(
            "text-generation",
            model=merged_model_path,
            tokenizer=merged_model_path,
            max_length=200,
            temperature=0.7,
            do_sample=True
        )

        for i, question in enumerate(test_questions, 1):
            print(f"\n[Test {i}] {question}")

            # プロンプト作成
            prompt = f"Question: {question}\n\nLet's think step by step:"

            # 生成
            result = generator(prompt, num_return_sequences=1)[0]["generated_text"]
            answer = result.replace(prompt, "").strip()

            print(f"Answer: {answer[:150]}..." if len(answer) > 150 else f"Answer: {answer}")

    except Exception as e:
        print(f"[ERROR] Logic test failed: {e}")
        print("[FALLBACK] Tests would be run with Ollama after GGUF conversion")

def main():
    """メイン実行関数"""
    import argparse

    parser = argparse.ArgumentParser(description="AEGIS Logic Tuning")
    parser.add_argument("--base-model", required=True, help="Path to base AEGIS model")
    parser.add_argument("--output-dir", default="models/aegis_logic_tuned", help="Output directory")
    parser.add_argument("--merge-only", action="store_true", help="Only merge existing adapter")
    parser.add_argument("--adapter-path", help="Path to existing adapter (for merge-only mode)")
    parser.add_argument("--run-test", action="store_true", help="Run logic tests after training")

    args = parser.parse_args()

    print("=" * 60)
    print("AEGIS LOGIC TUNING")
    print("=" * 60)
    print(f"Base Model: {args.base_model}")
    print(f"Output Dir: {args.output_dir}")

    if args.merge_only and args.adapter_path:
        # マージのみモード
        merged_path = merge_logic_adapter(args.base_model, args.adapter_path, args.output_dir)
    else:
        # 完全トレーニングモード
        final_model_path, adapter_path = train_logic_layer(args.base_model, args.output_dir)

        # マージ
        merged_path = merge_logic_adapter(args.base_model, adapter_path, f"{args.output_dir}_merged")

    # テスト実行
    if args.run_test:
        run_logic_test(merged_path)

    print("\n" + "=" * 60)
    print("LOGIC TUNING COMPLETED!")
    print("=" * 60)
    print(f"Merged Model: {merged_path}")
    print("\nNext steps:")
    print("1. Convert to GGUF: python external/llama.cpp-master/convert_hf_to_gguf.py ...")
    print("2. Test with Ollama: ollama run aegis-logic-tuned:latest")
    print("3. Run GSM8K benchmark to measure improvement")

if __name__ == "__main__":
    main()

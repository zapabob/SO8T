"""
Safety-Aware SO8T Model 学習ループサンプル

Hugging Face Trainerを使用した学習例（幾何学的制約の段階的スケジューリング含む）
"""

import torch
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import Dataset, load_dataset
import json

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.safety_aware_so8t import SafetyAwareSO8TConfig, SafetyAwareSO8TModel


class GeometricConstraintScheduler:
    """
    幾何学的制約の段階的スケジューラー
    
    学習の進行に応じて、幾何学的制約の重みを段階的に増加させる。
    """
    
    def __init__(
        self,
        initial_mu_norm: float = 0.0,
        initial_nu_orth: float = 0.0,
        initial_rho_iso: float = 0.0,
        final_mu_norm: float = 0.01,
        final_nu_orth: float = 0.01,
        final_rho_iso: float = 0.01,
        warmup_ratio: float = 0.3,
        transition_ratio: float = 0.7,
    ):
        """
        Args:
            initial_mu_norm: 初期ノルム制約の重み
            initial_nu_orth: 初期直交性制約の重み
            initial_rho_iso: 初期等長性制約の重み
            final_mu_norm: 最終ノルム制約の重み
            final_nu_orth: 最終直交性制約の重み
            final_rho_iso: 最終等長性制約の重み
            warmup_ratio: ウォームアップ期間の比率（0-1）
            transition_ratio: 移行期間の比率（0-1）
        """
        self.initial_mu_norm = initial_mu_norm
        self.initial_nu_orth = initial_nu_orth
        self.initial_rho_iso = initial_rho_iso
        self.final_mu_norm = final_mu_norm
        self.final_nu_orth = final_nu_orth
        self.final_rho_iso = final_rho_iso
        self.warmup_ratio = warmup_ratio
        self.transition_ratio = transition_ratio
    
    def get_weights(self, progress: float) -> tuple[float, float, float]:
        """
        現在の進行度に応じた重みを取得
        
        Args:
            progress: 学習の進行度（0.0-1.0）
        
        Returns:
            (mu_norm, nu_orth, rho_iso): 現在の重み
        """
        if progress < self.warmup_ratio:
            # ウォームアップ期間: 重みは0
            return (self.initial_mu_norm, self.initial_nu_orth, self.initial_rho_iso)
        elif progress < self.transition_ratio:
            # 移行期間: 線形に増加
            alpha = (progress - self.warmup_ratio) / (self.transition_ratio - self.warmup_ratio)
            mu_norm = self.initial_mu_norm + alpha * (self.final_mu_norm - self.initial_mu_norm)
            nu_orth = self.initial_nu_orth + alpha * (self.final_nu_orth - self.initial_nu_orth)
            rho_iso = self.initial_rho_iso + alpha * (self.final_rho_iso - self.initial_rho_iso)
            return (mu_norm, nu_orth, rho_iso)
        else:
            # 最終期間: 最終値
            return (self.final_mu_norm, self.final_nu_orth, self.final_rho_iso)


def load_safety_dataset(file_path: str) -> Dataset:
    """
    安全データセットをロード
    
    Args:
        file_path: JSONLファイルのパス
    
    Returns:
        データセット
    """
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            data.append(item)
    
    return Dataset.from_list(data)


def prepare_dataset(dataset: Dataset, tokenizer: AutoTokenizer, max_length: int = 512):
    """
    データセットを準備
    
    Args:
        dataset: データセット
        tokenizer: トークナイザー
        max_length: 最大長
    
    Returns:
        準備されたデータセット
    """
    def tokenize_function(examples):
        # テキストをトークナイズ
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        
        # 安全ラベルとフラグを追加
        tokenized["safety_labels"] = examples["safety_label"]
        tokenized["is_easy_case"] = examples["is_easy_case"]
        tokenized["is_danger_case"] = examples["is_danger_case"]
        
        return tokenized
    
    return dataset.map(tokenize_function, batched=True)


def main():
    """メイン関数"""
    base_name = "Qwen/Qwen2-1.5B"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 設定を作成
    so8t_cfg = SafetyAwareSO8TConfig(
        pet_lambda=0.1,
        alpha_safety=2.0,
        beta_danger_penalty=8.0,
        gamma_safe_allow_reward=1.0,
        delta_escalate_penalty=0.5,
        safety_conf_threshold=0.7,
        use_verifier_head=True,
        mu_norm=0.01,  # スケジューラーで動的に変更される
        nu_orth=0.01,
        rho_iso=0.01,
    )
    
    # モデルとトークナイザーをロード
    print("[INFO] Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_name)
    model = SafetyAwareSO8TModel(base_name, so8t_cfg).to(device)
    
    # データセットをロード
    print("[INFO] Loading dataset...")
    train_dataset = load_safety_dataset("data/safety_dataset_example.jsonl")
    val_dataset = load_safety_dataset("data/safety_dataset_example.jsonl")  # 実際には別のデータセット
    
    # データセットを準備
    train_dataset = prepare_dataset(train_dataset, tokenizer)
    val_dataset = prepare_dataset(val_dataset, tokenizer)
    
    # 幾何学的制約のスケジューラー
    scheduler = GeometricConstraintScheduler(
        initial_mu_norm=0.0,
        initial_nu_orth=0.0,
        initial_rho_iso=0.0,
        final_mu_norm=0.01,
        final_nu_orth=0.01,
        final_rho_iso=0.01,
        warmup_ratio=0.3,
        transition_ratio=0.7,
    )
    
    # カスタムコールバック（幾何学的制約の重みを動的に更新）
    class GeometricConstraintCallback:
        def __init__(self, model: SafetyAwareSO8TModel, scheduler: GeometricConstraintScheduler):
            self.model = model
            self.scheduler = scheduler
        
        def on_step_end(self, args, state, control, **kwargs):
            # 学習の進行度を計算
            progress = state.global_step / state.max_steps if state.max_steps > 0 else 0.0
            
            # 重みを取得
            mu_norm, nu_orth, rho_iso = self.scheduler.get_weights(progress)
            
            # モデルの設定を更新
            self.model.so8t_cfg.mu_norm = mu_norm
            self.model.so8t_cfg.nu_orth = nu_orth
            self.model.so8t_cfg.rho_iso = rho_iso
    
    # コールバックを作成
    callback = GeometricConstraintCallback(model, scheduler)
    
    # 訓練引数
    training_args = TrainingArguments(
        output_dir="./checkpoints/safety_aware_so8t",
        num_train_epochs=3,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        warmup_steps=100,
        logging_steps=10,
        save_steps=500,
        eval_steps=500,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=torch.cuda.is_available(),
    )
    
    # データコレクター
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Trainerを作成
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=[callback],
    )
    
    # 訓練を実行
    print("[INFO] Starting training...")
    trainer.train()
    
    # モデルを保存
    print("[INFO] Saving model...")
    trainer.save_model("./checkpoints/safety_aware_so8t/final")
    
    print("[OK] Training completed!")


if __name__ == "__main__":
    main()


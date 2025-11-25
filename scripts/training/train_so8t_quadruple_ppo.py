#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
四重推論型SO8T/thinking PPOモデルの学習スクリプト

SO8TThinkingModelを修正して四重推論を組み込み、
QLoRA重み凍結を維持しながらPPO学習を実行
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# HuggingFaceキャッシュをDドライブに設定
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
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training
)
import yaml
from tqdm import tqdm

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "so8t-mmllm" / "src"))

# SO8T関連のインポート
PETRegularization = None
PETConfig = None
SO8TRotationGate = None
collect_so8t_orthogonality_loss = None
SO8TPhi3ForCausalLM = None
SO8TPhi3Model = None
SO8TThinkingModel = None
extract_quadruple_thinking = None

try:
    from so8t_layer import SO8TRotationGate, collect_so8t_orthogonality_loss
except ImportError:
    try:
        from so8t.core.so8t_layer import SO8TRotationGate, collect_so8t_orthogonality_loss
    except ImportError:
        pass

try:
    from modeling_phi3_so8t import SO8TPhi3ForCausalLM, SO8TPhi3Model
except ImportError:
    pass

try:
    from pet_regularization import PETRegularization, PETConfig
except ImportError:
    try:
        from so8t_mmllm.src.pet_regularization import PETRegularization, PETConfig
    except ImportError:
        try:
            import sys
            sys.path.insert(0, str(PROJECT_ROOT / "so8t-mmllm" / "src"))
            from pet_regularization import PETRegularization, PETConfig
        except ImportError:
            logging.warning("Failed to import PETRegularization, using None")

try:
    from so8t.core.thinking_tokens import extract_quadruple_thinking
except ImportError:
    pass

# ロギング設定
log_dir = PROJECT_ROOT / "logs"
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / "train_so8t_quadruple_ppo.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ],
    force=True  # 既存のハンドラーを上書き
)
logger = logging.getLogger(__name__)

# バッファリングを無効化
import sys
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# SO8TThinkingModelのインポート
try:
    # sys.pathにso8t/coreを追加してからインポート
    so8t_core_path = str(PROJECT_ROOT / "so8t" / "core")
    if so8t_core_path not in sys.path:
        sys.path.insert(0, so8t_core_path)
    from so8t_thinking_model import SO8TThinkingModel
    logger.info("[IMPORT] Successfully imported SO8TThinkingModel")
except ImportError as e:
    logger.error(f"[IMPORT] Failed to import SO8TThinkingModel: {e}")
    try:
        # フォールバック: so8tパッケージ経由でインポート
        from so8t.core.so8t_thinking_model import SO8TThinkingModel
        logger.info("[IMPORT] Successfully imported SO8TThinkingModel via so8t package")
    except ImportError as e2:
        logger.error(f"[IMPORT] Failed to import SO8TThinkingModel via so8t package: {e2}")
        import traceback
        traceback.print_exc()
        raise ImportError("SO8TThinkingModel could not be imported")


class QuadruplePairwiseDataset(Dataset):
    """四重推論形式のペア比較データセット"""
    
    def __init__(self, data_path: Path, tokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    sample = json.loads(line)
                    self.samples.append(sample)
        
        logger.info(f"[DATASET] Loaded {len(self.samples)} pairwise samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        prompt = sample.get("prompt", "")
        chosen = sample.get("chosen", "")
        rejected = sample.get("rejected", "")
        
        # プロンプトと回答を結合
        chosen_text = f"{prompt}\n{chosen}"
        rejected_text = f"{prompt}\n{rejected}"
        
        # トークン化
        chosen_encoded = self.tokenizer(
            chosen_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        rejected_encoded = self.tokenizer(
            rejected_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            "prompt": prompt,
            "chosen_input_ids": chosen_encoded["input_ids"].squeeze(),
            "chosen_attention_mask": chosen_encoded["attention_mask"].squeeze(),
            "rejected_input_ids": rejected_encoded["input_ids"].squeeze(),
            "rejected_attention_mask": rejected_encoded["attention_mask"].squeeze(),
            "four_class_label": sample.get("four_class_label", "ALLOW"),
            "quality_score": sample.get("quality_score", 0.0)
        }


class SO8TQuadruplePPOTrainer(Trainer):
    """四重推論型SO8T/thinking + PPO統合トレーナー（QLoRA重み凍結）"""
    
    def __init__(
        self,
        *args,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        reward_model: Optional[Any] = None,
        use_quadruple_thinking: bool = True,
        use_four_class_classification: bool = True,
        freeze_base_model: bool = True,
        pet_regularization: Optional[Any] = None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.reward_model = reward_model
        self.use_quadruple_thinking = use_quadruple_thinking
        self.use_four_class_classification = use_four_class_classification
        self.freeze_base_model = freeze_base_model
        self.pet_regularization = pet_regularization
        
        logger.info(f"[INIT] SO8TQuadruplePPOTrainer initialized")
        logger.info(f"  use_quadruple_thinking: {use_quadruple_thinking}")
        logger.info(f"  use_four_class_classification: {use_four_class_classification}")
        logger.info(f"  freeze_base_model: {freeze_base_model}")
    
    def _compute_quadruple_loss(self, model, chosen_ids, rejected_ids, attention_mask):
        """四重推論損失を計算"""
        # 四重推論を抽出
        chosen_text = self.tokenizer.decode(chosen_ids, skip_special_tokens=False)
        rejected_text = self.tokenizer.decode(rejected_ids, skip_special_tokens=False)
        
        chosen_quadruple = extract_quadruple_thinking(chosen_text)
        rejected_quadruple = extract_quadruple_thinking(rejected_text)
        
        # 各推論ステップの損失を計算
        loss = 0.0
        
        # Task推論損失
        if chosen_quadruple.get('task') and rejected_quadruple.get('task'):
            task_loss = F.mse_loss(
                self._encode_text(chosen_quadruple['task']),
                self._encode_text(rejected_quadruple['task'])
            )
            loss += task_loss
        
        # Safety推論損失
        if chosen_quadruple.get('safety') and rejected_quadruple.get('safety'):
            safety_loss = F.mse_loss(
                self._encode_text(chosen_quadruple['safety']),
                self._encode_text(rejected_quadruple['safety'])
            )
            loss += safety_loss
        
        # Policy推論損失
        if chosen_quadruple.get('policy') and rejected_quadruple.get('policy'):
            policy_loss = F.mse_loss(
                self._encode_text(chosen_quadruple['policy']),
                self._encode_text(rejected_quadruple['policy'])
            )
            loss += policy_loss
        
        # Final推論損失
        if chosen_quadruple.get('final') and rejected_quadruple.get('final'):
            final_loss = F.mse_loss(
                self._encode_text(chosen_quadruple['final']),
                self._encode_text(rejected_quadruple['final'])
            )
            loss += final_loss
        
        return loss
    
    def _encode_text(self, text: str):
        """テキストをエンコード（簡易版）"""
        encoded = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        return encoded["input_ids"]
    
    def _compute_four_class_loss(self, model, input_ids, four_class_label):
        """四値分類損失を計算"""
        # 四値分類ヘッドが存在する場合
        if hasattr(model, 'four_class_head'):
            outputs = model(input_ids=input_ids)
            logits = model.four_class_head(outputs.last_hidden_state[:, -1, :])
            
            # ラベルをIDに変換
            label_to_id = {'ALLOW': 0, 'ESCALATION': 1, 'DENY': 2, 'REFUSE': 3}
            label_id = label_to_id.get(four_class_label, 0)
            
            loss = F.cross_entropy(logits, torch.tensor([label_id], device=logits.device))
            return loss
        
        return torch.tensor(0.0, device=input_ids.device)
    
    def _compute_ppo_loss(
        self,
        logprobs: torch.Tensor,
        old_logprobs: torch.Tensor,
        rewards: torch.Tensor,
        values: torch.Tensor,
        advantages: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """PPO損失を計算"""
        # ポリシー比
        ratio = torch.exp(logprobs - old_logprobs)
        
        # クリッピングされたポリシー損失
        clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon)
        policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
        
        # 価値関数損失
        value_loss = F.mse_loss(values, rewards)
        
        # エントロピーボーナス
        entropy_loss = -logprobs.mean()
        
        return policy_loss, value_loss, entropy_loss
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """四重推論型SO8T/thinking + PPO損失を計算"""
        # 1. 四重推論損失（Task/Safety/Policy/Final）
        quadruple_loss = self._compute_quadruple_loss(
            model,
            inputs["chosen_input_ids"],
            inputs["rejected_input_ids"],
            inputs["chosen_attention_mask"]
        )
        
        # 2. 四値分類損失（ALLOW/ESCALATION/DENY/REFUSE）
        four_class_loss = self._compute_four_class_loss(
            model,
            inputs["chosen_input_ids"],
            inputs.get("four_class_label", "ALLOW")
        )
        
        # 3. SO8T損失（PET + 直交性損失）
        so8t_loss = torch.tensor(0.0, device=inputs["chosen_input_ids"].device)
        
        # 直交性損失
        if hasattr(model, 'get_orthogonality_loss'):
            so8t_loss += model.get_orthogonality_loss()
        
        # PET損失
        if self.pet_regularization:
            # PET損失の計算（簡易版）
            pass
        
        # 4. PPO損失（ポリシー + 価値 + エントロピー）
        # 簡易版: ログ確率を計算
        chosen_outputs = model(input_ids=inputs["chosen_input_ids"])
        chosen_logprobs = F.log_softmax(chosen_outputs.logits, dim=-1)
        
        rejected_outputs = model(input_ids=inputs["rejected_input_ids"])
        rejected_logprobs = F.log_softmax(rejected_outputs.logits, dim=-1)
        
        # 報酬を計算（品質スコアを使用）
        rewards = torch.tensor([inputs.get("quality_score", 0.0)], device=chosen_logprobs.device)
        
        # 価値関数の出力（簡易版）
        values = chosen_logprobs.mean(dim=-1).mean(dim=-1)
        
        # アドバンテージを計算
        advantages = rewards - values.detach()
        
        # PPO損失を計算
        policy_loss, value_loss, entropy_loss = self._compute_ppo_loss(
            logprobs=chosen_logprobs[:, -1, :].mean(dim=-1),
            old_logprobs=chosen_logprobs.detach()[:, -1, :].mean(dim=-1),
            rewards=rewards,
            values=values,
            advantages=advantages
        )
        
        # 5. 統合損失
        total_loss = (
            quadruple_loss +
            four_class_loss +
            so8t_loss +
            policy_loss +
            self.value_coef * value_loss -
            self.entropy_coef * entropy_loss
        )
        
        # ログ出力
        if self.state.global_step % 10 == 0:
            logger.info(
                f"[PPO] Step {self.state.global_step}: "
                f"Quadruple Loss={quadruple_loss.item():.4f}, "
                f"Four-Class Loss={four_class_loss.item():.4f}, "
                f"SO8T Loss={so8t_loss.item():.4f}, "
                f"Policy Loss={policy_loss.item():.4f}, "
                f"Value Loss={value_loss.item():.4f}, "
                f"Total Loss={total_loss.item():.4f}"
            )
        
        return (total_loss, chosen_outputs) if return_outputs else total_loss


def main():
    parser = argparse.ArgumentParser(
        description="Train SO8T Quadruple PPO model"
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Config file path (YAML)"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Pairwise dataset path (JSONL format)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory"
    )
    parser.add_argument(
        "--auto-resume",
        action="store_true",
        help="Auto-resume from checkpoint if available"
    )
    
    args = parser.parse_args()
    
    # 設定ファイル読み込み
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*80)
    logger.info("SO8T Quadruple PPO Training")
    logger.info("="*80)
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Output: {output_dir}")
    
    # データセットファイルの確認
    if not args.dataset.exists():
        error_msg = f"Dataset file not found: {args.dataset}"
        logger.error(f"[ERROR] {error_msg}")
        raise FileNotFoundError(error_msg)
    
    dataset_size = args.dataset.stat().st_size
    logger.info(f"Dataset file size: {dataset_size:,} bytes")
    
    if dataset_size == 0:
        error_msg = f"Dataset file is empty: {args.dataset}"
        logger.error(f"[ERROR] {error_msg}")
        raise ValueError(error_msg)
    
    # トークナイザー読み込み
    model_path = config["model"]["base_model"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # モデル読み込み
    logger.info("Loading model...")
    load_in_8bit = config.get("quantization", {}).get("load_in_8bit", True)
    
    quantization_config = None
    if load_in_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=config.get("quantization", {}).get("llm_int8_threshold", 6.0)
        )
    
    # SO8TThinkingModelを読み込み
    from so8t.core.safety_aware_so8t import SafetyAwareSO8TConfig
    so8t_config_dict = config.get("so8t", {})
    so8t_config = SafetyAwareSO8TConfig(
        pet_lambda=so8t_config_dict.get("pet_lambda", 0.1),
        nu_orth=so8t_config_dict.get("orthogonal_reg", 1e-4),  # nu_orthが直交性制約の重み
        mu_norm=so8t_config_dict.get("norm_reg", 0.01),
        rho_iso=so8t_config_dict.get("isometry_reg", 0.01)
    )
    
    model = SO8TThinkingModel(
        base_model_name_or_path=model_path,
        so8t_config=so8t_config,
        use_quadruple_thinking=True
    )
    
    # QLoRA設定
    if config.get("qlora", {}).get("enabled", True):
        model = prepare_model_for_kbit_training(model)
        qlora_config = config.get("qlora", {})
        lora_config = LoraConfig(
            r=qlora_config.get("r", 64),
            lora_alpha=qlora_config.get("lora_alpha", 128),
            target_modules=qlora_config.get("target_modules", [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]),
            lora_dropout=qlora_config.get("lora_dropout", 0.05),
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        model = get_peft_model(model, lora_config)
        logger.info("[OK] QLoRA applied")
    
    # 重み凍結
    if config.get("model", {}).get("freeze_base_model", True):
        from scripts.training.train_borea_phi35_so8t_thinking import freeze_base_model_weights
        freeze_base_model_weights(model, config)
    
    # データセット読み込み
    train_dataset = QuadruplePairwiseDataset(
        data_path=args.dataset,
        tokenizer=tokenizer,
        max_length=config.get("data", {}).get("max_seq_length", 2048)
    )
    
    # チェックポイントからの再開
    resume_from_checkpoint = None
    if args.auto_resume:
        # 最新のチェックポイントを探す
        checkpoint_dirs = list(output_dir.glob("checkpoint-*"))
        if checkpoint_dirs:
            # チェックポイント番号でソート
            checkpoint_dirs.sort(key=lambda x: int(x.name.split("-")[-1]))
            resume_from_checkpoint = str(checkpoint_dirs[-1])
            logger.info(f"[RESUME] Resuming from checkpoint: {resume_from_checkpoint}")
        else:
            logger.info("[RESUME] No checkpoint found, starting from scratch")
    
    # トレーニング引数
    training_config = config.get("training", {})
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=training_config.get("num_train_epochs", 3),
        per_device_train_batch_size=training_config.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 16),
        learning_rate=training_config.get("learning_rate", 1.0e-5),
        weight_decay=training_config.get("weight_decay", 0.01),
        warmup_steps=training_config.get("warmup_steps", 100),
        logging_steps=training_config.get("logging_steps", 10),
        save_steps=training_config.get("save_steps", 500),
        fp16=training_config.get("fp16", True),
        gradient_checkpointing=training_config.get("gradient_checkpointing", True),
        optim=training_config.get("optim", "paged_adamw_8bit"),
        report_to=[],
    )
    
    # PET正則化
    pet_regularization = None
    if config.get("pet", {}).get("enabled", False):
        pet_config_dict = config.get("pet", {})
        pet_config = PETConfig(
            lambda_exploration=pet_config_dict.get("lambda_exploration", 0.01),
            lambda_transition=pet_config_dict.get("lambda_transition", 0.05),
            lambda_stabilization=pet_config_dict.get("lambda_stabilization", 0.1),
            exploration_ratio=pet_config_dict.get("exploration_ratio", 0.2),
            transition_ratio=pet_config_dict.get("transition_ratio", 0.4),
            stabilization_ratio=pet_config_dict.get("stabilization_ratio", 0.4)
        )
        total_steps = (len(train_dataset) // (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps)) * training_args.num_train_epochs
        if PETRegularization is not None:
            pet_regularization = PETRegularization(config=pet_config, total_steps=total_steps)
        else:
            logger.warning("[WARNING] PETRegularization not available, skipping PET regularization")
            pet_regularization = None
    
    # PPOトレーナー
    reward_config = config.get("reward_learning", {})
    trainer = SO8TQuadruplePPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        clip_epsilon=reward_config.get("clip_epsilon", 0.2),
        value_coef=reward_config.get("value_coef", 0.5),
        entropy_coef=reward_config.get("entropy_coef", 0.01),
        use_quadruple_thinking=True,
        use_four_class_classification=True,
        freeze_base_model=config.get("model", {}).get("freeze_base_model", True),
        pet_regularization=pet_regularization
    )
    
    # 学習実行
    logger.info("Starting SO8T Quadruple PPO training...")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    
    # 最終モデル保存
    final_model_dir = output_dir / "final_model"
    trainer.save_model(str(final_model_dir))
    tokenizer.save_pretrained(str(final_model_dir))
    
    logger.info(f"[SUCCESS] SO8T Quadruple PPO training completed. Model saved to {final_model_dir}")


if __name__ == "__main__":
    main()


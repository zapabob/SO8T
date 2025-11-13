#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Borea-Phi-3.5-mini-Instruct-Jp SO8T/thinking学習スクリプト

選択的SO8T統合 + PET正則化 + /think形式データセットで学習
"""

import os
import sys
import json
import time
import signal
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
    Phi3Config
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

# SO8T統合モデルをインポート
try:
    sys.path.insert(0, str(PROJECT_ROOT / "models" / "Borea-Phi-3.5-mini-Instruct-Jp"))
    from modeling_phi3_so8t import (
        SO8TPhi3Model
    )
    # SO8TPhi3ForCausalLMが存在しない場合は、後で作成
    try:
        from modeling_phi3_so8t import SO8TPhi3ForCausalLM
    except ImportError:
        SO8TPhi3ForCausalLM = None
except ImportError as e:
    # loggerは後で定義されるので、ここではprintを使用
    print(f"[WARNING] Failed to import SO8T models: {e}")
    SO8TPhi3Model = None
    SO8TPhi3ForCausalLM = None

# PET正則化をインポート
from pet_regularization import PETRegularization, PETConfig

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/train_borea_phi35_so8t_thinking.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ThinkingSFTDataset(Dataset):
    """/think形式SFTデータセット"""
    
    def __init__(
        self,
        data_path: Path,
        tokenizer,
        max_length: int = 2048
    ):
        """
        Args:
            data_path: JSONLファイルパス（/think形式）
            tokenizer: トークナイザー
            max_length: 最大シーケンス長
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        
        logger.info(f"Loading /think format dataset from {data_path}...")
        
        with open(data_path, 'r', encoding='utf-8') as f:
            for line_no, line in enumerate(f, 1):
                try:
                    sample = json.loads(line.strip())
                    output = sample.get("output", "")
                    
                    # outputが既にPhi-3.5チャットテンプレート形式であることを確認
                    if "<|system|>" in output and "<|assistant|>" in output:
                        self.samples.append({
                            "text": output,
                            "instruction": sample.get("instruction", ""),
                            "input": sample.get("input", "")
                        })
                    else:
                        logger.warning(f"Line {line_no}: Invalid format, skipping")
                        continue
                        
                except json.JSONDecodeError as e:
                    logger.warning(f"Line {line_no}: JSON decode error: {e}")
                    continue
        
        logger.info(f"[OK] Loaded {len(self.samples):,} training samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        text = sample["text"]
        
        # トークナイズ
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoded["input_ids"].squeeze(),
            "attention_mask": encoded["attention_mask"].squeeze(),
            "labels": encoded["input_ids"].squeeze()
        }


class SO8TPETTrainer(Trainer):
    """SO8T + PET統合Trainer"""
    
    def __init__(
        self,
        *args,
        pet_regularization: Optional[PETRegularization] = None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.pet_regularization = pet_regularization
        self.hidden_states_history = []
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        損失計算（PET統合）
        """
        # hidden_statesを取得するためにoutput_hidden_states=Trueを設定
        inputs_with_hidden = {**inputs, "output_hidden_states": True}
        
        # 標準損失
        outputs = model(**inputs_with_hidden)
        loss = outputs.loss
        
        # PET損失を追加
        if self.pet_regularization is not None and hasattr(outputs, 'hidden_states') and outputs.hidden_states:
            # hidden_statesが利用可能な場合
            hidden_states = outputs.hidden_states[-1]  # 最後のレイヤーの隠れ状態
            step = self.state.global_step
            
            pet_loss, pet_info = self.pet_regularization.compute_pet_loss(
                hidden_states=hidden_states,
                step=step,
                mask=inputs.get("attention_mask")
            )
            
            loss = loss + pet_loss
            
            # ログ出力（定期的に）
            if self.state.global_step % 100 == 0:
                logger.info(f"[PET] Step {step}: Loss={pet_loss.item():.6e}, "
                          f"Phase={pet_info.get('phase', 'unknown')}, "
                          f"Lambda={pet_info.get('lambda', 0.0):.4f}")
        
        return (loss, outputs) if return_outputs else loss


def load_model_with_so8t(
    model_path: str,
    so8t_layer_indices: Optional[List[int]] = None,
    load_in_8bit: bool = True
) -> Tuple[Any, Any]:
    """
    SO8T統合モデルを読み込み
    
    Args:
        model_path: モデルパス
        so8t_layer_indices: SO8T適用レイヤーインデックス
        load_in_8bit: 8bit量子化を使用するか
    
    Returns:
        (model, tokenizer)
    """
    logger.info(f"Loading model from {model_path}...")
    
    # トークナイザー読み込み
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 設定読み込み
    config = Phi3Config.from_pretrained(model_path, trust_remote_code=True)
    
    # 量子化設定
    quantization_config = None
    if load_in_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0
        )
    
    # SO8T統合モデルを読み込み
    if SO8TPhi3Model is not None and so8t_layer_indices is not None:
        try:
            # SO8TPhi3Modelを使用してモデルを構築
            # 注意: SO8TPhi3ForCausalLMが存在しない場合は、標準モデルを読み込んでSO8Tを適用
            if SO8TPhi3ForCausalLM is not None:
                # 標準モデルを読み込んでからSO8Tモデルに置き換え
                base_model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    quantization_config=quantization_config,
                    device_map="auto",
                    trust_remote_code=True,
                    torch_dtype=torch.float16
                )
                
                # SO8TForCausalLMを作成
                so8t_model = SO8TPhi3ForCausalLM(
                    config=config,
                    so8t_layer_indices=so8t_layer_indices
                )
                
                # 重みをコピー（embed_tokensとlm_head）
                so8t_model.model.embed_tokens.weight.data = base_model.model.embed_tokens.weight.data
                so8t_model.lm_head.weight.data = base_model.lm_head.weight.data
                
                # レイヤーの重みをコピー（SO8T適用レイヤーと標準レイヤーを適切に処理）
                for i, (so8t_layer, base_layer) in enumerate(zip(so8t_model.model.layers, base_model.model.layers)):
                    # アテンション重みをコピー
                    if hasattr(so8t_layer, 'self_attn') and hasattr(base_layer, 'self_attn'):
                        for param_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                            if hasattr(so8t_layer.self_attn, param_name) and hasattr(base_layer.self_attn, param_name):
                                so8t_param = getattr(so8t_layer.self_attn, param_name)
                                base_param = getattr(base_layer.self_attn, param_name)
                                
                                # 重みをコピー
                                if so8t_param.weight.shape == base_param.weight.shape:
                                    so8t_param.weight.data.copy_(base_param.weight.data)
                                else:
                                    logger.warning(f"Layer {i}.self_attn.{param_name}: shape mismatch "
                                                f"({so8t_param.weight.shape} vs {base_param.weight.shape})")
                                
                                # バイアスをコピー（存在する場合）
                                if hasattr(so8t_param, 'bias') and hasattr(base_param, 'bias'):
                                    if so8t_param.bias is not None and base_param.bias is not None:
                                        if so8t_param.bias.shape == base_param.bias.shape:
                                            so8t_param.bias.data.copy_(base_param.bias.data)
                    
                    # MLP重みをコピー
                    if hasattr(so8t_layer, 'mlp') and hasattr(base_layer, 'mlp'):
                        for param_name in ['gate_proj', 'up_proj', 'down_proj']:
                            if hasattr(so8t_layer.mlp, param_name) and hasattr(base_layer.mlp, param_name):
                                so8t_param = getattr(so8t_layer.mlp, param_name)
                                base_param = getattr(base_layer.mlp, param_name)
                                
                                # 重みをコピー
                                if so8t_param.weight.shape == base_param.weight.shape:
                                    so8t_param.weight.data.copy_(base_param.weight.data)
                                else:
                                    logger.warning(f"Layer {i}.mlp.{param_name}: shape mismatch "
                                                f"({so8t_param.weight.shape} vs {base_param.weight.shape})")
                                
                                # バイアスをコピー（存在する場合）
                                if hasattr(so8t_param, 'bias') and hasattr(base_param, 'bias'):
                                    if so8t_param.bias is not None and base_param.bias is not None:
                                        if so8t_param.bias.shape == base_param.bias.shape:
                                            so8t_param.bias.data.copy_(base_param.bias.data)
                    
                    # RMSNorm重みをコピー
                    if hasattr(so8t_layer, 'input_layernorm') and hasattr(base_layer, 'input_layernorm'):
                        if hasattr(so8t_layer.input_layernorm, 'weight') and hasattr(base_layer.input_layernorm, 'weight'):
                            if so8t_layer.input_layernorm.weight.shape == base_layer.input_layernorm.weight.shape:
                                so8t_layer.input_layernorm.weight.data.copy_(base_layer.input_layernorm.weight.data)
                    
                    if hasattr(so8t_layer, 'post_attention_layernorm') and hasattr(base_layer, 'post_attention_layernorm'):
                        if hasattr(so8t_layer.post_attention_layernorm, 'weight') and hasattr(base_layer.post_attention_layernorm, 'weight'):
                            if so8t_layer.post_attention_layernorm.weight.shape == base_layer.post_attention_layernorm.weight.shape:
                                so8t_layer.post_attention_layernorm.weight.data.copy_(base_layer.post_attention_layernorm.weight.data)
                
                # RMSNorm重みをコピー（モデル全体）
                if hasattr(so8t_model.model, 'norm') and hasattr(base_model.model, 'norm'):
                    if hasattr(so8t_model.model.norm, 'weight') and hasattr(base_model.model.norm, 'weight'):
                        if so8t_model.model.norm.weight.shape == base_model.model.norm.weight.shape:
                            so8t_model.model.norm.weight.data.copy_(base_model.model.norm.weight.data)
                
                logger.info("[OK] Copied weights from base model to SO8T model")
                
                # 量子化を再適用（必要な場合）
                if quantization_config:
                    from peft import prepare_model_for_kbit_training
                    so8t_model = prepare_model_for_kbit_training(so8t_model)
                
                model = so8t_model
            else:
                # 標準モデルを読み込んで、SO8Tモデルに置き換え
                logger.info("SO8TPhi3ForCausalLM not available, using standard model with SO8T integration")
                base_model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    quantization_config=quantization_config,
                    device_map="auto",
                    trust_remote_code=True,
                    torch_dtype=torch.float16
                )
                
                # SO8Tモデルに置き換え（model.modelをSO8TPhi3Modelに置き換え）
                if hasattr(base_model, 'model'):
                    so8t_model = SO8TPhi3Model(config, so8t_layer_indices=so8t_layer_indices)
                    # SO8Tモデルへの重み転送: 標準モデルの重みを新SO8Tモデルに適切にコピー
                    def transfer_weights_to_so8t_model(so8t_model, base_model):
                        """
                        標準モデル (base_model.model) の重みを so8t_model に転送します。
                        レイヤーごとに so8t_layer_indices で指定されたレイヤーを対象に転送します。
                        """
                        # 基本ブロックの名前またはアクセス
                        if not hasattr(base_model, 'model'):
                            logger.warning("base_modelに'model'属性がありません。重み転送をスキップします。")
                            return

                        base_transformer = base_model.model
                        so8t_transformer = so8t_model

                        # レイヤーリストの取得
                        base_layers = getattr(base_transformer, 'layers', None)
                        so8t_layers = getattr(so8t_transformer, 'layers', None)
                        if base_layers is None or so8t_layers is None:
                            logger.warning("レイヤーリストが見つかりません。重み転送をスキップします。")
                            return

                        # 各so8t_layer_indicesで指定されたレイヤーについて重み転送
                        for idx, base_layer in enumerate(base_layers):
                            if idx in so8t_layer_indices:
                                so8t_layer = so8t_layers[idx]

                                # Attention部分の重み転送
                                for att_name in ["self_attn", "attn"]:
                                    if hasattr(so8t_layer, att_name) and hasattr(base_layer, att_name):
                                        so8t_att = getattr(so8t_layer, att_name)
                                        base_att = getattr(base_layer, att_name)
                                        for param_name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                                            if hasattr(so8t_att, param_name) and hasattr(base_att, param_name):
                                                so8t_param = getattr(so8t_att, param_name)
                                                base_param = getattr(base_att, param_name)
                                                if hasattr(so8t_param, 'weight') and hasattr(base_param, 'weight'):
                                                    if so8t_param.weight.shape == base_param.weight.shape:
                                                        so8t_param.weight.data.copy_(base_param.weight.data)
                    # 本番環境: 標準モデル内のmodel部分をSO8Tモデルで置き換え
                    base_model.model = so8t_model
                    model = base_model
                    # 必要であれば、この部分で重み転送/変換などをさらに実装
        except Exception as e:
            logger.warning(f"Failed to load SO8T model: {e}")
            logger.info("Falling back to standard model loading...")
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16
            )
    else:
        # 標準モデルを読み込み
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
    
    logger.info("[OK] Model and tokenizer loaded")
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(
        description="Train Borea-Phi-3.5 with SO8T/thinking"
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Config file path (YAML)"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="AXCXEPT/Borea-Phi-3.5-mini-Instruct-Jp",
        help="Base model path"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Training dataset path (JSONL, /think format)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory"
    )
    
    args = parser.parse_args()
    
    # 設定ファイル読み込み
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*80)
    logger.info("Borea-Phi-3.5 SO8T/thinking Training")
    logger.info("="*80)
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Output: {output_dir}")
    
    # モデルとトークナイザー読み込み
    so8t_config = config.get("so8t", {})
    so8t_layer_indices = so8t_config.get("layer_indices", None)
    model, tokenizer = load_model_with_so8t(
        model_path=args.model_path,
        so8t_layer_indices=so8t_layer_indices,
        load_in_8bit=config.get("quantization", {}).get("load_in_8bit", True)
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
    
    # データセット読み込み
    train_dataset = ThinkingSFTDataset(
        data_path=args.dataset,
        tokenizer=tokenizer,
        max_length=config.get("data", {}).get("max_seq_length", 2048)
    )
    
    # PET正則化設定
    pet_regularization = None
    if config.get("pet", {}).get("enabled", True):
        pet_config_dict = config.get("pet", {})
        pet_config = PETConfig(
            lambda_exploration=pet_config_dict.get("lambda_exploration", 0.01),
            lambda_transition=pet_config_dict.get("lambda_transition", 0.05),
            lambda_stabilization=pet_config_dict.get("lambda_stabilization", 0.1),
            exploration_ratio=pet_config_dict.get("exploration_ratio", 0.2),
            transition_ratio=pet_config_dict.get("transition_ratio", 0.4),
            stabilization_ratio=pet_config_dict.get("stabilization_ratio", 0.4)
        )
        
        # 総ステップ数を計算
        training_config = config.get("training", {})
        num_epochs = training_config.get("num_train_epochs", 3)
        batch_size = training_config.get("per_device_train_batch_size", 1)
        gradient_accumulation = training_config.get("gradient_accumulation_steps", 16)
        total_steps = (len(train_dataset) // (batch_size * gradient_accumulation)) * num_epochs
        
        pet_regularization = PETRegularization(
            config=pet_config,
            total_steps=total_steps
        )
        logger.info(f"[OK] PET regularization initialized (total_steps={total_steps})")
    
    # データコレクター
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # トレーニング引数
    training_config = config.get("training", {})
    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=training_config.get("num_train_epochs", 3),
        per_device_train_batch_size=training_config.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 16),
        learning_rate=training_config.get("learning_rate", 2.0e-4),
        weight_decay=training_config.get("weight_decay", 0.01),
        warmup_ratio=training_config.get("warmup_ratio", 0.1),
        lr_scheduler_type=training_config.get("lr_scheduler_type", "cosine"),
        logging_steps=training_config.get("logging_steps", 10),
        save_steps=training_config.get("save_steps", 500),
        save_total_limit=training_config.get("save_total_limit", 5),
        fp16=training_config.get("fp16", True),
        bf16=training_config.get("bf16", False),
        gradient_checkpointing=training_config.get("gradient_checkpointing", True),
        optim=training_config.get("optim", "paged_adamw_8bit"),
        report_to=[],
    )
    
    # トレーナー
    trainer = SO8TPETTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        pet_regularization=pet_regularization
    )
    
    # 学習実行
    logger.info("Starting training...")
    trainer.train()
    
    # 最終モデル保存
    final_model_dir = output_dir / "final_model"
    trainer.save_model(str(final_model_dir))
    tokenizer.save_pretrained(str(final_model_dir))
    
    logger.info(f"[SUCCESS] Training completed. Model saved to {final_model_dir}")


if __name__ == "__main__":
    main()


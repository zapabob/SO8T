"""
QLoRA 8bit学習実装
SO8T回転ゲート + PET損失対応
"""

import torch
import torch.nn as nn
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoTokenizer,
    AutoProcessor,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from bitsandbytes import BitsAndBytesConfig
from typing import Dict, Any, Optional, List, Union
import json
import os
from tqdm import tqdm
import logging

from ..modules.rotation_gate import SO8TRotationGate
from ..losses.pet import PETLoss, CombinedLoss


class SO8TQLoRATrainer:
    """
    SO8T回転ゲート + PET損失対応QLoRA学習器
    """
    
    def __init__(
        self,
        model_path: str,
        config_path: str,
        output_dir: str = "./outputs",
        device_map: str = "auto"
    ):
        """
        Args:
            model_path: モデルパス
            config_path: 設定ファイルパス
            output_dir: 出力ディレクトリ
            device_map: デバイスマップ
        """
        self.model_path = model_path
        self.config_path = config_path
        self.output_dir = output_dir
        self.device_map = device_map
        
        # 設定を読み込み
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        # ログ設定
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # モデルとトークナイザーを初期化
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.trainer = None
        
        # SO8T回転ゲート
        self.rotation_gate = None
        self.pet_loss = None
        
    def setup_model(self) -> None:
        """モデルとトークナイザーをセットアップ"""
        self.logger.info("モデルとトークナイザーをセットアップ中...")
        
        # 量子化設定
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            llm_int8_enable_fp32_cpu_offload=True
        )
        
        # モデル読み込み
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_path,
            quantization_config=quantization_config,
            device_map=self.device_map,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2" if torch.cuda.is_available() else None
        )
        
        # プロセッサ読み込み
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.tokenizer = self.processor.tokenizer
        
        # SO8T回転ゲートを追加
        if self.config.get("rotation_gate_enabled", True):
            self.rotation_gate = SO8TRotationGate(
                hidden_size=self.model.config.hidden_size,
                learnable=True
            )
            self.model.rotation_gate = self.rotation_gate
        
        # PET損失を設定
        if self.config.get("pet_loss_enabled", True):
            pet_config = self.config.get("pet_lambda_schedule", {})
            self.pet_loss = PETLoss(
                max_lambda=pet_config.get("max_lambda", 0.1),
                warmup_steps=pet_config.get("warmup_steps", 100),
                main_steps=pet_config.get("main_steps", 1000),
                anneal_steps=pet_config.get("anneal_steps", 200)
            )
        
        self.logger.info("モデルセットアップ完了")
    
    def setup_lora(self) -> None:
        """LoRA設定をセットアップ"""
        self.logger.info("LoRA設定をセットアップ中...")
        
        # LoRA設定
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.get("lora_rank", 64),
            lora_alpha=self.config.get("lora_alpha", 128),
            lora_dropout=self.config.get("lora_dropout", 0.1),
            target_modules=self.config.get("target_modules", [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]),
            bias="none"
        )
        
        # モデルにLoRAを適用
        self.model = get_peft_model(self.model, lora_config)
        
        # 回転ゲートもLoRA対象に追加
        if self.rotation_gate is not None:
            # 回転ゲートのパラメータをLoRA対象に追加
            for name, param in self.rotation_gate.named_parameters():
                if param.requires_grad:
                    param.data = param.data.to(self.model.device)
        
        self.logger.info("LoRA設定完了")
    
    def create_custom_trainer(self, train_dataset, eval_dataset=None):
        """カスタムTrainerを作成"""
        
        class SO8TTrainer(Trainer):
            def __init__(self, pet_loss, rotation_gate, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.pet_loss = pet_loss
                self.rotation_gate = rotation_gate
                self.step_count = 0
            
            def compute_loss(self, model, inputs, return_outputs=False):
                """カスタム損失計算"""
                # 標準の前向き計算
                outputs = model(**inputs)
                logits = outputs.logits
                labels = inputs["labels"]
                
                # 標準のクロスエントロピー損失
                loss_fct = nn.CrossEntropyLoss()
                task_loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
                
                # PET損失を追加
                if self.pet_loss is not None and self.rotation_gate is not None:
                    # 隠れ状態を取得（最後のレイヤー）
                    hidden_states = outputs.hidden_states[-1] if hasattr(outputs, 'hidden_states') else None
                    
                    if hidden_states is not None:
                        pet_loss_val = self.pet_loss(hidden_states, self.step_count)
                        task_loss = task_loss + pet_loss_val
                
                self.step_count += 1
                
                return (task_loss, outputs) if return_outputs else task_loss
        
        # 学習引数
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=self.config.get("batch_size", 1),
            gradient_accumulation_steps=self.config.get("gradient_accumulation_steps", 8),
            num_train_epochs=self.config.get("num_epochs", 3),
            learning_rate=self.config.get("learning_rate", 2e-4),
            warmup_steps=self.config.get("warmup_steps", 100),
            max_grad_norm=self.config.get("max_grad_norm", 1.0),
            weight_decay=self.config.get("weight_decay", 0.01),
            logging_steps=10,
            save_steps=500,
            eval_steps=500,
            evaluation_strategy="steps" if eval_dataset is not None else "no",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=torch.cuda.is_available(),
            dataloader_pin_memory=True,
            remove_unused_columns=False,
            report_to="tensorboard"
        )
        
        # データコレーター
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # カスタムTrainerを作成
        self.trainer = SO8TTrainer(
            pet_loss=self.pet_loss,
            rotation_gate=self.rotation_gate,
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer
        )
    
    def train(self, train_dataset, eval_dataset=None) -> None:
        """学習を実行"""
        self.logger.info("学習を開始...")
        
        # モデルセットアップ
        self.setup_model()
        
        # LoRAセットアップ
        self.setup_lora()
        
        # カスタムTrainer作成
        self.create_custom_trainer(train_dataset, eval_dataset)
        
        # 学習実行
        self.trainer.train()
        
        # モデル保存
        self.save_model()
        
        self.logger.info("学習完了")
    
    def save_model(self) -> None:
        """モデルを保存"""
        self.logger.info("モデルを保存中...")
        
        # 出力ディレクトリ作成
        os.makedirs(self.output_dir, exist_ok=True)
        
        # モデル保存
        self.trainer.save_model()
        
        # 回転ゲートの回転行列を保存
        if self.rotation_gate is not None:
            rotation_matrices = self.rotation_gate.get_rotation_matrices()
            torch.save(rotation_matrices, os.path.join(self.output_dir, "rotation_matrices.pt"))
        
        # 設定保存
        with open(os.path.join(self.output_dir, "config.json"), 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"モデルを {self.output_dir} に保存しました")
    
    def evaluate(self, eval_dataset) -> Dict[str, float]:
        """評価を実行"""
        self.logger.info("評価を実行中...")
        
        if self.trainer is None:
            raise ValueError("Trainerが初期化されていません。先にtrain()を実行してください。")
        
        # 評価実行
        eval_results = self.trainer.evaluate(eval_dataset)
        
        self.logger.info(f"評価結果: {eval_results}")
        return eval_results
    
    def generate(
        self,
        input_text: str,
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True
    ) -> str:
        """テキスト生成"""
        self.logger.info("テキスト生成中...")
        
        if self.model is None:
            raise ValueError("モデルが初期化されていません。")
        
        # 入力エンコード
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        # デバイスに移動
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # 生成
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # デコード
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return generated_text

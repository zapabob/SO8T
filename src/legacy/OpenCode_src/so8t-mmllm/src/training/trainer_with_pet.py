"""
PET損失統合学習器
SO8T回転ゲート + PET正則化 + QLoRA学習の統合実装
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
from typing import Dict, Any, Optional, List, Union, Tuple
import json
import os
from tqdm import tqdm
import logging
import numpy as np

from ..modules.rotation_gate import SO8TRotationGate
from ..losses.pet import PETLoss, CombinedLoss, create_pet_schedule
from ..modules.qwen2vl_wrapper import create_so8t_qwen2vl_model
from ..io.ocr_summary import OCRSummaryProcessor
from ..audit.sqlite_logger import SQLiteAuditLogger


class SO8TPETTrainer(Trainer):
    """
    SO8T回転ゲート + PET損失対応カスタムTrainer
    """
    
    def __init__(
        self,
        pet_loss: Optional[PETLoss] = None,
        rotation_gate: Optional[SO8TRotationGate] = None,
        audit_logger: Optional[SQLiteAuditLogger] = None,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        
        self.pet_loss = pet_loss
        self.rotation_gate = rotation_gate
        self.audit_logger = audit_logger
        self.step_count = 0
        
        # 学習統計
        self.training_stats = {
            "total_steps": 0,
            "pet_loss_values": [],
            "rotation_norms": [],
            "task_loss_values": [],
            "combined_loss_values": []
        }
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """カスタム損失計算（PET損失統合）"""
        # 標準の前向き計算
        outputs = model(**inputs)
        logits = outputs.logits
        labels = inputs["labels"]
        
        # 標準のクロスエントロピー損失
        loss_fct = nn.CrossEntropyLoss()
        task_loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        # PET損失を追加
        pet_loss_val = 0.0
        if self.pet_loss is not None:
            # 隠れ状態を取得
            hidden_states = None
            if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                # 最後のレイヤーの隠れ状態を使用
                hidden_states = outputs.hidden_states[-1]
            elif hasattr(outputs, 'last_hidden_state'):
                hidden_states = outputs.last_hidden_state
            
            if hidden_states is not None:
                pet_loss_val = self.pet_loss(hidden_states, self.step_count)
        
        # 合計損失
        total_loss = task_loss + pet_loss_val
        
        # 統計を記録
        self.training_stats["total_steps"] = self.step_count
        self.training_stats["task_loss_values"].append(task_loss.item())
        self.training_stats["pet_loss_values"].append(pet_loss_val.item() if isinstance(pet_loss_val, torch.Tensor) else pet_loss_val)
        self.training_stats["combined_loss_values"].append(total_loss.item())
        
        # 回転ゲートの統計を記録
        if self.rotation_gate is not None:
            rotation_norm = torch.norm(self.rotation_gate.theta).item()
            self.training_stats["rotation_norms"].append(rotation_norm)
        
        # 監査ログに記録（定期的に）
        if self.audit_logger is not None and self.step_count % 100 == 0:
            self.audit_logger.log_audit(
                change_type="training_step",
                change_description=f"学習ステップ {self.step_count}",
                change_data={
                    "step": self.step_count,
                    "task_loss": task_loss.item(),
                    "pet_loss": pet_loss_val.item() if isinstance(pet_loss_val, torch.Tensor) else pet_loss_val,
                    "total_loss": total_loss.item(),
                    "rotation_norm": rotation_norm if self.rotation_gate is not None else None
                }
            )
        
        self.step_count += 1
        
        return (total_loss, outputs) if return_outputs else total_loss
    
    def get_training_stats(self) -> Dict[str, Any]:
        """学習統計を取得"""
        return self.training_stats.copy()
    
    def reset_stats(self):
        """統計をリセット"""
        self.training_stats = {
            "total_steps": 0,
            "pet_loss_values": [],
            "rotation_norms": [],
            "task_loss_values": [],
            "combined_loss_values": []
        }


class SO8TIntegratedTrainer:
    """
    SO8T回転ゲート + PET損失 + OCR要約 + SQLite監査の統合学習器
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
        
        # コンポーネント初期化
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.trainer = None
        self.rotation_gate = None
        self.pet_loss = None
        self.ocr_processor = None
        self.audit_logger = None
    
    def setup_components(self) -> None:
        """全コンポーネントをセットアップ"""
        self.logger.info("コンポーネントをセットアップ中...")
        
        # 1. モデルとトークナイザー
        self._setup_model()
        
        # 2. LoRA設定
        self._setup_lora()
        
        # 3. SO8T回転ゲート
        self._setup_rotation_gate()
        
        # 4. PET損失
        self._setup_pet_loss()
        
        # 5. OCR要約プロセッサ
        self._setup_ocr_processor()
        
        # 6. SQLite監査ロガー
        self._setup_audit_logger()
        
        self.logger.info("コンポーネントセットアップ完了")
    
    def _setup_model(self) -> None:
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
    
    def _setup_lora(self) -> None:
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
    
    def _setup_rotation_gate(self) -> None:
        """SO8T回転ゲートをセットアップ"""
        if self.config.get("rotation_gate_enabled", True):
            self.logger.info("SO8T回転ゲートをセットアップ中...")
            
            self.rotation_gate = SO8TRotationGate(
                hidden_size=self.model.config.hidden_size,
                learnable=True,
                init_scale=self.config.get("rotation_init_scale", 0.1)
            )
            
            # モデルに回転ゲートを追加
            self.model.rotation_gate = self.rotation_gate
    
    def _setup_pet_loss(self) -> None:
        """PET損失をセットアップ"""
        if self.config.get("pet_loss_enabled", True):
            self.logger.info("PET損失をセットアップ中...")
            
            pet_config = self.config.get("pet_lambda_schedule", {})
            self.pet_loss = PETLoss(
                max_lambda=pet_config.get("max_lambda", 0.1),
                warmup_steps=pet_config.get("warmup_steps", 100),
                main_steps=pet_config.get("main_steps", 1000),
                anneal_steps=pet_config.get("anneal_steps", 200),
                huber_delta=pet_config.get("huber_delta", 0.0)
            )
    
    def _setup_ocr_processor(self) -> None:
        """OCR要約プロセッサをセットアップ"""
        self.logger.info("OCR要約プロセッサをセットアップ中...")
        
        self.ocr_processor = OCRSummaryProcessor(
            tesseract_config="--oem 3 --psm 6",
            languages="jpn+eng",
            min_confidence=30.0
        )
    
    def _setup_audit_logger(self) -> None:
        """SQLite監査ロガーをセットアップ"""
        self.logger.info("SQLite監査ロガーをセットアップ中...")
        
        self.audit_logger = SQLiteAuditLogger(
            db_path=os.path.join(self.output_dir, "audit.db"),
            synchronous="FULL",
            journal_mode="WAL"
        )
        
        # 学習開始をログに記録
        self.audit_logger.log_audit(
            change_type="training_start",
            change_description="SO8T×マルチモーダルLLM学習開始",
            change_data={
                "model_path": self.model_path,
                "config_path": self.config_path,
                "output_dir": self.output_dir,
                "rotation_enabled": self.config.get("rotation_gate_enabled", True),
                "pet_enabled": self.config.get("pet_loss_enabled", True)
            }
        )
    
    def create_custom_trainer(self, train_dataset, eval_dataset=None):
        """カスタムTrainerを作成"""
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
        self.trainer = SO8TPETTrainer(
            pet_loss=self.pet_loss,
            rotation_gate=self.rotation_gate,
            audit_logger=self.audit_logger,
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
        
        # 全コンポーネントをセットアップ
        self.setup_components()
        
        # カスタムTrainer作成
        self.create_custom_trainer(train_dataset, eval_dataset)
        
        # 学習実行
        self.trainer.train()
        
        # モデル保存
        self.save_model()
        
        # 学習完了をログに記録
        if self.audit_logger is not None:
            self.audit_logger.log_audit(
                change_type="training_complete",
                change_description="SO8T×マルチモーダルLLM学習完了",
                change_data={
                    "final_stats": self.trainer.get_training_stats()
                }
            )
        
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
        
        # 学習統計を保存
        stats = self.trainer.get_training_stats()
        with open(os.path.join(self.output_dir, "training_stats.json"), 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
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
    
    def generate_with_ocr(
        self,
        input_text: str,
        image_paths: Optional[List[str]] = None,
        max_length: int = 512,
        temperature: float = 0.7
    ) -> str:
        """OCR要約付きテキスト生成"""
        self.logger.info("OCR要約付きテキスト生成中...")
        
        if self.model is None:
            raise ValueError("モデルが初期化されていません。")
        
        # OCR要約を生成
        ocr_summaries = []
        if image_paths:
            for image_path in image_paths:
                try:
                    summary = self.ocr_processor.process_image(image_path)
                    ocr_summaries.append(summary)
                except Exception as e:
                    self.logger.warning(f"OCR処理エラー: {e}")
        
        # プロンプトを作成
        if ocr_summaries:
            prompt = self.ocr_processor.create_prompt_with_ocr(input_text, ocr_summaries)
        else:
            prompt = input_text
        
        # 入力エンコード
        inputs = self.tokenizer(
            prompt,
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
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # デコード
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 監査ログに記録
        if self.audit_logger is not None:
            self.audit_logger.log_decision(
                input_text=prompt,
                decision="ALLOW",  # 生成成功
                confidence=0.9,
                reasoning="OCR要約付きテキスト生成完了",
                meta={
                    "image_count": len(image_paths) if image_paths else 0,
                    "ocr_summaries": ocr_summaries
                }
            )
        
        return generated_text

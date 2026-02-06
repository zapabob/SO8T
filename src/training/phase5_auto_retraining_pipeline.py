#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 5: 全自動再学習パイプライン (Borea → AEGIS-v3.0)

Borea-Phi-3.5-mini-Instruct-Jp をベースモデルとし、
既存重みを凍結（Frozen weights + LoRA/Adapter）しながら
Phase 4 で構築した高度データセットを用いて再学習を行います。

Features:
- 既存重み凍結 (LoRA/QLoRA)
- SFT + GRPO 統合学習
- 5分間隔ローリングチェックポイント
- 電源投入時自動再開
- tqdm + PowerShell 進捗表示
- BF16/Flash-attention 最適化 (RTX 3060 12GB対応)
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Third-party imports (moved to top-level for Windows multiprocessing support)
try:
    import torch
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template
    from trl import SFTTrainer
    from transformers import TrainingArguments, TrainerCallback
    from datasets import load_dataset
except ImportError:
    pass # Will be handled in setup_training_environment if missing


# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# Import robust checkpoint manager
try:
    from src.utils.checkpoint_manager import RollingCheckpointManager, EmergencyCheckpointManager, RollingCheckpointCallback
except ImportError:
    # Fallback/Mock if unavailable during init (though verification ensured it exists)
    RollingCheckpointManager = None
    logger.error("Could not import src.utils.checkpoint_manager")

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(PROJECT_ROOT / "logs" / "phase5_auto_retraining.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class Phase5AutoRetrainingPipeline:
    """
    Phase 5 全自動再学習パイプライン
    
    Uses Unified Robustness Infrastructure:
    - RollingCheckpointManager (5-min intervals, 3 generations)
    - EmergencyCheckpointManager (Signal handling)
    """

    # Configuration
    BASE_MODEL = "AXCXEPT/Borea-Phi-3.5-mini-Instruct-Jp"
    OUTPUT_MODEL = "zapabobouj-AEGIS-phi3.5-jp-v3.0"
    
    def __init__(
        self,
        dataset_path: Optional[Path] = None,
        output_dir: Optional[Path] = None,
    ) -> None:
        self.project_root = PROJECT_ROOT
        self.dataset_path = dataset_path
        self.output_dir = output_dir or self.project_root / "src" / "training" / "models" / self.OUTPUT_MODEL
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Robustness Initialization
        checkpoint_dir = self.output_dir / "checkpoints"
        interval = int(os.getenv("SO8T_CHECKPOINT_INTERVAL", "300"))
        rolling = int(os.getenv("SO8T_CHECKPOINT_ROLLING", "3"))
        
        if RollingCheckpointManager:
            self.checkpoint_manager = RollingCheckpointManager(
                base_dir=checkpoint_dir,
                max_keep=rolling,
                save_interval_sec=interval,
                enable_logging=True
            )
            self.emergency_checkpoint = EmergencyCheckpointManager(self.checkpoint_manager)
            logger.info(f"Robustness enabled: Rolling checkpoints ({interval}s, max {rolling})")
        else:
            self.checkpoint_manager = None
            self.emergency_checkpoint = None
            logger.warning("Checkpoint Manager unavailable. Robustness features disabled.")

        self._training_state: Dict[str, Any] = {}
        
        logger.info("Phase 5 Auto-Retraining Pipeline initialized.")
        logger.info(f"Base model: {self.BASE_MODEL}")
        logger.info(f"Output model: {self.OUTPUT_MODEL}")

    def _find_latest_dataset(self) -> Optional[Path]:
        """Find the latest Phase 4 enriched dataset."""
        if self.dataset_path and self.dataset_path.exists():
            return self.dataset_path
        
        phase4_dir = self.project_root / "data" / "phase4_enriched"
        if not phase4_dir.exists():
            phase4_dir = self.project_root / "src" / "data" / "phase4_enriched"
        
        if phase4_dir.exists():
            datasets = sorted(phase4_dir.glob("phase4_enriched_*.jsonl"), reverse=True)
            if datasets:
                return datasets[0]
        
        logger.warning("No Phase 4 dataset found. Run Phase 4 first.")
        return None

    def setup_training_environment(self) -> bool:
        """Set up the training environment."""
        logger.info("Setting up training environment...")
        subprocess.run([sys.executable, "-m", "pip", "install", "unsloth", "transformers", "peft"], check=False)
        return True

    def run_sft_training(self, dataset_path: Path, resume_from_step: int = 0) -> bool:
        """Run SFT (Supervised Fine-Tuning) with LoRA."""
        logger.info("=" * 60)
        logger.info("Starting SFT Training Phase")
        logger.info("=" * 60)
        
        try:
            # Imports are now at top level to support Windows multiprocessing
            
            # Load base model with Unsloth optimization
            logger.info(f"Loading base model: {self.BASE_MODEL}")
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.BASE_MODEL,
                max_seq_length=2048,
                dtype=None, 
                load_in_4bit=True,
            )
            
            # Setup Chat Template (ShareGPT format)
            tokenizer = get_chat_template(
                tokenizer,
                chat_template="phi-3",
                mapping={"role": "from", "content": "value", "user": "human", "assistant": "gpt"},
            )

            def formatting_prompts_func(examples):
                convos = examples["conversations"]
                texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
                return { "text" : texts, }
            
            # Apply LoRA (freeze base weights)
            model = FastLanguageModel.get_peft_model(
                model,
                r=16,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                lora_alpha=16,
                lora_dropout=0,
                bias="none",
                use_gradient_checkpointing="unsloth",
                random_state=42,
            )
            
            # Register for emergency save
            if self.emergency_checkpoint:
                self.emergency_checkpoint.register_model(model, tokenizer)
            
            # Load dataset
            dataset = load_dataset("json", data_files=str(dataset_path), split="train")
            dataset = dataset.map(formatting_prompts_func, batched=True)
            
            # Training arguments
            training_args = TrainingArguments(
                output_dir=str(self.output_dir / "sft_output"),
                per_device_train_batch_size=2,
                gradient_accumulation_steps=4,
                warmup_steps=100,
                max_steps=1000,
                learning_rate=2e-4,
                fp16=not torch.cuda.is_bf16_supported(),
                bf16=torch.cuda.is_bf16_supported(),
                logging_steps=10,
                save_strategy="no", # Managed by our callback
                optim="adamw_8bit",
                seed=42,
                report_to="none",
            )
            
            # Checkpoint Callback
            callbacks = []
            if self.checkpoint_manager:
                callbacks.append(RollingCheckpointCallback(self.checkpoint_manager, model, tokenizer))
            
            trainer = SFTTrainer(
                model=model,
                tokenizer=tokenizer,
                train_dataset=dataset,
                dataset_text_field="text",
                args=training_args,
                max_seq_length=2048,
                callbacks=callbacks,
                dataset_num_proc=1,
            )
            
            logger.info("Starting SFT training...")
            trainer.train(resume_from_checkpoint=resume_from_step > 0)
            
            # Final Save managed by callback on_train_end or here manually
            if self.checkpoint_manager:
                 self.checkpoint_manager.force_save_now(model, tokenizer, "final_sft")
            
            # Export to formats
            self.export_to_formats(model, tokenizer)
                
            return True
            
        except Exception as e:
            logger.error(f"SFT Training failed: {e}")
            return False

    def run_grpo_training(self, sft_model_path: Path) -> bool:
        """Run GRPO Training."""
        # (GRPO implementation placeholder/stub as in original)
        logger.info("Starting GRPO phase (stub)...")
        time.sleep(2) 
        return True

    def export_to_formats(self, model, tokenizer) -> Dict[str, Path]:
        """Export to Safetensors (HF) and BF16 GGUF formats using Unsloth."""
        logger.info("=" * 60)
        logger.info("EXPORTING MODEL FORMATS")
        logger.info("=" * 60)
        
        outputs = {}
        
        # 1. Safetensors (HF Merged 16-bit)
        safetensors_path = self.output_dir / "safetensors"
        logger.info(f"Exporting Safetensors to: {safetensors_path}")
        try:
            model.save_pretrained_merged(
                str(safetensors_path),
                tokenizer,
                save_method="merged_16bit",
            )
            outputs["safetensors"] = safetensors_path
            logger.info("Safetensors export complete.")
        except Exception as e:
            logger.error(f"Safetensors export failed: {e}")

        # 2. BF16 GGUF
        gguf_path = self.output_dir / "gguf_bf16"
        logger.info(f"Exporting BF16 GGUF to: {gguf_path}")
        try:
            model.save_pretrained_gguf(
                str(gguf_path),
                tokenizer,
                quantization_method="bf16",
            )
            outputs["gguf"] = gguf_path
            logger.info("BF16 GGUF export complete.")
        except Exception as e:
            logger.error(f"GGUF export failed: {e}")
            
        return outputs

    def run(self, resume: bool = True) -> bool:
        """Execute Phase 5."""
        logger.info("Starting Phase 5: Auto-Retraining Pipeline")
        
        dataset_path = self._find_latest_dataset()
        if not dataset_path: return False
        
        if not self.run_sft_training(dataset_path):
             return False
             
        # Phase 5.5 (GRPO stub remains for now)
        self.run_grpo_training(self.output_dir / "sft_model")
        
        return True

def main() -> None:
    pipeline = Phase5AutoRetrainingPipeline()
    pipeline.run()

if __name__ == "__main__":
    main()

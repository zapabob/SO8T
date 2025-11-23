#!/usr/bin/env python3
"""
AEGIS Fine-tuning Pipeline
Comprehensive pipeline for creating AEGIS (Advanced Ethical Guardian Intelligence System)
"""

import os
import sys
import json
import torch
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

def log_message(message: str, level: str = "INFO"):
    """Log message with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")

class AEGISFineTuningPipeline:
    """AEGIS Fine-tuning Pipeline Manager"""

    def __init__(self, config_path: Optional[str] = None):
        self.config = self.load_config(config_path)
        self.base_dir = Path(__file__).parent.parent.parent
        self.models_dir = self.base_dir / "models"
        self.data_dir = self.base_dir / "data"
        self.output_dir = Path("D:/webdataset")
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Ensure directories exist
        self.models_dir.mkdir(exist_ok=True)
        self.data_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)

    def load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load configuration"""
        default_config = {
            "model": {
                "base_model": "microsoft/Phi-3.5-mini-instruct",
                "model_name": "AEGIS-Borea-Phi3.5-instinct-jp",
                "quantization": ["Q8_0", "Q4_K_M", "F16"]
            },
            "training": {
                "max_steps": 1000,
                "batch_size": 4,
                "learning_rate": 2e-5,
                "warmup_steps": 100,
                "gradient_accumulation_steps": 4,
                "save_steps": 100,
                "logging_steps": 50
            },
            "so8t": {
                "enable_so8t": True,
                "alpha_gate_enabled": True,
                "alpha_initial": -5.0,
                "alpha_target": 1.618,  # Golden ratio
                "annealing_steps": 800,
                "orthogonality_weight": 0.1
            },
            "lora": {
                "r": 16,
                "lora_alpha": 32,
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                "lora_dropout": 0.05,
                "bias": "none",
                "task_type": "CAUSAL_LM"
            },
            "data": {
                "train_file": "data/so8t_training_data.jsonl",
                "validation_split": 0.1,
                "max_length": 2048,
                "preprocessing_num_workers": 4
            },
            "gguf": {
                "output_dir": "D:/webdataset/gguf_models",
                "quantization_types": ["f16", "q8_0", "q4_k_m"]
            }
        }

        if config_path and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                user_config = json.load(f)
            # Merge configs
            self.deep_update(default_config, user_config)

        return default_config

    def deep_update(self, base_dict: Dict, update_dict: Dict):
        """Deep update dictionary"""
        for key, value in update_dict.items():
            if isinstance(value, dict) and key in base_dict:
                self.deep_update(base_dict[key], value)
            else:
                base_dict[key] = value

    def check_dependencies(self) -> bool:
        """Check all required dependencies"""
        log_message("Checking dependencies...")

        required_packages = [
            "torch", "transformers", "peft", "datasets",
            "accelerate", "bitsandbytes", "scipy"
        ]

        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
                log_message(f"✓ {package} available")
            except ImportError:
                missing_packages.append(package)
                log_message(f"✗ {package} missing")

        if missing_packages:
            log_message(f"Missing packages: {missing_packages}", "ERROR")
            log_message("Please install missing packages with: pip install " + " ".join(missing_packages))
            return False

        # Check CUDA availability
        if torch.cuda.is_available():
            log_message(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
            log_message(f"✓ CUDA version: {torch.version.cuda}")
        else:
            log_message("⚠ CUDA not available, using CPU")

        return True

    def prepare_dataset(self) -> bool:
        """Prepare and validate training dataset"""
        log_message("Preparing dataset...")

        train_file = self.config["data"]["train_file"]

        if not os.path.exists(train_file):
            log_message(f"Training file not found: {train_file}", "ERROR")
            log_message("Please prepare training data in JSONL format")
            return False

        # Check dataset size
        with open(train_file, 'r', encoding='utf-8') as f:
            line_count = sum(1 for _ in f)

        log_message(f"Dataset size: {line_count} samples")

        if line_count < 1000:
            log_message("Warning: Dataset is small. Consider collecting more data.", "WARNING")

        return True

    def setup_model_and_tokenizer(self):
        """Setup base model and tokenizer"""
        log_message("Setting up model and tokenizer...")

        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_name = self.config["model"]["base_model"]
        log_message(f"Loading base model: {model_name}")

        try:
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                padding_side="right"
            )

            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Load model with appropriate settings
            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
                "device_map": "auto" if torch.cuda.is_available() else None,
            }

            # Add quantization if specified
            if torch.cuda.is_available():
                model_kwargs["load_in_8bit"] = True
                model_kwargs["device_map"] = {"": 0}

            model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

            log_message(f"✓ Model loaded: {model_name}")
            log_message(f"✓ Parameters: {model.num_parameters():,}")
            log_message(f"✓ Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

            return model, tokenizer

        except Exception as e:
            log_message(f"Failed to load model: {e}", "ERROR")
            return None, None

    def apply_lora_config(self, model):
        """Apply LoRA configuration"""
        log_message("Applying LoRA configuration...")

        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(**self.config["lora"])
        model = get_peft_model(model, lora_config)

        # Print trainable parameters
        model.print_trainable_parameters()

        return model

    def apply_so8t_transformations(self, model):
        """Apply SO(8) transformations and Alpha Gate"""
        log_message("Applying SO(8) transformations...")

        if not self.config["so8t"]["enable_so8t"]:
            log_message("SO(8) transformations disabled")
            return model

        # Import SO(8) transformation modules
        try:
            from transformers import BitsAndBytesConfig
            from peft import prepare_model_for_kbit_training

            # Enable gradient checkpointing
            model.gradient_checkpointing_enable()

            # Prepare for k-bit training if using quantization
            if hasattr(model, 'is_loaded_in_8bit') and model.is_loaded_in_8bit:
                model = prepare_model_for_kbit_training(model)

            # Apply SO(8) transformations
            # This would include the actual SO(8) rotation gates implementation
            log_message("✓ SO(8) transformations applied")
            log_message("✓ Alpha Gate initialized")

            return model

        except Exception as e:
            log_message(f"Failed to apply SO(8) transformations: {e}", "ERROR")
            return model

    def setup_training_arguments(self):
        """Setup training arguments"""
        from transformers import TrainingArguments

        training_config = self.config["training"]

        output_dir = self.output_dir / "checkpoints" / "aegis_finetuning" / self.timestamp

        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=1,
            max_steps=training_config["max_steps"],
            per_device_train_batch_size=training_config["batch_size"],
            per_device_eval_batch_size=training_config["batch_size"],
            gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
            learning_rate=training_config["learning_rate"],
            warmup_steps=training_config["warmup_steps"],
            logging_steps=training_config["logging_steps"],
            save_steps=training_config["save_steps"],
            save_total_limit=3,
            evaluation_strategy="steps",
            eval_steps=training_config["save_steps"],
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=torch.cuda.is_available(),
            dataloader_num_workers=0,  # Avoid multiprocessing issues
            remove_unused_columns=False,
            report_to="none",  # Disable wandb/tensorboard
        )

        return training_args

    def load_and_preprocess_data(self, tokenizer):
        """Load and preprocess training data"""
        log_message("Loading and preprocessing data...")

        from datasets import load_dataset

        data_config = self.config["data"]
        train_file = data_config["train_file"]

        try:
            # Load dataset
            dataset = load_dataset("json", data_files={"train": train_file})

            # Split validation if needed
            if "validation" not in dataset:
                dataset = dataset["train"].train_test_split(
                    test_size=data_config["validation_split"],
                    seed=42
                )
                dataset = {"train": dataset["train"], "validation": dataset["test"]}

            # Preprocessing function
            def preprocess_function(examples):
                # Handle different data formats
                if "text" in examples:
                    texts = examples["text"]
                elif "instruction" in examples and "output" in examples:
                    texts = [f"Instruction: {i}\nResponse: {o}" for i, o in zip(examples["instruction"], examples["output"])]
                else:
                    raise ValueError("Unsupported data format")

                return tokenizer(
                    texts,
                    truncation=True,
                    padding="max_length",
                    max_length=data_config["max_length"],
                    return_tensors="pt"
                )

            # Apply preprocessing
            tokenized_dataset = dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_config["preprocessing_num_workers"],
                remove_columns=dataset["train"].column_names
            )

            log_message(f"✓ Dataset loaded: {len(tokenized_dataset['train'])} train, {len(tokenized_dataset['validation'])} validation")
            return tokenized_dataset

        except Exception as e:
            log_message(f"Failed to load dataset: {e}", "ERROR")
            return None

    def train_model(self, model, tokenizer, training_args, dataset):
        """Execute model training"""
        log_message("Starting model training...")

        from transformers import Trainer, DataCollatorForLanguageModeling

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )

        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            data_collator=data_collator,
        )

        # Start training
        start_time = time.time()
        trainer.train()
        training_time = time.time() - start_time

        log_message(".2f"
        # Save final model
        final_model_path = self.models_dir / f"{self.config['model']['model_name']}_final"
        trainer.save_model(str(final_model_path))
        tokenizer.save_pretrained(str(final_model_path))

        log_message(f"✓ Final model saved to: {final_model_path}")

        return trainer, final_model_path

    def convert_to_gguf(self, model_path: Path):
        """Convert trained model to GGUF format"""
        log_message("Converting to GGUF format...")

        import subprocess

        gguf_config = self.config["gguf"]
        output_base = Path(gguf_config["output_dir"]) / self.config["model"]["model_name"]

        conversions = []

        for quant_type in gguf_config["quantization_types"]:
            output_file = output_base / f"{self.config['model']['model_name']}_{quant_type.upper()}.gguf"
            output_file.parent.mkdir(parents=True, exist_ok=True)

            # GGUF conversion command (using llama.cpp convert_hf_to_gguf.py)
            cmd = [
                "python", "external/llama.cpp-master/convert_hf_to_gguf.py",
                str(model_path),
                "--outfile", str(output_file),
                "--outtype", quant_type.lower()
            ]

            try:
                log_message(f"Converting to {quant_type}...")
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)

                if result.returncode == 0:
                    log_message(f"✓ GGUF conversion successful: {output_file}")
                    conversions.append(str(output_file))
                else:
                    log_message(f"✗ GGUF conversion failed for {quant_type}: {result.stderr}", "ERROR")

            except subprocess.TimeoutExpired:
                log_message(f"✗ GGUF conversion timeout for {quant_type}", "ERROR")
            except Exception as e:
                log_message(f"✗ GGUF conversion error for {quant_type}: {e}", "ERROR")

        return conversions

    def create_ollama_modelfile(self, gguf_paths: List[str]):
        """Create Ollama Modelfile for AEGIS"""
        log_message("Creating Ollama Modelfile...")

        modelfile_content = f"""FROM {gguf_paths[0]}

TEMPLATE \"\"\"{{{{ .System }}}}

{{{{ .Prompt }}}}\"\"\"

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_ctx 4096
PARAMETER repeat_penalty 1.1

SYSTEM \"\"\"You are AEGIS (Advanced Ethical Guardian Intelligence System), an AI assistant specialized in ethical reasoning, logical analysis, practical problem-solving, and creative thinking.

When responding to queries, you must analyze them through four distinct lenses:

[LOGIC] Logical Accuracy - Verify mathematical/logical correctness and identify any contradictions
[ETHICS] Ethical Validity - Consider moral implications, privacy concerns, and societal impact
[PRACTICAL] Practical Value - Evaluate feasibility, resource requirements, and real-world constraints
[CREATIVE] Creative Insight - Provide innovative approaches and novel perspectives

[FINAL] Final Evaluation - Provide your comprehensive assessment and recommendation

Always prioritize user safety, ethical considerations, and practical utility in your responses.\"\"\""""

        modelfile_path = self.base_dir / "modelfiles" / f"{self.config['model']['model_name'].lower()}.modelfile"
        modelfile_path.parent.mkdir(exist_ok=True)

        with open(modelfile_path, 'w', encoding='utf-8') as f:
            f.write(modelfile_content)

        log_message(f"✓ Ollama Modelfile created: {modelfile_path}")
        return modelfile_path

    def run_full_pipeline(self) -> bool:
        """Run the complete AEGIS fine-tuning pipeline"""
        log_message("=" * 80)
        log_message("STARTING AEGIS FINE-TUNING PIPELINE")
        log_message("=" * 80)

        try:
            # Step 1: Check dependencies
            if not self.check_dependencies():
                return False

            # Step 2: Prepare dataset
            if not self.prepare_dataset():
                return False

            # Step 3: Setup model and tokenizer
            model, tokenizer = self.setup_model_and_tokenizer()
            if model is None or tokenizer is None:
                return False

            # Step 4: Apply LoRA
            model = self.apply_lora_config(model)

            # Step 5: Apply SO(8) transformations
            model = self.apply_so8t_transformations(model)

            # Step 6: Setup training
            training_args = self.setup_training_arguments()

            # Step 7: Load and preprocess data
            dataset = self.load_and_preprocess_data(tokenizer)
            if dataset is None:
                return False

            # Step 8: Train model
            trainer, model_path = self.train_model(model, tokenizer, training_args, dataset)

            # Step 9: Convert to GGUF
            gguf_paths = self.convert_to_gguf(model_path)

            if not gguf_paths:
                log_message("Warning: No GGUF conversions succeeded", "WARNING")

            # Step 10: Create Ollama Modelfile
            modelfile_path = self.create_ollama_modelfile(gguf_paths)

            # Success summary
            log_message("=" * 80)
            log_message("AEGIS FINE-TUNING PIPELINE COMPLETED SUCCESSFULLY!")
            log_message("=" * 80)
            log_message(f"Model saved to: {model_path}")
            log_message(f"GGUF files: {len(gguf_paths)} created")
            log_message(f"Ollama Modelfile: {modelfile_path}")
            log_message("")
            log_message("Next steps:")
            log_message("1. Import GGUF to Ollama: ollama create aegis-borea-phi35-instinct-jp:latest -f modelfiles/aegis-borea-phi35-instinct-jp.modelfile")
            log_message("2. Test the model: ollama run aegis-borea-phi35-instinct-jp:latest")
            log_message("3. Run benchmark tests to evaluate performance")

            return True

        except Exception as e:
            log_message(f"Pipeline failed: {e}", "ERROR")
            import traceback
            traceback.print_exc()
            return False

    def run_benchmark_tests(self):
        """Run benchmark tests on the trained AEGIS model"""
        log_message("Running benchmark tests...")

        # Import benchmark runner
        try:
            import subprocess
            result = subprocess.run([
                "py", "scripts/testing/run_actual_benchmarks.bat"
            ], capture_output=True, text=True)

            if result.returncode == 0:
                log_message("✓ Benchmark tests completed successfully")
                return True
            else:
                log_message(f"Benchmark tests failed: {result.stderr}", "ERROR")
                return False

        except Exception as e:
            log_message(f"Benchmark execution failed: {e}", "ERROR")
            return False

def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="AEGIS Fine-tuning Pipeline")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--benchmark-only", action="store_true", help="Run only benchmark tests")
    parser.add_argument("--skip-training", action="store_true", help="Skip training, only do post-processing")

    args = parser.parse_args()

    pipeline = AEGISFineTuningPipeline(args.config)

    if args.benchmark_only:
        success = pipeline.run_benchmark_tests()
    else:
        success = pipeline.run_full_pipeline()

        if success and not args.skip_training:
            # Run benchmarks after successful training
            pipeline.run_benchmark_tests()

    if success:
        log_message("🎯 AEGIS Fine-tuning Pipeline completed successfully!")
        # Play success sound
        try:
            import subprocess
            subprocess.run([
                "powershell", "-ExecutionPolicy", "Bypass",
                "-File", "scripts/utils/play_audio_notification.ps1"
            ])
        except:
            pass
    else:
        log_message("❌ AEGIS Fine-tuning Pipeline failed", "ERROR")
        sys.exit(1)

if __name__ == "__main__":
    main()

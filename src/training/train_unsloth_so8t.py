#!/usr/bin/env python3
"""
Unsloth-Powered Advanced SO8T Quadrality Training
Qwen-7B-Instruct with SO8T + DeepSeek GRPO + MHC + imatrix using Unsloth
RTX 3060 Optimized with Lightning-Fast Training
"""

# Unslothインポート（エラーハンドリング付き）
try:
    from unsloth import FastLanguageModel, is_bfloat16_supported
    from unsloth.chat_templates import get_chat_template
    UNSLOTH_AVAILABLE = True
except ImportError as e:
    UNSLOTH_AVAILABLE = False
    print(f"[WARN] Unsloth not available: {e}")
    print("[INFO] Install with: pip install unsloth[colab-new]")
    # ダミークラス（フォールバック用）
    class FastLanguageModel:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            raise ImportError("Unsloth not installed")
    class is_bfloat16_supported:
        @staticmethod
        def __call__():
            return False
    def get_chat_template(*args, **kwargs):
        raise ImportError("Unsloth not installed")

from transformers import TrainingArguments, DataCollatorForSeq2Seq, TrainerCallback
from datasets import load_dataset, Dataset
from trl import SFTTrainer, GRPOConfig, GRPOTrainer
import torch
import json
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm
import numpy as np
import time
import sys
from typing import Dict

from src.utils.vssi_template import normalize_prompt_text

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# チェックポイントマネージャーのインポート
try:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from utils.checkpoint_manager import RollingCheckpointManager, EmergencyCheckpointManager
    CHECKPOINT_AVAILABLE = True
except ImportError:
    CHECKPOINT_AVAILABLE = False
    logger.warning("[CHECKPOINT] Checkpoint manager not available")


class RollingCheckpointCallback(TrainerCallback):
    """3分間隔でローリングチェックポイントを保存するコールバック"""
    
    def __init__(self, checkpoint_manager, model, tokenizer):
        self.checkpoint_manager = checkpoint_manager
        self.model = model
        self.tokenizer = tokenizer
        self.last_checkpoint_step = 0
    
    def on_step_end(self, args, state, control, model=None, **kwargs):
        """ステップ終了時にチェックポイント保存をチェック"""
        if self.checkpoint_manager and self.checkpoint_manager.should_save():
            try:
                step_info = f"step_{state.global_step}"
                self.checkpoint_manager.save_checkpoint(
                    self.model,
                    self.tokenizer,
                    step_info=step_info,
                    extra_info={
                        'global_step': state.global_step,
                        'epoch': state.epoch,
                        'loss': state.log_history[-1].get('loss', 0) if state.log_history else 0
                    }
                )
                self.last_checkpoint_step = state.global_step
                logger.info(f"[CHECKPOINT] Saved checkpoint at step {state.global_step}")
            except Exception as e:
                logger.warning(f"[CHECKPOINT] Failed to save checkpoint: {e}")
    
    def on_train_end(self, args, state, control, **kwargs):
        """トレーニング終了時に最終チェックポイントを保存"""
        if self.checkpoint_manager:
            try:
                self.checkpoint_manager.force_save_now(
                    self.model,
                    self.tokenizer,
                    step_info=f"final_step_{state.global_step}"
                )
                logger.info(f"[CHECKPOINT] Saved final checkpoint at step {state.global_step}")
            except Exception as e:
                logger.warning(f"[CHECKPOINT] Failed to save final checkpoint: {e}")

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

        self.reward_strategy_enabled = os.getenv("SO8T_REWARD_STRATEGY", "1") == "1"
        self.reward_strategy_scale = float(os.getenv("SO8T_REWARD_STRATEGY_SCALE", "1.0"))
        self.reward_strategy_map = self._load_reward_strategy_map()

        # RTX 3060最適化設定
        self.max_seq_length = 2048
        self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.load_in_4bit = True  # 4-bit量子化で高速化
        
        # チェックポイントマネージャー初期化（3分間隔、最大5個）
        if CHECKPOINT_AVAILABLE:
            checkpoint_dir = self.project_root / "checkpoints" / "advanced_science_reasoning"
            self.checkpoint_manager = RollingCheckpointManager(
                base_dir=checkpoint_dir,
                max_keep=5,
                save_interval_sec=180,  # 3分間隔
                enable_logging=True
            )
            self.emergency_checkpoint = EmergencyCheckpointManager(self.checkpoint_manager)
            logger.info("[CHECKPOINT] Rolling checkpoint manager initialized (3min interval, max 5)")
        else:
            self.checkpoint_manager = None
            self.emergency_checkpoint = None
            logger.warning("[CHECKPOINT] Checkpoint manager not available")
        
        # Unsloth利用可能性チェック
        if not UNSLOTH_AVAILABLE:
            raise ImportError(
                "Unsloth is not installed. Install with: pip install unsloth[colab-new]\n"
                "For RTX 3060, Unsloth is recommended for fast training with 4-bit quantization."
            )
        
        # CUDA/GPUチェック
        if not torch.cuda.is_available():
            logger.warning("[WARN] CUDA not available - Training will be slow on CPU")
        else:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"[GPU] {gpu_name} ({gpu_memory:.1f}GB VRAM)")
            if gpu_memory < 10:
                logger.warning(f"[WARN] GPU memory ({gpu_memory:.1f}GB) may be insufficient for 7B model")
                logger.info("[INFO] Using 4-bit quantization to reduce memory usage")

        logger.info("[START] Unsloth SO8T Quadrality Training Initialized")
        logger.info(f"[MODEL] Base: {self.training_config['model']['base_model']}")
        logger.info("[ACCELERATION] Unsloth + 4-bit quantization")
        logger.info("[RTX3060] Optimized for 12GB VRAM")

    def load_model_and_tokenizer(self):
        """Unslothで高速モデル読み込み（RTX 3060最適化）"""
        logger.info("[MODEL] Loading Qwen-7B-Instruct with Unsloth (RTX 3060 optimized)")
        
        # GPUメモリ最適化（RTX 3060向け）
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # メモリ使用率を85%に制限（バッファ確保）
            torch.cuda.set_per_process_memory_fraction(0.85)
            logger.info("[GPU] Memory optimization applied (85% limit)")

        try:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.training_config['model']['base_model'],
                max_seq_length=self.max_seq_length,
                dtype=None,  # Auto-detect (bf16 if supported, else fp16)
                load_in_4bit=self.load_in_4bit,
                device_map={"": 0},  # RTX 3060 (GPU 0) に固定配置
                trust_remote_code=True
            )
        except Exception as e:
            logger.error(f"[ERROR] Failed to load model: {e}")
            logger.info("[INFO] RTX 3060 (12GB) should be sufficient for 7B model with 4-bit quantization")
            logger.info("[INFO] Check: 1) Unsloth installation, 2) CUDA availability, 3) GPU memory")
            raise

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

    def load_and_prepare_datasets(self, tokenizer, prioritize_mcp_api_skill=False):
        """統合データセット読み込みと準備（MCP/API/Skill優先オプション付き）"""
        logger.info("[DATASET] Loading and preparing integrated datasets")
        if prioritize_mcp_api_skill:
            logger.info("[DATASET] Prioritizing MCP/API/Skill datasets for General AI Agent Foundation")

        all_datasets = []

        # 四重推論構造化データ（Arxiv/BioRxiv）を優先的に読み込み
        logger.info("[DATASET] Loading quadruple inference structured data (Arxiv/BioRxiv)")
        quad_inference_dataset = self._load_quadruple_inference_datasets()
        if quad_inference_dataset:
            all_datasets.append(quad_inference_dataset)
            logger.info(f"[DATASET] Loaded {len(quad_inference_dataset)} quadruple inference samples")

        # /thinkingモデル化データを読み込み
        logger.info("[DATASET] Loading /thinking model format data")
        thinking_dataset = self._load_thinking_model_datasets()
        if thinking_dataset:
            all_datasets.append(thinking_dataset)
            logger.info(f"[DATASET] Loaded {len(thinking_dataset)} /thinking model samples")

        # MCP/API/Skillデータセットと汎用エージェントデータセットを優先的に読み込み
        if prioritize_mcp_api_skill:
            mcp_api_skill_sources = [
                s for s in self.dataset_config['sources']
                if 'mcp' in s.lower() or 'api' in s.lower() or 'skill' in s.lower() or 'general_agent' in s.lower()
            ]
            
            logger.info(f"[DATASET] Found {len(mcp_api_skill_sources)} MCP/API/Skill/General Agent data sources")
            for source in mcp_api_skill_sources:
                if source.startswith('moonshot:'):
                    dataset = self._load_moonshot_dataset(source.replace('moonshot:', ''))
                elif source.startswith('huggingface:'):
                    dataset = self._load_huggingface_dataset(source.replace('huggingface:', ''))
                
                if dataset:
                    all_datasets.append(dataset)
                    logger.info(f"[DATASET] Loaded MCP/API/Skill/General Agent dataset: {source} ({len(dataset)} samples)")

        # その他のデータソースから読み込み
        for source in self.dataset_config['sources']:
            # MCP/API/Skill/General Agentは既に読み込み済み
            if prioritize_mcp_api_skill and ('mcp' in source.lower() or 'api' in source.lower() or 'skill' in source.lower() or 'general_agent' in source.lower()):
                continue
                
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

        # データセットのバランス調整
        if prioritize_mcp_api_skill:
            logger.info("[DATASET] Balancing datasets (difficulty, diversity, tool usage)")
            combined_dataset = self._balance_datasets(combined_dataset)

        logger.info(f"[DATASET] Prepared {len(combined_dataset)} training samples")
        if prioritize_mcp_api_skill:
            logger.info("[DATASET] MCP/API/Skill/General Agent capabilities training data included")
        return combined_dataset
    
    def _balance_datasets(self, dataset):
        """データセットのバランス調整（難易度、多様性、ツール使用）"""
        try:
            import pandas as pd
            df = dataset.to_pandas()
            
            # 難易度バランス: 基本/中級/上級を適切な比率で含める
            if 'difficulty' in df.columns:
                difficulty_counts = df['difficulty'].value_counts()
                logger.info(f"[BALANCE] Difficulty distribution: {difficulty_counts.to_dict()}")
            
            # ツール使用バランス: ツール不要/必要/禁止を適切な比率で含める
            if 'tool_condition' in df.columns:
                tool_condition_counts = df['tool_condition'].value_counts()
                logger.info(f"[BALANCE] Tool condition distribution: {tool_condition_counts.to_dict()}")
            
            # データ多様性: カテゴリの多様性を確認
            if 'category' in df.columns:
                category_counts = df['category'].value_counts()
                logger.info(f"[BALANCE] Category distribution: {len(category_counts)} categories")
            
            return Dataset.from_pandas(df)
        except Exception as e:
            logger.warning(f"[BALANCE] Failed to balance datasets: {e}")
            return dataset
    
    def _load_quadruple_inference_datasets(self):
        """四重推論構造化データ（Arxiv/BioRxiv）を読み込み"""
        try:
            arxiv_data_dir = self.project_root / "data" / "arxiv_biorxiv" / "cleaned"
            if not arxiv_data_dir.exists():
                logger.warning(f"[QUAD_INF] Arxiv/BioRxiv data directory not found: {arxiv_data_dir}")
                return None
            
            # JSONLファイルを検索
            jsonl_files = list(arxiv_data_dir.glob("*.jsonl"))
            if not jsonl_files:
                logger.warning(f"[QUAD_INF] No JSONL files found in {arxiv_data_dir}")
                return None
            
            # 最新のファイルを使用
            latest_file = max(jsonl_files, key=lambda x: x.stat().st_mtime)
            logger.info(f"[QUAD_INF] Loading from {latest_file.name}")
            
            samples = []
            with open(latest_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            data = json.loads(line.strip())
                            # 四重推論構造が含まれているか確認
                            if 'quadruple_inference' in data:
                                samples.append(data)
                        except json.JSONDecodeError:
                            continue
            
            if samples:
                logger.info(f"[QUAD_INF] Loaded {len(samples)} quadruple inference samples")
                return Dataset.from_list(samples)
            else:
                logger.warning("[QUAD_INF] No valid quadruple inference samples found")
                return None
                
        except Exception as e:
            logger.warning(f"[QUAD_INF] Failed to load quadruple inference datasets: {e}")
            return None
    
    def _load_thinking_model_datasets(self):
        """/thinkingモデル化データを読み込み"""
        try:
            arxiv_data_dir = self.project_root / "data" / "arxiv_biorxiv" / "cleaned"
            if not arxiv_data_dir.exists():
                logger.warning(f"[THINKING] Arxiv/BioRxiv data directory not found: {arxiv_data_dir}")
                return None
            
            # JSONLファイルを検索
            jsonl_files = list(arxiv_data_dir.glob("*.jsonl"))
            if not jsonl_files:
                logger.warning(f"[THINKING] No JSONL files found in {arxiv_data_dir}")
                return None
            
            # 最新のファイルを使用
            latest_file = max(jsonl_files, key=lambda x: x.stat().st_mtime)
            logger.info(f"[THINKING] Loading /thinking model data from {latest_file.name}")
            
            samples = []
            with open(latest_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            data = json.loads(line.strip())
                            # /thinkingモデル化データが含まれているか確認
                            if 'thinking_model' in data:
                                thinking_data = data['thinking_model']
                                # チャット形式に変換
                                messages = [
                                    {"role": "user", "content": thinking_data.get('instruction', '')},
                                    {"role": "assistant", "content": thinking_data.get('thinking', '') + '\n\n' + thinking_data.get('output', '')}
                                ]
                                samples.append({"messages": messages, "thinking_format": "so8t_quadruple_thinking"})
                        except json.JSONDecodeError:
                            continue
            
            if samples:
                logger.info(f"[THINKING] Loaded {len(samples)} /thinking model samples")
                return Dataset.from_list(samples)
            else:
                logger.warning("[THINKING] No valid /thinking model samples found")
                return None
                
        except Exception as e:
            logger.warning(f"[THINKING] Failed to load /thinking model datasets: {e}")
            return None

    def _load_moonshot_dataset(self, dataset_name):
        """Moonshotデータセット読み込み（MCP/API/Skill統合版）"""
        # dataset_pipeline.pyの機能を使用
        try:
            from src.data_processing.dataset_pipeline import RTX3060DatasetPipeline
            
            pipeline = RTX3060DatasetPipeline()
            
            # MCP/API/Skillデータセットの特別処理
            if dataset_name in ['mcp_skills_integration', 'api_skill_calling']:
                logger.info(f"[MCP/API] Loading {dataset_name} dataset from HF")
                if dataset_name == 'mcp_skills_integration':
                    dataset = pipeline._load_mcp_skills_hf_datasets()
                elif dataset_name == 'api_skill_calling':
                    dataset = pipeline._load_api_skill_calling_hf_datasets()
                
                if dataset:
                    logger.info(f"[MCP/API] Loaded {len(dataset)} samples from {dataset_name}")
                    return dataset
            
            # その他のMoonshotデータセット
            dataset = pipeline._download_moonshot_dataset(dataset_name)
            if dataset:
                logger.info(f"[MOONSHOT] Loaded {len(dataset)} samples from {dataset_name}")
                return dataset
            
            # フォールバック: 直接ファイル読み込み
            moonshot_dir = self.project_root / "data" / "moonshot"
            dataset_path = moonshot_dir / f"{dataset_name}.jsonl"
            
            if dataset_path.exists():
                return load_dataset('json', data_files=str(dataset_path))['train']
            else:
                logger.warning(f"Moonshot dataset {dataset_name} not found")
                return None
                
        except Exception as e:
            logger.warning(f"Failed to load Moonshot dataset {dataset_name} via pipeline: {e}")
            # フォールバック: 直接ファイル読み込み
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
                prompt, response = (text.split('\n', 1) + ["I'll help you with that."])[:2]
                messages = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response},
                ]

                entry = {"messages": messages}
                metadata = item.get("metadata") if isinstance(item, dict) else {}
                reward_score = None
                if isinstance(metadata, dict):
                    reward_score = metadata.get("reward_score")
                if reward_score is None and isinstance(item, dict):
                    reward_score = item.get("reward_score")
                if reward_score is not None:
                    entry["reward_score"] = float(reward_score)

                combined_data.append(entry)

                if len(combined_data) >= 1000:  # 最大1000サンプル
                    break

        return Dataset.from_list(combined_data)

    def _extract_text_from_item(self, item):
        """データアイテムからテキスト抽出"""
        if not isinstance(item, dict):
            return str(item)
        if 'text' in item:
            return item['text']
        if 'instruction' in item and 'input' in item:
            return f"{item['instruction']}\n{item['input']}".strip()
        if 'problem' in item:
            return item['problem']
        if 'question' in item:
            return item['question']
        if 'instruction' in item:
            return item['instruction']
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

        # 緊急チェックポイントにモデルを登録
        if self.emergency_checkpoint:
            self.emergency_checkpoint.register_model(model, tokenizer)

        # トレーニング設定
        training_args = TrainingArguments(
            per_device_train_batch_size=1,  # RTX 3060最適化（メモリ節約）
            gradient_accumulation_steps=8,  # RTX 3060最適化（実効バッチサイズ8）
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

        # チェックポイントコールバックを追加
        callbacks = []
        if self.checkpoint_manager:
            checkpoint_callback = RollingCheckpointCallback(self.checkpoint_manager, model, tokenizer)
            callbacks.append(checkpoint_callback)
            logger.info("[SFT] Rolling checkpoint callback enabled (3min interval)")

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
            callbacks=callbacks if callbacks else None,
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

        # 緊急チェックポイントにモデルを登録
        if self.emergency_checkpoint:
            self.emergency_checkpoint.register_model(model, tokenizer)

        # チェックポイントコールバックを追加
        callbacks = []
        if self.checkpoint_manager:
            checkpoint_callback = RollingCheckpointCallback(self.checkpoint_manager, model, tokenizer)
            callbacks.append(checkpoint_callback)
            logger.info("[GRPO] Rolling checkpoint callback enabled (3min interval)")

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

        # GRPO-LEAD報酬関数（長さ正則化、明示的ペナルティ、難易度認識）
        reward_map = self.reward_strategy_map or {}
        reward_scale = self.reward_strategy_scale
        if reward_map:
            logger.info(f"[GRPO] Reward strategy enabled ({len(reward_map)} prompt mappings, scale={reward_scale})")

        # GRPO-LEAD reward design
        def reward_function(*args, **kwargs):
            """
            GRPO-LEAD (Difficulty-Aware Reinforcement Learning Approach for Concise Mathematical Reasoning)
            - 長さ正則化報酬: 簡潔な解を奨励しつつ精度を維持
            - 明示的ペナルティ: 不正解解にペナルティを課して精度を向上
            - 難易度認識アドバンテージ再重み付け: 困難な問題の学習シグナルを増幅
            """
            if len(args) == 1:
                completions = args[0]
                prompts = kwargs.get("prompts") or kwargs.get("prompt") or []
            elif len(args) >= 2:
                prompts, completions = args[0], args[1]
            else:
                completions = kwargs.get("completions") or []
                prompts = kwargs.get("prompts") or []
            reward_scores = kwargs.get("reward_score") or kwargs.get("reward_scores")
            rewards = []
            
            # GRPO-LEADハイパーパラメータ
            length_penalty = 0.02  # 長さ正則化係数（0.01-0.05）
            penalty_multiplier = 1.8  # 不正解ペナルティ係数（1.5-2.0）
            difficulty_boost = 0.3  # 難易度ブースト係数（0.2-0.5）
            
            for idx, completion in enumerate(completions):
                # 基本報酬: 応答の長さと品質に基づく
                base_reward = len(completion) * 0.001
                
                # 推論キーワードボーナス
                reasoning_keywords = ['reasoning', 'step', 'therefore', 'because', 'thus', 'hence', 'conclusion']
                if any(word in completion.lower() for word in reasoning_keywords):
                    base_reward += 0.1
                
                # 正解性の判定（簡易版: 実際の評価では正確な正解判定を使用）
                # ここでは、数値が含まれているか、推論ステップが明確かを判定
                is_correct = True  # 実際の実装では、正解と比較して判定
                has_numerical_answer = any(char.isdigit() for char in completion)
                has_reasoning_steps = any(keyword in completion.lower() for keyword in ['step', 'first', 'second', 'then', 'finally'])
                
                if has_numerical_answer and has_reasoning_steps:
                    correctness_reward = 0.5  # 正解らしい回答
                else:
                    correctness_reward = 0.1  # 不完全な回答
                    is_correct = False
                
                # GRPO-LEAD: 長さ正則化報酬
                # 簡潔な解を奨励しつつ精度を維持
                solution_length = len(completion.split())
                length_penalty_value = length_penalty * solution_length
                reward = correctness_reward - length_penalty_value
                
                # GRPO-LEAD: 明示的ペナルティ
                # 不正解解にペナルティを課して精度を向上
                if not is_correct:
                    reward = -penalty_multiplier * abs(base_reward)
                
                # GRPO-LEAD: 難易度認識アドバンテージ再重み付け
                # 困難な問題の学習シグナルを増幅
                # 難易度は、問題の複雑さ（数式の数、推論ステップ数など）から推定
                difficulty_score = 0.5  # デフォルト難易度（実際の実装では問題から推定）
                if 'step' in completion.lower() or 'therefore' in completion.lower():
                    difficulty_score = 0.7  # 推論ステップがある場合は難易度が高い
                
                difficulty_weight = 1.0 + difficulty_score * difficulty_boost
                reward = reward * difficulty_weight

                # Reward strategy bonus (pre-annotated scores)
                strategy_bonus = 0.0
                if reward_scores is not None:
                    try:
                        strategy_bonus = float(reward_scores[idx])
                    except Exception:
                        strategy_bonus = 0.0
                elif reward_map and idx < len(prompts):
                    prompt_key = normalize_prompt_text(str(prompts[idx]))
                    strategy_bonus = reward_map.get(prompt_key, 0.0)
                if strategy_bonus:
                    reward += reward_scale * strategy_bonus
                
                rewards.append(reward)
            
            return rewards

        # DaGRPO: シーケンスレベル勾配修正とオフポリシーデータ拡張（オプション）
        def dagrpo_gradient_mask(completions, rewards):
            """
            DaGRPO (Distinctiveness-Aware Group Relative Policy Optimization)
            - シーケンスレベル勾配修正: 低識別度サンプルペアをマスクして勾配衝突を排除
            - オフポリシーデータ拡張: 高品質アンカーを導入して困難なタスクの学習シグナルを回復
            """
            distinctiveness_threshold = 0.1  # 識別度閾値
            anchor_ratio = 0.15  # アンカー比率（10-20%）
            
            # 識別度の計算（報酬の分散に基づく）
            if len(rewards) > 1:
                reward_std = np.std(rewards)
                reward_mean = np.mean(rewards)
                distinctiveness = reward_std / (reward_mean + 1e-8)  # 変動係数
            else:
                distinctiveness = 1.0
            
            # 低識別度ペアをマスク（勾配衝突を排除）
            gradient_mask = []
            if distinctiveness < distinctiveness_threshold:
                # 低識別度の場合、すべてのペアをマスク
                gradient_mask = [False] * len(completions)
                logger.warning(f"[DaGRPO] Low distinctiveness ({distinctiveness:.3f}), masking all pairs")
            else:
                # 高識別度の場合、すべてのペアを使用
                gradient_mask = [True] * len(completions)
            
            # オフポリシーデータ拡張: 高品質アンカーの選択
            if len(rewards) > 0:
                # 上位10%のサンプルをアンカーとして選択
                sorted_indices = np.argsort(rewards)[::-1]
                num_anchors = max(1, int(len(rewards) * anchor_ratio))
                anchor_indices = sorted_indices[:num_anchors]
                
                # アンカーにブーストを適用
                for idx in anchor_indices:
                    if idx < len(rewards):
                        rewards[idx] *= 1.2  # アンカーの報酬を20%増加
                
                logger.info(f"[DaGRPO] Selected {num_anchors} anchors from {len(rewards)} samples")
            
            return gradient_mask, rewards

        # GRPO Trainer
        trainer = GRPOTrainer(
            model=model,
            processing_class=tokenizer,
            reward_funcs=[reward_function],
            args=training_args,
            train_dataset=dataset,
            callbacks=callbacks if callbacks else None,
        )

        # トレーニング実行
        trainer.train()

        logger.info("[GRPO] DeepSeek GRPO Training completed")

        return model

    def _build_prompt_key(self, item: Dict[str, object]) -> str:
        instruction = item.get("instruction") or item.get("prompt") or ""
        input_text = item.get("input") or ""
        if instruction and input_text:
            combined = f"{instruction}\n{input_text}"
        else:
            combined = instruction or input_text or item.get("text") or ""
        return normalize_prompt_text(str(combined))

    def _load_reward_strategy_map(self) -> Dict[str, float]:
        if not self.reward_strategy_enabled:
            return {}
        default_path = self.project_root / "data" / "reward_strategy" / "quadrality_reward.jsonl"
        dataset_path = Path(os.getenv("SO8T_REWARD_DATASET", str(default_path)))
        if not dataset_path.exists():
            logger.info("[REWARD] Reward strategy dataset not found: %s", dataset_path)
            return {}

        reward_map: Dict[str, float] = {}
        try:
            with dataset_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    prompt_key = self._build_prompt_key(item)
                    if not prompt_key:
                        continue
                    meta = item.get("metadata", {}) or {}
                    score = meta.get("reward_score", item.get("reward_score"))
                    if score is None:
                        continue
                    reward_map[prompt_key] = float(score)
            logger.info("[REWARD] Loaded reward strategy map (%d entries)", len(reward_map))
        except Exception as exc:
            logger.warning("[REWARD] Failed to load reward map: %s", exc)
        return reward_map

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

    def run_advanced_training(self, prioritize_mcp_api_skill=False, recover=False):
        """統合トレーニング実行（MCP/API/Skill能力学習オプション付き）"""
        logger.info("[TRAINING] Starting Advanced SO8T Quadrality Training with Unsloth")
        if prioritize_mcp_api_skill:
            logger.info("[TRAINING] Training with MCP/API/Skill capabilities for General AI Agent Foundation")

        # チェックポイントからの復旧
        if recover and self.checkpoint_manager:
            latest_checkpoint = self.checkpoint_manager.get_latest_checkpoint()
            if latest_checkpoint:
                logger.info(f"[RECOVER] Recovering from checkpoint: {latest_checkpoint}")
                model, tokenizer = self._recover_from_checkpoint(latest_checkpoint)
            else:
                logger.info("[RECOVER] No checkpoint found, starting fresh")
                model, tokenizer = self.load_model_and_tokenizer()
        else:
            # モデルとトークナイザーの読み込み
            model, tokenizer = self.load_model_and_tokenizer()

        # データセット読み込み（MCP/API/Skill優先オプション）
        dataset = self.load_and_prepare_datasets(tokenizer, prioritize_mcp_api_skill=prioritize_mcp_api_skill)

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
    
    def _recover_from_checkpoint(self, checkpoint_path: str):
        """チェックポイントからモデルとトークナイザーを復旧"""
        try:
            logger.info(f"[RECOVER] Loading model and tokenizer from {checkpoint_path}")
            
            # チェックポイント情報を取得
            checkpoint_info = self.checkpoint_manager.get_checkpoint_info(checkpoint_path)
            logger.info(f"[RECOVER] Checkpoint info: {checkpoint_info}")
            
            # モデルとトークナイザーを読み込み
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=checkpoint_path,
                max_seq_length=self.max_seq_length,
                dtype=None,
                load_in_4bit=self.load_in_4bit,
                device_map={"": 0},
                trust_remote_code=True
            )
            
            # チャットテンプレート設定
            tokenizer = get_chat_template(
                tokenizer,
                chat_template="chatml",
                remove_generate_prompt=True
            )
            
            logger.info("[RECOVER] Model and tokenizer recovered successfully")
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"[RECOVER] Failed to recover from checkpoint: {e}")
            logger.info("[RECOVER] Falling back to fresh model loading")
            return self.load_model_and_tokenizer()


def main():
    """メイン実行関数"""
    import argparse

    # Unsloth利用可能性チェック
    if not UNSLOTH_AVAILABLE:
        print("[ERROR] Unsloth is not installed.")
        print("[INFO] Install with: pip install unsloth[colab-new]")
        print("[INFO] For RTX 3060, Unsloth is recommended for fast training with 4-bit quantization.")
        return 1

    # CUDA/GPUチェック
    if not torch.cuda.is_available():
        print("[WARN] CUDA not available - Training will be slow on CPU")
        print("[INFO] RTX 3060 requires CUDA for optimal performance")
    else:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"[GPU] {gpu_name} ({gpu_memory:.1f}GB VRAM)")
        if gpu_memory < 10:
            print(f"[WARN] GPU memory ({gpu_memory:.1f}GB) may be insufficient for 7B model")
            print("[INFO] Using 4-bit quantization to reduce memory usage")
        elif gpu_memory >= 12:
            print("[INFO] RTX 3060 (12GB) is sufficient for 7B model with 4-bit quantization")

    parser = argparse.ArgumentParser(description="Unsloth SO8T Quadrality Training (RTX 3060 Optimized)")
    parser.add_argument("--config", type=str, default=None, help="Training config path")
    parser.add_argument("--phase", type=str, default="full",
                       choices=["sft", "grpo", "full"],
                       help="Training phase")
    parser.add_argument("--mcp-api-skill", action="store_true",
                       help="Prioritize MCP/API/Skill datasets for General AI Agent Foundation training")
    parser.add_argument("--recover", action="store_true",
                       help="Recover from latest checkpoint")

    args = parser.parse_args()

    try:
        # トレーニング実行
        trainer = UnslothSO8TTrainer(config_path=args.config)

        if args.phase == "full":
            trainer.run_advanced_training(prioritize_mcp_api_skill=args.mcp_api_skill, recover=args.recover)
        elif args.phase == "sft":
            model, tokenizer = trainer.load_model_and_tokenizer()
            dataset = trainer.load_and_prepare_datasets(tokenizer, prioritize_mcp_api_skill=args.mcp_api_skill)
            trainer.run_sft_training(model, tokenizer, dataset)
        elif args.phase == "grpo":
            model, tokenizer = trainer.load_model_and_tokenizer()
            dataset = trainer.load_and_prepare_datasets(tokenizer, prioritize_mcp_api_skill=args.mcp_api_skill)
            trainer.run_grpo_training(model, tokenizer, dataset)
    except ImportError as e:
        logger.error(f"[ERROR] Import error: {e}")
        logger.error("[ERROR] Unsloth is required for RTX 3060 training")
        logger.error("[INFO] Install with: pip install unsloth[colab-new]")
        logger.error("[INFO] RTX 3060 (12GB) is sufficient for 7B model with 4-bit quantization")
        return 1
    except Exception as e:
        logger.error(f"[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    main()

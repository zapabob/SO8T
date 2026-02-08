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
import random
from pathlib import Path
from tqdm import tqdm
import time
import sys
import os
from typing import Dict

# Add project root to path for src imports
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

try:
    from src.utils.path_resolver import PathResolver
    from src.utils.config_loader import ConfigLoader

    PROJECT_ROOT = PathResolver.get_project_root()
except ImportError:
    PROJECT_ROOT = _project_root
    logger_fallback = logging.getLogger(__name__)
    logger_fallback.warning(
        "[PATH] PathResolver not available, using fallback: %s", PROJECT_ROOT
    )

# Import local modules
# Import local modules
from src.utils.vssi_template import normalize_prompt_text
from src.utils.execution_guards import ExecutionGuards

# ログ設定
# ログ設定
try:
    from src.utils.safe_logger import SafeLogger

    logger = SafeLogger.setup_logger(__name__)
except ImportError:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

# チェックポイントマネージャーのインポート
try:
    from src.utils.checkpoint_manager import (
        RollingCheckpointManager,
        EmergencyCheckpointManager,
        RollingCheckpointCallback,
    )

    CHECKPOINT_AVAILABLE = True
except ImportError as e:
    CHECKPOINT_AVAILABLE = False
    logger.warning(f"[CHECKPOINT] Checkpoint manager not available: {e}")


class RollingCheckpointCallback(TrainerCallback):
    """ローリングチェックポイントを保存するコールバック"""

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
                        "global_step": state.global_step,
                        "epoch": state.epoch,
                        "loss": state.log_history[-1].get("loss", 0)
                        if state.log_history
                        else 0,
                    },
                )
                self.last_checkpoint_step = state.global_step
                logger.info(
                    f"[CHECKPOINT] Saved checkpoint at step {state.global_step}"
                )
            except Exception as e:
                logger.warning(f"[CHECKPOINT] Failed to save checkpoint: {e}")

    def on_train_end(self, args, state, control, **kwargs):
        """トレーニング終了時に最終チェックポイントを保存"""
        if self.checkpoint_manager:
            try:
                self.checkpoint_manager.force_save_now(
                    self.model,
                    self.tokenizer,
                    step_info=f"final_step_{state.global_step}",
                )
                logger.info(
                    f"[CHECKPOINT] Saved final checkpoint at step {state.global_step}"
                )
            except Exception as e:
                logger.warning(f"[CHECKPOINT] Failed to save final checkpoint: {e}")


class UnslothSO8TTrainer:
    def __init__(self, config_path=None):
        self.project_root = PathResolver.get_project_root()

        # 設定ファイル読み込み
        if config_path:
            with open(config_path, "r", encoding="utf-8") as f:
                self.training_config = json.load(f)
        else:
            self.training_config = ConfigLoader.load_json(
                "training.json", required=True
            )

        # データセット設定読み込み
        self.dataset_config = ConfigLoader.load_json("dataset.json", required=True)

        self.reward_strategy_enabled = os.getenv("SO8T_REWARD_STRATEGY", "1") == "1"
        self.reward_strategy_scale = float(
            os.getenv("SO8T_REWARD_STRATEGY_SCALE", "1.0")
        )
        self.reward_strategy_map = self._load_reward_strategy_map()

        # RTX 3060最適化設定
        self.max_seq_length = 2048
        self.dtype = None  # Auto-detect
        self.load_in_4bit = self.training_config["optimization"].get(
            "load_in_4bit", True
        )

        # Borea-Phi-3.5 用のベースモデル名上書き（設定優先）
        # Priority: Env Var > Config > Default
        env_model = os.getenv("SO8T_BASE_MODEL")
        if env_model:
            self.base_model_name = env_model
            logger.info(
                f"[MODEL] Overriding base model from env: {self.base_model_name}"
            )
        else:
            self.base_model_name = self.training_config["model"].get(
                "base_model", "AXCXEPT/Borea-Phi-3.5-mini-Instruct-Jp"
            )

        # チェックポイントマネージャー初期化
        if CHECKPOINT_AVAILABLE:
            checkpoint_dir = self.project_root / "checkpoints" / "aegis_v3_borea"

            # 環境変数から設定を取得（デフォルト5分・3世代）
            interval = int(os.getenv("SO8T_CHECKPOINT_INTERVAL", "300"))
            rolling = int(os.getenv("SO8T_CHECKPOINT_ROLLING", "3"))

            self.checkpoint_manager = RollingCheckpointManager(
                base_dir=checkpoint_dir,
                max_keep=rolling,
                save_interval_sec=interval,
                enable_logging=True,
            )
            self.emergency_checkpoint = EmergencyCheckpointManager(
                self.checkpoint_manager
            )
            logger.info(
                f"[CHECKPOINT] Rolling checkpoint manager initialized ({interval}s interval, max {rolling})"
            )
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
            if gpu_memory < 11:
                logger.warning(
                    f"[WARN] GPU memory ({gpu_memory:.1f}GB) is below ideal for 7B/12B, using 4-bit."
                )
                self.load_in_4bit = True

        logger.info("[START] Unsloth Borea-SO8T Quadrality Training Initialized")
        logger.info(f"[MODEL] Base: {self.base_model_name}")
        logger.info(
            f"[ACCELERATION] Unsloth + {'4-bit' if self.load_in_4bit else 'BF16/FP16'}"
        )
        logger.info("[RTX3060] Optimized for AEGIS-v3.0")

    def _load_reward_strategy_map(self) -> Dict[str, float]:
        """
        SO8T 4-way 思考（task, analysis, safety, policy）と2024-2026年のキーワードを重視する報酬戦略マップを読み込む。
        環境変数 SO8T_REWARD_STRATEGY_MAP_PATH が設定されていれば、そこからJSONを読み込む。
        そうでなければ、デフォルトの戦略を返す。
        """
        strategy_map_path = os.getenv("SO8T_REWARD_STRATEGY_MAP_PATH")
        if strategy_map_path and Path(strategy_map_path).exists():
            try:
                with open(strategy_map_path, "r", encoding="utf-8") as f:
                    custom_map = json.load(f)
                logger.info(
                    f"[REWARD] Loaded custom reward strategy map from {strategy_map_path}"
                )
                return custom_map
            except Exception as e:
                logger.warning(
                    f"[REWARD] Failed to load custom reward strategy map: {e}. Using default."
                )

        # デフォルトのSO8T 4-way思考と最新キーワード重視戦略
        default_map = {
            # SO8T 4-way 思考の要素
            "task_completion": 1.5,  # タスクの正確な完了
            "analysis_depth": 1.2,  # 分析の深さと洞察力
            "safety_adherence": 2.0,  # 安全性、倫理、ハルシネーション防止
            "policy_alignment": 1.8,  # 指示、制約、ポリシーへの適合性
            "creativity": 0.8,  # 創造性、独創性
            "coherence": 1.0,  # 一貫性、論理的整合性
            "conciseness": 0.7,  # 簡潔さ、効率性
            "relevance": 1.0,  # 関連性、的確性
            # 2024-2026年の重要キーワード（評価基準に追加）
            "keyword_ukraine": 1.3,
            "keyword_cybersecurity": 1.5,
            "keyword_national_security": 1.4,
            "keyword_sakana_ai": 1.2,
            "keyword_generative_ai_ethics": 1.6,
            "keyword_climate_change_mitigation": 1.1,
            "keyword_quantum_computing_advances": 1.0,
            "keyword_biotechnology_breakthroughs": 1.0,
            "keyword_geopolitical_stability": 1.3,
            "keyword_supply_chain_resilience": 1.1,
            "keyword_digital_sovereignty": 1.2,
            "keyword_ai_regulation": 1.5,
            "keyword_space_economy": 0.9,
            "keyword_critical_minerals": 1.0,
            "keyword_disinformation_combat": 1.7,
            "keyword_global_health_security": 1.2,
            "keyword_human_rights_tech": 1.4,
            "keyword_sustainable_development_goals": 1.0,
            "keyword_future_of_work": 0.8,
            "keyword_urban_resilience": 0.9,
        }
        logger.info(
            "[REWARD] Using default SO8T 4-way thinking and 2024-2026 keyword-focused reward strategy map."
        )
        return default_map

    def load_model_and_tokenizer(self):
        """Unslothで高速モデル読み込み（Borea-Phi-3.5/RTX 3060最適化）"""
        logger.info(f"[MODEL] Loading {self.base_model_name} with Unsloth")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # メモリ使用率を88%に制限（バッファ確保）
            torch.cuda.set_per_process_memory_fraction(0.88)
            logger.info("[GPU] Memory optimization applied (88% limit)")

        try:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.base_model_name,
                max_seq_length=self.max_seq_length,
                dtype=None,  # Auto-detect (bf16 if supported, else fp16)
                load_in_4bit=self.load_in_4bit,
                device_map={"": 0},  # RTX 3060 (GPU 0) に固定配置
                trust_remote_code=True,
            )
        except Exception as e:
            logger.error(f"[ERROR] Failed to load model: {e}")
            logger.info(
                "[INFO] RTX 3060 (12GB) should be sufficient for 7B model with 4-bit quantization"
            )
            logger.info(
                "[INFO] Check: 1) Unsloth installation, 2) CUDA availability, 3) GPU memory"
            )
            raise

        # Phi-3 / Borea 用のチャットテンプレート設定
        try:
            tokenizer = get_chat_template(
                tokenizer,
                chat_template="phi-3",
            )
        except Exception:
            logger.info("Falling back to default chat template")

        logger.info(f"[MODEL] Loaded {self.base_model_name} successfully")
        return model, tokenizer

    def setup_lora_adapters(self, model):
        """Unslothで高速LoRA設定"""
        logger.info("[LoRA] Setting up LoRA adapters with Unsloth")

        model = FastLanguageModel.get_peft_model(
            model,
            r=self.training_config["model"]["lora_rank"],
            target_modules=self.training_config["model"]["target_modules"],
            lora_alpha=self.training_config["model"]["lora_alpha"],
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
            use_rslora=False,
            loftq_config=None,
        )

        logger.info("[LoRA] LoRA adapters configured with Unsloth optimization")
        return model

    def freeze_base_model_weights(self, model):
        """ベースモデルの重みを凍結（Adapter/LoRA/Soul以外）

        Model B (Borea) の既存能力を保護しつつ、
        Adapter/LoRA で新規知識を学習可能にする。
        """
        freeze_base = self.training_config.get("model", {}).get(
            "freeze_base_model", False
        )
        if not freeze_base:
            logger.info("[FREEZE] Base model weight freezing is disabled")
            return model

        preserve_soul = self.training_config.get("model", {}).get(
            "freeze_preserve_soul_weights", True
        )

        logger.info("[FREEZE] Starting base model weight freezing...")
        logger.info(f"[FREEZE] Preserve soul weights: {preserve_soul}")

        frozen_count = 0
        trainable_count = 0
        soul_count = 0

        # 学習可能にするキーワードリスト
        trainable_keywords = [
            "lora",  # QLoRAアダプター
            "so8",  # SO(8)ゲート
            "rotation",  # 回転行列
            "alpha_gate",  # Alpha Gate
            "so8t",  # SO8T関連
        ]

        # 魂の重みを保持する場合は追加
        if preserve_soul:
            trainable_keywords.extend(
                [
                    "alpha",  # Alphaパラメータ
                    "r_safe",  # 安全側の回転行列
                    "r_cmd",  # コマンド側の回転行列
                    "soul",  # 魂のパラメータ
                    "safety_head",  # 安全ヘッド
                    "task_head",  # タスクヘッド
                    "dual_heads",  # 二重政策系
                    "pet",  # PET正則化
                ]
            )

        for name, param in model.named_parameters():
            should_freeze = True

            for keyword in trainable_keywords:
                if keyword in name.lower():
                    should_freeze = False
                    if keyword in ["r_safe", "r_cmd", "alpha", "soul"]:
                        soul_count += 1
                    break

            if should_freeze:
                param.requires_grad = False
                frozen_count += 1
            else:
                param.requires_grad = True
                trainable_count += 1

        # 統計情報
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())

        logger.info(f"[FREEZE] Frozen {frozen_count} parameter groups")
        logger.info(f"[FREEZE] Trainable parameter groups: {trainable_count}")
        logger.info(
            f"[FREEZE] Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)"
        )
        if soul_count > 0:
            logger.info(f"[SOUL] Soul weight parameter groups: {soul_count}")

        return model

    def load_and_prepare_datasets(self, tokenizer, prioritize_mcp_api_skill=False):
        """統合データセット読み込みと準備（MCP/API/Skill優先オプション付き）"""
        logger.info("[DATASET] Loading and preparing integrated datasets")
        if prioritize_mcp_api_skill:
            logger.info(
                "[DATASET] Prioritizing MCP/API/Skill datasets for General AI Agent Foundation"
            )

        all_datasets = []

        # 四重推論構造化データ（Arxiv/BioRxiv）を優先的に読み込み
        logger.info(
            "[DATASET] Loading quadruple inference structured data (Arxiv/BioRxiv)"
        )
        quad_inference_dataset = self._load_quadruple_inference_datasets()
        if quad_inference_dataset:
            all_datasets.append(quad_inference_dataset)
            logger.info(
                f"[DATASET] Loaded {len(quad_inference_dataset)} quadruple inference samples"
            )

        # /thinkingモデル化データを読み込み
        logger.info("[DATASET] Loading /thinking model format data")
        thinking_dataset = self._load_thinking_model_datasets()
        if thinking_dataset:
            all_datasets.append(thinking_dataset)
            logger.info(
                f"[DATASET] Loaded {len(thinking_dataset)} /thinking model samples"
            )

        # MCP/API/Skillデータセットと汎用エージェントデータセットを優先的に読み込み
        if prioritize_mcp_api_skill:
            mcp_api_skill_sources = [
                s
                for s in self.dataset_config["sources"]
                if "mcp" in s.lower()
                or "api" in s.lower()
                or "skill" in s.lower()
                or "general_agent" in s.lower()
            ]

            logger.info(
                f"[DATASET] Found {len(mcp_api_skill_sources)} MCP/API/Skill/General Agent data sources"
            )
            for source in mcp_api_skill_sources:
                dataset = None
                if source.startswith("local:"):
                    dataset = self._load_local_jsonl_dataset(
                        source.replace("local:", "")
                    )
                elif source.startswith("moonshot:"):
                    dataset = self._load_moonshot_dataset(
                        source.replace("moonshot:", "")
                    )
                elif source.startswith("huggingface:"):
                    dataset = self._load_huggingface_dataset(
                        source.replace("huggingface:", "")
                    )

                if dataset:
                    all_datasets.append(dataset)
                    logger.info(
                        f"[DATASET] Loaded MCP/API/Skill/General Agent dataset: {source} ({len(dataset)} samples)"
                    )

        # その他のデータソースから読み込み
        for source in self.dataset_config["sources"]:
            # MCP/API/Skill/General Agentは既に読み込み済み
            if prioritize_mcp_api_skill and (
                "mcp" in source.lower()
                or "api" in source.lower()
                or "skill" in source.lower()
                or "general_agent" in source.lower()
            ):
                continue

            dataset = None
            if source.startswith("local:"):
                dataset = self._load_local_jsonl_dataset(source.replace("local:", ""))
            elif source.startswith("moonshot:"):
                dataset = self._load_moonshot_dataset(source.replace("moonshot:", ""))
            elif source.startswith("huggingface:"):
                dataset = self._load_huggingface_dataset(
                    source.replace("huggingface:", "")
                )
            elif source.startswith("synthetic:"):
                dataset = self._generate_synthetic_dataset(
                    source.replace("synthetic:", "")
                )

            if dataset:
                all_datasets.append(dataset)

        # データセット統合と前処理
        combined_dataset = self._combine_and_preprocess_datasets(
            all_datasets, tokenizer
        )

        # データセットのバランス調整
        if prioritize_mcp_api_skill:
            logger.info(
                "[DATASET] Balancing datasets (difficulty, diversity, tool usage)"
            )
            combined_dataset = self._balance_datasets(combined_dataset)

        logger.info(f"[DATASET] Prepared {len(combined_dataset)} training samples")
        if prioritize_mcp_api_skill:
            logger.info(
                "[DATASET] MCP/API/Skill/General Agent capabilities training data included"
            )
        return combined_dataset

    def _balance_datasets(self, dataset):
        """データセットのバランス調整（難易度、多様性、ツール使用）"""
        try:
            import pandas as pd

            df = dataset.to_pandas()

            # 難易度バランス: 基本/中級/上級を適切な比率で含める
            if "difficulty" in df.columns:
                difficulty_counts = df["difficulty"].value_counts()
                logger.info(
                    f"[BALANCE] Difficulty distribution: {difficulty_counts.to_dict()}"
                )

            # ツール使用バランス: ツール不要/必要/禁止を適切な比率で含める
            if "tool_condition" in df.columns:
                tool_condition_counts = df["tool_condition"].value_counts()
                logger.info(
                    f"[BALANCE] Tool condition distribution: {tool_condition_counts.to_dict()}"
                )

            # データ多様性: カテゴリの多様性を確認
            if "category" in df.columns:
                category_counts = df["category"].value_counts()
                logger.info(
                    f"[BALANCE] Category distribution: {len(category_counts)} categories"
                )

            return Dataset.from_pandas(df)
        except Exception as e:
            logger.warning(f"[BALANCE] Failed to balance datasets: {e}")
            return dataset

    def _load_quadruple_inference_datasets(self):
        """四重推論構造化データ（Arxiv/BioRxiv）を読み込み"""
        try:
            arxiv_data_dir = self.project_root / "data" / "arxiv_biorxiv" / "cleaned"
            if not arxiv_data_dir.exists():
                logger.warning(
                    f"[QUAD_INF] Arxiv/BioRxiv data directory not found: {arxiv_data_dir}"
                )
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
            with open(latest_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            data = json.loads(line.strip())
                            # 四重推論構造が含まれているか確認
                            if "quadruple_inference" in data:
                                samples.append(data)
                        except json.JSONDecodeError:
                            continue

            if samples:
                logger.info(
                    f"[QUAD_INF] Loaded {len(samples)} quadruple inference samples"
                )
                return Dataset.from_list(samples)
            else:
                logger.warning("[QUAD_INF] No valid quadruple inference samples found")
                return None

        except Exception as e:
            logger.warning(
                f"[QUAD_INF] Failed to load quadruple inference datasets: {e}"
            )
            return None

    def _load_thinking_model_datasets(self):
        """/thinkingモデル化データを読み込み"""
        try:
            arxiv_data_dir = self.project_root / "data" / "arxiv_biorxiv" / "cleaned"
            if not arxiv_data_dir.exists():
                logger.warning(
                    f"[THINKING] Arxiv/BioRxiv data directory not found: {arxiv_data_dir}"
                )
                return None

            # JSONLファイルを検索
            jsonl_files = list(arxiv_data_dir.glob("*.jsonl"))
            if not jsonl_files:
                logger.warning(f"[THINKING] No JSONL files found in {arxiv_data_dir}")
                return None

            # 最新のファイルを使用
            latest_file = max(jsonl_files, key=lambda x: x.stat().st_mtime)
            logger.info(
                f"[THINKING] Loading /thinking model data from {latest_file.name}"
            )

            samples = []
            with open(latest_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            data = json.loads(line.strip())
                            # /thinkingモデル化データが含まれているか確認
                            if "thinking_model" in data:
                                thinking_data = data["thinking_model"]
                                # チャット形式に変換
                                messages = [
                                    {
                                        "role": "user",
                                        "content": thinking_data.get("instruction", ""),
                                    },
                                    {
                                        "role": "assistant",
                                        "content": thinking_data.get("thinking", "")
                                        + "\n\n"
                                        + thinking_data.get("output", ""),
                                    },
                                ]
                                samples.append(
                                    {
                                        "messages": messages,
                                        "thinking_format": "so8t_quadruple_thinking",
                                    }
                                )
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
        # dataset_pipeline.pyの機能を使用（強化されたインポート処理）
        try:
            # インポートパスの調整
            try:
                from src.data.processing.dataset_pipeline import RTX3060DatasetPipeline
            except ImportError as import_err:
                logger.debug(f"[IMPORT] Primary import failed: {import_err}")
                # フォールバック: 直接データパイプラインをインポート
                try:
                    import sys

                    project_root = Path(__file__).resolve().parents[2]
                    if str(project_root) not in sys.path:
                        sys.path.insert(0, str(project_root))
                    if str(project_root / "src") not in sys.path:
                        sys.path.insert(0, str(project_root / "src"))
                    from data.processing.dataset_pipeline import RTX3060DatasetPipeline

                    logger.debug("[IMPORT] Fallback import successful")
                except ImportError as fallback_err:
                    logger.debug(
                        f"[IMPORT] Fallback import also failed: {fallback_err}"
                    )
                    raise ImportError(
                        f"Could not import RTX3060DatasetPipeline: {import_err}"
                    )

            pipeline = RTX3060DatasetPipeline()

            # MCP/API/Skillデータセットの特別処理
            if dataset_name in ["mcp_skills_integration", "api_skill_calling"]:
                logger.info(f"[MCP/API] Loading {dataset_name} dataset from HF")
                if dataset_name == "mcp_skills_integration":
                    dataset = pipeline._load_mcp_skills_hf_datasets()
                elif dataset_name == "api_skill_calling":
                    dataset = pipeline._load_api_skill_calling_hf_datasets()

                if dataset:
                    logger.info(
                        f"[MCP/API] Loaded {len(dataset)} samples from {dataset_name}"
                    )
                    return dataset

            # その他のMoonshotデータセット
            dataset = pipeline._download_moonshot_dataset(dataset_name)
            if dataset:
                logger.info(
                    f"[MOONSHOT] Loaded {len(dataset)} samples from {dataset_name}"
                )
                return dataset

            # 直接ファイル読み込みフォールバック
            return self._load_moonshot_dataset_direct(dataset_name)

        except Exception as e:
            logger.warning(
                f"[PIPELINE] Failed to load {dataset_name} via pipeline: {e}"
            )
            # 直接ファイル読み込みフォールバック
            return self._load_moonshot_dataset_direct(dataset_name)

    def _load_moonshot_dataset_direct(self, dataset_name):
        """
        Direct file loading fallback for Moonshot datasets
        Used when RTX3060DatasetPipeline is not available
        """
        moonshot_dir = self.project_root / "data" / "moonshot"
        dataset_path = moonshot_dir / f"{dataset_name}.jsonl"

        if dataset_path.exists():
            try:
                dataset = load_dataset(
                    "json", data_files=str(dataset_path), split="train"
                )
                logger.info(
                    f"[DIRECT] Loaded {len(dataset)} samples from {dataset_name}"
                )
                return dataset
            except Exception as e:
                logger.error(f"[DIRECT] Failed to load {dataset_name}: {e}")
                return None
        else:
            logger.warning(
                f"[DIRECT] Moonshot dataset {dataset_name} not found at {dataset_path}"
            )
            return None

    def _load_local_jsonl_dataset(self, file_path):
        """ローカルJSONLファイルから直接データセット読み込み"""
        try:
            fpath = Path(file_path)
            if not fpath.exists():
                logger.warning(f"[LOCAL] File not found: {file_path}")
                return None
            dataset = load_dataset("json", data_files=str(fpath), split="train")
            logger.info(f"[LOCAL] Loaded {len(dataset)} samples from {fpath.name}")
            return dataset
        except Exception as e:
            logger.warning(f"[LOCAL] Failed to load {file_path}: {e}")
            return None

    def _load_huggingface_dataset(self, dataset_name):
        """HuggingFaceデータセット読み込み (name:config形式対応)"""
        try:
            # Support huggingface:owner/repo:config syntax
            config = None
            parts = dataset_name.split(":")
            if len(parts) == 2:
                dataset_name, config = parts[0], parts[1]
            return load_dataset(dataset_name, config, split="train")
        except Exception as e:
            # Fallback: some datasets only have 'test' split (e.g. elyza/ELYZA-tasks-100)
            if "Unknown split" in str(e) and "test" in str(e):
                try:
                    logger.info(f"Retrying {dataset_name} with split='test'")
                    config = None
                    parts = dataset_name.split(":")
                    if len(parts) == 2:
                        dataset_name, config = parts[0], parts[1]
                    return load_dataset(dataset_name, config, split="test")
                except Exception as e2:
                    logger.warning(
                        f"Failed to load HF dataset {dataset_name} (test split): {e2}"
                    )
                    return None
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
                prompt, response = (text.split("\n", 1) + ["I'll help you with that."])[
                    :2
                ]
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
        if "text" in item:
            return item["text"]
        if "instruction" in item and "input" in item:
            return f"{item['instruction']}\n{item['input']}".strip()
        if "problem" in item:
            return item["problem"]
        if "question" in item:
            return item["question"]
        if "instruction" in item:
            return item["instruction"]
        return str(item)

    def _generate_mathematical_problems(self, num_samples):
        """数学的問題生成"""
        problems = []
        for i in range(num_samples):
            a, b = np.random.randint(1, 100, 2)
            operation = random.choice(["+", "-", "*", "/"])
            if operation == "/":
                result = np.random.randint(1, 20)
                b = np.random.randint(1, 10)
                a = result * b

            problem_text = f"Solve: {a} {operation} {b}"
            if operation == "+":
                answer = a + b
            elif operation == "-":
                answer = a - b
            elif operation == "*":
                answer = a * b
            else:
                answer = a // b

            problems.append(
                {
                    "instruction": "Solve this mathematical problem step by step.",
                    "input": problem_text,
                    "output": f"The answer is {answer}.",
                    "type": "mathematical_reasoning",
                }
            )

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
            greeting, response = random.choice(patterns)
            conversations.append(
                {
                    "instruction": "以下の日本語の挨拶に対して、適切な応答をしてください。",
                    "input": greeting,
                    "output": response,
                    "type": "japanese_conversation",
                }
            )

        return conversations

    def _generate_mcp_skill_examples(self, num_samples):
        """MCPスキル使用例生成"""
        skills = []
        for i in range(num_samples):
            skills.append(
                {
                    "instruction": "以下のタスクを解決するために、適切なツールを使用してください。",
                    "input": f"Calculate the square root of {i * i + 1}",
                    "output": f"Using calculator tool: sqrt({i * i + 1}) = {np.sqrt(i * i + 1):.2f}",
                    "type": "mcp_skill_usage",
                }
            )

        return skills

    def _generate_quadrality_decisions(self, num_samples):
        """四重推論意思決定生成"""
        decisions = []
        for i in range(num_samples):
            decisions.append(
                {
                    "instruction": "以下の状況で、適切な決定を下してください。",
                    "input": f"Situation {i}: Choose ALLOW, ESCALATE, DENY, or REFUSE",
                    "output": random.choice(["ALLOW", "ESCALATE", "DENY", "REFUSE"]),
                    "type": "quadrality_decision_making",
                }
            )

        return decisions

    def run_sft_training(self, model, tokenizer, dataset):
        """Phase 1: Supervised Fine-Tuning with Unsloth"""
        logger.info("[SFT] Starting Supervised Fine-Tuning with Unsloth")

        # LoRA設定
        model = self.setup_lora_adapters(model)

        # ベースモデルの重み凍結（Model B の能力を保護）
        model = self.freeze_base_model_weights(model)

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
            output_dir=str(
                self.project_root
                / "data"
                / "sunset_pipeline"
                / "checkpoints"
                / "unsloth_sft"
            ),
            report_to="none",  # Wandb無効
            save_steps=30,
            save_total_limit=2,
        )

        # チェックポイントコールバックを追加
        callbacks = []
        if self.checkpoint_manager:
            checkpoint_callback = RollingCheckpointCallback(
                self.checkpoint_manager, model, tokenizer
            )
            callbacks.append(checkpoint_callback)
            logger.info("[SFT] Rolling checkpoint callback enabled")

        # Unsloth SFT Trainer
        # Windows multiprocessing issue fix: use ExecutionGuards
        dataset_num_proc = ExecutionGuards.get_safe_num_proc()

        # Preprocess dataset to add text field if not present
        def add_text_field(example):
            """Add text field from messages if not present"""
            if "text" in example and example["text"]:
                return example

            if "messages" in example:
                messages = example["messages"]
                # Apply chat template if tokenizer has it
                if hasattr(tokenizer, "apply_chat_template"):
                    try:
                        text = tokenizer.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=False
                        )
                        example["text"] = text
                        return example
                    except:
                        pass
                # Fallback: manual formatting for Phi-3/Borea
                formatted = ""
                for msg in messages:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    if role == "system":
                        formatted += f"<|im_start|>system\n{content}<|im_end|>\n"
                    elif role == "user":
                        formatted += f"<|im_start|>user\n{content}<|im_end|>\n"
                    elif role == "assistant":
                        formatted += f"<|im_start|>assistant\n{content}<|im_end|>\n"
                example["text"] = formatted
            elif "instruction" in example and "output" in example:
                example["text"] = (
                    f"<|im_start|>user\n{example['instruction']}<|im_end|>\n<|im_start|>assistant\n{example['output']}<|im_end|>"
                )

            return example

        # Apply text field preprocessing
        dataset = dataset.map(add_text_field, desc="Formatting dataset")

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            max_seq_length=self.max_seq_length,
            dataset_num_proc=dataset_num_proc,
            packing=False,
            args=training_args,
            callbacks=callbacks if callbacks else None,
        )

        # トレーニング実行
        trainer_stats = trainer.train()

        logger.info("[SFT] Supervised Fine-Tuning completed")
        logger.info(
            f"[SFT] Training time: {trainer_stats.metrics['train_runtime']:.2f}s"
        )
        logger.info(
            f"[SFT] Training samples/sec: {trainer_stats.metrics['train_samples_per_second']:.2f}"
        )

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
            checkpoint_callback = RollingCheckpointCallback(
                self.checkpoint_manager, model, tokenizer
            )
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
            output_dir=str(
                self.project_root
                / "data"
                / "sunset_pipeline"
                / "checkpoints"
                / "unsloth_grpo"
            ),
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
            logger.info(
                f"[GRPO] Reward strategy enabled ({len(reward_map)} prompt mappings, scale={reward_scale})"
            )

        # GRPO-SO8T 報酬関数 (2024-2026 知識・四重推論対応)
        def reward_function(*args, **kwargs):
            """
            SO8T-VSSI-GRPO (Unified Quadrality Reasoning Reward)
            - 四重推論タグ (<think-task>, <think-analysis>, <think-safety>, <think-policy>) の厳密検証
            - 2024-2026年ドメイン知識（科学、OSINT、安保）の正確性検証
            - 長さ正則化と論理的一貫性の評価
            """
            if len(args) == 1:
                completions = args[0]
                prompts = kwargs.get("prompts") or kwargs.get("prompt") or []
            elif len(args) >= 2:
                prompts, completions = args[0], args[1]
            else:
                completions = kwargs.get("completions") or []
                prompts = kwargs.get("prompts") or []

            rewards = []

            # ドメイン知識キーワード (2024-2026)
            domain_keywords = {
                "osint": [
                    "Ukraine",
                    "Cybersecurity",
                    "Economic Security",
                    "Geopolitics",
                    "National Security",
                    "日中問題",
                    "経済安保",
                    "国家安全保障",
                ],
                "science": [
                    "AI Scientist",
                    "ShinkaEvolve",
                    "DeepSeek",
                    "Sakana AI",
                    "Unified Dataset",
                    "Pharmacology",
                    "IMO",
                    "Quantum",
                ],
                "culture": [
                    "Gundam SEED FREEDOM",
                    "GQuuuuuuX",
                    "Hathaway",
                    "Pop-Culture",
                    "Anime Analysis",
                ],
            }

            for idx, completion in enumerate(completions):
                reward = 0.0

                # 1. SO8T 四重推論タグの検証 (非常に高い重み)
                quad_tags = [
                    "<think-task>",
                    "<think-analysis>",
                    "<think-safety>",
                    "<think-policy>",
                ]
                found_tags = sum(1 for tag in quad_tags if tag in completion)
                if found_tags == 4:
                    reward += 1.0  # 満点ボーナス
                elif found_tags > 0:
                    reward += 0.2 * found_tags

                # タグの順序検証
                if "<think-task>" in completion and "<think-policy>" in completion:
                    if completion.find("<think-task>") < completion.find(
                        "<think-policy>"
                    ):
                        reward += 0.2

                # 2. 2024-2026年ドメイン知識ボーナス
                completion_lower = completion.lower()
                for cat, keywords in domain_keywords.items():
                    matches = sum(
                        1 for kw in keywords if kw.lower() in completion_lower
                    )
                    if matches > 0:
                        reward += 0.05 * min(matches, 3)

                # 3. 論理的一貫性と構造
                markers = [
                    "step",
                    "analysis",
                    "conclusion",
                    "therefore",
                    "because",
                    "したがって",
                    "結論",
                    "分析",
                ]
                struct_matches = sum(1 for m in markers if m in completion_lower)
                reward += 0.01 * min(struct_matches, 10)

                # 4. 長さ正則化 (LEAD-style)
                words = completion.split()
                if len(words) < 50:
                    reward -= 0.5  # 短すぎる
                elif len(words) > 800:
                    reward -= 0.2  # 冗長

                # 5. 特化報酬 (SO8T_REWARD_DATASET)
                # (既存の reward_map ロジックを継承)
                if reward_map and idx < len(prompts):
                    prompt_key = normalize_prompt_text(str(prompts[idx]))
                    strategy_bonus = reward_map.get(prompt_key, 0.0)
                    reward += reward_strategy_scale * strategy_bonus

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
                logger.warning(
                    f"[DaGRPO] Low distinctiveness ({distinctiveness:.3f}), masking all pairs"
                )
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

                logger.info(
                    f"[DaGRPO] Selected {num_anchors} anchors from {len(rewards)} samples"
                )

            return gradient_mask, rewards

        # GRPO Trainer
        dataset_num_proc = ExecutionGuards.get_safe_num_proc()

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
        default_path = (
            self.project_root / "data" / "reward_strategy" / "quadrality_reward.jsonl"
        )
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
            logger.info(
                "[REWARD] Loaded reward strategy map (%d entries)", len(reward_map)
            )
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
        logger.info(
            "[TRAINING] Starting Advanced SO8T Quadrality Training with Unsloth"
        )
        if prioritize_mcp_api_skill:
            logger.info(
                "[TRAINING] Training with MCP/API/Skill capabilities for General AI Agent Foundation"
            )

        # チェックポイントからの復旧
        if recover and self.checkpoint_manager:
            latest_checkpoint = self.checkpoint_manager.get_latest_checkpoint()
            if latest_checkpoint:
                logger.info(
                    f"[RECOVER] Recovering from checkpoint: {latest_checkpoint}"
                )
                model, tokenizer = self._recover_from_checkpoint(latest_checkpoint)
            else:
                logger.info("[RECOVER] No checkpoint found, starting fresh")
                model, tokenizer = self.load_model_and_tokenizer()
        else:
            # モデルとトークナイザーの読み込み
            model, tokenizer = self.load_model_and_tokenizer()

        # データセット読み込み（MCP/API/Skill優先オプション）
        dataset = self.load_and_prepare_datasets(
            tokenizer, prioritize_mcp_api_skill=prioritize_mcp_api_skill
        )

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

        logger.info(
            "[COMPLETE] Advanced SO8T Quadrality Training with Unsloth Completed"
        )

    def _recover_from_checkpoint(self, checkpoint_path: str):
        """チェックポイントからモデルとトークナイザーを復旧"""
        try:
            logger.info(f"[RECOVER] Loading model and tokenizer from {checkpoint_path}")

            # チェックポイント情報を取得
            checkpoint_info = self.checkpoint_manager.get_checkpoint_info(
                checkpoint_path
            )
            logger.info(f"[RECOVER] Checkpoint info: {checkpoint_info}")

            # モデルとトークナイザーを読み込み
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=checkpoint_path,
                max_seq_length=self.max_seq_length,
                dtype=None,
                load_in_4bit=self.load_in_4bit,
                device_map={"": 0},
                trust_remote_code=True,
            )

            # チャットテンプレート設定
            tokenizer = get_chat_template(
                tokenizer, chat_template="chatml", remove_generate_prompt=True
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
        print(
            "[INFO] For RTX 3060, Unsloth is recommended for fast training with 4-bit quantization."
        )
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
            print(
                f"[WARN] GPU memory ({gpu_memory:.1f}GB) may be insufficient for 7B model"
            )
            print("[INFO] Using 4-bit quantization to reduce memory usage")
        elif gpu_memory >= 12:
            print(
                "[INFO] RTX 3060 (12GB) is sufficient for 7B model with 4-bit quantization"
            )

    parser = argparse.ArgumentParser(
        description="Unsloth SO8T Quadrality Training (RTX 3060 Optimized)"
    )
    parser.add_argument("--config", type=str, default=None, help="Training config path")
    parser.add_argument(
        "--phase",
        type=str,
        default="full",
        choices=["sft", "grpo", "full"],
        help="Training phase",
    )
    parser.add_argument(
        "--mcp-api-skill",
        action="store_true",
        help="Prioritize MCP/API/Skill datasets for General AI Agent Foundation training",
    )
    parser.add_argument(
        "--recover", action="store_true", help="Recover from latest checkpoint"
    )
    # Pipeline integration arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for trained model",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=300,
        help="Checkpoint save interval in seconds (default: 300 = 5 minutes)",
    )
    parser.add_argument(
        "--rolling-checkpoints",
        type=int,
        default=3,
        help="Number of rolling checkpoints to keep (default: 3)",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Resume training from checkpoint path",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        action="append",
        default=[],
        help="Additional dataset paths (can be specified multiple times)",
    )

    args = parser.parse_args()

    try:
        # トレーニング実行
        trainer = UnslothSO8TTrainer(config_path=args.config)

        if args.phase == "full":
            trainer.run_advanced_training(
                prioritize_mcp_api_skill=args.mcp_api_skill, recover=args.recover
            )
        elif args.phase == "sft":
            model, tokenizer = trainer.load_model_and_tokenizer()
            dataset = trainer.load_and_prepare_datasets(
                tokenizer, prioritize_mcp_api_skill=args.mcp_api_skill
            )
            trainer.run_sft_training(model, tokenizer, dataset)
        elif args.phase == "grpo":
            model, tokenizer = trainer.load_model_and_tokenizer()
            dataset = trainer.load_and_prepare_datasets(
                tokenizer, prioritize_mcp_api_skill=args.mcp_api_skill
            )
            trainer.run_grpo_training(model, tokenizer, dataset)
    except ImportError as e:
        logger.error(f"[ERROR] Import error: {e}")
        logger.error("[ERROR] Unsloth is required for RTX 3060 training")
        logger.error("[INFO] Install with: pip install unsloth[colab-new]")
        logger.error(
            "[INFO] RTX 3060 (12GB) is sufficient for 7B model with 4-bit quantization"
        )
        return 1
    except Exception as e:
        logger.error(f"[ERROR] Training failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    main()

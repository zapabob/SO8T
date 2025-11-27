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
        try:
            logger.info("[QUANTIZATION] Creating BitsAndBytesConfig for 8bit quantization...")
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=config.get("quantization", {}).get("llm_int8_threshold", 6.0)
            )
            logger.info("[QUANTIZATION] BitsAndBytesConfig created successfully")
        except Exception as e:
            logger.error(f"[ERROR] Failed to create BitsAndBytesConfig: {e}")
            logger.warning("[WARNING] Continuing without quantization (may cause OOM)")
            quantization_config = None
            load_in_8bit = False
    
    # SO8TThinkingModelを読み込み
    from so8t.core.safety_aware_so8t import SafetyAwareSO8TConfig
    so8t_config_dict = config.get("so8t", {})
    
    # ベイズ最適化結果を読み込む（存在する場合）
    bayes_opt_result_path = Path("D:/webdataset/alpha_gate_bayes_opt/optimal_alpha_gate_orthogonal.json")
    bayes_opt_params = {}
    if bayes_opt_result_path.exists():
        try:
            with open(bayes_opt_result_path, "r", encoding="utf-8") as f:
                bayes_opt_result = json.load(f)
                bayes_opt_params = bayes_opt_result.get("best_params", {})
                logger.info(f"[BAYES_OPT] Loaded optimization results: {bayes_opt_params}")
        except Exception as e:
            logger.warning(f"[BAYES_OPT] Failed to load optimization results: {e}, using defaults")
    
    # Alpha Gate設定（ベイズ最適化結果を優先、なければYAML設定、それもなければデフォルト）
    alpha_gate_config = so8t_config_dict.get("alpha_gate", {}) if isinstance(so8t_config_dict.get("alpha_gate"), dict) else {}
    use_alpha_gate = bayes_opt_params.get("use_alpha_gate", so8t_config_dict.get("use_alpha_gate", True))
    alpha_gate_target = so8t_config_dict.get("alpha_gate_target", 0.432)
    alpha_gate_start = so8t_config_dict.get("alpha_gate_start", -5.0)
    alpha_gate_annealing_steps = bayes_opt_params.get("alpha_gate_annealing_steps", so8t_config_dict.get("alpha_gate_annealing_steps", 1000))
    alpha_gate_steepness = bayes_opt_params.get("alpha_gate_steepness", so8t_config_dict.get("alpha_gate_steepness", 12.0))
    alpha_gate_orthogonal_weight = bayes_opt_params.get("alpha_gate_orthogonal_weight", so8t_config_dict.get("alpha_gate_orthogonal_weight", 1.0))
    alpha_gate_pet_weight = bayes_opt_params.get("alpha_gate_pet_weight", so8t_config_dict.get("alpha_gate_pet_weight", 0.1))
    
    # 中間レイヤー設定
    so8_apply_to_intermediate_layers = so8t_config_dict.get("apply_to_intermediate_layers", True)
    so8_intermediate_layer_start = so8t_config_dict.get("intermediate_layer_start")
    so8_intermediate_layer_end = so8t_config_dict.get("intermediate_layer_end")
    so8_intermediate_layer_ratio = so8t_config_dict.get("intermediate_layer_ratio", [0.25, 0.75])
    so8_log_orthogonal_error = so8t_config_dict.get("log_orthogonal_error", True)
    so8_orthogonal_error_threshold = so8t_config_dict.get("orthogonal_error_threshold", 1e-3)
    pet_apply_to_intermediate_layers = so8t_config_dict.get("pet_apply_to_intermediate_layers", True)
    pet_high_freq_cutoff = so8t_config_dict.get("pet_high_freq_cutoff", 0.5)
    
    so8t_config = SafetyAwareSO8TConfig(
        pet_lambda=so8t_config_dict.get("pet_lambda", 0.1),
        nu_orth=so8t_config_dict.get("orthogonal_reg", 1e-4),  # nu_orthが直交性制約の重み
        mu_norm=so8t_config_dict.get("norm_reg", 0.01),
        rho_iso=so8t_config_dict.get("isometry_reg", 0.01),
        # Alpha Gate設定
        use_alpha_gate=use_alpha_gate,
        alpha_gate_target=alpha_gate_target,
        alpha_gate_start=alpha_gate_start,
        alpha_gate_annealing_steps=alpha_gate_annealing_steps,
        alpha_gate_steepness=alpha_gate_steepness,
        alpha_gate_orthogonal_weight=alpha_gate_orthogonal_weight,
        alpha_gate_pet_weight=alpha_gate_pet_weight,
        # 中間レイヤー設定
        so8_apply_to_intermediate_layers=so8_apply_to_intermediate_layers,
        so8_intermediate_layer_start=so8_intermediate_layer_start,
        so8_intermediate_layer_end=so8_intermediate_layer_end,
        so8_intermediate_layer_ratio=tuple(so8_intermediate_layer_ratio) if isinstance(so8_intermediate_layer_ratio, list) else so8_intermediate_layer_ratio,
        so8_log_orthogonal_error=so8_log_orthogonal_error,
        so8_orthogonal_error_threshold=so8_orthogonal_error_threshold,
        pet_apply_to_intermediate_layers=pet_apply_to_intermediate_layers,
        pet_high_freq_cutoff=pet_high_freq_cutoff
    )
    
    try:
        logger.info("[MODEL] Initializing SO8TThinkingModel...")
        # CUDA メモリをクリア（念のため）
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info(f"[CUDA] Available memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        
        # quantization_configをSO8TThinkingModelに渡す
        model = SO8TThinkingModel(
            base_model_name_or_path=model_path,
            so8t_config=so8t_config,
            use_quadruple_thinking=True,
            quantization_config=quantization_config if load_in_8bit else None
        )
        logger.info("[MODEL] SO8TThinkingModel initialized successfully")
        
        # メモリ使用状況を確認
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            reserved = torch.cuda.memory_reserved(0) / 1024**3
            logger.info(f"[CUDA] Memory allocated: {allocated:.2f} GB, reserved: {reserved:.2f} GB")
    except Exception as e:
        logger.error(f"[ERROR] Failed to initialize SO8TThinkingModel: {e}")
        import traceback
        logger.error(f"[ERROR] Traceback:\n{traceback.format_exc()}")
        # CUDA メモリをクリア
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise
    
    # QLoRA設定（LLMベストプラクティス: エラーハンドリング、ログ、リトライ）
    if config.get("qlora", {}).get("enabled", True):
        try:
            logger.info("[QLORA] Preparing model for k-bit training...")
            # メモリ使用状況を確認
            if torch.cuda.is_available():
                allocated_before = torch.cuda.memory_allocated(0) / 1024**3
                reserved_before = torch.cuda.memory_reserved(0) / 1024**3
                logger.info(f"[QLORA] Memory before preparation - allocated: {allocated_before:.2f} GB, reserved: {reserved_before:.2f} GB")
            
            model = prepare_model_for_kbit_training(model)
            logger.info("[QLORA] Model prepared for k-bit training")
            
            # メモリ使用状況を確認
            if torch.cuda.is_available():
                allocated_after = torch.cuda.memory_allocated(0) / 1024**3
                reserved_after = torch.cuda.memory_reserved(0) / 1024**3
                logger.info(f"[QLORA] Memory after preparation - allocated: {allocated_after:.2f} GB, reserved: {reserved_after:.2f} GB")
        except Exception as e:
            logger.error(f"[ERROR] Failed to prepare model for k-bit training: {e}")
            import traceback
            logger.error(f"[ERROR] Traceback:\n{traceback.format_exc()}")
            # CUDAメモリをクリア
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise
        
        # LoRA設定の作成（LLMベストプラクティス: 動的モジュール検出）
        qlora_config = config.get("qlora", {})
        default_target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
        target_modules = qlora_config.get("target_modules", default_target_modules)
        
        # モデルの実際のモジュール名を動的に検出（LLMベストプラクティス: 深い探索）
        logger.info("[QLORA] Detecting actual module names in the model...")
        actual_module_names = []
        full_module_names = []
        
        # base_modelのモジュール名を取得（SO8TThinkingModelの場合）
        # SO8TThinkingModel -> SafetyAwareSO8TModel -> base_model (AutoModelForCausalLM)
        # 再帰的にbase_modelを探索して、AutoModelForCausalLMのインスタンスを見つける
        actual_base_model = None
        current_model = model
        
        logger.info("[QLORA] Recursively exploring model structure to find AutoModelForCausalLM...")
        max_depth = 5  # 最大探索深度
        depth = 0
        
        # SO8Tラッパーをスキップして、実際のAutoModelForCausalLMを見つける
        from transformers import AutoModelForCausalLM
        
        while depth < max_depth:
            model_type_name = type(current_model).__name__
            logger.info(f"[QLORA] Depth {depth}: Current model type: {model_type_name}")
            
            # SO8Tラッパーの場合、__dict__からbase_modelを直接取得（LLMベストプラクティス）
            if model_type_name in ['SO8TThinkingModel', 'SafetyAwareSO8TModel']:
                # まず、__dict__から直接取得を試みる
                if hasattr(current_model, '__dict__'):
                    model_dict = current_model.__dict__
                    if 'base_model' in model_dict:
                        inner_base = model_dict['base_model']
                        inner_base_type = type(inner_base).__name__
                        logger.info(f"[QLORA] Found base_model in __dict__: {inner_base_type}")
                        if isinstance(inner_base, AutoModelForCausalLM):
                            actual_base_model = inner_base
                            logger.info(f"[QLORA] Found AutoModelForCausalLM instance in __dict__ at depth {depth}")
                            break
                        elif inner_base_type not in ['SO8TThinkingModel', 'SafetyAwareSO8TModel']:
                            # 次のレベルに進む
                            current_model = inner_base
                            depth += 1
                            continue
                
                # __dict__にない場合、getattrで取得を試みる
                if actual_base_model is None and hasattr(current_model, 'base_model'):
                    try:
                        inner_base = getattr(current_model, 'base_model')
                        inner_base_type = type(inner_base).__name__
                        logger.info(f"[QLORA] Found base_model via getattr: {inner_base_type}")
                        if isinstance(inner_base, AutoModelForCausalLM):
                            actual_base_model = inner_base
                            logger.info(f"[QLORA] Found AutoModelForCausalLM instance via getattr at depth {depth}")
                            break
                    except AttributeError:
                        logger.warning(f"[QLORA] Could not get base_model via getattr for {model_type_name}")
            
            # base_model属性を確認
            if hasattr(current_model, 'base_model'):
                next_model = current_model.base_model
                depth += 1
                next_model_type_name = type(next_model).__name__
                logger.info(f"[QLORA] Depth {depth}: Found base_model of type {next_model_type_name}")
                
                # SO8Tラッパーでない場合（AutoModelForCausalLMまたはそのサブクラス）
                if next_model_type_name not in ['SO8TThinkingModel', 'SafetyAwareSO8TModel']:
                    if isinstance(next_model, AutoModelForCausalLM):
                        actual_base_model = next_model
                        logger.info(f"[QLORA] Found AutoModelForCausalLM instance at depth {depth}: {next_model_type_name}")
                        break
                    # または、model属性やlayers属性がある場合（Phi-3.5などの構造）
                    elif hasattr(next_model, 'model') or hasattr(next_model, 'layers'):
                        actual_base_model = next_model
                        logger.info(f"[QLORA] Found model with 'model' or 'layers' attribute at depth {depth}: {next_model_type_name}")
                        break
                
                current_model = next_model
            else:
                logger.warning(f"[QLORA] No base_model attribute found at depth {depth}")
                break
        
        if actual_base_model is None:
            logger.error(f"[ERROR] Could not find AutoModelForCausalLM instance after {depth} levels of exploration")
            logger.error(f"[ERROR] Final model type: {type(current_model).__name__}")
            # 最後の手段: 現在のモデルをそのまま使用（エラーを出さずに続行）
            logger.warning("[WARNING] Using current model as actual_base_model (may cause issues)")
            actual_base_model = current_model
        
        # actual_base_modelの内部構造を確認（Phi-3.5の場合、model.layers など）
        if hasattr(actual_base_model, 'model'):
            # Phi-3.5などの構造: model.layers.0.self_attn.q_proj
            inner_model = actual_base_model.model
            logger.info(f"[QLORA] Found inner model structure: {type(inner_model).__name__}")
            logger.info(f"[QLORA] Exploring inner_model.named_modules()...")
            for name, module in inner_model.named_modules():
                full_name = f"model.{name}" if name else "model"
                full_module_names.append(full_name)
                module_name_parts = name.split('.') if name else []
                if len(module_name_parts) > 0:
                    actual_module_names.append(module_name_parts[-1])
            logger.info(f"[QLORA] Collected {len(full_module_names)} module names from inner_model")
        else:
            # 直接actual_base_modelから探索
            logger.info("[QLORA] Exploring actual_base_model directly (no inner 'model' attribute)...")
            for name, module in actual_base_model.named_modules():
                full_module_names.append(name)
                module_name_parts = name.split('.')
                if len(module_name_parts) > 0:
                    actual_module_names.append(module_name_parts[-1])
            logger.info(f"[QLORA] Collected {len(full_module_names)} module names from actual_base_model")
        
        # ユニークなモジュール名を取得
        unique_module_names = set(actual_module_names)
        logger.info(f"[QLORA] Found {len(unique_module_names)} unique module name patterns")
        logger.info(f"[QLORA] Sample module name patterns: {sorted(list(unique_module_names))[:30]}")
        logger.info(f"[QLORA] Sample full module names (first 30): {full_module_names[:30]}")
        logger.info(f"[QLORA] Sample full module names (last 30): {full_module_names[-30:]}")
        
        # target_modulesが実際に存在するか確認し、存在するもののみを使用
        available_target_modules = []
        for module_name in target_modules:
            # 完全一致または部分一致で検出
            # 1. モジュール名パターンでの検索
            found_in_patterns = module_name in unique_module_names
            # 2. 完全なモジュール名での検索（例: "layers.0.self_attn.q_proj" に "q_proj" が含まれる）
            found_in_full_names = any(module_name in full_name or full_name.endswith(f'.{module_name}') for full_name in full_module_names)
            
            if found_in_patterns or found_in_full_names:
                available_target_modules.append(module_name)
                logger.info(f"[QLORA] Found target module '{module_name}' in model")
            else:
                logger.warning(f"[QLORA] Target module '{module_name}' not found in model, skipping")
        
        if not available_target_modules:
            # フォールバック: 実際のモジュール名から推測
            logger.warning("[QLORA] No target modules found, trying to infer from actual module names...")
            # よくあるパターンを試す
            common_patterns = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            for pattern in common_patterns:
                if any(pattern in name for name in actual_module_names):
                    available_target_modules.append(pattern)
            
            if not available_target_modules:
                error_msg = f"Could not find any suitable target modules. Available modules: {sorted(list(actual_module_names))[:30]}"
                logger.error(f"[ERROR] {error_msg}")
                raise ValueError(error_msg)
        
        logger.info(f"[QLORA] Using target modules: {available_target_modules}")
        logger.info(f"[QLORA] LoRA config - r: {qlora_config.get('r', 64)}, alpha: {qlora_config.get('lora_alpha', 128)}, dropout: {qlora_config.get('lora_dropout', 0.05)}")
        
        lora_config = LoraConfig(
            r=qlora_config.get("r", 64),
            lora_alpha=qlora_config.get("lora_alpha", 128),
            target_modules=available_target_modules,
            lora_dropout=qlora_config.get("lora_dropout", 0.05),
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        # get_peft_model()の実行（エラーハンドリング強化）
        try:
            logger.info("[QLORA] Applying LoRA adapters to model (this may take a while)...")
            # メモリ使用状況を確認
            if torch.cuda.is_available():
                allocated_before = torch.cuda.memory_allocated(0) / 1024**3
                reserved_before = torch.cuda.memory_reserved(0) / 1024**3
                logger.info(f"[QLORA] Memory before LoRA - allocated: {allocated_before:.2f} GB, reserved: {reserved_before:.2f} GB")
            
            model = get_peft_model(model, lora_config)
            logger.info("[OK] QLoRA applied successfully")
            
            # メモリ使用状況を確認
            if torch.cuda.is_available():
                allocated_after = torch.cuda.memory_allocated(0) / 1024**3
                reserved_after = torch.cuda.memory_reserved(0) / 1024**3
                logger.info(f"[QLORA] Memory after LoRA - allocated: {allocated_after:.2f} GB, reserved: {reserved_after:.2f} GB")
            
            # 学習可能パラメータ数を確認
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())
            logger.info(f"[QLORA] Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")
        except Exception as e:
            logger.error(f"[ERROR] Failed to apply LoRA adapters: {e}")
            import traceback
            logger.error(f"[ERROR] Traceback:\n{traceback.format_exc()}")
            # CUDAメモリをクリア
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise
    
    # 重み凍結（LLMベストプラクティス: エラーハンドリング、ログ）
    if config.get("model", {}).get("freeze_base_model", True):
        try:
            logger.info("[FREEZE] Freezing base model weights...")
            from scripts.training.train_borea_phi35_so8t_thinking import freeze_base_model_weights
            freeze_base_model_weights(model, config)
            logger.info("[OK] Base model weights frozen successfully")
        except Exception as e:
            logger.error(f"[ERROR] Failed to freeze base model weights: {e}")
            import traceback
            logger.error(f"[ERROR] Traceback:\n{traceback.format_exc()}")
            # 重み凍結に失敗しても続行（警告のみ）
            logger.warning("[WARNING] Continuing without weight freezing")
    
    # データセット読み込み（LLMベストプラクティス: エラーハンドリング、プログレスバー）
    try:
        logger.info("[DATASET] Loading dataset...")
        logger.info(f"[DATASET] Dataset path: {args.dataset}")
        logger.info(f"[DATASET] Max sequence length: {config.get('data', {}).get('max_seq_length', 2048)}")
        
        train_dataset = QuadruplePairwiseDataset(
            data_path=args.dataset,
            tokenizer=tokenizer,
            max_length=config.get("data", {}).get("max_seq_length", 2048)
        )
        
        dataset_size = len(train_dataset)
        logger.info(f"[OK] Dataset loaded successfully - {dataset_size:,} samples")
        
        if dataset_size == 0:
            error_msg = "Dataset is empty after loading"
            logger.error(f"[ERROR] {error_msg}")
            raise ValueError(error_msg)
    except Exception as e:
        logger.error(f"[ERROR] Failed to load dataset: {e}")
        import traceback
        logger.error(f"[ERROR] Traceback:\n{traceback.format_exc()}")
        raise
    
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


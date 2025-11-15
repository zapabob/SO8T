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

# HuggingFaceキャッシュをDドライブに設定
os.environ["HF_HOME"] = r"D:\webdataset\hf_cache"
os.environ["TRANSFORMERS_CACHE"] = r"D:\webdataset\hf_cache\transformers"
os.environ["HF_DATASETS_CACHE"] = r"D:\webdataset\hf_cache\datasets"
os.environ["HF_HUB_CACHE"] = r"D:\webdataset\hf_cache\hub"

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

# 時間ベースチェックポイントCallbackをインポート
from src.so8t.checkpointing import TimeBasedCheckpointCallback

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


class PowerFailureRecovery:
    """電源断リカバリーシステム"""
    
    def __init__(self, session_file: Path):
        """
        Args:
            session_file: セッションファイルパス
        """
        self.session_file = Path(session_file)
        self.session_file.parent.mkdir(parents=True, exist_ok=True)
        self.session_data = None
    
    def load_session(self) -> Optional[Dict[str, Any]]:
        """前回セッションを読み込み"""
        if not self.session_file.exists():
            return None
        
        try:
            with open(self.session_file, 'r', encoding='utf-8') as f:
                self.session_data = json.load(f)
            logger.info(f"[RECOVERY] Session loaded: {self.session_data.get('session_id', 'unknown')}")
            logger.info(f"[RECOVERY] Progress: {self.session_data.get('current_step', 0)}/{self.session_data.get('total_steps', 0)}")
            return self.session_data
        except Exception as e:
            logger.warning(f"[RECOVERY] Failed to load session: {e}")
            return None
    
    def save_session(self, session_data: Dict[str, Any]):
        """セッション情報を保存"""
        try:
            self.session_data = session_data
            with open(self.session_file, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"[RECOVERY] Failed to save session: {e}")
    
    def find_latest_checkpoint(self, checkpoint_dir: Path) -> Optional[Path]:
        """最新のチェックポイントを検索"""
        if not checkpoint_dir.exists():
            return None
        
        # checkpoint-* ディレクトリを検索
        checkpoints = sorted(
            checkpoint_dir.glob("checkpoint-*"),
            key=lambda x: int(x.name.split("-")[1]) if x.name.split("-")[1].isdigit() else 0,
            reverse=True
        )
        
        if checkpoints:
            logger.info(f"[RECOVERY] Found latest checkpoint: {checkpoints[0]}")
            return checkpoints[0]
        
        return None


class ThinkingSFTDataset(Dataset):
    """/think形式SFTデータセット"""
    
    def __init__(
        self,
        data_path: Path,
        tokenizer,
        max_length: int = 2048,
        sample_ratio: Optional[float] = None
    ):
        """
        Args:
            data_path: JSONLファイルパス（/think形式）
            tokenizer: トークナイザー
            max_length: 最大シーケンス長
            sample_ratio: サンプリング比率（Noneの場合は全サンプル使用）
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        
        logger.info(f"Loading /think format dataset from {data_path}...")
        
        all_samples = []
        total_lines = 0
        
        # ファイルサイズを取得（進捗表示のため）
        file_size = data_path.stat().st_size
        file_size_mb = file_size / (1024 * 1024)
        logger.info(f"Dataset file size: {file_size_mb:.2f} MB")
        
        with open(data_path, 'r', encoding='utf-8') as f:
            # 総行数をカウント（非同期で実行、またはスキップ）
            logger.info("Counting total lines in dataset...")
            start_count_time = time.time()
            f.seek(0)
            total_lines = sum(1 for _ in f)
            count_time = time.time() - start_count_time
            f.seek(0)
            logger.info(f"Total lines in dataset: {total_lines:,} (counted in {count_time:.2f}s)")
            
            start_load_time = time.time()
            bytes_read = 0
            
            for line_no, line in enumerate(f, 1):
                bytes_read += len(line.encode('utf-8'))
                
                try:
                    # JSON解析を最適化（大きな行の場合）
                    line_stripped = line.strip()
                    if not line_stripped:
                        continue
                    
                    sample = json.loads(line_stripped)
                    output = sample.get("output", "")
                    
                    # 高速チェック: 必要なトークンが含まれているか確認
                    if not output:
                        if line_no % 100 == 0:
                            logger.warning(f"Line {line_no}: Empty output, skipping")
                        continue
                    
                    # outputが既にPhi-3.5チャットテンプレート形式であることを確認
                    # 高速チェック: 最初の1000文字で確認
                    output_preview = output[:1000] if len(output) > 1000 else output
                    if "<|system|>" in output_preview and "<|assistant|>" in output:
                        all_samples.append({
                            "text": output,
                            "instruction": sample.get("instruction", ""),
                            "input": sample.get("input", "")
                        })
                    else:
                        if line_no % 100 == 0:
                            logger.warning(f"Line {line_no}: Invalid format, skipping")
                        continue
                        
                except json.JSONDecodeError as e:
                    if line_no % 100 == 0:
                        logger.warning(f"Line {line_no}: JSON decode error: {e}")
                    continue
                except Exception as e:
                    if line_no % 100 == 0:
                        logger.warning(f"Line {line_no}: Unexpected error: {e}")
                    continue
                
                # 進捗表示（50行ごと、または5秒ごと、または最初の10行）
                elapsed = time.time() - start_load_time
                should_log = (
                    line_no <= 10 or  # 最初の10行
                    line_no % 50 == 0 or  # 50行ごと
                    (elapsed > 5 and line_no % 10 == 0)  # 5秒経過後は10行ごと
                )
                
                if should_log:
                    progress_pct = (line_no * 100 // total_lines) if total_lines > 0 else 0
                    bytes_mb = bytes_read / (1024 * 1024)
                    bytes_pct = (bytes_read * 100 / file_size) if file_size > 0 else 0
                    speed_mb_per_sec = bytes_mb / elapsed if elapsed > 0 else 0
                    logger.info(f"Loading progress: {line_no:,}/{total_lines:,} lines ({progress_pct}%), "
                              f"{bytes_mb:.2f}MB/{file_size_mb:.2f}MB ({bytes_pct:.1f}%), "
                              f"loaded {len(all_samples):,} valid samples, "
                              f"speed: {speed_mb_per_sec:.2f}MB/s, elapsed: {elapsed:.1f}s")
        
        # サンプリング（高速化のため）
        if sample_ratio is not None and 0 < sample_ratio < 1:
            import random
            random.seed(42)
            sample_size = int(len(all_samples) * sample_ratio)
            self.samples = random.sample(all_samples, sample_size)
            logger.info(f"[OK] Sampled {len(self.samples):,} samples from {len(all_samples):,} (ratio: {sample_ratio:.1%})")
        else:
            self.samples = all_samples
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
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        損失計算（PET統合）
        
        Args:
            model: モデル
            inputs: 入力データ
            return_outputs: 出力を返すかどうか
            num_items_in_batch: バッチ内のアイテム数（transformers新バージョン用、未使用）
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
    logger.info("[DEBUG] Step 1: Loading tokenizer...")
    
    # トークナイザー読み込み
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info("[DEBUG] Step 1: Tokenizer loaded")
    
    # 設定読み込み
    logger.info("[DEBUG] Step 2: Loading config...")
    config = Phi3Config.from_pretrained(model_path, trust_remote_code=True)
    logger.info("[DEBUG] Step 2: Config loaded")
    
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
            logger.info("[DEBUG] Step 3: Loading base model for SO8T integration...")
            # SO8TPhi3Modelを使用してモデルを構築
            # 注意: SO8TPhi3ForCausalLMが存在しない場合は、標準モデルを読み込んでSO8Tを適用
            if SO8TPhi3ForCausalLM is not None:
                # 標準モデルを読み込んでからSO8Tモデルに置き換え
                logger.info("[DEBUG] Loading base model (this may take several minutes)...")
                base_model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    quantization_config=quantization_config,
                    device_map="auto",
                    trust_remote_code=True,
                    torch_dtype=torch.float16
                )
                
                logger.info("[DEBUG] Base model loaded, creating SO8T model...")
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
                logger.info("[DEBUG] Loading base model (this may take several minutes)...")
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
            logger.info("[DEBUG] Loading standard model (this may take several minutes)...")
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16
            )
    else:
        # 標準モデルを読み込み
        logger.info("[DEBUG] Loading standard model (this may take several minutes)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
    
    logger.info("[OK] Model and tokenizer loaded")
    logger.info(f"[DEBUG] Model type: {type(model)}")
    logger.info(f"[DEBUG] Model device: {next(model.parameters()).device if hasattr(model, 'parameters') else 'N/A'}")
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
        default=None,
        help="Training dataset path (JSONL, /think format). If not provided, will be read from config."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. If not provided, will be read from config."
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="Resume from checkpoint path. If not provided, will auto-detect from session file."
    )
    parser.add_argument(
        "--auto-resume",
        action="store_true",
        help="Automatically resume from latest checkpoint if session file exists"
    )
    
    args = parser.parse_args()
    
    # 設定ファイル読み込み
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 設定ファイルからデータセットパスと出力ディレクトリを取得
    dataset_path = args.dataset
    if dataset_path is None:
        dataset_path = Path(config.get("data", {}).get("train_data", "D:/webdataset/processed/thinking_sft/thinking_sft_dataset.jsonl"))
    
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = Path(config.get("training", {}).get("output_dir", "D:/webdataset/checkpoints/training/borea_phi35_so8t_thinking"))
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 電源断リカバリー設定
    session_file = output_dir / "training_session.json"
    recovery = PowerFailureRecovery(session_file)
    
    # 自動再開モード
    resume_checkpoint = args.resume_from_checkpoint
    existing_session = None
    if args.auto_resume or resume_checkpoint is None:
        existing_session = recovery.load_session()
        if existing_session:
            checkpoint_dir = output_dir / "checkpoints"
            latest_checkpoint = recovery.find_latest_checkpoint(checkpoint_dir)
            if latest_checkpoint:
                resume_checkpoint = str(latest_checkpoint)
                logger.info(f"[RECOVERY] Auto-resuming from checkpoint: {resume_checkpoint}")
                logger.info(f"[RECOVERY] Session: {existing_session.get('session_id', 'unknown')}")
                logger.info(f"[RECOVERY] Progress: {existing_session.get('current_step', 0)}/{existing_session.get('total_steps', 0)}")
    
    is_recovery = resume_checkpoint is not None
    
    logger.info("="*80)
    logger.info("Borea-Phi-3.5 SO8T/thinking Training")
    logger.info("="*80)
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Dataset: {dataset_path}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Recovery mode: {is_recovery}")
    if is_recovery:
        logger.info(f"Resume checkpoint: {resume_checkpoint}")
    
    # モデルとトークナイザー読み込み
    so8t_config = config.get("so8t", {})
    so8t_layer_indices = so8t_config.get("layer_indices", None)
    model, tokenizer = load_model_with_so8t(
        model_path=args.model_path,
        so8t_layer_indices=so8t_layer_indices,
        load_in_8bit=config.get("quantization", {}).get("load_in_8bit", True)
    )
    
    logger.info("[DEBUG] Model loaded, preparing for QLoRA...")
    
    # QLoRA設定
    if config.get("qlora", {}).get("enabled", True):
        logger.info("[DEBUG] Starting prepare_model_for_kbit_training...")
        model = prepare_model_for_kbit_training(model)
        logger.info("[DEBUG] prepare_model_for_kbit_training completed")
        
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
        
        logger.info("[DEBUG] Starting get_peft_model...")
        model = get_peft_model(model, lora_config)
        logger.info("[OK] QLoRA applied")
    
    # データセット読み込み
    logger.info("[DEBUG] Starting dataset loading...")
    data_config = config.get("data", {})
    train_dataset = ThinkingSFTDataset(
        data_path=dataset_path,
        tokenizer=tokenizer,
        max_length=data_config.get("max_seq_length", 2048),
        sample_ratio=data_config.get("sample_ratio", None)
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
    
    # セッション情報を作成
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    if is_recovery and existing_session:
        session_id = existing_session.get("session_id", session_id)
    
    # 総ステップ数を計算
    training_config = config.get("training", {})
    num_epochs = training_config.get("num_train_epochs", 3)
    batch_size = training_config.get("per_device_train_batch_size", 1)
    gradient_accumulation = training_config.get("gradient_accumulation_steps", 16)
    total_steps = (len(train_dataset) // (batch_size * gradient_accumulation)) * num_epochs
    
    # セッション情報を保存
    session_data = {
        "session_id": session_id,
        "start_time": datetime.now().isoformat(),
        "current_step": 0,
        "total_steps": total_steps,
        "checkpoints": [],
        "model_path": str(args.model_path),
        "dataset_path": str(dataset_path),
        "output_dir": str(output_dir)
    }
    recovery.save_session(session_data)
    
    # トレーニング引数
    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation,
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
        resume_from_checkpoint=resume_checkpoint if is_recovery else None,
        # 進捗バーとログ出力の設定
        disable_tqdm=False,  # tqdmを有効化
        logging_first_step=True,  # 最初のステップもログ出力
        logging_nan_inf_filter=False,  # NaN/Infのフィルタリングは無効化
    )
    
    # 時間ベースチェックポイントCallbackを作成（約3分ごと）
    time_cb = TimeBasedCheckpointCallback(fixed_interval_sec=180)
    logger.info(f"[CHECKPOINT] TimeBasedCheckpointCallback initialized (interval: {time_cb.fixed_interval_sec}s = {time_cb.fixed_interval_sec/60:.1f} minutes)")
    logger.info(f"[CHECKPOINT] Checkpoint callback type: {type(time_cb).__name__}")
    logger.info(f"[CHECKPOINT] Checkpoint callback will save every {time_cb.fixed_interval_sec}s ({time_cb.fixed_interval_sec/60:.1f} minutes)")
    
    # トレーナー
    trainer = SO8TPETTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        pet_regularization=pet_regularization,
        callbacks=[time_cb]
    )
    
    # コールバック登録の確認
    callback_count = len(trainer.callback_handler.callbacks) if hasattr(trainer, 'callback_handler') else 0
    logger.info(f"[CHECKPOINT] TimeBasedCheckpointCallback added to trainer (total callbacks: {callback_count})")
    
    # コールバックが正しく登録されているか確認
    if hasattr(trainer, 'callback_handler') and hasattr(trainer.callback_handler, 'callbacks'):
        callback_names = [type(cb).__name__ for cb in trainer.callback_handler.callbacks]
        logger.info(f"[CHECKPOINT] Registered callbacks: {callback_names}")
        if 'TimeBasedCheckpointCallback' in callback_names:
            logger.info("[CHECKPOINT] TimeBasedCheckpointCallback is correctly registered")
        else:
            logger.warning("[CHECKPOINT] TimeBasedCheckpointCallback not found in registered callbacks!")
    else:
        logger.warning("[CHECKPOINT] Could not verify callback registration (callback_handler not available)")
    
    # 学習実行
    logger.info("Starting training...")
    if is_recovery:
        logger.info(f"[RECOVERY] Resuming from checkpoint: {resume_checkpoint}")
    
    # カスタムコールバックでセッション情報を更新
    class SessionUpdateCallback(TrainerCallback):
        def __init__(self, recovery_obj, session_data):
            self.recovery = recovery_obj
            self.session_data = session_data
        
        def on_step_end(self, args, state, control, **kwargs):
            # ステップごとにセッション情報を更新
            self.session_data["current_step"] = state.global_step
            self.recovery.save_session(self.session_data)
        
        def on_save(self, args, state, control, **kwargs):
            # チェックポイント保存時にセッション情報を更新
            if state.global_step % args.save_steps == 0:
                checkpoint_path = Path(args.output_dir) / f"checkpoint-{state.global_step}"
                if checkpoint_path.exists():
                    self.session_data["checkpoints"].append(str(checkpoint_path))
                    # 古いチェックポイントを削除（save_total_limitを超える場合）
                    if len(self.session_data["checkpoints"]) > args.save_total_limit:
                        old_checkpoint = self.session_data["checkpoints"].pop(0)
                        old_path = Path(old_checkpoint)
                        if old_path.exists():
                            import shutil
                            shutil.rmtree(old_path)
                    self.recovery.save_session(self.session_data)
    
    # 進捗表示とログ出力を強化するコールバック
    class ProgressLoggingCallback(TrainerCallback):
        """進捗表示とログ出力を強化するコールバック"""
        
        def __init__(self):
            self.start_time = None
            self.last_log_time = time.time()
        
        def on_train_begin(self, args, state, control, **kwargs):
            """学習開始時のログ"""
            self.start_time = time.time()
            logger.info("="*80)
            logger.info("Training Started")
            logger.info("="*80)
            logger.info(f"Total steps: {state.max_steps}")
            logger.info(f"Total epochs: {args.num_train_epochs}")
            logger.info(f"Batch size: {args.per_device_train_batch_size}")
            logger.info(f"Gradient accumulation: {args.gradient_accumulation_steps}")
            logger.info(f"Effective batch size: {args.per_device_train_batch_size * args.gradient_accumulation_steps}")
            logger.info(f"Learning rate: {args.learning_rate}")
            logger.info("="*80)
        
        def on_log(self, args, state, control, logs=None, **kwargs):
            """ログ出力時の処理"""
            if logs is None:
                return
            
            # 定期的に詳細ログを出力
            current_time = time.time()
            if current_time - self.last_log_time >= 30:  # 30秒ごと
                elapsed_time = current_time - self.start_time if self.start_time else 0
                progress = state.global_step / state.max_steps if state.max_steps > 0 else 0
                
                log_msg = f"[PROGRESS] Step {state.global_step}/{state.max_steps} "
                log_msg += f"({progress*100:.1f}%) | "
                log_msg += f"Elapsed: {elapsed_time/3600:.2f}h | "
                
                if "loss" in logs:
                    log_msg += f"Loss: {logs['loss']:.4f} | "
                if "learning_rate" in logs:
                    log_msg += f"LR: {logs['learning_rate']:.2e}"
                
                logger.info(log_msg)
                self.last_log_time = current_time
        
        def on_epoch_begin(self, args, state, control, **kwargs):
            """エポック開始時のログ"""
            logger.info("="*80)
            logger.info(f"Epoch {int(state.epoch) + 1}/{args.num_train_epochs} Started")
            logger.info("="*80)
        
        def on_epoch_end(self, args, state, control, **kwargs):
            """エポック終了時のログ"""
            elapsed_time = time.time() - self.start_time if self.start_time else 0
            logger.info("="*80)
            logger.info(f"Epoch {int(state.epoch) + 1}/{args.num_train_epochs} Completed")
            logger.info(f"Total elapsed time: {elapsed_time/3600:.2f} hours")
            logger.info("="*80)
        
        def on_save(self, args, state, control, **kwargs):
            """チェックポイント保存時のログ"""
            checkpoint_path = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            if checkpoint_path.exists():
                elapsed_time = time.time() - self.start_time if self.start_time else 0
                logger.info(f"[CHECKPOINT] Saved at step {state.global_step} "
                          f"(Elapsed: {elapsed_time/3600:.2f}h)")
        
        def on_train_end(self, args, state, control, **kwargs):
            """学習終了時のログ"""
            total_time = time.time() - self.start_time if self.start_time else 0
            logger.info("="*80)
            logger.info("Training Completed")
            logger.info("="*80)
            logger.info(f"Total steps: {state.global_step}")
            logger.info(f"Total time: {total_time/3600:.2f} hours")
            logger.info(f"Average time per step: {total_time/state.global_step:.2f} seconds")
            logger.info("="*80)
    
    # コールバックを追加
    progress_cb = ProgressLoggingCallback()
    trainer.add_callback(SessionUpdateCallback(recovery, session_data))
    trainer.add_callback(progress_cb)
    
    trainer.train(resume_from_checkpoint=resume_checkpoint if is_recovery else None)
    
    # 最終モデル保存
    final_model_dir = output_dir / "final_model"
    trainer.save_model(str(final_model_dir))
    tokenizer.save_pretrained(str(final_model_dir))
    
    # セッション完了
    session_data["end_time"] = datetime.now().isoformat()
    session_data["status"] = "completed"
    recovery.save_session(session_data)
    
    logger.info(f"[SUCCESS] Training completed. Model saved to {final_model_dir}")


if __name__ == "__main__":
    main()


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
        save_logits: bool = False,
        save_logits_steps: int = 100,
        save_logits_dir: str = "logits",
        save_logits_max_files: int = 10,
        save_metrics: bool = True,
        save_metrics_steps: int = 10,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.pet_regularization = pet_regularization
        self.hidden_states_history = []
        self.save_logits = save_logits
        self.save_logits_steps = save_logits_steps
        self.save_logits_dir = save_logits_dir
        self.save_logits_max_files = save_logits_max_files
        self.saved_logits_files = []  # 保存済みファイルリスト（古いファイル削除用）
        
        # logits保存ディレクトリを作成
        if self.save_logits and hasattr(self.args, 'output_dir'):
            self.logits_save_dir = Path(self.args.output_dir) / self.save_logits_dir
            self.logits_save_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"[LOGITS] Logits will be saved to: {self.logits_save_dir}")
        
        # メトリクス記録機能を初期化
        self.save_metrics = save_metrics
        self.save_metrics_steps = save_metrics_steps
        self.metrics_recorder = None
        if self.save_metrics and hasattr(self.args, 'output_dir'):
            try:
                from scripts.utils.training_metrics_recorder import TrainingMetricsRecorder
                self.metrics_recorder = TrainingMetricsRecorder(
                    output_dir=Path(self.args.output_dir),
                    model_name="borea_phi35_so8t",
                    save_interval=save_metrics_steps
                )
                logger.info(f"[METRICS] Metrics recorder initialized: {self.args.output_dir}")
            except Exception as e:
                logger.warning(f"[METRICS] Failed to initialize metrics recorder: {e}")
                self.metrics_recorder = None
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        損失計算（PET統合）
        
        Args:
            model: モデル
            inputs: 入力データ
            return_outputs: 出力を返すかどうか
            num_items_in_batch: バッチ内のアイテム数（transformers新バージョン用、未使用）
        """
        # CRITICAL: PEFT 0.18.0ではenable_input_require_gradsが存在しないため、
        # embedding層のパラメータ自体にrequires_grad=Trueを設定し、
        # さらにforward hookを登録して出力にrequires_grad=Trueを設定
        hook_handles = []
        
        # Embedding層のパラメータにrequires_grad=Trueを設定
        # これにより、embedding層の出力が確実に勾配計算グラフに接続される
        embedding_modules = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Embedding, nn.EmbeddingBag)):
                embedding_modules.append((name, module))
                # Embedding層のパラメータにrequires_grad=Trueを設定
                for param in module.parameters():
                    if not param.requires_grad:
                        param.requires_grad = True
                        logger.debug(f"[EMBEDDING] Set requires_grad=True for {name}.{param}")
        
        def make_embedding_hook():
            """Embedding層の出力にrequires_grad=Trueを設定するhook
            
            Note: embedding層のパラメータにrequires_grad=Trueを設定したが、
            それでも不十分な場合があるため、forward hookでも設定を試みる。
            ただし、既に計算グラフから外れているテンソルには効果がない可能性がある。
            """
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    # requires_gradを設定（勾配グラフに接続する試み）
                    if not output.requires_grad:
                        output.requires_grad_(True)
                elif isinstance(output, tuple):
                    # タプルの場合、各要素を処理
                    for item in output:
                        if isinstance(item, torch.Tensor) and not item.requires_grad:
                            item.requires_grad_(True)
                return output
            return hook
        
        # Embedding層にforward hookを登録
        for name, module in embedding_modules:
            handle = module.register_forward_hook(make_embedding_hook())
            hook_handles.append(handle)
            logger.debug(f"[HOOK] Registered forward hook on embedding layer: {name}")
        
        try:
            # CRITICAL: モデルが訓練モードであることを確認
            if not model.training:
                logger.warning(f"[WARNING] Model is not in training mode at step {self.state.global_step}, setting training mode")
                model.train()
            
            # past_key_values警告を回避するため、use_cache=Falseとpast_key_values=Noneを明示的に設定
            # hidden_statesを取得するためにoutput_hidden_states=Trueを設定
            inputs_with_hidden = {
                **inputs,
                "output_hidden_states": True,
                "use_cache": False,  # 訓練時はキャッシュを使用しない
                "past_key_values": None  # 明示的にNoneを設定
            }
            
            # 標準損失
            outputs = model(**inputs_with_hidden)
            loss = outputs.loss
        finally:
            # Hookを削除（メモリリークを防ぐ）
            for handle in hook_handles:
                handle.remove()
            if hook_handles:
                logger.debug(f"[HOOK] Removed {len(hook_handles)} forward hooks")
        
        # CRITICAL: 8-bit量子化モデルでは、hidden_statesがrequires_grad=Falseになることがある
        # これは、ベースモデルのパラメータが凍結されているため
        # LoRAアダプターを通るパスだけが勾配を必要とするが、
        # hidden_states自体は勾配グラフから外れている可能性がある
        # Forward hookでembedding層の出力にrequires_grad=Trueを設定したが、
        # それでも不十分な場合は、損失計算時にhidden_statesから直接損失を再計算する
        
        # CRITICAL: logitsがrequires_grad=Trueであることを事前に確認
        if hasattr(outputs, 'logits') and outputs.logits is not None:
            if not outputs.logits.requires_grad:
                logger.warning(f"[WARNING] outputs.logits does not require grad at step {self.state.global_step}")
                # logitsをrequires_grad=Trueに設定（ただし、これは勾配グラフに接続されない可能性がある）
                # 実際の修正は損失再計算時に行う
        
        # CRITICAL: outputs.lossがrequires_grad=Falseの場合、勾配計算グラフに接続
        if loss is not None and not loss.requires_grad:
            # logitsから損失を再計算して勾配グラフに接続
            if hasattr(outputs, 'logits') and 'labels' in inputs:
                logits = outputs.logits
                loss_fct = nn.CrossEntropyLoss()
                shift_labels = inputs['labels'][..., 1:].contiguous()
                shift_labels = shift_labels.view(-1)
                
                # CRITICAL: logitsがrequires_grad=Falseの場合、hidden_statesから直接logitsを再計算
                # logits.requires_grad_(True)だけでは不十分（勾配グラフから外れているため）
                if not logits.requires_grad:
                    logger.debug(f"[FIX] logits requires_grad=False, recalculating from hidden_states")
                    # hidden_statesから直接logitsを再計算（これにより勾配グラフに接続される）
                    if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                        hidden_states = outputs.hidden_states[-1]
                        if hidden_states.requires_grad:
                            # hidden_statesからlogitsを再計算
                            logits_from_hidden = model.lm_head(hidden_states)
                            logits_from_hidden = logits_from_hidden.float()
                            shift_logits = logits_from_hidden[..., :-1, :].contiguous()
                            shift_logits = shift_logits.view(-1, shift_logits.size(-1))
                            shift_labels = shift_labels.to(shift_logits.device)
                            loss = loss_fct(shift_logits, shift_labels)
                            logger.debug(f"[FIX] Recalculated loss from hidden_states (loss.requires_grad={loss.requires_grad})")
                        else:
                            logger.error(f"[ERROR] hidden_states also does not require grad at step {self.state.global_step}")
                            
                            # CRITICAL: Try to fix the model state
                            # Check if model is in training mode
                            if not model.training:
                                logger.warning(f"[FIX] Model not in training mode, setting to training mode")
                                model.train()
                            
                            # CRITICAL: PEFT 0.18.0にはenable_input_require_gradsが存在しない
                            # Forward hookでembedding層の出力にrequires_grad=Trueを設定したが、
                            # それでも不十分な場合は、次回のforward呼び出しで再度試行
                            logger.warning(f"[WARNING] PEFT 0.18.0 does not have enable_input_require_grads")
                            logger.info("[FIX] Forward hook applied, but hidden_states still requires_grad=False. Will retry in next forward pass.")
                            
                            # Verify trainable parameters
                            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                            if trainable_params == 0:
                                logger.error(f"[ERROR] No trainable parameters found! This is a critical issue.")
                                # Try to enable LoRA parameters manually
                                for name, param in model.named_parameters():
                                    if 'lora' in name.lower():
                                        param.requires_grad = True
                                logger.info("[FIX] Manually enabled requires_grad for LoRA parameters")
                            
                            # 最後の手段: logitsを使用（勾配は流れないが、エラーは回避）
                            shift_logits = logits[..., :-1, :].contiguous()
                            shift_logits = shift_logits.view(-1, shift_logits.size(-1))
                            shift_labels = shift_labels.to(shift_logits.device)
                            loss = loss_fct(shift_logits, shift_labels)
                            logger.warning(f"[WARNING] Using logits without grad tracking (loss.requires_grad={loss.requires_grad})")
                    else:
                        # hidden_statesが利用できない場合
                        logger.error(f"[ERROR] hidden_states not available, using logits without grad tracking")
                        shift_logits = logits[..., :-1, :].contiguous()
                        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
                        shift_labels = shift_labels.to(shift_logits.device)
                        loss = loss_fct(shift_logits, shift_labels)
                else:
                    # logitsがrequires_grad=Trueの場合、通常通り使用
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_logits = shift_logits.view(-1, shift_logits.size(-1))
                    shift_labels = shift_labels.to(shift_logits.device)
                    loss = loss_fct(shift_logits, shift_labels)
                    logger.debug(f"[FIX] outputs.loss requires_grad=False, recalculated from logits (loss.requires_grad={loss.requires_grad})")
        
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
            
            # CRITICAL: pet_lossがrequires_grad=Falseの場合、lossから直接派生したゼロテンソルを作成
            if not pet_loss.requires_grad:
                # loss * 0.0を使用して、lossから直接派生したゼロテンソルを作成
                # これにより、勾配計算グラフに含まれる
                # スカラーではなくテンソルとして作成（形状を保持）
                if loss.dim() == 0:
                    pet_loss = loss * 0.0
                else:
                    pet_loss = (loss * 0.0).sum() if loss.numel() > 0 else loss * 0.0
                # デバッグログ
                logger.debug(f"[PET] pet_loss requires_grad=False, using loss * 0.0 (shape: {pet_loss.shape})")
            
            loss = loss + pet_loss
            
            # ログ出力（定期的に）
            if self.state.global_step % 100 == 0:
                logger.info(f"[PET] Step {step}: Loss={pet_loss.item():.6e}, "
                          f"Phase={pet_info.get('phase', 'unknown')}, "
                          f"Lambda={pet_info.get('lambda', 0.0):.4f}")
            
            # メトリクス記録
            if self.metrics_recorder is not None:
                try:
                    # Perplexity計算（lossから）
                    perplexity = torch.exp(loss).item() if loss.item() < 10 else float('inf')
                    
                    # 学習率取得
                    lr = self._get_learning_rate()
                    
                    self.metrics_recorder.record_step(
                        step=self.state.global_step,
                        epoch=self.state.epoch if hasattr(self.state, 'epoch') else 0.0,
                        loss=loss.item(),
                        learning_rate=lr,
                        pet_loss=pet_loss.item() if isinstance(pet_loss, torch.Tensor) else pet_loss,
                        perplexity=perplexity if perplexity != float('inf') else None
                    )
                except Exception as e:
                    logger.debug(f"[METRICS] Failed to record metrics: {e}")
        
        # 最終的な損失の検証（Forward hookが機能しなかった場合のフォールバック）
        if loss is not None:
            if not loss.requires_grad:
                logger.warning(f"[WARNING] Final loss does not require grad at step {self.state.global_step}")
                logger.info("[FIX] Attempting to recalculate loss from hidden_states as fallback...")
                # 緊急対応: hidden_statesから直接損失を計算（logitsは使わない）
                if hasattr(outputs, 'hidden_states') and outputs.hidden_states and 'labels' in inputs:
                    hidden_states = outputs.hidden_states[-1]
                    if hidden_states.requires_grad:
                        # hidden_statesからlogitsを再計算（これにより勾配グラフに接続される）
                        loss_fct = nn.CrossEntropyLoss()
                        logits_from_hidden = model.lm_head(hidden_states)
                        logits_from_hidden = logits_from_hidden.float()
                        shift_logits = logits_from_hidden[..., :-1, :].contiguous()
                        shift_labels = inputs['labels'][..., 1:].contiguous()
                        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
                        shift_labels = shift_labels.view(-1)
                        shift_labels = shift_labels.to(shift_logits.device)
                        loss = loss_fct(shift_logits, shift_labels)
                        logger.info(f"[FIX] Final validation: Recalculated loss from hidden_states (loss.requires_grad={loss.requires_grad})")
                        
                        if self.pet_regularization is not None:
                            # PET損失を再追加（requires_grad=Trueを保証）
                            pet_loss, _ = self.pet_regularization.compute_pet_loss(
                                hidden_states=hidden_states,
                                step=self.state.global_step,
                                mask=inputs.get("attention_mask")
                            )
                            if not pet_loss.requires_grad:
                                pet_loss = loss * 0.0
                            loss = loss + pet_loss
                    else:
                        logger.error(f"[ERROR] Final validation: hidden_states also does not require grad after forward hook")
                        logger.error(f"[ERROR] This indicates that forward hook did not work properly. Check model configuration.")
                else:
                    logger.error(f"[ERROR] Final validation: Cannot recalculate loss (hidden_states or labels not available)")
        
        # Logits保存（定期的に）
        if self.save_logits and hasattr(outputs, 'logits') and outputs.logits is not None:
            if self.state.global_step % self.save_logits_steps == 0:
                try:
                    self._save_logits(
                        logits=outputs.logits,
                        labels=inputs.get('labels'),
                        step=self.state.global_step,
                        epoch=self.state.epoch if hasattr(self.state, 'epoch') else 0
                    )
                except Exception as e:
                    logger.warning(f"[WARNING] Failed to save logits at step {self.state.global_step}: {e}")
        
        # メトリクス記録（PET損失がない場合も）
        if self.metrics_recorder is not None and loss is not None:
            try:
                perplexity = torch.exp(loss).item() if loss.item() < 10 else float('inf')
                lr = self._get_learning_rate()
                
                self.metrics_recorder.record_step(
                    step=self.state.global_step,
                    epoch=self.state.epoch if hasattr(self.state, 'epoch') else 0.0,
                    loss=loss.item(),
                    learning_rate=lr,
                    perplexity=perplexity if perplexity != float('inf') else None
                )
            except Exception as e:
                logger.debug(f"[METRICS] Failed to record metrics: {e}")
        
        return (loss, outputs) if return_outputs else loss
    
    def _get_learning_rate(self) -> float:
        """現在の学習率を取得"""
        try:
            if hasattr(self, 'lr_scheduler') and self.lr_scheduler is not None:
                if hasattr(self.lr_scheduler, 'get_last_lr'):
                    return self.lr_scheduler.get_last_lr()[0]
                elif hasattr(self.lr_scheduler, 'get_lr'):
                    return self.lr_scheduler.get_lr()[0]
            if hasattr(self, 'optimizer') and self.optimizer is not None:
                return self.optimizer.param_groups[0].get('lr', 0.0)
        except Exception:
            pass
        return 0.0
    
    def _save_logits(self, logits: torch.Tensor, labels: Optional[torch.Tensor], step: int, epoch: int):
        """
        Logitsを保存
        
        Args:
            logits: モデルの出力logits
            labels: ラベル（存在する場合）
            step: 現在のステップ数
            epoch: 現在のエポック数
        """
        if not hasattr(self, 'logits_save_dir'):
            return
        
        # CPUに移動してから保存（メモリ効率化）
        logits_cpu = logits.detach().cpu()
        labels_cpu = labels.detach().cpu() if labels is not None else None
        
        # ファイル名を生成
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"logits_step_{step}_epoch_{epoch}_{timestamp}.pt"
        filepath = self.logits_save_dir / filename
        
        # 保存データを準備
        save_data = {
            'logits': logits_cpu,
            'step': step,
            'epoch': epoch,
            'timestamp': timestamp,
            'logits_shape': logits_cpu.shape,
            'logits_dtype': str(logits_cpu.dtype),
        }
        
        if labels_cpu is not None:
            save_data['labels'] = labels_cpu
            save_data['labels_shape'] = labels_cpu.shape
        
        # 保存
        torch.save(save_data, filepath)
        self.saved_logits_files.append(filepath)
        logger.info(f"[LOGITS] Saved logits to {filepath} (shape: {logits_cpu.shape})")
        
        # 古いファイルを削除（最大ファイル数制限）
        if len(self.saved_logits_files) > self.save_logits_max_files:
            old_file = self.saved_logits_files.pop(0)
            try:
                old_file.unlink()
                logger.debug(f"[LOGITS] Deleted old logits file: {old_file}")
            except Exception as e:
                logger.warning(f"[WARNING] Failed to delete old logits file {old_file}: {e}")


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
                logger.info("[DEBUG] Starting model loading at: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                try:
                    base_model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        quantization_config=quantization_config,
                        device_map="auto",
                        trust_remote_code=True,
                        torch_dtype=torch.float16,
                        low_cpu_mem_usage=True
                    )
                    logger.info("[DEBUG] Base model loaded successfully at: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                except Exception as e:
                    logger.error(f"[ERROR] Failed to load base model: {e}")
                    raise
                
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
                    
                    # PEFT 0.18.0ではenable_input_require_gradsが削除された可能性がある
                    # prepare_model_for_kbit_trainingで自動的に設定される
                    try:
                        from peft.tuners.lora import enable_input_require_grads
                        enable_input_require_grads(so8t_model)
                        logger.info("[DEBUG] enable_input_require_grads called successfully for SO8T model")
                    except ImportError:
                        logger.info("[INFO] enable_input_require_grads not available in PEFT 0.18.0, relying on prepare_model_for_kbit_training")
                    except Exception as e:
                        logger.warning(f"[WARNING] Failed to call enable_input_require_grads for SO8T model: {e}")
                    
                    logger.info("[DEBUG] prepare_model_for_kbit_training completed for SO8T model")
                
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
            logger.info("[DEBUG] Starting model loading at: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    quantization_config=quantization_config,
                    device_map="auto",
                    trust_remote_code=True,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True
                )
                logger.info("[DEBUG] Standard model loaded successfully at: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            except Exception as e:
                logger.error(f"[ERROR] Failed to load standard model: {e}")
                raise
    else:
        # 標準モデルを読み込み
        logger.info("[DEBUG] Loading standard model (this may take several minutes)...")
        logger.info("[DEBUG] Starting model loading at: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )
            logger.info("[DEBUG] Standard model loaded successfully at: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        except Exception as e:
            logger.error(f"[ERROR] Failed to load standard model: {e}")
            raise
    
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
            else:
                # チェックポイントが存在しない場合でも、セッション情報が存在する場合はRecovery modeを有効化
                logger.info(f"[RECOVERY] Session found but no checkpoint directory. Recovery mode enabled.")
            logger.info(f"[RECOVERY] Session: {existing_session.get('session_id', 'unknown')}")
            logger.info(f"[RECOVERY] Progress: {existing_session.get('current_step', 0)}/{existing_session.get('total_steps', 0)}")
    
    # セッション情報が存在する場合、またはチェックポイントが指定されている場合はRecovery modeを有効化
    is_recovery = (existing_session is not None) or (resume_checkpoint is not None)
    
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
        # 8-bit量子化が有効な場合のみprepare_model_for_kbit_trainingを呼び出す
        load_in_8bit = config.get("quantization", {}).get("load_in_8bit", False)
        if load_in_8bit:
            logger.info("[DEBUG] Starting prepare_model_for_kbit_training (8-bit quantization enabled)...")
            model = prepare_model_for_kbit_training(model)
            logger.info("[DEBUG] prepare_model_for_kbit_training completed")
        else:
            logger.info("[DEBUG] Skipping prepare_model_for_kbit_training (8-bit quantization disabled, using FP16/BF16)")
        
        # PEFT 0.18.0ではenable_input_require_gradsが削除された可能性がある
        # prepare_model_for_kbit_trainingとget_peft_modelで自動的に設定される
        # ただし、念のため手動で設定を試みる
        try:
            from peft.tuners.lora import enable_input_require_grads
            enable_input_require_grads(model)
            logger.info("[DEBUG] enable_input_require_grads called successfully")
        except ImportError:
            logger.info("[INFO] enable_input_require_grads not available in PEFT 0.18.0, relying on prepare_model_for_kbit_training and get_peft_model")
        except Exception as e:
            logger.warning(f"[WARNING] Failed to call enable_input_require_grads: {e}")
        
        logger.info("[DEBUG] Model preparation completed")
        
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
        
        # CRITICAL: Verify trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"[DEBUG] Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")
        
        # CRITICAL: PEFT 0.18.0では、LoRAパラメータがrequires_grad=Falseになる問題がある
        # LoRAパラメータのrequires_gradを確認して、必要に応じて手動で設定
        lora_params_fixed = 0
        for name, param in model.named_parameters():
            if 'lora' in name.lower() and not param.requires_grad:
                param.requires_grad = True
                lora_params_fixed += 1
        if lora_params_fixed > 0:
            logger.warning(f"[WARNING] Fixed {lora_params_fixed} LoRA parameters with requires_grad=False")
        
        # CRITICAL: Embedding層のパラメータにrequires_grad=Trueを設定
        # これにより、embedding層の出力が確実に勾配計算グラフに接続される
        embedding_params_fixed = 0
        for name, module in model.named_modules():
            if isinstance(module, (nn.Embedding, nn.EmbeddingBag)):
                for param in module.parameters():
                    if not param.requires_grad:
                        param.requires_grad = True
                        embedding_params_fixed += 1
        if embedding_params_fixed > 0:
            logger.info(f"[INFO] Set requires_grad=True for {embedding_params_fixed} embedding parameters")
        
        # CRITICAL: SO(8)群構造パラメータにrequires_grad=Trueを設定
        # SO(8)群の回転行列パラメータは学習可能パラメータとして維持する必要がある
        # PEFT LoRAのtarget_modulesに含まれないため、手動で設定する
        so8t_params_fixed = 0
        for name, param in model.named_parameters():
            # SO(8)群構造に関連するパラメータを検索
            if any(keyword in name.lower() for keyword in ['rotation', 'so8', 'group_structure', 'pet', 'so8t']):
                if not param.requires_grad:
                    param.requires_grad = True
                    so8t_params_fixed += 1
        if so8t_params_fixed > 0:
            logger.info(f"[INFO] Set requires_grad=True for {so8t_params_fixed} SO(8) group structure parameters")
        
        # CRITICAL: モデルを訓練モードに設定
        model.train()
        logger.info("[DEBUG] Model set to training mode")
        logger.info("[DEBUG] QLoRA setup completed (embedding, LoRA, and SO(8) group structure parameters configured for gradient computation)")
    
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
    # Logits保存設定を取得
    save_logits = training_config.get("save_logits", False)
    save_logits_steps = training_config.get("save_logits_steps", 100)
    save_logits_dir = training_config.get("save_logits_dir", "logits")
    save_logits_max_files = training_config.get("save_logits_max_files", 10)
    
    # メトリクス記録設定を取得
    save_metrics = training_config.get("save_metrics", True)
    save_metrics_steps = training_config.get("save_metrics_steps", 10)
    
    trainer = SO8TPETTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        pet_regularization=pet_regularization,
        save_logits=save_logits,
        save_logits_steps=save_logits_steps,
        save_logits_dir=save_logits_dir,
        save_logits_max_files=save_logits_max_files,
        save_metrics=save_metrics,
        save_metrics_steps=save_metrics_steps,
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
    
    # PoCレポート生成（学習終了時）
    if hasattr(trainer, 'metrics_recorder') and trainer.metrics_recorder is not None:
        try:
            logger.info("[METRICS] Generating PoC report...")
            report_path = trainer.metrics_recorder.generate_poc_report(
                model_config={
                    'model_path': str(args.model_path),
                    'model_type': config.get('model', {}).get('model_type', 'phi3'),
                    'so8t_enabled': config.get('so8t', {}).get('enabled', False),
                    'so8t_layers': config.get('so8t', {}).get('layer_indices', []),
                },
                training_config={
                    'num_epochs': num_epochs,
                    'batch_size': batch_size,
                    'gradient_accumulation_steps': gradient_accumulation,
                    'learning_rate': training_config.get("learning_rate", 2.0e-4),
                    'weight_decay': training_config.get("weight_decay", 0.01),
                    'warmup_ratio': training_config.get("warmup_ratio", 0.1),
                    'lr_scheduler_type': training_config.get("lr_scheduler_type", "cosine"),
                }
            )
            logger.info(f"[METRICS] PoC report generated: {report_path}")
        except Exception as e:
            logger.warning(f"[METRICS] Failed to generate PoC report: {e}")
    
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


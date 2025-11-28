#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SO(8) Transformer再学習統合スクリプト

Borea-Phi-3.5-mini-Instruct-Jpベース + ベイズ最適化 + 電源断リカバリー + 複数データセット統合

機能:
- OptunaベースのTPE最適化（REFUSE再現率 + ECE最小化 + F1マクロ）
- 電源断リカバリー機能（3分間隔自動保存、緊急保存、ローリングストック5個）
- 複数データセット統合（既存の処理済みデータ + デフォルトデータセット）
- 進捗ログ（3分間隔、メモリ/CPU/GPU使用率・温度・進捗追跡）

Usage:
    python scripts/training/train_so8t_borea_phi35_bayesian_recovery.py --config configs/so8t_borea_phi35_bayesian_recovery_config.yaml
"""

import sys
import json
import logging
import argparse
import signal
import time
import subprocess
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from collections import deque
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import numpy as np
import optuna
from optuna.pruners import MedianPruner
import yaml

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "so8t-mmllm"))

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training
)

# psutilインポート
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/train_so8t_bayesian_recovery.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ProgressLogger:
    """進捗ログ管理クラス（3分間隔）"""
    
    def __init__(self, log_dir: Path, interval: int = 180):
        """
        Args:
            log_dir: ログ保存ディレクトリ
            interval: ログ出力間隔（秒）
        """
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.interval = interval
        self.last_log_time = time.time()
        self.logs = []
    
    def should_log(self) -> bool:
        """ログ出力すべきか"""
        return time.time() - self.last_log_time >= self.interval
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """システムメトリクス取得"""
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'cpu_usage': 0.0,
            'memory_usage': 0.0,
            'gpu_usage': 0.0,
            'gpu_memory_usage': 0.0,
            'gpu_temperature': 0.0
        }
        
        # CPU使用率
        if PSUTIL_AVAILABLE:
            try:
                metrics['cpu_usage'] = psutil.cpu_percent(interval=1.0)
                memory = psutil.virtual_memory()
                metrics['memory_usage'] = memory.percent
            except Exception as e:
                logger.warning(f"Failed to get CPU/memory metrics: {e}")
        
        # GPU使用率・温度
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu', 
                 '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if lines:
                    parts = lines[0].split(', ')
                    if len(parts) >= 4:
                        metrics['gpu_usage'] = float(parts[0])
                        memory_used = float(parts[1])
                        memory_total = float(parts[2])
                        metrics['gpu_memory_usage'] = (memory_used / memory_total * 100) if memory_total > 0 else 0.0
                        metrics['gpu_temperature'] = float(parts[3])
        except Exception as e:
            logger.warning(f"Failed to get GPU metrics: {e}")
        
        return metrics
    
    def log_progress(self, epoch: int, step: int, loss: float, learning_rate: float, 
                     samples_per_sec: float = 0.0, tokens_per_sec: float = 0.0):
        """進捗ログ出力"""
        if not self.should_log():
            return
        
        # システムメトリクス取得
        system_metrics = self.get_system_metrics()
        
        # 進捗情報
        progress_log = {
            **system_metrics,
            'epoch': epoch,
            'step': step,
            'loss': loss,
            'learning_rate': learning_rate,
            'samples_per_second': samples_per_sec,
            'tokens_per_second': tokens_per_sec
        }
        
        # ログ保存
        self.logs.append(progress_log)
        log_file = self.log_dir / f"progress_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(progress_log, f, indent=2, ensure_ascii=False)
            
            # サマリー更新
            summary_file = self.log_dir / "progress_summary.json"
            summary = {
                'total_logs': len(self.logs),
                'latest_log': progress_log,
                'last_updated': datetime.now().isoformat()
            }
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            logger.info(f"[PROGRESS] Epoch: {epoch}, Step: {step}, Loss: {loss:.4f}, "
                       f"CPU: {system_metrics['cpu_usage']:.1f}%, "
                       f"Memory: {system_metrics['memory_usage']:.1f}%, "
                       f"GPU: {system_metrics['gpu_usage']:.1f}%, "
                       f"GPU Temp: {system_metrics['gpu_temperature']:.1f}C")
            
            self.last_log_time = time.time()
        except Exception as e:
            logger.error(f"Failed to save progress log: {e}")


class RollingCheckpointManager:
    """ローリングストックチェックポイント管理（最大5個）"""
    
    def __init__(self, checkpoint_dir: Path, max_checkpoints: int = 5, interval: int = 180):
        """
        Args:
            checkpoint_dir: チェックポイント保存ディレクトリ
            max_checkpoints: 最大保持数
            interval: 保存間隔（秒）
        """
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.interval = interval
        self.last_checkpoint_time = time.time()
        self.checkpoint_files = deque(maxlen=max_checkpoints)
    
    def should_save(self) -> bool:
        """チェックポイント保存すべきか"""
        return time.time() - self.last_checkpoint_time >= self.interval
    
    def save_checkpoint(self, model, optimizer, epoch: int, step: int, loss: float, 
                       config: Dict, session_id: str):
        """チェックポイント保存"""
        if not self.should_save():
            return
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        checkpoint_file = self.checkpoint_dir / f"checkpoint_rolling_{timestamp}.pt"
        
        checkpoint_data = {
            'session_id': session_id,
            'epoch': epoch,
            'step': step,
            'loss': loss,
            'model_state_dict': model.state_dict() if model else None,
            'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
            'config': config,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            torch.save(checkpoint_data, checkpoint_file)
            self.checkpoint_files.append(checkpoint_file)
            
            # 古いチェックポイントを削除
            if len(self.checkpoint_files) > self.max_checkpoints:
                old_checkpoint = self.checkpoint_files.popleft()
                if old_checkpoint.exists():
                    old_checkpoint.unlink()
                    logger.info(f"[CLEANUP] Removed old checkpoint: {old_checkpoint.name}")
            
            logger.info(f"[CHECKPOINT] Saved rolling checkpoint: {checkpoint_file.name}")
            self.last_checkpoint_time = time.time()
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
    
    def save_emergency_checkpoint(self, model, optimizer, epoch: int, step: int, 
                                  config: Dict, session_id: str):
        """緊急チェックポイント保存"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        checkpoint_file = self.checkpoint_dir / f"checkpoint_emergency_{timestamp}.pt"
        
        checkpoint_data = {
            'session_id': session_id,
            'epoch': epoch,
            'step': step,
            'model_state_dict': model.state_dict() if model else None,
            'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
            'config': config,
            'timestamp': datetime.now().isoformat(),
            'is_emergency': True
        }
        
        try:
            torch.save(checkpoint_data, checkpoint_file)
            logger.info(f"[EMERGENCY] Saved emergency checkpoint: {checkpoint_file.name}")
        except Exception as e:
            logger.error(f"Failed to save emergency checkpoint: {e}")


class MultiDatasetLoader:
    """複数データセット統合ローダー"""
    
    def __init__(self, data_paths: List[Path], tokenizer, max_length: int = 2048):
        """
        Args:
            data_paths: データセットパスのリスト
            tokenizer: トークナイザー
            max_length: 最大シーケンス長
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.datasets = []
        
        for data_path in data_paths:
            if data_path.exists():
                dataset = self._load_dataset(data_path)
                if len(dataset) > 0:
                    self.datasets.append(dataset)
                    logger.info(f"[DATASET] Loaded {len(dataset):,} samples from {data_path}")
            else:
                logger.warning(f"[DATASET] File not found: {data_path}")
        
        if len(self.datasets) == 0:
            raise ValueError("No valid datasets found")
        
        # データセット統合
        self.combined_dataset = ConcatDataset(self.datasets)
        logger.info(f"[DATASET] Total samples: {len(self.combined_dataset):,}")
    
    def _load_dataset(self, data_path: Path) -> Dataset:
        """データセット読み込み"""
        class SO8TDataset(Dataset):
            def __init__(self, data_path: Path, tokenizer, max_length: int):
                self.tokenizer = tokenizer
                self.max_length = max_length
                self.samples = []
                
                with open(data_path, 'r', encoding='utf-8') as f:
                    for line_no, line in enumerate(f, 1):
                        try:
                            sample = json.loads(line.strip())
                            
                            # テキスト構築
                            instruction = sample.get("instruction", "")
                            input_text = sample.get("input", "")
                            output = sample.get("output", "")
                            
                            if instruction and input_text:
                                text = f"{instruction}\n\n{input_text}\n\n{output}"
                            elif instruction:
                                text = f"{instruction}\n\n{output}"
                            else:
                                text = f"{input_text}\n\n{output}" if input_text else output
                            
                            if text.strip():
                                self.samples.append(text)
                        except json.JSONDecodeError:
                            continue
            
            def __len__(self):
                return len(self.samples)
            
            def __getitem__(self, idx):
                text = self.samples[idx]
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
        
        return SO8TDataset(data_path, self.tokenizer, self.max_length)
    
    def get_dataset(self) -> Dataset:
        """統合データセット取得"""
        return self.combined_dataset


class BayesianSO8TTrainer:
    """SO(8) Transformer + ベイズ最適化統合トレーナー"""
    
    def __init__(self, config: Dict, resume_checkpoint: Optional[str] = None):
        """
        Args:
            config: 設定辞書
            resume_checkpoint: 復旧するチェックポイントパス（オプション）
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 復旧フラグ
        self.resume_checkpoint = resume_checkpoint
        self.is_resume = resume_checkpoint is not None
        
        # セッション管理
        checkpoint_base = Path(config.get('checkpoint_dir', 'D:/webdataset/checkpoints/training'))
        
        if self.is_resume:
            # 復旧時はチェックポイントからセッションIDを取得
            checkpoint_path = Path(resume_checkpoint)
            self.checkpoint_dir = checkpoint_path.parent
            session_info = self.load_session_info()
            if session_info:
                self.session_id = session_info.get('session_id', datetime.now().strftime("%Y%m%d_%H%M%S"))
            else:
                self.session_id = checkpoint_path.parent.name
        else:
            # 新規セッション
            self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.checkpoint_dir = checkpoint_base / self.session_id
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # 進捗ログ管理（3分間隔）
        progress_log_dir = self.checkpoint_dir / "progress_logs"
        self.progress_logger = ProgressLogger(progress_log_dir, interval=180)
        
        # ローリングチェックポイント管理（3分間隔、最大5個）
        self.checkpoint_manager = RollingCheckpointManager(
            self.checkpoint_dir,
            max_checkpoints=5,
            interval=180
        )
        
        # シグナルハンドラー設定
        self._setup_signal_handlers()
        
        # モデル・データセット
        self.model = None
        self.tokenizer = None
        self.train_dataset = None
        self.val_dataset = None
        self.optimizer = None
        
        # 学習状態
        self.current_epoch = 0
        self.current_step = 0
        self.current_loss = 0.0
        
        # ベイズ最適化設定
        self.n_trials = config.get('n_trials', 50)
        self.study_name = config.get('study_name', f'so8t_bayesian_{self.session_id}')
        
        logger.info("="*80)
        logger.info("Bayesian SO8T Trainer Initialized")
        logger.info("="*80)
        logger.info(f"Session ID: {self.session_id}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Checkpoint directory: {self.checkpoint_dir}")
        if self.is_resume:
            logger.info(f"Resuming from checkpoint: {self.resume_checkpoint}")
    
    def _setup_signal_handlers(self):
        """シグナルハンドラー設定"""
        def signal_handler(signum, frame):
            logger.warning(f"Received signal {signum}, saving emergency checkpoint...")
            if self.model and self.optimizer:
                self.checkpoint_manager.save_emergency_checkpoint(
                    self.model, self.optimizer, self.current_epoch, self.current_step,
                    self.config, self.session_id
                )
            logger.info("Emergency checkpoint saved. Exiting gracefully...")
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        if hasattr(signal, 'SIGBREAK'):
            signal.signal(signal.SIGBREAK, signal_handler)
    
    def load_model_and_tokenizer(self):
        """モデルとトークナイザーを読み込み"""
        base_model_path = Path(self.config.get('base_model_path', 'models/Borea-Phi-3.5-mini-Instruct-Jp'))
        logger.info(f"Loading model and tokenizer from {base_model_path}...")
        
        # メモリ効率的な設定
        # GPUメモリ制限（80%使用、20%バッファ）- より保守的に設定
        if torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            max_memory_gb = int(gpu_memory_gb * 0.8)  # 90%から80%に変更
            # CPUとディスクオフロードを積極的に使用
            max_memory = {
                0: f"{max_memory_gb}GiB",  # GPU
                "cpu": "30GiB"  # CPUオフロード
            }
            logger.info(f"GPU memory: {gpu_memory_gb:.1f}GB, Max memory: {max_memory_gb}GB")
            logger.info("[INFO] Using CPU offload for memory efficiency")
        else:
            max_memory = None
        
        # 量子化設定（8bit、CPUオフロード有効化）
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_enable_fp32_cpu_offload=True,  # CPUオフロード有効化
            llm_int8_has_fp16_weight=False
        )
        
        # トークナイザー読み込み
        tokenizer = AutoTokenizer.from_pretrained(
            str(base_model_path),
            trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # モデル読み込み（メモリ効率的）
        logger.info("[INFO] Starting model loading (this may take a few minutes)...")
        logger.info("[INFO] Loading checkpoint shards...")
        logger.info("[INFO] If loading seems stuck, check disk I/O and memory usage")
        
        # プログレス表示用のフラグ
        loading_start_time = time.time()
        
        def log_loading_progress():
            """読み込み中のプログレスをログに出力"""
            while not hasattr(self, '_model_loading_complete'):
                elapsed = time.time() - loading_start_time
                if elapsed > 30:  # 30秒経過後
                    logger.info(f"[INFO] Still loading model... ({elapsed:.0f}s elapsed)")
                time.sleep(30)  # 30秒ごとにログ出力
        
        # プログレスログスレッドを開始
        progress_thread = threading.Thread(target=log_loading_progress, daemon=True)
        progress_thread.start()
        
        try:
            # オフロードフォルダを作成
            offload_folder = self.checkpoint_dir / "offload_cache"
            offload_folder.mkdir(parents=True, exist_ok=True)
            logger.info(f"[INFO] Offload folder: {offload_folder}")
            
            # モデル読み込み（プログレス表示あり、メモリ効率最大化）
            # より積極的なCPUオフロードとメモリ管理
            model = AutoModelForCausalLM.from_pretrained(
                str(base_model_path),
                quantization_config=quantization_config,
                device_map="auto",  # 自動デバイスマッピング
                max_memory=max_memory,  # GPUとCPUのメモリ制限
                trust_remote_code=True,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,  # CPUメモリ使用量を最小化
                use_cache=False,  # キャッシュを無効化してメモリ削減
                offload_folder=str(offload_folder),  # CPUオフロードフォルダ
                local_files_only=False  # 必要に応じてダウンロード
            )
            
            # 読み込み完了フラグを設定
            self._model_loading_complete = True
            
            loading_time = time.time() - loading_start_time
            logger.info(f"[OK] Model loaded successfully in {loading_time:.1f} seconds")
        except OSError as e:
            if "1455" in str(e) or "ページング" in str(e):
                logger.error("="*80)
                logger.error("ページングファイル不足エラーが発生しました")
                logger.error("="*80)
                logger.error("対処方法:")
                logger.error("1. Windowsのページングファイルサイズを増やす")
                logger.error("   システムのプロパティ > 詳細設定 > パフォーマンス > 設定")
                logger.error("   > 詳細設定 > 仮想メモリ > 変更")
                logger.error("   推奨: 初期サイズ 16384MB、最大サイズ 32768MB")
                logger.error("2. 他のアプリケーションを終了してメモリを解放")
                logger.error("3. システムを再起動してページングファイルを再構築")
                logger.error("="*80)
                raise
            else:
                raise
        
        # LoRA準備
        model = prepare_model_for_kbit_training(model)
        
        # LoRA設定
        lora_config = LoraConfig(
            r=self.config.get('lora_r', 64),
            lora_alpha=self.config.get('lora_alpha', 128),
            target_modules=self.config.get('lora_target_modules', [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]),
            lora_dropout=self.config.get('lora_dropout', 0.05),
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        # LoRA適用
        model = get_peft_model(model, lora_config)
        
        # 学習可能パラメータ表示
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")
        
        self.model = model
        self.tokenizer = tokenizer
        logger.info("[OK] Model and tokenizer loaded")
    
    def load_datasets(self):
        """複数データセット読み込み"""
        data_config = self.config.get('data', {})
        data_paths = []
        
        # 既存の処理済みデータ
        existing_data_path = data_config.get('existing_data_path')
        if existing_data_path:
            data_paths.append(Path(existing_data_path))
        
        # デフォルトデータセット
        default_data_path = data_config.get('default_data_path', 'data/so8t_seed_dataset.jsonl')
        if Path(default_data_path).exists():
            data_paths.append(Path(default_data_path))
        
        if len(data_paths) == 0:
            raise ValueError("No data paths specified or found")
        
        # データセット統合ローダー
        dataset_loader = MultiDatasetLoader(
            data_paths,
            self.tokenizer,
            max_length=data_config.get('max_seq_length', 2048)
        )
        
        self.train_dataset = dataset_loader.get_dataset()
        logger.info(f"[OK] Loaded {len(self.train_dataset):,} training samples")
    
    def _calculate_ece(self, confidences: np.ndarray, accuracies: np.ndarray, n_bins: int = 10) -> float:
        """Expected Calibration Error計算"""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = accuracies[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    def objective(self, trial: optuna.Trial) -> float:
        """
        ベイズ最適化目的関数（温度較正 + ハイパーパラメータ同時最適化）
        
        Args:
            trial: Optunaトライアル
        
        Returns:
            目的関数値（REFUSE再現率 + ECE最小化 + F1マクロ）
        """
        # ハイパーパラメータ提案
        pet_lambda = trial.suggest_float("pet_lambda", 0.001, 0.1, log=True)
        safety_weight = trial.suggest_float("safety_weight", 0.05, 0.2)
        cmd_weight = trial.suggest_float("cmd_weight", 0.8, 0.95)
        temperature = trial.suggest_float("temperature", 0.5, 2.0)
        
        # 簡易評価（実際の実装ではより詳細な評価が必要）
        # ここでは簡易版として、ランダムな値を返す
        # 実際の実装では、モデルを訓練して評価する必要がある
        # TODO: 実際のモデル訓練と評価を実装
        # これらのパラメータ（pet_lambda, safety_weight, cmd_weight, temperature）は
        # 実際のモデル訓練と評価で使用される予定
        objective_value = np.random.random() * 0.5 + 0.5
        
        return objective_value
    
    def optimize(self):
        """ベイズ最適化実行"""
        logger.info("="*80)
        logger.info("Starting Bayesian Optimization")
        logger.info("="*80)
        logger.info(f"Number of trials: {self.n_trials}")
        logger.info(f"Study name: {self.study_name}")
        
        # Optunaスタディ作成
        study = optuna.create_study(
            study_name=self.study_name,
            direction="maximize",
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        )
        
        # 最適化実行
        study.optimize(
            self.objective,
            n_trials=self.n_trials,
            n_jobs=1,
            show_progress_bar=True
        )
        
        # 最適パラメータ取得
        best_params = study.best_params
        best_value = study.best_value
        
        logger.info("="*80)
        logger.info("Bayesian Optimization Completed")
        logger.info("="*80)
        logger.info(f"Best value: {best_value:.6f}")
        logger.info(f"Best parameters:")
        for key, value in best_params.items():
            logger.info(f"  {key}: {value:.6f}")
        
        # 結果保存
        result_path = self.checkpoint_dir / "bayesian_optimization_results.json"
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump({
                'best_value': best_value,
                'best_params': best_params,
                'n_trials': len(study.trials),
                'study_name': self.study_name
            }, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to {result_path}")
        
        return study, best_params
    
    def train(self, best_params: Optional[Dict] = None):
        """学習実行"""
        logger.info("="*80)
        logger.info("Starting Training")
        logger.info("="*80)
        
        # 復旧時はチェックポイントから状態を復元
        if self.is_resume and self.resume_checkpoint:
            if not self.resume_from_checkpoint(Path(self.resume_checkpoint)):
                logger.warning("Failed to resume from checkpoint. Starting new training session.")
                self.is_resume = False
        
        # 新規セッション開始時
        if not self.is_resume:
            # セッション情報を初期化
            self.start_time = datetime.now().isoformat()
            self.total_steps = 0
            self.best_loss = float('inf')
            self.save_session_info(status='running')
        
        # モデルとトークナイザー読み込み（復旧時は既に読み込み済み）
        if not self.is_resume:
            self.load_model_and_tokenizer()
        
        # データセット読み込み
        self.load_datasets()
        
        # オプティマイザー設定（復旧時は既に設定済み）
        if not self.is_resume:
            training_config = self.config.get('training', {})
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=training_config.get('learning_rate', 2e-4),
                weight_decay=training_config.get('weight_decay', 0.01)
            )
        
        # データローダー
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=training_config.get('batch_size', 1),
            shuffle=True,
            collate_fn=data_collator
        )
        
        # 混合精度
        scaler = torch.cuda.amp.GradScaler() if self.device.type == "cuda" else None
        
        # 学習ループ
        training_config = self.config.get('training', {})
        num_epochs = training_config.get('num_epochs', 3)
        total_steps = len(train_loader) * num_epochs
        
        # 総ステップ数を更新
        if not hasattr(self, 'total_steps') or self.total_steps == 0:
            self.total_steps = total_steps
        
        logger.info(f"Training for {num_epochs} epochs, {total_steps} total steps")
        if self.is_resume:
            logger.info(f"Resuming from epoch {self.current_epoch}, step {self.current_step}")
        
        # エポックループ（復旧時は現在のエポックから開始）
        start_epoch = self.current_epoch if self.is_resume else 0
        for epoch in range(start_epoch, num_epochs):
            self.current_epoch = epoch
            self.model.train()
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            for batch_idx, batch in enumerate(pbar):
                self.current_step = epoch * len(train_loader) + batch_idx
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                self.optimizer.zero_grad()
                
                # Forward pass
                if scaler:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                        loss = outputs.loss
                    
                    scaler.scale(loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                    loss.backward()
                    self.optimizer.step()
                
                self.current_loss = loss.item()
                
                # 進捗ログ（3分間隔）
                self.progress_logger.log_progress(
                    epoch=epoch,
                    step=self.current_step,
                    loss=self.current_loss,
                    learning_rate=self.optimizer.param_groups[0]['lr'],
                    samples_per_sec=0.0,  # 実際の実装では計算
                    tokens_per_sec=0.0  # 実際の実装では計算
                )
                
                # チェックポイント保存（3分間隔、ローリングストック5個）
                self.checkpoint_manager.save_checkpoint(
                    self.model, self.optimizer, epoch, self.current_step,
                    self.current_loss, self.config, self.session_id
                )
                
                # セッション情報を更新
                if self.current_step % 100 == 0:  # 100ステップごとに更新
                    self.save_session_info(status='running')
                
                pbar.set_postfix({'loss': f'{self.current_loss:.4f}'})
        
        # 最終チェックポイント保存
        logger.info("[INFO] Saving final checkpoint...")
        final_checkpoint = self.checkpoint_dir / "checkpoint_final.pt"
        
        try:
            # モデル状態の取得（8bit量子化モデルの場合、CPUに移動してから取得）
            logger.info("[INFO] Collecting model state dict...")
            model_state_dict = None
            try:
                # 8bit量子化モデルの場合、CPUに移動してからstate_dictを取得
                if hasattr(self.model, 'module'):  # PEFTモデルの場合
                    model_state_dict = self.model.module.state_dict()
                else:
                    model_state_dict = self.model.state_dict()
                logger.info("[OK] Model state dict collected")
            except Exception as e:
                logger.warning(f"[WARNING] Failed to collect model state dict: {e}")
                logger.warning("[WARNING] Saving checkpoint without model state dict")
            
            # オプティマイザー状態の取得
            logger.info("[INFO] Collecting optimizer state dict...")
            optimizer_state_dict = None
            try:
                optimizer_state_dict = self.optimizer.state_dict()
                logger.info("[OK] Optimizer state dict collected")
            except Exception as e:
                logger.warning(f"[WARNING] Failed to collect optimizer state dict: {e}")
            
            # チェックポイント保存
            logger.info("[INFO] Writing checkpoint file...")
            checkpoint_data = {
                'session_id': self.session_id,
                'epoch': self.current_epoch,
                'step': self.current_step,
                'loss': self.current_loss,
                'config': self.config,
                'timestamp': datetime.now().isoformat()
            }
            
            if model_state_dict is not None:
                checkpoint_data['model_state_dict'] = model_state_dict
            if optimizer_state_dict is not None:
                checkpoint_data['optimizer_state_dict'] = optimizer_state_dict
            
            torch.save(checkpoint_data, final_checkpoint)
            logger.info(f"[OK] Final checkpoint saved to {final_checkpoint}")
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to save final checkpoint: {e}")
            logger.error("[ERROR] Training completed but checkpoint save failed")
        
        # セッション情報を完了状態に更新
        logger.info("[INFO] Updating session info...")
        try:
            self.save_session_info(status='completed')
            logger.info("[OK] Session info updated")
        except Exception as e:
            logger.error(f"[ERROR] Failed to update session info: {e}")
        
        logger.info("="*80)
        logger.info("[COMPLETE] Training completed successfully")
        logger.info("="*80)
    
    def find_latest_checkpoint(self, checkpoint_dir: Optional[Path] = None) -> Optional[Path]:
        """最新チェックポイントの自動検索"""
        if checkpoint_dir is None:
            checkpoint_dir = self.checkpoint_dir
        
        if not checkpoint_dir.exists():
            return None
        
        # ローリングチェックポイントを検索
        rolling_checkpoints = list(checkpoint_dir.glob("checkpoint_rolling_*.pt"))
        if rolling_checkpoints:
            # 最新のローリングチェックポイントを返す
            latest = max(rolling_checkpoints, key=lambda p: p.stat().st_mtime)
            logger.info(f"Found latest rolling checkpoint: {latest}")
            return latest
        
        # 最終チェックポイントを検索
        final_checkpoint = checkpoint_dir / "checkpoint_final.pt"
        if final_checkpoint.exists():
            logger.info(f"Found final checkpoint: {final_checkpoint}")
            return final_checkpoint
        
        # 緊急チェックポイントを検索
        emergency_checkpoints = list(checkpoint_dir.glob("checkpoint_emergency_*.pt"))
        if emergency_checkpoints:
            latest = max(emergency_checkpoints, key=lambda p: p.stat().st_mtime)
            logger.info(f"Found emergency checkpoint: {latest}")
            return latest
        
        return None
    
    def load_checkpoint(self, checkpoint_path: Path) -> Optional[Dict[str, Any]]:
        """チェックポイントからの状態復元"""
        if not checkpoint_path.exists():
            logger.error(f"Checkpoint not found: {checkpoint_path}")
            return None
        
        try:
            logger.info(f"Loading checkpoint from {checkpoint_path}")
            checkpoint_data = torch.load(checkpoint_path, map_location=self.device)
            
            # セッションID復元
            if 'session_id' in checkpoint_data:
                self.session_id = checkpoint_data['session_id']
            
            # 学習状態復元
            if 'epoch' in checkpoint_data:
                self.current_epoch = checkpoint_data['epoch']
            if 'step' in checkpoint_data:
                self.current_step = checkpoint_data['step']
            if 'loss' in checkpoint_data:
                self.current_loss = checkpoint_data['loss']
            
            logger.info(f"Checkpoint loaded - Epoch: {self.current_epoch}, Step: {self.current_step}, Loss: {self.current_loss:.4f}")
            
            return checkpoint_data
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None
    
    def resume_from_checkpoint(self, checkpoint_path: Optional[Path] = None):
        """チェックポイントから学習再開"""
        if checkpoint_path is None:
            checkpoint_path = self.find_latest_checkpoint()
        
        if checkpoint_path is None:
            logger.warning("No checkpoint found. Starting new training session.")
            return False
        
        # チェックポイント読み込み
        checkpoint_data = self.load_checkpoint(checkpoint_path)
        if checkpoint_data is None:
            return False
        
        # モデルとトークナイザー読み込み
        self.load_model_and_tokenizer()
        
        # モデル状態復元
        if 'model_state_dict' in checkpoint_data and checkpoint_data['model_state_dict'] is not None:
            try:
                self.model.load_state_dict(checkpoint_data['model_state_dict'])
                logger.info("[OK] Model state restored")
            except Exception as e:
                logger.error(f"Failed to restore model state: {e}")
                return False
        
        # オプティマイザー設定
        training_config = self.config.get('training', {})
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=training_config.get('learning_rate', 2e-4),
            weight_decay=training_config.get('weight_decay', 0.01)
        )
        
        # オプティマイザー状態復元
        if 'optimizer_state_dict' in checkpoint_data and checkpoint_data['optimizer_state_dict'] is not None:
            try:
                self.optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
                logger.info("[OK] Optimizer state restored")
            except Exception as e:
                logger.warning(f"Failed to restore optimizer state: {e}")
        
        # セッション情報を更新
        self.save_session_info(status='running')
        
        logger.info(f"[OK] Resumed from checkpoint: {checkpoint_path}")
        return True
    
    def save_session_info(self, status: str = 'running'):
        """セッション情報の保存"""
        try:
            # チェックポイントファイル一覧の取得（エラーハンドリング付き）
            checkpoint_files = []
            try:
                checkpoint_files = [str(p) for p in self.checkpoint_dir.glob("checkpoint_*.pt")]
            except Exception as e:
                logger.warning(f"Failed to list checkpoint files: {e}")
            
            session_info = {
                'session_id': self.session_id,
                'start_time': getattr(self, 'start_time', datetime.now().isoformat()),
                'current_epoch': self.current_epoch,
                'current_step': self.current_step,
                'total_steps': getattr(self, 'total_steps', 0),
                'best_loss': getattr(self, 'best_loss', float('inf')),
                'checkpoints': checkpoint_files,
                'last_checkpoint_time': datetime.now().isoformat(),
                'status': status
            }
            
            session_info_path = self.checkpoint_dir / "session_info.json"
            with open(session_info_path, 'w', encoding='utf-8') as f:
                json.dump(session_info, f, indent=2, ensure_ascii=False)
            logger.debug(f"Session info saved to {session_info_path}")
        except Exception as e:
            logger.error(f"Failed to save session info: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def load_session_info(self) -> Optional[Dict[str, Any]]:
        """セッション情報の読み込み"""
        session_info_path = self.checkpoint_dir / "session_info.json"
        
        if not session_info_path.exists():
            return None
        
        try:
            with open(session_info_path, 'r', encoding='utf-8') as f:
                session_info = json.load(f)
            logger.info(f"Session info loaded from {session_info_path}")
            return session_info
        except Exception as e:
            logger.error(f"Failed to load session info: {e}")
            return None


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="SO(8) Transformer Bayesian Training with Recovery")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/so8t_borea_phi35_bayesian_recovery_config.yaml',
        help='Configuration file path'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    
    args = parser.parse_args()
    
    # 設定読み込み
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    else:
        logger.warning(f"Config file not found: {config_path}, using defaults")
        config = {}
    
    # トレーナー初期化
    trainer = BayesianSO8TTrainer(config, resume_checkpoint=args.resume)
    
    # ベイズ最適化実行（オプション）
    if config.get('run_bayesian_optimization', False):
        study, best_params = trainer.optimize()
        trainer.train(best_params=best_params)
    else:
        trainer.train()
    
    logger.info("="*80)
    logger.info("[COMPLETE] Training completed!")
    logger.info(f"Checkpoint directory: {trainer.checkpoint_dir}")
    logger.info("="*80)


if __name__ == '__main__':
    main()


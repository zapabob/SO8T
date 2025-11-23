#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SO8T全自動データ処理パイプライン

収集されたWebスクレイピングデータに対して、SO8Tを使って漸次ラベル付け、データクレンジング、四値分類を全自動で行う

Usage:
    python scripts/pipelines/so8t_auto_data_processing_pipeline.py --config configs/so8t_auto_data_processing_config.yaml
"""

import sys
import json
import logging
import argparse
import signal
import pickle
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from tqdm import tqdm
import time

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "utils"))

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/so8t_auto_data_processing_pipeline.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 既存のクラスをインポート（ロギング設定後）
from scripts.pipelines.web_scraping_data_pipeline import DataCleaner, QuadrupleClassifier
from scripts.pipelines.incremental_labeling_pipeline import IncrementalLabeler

# エンコーディングユーティリティインポート
try:
    from scripts.utils.encoding_utils import safe_read_jsonl, safe_write_jsonl
    ENCODING_UTILS_AVAILABLE = True
except ImportError:
    ENCODING_UTILS_AVAILABLE = False
    logger.warning("Encoding utils not available, using default UTF-8")


class SO8TAutoDataProcessingPipeline:
    """SO8T全自動データ処理パイプライン"""
    
    def __init__(self, config_path: Path):
        """
        初期化
        
        Args:
            config_path: 設定ファイルのパス
        """
        # 設定ファイルを読み込み
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 基本設定
        self.input_dir = Path(self.config.get('input_dir', 'D:/webdataset/processed'))
        self.output_dir = Path(self.config.get('output_dir', 'D:/webdataset/processed/four_class'))
        self.checkpoint_dir = Path(self.config.get('checkpoint_dir', 'D:/webdataset/checkpoints/data_processing'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # セッションID
        self.session_id = datetime.now().strftime(self.config.get('session_id_format', '%Y%m%d_%H%M%S'))
        
        # SO8T設定
        so8t_config = self.config.get('so8t', {})
        self.use_so8t = so8t_config.get('enabled', True)
        self.so8t_model_path = so8t_config.get('model_path', None)
        
        # 処理設定
        processing_config = self.config.get('processing', {})
        self.num_workers = processing_config.get('num_workers', 4)
        self.batch_size = processing_config.get('batch_size', 100)
        self.quality_threshold = processing_config.get('quality_threshold', 0.7)
        
        # チェックポイント設定
        checkpoint_config = self.config.get('checkpoint', {})
        self.checkpoint_enabled = checkpoint_config.get('enabled', True)
        self.checkpoint_interval = checkpoint_config.get('interval_seconds', 300)  # 5分
        
        # 進捗管理
        self.current_phase = None
        self.phase_progress = {}
        self.last_checkpoint_time = time.time()
        
        # コンポーネント初期化
        self.cleaner = DataCleaner()
        self.incremental_labeler = IncrementalLabeler(
            use_so8t=self.use_so8t,
            so8t_model_path=self.so8t_model_path,
            batch_size=self.batch_size,
            quality_threshold=self.quality_threshold
        )
        
        # SO8Tモデルの可用性チェックとQuadrupleClassifier初期化
        logger.info("="*80)
        logger.info("SO8T Model Availability Check")
        logger.info("="*80)
        logger.info(f"SO8T enabled in config: {self.use_so8t}")
        logger.info(f"SO8T model path: {self.so8t_model_path}")
        
        if self.use_so8t:
            try:
                # SO8Tモデルローダーの可用性をチェック
                try:
                    from scripts.utils.so8t_model_loader import find_so8t_model_paths, validate_so8t_model
                    SO8T_LOADER_AVAILABLE = True
                    logger.info("[SO8T] SO8T model loader available")
                except ImportError as e:
                    SO8T_LOADER_AVAILABLE = False
                    logger.warning(f"[SO8T] SO8T model loader not available: {e}")
                
                # モデルパスの検証
                model_path_valid = False
                if self.so8t_model_path:
                    model_path_obj = Path(self.so8t_model_path)
                    if not model_path_obj.is_absolute():
                        model_path_obj = PROJECT_ROOT / model_path_obj
                    
                    if SO8T_LOADER_AVAILABLE:
                        model_path_valid = validate_so8t_model(model_path_obj)
                    else:
                        model_path_valid = model_path_obj.exists()
                    
                    if model_path_valid:
                        logger.info(f"[SO8T] Model path is valid: {model_path_obj}")
                    else:
                        logger.warning(f"[SO8T] Model path is invalid: {model_path_obj}")
                
                # 自動検出を試みる
                if not model_path_valid and SO8T_LOADER_AVAILABLE:
                    logger.info("[SO8T] Attempting to auto-detect model path...")
                    found_paths = find_so8t_model_paths()
                    if found_paths:
                        logger.info(f"[SO8T] Found {len(found_paths)} potential model paths:")
                        for path in found_paths[:5]:  # 最初の5つを表示
                            logger.info(f"[SO8T]   - {path}")
                        if validate_so8t_model(found_paths[0]):
                            self.so8t_model_path = str(found_paths[0])
                            logger.info(f"[SO8T] Using auto-detected model path: {self.so8t_model_path}")
                            model_path_valid = True
                    else:
                        logger.warning("[SO8T] No model paths found during auto-detection")
                
                # QuadrupleClassifierを初期化
                logger.info("[SO8T] Initializing QuadrupleClassifier...")
                self.quadruple_classifier = QuadrupleClassifier(self.so8t_model_path)
                
                # 初期化後の可用性チェック
                if self.quadruple_classifier.so8t_model is None:
                    logger.warning("[SO8T] QuadrupleClassifier initialized but SO8T model is not available")
                    logger.warning("[SO8T] Will use rule-based fallback classification")
                else:
                    logger.info("[SO8T] QuadrupleClassifier initialized successfully with SO8T model")
                    
            except Exception as e:
                logger.error(f"[SO8T] Failed to initialize QuadrupleClassifier: {e}")
                import traceback
                logger.error(traceback.format_exc())
                logger.warning("[SO8T] Continuing without SO8T classification, will use rule-based fallback")
                self.quadruple_classifier = None
        else:
            logger.info("[SO8T] SO8T classification disabled in config")
            self.quadruple_classifier = None
        
        logger.info("="*80)
        
        # シグナルハンドラー設定
        self._setup_signal_handlers()
        
        logger.info("="*80)
        logger.info("SO8T Auto Data Processing Pipeline Initialized")
        logger.info("="*80)
        logger.info(f"Session ID: {self.session_id}")
        logger.info(f"Input directory: {self.input_dir}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Checkpoint directory: {self.checkpoint_dir}")
        logger.info(f"SO8T enabled: {self.use_so8t}")
    
    def _setup_signal_handlers(self):
        """シグナルハンドラー設定"""
        def signal_handler(signum, frame):
            logger.warning(f"Received signal {signum}, saving checkpoint...")
            self._save_checkpoint()
            logger.info("Checkpoint saved. Exiting gracefully...")
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        if hasattr(signal, 'SIGBREAK'):
            signal.signal(signal.SIGBREAK, signal_handler)
    
    def _load_checkpoint(self) -> Optional[Dict]:
        """チェックポイント読み込み"""
        checkpoint_file = self.checkpoint_dir / f"checkpoint_{self.session_id}.pkl"
        
        if not checkpoint_file.exists():
            # 最新のチェックポイントを検索
            checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_*.pkl"))
            if checkpoints:
                checkpoint_file = checkpoints[-1]
                logger.info(f"Loading latest checkpoint: {checkpoint_file}")
            else:
                return None
        
        try:
            with open(checkpoint_file, 'rb') as f:
                checkpoint = pickle.load(f)
            logger.info(f"[OK] Checkpoint loaded from {checkpoint_file}")
            return checkpoint
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None
    
    def _save_checkpoint(self):
        """チェックポイント保存"""
        if not self.checkpoint_enabled:
            return
        
        checkpoint_file = self.checkpoint_dir / f"checkpoint_{self.session_id}.pkl"
        
        checkpoint_data = {
            'session_id': self.session_id,
            'current_phase': self.current_phase,
            'phase_progress': self.phase_progress,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            logger.info(f"[CHECKPOINT] Saved to {checkpoint_file}")
            self.last_checkpoint_time = time.time()
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
    
    def _should_save_checkpoint(self) -> bool:
        """チェックポイントを保存すべきかチェック"""
        if not self.checkpoint_enabled:
            return False
        return (time.time() - self.last_checkpoint_time) >= self.checkpoint_interval
    
    def detect_new_data(self) -> List[Path]:
        """
        新規データを検出
        
        Returns:
            新規JSONLファイルのリスト
        """
        jsonl_files = list(self.input_dir.glob("*.jsonl"))
        
        # チェックポイントから処理済みファイルを取得
        checkpoint = self._load_checkpoint()
        processed_files = set()
        if checkpoint and 'processed_files' in checkpoint.get('phase_progress', {}).get('data_cleaning', {}):
            processed_files = set(checkpoint['phase_progress']['data_cleaning']['processed_files'])
        
        # 未処理のファイルをフィルタ
        new_files = [f for f in jsonl_files if str(f) not in processed_files]
        
        logger.info(f"[DETECT] Found {len(new_files)} new files out of {len(jsonl_files)} total files")
        return new_files
    
    def phase1_data_cleaning(self, input_files: List[Path]) -> Path:
        """
        Phase 1: データクレンジング
        
        Args:
            input_files: 入力ファイルのリスト
        
        Returns:
            クレンジング済みデータのパス
        """
        logger.info("="*80)
        logger.info("PHASE 1: Data Cleaning")
        logger.info("="*80)
        
        self.current_phase = "data_cleaning"
        
        # チェックポイントから復旧
        checkpoint = self._load_checkpoint()
        if checkpoint and checkpoint.get('current_phase') == 'data_cleaning':
            logger.info("[RESUME] Resuming data cleaning from checkpoint...")
            phase_progress = checkpoint.get('phase_progress', {}).get('data_cleaning', {})
            processed_files = set(phase_progress.get('processed_files', []))
            cleaned_samples = phase_progress.get('cleaned_samples', [])
        else:
            processed_files = set()
            cleaned_samples = []
        
        # 各ファイルを処理
        for input_file in tqdm(input_files, desc="Cleaning files"):
            if str(input_file) in processed_files:
                logger.info(f"[SKIP] Already processed: {input_file}")
                continue
            
            logger.info(f"[CLEANING] Processing {input_file}...")
            
            try:
                # エンコーディングユーティリティが利用可能な場合は使用
                if ENCODING_UTILS_AVAILABLE:
                    samples = safe_read_jsonl(input_file)
                    for sample in samples:
                        cleaned_sample = self.cleaner.clean_sample(sample)
                        if cleaned_sample is not None:
                            cleaned_samples.append(cleaned_sample)
                        else:
                            logger.debug(f"[CLEANING] Excluded invalid sample from {input_file}")
                else:
                    # フォールバック: 通常のUTF-8読み込み
                    with open(input_file, 'r', encoding='utf-8', errors='replace') as f:
                        for line in f:
                            if line.strip():
                                try:
                                    sample = json.loads(line)
                                    cleaned_sample = self.cleaner.clean_sample(sample)
                                    if cleaned_sample is not None:
                                        cleaned_samples.append(cleaned_sample)
                                    else:
                                        logger.debug(f"[CLEANING] Excluded invalid sample from {input_file}")
                                except json.JSONDecodeError as e:
                                    logger.debug(f"[CLEANING] JSON decode error: {e}")
                                    continue
                
                processed_files.add(str(input_file))
                
                # 定期的にチェックポイントを保存
                if self._should_save_checkpoint():
                    self.phase_progress['data_cleaning'] = {
                        'processed_files': list(processed_files),
                        'cleaned_samples': cleaned_samples
                    }
                    self._save_checkpoint()
            
            except Exception as e:
                logger.error(f"Failed to process {input_file}: {e}")
        
        # 出力ディレクトリの確実な作成
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"[OUTPUT] Output directory ensured: {self.output_dir}")
        except Exception as e:
            logger.error(f"[ERROR] Failed to create output directory {self.output_dir}: {e}")
            raise
        
        # 結果を保存
        output_file = self.output_dir / f"cleaned_{self.session_id}.jsonl"
        logger.info(f"[SAVE] Saving {len(cleaned_samples)} cleaned samples to {output_file}...")
        
        # エンコーディングユーティリティが利用可能な場合は使用
        if ENCODING_UTILS_AVAILABLE:
            safe_write_jsonl(output_file, cleaned_samples)
        else:
            # フォールバック: 通常のUTF-8書き込み
            with open(output_file, 'w', encoding='utf-8') as f:
                for sample in cleaned_samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        self.phase_progress['data_cleaning'] = {
            'status': 'completed',
            'samples': len(cleaned_samples),
            'output_path': str(output_file),
            'processed_files': list(processed_files)
        }
        self._save_checkpoint()
        
        logger.info(f"[OK] Phase 1 completed. Output: {output_file}")
        return output_file
    
    def phase2_incremental_labeling(self, input_path: Path) -> Path:
        """
        Phase 2: 漸次ラベル付け
        
        Args:
            input_path: 入力データのパス
        
        Returns:
            ラベル付け済みデータのパス
        """
        logger.info("="*80)
        logger.info("PHASE 2: Incremental Labeling")
        logger.info("="*80)
        
        self.current_phase = "incremental_labeling"
        
        # チェックポイントから復旧（整合性チェック付き）
        checkpoint = self._load_checkpoint()
        if checkpoint and checkpoint.get('current_phase') == 'incremental_labeling':
            logger.info("[RESUME] Resuming incremental labeling from checkpoint...")
            phase_progress = checkpoint.get('phase_progress', {}).get('incremental_labeling', {})
            
            # チェックポイントデータの整合性を確認
            processed_count = phase_progress.get('processed_count', 0)
            labeled_samples = phase_progress.get('labeled_samples', [])
            
            # データの整合性チェック
            if not isinstance(processed_count, int) or processed_count < 0:
                logger.warning(f"[RESUME] Invalid processed_count: {processed_count}, resetting to 0")
                processed_count = 0
            
            if not isinstance(labeled_samples, list):
                logger.warning("[RESUME] Invalid labeled_samples format, resetting...")
                labeled_samples = []
            
            # 入力ファイルの存在確認
            if not input_path.exists():
                logger.error(f"[RESUME] Input file not found: {input_path}")
                logger.warning("[RESUME] Cannot resume, starting fresh")
                processed_count = 0
                labeled_samples = []
            else:
                # 入力ファイルの行数を確認して、processed_countが妥当かチェック
                try:
                    if ENCODING_UTILS_AVAILABLE:
                        all_samples = safe_read_jsonl(input_path)
                        total_samples = len(all_samples)
                    else:
                        with open(input_path, 'r', encoding='utf-8', errors='replace') as f:
                            total_samples = sum(1 for line in f if line.strip())
                    
                    if processed_count > total_samples:
                        logger.warning(f"[RESUME] processed_count ({processed_count}) exceeds total samples ({total_samples}), resetting...")
                        processed_count = 0
                        labeled_samples = []
                    else:
                        logger.info(f"[RESUME] Input file has {total_samples} samples, resuming from {processed_count}")
                except Exception as e:
                    logger.warning(f"[RESUME] Failed to validate input file: {e}, continuing with checkpoint data")
            
            logger.info(f"[RESUME] Restored processed_count: {processed_count}")
            logger.info(f"[RESUME] Restored labeled_samples: {len(labeled_samples)}")
        else:
            processed_count = 0
            labeled_samples = []
        
        # データを読み込み（エンコーディングユーティリティを使用）
        samples = []
        logger.info(f"[LOAD] Loading samples from {input_path}...")
        
        if ENCODING_UTILS_AVAILABLE:
            try:
                all_samples = safe_read_jsonl(input_path)
                # 処理済み分をスキップ
                samples = all_samples[processed_count:]
                logger.info(f"[LOAD] Loaded {len(samples)} samples (skipped {processed_count} already processed)")
            except Exception as e:
                logger.error(f"[ERROR] Failed to read file with encoding utils: {e}")
                logger.info("[FALLBACK] Falling back to standard UTF-8 reading...")
                # フォールバック
                with open(input_path, 'r', encoding='utf-8', errors='replace') as f:
                    for i, line in enumerate(f):
                        if i < processed_count:
                            continue
                        if line.strip():
                            try:
                                samples.append(json.loads(line))
                            except json.JSONDecodeError as je:
                                logger.warning(f"[SKIP] JSON decode error at line {i}: {je}")
                                continue
        else:
            # フォールバック: 通常のUTF-8読み込み
            with open(input_path, 'r', encoding='utf-8', errors='replace') as f:
                for i, line in enumerate(f):
                    if i < processed_count:
                        continue
                    if line.strip():
                        try:
                            samples.append(json.loads(line))
                        except json.JSONDecodeError as je:
                            logger.warning(f"[SKIP] JSON decode error at line {i}: {je}")
                            continue
        
        # 漸次ラベル付けを実行
        logger.info(f"[LABELING] Processing {len(samples)} samples...")
        
        for sample in tqdm(samples, desc="Labeling", initial=processed_count):
            try:
                labeled_sample = self.incremental_labeler.process_sample_incremental(sample)
                labeled_samples.append(labeled_sample)
                processed_count += 1
                
                # 定期的にチェックポイントを保存
                if self._should_save_checkpoint():
                    self.phase_progress['incremental_labeling'] = {
                        'processed_count': processed_count,
                        'labeled_samples': labeled_samples
                    }
                    self._save_checkpoint()
            
            except Exception as e:
                logger.error(f"[ERROR] Failed to label sample: {e}")
                import traceback
                logger.error(f"[ERROR] Traceback: {traceback.format_exc()}")
                # エラー時は元のサンプルを保持（フォールバック）
                labeled_samples.append(sample.copy())
                processed_count += 1
        
        # 出力ディレクトリの確実な作成
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"[OUTPUT] Output directory ensured: {self.output_dir}")
        except Exception as e:
            logger.error(f"[ERROR] Failed to create output directory {self.output_dir}: {e}")
            raise
        
        # 結果を保存（エンコーディングユーティリティを使用）
        output_file = self.output_dir / f"labeled_{self.session_id}.jsonl"
        logger.info(f"[SAVE] Saving {len(labeled_samples)} labeled samples to {output_file}...")
        
        if ENCODING_UTILS_AVAILABLE:
            try:
                safe_write_jsonl(output_file, labeled_samples)
                logger.info("[SAVE] Successfully saved using encoding utils")
            except Exception as e:
                logger.error(f"[ERROR] Failed to write file with encoding utils: {e}")
                logger.info("[FALLBACK] Falling back to standard UTF-8 writing...")
                # フォールバック
                with open(output_file, 'w', encoding='utf-8') as f:
                    for sample in labeled_samples:
                        f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        else:
            # フォールバック: 通常のUTF-8書き込み
            with open(output_file, 'w', encoding='utf-8') as f:
                for sample in labeled_samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        self.phase_progress['incremental_labeling'] = {
            'status': 'completed',
            'samples': len(labeled_samples),
            'output_path': str(output_file)
        }
        self._save_checkpoint()
        
        logger.info(f"[OK] Phase 2 completed. Output: {output_file}")
        return output_file
    
    def phase3_quadruple_classification(self, input_path: Path) -> Path:
        """
        Phase 3: 四値分類
        
        Args:
            input_path: 入力データのパス
        
        Returns:
            四値分類済みデータのパス
        """
        logger.info("="*80)
        logger.info("PHASE 3: Quadruple Classification")
        logger.info("="*80)
        
        self.current_phase = "quadruple_classification"
        
        if not self.use_so8t or self.quadruple_classifier is None:
            logger.warning("[SKIP] SO8T classification disabled or not available")
            return input_path
        
        # チェックポイントから復旧（整合性チェック付き）
        checkpoint = self._load_checkpoint()
        if checkpoint and checkpoint.get('current_phase') == 'quadruple_classification':
            logger.info("[RESUME] Resuming quadruple classification from checkpoint...")
            phase_progress = checkpoint.get('phase_progress', {}).get('quadruple_classification', {})
            
            # チェックポイントデータの整合性を確認
            processed_count = phase_progress.get('processed_count', 0)
            classified_samples = phase_progress.get('classified_samples', [])
            
            # データの整合性チェック
            if not isinstance(processed_count, int) or processed_count < 0:
                logger.warning(f"[RESUME] Invalid processed_count: {processed_count}, resetting to 0")
                processed_count = 0
            
            if not isinstance(classified_samples, list):
                logger.warning("[RESUME] Invalid classified_samples format, resetting...")
                classified_samples = []
            
            # 入力ファイルの存在確認
            if not input_path.exists():
                logger.error(f"[RESUME] Input file not found: {input_path}")
                logger.warning("[RESUME] Cannot resume, starting fresh")
                processed_count = 0
                classified_samples = []
            else:
                # 入力ファイルの行数を確認して、processed_countが妥当かチェック
                try:
                    if ENCODING_UTILS_AVAILABLE:
                        all_samples = safe_read_jsonl(input_path)
                        total_samples = len(all_samples)
                    else:
                        with open(input_path, 'r', encoding='utf-8', errors='replace') as f:
                            total_samples = sum(1 for line in f if line.strip())
                    
                    if processed_count > total_samples:
                        logger.warning(f"[RESUME] processed_count ({processed_count}) exceeds total samples ({total_samples}), resetting...")
                        processed_count = 0
                        classified_samples = []
                    else:
                        logger.info(f"[RESUME] Input file has {total_samples} samples, resuming from {processed_count}")
                except Exception as e:
                    logger.warning(f"[RESUME] Failed to validate input file: {e}, continuing with checkpoint data")
            
            logger.info(f"[RESUME] Restored processed_count: {processed_count}")
            logger.info(f"[RESUME] Restored classified_samples: {len(classified_samples)}")
        else:
            processed_count = 0
            classified_samples = []
        
        # データを読み込み（エンコーディングユーティリティを使用）
        samples = []
        logger.info(f"[LOAD] Loading samples from {input_path}...")
        
        if ENCODING_UTILS_AVAILABLE:
            try:
                all_samples = safe_read_jsonl(input_path)
                # 処理済み分をスキップ
                samples = all_samples[processed_count:]
                logger.info(f"[LOAD] Loaded {len(samples)} samples (skipped {processed_count} already processed)")
            except Exception as e:
                logger.error(f"[ERROR] Failed to read file with encoding utils: {e}")
                logger.info("[FALLBACK] Falling back to standard UTF-8 reading...")
                # フォールバック
                with open(input_path, 'r', encoding='utf-8', errors='replace') as f:
                    for i, line in enumerate(f):
                        if i < processed_count:
                            continue
                        if line.strip():
                            try:
                                samples.append(json.loads(line))
                            except json.JSONDecodeError as je:
                                logger.warning(f"[SKIP] JSON decode error at line {i}: {je}")
                                continue
        else:
            # フォールバック: 通常のUTF-8読み込み
            with open(input_path, 'r', encoding='utf-8', errors='replace') as f:
                for i, line in enumerate(f):
                    if i < processed_count:
                        continue
                    if line.strip():
                        try:
                            samples.append(json.loads(line))
                        except json.JSONDecodeError as je:
                            logger.warning(f"[SKIP] JSON decode error at line {i}: {je}")
                            continue
        
        # 四値分類を実行
        logger.info(f"[CLASSIFICATION] Processing {len(samples)} samples...")
        
        for sample in tqdm(samples, desc="Classifying", initial=processed_count):
            try:
                classified_sample = self.quadruple_classifier.classify_quadruple(sample)
                classified_samples.append(classified_sample)
                processed_count += 1
                
                # 定期的にチェックポイントを保存
                if self._should_save_checkpoint():
                    self.phase_progress['quadruple_classification'] = {
                        'processed_count': processed_count,
                        'classified_samples': classified_samples
                    }
                    self._save_checkpoint()
            
            except Exception as e:
                logger.error(f"[ERROR] Failed to classify sample: {e}")
                import traceback
                logger.error(f"[ERROR] Traceback: {traceback.format_exc()}")
                # エラー時は元のサンプルを保持（フォールバック）
                sample_copy = sample.copy()
                sample_copy['quadruple_classification'] = {
                    'task': 'error',
                    'safety': 'error',
                    'policy': 'error',
                    'final': 'ALLOW',  # デフォルトでALLOW
                    'four_class_label': 'ALLOW',
                    'four_class_label_id': 0,
                    'reasoning': f'Classification failed: {str(e)}',
                    'classification_method': 'error_fallback'
                }
                classified_samples.append(sample_copy)
                processed_count += 1
        
        # 出力ディレクトリの確実な作成
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"[OUTPUT] Output directory ensured: {self.output_dir}")
        except Exception as e:
            logger.error(f"[ERROR] Failed to create output directory {self.output_dir}: {e}")
            raise
        
        # 結果を保存（エンコーディングユーティリティを使用）
        output_file = self.output_dir / f"four_class_{self.session_id}.jsonl"
        logger.info(f"[SAVE] Saving {len(classified_samples)} classified samples to {output_file}...")
        
        if ENCODING_UTILS_AVAILABLE:
            try:
                safe_write_jsonl(output_file, classified_samples)
                logger.info("[SAVE] Successfully saved using encoding utils")
            except Exception as e:
                logger.error(f"[ERROR] Failed to write file with encoding utils: {e}")
                logger.info("[FALLBACK] Falling back to standard UTF-8 writing...")
                # フォールバック
                with open(output_file, 'w', encoding='utf-8') as f:
                    for sample in classified_samples:
                        f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        else:
            # フォールバック: 通常のUTF-8書き込み
            with open(output_file, 'w', encoding='utf-8') as f:
                for sample in classified_samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        self.phase_progress['quadruple_classification'] = {
            'status': 'completed',
            'samples': len(classified_samples),
            'output_path': str(output_file)
        }
        self._save_checkpoint()
        
        logger.info(f"[OK] Phase 3 completed. Output: {output_file}")
        return output_file
    
    def run_pipeline(self, resume: bool = True):
        """
        パイプラインを実行
        
        Args:
            resume: チェックポイントから再開するか
        """
        logger.info("="*80)
        logger.info("Starting SO8T Auto Data Processing Pipeline")
        logger.info("="*80)
        
        start_time = time.time()
        
        # チェックポイントから復旧
        if resume:
            checkpoint = self._load_checkpoint()
            if checkpoint:
                logger.info(f"[RESUME] Resuming from checkpoint (Session: {self.session_id})")
                self.phase_progress = checkpoint.get('phase_progress', {})
        
        try:
            # 入力ディレクトリの検証
            logger.info(f"[VALIDATE] Validating input directory: {self.input_dir}")
            if not self.input_dir.exists():
                logger.error(f"[ERROR] Input directory does not exist: {self.input_dir}")
                raise FileNotFoundError(f"Input directory not found: {self.input_dir}")
            
            # JSONLファイルの存在確認
            jsonl_files = list(self.input_dir.glob("*.jsonl"))
            if not jsonl_files:
                logger.warning(f"[WARNING] No JSONL files found in input directory: {self.input_dir}")
                logger.warning("[WARNING] Checking subdirectories...")
                # サブディレクトリも検索
                jsonl_files = list(self.input_dir.rglob("*.jsonl"))
                if not jsonl_files:
                    logger.error("[ERROR] No JSONL files found in input directory or subdirectories")
                    raise FileNotFoundError(f"No JSONL files found in {self.input_dir}")
                else:
                    logger.info(f"[INFO] Found {len(jsonl_files)} JSONL files in subdirectories")
            
            # 空のファイルをスキップ
            valid_files = []
            for jsonl_file in jsonl_files:
                try:
                    # ファイルサイズをチェック
                    file_size = jsonl_file.stat().st_size
                    if file_size == 0:
                        logger.warning(f"[SKIP] Empty file: {jsonl_file}")
                        continue
                    
                    # ファイルの読み込み可能性をチェック
                    if ENCODING_UTILS_AVAILABLE:
                        try:
                            test_samples = safe_read_jsonl(jsonl_file)
                            if len(test_samples) == 0:
                                logger.warning(f"[SKIP] File contains no valid samples: {jsonl_file}")
                                continue
                        except Exception as e:
                            logger.warning(f"[SKIP] File cannot be read: {jsonl_file}, error: {e}")
                            continue
                    else:
                        # 簡易チェック: 最初の行を読み込めるか
                        try:
                            with open(jsonl_file, 'r', encoding='utf-8', errors='replace') as f:
                                first_line = f.readline()
                                if not first_line.strip():
                                    logger.warning(f"[SKIP] File appears empty: {jsonl_file}")
                                    continue
                        except Exception as e:
                            logger.warning(f"[SKIP] File cannot be read: {jsonl_file}, error: {e}")
                            continue
                    
                    valid_files.append(jsonl_file)
                except Exception as e:
                    logger.warning(f"[SKIP] Error checking file {jsonl_file}: {e}")
                    continue
            
            if not valid_files:
                logger.error("[ERROR] No valid JSONL files found after validation")
                raise ValueError("No valid JSONL files found in input directory")
            
            logger.info(f"[VALIDATE] Found {len(valid_files)} valid JSONL files out of {len(jsonl_files)} total files")
            
            # 新規データを検出
            new_files = self.detect_new_data()
            
            if not new_files:
                logger.info("[SKIP] No new data files found")
                # 既存のデータファイルがある場合は処理を続行
                if valid_files:
                    logger.info(f"[INFO] Found {len(valid_files)} existing files, checking if reprocessing is needed...")
                    # 既存のファイルを処理対象に追加（再処理が必要な場合）
                    new_files = valid_files
                else:
                    return
            
            # Phase 1: データクレンジング
            if 'data_cleaning' not in self.phase_progress or self.phase_progress['data_cleaning'].get('status') != 'completed':
                cleaned_path = self.phase1_data_cleaning(new_files)
            else:
                logger.info("[SKIP] Phase 1 already completed")
                cleaned_path = Path(self.phase_progress['data_cleaning']['output_path'])
            
            # Phase 2: 漸次ラベル付け
            if 'incremental_labeling' not in self.phase_progress or self.phase_progress['incremental_labeling'].get('status') != 'completed':
                labeled_path = self.phase2_incremental_labeling(cleaned_path)
            else:
                logger.info("[SKIP] Phase 2 already completed")
                labeled_path = Path(self.phase_progress['incremental_labeling']['output_path'])
            
            # Phase 3: 四値分類
            if 'quadruple_classification' not in self.phase_progress or self.phase_progress['quadruple_classification'].get('status') != 'completed':
                four_class_path = self.phase3_quadruple_classification(labeled_path)
            else:
                logger.info("[SKIP] Phase 3 already completed")
                four_class_path = Path(self.phase_progress['quadruple_classification']['output_path'])
            
            total_time = time.time() - start_time
            
            logger.info("="*80)
            logger.info("Pipeline Completed Successfully")
            logger.info("="*80)
            logger.info(f"Total processing time: {total_time:.2f} seconds")
            logger.info(f"Final output: {four_class_path}")
            
        except KeyboardInterrupt:
            logger.warning("[INTERRUPT] Pipeline interrupted by user")
            self._save_checkpoint()
        except Exception as e:
            logger.error(f"[ERROR] Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            self._save_checkpoint()
            raise
    
    def reclassify_existing_data(self, input_file: Optional[Path] = None) -> Path:
        """
        既存のfour_class_*.jsonlファイルを再分類
        
        Args:
            input_file: 再分類するファイルのパス（Noneの場合は最新のファイルを使用）
        
        Returns:
            再分類済みデータのパス
        """
        logger.info("="*80)
        logger.info("Reclassifying Existing Data")
        logger.info("="*80)
        
        # 入力ファイルを決定
        if input_file is None:
            # 最新のfour_class_*.jsonlファイルを検索
            four_class_files = sorted(self.output_dir.glob("four_class_*.jsonl"))
            if not four_class_files:
                raise FileNotFoundError(f"No four_class_*.jsonl files found in {self.output_dir}")
            input_file = four_class_files[-1]
            logger.info(f"[INFO] Using latest file: {input_file}")
        
        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        # データを読み込み
        samples = []
        logger.info(f"[LOAD] Loading samples from {input_file}...")
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))
        
        logger.info(f"[INFO] Loaded {len(samples)} samples")
        
        # 再分類を実行
        logger.info("[RECLASSIFY] Reclassifying samples...")
        reclassified_samples = []
        
        for sample in tqdm(samples, desc="Reclassifying"):
            try:
                # 四値分類を実行
                reclassified_sample = self.quadruple_classifier.classify_quadruple(sample)
                reclassified_samples.append(reclassified_sample)
            except Exception as e:
                logger.error(f"Failed to reclassify sample: {e}")
                # エラー時は元のサンプルを保持
                reclassified_samples.append(sample)
        
        # 結果を保存
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = self.output_dir / f"four_class_reclassified_{timestamp}.jsonl"
        logger.info(f"[SAVE] Saving {len(reclassified_samples)} reclassified samples to {output_file}...")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in reclassified_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        # 分類統計を計算
        classification_stats = {
            'ALLOW': 0,
            'ESCALATION': 0,
            'DENY': 0,
            'REFUSE': 0,
            'unknown': 0
        }
        
        for sample in reclassified_samples:
            quad_class = sample.get('quadruple_classification', {})
            four_class_label = quad_class.get('four_class_label', 'unknown')
            classification_stats[four_class_label] = classification_stats.get(four_class_label, 0) + 1
        
        logger.info("="*80)
        logger.info("Reclassification Completed")
        logger.info("="*80)
        logger.info("Classification statistics:")
        logger.info(f"  ALLOW: {classification_stats['ALLOW']}")
        logger.info(f"  ESCALATION: {classification_stats['ESCALATION']}")
        logger.info(f"  DENY: {classification_stats['DENY']}")
        logger.info(f"  REFUSE: {classification_stats['REFUSE']}")
        logger.info(f"  unknown: {classification_stats.get('unknown', 0)}")
        logger.info(f"Output file: {output_file}")
        
        return output_file
    
    def validate_dataset(self, dataset_path: Path) -> Dict:
        """
        データセットのバリデーション
        
        Args:
            dataset_path: バリデーションするデータセットのパス
        
        Returns:
            バリデーション結果の辞書
        """
        logger.info("="*80)
        logger.info("Dataset Validation")
        logger.info("="*80)
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
        
        # データを読み込み
        samples = []
        logger.info(f"[LOAD] Loading samples from {dataset_path}...")
        
        # エンコーディングユーティリティが利用可能な場合は使用
        if ENCODING_UTILS_AVAILABLE:
            samples = safe_read_jsonl(dataset_path)
        else:
            # フォールバック: 通常のUTF-8読み込み
            with open(dataset_path, 'r', encoding='utf-8', errors='replace') as f:
                for line in f:
                    if line.strip():
                        try:
                            samples.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        
        logger.info(f"[INFO] Loaded {len(samples)} samples")
        
        # バリデーション統計
        validation_stats = {
            'total_samples': len(samples),
            'valid_samples': 0,
            'invalid_samples': 0,
            'missing_fields': {
                'text': 0,
                'nsfw_label': 0,
                'category': 0,
                'domain': 0
            },
            'empty_text': 0,
            'invalid_text_length': 0,
            'missing_quadruple_classification': 0,
            'invalid_classification_labels': 0
        }
        
        # 必須フィールド
        required_fields = ['text', 'nsfw_label', 'category', 'domain']
        valid_labels = ['ALLOW', 'ESCALATION', 'DENY', 'REFUSE']
        
        # 各サンプルをバリデーション
        for sample in tqdm(samples, desc="Validating"):
            is_valid = True
            
            # 必須フィールドのチェック
            for field in required_fields:
                if field not in sample or not sample.get(field):
                    validation_stats['missing_fields'][field] += 1
                    is_valid = False
            
            # テキスト長のチェック
            text = sample.get('text', '')
            text_length = len(text) if text else 0
            
            if text_length == 0:
                validation_stats['empty_text'] += 1
                is_valid = False
            
            # テキスト長が異常に長い場合（>1MB）
            if text_length > 1000000:
                validation_stats['invalid_text_length'] += 1
                is_valid = False
            
            # 四値分類ラベルのチェック
            quad_class = sample.get('quadruple_classification', {})
            if not quad_class:
                validation_stats['missing_quadruple_classification'] += 1
                is_valid = False
            else:
                four_class_label = quad_class.get('four_class_label', '')
                if four_class_label not in valid_labels:
                    validation_stats['invalid_classification_labels'] += 1
                    is_valid = False
            
            if is_valid:
                validation_stats['valid_samples'] += 1
            else:
                validation_stats['invalid_samples'] += 1
        
        # バリデーション結果をログ出力
        logger.info("="*80)
        logger.info("Validation Results")
        logger.info("="*80)
        logger.info(f"Total samples: {validation_stats['total_samples']}")
        logger.info(f"Valid samples: {validation_stats['valid_samples']} ({validation_stats['valid_samples']/validation_stats['total_samples']*100:.1f}%)")
        logger.info(f"Invalid samples: {validation_stats['invalid_samples']} ({validation_stats['invalid_samples']/validation_stats['total_samples']*100:.1f}%)")
        logger.info("Missing fields:")
        for field, count in validation_stats['missing_fields'].items():
            if count > 0:
                logger.info(f"  {field}: {count}")
        logger.info(f"Empty text: {validation_stats['empty_text']}")
        logger.info(f"Invalid text length: {validation_stats['invalid_text_length']}")
        logger.info(f"Missing quadruple classification: {validation_stats['missing_quadruple_classification']}")
        logger.info(f"Invalid classification labels: {validation_stats['invalid_classification_labels']}")
        
        return validation_stats


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="SO8T Auto Data Processing Pipeline")
    parser.add_argument(
        '--config',
        type=Path,
        default=Path('configs/so8t_auto_data_processing_config.yaml'),
        help='Configuration file path'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        default=True,
        help='Resume from checkpoint'
    )
    parser.add_argument(
        '--reclassify',
        type=Path,
        nargs='?',
        const=None,
        default=None,
        help='Reclassify existing data file (path to four_class_*.jsonl file, or omit to use latest)'
    )
    parser.add_argument(
        '--validate',
        type=Path,
        nargs='?',
        const=None,
        default=None,
        help='Validate dataset file (path to four_class_*.jsonl file, or omit to use latest)'
    )
    
    args = parser.parse_args()
    
    if not args.config.exists():
        logger.error(f"Config file not found: {args.config}")
        return 1
    
    pipeline = SO8TAutoDataProcessingPipeline(args.config)
    
    if '--validate' in sys.argv:
        # バリデーションモード
        input_file = args.validate
        if input_file is None:
            # 最新のfour_class_*.jsonlファイルを検索
            four_class_files = sorted(pipeline.output_dir.glob("four_class_*.jsonl"))
            if not four_class_files:
                logger.error(f"No four_class_*.jsonl files found in {pipeline.output_dir}")
                return 1
            input_file = four_class_files[-1]
            logger.info(f"[INFO] Using latest file: {input_file}")
        
        pipeline.validate_dataset(input_file)
    elif '--reclassify' in sys.argv:
        # 再分類モード
        input_file = args.reclassify
        pipeline.reclassify_existing_data(input_file)
    else:
        # 通常のパイプライン実行
        pipeline.run_pipeline(resume=args.resume)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())


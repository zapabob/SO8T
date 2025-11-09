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
        self.quadruple_classifier = QuadrupleClassifier(self.so8t_model_path) if self.use_so8t else None
        
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
                with open(input_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            sample = json.loads(line)
                            cleaned_sample = self.cleaner.clean_sample(sample)
                            if cleaned_sample is not None:
                                cleaned_samples.append(cleaned_sample)
                            else:
                                logger.debug(f"[CLEANING] Excluded invalid sample from {input_file}")
                
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
        
        # 結果を保存
        output_file = self.output_dir / f"cleaned_{self.session_id}.jsonl"
        logger.info(f"[SAVE] Saving {len(cleaned_samples)} cleaned samples to {output_file}...")
        
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
        
        # チェックポイントから復旧
        checkpoint = self._load_checkpoint()
        if checkpoint and checkpoint.get('current_phase') == 'incremental_labeling':
            logger.info("[RESUME] Resuming incremental labeling from checkpoint...")
            phase_progress = checkpoint.get('phase_progress', {}).get('incremental_labeling', {})
            processed_count = phase_progress.get('processed_count', 0)
            labeled_samples = phase_progress.get('labeled_samples', [])
        else:
            processed_count = 0
            labeled_samples = []
        
        # データを読み込み
        samples = []
        with open(input_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i < processed_count:
                    continue
                if line.strip():
                    samples.append(json.loads(line))
        
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
                logger.error(f"Failed to label sample: {e}")
        
        # 結果を保存
        output_file = self.output_dir / f"labeled_{self.session_id}.jsonl"
        logger.info(f"[SAVE] Saving {len(labeled_samples)} labeled samples to {output_file}...")
        
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
        
        # チェックポイントから復旧
        checkpoint = self._load_checkpoint()
        if checkpoint and checkpoint.get('current_phase') == 'quadruple_classification':
            logger.info("[RESUME] Resuming quadruple classification from checkpoint...")
            phase_progress = checkpoint.get('phase_progress', {}).get('quadruple_classification', {})
            processed_count = phase_progress.get('processed_count', 0)
            classified_samples = phase_progress.get('classified_samples', [])
        else:
            processed_count = 0
            classified_samples = []
        
        # データを読み込み
        samples = []
        with open(input_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i < processed_count:
                    continue
                if line.strip():
                    samples.append(json.loads(line))
        
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
                logger.error(f"Failed to classify sample: {e}")
        
        # 結果を保存
        output_file = self.output_dir / f"four_class_{self.session_id}.jsonl"
        logger.info(f"[SAVE] Saving {len(classified_samples)} classified samples to {output_file}...")
        
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
            # 新規データを検出
            new_files = self.detect_new_data()
            
            if not new_files:
                logger.info("[SKIP] No new data files found")
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
        logger.info(f"Classification statistics:")
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
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))
        
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
        logger.info(f"Missing fields:")
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


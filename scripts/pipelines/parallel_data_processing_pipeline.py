#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
逐次データクレンジング・データセット作成並行パイプライン

スクレイピングと並行実行、マルチプロセス・マルチスレッドによる効率化

Usage:
    python scripts/pipelines/parallel_data_processing_pipeline.py --input D:/webdataset/processed --output D:/webdataset/cleaned
"""

import sys
import json
import logging
import argparse
import asyncio
import threading
import queue
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "pipelines"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "audit"))

# データクレンジングインポート
try:
    from scripts.pipelines.web_scraping_data_pipeline import DataCleaner, QuadrupleClassifier
    DATA_CLEANER_AVAILABLE = True
except ImportError:
    DATA_CLEANER_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("Data cleaner not available")

# 監査ログインポート
try:
    from scripts.audit.scraping_audit_logger import ScrapingAuditLogger, DataCleaningEvent, DatasetCreationEvent
    AUDIT_LOGGER_AVAILABLE = True
except ImportError:
    AUDIT_LOGGER_AVAILABLE = False

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/parallel_data_processing_pipeline.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ParallelDataProcessingPipeline:
    """逐次データクレンジング・データセット作成並行パイプライン"""
    
    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        num_workers: int = 4,
        batch_size: int = 100,
        audit_logger: Optional[ScrapingAuditLogger] = None
    ):
        """
        初期化
        
        Args:
            input_dir: 入力ディレクトリ
            output_dir: 出力ディレクトリ
            num_workers: 並列処理ワーカー数
            batch_size: バッチサイズ
            audit_logger: 監査ロガー
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.audit_logger = audit_logger
        
        # データクレーナー初期化
        self.cleaner = None
        if DATA_CLEANER_AVAILABLE:
            try:
                self.cleaner = DataCleaner()
                logger.info("[CLEANER] Data cleaner initialized")
            except Exception as e:
                logger.warning(f"[CLEANER] Failed to initialize data cleaner: {e}")
        
        # 四値分類器初期化
        self.classifier = None
        if DATA_CLEANER_AVAILABLE:
            try:
                self.classifier = QuadrupleClassifier()
                logger.info("[CLASSIFIER] Quadruple classifier initialized")
            except Exception as e:
                logger.warning(f"[CLASSIFIER] Failed to initialize classifier: {e}")
        
        # 処理キュー
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()
        
        # 進捗管理
        self.processed_count = 0
        self.cleaned_count = 0
        self.classified_count = 0
        
        logger.info("="*80)
        logger.info("Parallel Data Processing Pipeline Initialized")
        logger.info("="*80)
        logger.info(f"Input directory: {self.input_dir}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Number of workers: {num_workers}")
        logger.info(f"Batch size: {batch_size}")
    
    def load_input_files(self) -> List[Path]:
        """
        入力ファイルを読み込み
        
        Returns:
            files: JSONLファイルのリスト
        """
        jsonl_files = list(self.input_dir.glob("*.jsonl"))
        logger.info(f"[LOAD] Found {len(jsonl_files)} JSONL files")
        return jsonl_files
    
    def process_sample(self, sample: Dict) -> Optional[Dict]:
        """
        サンプルを処理（クレンジング・分類）
        
        Args:
            sample: サンプル辞書
            
        Returns:
            processed_sample: 処理済みサンプル（Noneの場合は無効）
        """
        try:
            # データクレンジング
            if self.cleaner:
                cleaned_sample = self.cleaner.clean_sample(sample)
                if not cleaned_sample:
                    return None
            else:
                cleaned_sample = sample.copy()
            
            # 四値分類
            if self.classifier:
                classified_sample = self.classifier.classify_quadruple(cleaned_sample)
            else:
                classified_sample = cleaned_sample.copy()
                classified_sample['four_class_label'] = 'ALLOW'
                classified_sample['four_class_label_id'] = 0
            
            return classified_sample
            
        except Exception as e:
            logger.error(f"[PROCESS] Failed to process sample: {e}")
            return None
    
    def process_batch(self, samples: List[Dict]) -> List[Dict]:
        """
        バッチを処理
        
        Args:
            samples: サンプルのリスト
            
        Returns:
            processed_samples: 処理済みサンプルのリスト
        """
        processed_samples = []
        
        for sample in samples:
            processed = self.process_sample(sample)
            if processed:
                processed_samples.append(processed)
        
        return processed_samples
    
    def worker_thread(self, worker_id: int):
        """
        ワーカースレッド
        
        Args:
            worker_id: ワーカーID
        """
        logger.info(f"[WORKER {worker_id}] Started")
        
        while True:
            try:
                # キューからバッチを取得
                batch = self.input_queue.get(timeout=5.0)
                
                if batch is None:  # 終了シグナル
                    break
                
                # バッチを処理
                processed_batch = self.process_batch(batch)
                
                # 出力キューに追加
                self.output_queue.put(processed_batch)
                
                # 進捗更新
                self.processed_count += len(batch)
                self.cleaned_count += len(processed_batch)
                
                self.input_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"[WORKER {worker_id}] Error: {e}")
        
        logger.info(f"[WORKER {worker_id}] Finished")
    
    def writer_thread(self):
        """ライタースレッド（処理済みデータを書き込み）"""
        logger.info("[WRITER] Started")
        
        output_file = self.output_dir / f"cleaned_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            while True:
                try:
                    # 出力キューからバッチを取得
                    batch = self.output_queue.get(timeout=5.0)
                    
                    if batch is None:  # 終了シグナル
                        break
                    
                    # ファイルに書き込み
                    for sample in batch:
                        f.write(json.dumps(sample, ensure_ascii=False) + '\n')
                        f.flush()
                    
                    self.output_queue.task_done()
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"[WRITER] Error: {e}")
        
        logger.info(f"[WRITER] Finished. Output file: {output_file}")
        return output_file
    
    def run_parallel_processing(self):
        """
        並行処理を実行
        
        Returns:
            output_file: 出力ファイルパス
        """
        logger.info("="*80)
        logger.info("Starting Parallel Data Processing")
        logger.info("="*80)
        
        # 入力ファイルを読み込み
        input_files = self.load_input_files()
        
        if not input_files:
            logger.warning("[PROCESS] No input files found")
            return None
        
        # ワーカースレッド起動
        workers = []
        for i in range(self.num_workers):
            worker = threading.Thread(target=self.worker_thread, args=(i,))
            worker.daemon = True
            worker.start()
            workers.append(worker)
        
        # ライタースレッド起動
        writer = threading.Thread(target=self.writer_thread)
        writer.daemon = True
        writer.start()
        
        # 入力ファイルを読み込んでキューに追加
        total_samples = 0
        for input_file in tqdm(input_files, desc="Loading files"):
            try:
                with open(input_file, 'r', encoding='utf-8') as f:
                    batch = []
                    for line in f:
                        if line.strip():
                            try:
                                sample = json.loads(line)
                                batch.append(sample)
                                total_samples += 1
                                
                                # バッチサイズに達したらキューに追加
                                if len(batch) >= self.batch_size:
                                    self.input_queue.put(batch)
                                    batch = []
                            except json.JSONDecodeError:
                                continue
                    
                    # 残りのバッチを追加
                    if batch:
                        self.input_queue.put(batch)
            
            except Exception as e:
                logger.error(f"[LOAD] Failed to load {input_file}: {e}")
        
        # 終了シグナルを送信
        for _ in range(self.num_workers):
            self.input_queue.put(None)
        
        self.output_queue.put(None)
        
        # ワーカーの完了を待機
        for worker in workers:
            worker.join()
        
        writer.join()
        
        logger.info(f"[PROCESS] Processed {self.processed_count} samples")
        logger.info(f"[PROCESS] Cleaned {self.cleaned_count} samples")
        
        # 監査ログ記録
        if self.audit_logger:
            event = DataCleaningEvent(
                event_id=f"cleaning_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                timestamp=datetime.now().isoformat(),
                input_samples=self.processed_count,
                output_samples=self.cleaned_count,
                cleaning_method="parallel_processing",
                quality_score=self.cleaned_count / max(self.processed_count, 1)
            )
            self.audit_logger.log_data_cleaning_event(event)
        
        return self.output_dir / f"cleaned_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    
    def create_dataset(
        self,
        cleaned_file: Path,
        dataset_name: str,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1
    ) -> Dict[str, Path]:
        """
        データセットを作成
        
        Args:
            cleaned_file: クレンジング済みファイル
            dataset_name: データセット名
            train_ratio: 訓練データ比率
            val_ratio: 検証データ比率
            
        Returns:
            dataset_files: データセットファイルの辞書
        """
        logger.info(f"[DATASET] Creating dataset: {dataset_name}")
        
        # サンプルを読み込み
        samples = []
        with open(cleaned_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))
        
        # シャッフル
        import random
        random.shuffle(samples)
        
        # 分割
        total = len(samples)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)
        
        train_samples = samples[:train_end]
        val_samples = samples[train_end:val_end]
        test_samples = samples[val_end:]
        
        # データセットディレクトリ作成
        dataset_dir = self.output_dir / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # ファイル保存
        dataset_files = {}
        
        if train_samples:
            train_file = dataset_dir / "train.jsonl"
            with open(train_file, 'w', encoding='utf-8') as f:
                for sample in train_samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            dataset_files['train'] = train_file
            logger.info(f"[DATASET] Train: {len(train_samples)} samples")
        
        if val_samples:
            val_file = dataset_dir / "val.jsonl"
            with open(val_file, 'w', encoding='utf-8') as f:
                for sample in val_samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            dataset_files['val'] = val_file
            logger.info(f"[DATASET] Val: {len(val_samples)} samples")
        
        if test_samples:
            test_file = dataset_dir / "test.jsonl"
            with open(test_file, 'w', encoding='utf-8') as f:
                for sample in test_samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            dataset_files['test'] = test_file
            logger.info(f"[DATASET] Test: {len(test_samples)} samples")
        
        # 監査ログ記録
        if self.audit_logger:
            for dataset_type, dataset_file in dataset_files.items():
                event = DatasetCreationEvent(
                    event_id=f"dataset_{dataset_name}_{dataset_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    timestamp=datetime.now().isoformat(),
                    dataset_name=dataset_name,
                    dataset_type=dataset_type,
                    samples_count=len(train_samples if dataset_type == 'train' else val_samples if dataset_type == 'val' else test_samples),
                    source_sessions=[],
                    details={
                        'train_ratio': train_ratio,
                        'val_ratio': val_ratio,
                        'test_ratio': 1.0 - train_ratio - val_ratio
                    }
                )
                self.audit_logger.log_dataset_creation_event(event)
        
        logger.info(f"[DATASET] Dataset created: {dataset_name}")
        return dataset_files


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="Parallel Data Processing Pipeline")
    parser.add_argument('--input', type=Path, required=True, help='Input directory')
    parser.add_argument('--output', type=Path, required=True, help='Output directory')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--batch-size', type=int, default=100, help='Batch size')
    parser.add_argument('--dataset-name', type=str, help='Dataset name (optional)')
    parser.add_argument('--create-dataset', action='store_true', help='Create dataset after cleaning')
    
    args = parser.parse_args()
    
    # 監査ロガー初期化
    audit_logger = None
    if AUDIT_LOGGER_AVAILABLE:
        try:
            audit_logger = ScrapingAuditLogger()
        except Exception as e:
            logger.warning(f"Failed to initialize audit logger: {e}")
    
    # パイプライン初期化
    pipeline = ParallelDataProcessingPipeline(
        input_dir=args.input,
        output_dir=args.output,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        audit_logger=audit_logger
    )
    
    # 並行処理実行
    cleaned_file = pipeline.run_parallel_processing()
    
    # データセット作成
    if args.create_dataset and cleaned_file:
        dataset_name = args.dataset_name or f"dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        dataset_files = pipeline.create_dataset(cleaned_file, dataset_name)
        logger.info(f"[OK] Dataset created: {dataset_files}")
    
    logger.info("[OK] Processing completed")


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
統合マスターパイプライン

すべての全自動パイプラインを統合して、電源投入時に自動実行できるようにする

Usage:
    python scripts/pipelines/unified_master_pipeline.py --config configs/unified_master_pipeline_config.yaml
    python scripts/pipelines/unified_master_pipeline.py --setup  # Windowsタスクスケジューラ登録
    python scripts/pipelines/unified_master_pipeline.py --run    # パイプライン実行
"""

import sys
import json
import logging
import argparse
import subprocess
import signal
import pickle
import yaml
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import warnings
warnings.filterwarnings("ignore")

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/unified_master_pipeline.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class AudioNotifier:
    """音声通知クラス"""
    
    @staticmethod
    def play_notification():
        """音声通知を再生"""
        audio_path = PROJECT_ROOT / ".cursor" / "marisa_owattaze.wav"
        if audio_path.exists():
            try:
                subprocess.run([
                    "powershell", "-ExecutionPolicy", "Bypass", "-File",
                    str(PROJECT_ROOT / "scripts" / "utils" / "play_audio_notification.ps1")
                ], check=False, timeout=10)
            except Exception as e:
                logger.warning(f"Failed to play audio notification: {e}")
                try:
                    import winsound
                    winsound.Beep(1000, 500)
                except Exception:
                    pass


def check_admin_privileges() -> bool:
    """管理者権限をチェック"""
    try:
        result = subprocess.run(
            ["net", "session"],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0
    except Exception:
        return False


class UnifiedMasterPipeline:
    """統合マスターパイプライン"""
    
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
        self.session_id = datetime.now().strftime(self.config.get('session_id_format', '%Y%m%d_%H%M%S'))
        self.checkpoint_dir = Path(self.config.get('checkpoint_dir', 'D:/webdataset/checkpoints/unified_master_pipeline'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # 各パイプラインの設定
        self.phase1_config = self.config.get('phase1_parallel_scraping', {})
        self.phase2_config = self.config.get('phase2_data_processing', {})
        self.phase3_config = self.config.get('phase3_ab_test', {})
        self.phase4_config = self.config.get('phase4_github_scraping', {})
        self.phase5_config = self.config.get('phase5_engineer_sites', {})
        self.phase6_config = self.config.get('phase6_coding_extraction', {})
        self.phase7_config = self.config.get('phase7_coding_training_data', {})
        self.phase8_config = self.config.get('phase8_coding_retraining', {})
        self.phase9_config = self.config.get('phase9_documentation_scraping', {})
        self.phase10_config = self.config.get('phase10_unified_agent_base', {})
        self.phase11_config = self.config.get('phase11_nsfw_detection_dataset', {})
        
        # 進捗管理
        self.current_phase = None
        self.phase_progress = {}
        
        # シグナルハンドラー設定
        self._setup_signal_handlers()
        
        logger.info("="*80)
        logger.info("Unified Master Pipeline Initialized")
        logger.info("="*80)
        logger.info(f"Session ID: {self.session_id}")
        logger.info(f"Checkpoint directory: {self.checkpoint_dir}")
    
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
    
    def _count_current_samples(self, use_cache: bool = True) -> int:
        """
        現在のサンプル数をカウント（キャッシュ対応）
        
        Args:
            use_cache: キャッシュを使用するか
        
        Returns:
            サンプル数
        """
        # チェックポイントからキャッシュを読み込み
        if use_cache:
            checkpoint = self._load_checkpoint()
            if checkpoint:
                sample_cache = checkpoint.get('sample_cache', {})
                cached_count = sample_cache.get('count')
                cached_time = sample_cache.get('timestamp')
                
                if cached_count is not None and cached_time:
                    cache_age = (datetime.now() - datetime.fromisoformat(cached_time)).total_seconds()
                    cache_ttl = 300  # 5分間のキャッシュ有効期限
                    
                    if cache_age < cache_ttl:
                        logger.debug(f"[CACHE] Using cached sample count: {cached_count} (age: {cache_age:.1f}s)")
                        return cached_count
        
        # キャッシュが無効または存在しない場合は再カウント
        base_output = Path(self.phase1_config.get('base_output', 'D:/webdataset/processed'))
        total_samples = 0
        
        try:
            jsonl_files = list(base_output.rglob("*.jsonl"))
            logger.info(f"[COUNT] Found {len(jsonl_files)} JSONL files to count")
            
            for jsonl_file in jsonl_files:
                try:
                    # エンコーディングユーティリティが利用可能な場合は使用
                    try:
                        from scripts.utils.encoding_utils import safe_read_jsonl
                        samples = safe_read_jsonl(jsonl_file)
                        total_samples += len(samples)
                    except ImportError:
                        # フォールバック: 通常のカウント
                        count = sum(1 for _ in open(jsonl_file, 'r', encoding='utf-8', errors='ignore'))
                        total_samples += count
                except Exception as e:
                    logger.debug(f"[COUNT] Failed to count {jsonl_file}: {e}")
                    continue
            
            logger.info(f"[COUNT] Total samples counted: {total_samples}")
            
            # カウント結果をキャッシュに保存
            self.phase_progress['sample_cache'] = {
                'count': total_samples,
                'timestamp': datetime.now().isoformat()
            }
            self._save_checkpoint()
            
        except Exception as e:
            logger.warning(f"[COUNT] Failed to count samples: {e}")
        
        return total_samples
    
    def _get_dynamic_threshold(self) -> int:
        """
        データセットの種類や品質に基づいて動的に最小サンプル数を計算
        
        Returns:
            動的に計算された最小サンプル数
        """
        if not self.phase3_config.get('dynamic_threshold', False):
            return self.phase3_config.get('min_samples_for_retraining', 50000)
        
        # データセットタイプと品質を分析
        dataset_type = self._analyze_dataset_type()
        quality_score = self._calculate_quality_score()
        
        # タイプごとの閾値を取得
        threshold_by_type = self.phase3_config.get('threshold_by_dataset_type', {})
        base_threshold = threshold_by_type.get(dataset_type, 50000)
        
        # 品質スコアに基づいて調整
        if quality_score >= 0.8:
            quality_multiplier = threshold_by_type.get('high_quality', 30000) / 50000
        elif quality_score >= 0.5:
            quality_multiplier = threshold_by_type.get('medium_quality', 50000) / 50000
        else:
            quality_multiplier = threshold_by_type.get('low_quality', 100000) / 50000
        
        dynamic_threshold = int(base_threshold * quality_multiplier)
        
        logger.info(f"[DYNAMIC_THRESHOLD] Dataset type: {dataset_type}, Quality score: {quality_score:.2f}")
        logger.info(f"[DYNAMIC_THRESHOLD] Base threshold: {base_threshold}, Dynamic threshold: {dynamic_threshold}")
        
        return dynamic_threshold
    
    def _analyze_dataset_type(self) -> str:
        """
        データセットタイプを分析
        
        Returns:
            データセットタイプ（'nsfw_detection', 'general'）
        """
        base_output = Path(self.phase1_config.get('base_output', 'D:/webdataset/processed'))
        
        # NSFW検知用データセットのパスをチェック
        nsfw_paths = [
            base_output / "nsfw_detection",
            base_output / "nsfw",
            base_output / "safety"
        ]
        
        for nsfw_path in nsfw_paths:
            if nsfw_path.exists() and list(nsfw_path.rglob("*.jsonl")):
                return "nsfw_detection"
        
        return "general"
    
    def _calculate_quality_score(self, use_cache: bool = True) -> float:
        """
        データセットの品質スコアを計算（キャッシュ対応）
        
        Args:
            use_cache: キャッシュを使用するか
        
        Returns:
            品質スコア（0.0-1.0）
        """
        # チェックポイントからキャッシュを読み込み
        if use_cache:
            checkpoint = self._load_checkpoint()
            if checkpoint:
                quality_cache = checkpoint.get('quality_cache', {})
                cached_score = quality_cache.get('score')
                cached_time = quality_cache.get('timestamp')
                cached_dataset_hash = quality_cache.get('dataset_hash')
                
                # 現在のデータセットハッシュを計算
                current_hash = self._calculate_dataset_hash()
                
                if cached_score is not None and cached_time and cached_dataset_hash == current_hash:
                    cache_age = (datetime.now() - datetime.fromisoformat(cached_time)).total_seconds()
                    cache_ttl = 600  # 10分間のキャッシュ有効期限
                    
                    if cache_age < cache_ttl:
                        logger.debug(f"[CACHE] Using cached quality score: {cached_score:.2f} (age: {cache_age:.1f}s)")
                        return cached_score
        
        # キャッシュが無効または存在しない場合は再計算
        logger.info("[INFO] Calculating quality score (this may take a while)...")
        
        # 品質スコア計算ロジック
        quality_score = self._compute_quality_metrics()
        
        # 計算結果をキャッシュに保存
        current_hash = self._calculate_dataset_hash()
        self.phase_progress['quality_cache'] = {
            'score': quality_score,
            'timestamp': datetime.now().isoformat(),
            'dataset_hash': current_hash
        }
        self._save_checkpoint()
        
        logger.info(f"[INFO] Quality score calculated: {quality_score:.2f}")
        
        return quality_score
    
    def _calculate_dataset_hash(self) -> str:
        """
        データセットのハッシュ値を計算（データセットが変更されたかを検出）
        
        Returns:
            データセットハッシュ値
        """
        import hashlib
        
        # データセットファイルのパスとサイズからハッシュを計算
        base_output = Path(self.phase1_config.get('base_output', 'D:/webdataset/processed'))
        dataset_paths = [
            base_output / "finetuning" / "train.jsonl",
            base_output / "finetuning" / "val.jsonl",
        ]
        
        hash_input = ""
        for path in dataset_paths:
            if path.exists():
                hash_input += f"{path}:{path.stat().st_size}:{path.stat().st_mtime}"
        
        return hashlib.md5(hash_input.encode()).hexdigest()
    
    def _compute_quality_metrics(self) -> float:
        """
        データセットの品質メトリクスを計算
        
        Returns:
            品質スコア（0.0-1.0）
        """
        base_output = Path(self.phase1_config.get('base_output', 'D:/webdataset/processed'))
        jsonl_files = list(base_output.rglob("*.jsonl"))
        
        if not jsonl_files:
            return 0.5  # デフォルト品質スコア
        
        total_samples = 0
        total_length = 0
        encoding_errors = 0
        
        try:
            for jsonl_file in jsonl_files[:100]:  # サンプリング（最大100ファイル）
                try:
                    from scripts.utils.encoding_utils import safe_read_jsonl
                    samples = safe_read_jsonl(jsonl_file)
                    total_samples += len(samples)
                    
                    for sample in samples:
                        if isinstance(sample, dict):
                            text = str(sample.get('text', ''))
                            total_length += len(text)
                except Exception as e:
                    encoding_errors += 1
                    logger.debug(f"[QUALITY] Error processing {jsonl_file}: {e}")
        except Exception as e:
            logger.warning(f"[QUALITY] Failed to compute quality metrics: {e}")
            return 0.5
        
        # 品質スコア計算（簡易版）
        # - 平均テキスト長が適切（50-10000文字）
        # - エンコーディングエラー率が低い
        avg_length = total_length / total_samples if total_samples > 0 else 0
        error_rate = encoding_errors / len(jsonl_files) if jsonl_files else 0
        
        # 品質スコア（0.0-1.0）
        length_score = min(1.0, max(0.0, (avg_length - 50) / 1000)) if avg_length > 50 else 0.0
        error_score = max(0.0, 1.0 - error_rate * 2)  # エラー率が高いほどスコアが低い
        
        quality_score = (length_score * 0.7 + error_score * 0.3)
        
        return min(1.0, max(0.0, quality_score))
    
    def _save_checkpoint(self):
        """チェックポイント保存"""
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
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
    
    def _is_phase_completed(self, phase_name: str) -> bool:
        """
        フェーズの完了状態を検証
        
        Args:
            phase_name: フェーズ名
        
        Returns:
            completed: 完了しているかどうか
        """
        # チェックポイントからステータスを確認
        phase_progress = self.phase_progress.get(phase_name, {})
        status = phase_progress.get('status', 'pending')
        
        if status != 'completed':
            return False
        
        # Phase 1: 並列DeepResearch Webスクレイピング
        if phase_name == 'phase1_parallel_scraping':
            # バックグラウンド実行のため、プロセスIDの確認のみ
            process_id = phase_progress.get('process_id')
            if process_id:
                # プロセスが実行中か確認（オプション）
                try:
                    import psutil
                    if psutil.pid_exists(process_id):
                        logger.debug(f"[VERIFY] Phase 1: Process {process_id} is still running")
                        return True  # 実行中なら完了とみなす（バックグラウンド実行のため）
                except ImportError:
                    pass  # psutilがインストールされていない場合はスキップ
            
            # 出力ディレクトリの存在確認
            base_output = Path(self.phase1_config.get('base_output', 'D:/webdataset/processed'))
            if base_output.exists():
                logger.info(f"[VERIFY] Phase 1: Output directory exists: {base_output}")
                return True
            else:
                logger.warning(f"[VERIFY] Phase 1: Output directory not found: {base_output}")
                return False
        
        # Phase 2: SO8T全自動データ処理
        elif phase_name == 'phase2_data_processing':
            phase2_config = self.phase2_config
            if not phase2_config.get('enabled', True):
                logger.info("[VERIFY] Phase 2: Disabled in config")
                return True
            
            # データ処理パイプラインの出力ディレクトリを確認
            # 設定ファイルから出力パスを取得
            config_path = PROJECT_ROOT / phase2_config.get('config', 'configs/so8t_auto_data_processing_config.yaml')
            if config_path.exists():
                try:
                    import yaml
                    with open(config_path, 'r', encoding='utf-8') as f:
                        data_processing_config = yaml.safe_load(f)
                    output_dir = Path(data_processing_config.get('output_dir', 'D:/webdataset/processed/four_class'))
                    
                    # 出力ディレクトリが存在し、ファイルが含まれているか確認
                    if output_dir.exists():
                        output_files = list(output_dir.glob("*.jsonl"))
                        if output_files:
                            logger.info(f"[VERIFY] Phase 2: Output directory exists with {len(output_files)} files: {output_dir}")
                            return True
                        else:
                            logger.warning(f"[VERIFY] Phase 2: Output directory exists but no files found: {output_dir}")
                            return False
                    else:
                        logger.warning(f"[VERIFY] Phase 2: Output directory not found: {output_dir}")
                        return False
                except Exception as e:
                    logger.warning(f"[VERIFY] Phase 2: Failed to verify output (fallback to status check): {e}")
                    return True  # エラー時はステータスのみで判定
            else:
                logger.debug("[VERIFY] Phase 2: Config file not found, using status check only")
                return True
        
        # Phase 3: SO8T完全統合A/Bテスト
        elif phase_name == 'phase3_ab_test':
            phase3_config = self.phase3_config
            if not phase3_config.get('enabled', True):
                logger.info("[VERIFY] Phase 3: Disabled in config")
                return True
            
            # A/Bテストの結果ファイルを確認
            # 設定ファイルから出力パスを取得
            config_path = PROJECT_ROOT / phase3_config.get('config', 'configs/complete_so8t_ab_test_pipeline_config.yaml')
            if config_path.exists():
                try:
                    import yaml
                    with open(config_path, 'r', encoding='utf-8') as f:
                        ab_test_config = yaml.safe_load(f)
                    output_base_dir = Path(ab_test_config.get('output_base_dir', 'D:/webdataset'))
                    ab_test_results_dir = output_base_dir / 'ab_test_results'
                    
                    # 結果ディレクトリが存在し、結果ファイルが含まれているか確認
                    if ab_test_results_dir.exists():
                        result_dirs = [d for d in ab_test_results_dir.iterdir() if d.is_dir() and d.name.startswith('complete_so8t_ab_test_')]
                        if result_dirs:
                            # 最新の結果ディレクトリを確認
                            latest_result_dir = max(result_dirs, key=lambda d: d.stat().st_mtime)
                            results_file = latest_result_dir / 'ab_test_results.json'
                            if results_file.exists():
                                logger.info(f"[VERIFY] Phase 3: A/B test results found: {results_file}")
                                return True
                            else:
                                logger.warning(f"[VERIFY] Phase 3: Results directory exists but no results file: {latest_result_dir}")
                                return False
                        else:
                            logger.warning(f"[VERIFY] Phase 3: No result directories found: {ab_test_results_dir}")
                            return False
                    else:
                        logger.warning(f"[VERIFY] Phase 3: Results directory not found: {ab_test_results_dir}")
                        return False
                except Exception as e:
                    logger.warning(f"[VERIFY] Phase 3: Failed to verify results (fallback to status check): {e}")
                    return True  # エラー時はステータスのみで判定
            else:
                logger.debug("[VERIFY] Phase 3: Config file not found, using status check only")
                return True
        
        # Phase 4: GitHubリポジトリ検索
        elif phase_name == 'phase4_github_scraping':
            output_dir = Path(self.phase4_config.get('output_dir', 'D:/webdataset/processed/github'))
            github_files = list(output_dir.glob('github_repositories_*.jsonl'))
            if github_files:
                latest_file = max(github_files, key=lambda p: p.stat().st_mtime)
                if latest_file.stat().st_size > 0:
                    logger.info(f"[VERIFY] Phase 4: GitHub scraping results found: {latest_file}")
                    return True
            return False
        
        # Phase 5: エンジニア向けサイトスクレイピング
        elif phase_name == 'phase5_engineer_sites':
            output_dir = Path(self.phase5_config.get('output_dir', 'D:/webdataset/processed/engineer_sites'))
            engineer_files = list(output_dir.glob('engineer_sites_*.jsonl'))
            if engineer_files:
                latest_file = max(engineer_files, key=lambda p: p.stat().st_mtime)
                if latest_file.stat().st_size > 0:
                    logger.info(f"[VERIFY] Phase 5: Engineer sites scraping results found: {latest_file}")
                    return True
            return False
        
        # Phase 6: コーディング関連データ抽出
        elif phase_name == 'phase6_coding_extraction':
            output_dir = Path(self.phase6_config.get('output_dir', 'D:/webdataset/coding_dataset'))
            coding_files = list(output_dir.glob('coding_*.jsonl'))
            if coding_files:
                latest_file = max(coding_files, key=lambda p: p.stat().st_mtime)
                if latest_file.stat().st_size > 0:
                    logger.info(f"[VERIFY] Phase 6: Coding extraction results found: {latest_file}")
                    return True
            return False
        
        # Phase 7: コーディングタスク用データセット作成
        elif phase_name == 'phase7_coding_training_data':
            output_dir = Path(self.phase7_config.get('output_dir', 'D:/webdataset/coding_training_data'))
            training_files = list(output_dir.glob('coding_training_*.jsonl'))
            if training_files:
                latest_file = max(training_files, key=lambda p: p.stat().st_mtime)
                if latest_file.stat().st_size > 0:
                    logger.info(f"[VERIFY] Phase 7: Coding training data found: {latest_file}")
                    return True
            return False
        
        # Phase 8: コーディング特化再学習
        elif phase_name == 'phase8_coding_retraining':
            config_path = PROJECT_ROOT / self.phase8_config.get('config_path', 'configs/coding_focused_retraining_config.yaml')
            if config_path.exists():
                try:
                    import yaml
                    with open(config_path, 'r', encoding='utf-8') as f:
                        retraining_config = yaml.safe_load(f)
                    final_model_dir = Path(retraining_config.get('output', {}).get('final_model_dir', 'D:/webdataset/models/coding_focused_retraining'))
                    
                    # 最終モデルディレクトリの存在を確認
                    if final_model_dir.exists():
                        model_files = list(final_model_dir.glob('*.pt')) + list(final_model_dir.glob('*.safetensors'))
                        if model_files:
                            logger.info(f"[VERIFY] Phase 8: Coding retraining model found: {final_model_dir}")
                            return True
                except Exception as e:
                    logger.warning(f"[VERIFY] Phase 8: Failed to verify model (fallback to status check): {e}")
                    return True  # エラー時はステータスのみで判定
            return False
        
        # Phase 9: ドキュメンテーション収集
        elif phase_name == 'phase9_documentation_scraping':
            phase9_config = self.phase9_config
            if not phase9_config.get('enabled', True):
                logger.info("[VERIFY] Phase 9: Disabled in config")
                return True
            
            output_dir = Path(phase9_config.get('output_dir', 'D:/webdataset/processed/documentation'))
            doc_files = list(output_dir.glob('documentation_*.jsonl'))
            if doc_files:
                latest_file = max(doc_files, key=lambda p: p.stat().st_mtime)
                if latest_file.stat().st_size > 0:
                    logger.info(f"[VERIFY] Phase 9: Documentation scraping results found: {latest_file}")
                    return True
            return False
        
        # Phase 10: 統合AIエージェント基盤の構築
        elif phase_name == 'phase10_unified_agent_base':
            phase10_config = self.phase10_config
            if not phase10_config.get('enabled', True):
                logger.info("[VERIFY] Phase 10: Disabled in config")
                return True
            
            # 結果ファイルの存在確認
            result_files = list(self.checkpoint_dir.glob('phase10_results_*.json'))
            if result_files:
                logger.info(f"[VERIFY] Phase 10: Result files found: {len(result_files)} files")
                return True
            else:
                logger.warning(f"[VERIFY] Phase 10: No result files found in checkpoint directory")
                return False
        
        # Phase 11: 検知用NSFWデータセット収集
        elif phase_name == 'phase11_nsfw_detection_dataset':
            phase11_config = self.phase11_config
            if not phase11_config.get('enabled', True):
                logger.info("[VERIFY] Phase 11: Disabled in config")
                return True
            
            # 出力ディレクトリの存在確認
            output_dir = Path(phase11_config.get('output_dir', 'D:/webdataset/nsfw_detection_dataset'))
            train_file = output_dir / "nsfw_detection_train.jsonl"
            val_file = output_dir / "nsfw_detection_val.jsonl"
            
            if train_file.exists() and val_file.exists():
                if train_file.stat().st_size > 0 and val_file.stat().st_size > 0:
                    logger.info(f"[VERIFY] Phase 11: Dataset files found: {train_file}, {val_file}")
                    return True
                else:
                    logger.warning(f"[VERIFY] Phase 11: Dataset files exist but are empty")
                    return False
            else:
                logger.warning(f"[VERIFY] Phase 11: Dataset files not found")
                return False
        
        # その他のフェーズはステータスのみで判定
        else:
            logger.debug(f"[VERIFY] Phase {phase_name}: Status check only (status: {status})")
            return status == 'completed'
    
    def phase1_parallel_scraping(self) -> bool:
        """
        Phase 1: SO8T/thinkingモデル統制並列DeepResearch Webスクレイピング
        
        Returns:
            success: 成功フラグ
        """
        logger.info("="*80)
        logger.info("PHASE 1: SO8T Thinking Controlled Parallel DeepResearch Web Scraping")
        logger.info("="*80)
        
        self.current_phase = "phase1_parallel_scraping"
        
        # チェックポイントから復旧
        checkpoint = self._load_checkpoint()
        if checkpoint:
            phase_progress = checkpoint.get('phase_progress', {}).get('phase1_parallel_scraping', {})
            if phase_progress.get('status') == 'completed':
                # 目標サンプル数を確認
                target_samples = self.phase1_config.get('target_samples')
                if target_samples:
                    # 現在のサンプル数を確認
                    current_samples = self._count_current_samples()
                    if current_samples < target_samples:
                        logger.info(f"[RESUME] Current samples ({current_samples}) < target ({target_samples}), restarting Phase 1")
                        # ステータスをリセットして再実行
                        phase_progress['status'] = 'running'
                        # チェックポイントを更新
                        self.phase_progress['phase1_parallel_scraping'] = phase_progress
                        self._save_checkpoint()
                    else:
                        logger.info(f"[SKIP] Phase 1 already completed and target samples reached ({current_samples}/{target_samples})")
                        return True
                else:
                    logger.info("[SKIP] Phase 1 already completed")
                    return True
        
        if not self.phase1_config.get('enabled', True):
            logger.info("[SKIP] Phase 1 disabled in config")
            return True
        
        # SO8T統制ChromeDev並列ブラウザCUDA分散処理オプションをチェック
        use_so8t_chromedev_daemon = self.phase1_config.get('use_so8t_chromedev_daemon', False)
        so8t_chromedev_daemon_config = self.phase1_config.get('so8t_chromedev_daemon', {})
        
        if use_so8t_chromedev_daemon and so8t_chromedev_daemon_config.get('enabled', False):
            # SO8T統制ChromeDev並列ブラウザCUDA分散処理を使用
            import asyncio
            return asyncio.run(self._phase1_so8t_chromedev_daemon_scraping())
        
        # 並列タブスクレイピングオプションをチェック
        use_parallel_tabs = self.phase1_config.get('use_parallel_tabs', False)
        
        if use_parallel_tabs:
            # 並列タブスクレイピングスクリプトを実行
            script_path = PROJECT_ROOT / "scripts" / "data" / "cursor_parallel_tab_scraping.py"
            
            if not script_path.exists():
                logger.error(f"Parallel tab scraper not found: {script_path}")
                return False
            
            base_output = str(self.phase1_config.get('base_output', 'D:/webdataset/processed'))
            num_tabs = self.phase1_config.get('num_tabs', 10)
            pages_per_tab = self.phase1_config.get('pages_per_tab', 10)
            
            # MCP Chrome DevTools設定を取得
            use_mcp_chrome_devtools = self.phase1_config.get('use_mcp_chrome_devtools', False)
            mcp_server_config = self.phase1_config.get('mcp_server', {})
            
            cmd = [
                sys.executable,
                str(script_path),
                "--output", base_output,
                "--num-tabs", str(num_tabs),
                "--pages-per-tab", str(pages_per_tab),
                "--use-cursor-browser",
                "--remote-debugging-port", str(self.phase1_config.get('base_port', 9222)),
                "--delay-per-action", str(self.phase1_config.get('restart_delay', 60.0) / 40.0)  # アクション間の遅延に変換
            ]
            
            # MCP Chrome DevToolsを使用する場合
            if use_mcp_chrome_devtools and mcp_server_config.get('enabled', True):
                cmd.append("--use-mcp-chrome-devtools")
                logger.info("[MCP] MCP Chrome DevTools enabled for parallel tab scraping")
            
            logger.info(f"[PARALLEL_TABS] Starting parallel tab scraping: {num_tabs} tabs, {pages_per_tab} pages per tab")
            if use_mcp_chrome_devtools:
                logger.info("[MCP] Using MCP Chrome DevTools for browser control")
        else:
            # SO8T/thinkingモデル統制スクレイピングスクリプトを実行
            script_path = PROJECT_ROOT / "scripts" / "data" / "so8t_thinking_controlled_scraping.py"
            
            if not script_path.exists():
                logger.warning(f"SO8T thinking controlled scraper not found: {script_path}")
                logger.info("Falling back to parallel_pipeline_manager.py")
                script_path = PROJECT_ROOT / "scripts" / "data" / "parallel_pipeline_manager.py"
            
            if not script_path.exists():
                logger.error(f"Script not found: {script_path}")
                return False
            
            # バックグラウンドで実行（daemon mode）
            base_output = str(self.phase1_config.get('base_output', 'D:/webdataset/processed'))
        if '#' in base_output:
            logger.error(f"[ERROR] Invalid character '#' in base_output path: {base_output}")
            logger.error("[ERROR] Please remove '#' from the path")
            self.phase_progress['phase1_parallel_scraping'] = {
                'status': 'failed',
                'error': f"Invalid character '#' in base_output path"
            }
            self._save_checkpoint()
            return False
        
        # 並列タブスクレイピングの場合は既にコマンドが構築されている
        if not use_parallel_tabs:
            # SO8T/thinkingモデル統制スクレイピングの場合は専用コマンドを使用
            if "so8t_thinking_controlled_scraping.py" in str(script_path):
                cmd = [
                sys.executable,
                str(script_path),
                "--output", base_output,
                "--num-browsers", str(self.phase1_config.get('num_instances', 10)),
                "--remote-debugging-port", str(self.phase1_config.get('base_port', 9222)),
                "--use-cursor-browser",
                "--resume"
            ]
            
            # データ量拡大パラメータを追加
            if self.phase1_config.get('target_samples'):
                cmd.extend(["--target-samples", str(self.phase1_config.get('target_samples'))])
            if self.phase1_config.get('min_samples_per_keyword'):
                cmd.extend(["--min-samples-per-keyword", str(self.phase1_config.get('min_samples_per_keyword'))])
            if self.phase1_config.get('max_samples_per_keyword'):
                cmd.extend(["--max-samples-per-keyword", str(self.phase1_config.get('max_samples_per_keyword'))])
        else:
            # 既存のparallel_pipeline_manager.pyを使用
            cmd = [
                sys.executable,
                str(script_path),
                "--run",
                "--daemon",
                "--num-instances", str(self.phase1_config.get('num_instances', 10)),
                "--base-output", base_output,
                "--base-port", str(self.phase1_config.get('base_port', 9222)),
                "--auto-restart",
                "--restart-delay", str(self.phase1_config.get('restart_delay', 60.0)),
                "--max-memory-gb", str(self.phase1_config.get('max_memory_gb', 8.0)),
                "--max-cpu-percent", str(self.phase1_config.get('max_cpu_percent', 80.0))
            ]
        
        # コマンド内に#が含まれていないか確認
        cmd_str = ' '.join(cmd)
        if '#' in cmd_str:
            logger.error(f"[ERROR] Invalid character '#' found in command: {cmd_str}")
            logger.error("[ERROR] Please check configuration files for '#' characters")
            self.phase_progress['phase1_parallel_scraping'] = {
                'status': 'failed',
                'error': f"Invalid character '#' in command"
            }
            self._save_checkpoint()
            return False
        
        try:
            logger.info(f"Starting SO8T thinking controlled parallel scraping pipeline...")
            logger.debug(f"Command: {cmd_str}")
            # バックグラウンドで実行（非同期）
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8'
            )
            
            # プロセスIDを記録
            self.phase_progress['phase1_parallel_scraping'] = {
                'status': 'running',
                'process_id': process.pid,
                'started_at': datetime.now().isoformat(),
                'script': str(script_path.name)
            }
            self._save_checkpoint()
            
            logger.info(f"[OK] Phase 1 started (PID: {process.pid})")
            logger.info("[INFO] Phase 1 runs in background with SO8T thinking control. Proceeding to Phase 2...")
            
            # バックグラウンド実行のため、すぐに成功として返す
            self.phase_progress['phase1_parallel_scraping'] = {
                'status': 'completed',
                'process_id': process.pid,
                'started_at': datetime.now().isoformat(),
                'note': 'Running in background with SO8T thinking control'
            }
            self._save_checkpoint()
            
            AudioNotifier.play_notification()
            
            return True
            
        except subprocess.SubprocessError as e:
            logger.error(f"[ERROR] Phase 1 failed (SubprocessError): {e}")
            logger.error(f"[ERROR] Command: {cmd_str}")
            import traceback
            traceback.print_exc()
            self.phase_progress['phase1_parallel_scraping'] = {
                'status': 'failed',
                'error': f"SubprocessError: {str(e)}"
            }
            self._save_checkpoint()
            return False
        except Exception as e:
            logger.error(f"[ERROR] Phase 1 failed: {e}")
            logger.error(f"[ERROR] Command: {cmd_str}")
            import traceback
            traceback.print_exc()
            self.phase_progress['phase1_parallel_scraping'] = {
                'status': 'failed',
                'error': str(e)
            }
            self._save_checkpoint()
            return False
    
    async def _phase1_so8t_chromedev_daemon_scraping(self) -> bool:
        """
        Phase 1: SO8T統制ChromeDev並列ブラウザCUDA分散処理スクレイピング
        
        Returns:
            success: 成功フラグ
        """
        logger.info("="*80)
        logger.info("PHASE 1: SO8T Controlled ChromeDev Parallel Browser CUDA Distributed Scraping")
        logger.info("="*80)
        
        try:
            # SO8TChromeDevDaemonManagerをインポート
            from scripts.data.so8t_chromedev_daemon_manager import SO8TChromeDevDaemonManager
            
            # 設定を取得
            so8t_chromedev_daemon_config = self.phase1_config.get('so8t_chromedev_daemon', {})
            total_parallel_tasks = so8t_chromedev_daemon_config.get('total_parallel_tasks', 200)
            base_port = so8t_chromedev_daemon_config.get('base_port', self.phase1_config.get('base_port', 9222))
            config_path = PROJECT_ROOT / so8t_chromedev_daemon_config.get('config_path', 'configs/so8t_chromedev_daemon_config.yaml')
            
            # num_browsersとnum_tabsを動的に計算（total_parallel_tasksから）
            # デフォルト: 20ブラウザ × 10タブ = 200タブ
            if 'num_browsers' in so8t_chromedev_daemon_config and 'num_tabs' in so8t_chromedev_daemon_config:
                num_browsers = so8t_chromedev_daemon_config.get('num_browsers', 20)
                num_tabs = so8t_chromedev_daemon_config.get('num_tabs', 10)
            else:
                # total_parallel_tasksから動的に計算
                # 最適な組み合わせを計算（例: 200 = 20 × 10）
                num_tabs = so8t_chromedev_daemon_config.get('num_tabs', 10)
                num_browsers = max(1, total_parallel_tasks // num_tabs)
                # 余りがある場合は最後のブラウザに追加
                if total_parallel_tasks % num_tabs != 0:
                    num_browsers += 1
            
            logger.info(f"[SO8T_CHROMEDEV] Initializing SO8T ChromeDev Daemon Manager...")
            logger.info(f"[SO8T_CHROMEDEV] Total parallel tasks: {total_parallel_tasks}")
            logger.info(f"[SO8T_CHROMEDEV] Number of browsers: {num_browsers}")
            logger.info(f"[SO8T_CHROMEDEV] Number of tabs per browser: {num_tabs}")
            logger.info(f"[SO8T_CHROMEDEV] Total tabs: {num_browsers * num_tabs}")
            logger.info(f"[SO8T_CHROMEDEV] Base port: {base_port}")
            
            # SO8TChromeDevDaemonManagerを初期化
            manager = SO8TChromeDevDaemonManager(
                num_browsers=num_browsers,
                num_tabs=num_tabs,
                base_port=base_port,
                config_path=config_path
            )
            
            # すべてのコンポーネントを起動
            logger.info("[SO8T_CHROMEDEV] Starting all components...")
            success = await manager.start_all()
            
            if not success:
                logger.error("[SO8T_CHROMEDEV] Failed to start all components")
                self.phase_progress['phase1_parallel_scraping'] = {
                    'status': 'failed',
                    'error': 'Failed to start SO8T ChromeDev Daemon Manager'
                }
                self._save_checkpoint()
                return False
            
            # URLリストを生成
            logger.info("[SO8T_CHROMEDEV] Generating URL list...")
            urls = self._generate_scraping_urls()
            
            # キーワードを生成（キーワードキューからも読み込み）
            keywords = self._generate_scraping_keywords()
            
            # キーワードキューからキーワードを追加（協調動作が有効な場合）
            coordination_config = self.phase1_config.get('keyword_coordination', {})
            if coordination_config.get('enabled', False):
                try:
                    from scripts.utils.keyword_coordinator import KeywordCoordinator
                    keyword_coordinator = KeywordCoordinator(
                        keyword_queue_file=coordination_config.get('keyword_queue_file', 'D:/webdataset/checkpoints/keyword_queue.json')
                    )
                    # キューからpending状態のキーワードを取得
                    pending_keywords = keyword_coordinator.get_all_keywords(status_filter=None)
                    queue_keywords = [kw.get('keyword') for kw in pending_keywords if kw.get('status') == 'pending']
                    if queue_keywords:
                        keywords.extend(queue_keywords)
                        logger.info(f"[SO8T_CHROMEDEV] Added {len(queue_keywords)} keywords from queue")
                except Exception as e:
                    logger.warning(f"[SO8T_CHROMEDEV] Failed to load keywords from queue: {e}")
            
            logger.info(f"[SO8T_CHROMEDEV] Generated {len(urls)} URLs and {len(keywords)} keywords")
            
            # SO8T統制でスクレイピング
            logger.info("[SO8T_CHROMEDEV] Starting SO8T-controlled scraping...")
            results = await manager.scrape_with_so8t_control(urls, keywords)
            
            # 結果を保存
            base_output = Path(self.phase1_config.get('base_output', 'D:/webdataset/processed'))
            output_file = base_output / f"so8t_chromedev_scraped_{self.session_id}.jsonl"
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            import json
            with open(output_file, 'w', encoding='utf-8') as f:
                for result in results:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
            
            logger.info(f"[OK] Saved {len(results)} samples to {output_file}")
            
            # すべてのコンポーネントを停止
            logger.info("[SO8T_CHROMEDEV] Stopping all components...")
            await manager.stop_all()
            
            # 進捗を更新
            self.phase_progress['phase1_parallel_scraping'] = {
                'status': 'completed',
                'started_at': datetime.now().isoformat(),
                'completed_at': datetime.now().isoformat(),
                'samples_scraped': len(results),
                'output_file': str(output_file),
                'method': 'so8t_chromedev_daemon'
            }
            self._save_checkpoint()
            
            AudioNotifier.play_notification()
            
            return True
            
        except ImportError as e:
            logger.error(f"[SO8T_CHROMEDEV] Failed to import SO8TChromeDevDaemonManager: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.phase_progress['phase1_parallel_scraping'] = {
                'status': 'failed',
                'error': f"ImportError: {str(e)}"
            }
            self._save_checkpoint()
            return False
        except KeyboardInterrupt:
            logger.warning("[SO8T_CHROMEDEV] Interrupted by user")
            try:
                if 'manager' in locals():
                    await manager.stop_all()
            except Exception:
                pass
            self.phase_progress['phase1_parallel_scraping'] = {
                'status': 'interrupted',
                'error': 'KeyboardInterrupt'
            }
            self._save_checkpoint()
            return False
        except Exception as e:
            logger.error(f"[SO8T_CHROMEDEV] Phase 1 failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # エラー時もコンポーネントを停止
            try:
                if 'manager' in locals():
                    await manager.stop_all()
            except Exception as stop_error:
                logger.warning(f"[SO8T_CHROMEDEV] Failed to stop components: {stop_error}")
            self.phase_progress['phase1_parallel_scraping'] = {
                'status': 'failed',
                'error': str(e)
            }
            self._save_checkpoint()
            return False
    
    def _generate_scraping_urls(self) -> List[str]:
        """
        スクレイピング用URLリストを生成
        
        Returns:
            urls: URLリスト
        """
        urls = []
        
        # 設定から百科事典ソースの設定を取得
        encyclopedia_sources = self.phase1_config.get('encyclopedia_sources', {})
        
        # ウィキペディア（日本語）
        if encyclopedia_sources.get('wikipedia_ja', True):
            urls.extend([
                "https://ja.wikipedia.org/wiki/メインページ",
                "https://ja.wikipedia.org/wiki/Category:コンピュータ",
                "https://ja.wikipedia.org/wiki/Category:プログラミング言語",
                "https://ja.wikipedia.org/wiki/Category:ソフトウェア",
                "https://ja.wikipedia.org/wiki/Category:軍事",
                "https://ja.wikipedia.org/wiki/Category:航空宇宙",
                "https://ja.wikipedia.org/wiki/Category:インフラ",
                "https://ja.wikipedia.org/wiki/Category:日本企業",
            ])
        
        # ウィキペディア（英語）
        if encyclopedia_sources.get('wikipedia_en', True):
            urls.extend([
                "https://en.wikipedia.org/wiki/Main_Page",
                "https://en.wikipedia.org/wiki/Category:Computer_science",
                "https://en.wikipedia.org/wiki/Category:Programming_languages",
                "https://en.wikipedia.org/wiki/Category:Software",
                "https://en.wikipedia.org/wiki/Category:Military",
                "https://en.wikipedia.org/wiki/Category:Aerospace",
                "https://en.wikipedia.org/wiki/Category:Infrastructure",
            ])
        
        # コトバンク
        if encyclopedia_sources.get('kotobank', True):
            urls.extend([
                "https://kotobank.jp/",
                "https://kotobank.jp/word/プログラミング",
                "https://kotobank.jp/word/コンピュータ",
                "https://kotobank.jp/word/ソフトウェア",
                "https://kotobank.jp/word/軍事",
                "https://kotobank.jp/word/航空宇宙",
                "https://kotobank.jp/word/インフラ",
            ])
        
        # ブリタニカ国際大百科事典
        if encyclopedia_sources.get('britannica', True):
            urls.extend([
                "https://www.britannica.com/",
                "https://www.britannica.com/technology/computer",
                "https://www.britannica.com/technology/software",
                "https://www.britannica.com/technology/programming-language",
                "https://www.britannica.com/topic/military",
                "https://www.britannica.com/topic/aerospace-industry",
                "https://www.britannica.com/topic/infrastructure",
            ])
        
        # 2025年最新の実用的なコーディング関連サイト（優先）
        coding_sources = self.phase1_config.get('coding_sources', {})
        if coding_sources.get('enabled', True):
            urls.extend([
                # GitHub（人気で有用なリポジトリ）
                "https://github.com/trending",
                "https://github.com/trending/python",
                "https://github.com/trending/rust",
                "https://github.com/trending/typescript",
                "https://github.com/trending/java",
                "https://github.com/trending/cpp",
                "https://github.com/trending/swift",
                "https://github.com/trending/kotlin",
                "https://github.com/trending/csharp",
                "https://github.com/trending/php",
                "https://github.com/explore",
                # PyTorch
                "https://pytorch.org/",
                "https://pytorch.org/docs/stable/index.html",
                "https://pytorch.org/tutorials/",
                # TensorFlow
                "https://www.tensorflow.org/",
                "https://www.tensorflow.org/api_docs",
                "https://www.tensorflow.org/tutorials",
                # Stack Overflow（技術スタック別）
                "https://stackoverflow.com/questions/tagged/python",
                "https://stackoverflow.com/questions/tagged/rust",
                "https://stackoverflow.com/questions/tagged/typescript",
                "https://stackoverflow.com/questions/tagged/javascript",
                "https://stackoverflow.com/questions/tagged/java",
                "https://stackoverflow.com/questions/tagged/c%2b%2b",
                "https://stackoverflow.com/questions/tagged/c",
                "https://stackoverflow.com/questions/tagged/swift",
                "https://stackoverflow.com/questions/tagged/kotlin",
                "https://stackoverflow.com/questions/tagged/c%23",
                "https://stackoverflow.com/questions/tagged/unity3d",
                "https://stackoverflow.com/questions/tagged/php",
                # エンジニア向けサイト
                "https://qiita.com/",
                "https://zenn.dev/",
                "https://dev.to/",
                "https://medium.com/tag/programming",
                # 技術ドキュメントサイト
                "https://docs.python.org/",
                "https://developer.mozilla.org/",
                "https://react.dev/",
                "https://vuejs.org/",
                "https://angular.io/",
                "https://docs.microsoft.com/en-us/dotnet/",
                "https://docs.microsoft.com/en-us/cpp/",
                "https://developer.apple.com/swift/",
                "https://kotlinlang.org/docs/home.html",
                # コーディング学習サイト
                "https://www.freecodecamp.org/",
                "https://www.codecademy.com/",
                "https://leetcode.com/",
                "https://www.codewars.com/",
                # 技術ブログ
                "https://techcrunch.com/",
                "https://www.infoq.com/",
                "https://www.oreilly.com/",
                # Reddit（技術関連）
                "https://www.reddit.com/r/programming/",
                "https://www.reddit.com/r/Python/",
                "https://www.reddit.com/r/rust/",
                "https://www.reddit.com/r/typescript/",
                "https://www.reddit.com/r/java/",
                "https://www.reddit.com/r/cpp/",
                "https://www.reddit.com/r/swift/",
                "https://www.reddit.com/r/Kotlin/",
                "https://www.reddit.com/r/csharp/",
                "https://www.reddit.com/r/Unity3D/",
                "https://www.reddit.com/r/PHP/",
                # Hacker News
                "https://news.ycombinator.com/",
            ])
        
        # NSFW検知目的のサイト（検知目的のみ）
        nsfw_sources = self.phase1_config.get('nsfw_sources', {})
        if nsfw_sources.get('enabled', False) and nsfw_sources.get('detection_only', True):
            urls.extend([
                # 検知目的のみ（生成目的ではない）
                "https://www.fanza.co.jp/",
                "https://www.xvideos.com/video",
                "https://missav.ai/"
                "https://www.pornhub.com/video"
                "https://www.xhamster.com/"
                "https://www.pornhub.com/video"
                "https://www.xhamster.com/video"
            ])
        
        # 設定から追加URLを取得
        additional_urls = self.phase1_config.get('additional_urls', [])
        if additional_urls:
            urls.extend(additional_urls)
        
        return urls
    
    def _generate_scraping_keywords(self) -> List[str]:
        """
        スクレイピング用キーワードリストを生成
        
        Returns:
            keywords: キーワードリスト
        """
        keywords = []
        
        # 2025年最新の実用的なコーディング関連キーワード
        coding_keywords = self.phase1_config.get('coding_keywords', {})
        if coding_keywords.get('enabled', True):
            keywords.extend([
                # プログラミング言語
                "Python", "Rust", "TypeScript", "JavaScript", "Java", "C++", "C", 
                "Swift", "Kotlin", "C#", "C# Unity", "PHP",
                # フレームワーク・ライブラリ
                "PyTorch", "TensorFlow", "React", "Vue.js", "Angular", "Unity",
                # ベストプラクティス
                "best practices", "coding standards", "design patterns", "clean code",
                "SOLID principles", "DRY principle", "KISS principle",
                # 2025年最新技術
                "AI programming", "machine learning", "deep learning", "LLM",
                "CUDA programming", "GPU computing", "parallel processing",
                # ドメイン別
                "military software", "aerospace programming", "infrastructure code",
                "on-premises environment", "enterprise software",
                # GitHub関連
                "GitHub repository", "open source", "contribution", "pull request",
                # コーディングタスク
                "code generation", "refactoring", "debugging", "testing",
                "code review", "version control", "CI/CD",
            ])
        
        # ドメイン別知識キーワード
        domain_keywords = self.phase1_config.get('domain_keywords', {})
        if domain_keywords.get('enabled', True):
            keywords.extend([
                "軍事", "航空宇宙", "インフラ", "日本企業", "運輸",
                "military", "aerospace", "infrastructure","japaneseCompany","transport",
            ])
        
        # 設定から追加キーワードを取得
        additional_keywords = self.phase1_config.get('additional_keywords', [])
        if additional_keywords:
            keywords.extend(additional_keywords)
        
        return keywords
    
    def phase2_data_processing(self) -> bool:
        """
        Phase 2: SO8T全自動データ処理
        
        Returns:
            success: 成功フラグ
        """
        logger.info("="*80)
        logger.info("PHASE 2: SO8T Auto Data Processing")
        logger.info("="*80)
        
        self.current_phase = "phase2_data_processing"
        
        # チェックポイントから復旧
        checkpoint = self._load_checkpoint()
        if checkpoint and checkpoint.get('current_phase') == 'phase2_data_processing':
            phase_progress = checkpoint.get('phase_progress', {}).get('phase2_data_processing', {})
            if phase_progress.get('status') == 'completed':
                logger.info("[SKIP] Phase 2 already completed")
                return True
        
        if not self.phase2_config.get('enabled', True):
            logger.info("[SKIP] Phase 2 disabled in config")
            return True
        
        # so8t_auto_data_processing_pipeline.pyを実行
        script_path = PROJECT_ROOT / "scripts" / "pipelines" / "so8t_auto_data_processing_pipeline.py"
        config_path = PROJECT_ROOT / self.phase2_config.get('config', 'configs/so8t_auto_data_processing_config.yaml')
        
        if not script_path.exists():
            logger.error(f"Script not found: {script_path}")
            return False
        
        if not config_path.exists():
            logger.error(f"Config file not found: {config_path}")
            return False
        
        # パスに#が含まれている場合はエラー
        config_path_str = str(config_path)
        if '#' in config_path_str:
            logger.error(f"[ERROR] Invalid character '#' in config path: {config_path_str}")
            logger.error("[ERROR] Please remove '#' from the path")
            self.phase_progress['phase2_data_processing'] = {
                'status': 'failed',
                'error': f"Invalid character '#' in config path"
            }
            self._save_checkpoint()
            return False
        
        cmd = [
            sys.executable,
            str(script_path),
            "--config", config_path_str,
            "--resume"
        ]
        
        # コマンド内に#が含まれていないか確認
        cmd_str = ' '.join(cmd)
        if '#' in cmd_str:
            logger.error(f"[ERROR] Invalid character '#' found in command: {cmd_str}")
            logger.error("[ERROR] Please check configuration files for '#' characters")
            self.phase_progress['phase2_data_processing'] = {
                'status': 'failed',
                'error': f"Invalid character '#' in command"
            }
            self._save_checkpoint()
            return False
        
        try:
            logger.info(f"Starting SO8T data processing pipeline with quadruple reasoning and four-value classification...")
            logger.info("[SO8T] Phase 2 will execute:")
            logger.info("[SO8T]  1. Data cleansing")
            logger.info("[SO8T]  2. Incremental labeling with SO8T/thinking model")
            logger.info("[SO8T]  3. Quadruple reasoning classification")
            logger.info("[SO8T]  4. Four-value classification (ALLOW/ESCALATION/DENY/REFUSE)")
            logger.debug(f"Command: {cmd_str}")
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=False,
                text=True,
                timeout=self.phase2_config.get('timeout', 86400)  # 24時間タイムアウト
            )
            
            self.phase_progress['phase2_data_processing'] = {
                'status': 'completed',
                'completed_at': datetime.now().isoformat()
            }
            self._save_checkpoint()
            
            logger.info("[OK] Phase 2 completed (SO8T quadruple reasoning and four-value classification)")
            
            AudioNotifier.play_notification()
            
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"[ERROR] Phase 2 failed: {e}")
            return False
        except subprocess.TimeoutExpired:
            logger.error("[ERROR] Phase 2 timeout")
            return False
        except Exception as e:
            logger.error(f"[ERROR] Phase 2 failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def phase3_ab_test(self) -> bool:
        """
        Phase 3: SO8T完全統合A/Bテスト
        
        Returns:
            success: 成功フラグ
        """
        logger.info("="*80)
        logger.info("PHASE 3: SO8T Complete A/B Test")
        logger.info("="*80)
        
        self.current_phase = "phase3_ab_test"
        
        # チェックポイントから復旧
        checkpoint = self._load_checkpoint()
        if checkpoint and checkpoint.get('current_phase') == 'phase3_ab_test':
            phase_progress = checkpoint.get('phase_progress', {}).get('phase3_ab_test', {})
            if phase_progress.get('status') == 'completed':
                logger.info("[SKIP] Phase 3 already completed")
                return True
        
        if not self.phase3_config.get('enabled', True):
            logger.info("[SKIP] Phase 3 disabled in config")
            return True
        
        # データセット量チェック（動的閾値対応）
        min_samples_for_retraining = self._get_dynamic_threshold()
        current_samples = self._count_current_samples()
        
        if current_samples < min_samples_for_retraining:
            remaining_samples = min_samples_for_retraining - current_samples
            progress_percent = (current_samples / min_samples_for_retraining * 100) if min_samples_for_retraining > 0 else 0.0
            logger.info("="*80)
            logger.info("[SKIP] Phase 3: A/B Test Skipped (Insufficient Dataset)")
            logger.info("="*80)
            logger.info(f"Current samples: {current_samples:,}")
            logger.info(f"Required samples: {min_samples_for_retraining:,}")
            logger.info(f"Remaining samples needed: {remaining_samples:,}")
            logger.info(f"Progress: {progress_percent:.1f}%")
            logger.info("="*80)
            logger.info("A/B test will be executed after collecting enough samples for SO8T retraining")
            logger.info("="*80)
            self.phase_progress['phase3_ab_test'] = {
                'status': 'skipped',
                'reason': 'insufficient_samples',
                'current_samples': current_samples,
                'required_samples': min_samples_for_retraining,
                'remaining_samples': remaining_samples,
                'progress_percent': progress_percent,
                'skipped_at': datetime.now().isoformat()
            }
            self._save_checkpoint()
            return True  # スキップは成功として扱う
        
        # complete_so8t_ab_test_pipeline.pyを実行
        script_path = PROJECT_ROOT / "scripts" / "pipelines" / "complete_so8t_ab_test_pipeline.py"
        config_path = PROJECT_ROOT / self.phase3_config.get('config', 'configs/complete_so8t_ab_test_pipeline_config.yaml')
        
        if not script_path.exists():
            logger.error(f"Script not found: {script_path}")
            return False
        
        if not config_path.exists():
            logger.error(f"Config file not found: {config_path}")
            return False
        
        # パスに#が含まれている場合はエラー
        config_path_str = str(config_path)
        if '#' in config_path_str:
            logger.error(f"[ERROR] Invalid character '#' in config path: {config_path_str}")
            logger.error("[ERROR] Please remove '#' from the path")
            self.phase_progress['phase3_ab_test'] = {
                'status': 'failed',
                'error': f"Invalid character '#' in config path"
            }
            self._save_checkpoint()
            return False
        
        cmd = [
            sys.executable,
            str(script_path),
            "--config", config_path_str,
            "--resume"
        ]
        
        # コマンド内に#が含まれていないか確認
        cmd_str = ' '.join(cmd)
        if '#' in cmd_str:
            logger.error(f"[ERROR] Invalid character '#' found in command: {cmd_str}")
            logger.error("[ERROR] Please check configuration files for '#' characters")
            self.phase_progress['phase3_ab_test'] = {
                'status': 'failed',
                'error': f"Invalid character '#' in command"
            }
            self._save_checkpoint()
            return False
        
        try:
            logger.info(f"Starting SO8T complete A/B test pipeline with Ollama check...")
            logger.info("[A/B TEST] Phase 3 will execute:")
            logger.info("[A/B TEST]  1. Model A GGUF conversion (base SO8T model)")
            logger.info("[A/B TEST]  2. SO8T retraining (QLoRA/fine-tuning)")
            logger.info("[A/B TEST]  3. Model B GGUF conversion (retrained SO8T model)")
            logger.info("[A/B TEST]  4. Ollama import (both models)")
            logger.info("[A/B TEST]  5. A/B test execution via Ollama")
            logger.info("[A/B TEST]  6. Ollama check and validation")
            logger.info("[A/B TEST]  7. Visualization and report generation")
            logger.debug(f"Command: {cmd_str}")
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=False,
                text=True,
                timeout=self.phase3_config.get('timeout', 172800)  # 48時間タイムアウト
            )
            
            self.phase_progress['phase3_ab_test'] = {
                'status': 'completed',
                'completed_at': datetime.now().isoformat()
            }
            self._save_checkpoint()
            
            logger.info("[OK] Phase 3 completed (A/B test with Ollama check)")
            
            AudioNotifier.play_notification()
            
            return True
            
        except subprocess.CalledProcessError as e:
            error_msg = str(e)
            # 量子化タイプエラーの場合は警告のみで続行（オプションフェーズのため）
            if 'q4_k_m' in error_msg or 'invalid choice' in error_msg or 'quantization' in error_msg.lower():
                logger.warning(f"[WARNING] Phase 3 failed due to quantization type error: {error_msg}")
                logger.warning("[WARNING] Phase 3 is optional, continuing pipeline...")
                self.phase_progress['phase3_ab_test'] = {
                    'status': 'skipped',
                    'error': error_msg,
                    'skipped_at': datetime.now().isoformat(),
                    'reason': 'quantization_type_error'
                }
                self._save_checkpoint()
                return True  # オプションフェーズのため、エラーでも続行
            else:
                logger.error(f"[ERROR] Phase 3 failed: {error_msg}")
                self.phase_progress['phase3_ab_test'] = {
                    'status': 'failed',
                    'error': error_msg
                }
                self._save_checkpoint()
                # Phase 3はオプションフェーズのため、エラーでも続行
                if not self.phase3_config.get('required', False):
                    logger.warning("[WARNING] Phase 3 is optional, continuing pipeline...")
                    return True
                return False
        except subprocess.TimeoutExpired:
            logger.error("[ERROR] Phase 3 timeout")
            # Phase 3はオプションフェーズのため、タイムアウトでも続行
            if not self.phase3_config.get('required', False):
                logger.warning("[WARNING] Phase 3 is optional, continuing pipeline...")
                self.phase_progress['phase3_ab_test'] = {
                    'status': 'skipped',
                    'error': 'timeout',
                    'skipped_at': datetime.now().isoformat()
                }
                self._save_checkpoint()
                return True
            return False
        except Exception as e:
            error_msg = str(e)
            logger.error(f"[ERROR] Phase 3 failed: {e}")
            import traceback
            traceback.print_exc()
            # Phase 3はオプションフェーズのため、エラーでも続行
            if not self.phase3_config.get('required', False):
                logger.warning("[WARNING] Phase 3 is optional, continuing pipeline...")
                self.phase_progress['phase3_ab_test'] = {
                    'status': 'skipped',
                    'error': error_msg,
                    'skipped_at': datetime.now().isoformat()
                }
                self._save_checkpoint()
                return True
            return False
    
    def phase4_github_scraping(self) -> bool:
        """Phase 4: GitHubリポジトリ検索"""
        try:
            logger.info("[PHASE 4] Starting GitHub repository scraping...")
            
            self.phase_progress['phase4_github_scraping'] = {
                'status': 'running',
                'started_at': datetime.now().isoformat()
            }
            self._save_checkpoint()
            
            # GitHubスクレイパースクリプトのパス
            script_path = PROJECT_ROOT / 'scripts' / 'data' / 'github_repository_scraper.py'
            
            if not script_path.exists():
                logger.error(f"[ERROR] GitHub scraper script not found: {script_path}")
                return False
            
            # 設定からパラメータを取得
            output_dir = Path(self.phase4_config.get('output_dir', 'D:/webdataset/processed/github'))
            github_token = self.phase4_config.get('github_token', None)
            queries = self.phase4_config.get('queries', ['best practices', 'tutorial', 'example'])
            languages = self.phase4_config.get('languages', None)
            
            # コマンドを構築
            cmd = [
                sys.executable,
                str(script_path),
                '--output', str(output_dir),
                '--max-repos', str(self.phase4_config.get('max_repos', 100)),
                '--min-stars', str(self.phase4_config.get('min_stars', 100)),
            ]
            
            if github_token:
                cmd.extend(['--github-token', github_token])
            
            cmd.extend(['--queries'] + queries)
            
            if languages:
                cmd.extend(['--languages'] + languages)
            
            # パスに#が含まれていないか確認
            cmd_str = ' '.join(cmd)
            if '#' in cmd_str:
                logger.error(f"[ERROR] Invalid character '#' found in command: {cmd_str}")
                return False
            
            logger.info(f"[PHASE 4] Executing: {' '.join(cmd)}")
            
            subprocess.run(
                cmd,
                check=True,
                capture_output=False,
                text=True,
                timeout=self.phase4_config.get('timeout', 3600)  # 1時間タイムアウト
            )
            
            self.phase_progress['phase4_github_scraping'] = {
                'status': 'completed',
                'completed_at': datetime.now().isoformat()
            }
            self._save_checkpoint()
            
            logger.info("[OK] Phase 4 completed")
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Phase 4 failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def phase5_engineer_sites(self) -> bool:
        """Phase 5: エンジニア向けサイトスクレイピング"""
        try:
            logger.info("[PHASE 5] Starting engineer sites scraping...")
            
            self.phase_progress['phase5_engineer_sites'] = {
                'status': 'running',
                'started_at': datetime.now().isoformat()
            }
            self._save_checkpoint()
            
            # エンジニアサイトスクレイパースクリプトのパス
            script_path = PROJECT_ROOT / 'scripts' / 'data' / 'engineer_site_scraper.py'
            
            if not script_path.exists():
                logger.error(f"[ERROR] Engineer site scraper script not found: {script_path}")
                return False
            
            # 設定からパラメータを取得
            output_dir = Path(self.phase5_config.get('output_dir', 'D:/webdataset/processed/engineer_sites'))
            queries = self.phase5_config.get('queries', ['Python', 'JavaScript', 'programming'])
            
            # コマンドを構築
            cmd = [
                sys.executable,
                str(script_path),
                '--output', str(output_dir),
                '--delay', str(self.phase5_config.get('delay', 2.0)),
                '--max-articles', str(self.phase5_config.get('max_articles', 100)),
            ]
            
            cmd.extend(['--queries'] + queries)
            
            # パスに#が含まれていないか確認
            cmd_str = ' '.join(cmd)
            if '#' in cmd_str:
                logger.error(f"[ERROR] Invalid character '#' found in command: {cmd_str}")
                return False
            
            logger.info(f"[PHASE 5] Executing: {' '.join(cmd)}")
            
            subprocess.run(
                cmd,
                check=True,
                capture_output=False,
                text=True,
                timeout=self.phase5_config.get('timeout', 7200)  # 2時間タイムアウト
            )
            
            self.phase_progress['phase5_engineer_sites'] = {
                'status': 'completed',
                'completed_at': datetime.now().isoformat()
            }
            self._save_checkpoint()
            
            logger.info("[OK] Phase 5 completed")
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Phase 5 failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def phase6_coding_extraction(self) -> bool:
        """Phase 6: コーディング関連データ抽出"""
        try:
            logger.info("[PHASE 6] Starting coding dataset extraction...")
            
            self.phase_progress['phase6_coding_extraction'] = {
                'status': 'running',
                'started_at': datetime.now().isoformat()
            }
            self._save_checkpoint()
            
            # コーディングデータ抽出スクリプトのパス
            script_path = PROJECT_ROOT / 'scripts' / 'pipelines' / 'extract_coding_dataset.py'
            
            if not script_path.exists():
                logger.error(f"[ERROR] Coding extraction script not found: {script_path}")
                return False
            
            # 設定からパラメータを取得
            input_dir = Path(self.phase6_config.get('input_dir', 'D:/webdataset/processed/four_class'))
            output_dir = Path(self.phase6_config.get('output_dir', 'D:/webdataset/coding_dataset'))
            
            # コマンドを構築
            cmd = [
                sys.executable,
                str(script_path),
                '--input', str(input_dir),
                '--output', str(output_dir),
                '--min-code-length', str(self.phase6_config.get('min_code_length', 10)),
                '--min-text-length', str(self.phase6_config.get('min_text_length', 50)),
            ]
            
            # パスに#が含まれていないか確認
            cmd_str = ' '.join(cmd)
            if '#' in cmd_str:
                logger.error(f"[ERROR] Invalid character '#' found in command: {cmd_str}")
                return False
            
            logger.info(f"[PHASE 6] Executing: {' '.join(cmd)}")
            
            subprocess.run(
                cmd,
                check=True,
                capture_output=False,
                text=True,
                timeout=self.phase6_config.get('timeout', 3600)  # 1時間タイムアウト
            )
            
            self.phase_progress['phase6_coding_extraction'] = {
                'status': 'completed',
                'completed_at': datetime.now().isoformat()
            }
            self._save_checkpoint()
            
            logger.info("[OK] Phase 6 completed")
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Phase 6 failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def phase7_coding_training_data(self) -> bool:
        """Phase 7: コーディングタスク用データセット作成"""
        try:
            logger.info("[PHASE 7] Starting coding training data preparation...")
            
            self.phase_progress['phase7_coding_training_data'] = {
                'status': 'running',
                'started_at': datetime.now().isoformat()
            }
            self._save_checkpoint()
            
            # コーディングトレーニングデータ準備スクリプトのパス
            script_path = PROJECT_ROOT / 'scripts' / 'pipelines' / 'prepare_coding_training_data.py'
            
            if not script_path.exists():
                logger.error(f"[ERROR] Coding training data script not found: {script_path}")
                return False
            
            # 設定からパラメータを取得
            input_dir = Path(self.phase7_config.get('input_dir', 'D:/webdataset/coding_dataset'))
            output_dir = Path(self.phase7_config.get('output_dir', 'D:/webdataset/coding_training_data'))
            
            # コマンドを構築
            cmd = [
                sys.executable,
                str(script_path),
                '--input', str(input_dir),
                '--output', str(output_dir),
            ]
            
            # パスに#が含まれていないか確認
            cmd_str = ' '.join(cmd)
            if '#' in cmd_str:
                logger.error(f"[ERROR] Invalid character '#' found in command: {cmd_str}")
                return False
            
            logger.info(f"[PHASE 7] Executing: {' '.join(cmd)}")
            
            subprocess.run(
                cmd,
                check=True,
                capture_output=False,
                text=True,
                timeout=self.phase7_config.get('timeout', 3600)  # 1時間タイムアウト
            )
            
            self.phase_progress['phase7_coding_training_data'] = {
                'status': 'completed',
                'completed_at': datetime.now().isoformat()
            }
            self._save_checkpoint()
            
            logger.info("[OK] Phase 7 completed")
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Phase 7 failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def phase8_coding_retraining(self) -> bool:
        """Phase 8: コーディング特化再学習"""
        try:
            logger.info("[PHASE 8] Starting coding-focused retraining...")
            
            self.phase_progress['phase8_coding_retraining'] = {
                'status': 'running',
                'started_at': datetime.now().isoformat()
            }
            self._save_checkpoint()
            
            # コーディング特化再学習パイプラインスクリプトのパス
            script_path = PROJECT_ROOT / 'scripts' / 'pipelines' / 'coding_focused_retraining_pipeline.py'
            
            if not script_path.exists():
                logger.error(f"[ERROR] Coding retraining pipeline script not found: {script_path}")
                return False
            
            # 設定からパラメータを取得（PROJECT_ROOTを使用して絶対パスに変換）
            config_path = PROJECT_ROOT / self.phase8_config.get('config_path', 'configs/coding_focused_retraining_config.yaml')
            
            if not config_path.exists():
                logger.error(f"[ERROR] Config file not found: {config_path}")
                return False
            
            # コマンドを構築
            cmd = [
                sys.executable,
                str(script_path),
                '--config', str(config_path),
            ]
            
            # パスに#が含まれていないか確認
            cmd_str = ' '.join(cmd)
            if '#' in cmd_str:
                logger.error(f"[ERROR] Invalid character '#' found in command: {cmd_str}")
                return False
            
            logger.info(f"[PHASE 8] Executing: {' '.join(cmd)}")
            
            subprocess.run(
                cmd,
                check=True,
                capture_output=False,
                text=True,
                timeout=self.phase8_config.get('timeout', 86400)  # 24時間タイムアウト
            )
            
            self.phase_progress['phase8_coding_retraining'] = {
                'status': 'completed',
                'completed_at': datetime.now().isoformat()
            }
            self._save_checkpoint()
            
            logger.info("[OK] Phase 8 completed")
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Phase 8 failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def phase9_documentation_scraping(self) -> bool:
        """Phase 9: ドキュメンテーション収集"""
        try:
            logger.info("[PHASE 9] Starting documentation scraping...")
            
            self.phase_progress['phase9_documentation_scraping'] = {
                'status': 'running',
                'started_at': datetime.now().isoformat()
            }
            self._save_checkpoint()
            
            # ドキュメンテーション収集スクリプトのパス
            script_path = PROJECT_ROOT / 'scripts' / 'data' / 'documentation_scraper.py'
            
            if not script_path.exists():
                logger.error(f"[ERROR] Documentation scraper script not found: {script_path}")
                return False
            
            # 設定からパラメータを取得
            output_dir = Path(self.phase9_config.get('output_dir', 'D:/webdataset/processed/documentation'))
            github_repos = self.phase9_config.get('github_repos', [])
            api_urls = self.phase9_config.get('api_urls', [])
            blog_urls = self.phase9_config.get('blog_urls', [])
            delay = self.phase9_config.get('delay', 2.0)
            max_docs = self.phase9_config.get('max_docs', 100)
            
            # コマンドを構築
            cmd = [
                sys.executable,
                str(script_path),
                '--output', str(output_dir),
                '--delay', str(delay),
                '--max-docs', str(max_docs)
            ]
            
            # GitHubリポジトリが指定されている場合
            if github_repos:
                cmd.extend(['--github-repos'] + github_repos)
            
            # API URLが指定されている場合
            if api_urls:
                cmd.extend(['--api-urls'] + api_urls)
            
            # ブログURLが指定されている場合
            if blog_urls:
                cmd.extend(['--blog-urls'] + blog_urls)
            
            # パスに#が含まれていないか確認
            cmd_str = ' '.join(cmd)
            if '#' in cmd_str:
                logger.error(f"[ERROR] Invalid character '#' found in command: {cmd_str}")
                return False
            
            logger.info(f"[PHASE 9] Executing: {' '.join(cmd)}")
            
            subprocess.run(
                cmd,
                check=True,
                capture_output=False,
                text=True,
                timeout=self.phase9_config.get('timeout', 7200)  # 2時間タイムアウト
            )
            
            self.phase_progress['phase9_documentation_scraping'] = {
                'status': 'completed',
                'completed_at': datetime.now().isoformat()
            }
            self._save_checkpoint()
            
            logger.info("[OK] Phase 9 completed")
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Phase 9 failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def phase10_unified_agent_base(self) -> bool:
        """Phase 10: 統合AIエージェント基盤の構築"""
        try:
            logger.info("[PHASE 10] Starting unified AI agent base construction...")
            
            self.phase_progress['phase10_unified_agent_base'] = {
                'status': 'running',
                'started_at': datetime.now().isoformat()
            }
            self._save_checkpoint()
            
            # 統合推論パイプラインスクリプトのパス
            script_path = PROJECT_ROOT / 'scripts' / 'agents' / 'integrated_reasoning_pipeline.py'
            
            if not script_path.exists():
                logger.error(f"[ERROR] Integrated reasoning pipeline script not found: {script_path}")
                return False
            
            # 設定からパラメータを取得
            model_path = self.phase10_config.get('model_path')
            knowledge_base_path = self.phase10_config.get('knowledge_base_path', 'database/so8t_memory.db')
            rag_store_path = self.phase10_config.get('rag_store_path', 'D:/webdataset/vector_stores')
            coding_data_path = self.phase10_config.get('coding_data_path', 'D:/webdataset/processed/coding')
            science_data_path = self.phase10_config.get('science_data_path', 'D:/webdataset/processed/science')
            test_queries = self.phase10_config.get('test_queries', [])
            
            # テストクエリが指定されている場合、テストを実行
            if test_queries:
                logger.info(f"[PHASE 10] Running test queries: {len(test_queries)} queries")
                
                # テストクエリファイルを作成
                test_queries_file = self.checkpoint_dir / f"test_queries_{self.session_id}.txt"
                with open(test_queries_file, 'w', encoding='utf-8') as f:
                    for query in test_queries:
                        f.write(f"{query}\n")
                
                # コマンドを構築
                cmd = [
                    sys.executable,
                    str(script_path),
                    '--queries-file', str(test_queries_file),
                    '--user-id', 'pipeline_test'
                ]
                
                if model_path:
                    cmd.extend(['--model-path', model_path])
                if knowledge_base_path:
                    cmd.extend(['--knowledge-base', knowledge_base_path])
                if rag_store_path:
                    cmd.extend(['--rag-store', rag_store_path])
                if coding_data_path:
                    cmd.extend(['--coding-data', coding_data_path])
                if science_data_path:
                    cmd.extend(['--science-data', science_data_path])
                
                # 出力ファイル
                output_file = self.checkpoint_dir / f"phase10_results_{self.session_id}.json"
                cmd.extend(['--output', str(output_file)])
                
                # パスに#が含まれていないか確認
                cmd_str = ' '.join(cmd)
                if '#' in cmd_str:
                    logger.error(f"[ERROR] Invalid character '#' found in command: {cmd_str}")
                    return False
                
                logger.info(f"[PHASE 10] Executing: {' '.join(cmd)}")
                
                result = subprocess.run(
                    cmd,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=self.phase10_config.get('timeout', 3600)  # 1時間タイムアウト
                )
                
                logger.info(f"[PHASE 10] Test execution completed")
                logger.debug(f"[PHASE 10] stdout: {result.stdout[:500]}")
                
                # 結果ファイルの存在確認
                if output_file.exists():
                    logger.info(f"[PHASE 10] Results saved to: {output_file}")
                else:
                    logger.warning(f"[PHASE 10] Results file not found: {output_file}")
            else:
                logger.info("[PHASE 10] No test queries specified, skipping test execution")
            
            self.phase_progress['phase10_unified_agent_base'] = {
                'status': 'completed',
                'completed_at': datetime.now().isoformat()
            }
            self._save_checkpoint()
            
            logger.info("[OK] Phase 10 completed")
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Phase 10 failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def phase11_nsfw_detection_dataset(self) -> bool:
        """Phase 11: 検知用NSFWデータセット収集"""
        try:
            logger.info("[PHASE 11] Starting NSFW detection dataset collection...")
            
            self.phase_progress['phase11_nsfw_detection_dataset'] = {
                'status': 'running',
                'started_at': datetime.now().isoformat()
            }
            self._save_checkpoint()
            
            # NSFW検知用データセット収集スクリプトのパス
            script_path = PROJECT_ROOT / 'scripts' / 'data' / 'collect_nsfw_detection_dataset.py'
            
            if not script_path.exists():
                logger.error(f"[ERROR] NSFW detection dataset collector script not found: {script_path}")
                return False
            
            # 設定からパラメータを取得
            input_dir = Path(self.phase11_config.get('input_dir', 'D:/webdataset/processed/four_class'))
            output_dir = Path(self.phase11_config.get('output_dir', 'D:/webdataset/nsfw_detection_dataset'))
            nsfw_classifier_path = self.phase11_config.get('nsfw_classifier_path')
            max_samples = self.phase11_config.get('max_samples', 50000)
            use_multimodal = self.phase11_config.get('use_multimodal', True)
            include_synthetic = self.phase11_config.get('include_synthetic', True)
            synthetic_samples = self.phase11_config.get('synthetic_samples', 1000)
            include_metadata = self.phase11_config.get('include_metadata', True)
            
            # コマンドを構築
            cmd = [
                sys.executable,
                str(script_path),
                '--output', str(output_dir),
                '--max-samples', str(max_samples)
            ]
            
            if input_dir.exists():
                cmd.extend(['--input', str(input_dir)])
            
            if nsfw_classifier_path:
                cmd.extend(['--nsfw-classifier', str(nsfw_classifier_path)])
            
            if use_multimodal:
                cmd.append('--use-multimodal')
            
            if include_synthetic:
                cmd.append('--include-synthetic')
                cmd.extend(['--synthetic-samples', str(synthetic_samples)])
            
            if include_metadata:
                cmd.append('--include-metadata')
            
            # パスに#が含まれていないか確認
            cmd_str = ' '.join(cmd)
            if '#' in cmd_str:
                logger.error(f"[ERROR] Invalid character '#' found in command: {cmd_str}")
                return False
            
            logger.info(f"[PHASE 11] Executing: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=self.phase11_config.get('timeout', 7200)  # 2時間タイムアウト
            )
            
            logger.info(f"[PHASE 11] Dataset collection completed")
            logger.debug(f"[PHASE 11] stdout: {result.stdout[:500]}")
            
            # 出力ファイルの存在確認
            train_file = output_dir / "nsfw_detection_train.jsonl"
            val_file = output_dir / "nsfw_detection_val.jsonl"
            
            if train_file.exists() and val_file.exists():
                logger.info(f"[PHASE 11] Dataset files created successfully")
            else:
                logger.warning(f"[PHASE 11] Some dataset files not found")
            
            self.phase_progress['phase11_nsfw_detection_dataset'] = {
                'status': 'completed',
                'completed_at': datetime.now().isoformat()
            }
            self._save_checkpoint()
            
            logger.info("[OK] Phase 11 completed")
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Phase 11 failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def start_streamlit_dashboard(self):
        """Streamlitダッシュボードを起動"""
        try:
            dashboard_script = PROJECT_ROOT / "scripts" / "dashboard" / "unified_scraping_monitoring_dashboard.py"
            
            if not dashboard_script.exists():
                logger.warning(f"Dashboard script not found: {dashboard_script}")
                return False
            
            # 設定からホストとポートを取得
            dashboard_config = self.config.get('dashboard', {})
            host = dashboard_config.get('host', '0.0.0.0')
            port = dashboard_config.get('port', 8501)
            
            logger.info("[DASHBOARD] Starting Streamlit dashboard...")
            
            cmd = [
                sys.executable,
                "-m", "streamlit", "run",
                str(dashboard_script),
                "--server.port", str(port),
                "--server.address", host,
                "--server.headless", "true"
            ]
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8'
            )
            
            logger.info(f"[DASHBOARD] Dashboard started (PID: {process.pid})")
            logger.info(f"[DASHBOARD] Access at: http://localhost:{port}")
            logger.info(f"[DASHBOARD] External access: http://{host}:{port}")
            
            return True
            
        except Exception as e:
            logger.error(f"[DASHBOARD] Failed to start dashboard: {e}")
            return False
    
    def run_complete_pipeline(self, resume: bool = True):
        """
        完全パイプラインを実行
        
        Args:
            resume: チェックポイントから再開するか
        """
        logger.info("="*80)
        logger.info("Starting Unified Master Pipeline (SO8T Thinking Controlled)")
        logger.info("="*80)
        
        start_time = datetime.now()
        
        # Streamlitダッシュボードを起動（設定で有効な場合）
        dashboard_config = self.config.get('dashboard', {})
        if dashboard_config.get('enabled', True) and dashboard_config.get('auto_start', True):
            self.start_streamlit_dashboard()
        
        # システム起動時の遅延処理（60秒待機）
        delay_seconds = 60
        logger.info(f"Waiting {delay_seconds} seconds before starting pipeline (system startup delay)...")
        time.sleep(delay_seconds)
        
        # チェックポイントから復旧
        if resume:
            checkpoint = self._load_checkpoint()
            if checkpoint:
                logger.info(f"[RESUME] Resuming from checkpoint (Session: {self.session_id})")
                self.phase_progress = checkpoint.get('phase_progress', {})
        
        try:
            # Phase 1: 並列DeepResearch Webスクレイピング
            if not self._is_phase_completed('phase1_parallel_scraping'):
                logger.info("[EXECUTE] Phase 1: Starting parallel DeepResearch web scraping")
                success = self.phase1_parallel_scraping()
                if not success and self.phase1_config.get('required', True):
                    raise RuntimeError("Phase 1 (parallel scraping) failed")
            else:
                logger.info("[SKIP] Phase 1 already completed and verified")
            
            # Phase 2: SO8T全自動データ処理
            if not self._is_phase_completed('phase2_data_processing'):
                logger.info("[EXECUTE] Phase 2: Starting SO8T auto data processing")
                success = self.phase2_data_processing()
                if not success and self.phase2_config.get('required', True):
                    raise RuntimeError("Phase 2 (data processing) failed")
            else:
                logger.info("[SKIP] Phase 2 already completed and verified")
            
            # Phase 3: SO8T完全統合A/Bテスト
            if not self._is_phase_completed('phase3_ab_test'):
                logger.info("[EXECUTE] Phase 3: Starting SO8T complete A/B test")
                success = self.phase3_ab_test()
                if not success and self.phase3_config.get('required', True):
                    raise RuntimeError("Phase 3 (A/B test) failed")
            else:
                logger.info("[SKIP] Phase 3 already completed and verified")
            
            # Phase 4: GitHubリポジトリ検索
            if not self._is_phase_completed('phase4_github_scraping'):
                logger.info("[EXECUTE] Phase 4: Starting GitHub repository scraping")
                success = self.phase4_github_scraping()
                if not success and self.phase4_config.get('required', False):
                    logger.warning("[WARNING] Phase 4 (GitHub scraping) failed, continuing...")
            else:
                logger.info("[SKIP] Phase 4 already completed and verified")
            
            # Phase 5: エンジニア向けサイトスクレイピング
            if not self._is_phase_completed('phase5_engineer_sites'):
                logger.info("[EXECUTE] Phase 5: Starting engineer sites scraping")
                success = self.phase5_engineer_sites()
                if not success and self.phase5_config.get('required', False):
                    logger.warning("[WARNING] Phase 5 (engineer sites) failed, continuing...")
            else:
                logger.info("[SKIP] Phase 5 already completed and verified")
            
            # Phase 6: コーディング関連データ抽出
            if not self._is_phase_completed('phase6_coding_extraction'):
                logger.info("[EXECUTE] Phase 6: Starting coding dataset extraction")
                success = self.phase6_coding_extraction()
                if not success and self.phase6_config.get('required', False):
                    logger.warning("[WARNING] Phase 6 (coding extraction) failed, continuing...")
            else:
                logger.info("[SKIP] Phase 6 already completed and verified")
            
            # Phase 7: コーディングタスク用データセット作成
            if not self._is_phase_completed('phase7_coding_training_data'):
                logger.info("[EXECUTE] Phase 7: Starting coding training data preparation")
                success = self.phase7_coding_training_data()
                if not success and self.phase7_config.get('required', False):
                    logger.warning("[WARNING] Phase 7 (coding training data) failed, continuing...")
            else:
                logger.info("[SKIP] Phase 7 already completed and verified")
            
            # Phase 8: コーディング特化再学習
            if not self._is_phase_completed('phase8_coding_retraining'):
                logger.info("[EXECUTE] Phase 8: Starting coding-focused retraining")
                success = self.phase8_coding_retraining()
                if not success and self.phase8_config.get('required', False):
                    logger.warning("[WARNING] Phase 8 (coding retraining) failed, continuing...")
            else:
                logger.info("[SKIP] Phase 8 already completed and verified")
            
            # Phase 9: ドキュメンテーション収集
            if not self._is_phase_completed('phase9_documentation_scraping'):
                logger.info("[EXECUTE] Phase 9: Starting documentation scraping")
                success = self.phase9_documentation_scraping()
                if not success and self.phase9_config.get('required', False):
                    logger.warning("[WARNING] Phase 9 (documentation scraping) failed, continuing...")
            else:
                logger.info("[SKIP] Phase 9 already completed and verified")
            
            # Phase 10: 統合AIエージェント基盤の構築
            if not self._is_phase_completed('phase10_unified_agent_base'):
                logger.info("[EXECUTE] Phase 10: Starting unified AI agent base construction")
                success = self.phase10_unified_agent_base()
                if not success and self.phase10_config.get('required', False):
                    logger.warning("[WARNING] Phase 10 (unified agent base) failed, continuing...")
            else:
                logger.info("[SKIP] Phase 10 already completed and verified")
            
            # Phase 11: 検知用NSFWデータセット収集
            if not self._is_phase_completed('phase11_nsfw_detection_dataset'):
                logger.info("[EXECUTE] Phase 11: Starting NSFW detection dataset collection")
                success = self.phase11_nsfw_detection_dataset()
                if not success and self.phase11_config.get('required', False):
                    logger.warning("[WARNING] Phase 11 (NSFW detection dataset) failed, continuing...")
            else:
                logger.info("[SKIP] Phase 11 already completed and verified")
            
            total_time = (datetime.now() - start_time).total_seconds()
            
            logger.info("="*80)
            logger.info("Unified Master Pipeline Completed Successfully")
            logger.info("="*80)
            logger.info(f"Total processing time: {total_time:.2f} seconds ({total_time/3600:.2f} hours)")
            
        except KeyboardInterrupt:
            logger.warning("[INTERRUPT] Pipeline interrupted by user")
            self._save_checkpoint()
        except Exception as e:
            logger.error(f"[ERROR] Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            self._save_checkpoint()
            raise


def setup_auto_start():
    """Windowsタスクスケジューラに自動実行タスクを登録"""
    logger.info("="*80)
    logger.info("Setting up auto-start task")
    logger.info("="*80)
    
    # 管理者権限チェック
    if not check_admin_privileges():
        logger.error("[ERROR] Administrator privileges required")
        logger.error("Please run this script as administrator")
        logger.error("Right-click and select 'Run as administrator'")
        return False
    
    task_name = 'SO8T-UnifiedMasterPipeline-AutoStart'
    
    # タスクスケジューラ用バッチファイルのパス
    batch_file_path = PROJECT_ROOT / 'scripts' / 'pipelines' / 'unified_master_pipeline_autostart.bat'
    
    if not batch_file_path.exists():
        logger.error(f"Batch file not found: {batch_file_path}")
        logger.error("Please ensure unified_master_pipeline_autostart.bat exists")
        return False
    
    # パスに#が含まれている場合はエラー
    batch_file_path_str = str(batch_file_path)
    if '#' in batch_file_path_str:
        logger.error(f"[ERROR] Invalid character '#' in batch file path: {batch_file_path_str}")
        logger.error("[ERROR] Please remove '#' from the path")
        return False
    
    # タスクスケジューラから呼び出されるコマンド（バッチファイルを実行）
    task_command = f'"{batch_file_path_str}"'
    
    # コマンド内に#が含まれていないか確認
    if '#' in task_command:
        logger.error(f"[ERROR] Invalid character '#' found in task command: {task_command}")
        logger.error("[ERROR] Please check batch file path for '#' characters")
        return False
    
    # 既存のタスクを削除
    try:
        result = subprocess.run(
            ["schtasks", "/query", "/tn", task_name],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            logger.info(f"Removing existing task: {task_name}")
            subprocess.run(
                ["schtasks", "/delete", "/tn", task_name, "/f"],
                check=False
            )
    except Exception as e:
        logger.warning(f"Failed to check/remove existing task: {e}")
    
    # 新しいタスクを作成
    logger.info(f"Creating new task: {task_name}")
    
    create_cmd = [
        "schtasks", "/create",
        "/tn", task_name,
        "/tr", task_command,
        "/sc", "onstart",  # システム起動時
        "/rl", "highest",  # 最高権限
        "/f"
    ]
    
    try:
        result = subprocess.run(create_cmd, capture_output=True, text=True, check=True)
        logger.info("[OK] Task created successfully")
        logger.info(f"Task name: {task_name}")
        logger.info(f"Trigger: On system start")
        logger.info(f"Command: {task_command}")
        
        subprocess.run(["schtasks", "/query", "/tn", task_name, "/fo", "list", "/v"], check=False)
        
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"[ERROR] Failed to create task: {e}")
        logger.error(f"Output: {e.stdout}")
        logger.error(f"Error: {e.stderr}")
        if "アクセスが拒否されました" in e.stderr or "Access is denied" in e.stderr:
            logger.error("[ERROR] Access denied. Administrator privileges required.")
            logger.error("Please run this script as administrator:")
            logger.error("  Right-click and select 'Run as administrator'")
            logger.error("  Or run: py -3 scripts\\pipelines\\unified_master_pipeline.py --setup")
        return False


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="Unified Master Pipeline")
    parser.add_argument(
        '--config',
        type=Path,
        default=Path('configs/unified_master_pipeline_config.yaml'),
        help='Configuration file path'
    )
    parser.add_argument(
        '--setup',
        action='store_true',
        help='Setup Windows Task Scheduler auto-start task'
    )
    parser.add_argument(
        '--run',
        action='store_true',
        help='Run the pipeline (called by task scheduler)'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        default=True,
        help='Resume from checkpoint'
    )
    
    args = parser.parse_args()
    
    if args.setup:
        # タスクスケジューラ登録
        success = setup_auto_start()
        if success:
            logger.info("[OK] Auto-start task setup completed")
            return 0
        else:
            logger.error("[ERROR] Auto-start task setup failed")
            return 1
    
    elif args.run:
        # タスクスケジューラから呼び出された場合の処理
        if not args.config.exists():
            logger.error(f"Config file not found: {args.config}")
            return 1
        
        pipeline = UnifiedMasterPipeline(args.config)
        pipeline.run_complete_pipeline(resume=args.resume)
        
        return 0
    
    else:
        # 通常実行
        if not args.config.exists():
            logger.error(f"Config file not found: {args.config}")
            return 1
        
        pipeline = UnifiedMasterPipeline(args.config)
        pipeline.run_complete_pipeline(resume=args.resume)
        
        return 0


if __name__ == "__main__":
    sys.exit(main())


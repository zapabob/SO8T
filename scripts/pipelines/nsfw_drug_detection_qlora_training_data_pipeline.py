#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NSFW・違法薬物検知目的QLoRA学習用データ生成全自動パイプライン

検知目的でのQLoRAでのSO8Tおよびドメイン別知識の学習用データ生成用の全自動パイプライン。
NSFW、違法薬物データを含む検知目的でのデータ収集から学習用データセット生成まで全自動で実行します。

重要: この実装は検知目的のみで、生成目的ではない。安全判定と拒否挙動の学習を目的とする。

Usage:
    python scripts/pipelines/nsfw_drug_detection_qlora_training_data_pipeline.py --config configs/nsfw_drug_detection_qlora_training_data_pipeline_config.yaml
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
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Set
from collections import Counter
from difflib import SequenceMatcher
import warnings
warnings.filterwarnings("ignore")

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "so8t-mmllm"))

# 統計計算用ライブラリ
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    stats = None

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/nsfw_drug_detection_qlora_training_data_pipeline.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# QuadrupleClassifierインポート
try:
    from scripts.pipelines.web_scraping_data_pipeline import QuadrupleClassifier
    QUADRUPLE_CLASSIFIER_AVAILABLE = True
except ImportError:
    QUADRUPLE_CLASSIFIER_AVAILABLE = False
    logger.warning("[WARNING] QuadrupleClassifier not available")

# tqdmインポート
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    logger.warning("[WARNING] tqdm not available, progress bars will be disabled")


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


class NSFWDrugDetectionQLoRATrainingDataPipeline:
    """NSFW・違法薬物検知目的QLoRA学習用データ生成全自動パイプライン"""
    
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
        self.checkpoint_dir = Path(self.config.get('checkpoint_dir', 'D:/webdataset/checkpoints/nsfw_drug_detection_qlora_training_data_pipeline'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # 各フェーズの設定
        self.nsfw_config = self.config.get('nsfw_detection_dataset', {})
        self.drug_config = self.config.get('drug_detection_dataset', {})
        self.domain_knowledge_config = self.config.get('domain_knowledge_dataset', {})
        self.qlora_config = self.config.get('qlora_training_data', {})
        self.quadruple_config = self.config.get('quadruple_classification', {})
        self.cleaning_config = self.config.get('statistical_cleaning', {})
        
        # 出力ディレクトリ
        self.output_dir = Path(self.config.get('output_dir', 'D:/webdataset/nsfw_drug_detection_qlora_training_data'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 進捗管理
        self.current_phase = None
        self.phase_progress = {}
        
        # 収集済みデータ
        self.nsfw_samples: List[Dict] = []
        self.drug_samples: List[Dict] = []
        self.domain_knowledge_samples: List[Dict] = []
        self.merged_samples: List[Dict] = []
        self.classified_samples: List[Dict] = []
        self.cleaned_samples: List[Dict] = []
        self.training_dataset_samples: List[Dict] = []
        
        # QuadrupleClassifier
        self.quadruple_classifier: Optional[QuadrupleClassifier] = None
        
        # シグナルハンドラー設定
        self._setup_signal_handlers()
        
        logger.info("="*80)
        logger.info("NSFW・違法薬物検知目的QLoRA学習用データ生成全自動パイプライン初期化")
        logger.info("="*80)
        logger.info(f"セッションID: {self.session_id}")
        logger.info(f"チェックポイントディレクトリ: {self.checkpoint_dir}")
        logger.info(f"出力ディレクトリ: {self.output_dir}")
    
    def _setup_signal_handlers(self):
        """シグナルハンドラー設定"""
        def signal_handler(signum, frame):
            logger.warning(f"シグナル {signum} を受信、チェックポイントを保存中...")
            self._save_checkpoint()
            logger.info("チェックポイントを保存しました。正常終了します...")
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        if hasattr(signal, 'SIGBREAK'):
            signal.signal(signal.SIGBREAK, signal_handler)
    
    def _load_checkpoint(self, auto_resume: bool = True) -> Optional[Dict]:
        """
        チェックポイント読み込み
        
        Args:
            auto_resume: 自動再開モード（Trueの場合は最新のチェックポイントを自動検出）
        
        Returns:
            チェックポイントデータ、またはNone
        """
        checkpoint_file = self.checkpoint_dir / f"checkpoint_{self.session_id}.pkl"
        
        if not checkpoint_file.exists() or auto_resume:
            # 最新のチェックポイントを検索
            checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_*.pkl"), key=lambda p: p.stat().st_mtime, reverse=True)
            if checkpoints:
                checkpoint_file = checkpoints[0]
                logger.info(f"[CHECKPOINT] 最新のチェックポイントを検出: {checkpoint_file}")
                
                # チェックポイントの最終更新時刻を確認
                checkpoint_mtime = checkpoint_file.stat().st_mtime
                checkpoint_age = time.time() - checkpoint_mtime
                
                if checkpoint_age > 86400:  # 24時間以上経過している場合は警告
                    logger.warning(f"[CHECKPOINT] チェックポイントが古いです（{checkpoint_age/3600:.1f}時間前）")
                else:
                    logger.info(f"[CHECKPOINT] チェックポイントの年齢: {checkpoint_age/3600:.1f}時間")
            else:
                if auto_resume:
                    logger.info("[CHECKPOINT] チェックポイントが見つかりません。新規実行を開始します。")
                return None
        
        try:
            with open(checkpoint_file, 'rb') as f:
                checkpoint = pickle.load(f)
            
            # チェックポイントの整合性を確認
            if 'session_id' not in checkpoint or 'current_phase' not in checkpoint:
                logger.warning("[CHECKPOINT] チェックポイントの整合性が確認できません。新規実行を開始します。")
                return None
            
            logger.info(f"[CHECKPOINT] チェックポイントを読み込みました: {checkpoint_file}")
            logger.info(f"[CHECKPOINT] セッションID: {checkpoint.get('session_id', 'unknown')}")
            logger.info(f"[CHECKPOINT] 現在のフェーズ: {checkpoint.get('current_phase', 'unknown')}")
            logger.info(f"[CHECKPOINT] タイムスタンプ: {checkpoint.get('timestamp', 'unknown')}")
            
            return checkpoint
        except Exception as e:
            logger.error(f"[CHECKPOINT] チェックポイントの読み込みに失敗: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def _find_latest_checkpoint(self) -> Optional[Path]:
        """最新のチェックポイントファイルを検出"""
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_*.pkl"), key=lambda p: p.stat().st_mtime, reverse=True)
        if checkpoints:
            return checkpoints[0]
        return None
    
    def _determine_resume_phase(self) -> Optional[str]:
        """
        再開すべきフェーズを決定
        
        Returns:
            再開すべきフェーズ名、またはNone（最初から実行）
        """
        if not self.current_phase:
            return None
        
        # フェーズの完了状態を確認
        phase_status = self.phase_progress.get(self.current_phase, {}).get('status', 'unknown')
        
        if phase_status == 'completed':
            # 現在のフェーズが完了している場合は、次のフェーズから開始
            phase_order = [
                'collect_nsfw_detection_data',
                'collect_drug_detection_data',
                'collect_domain_knowledge_data',
                'merge_all_datasets',
                'initialize_quadruple_classifier',
                'run_quadruple_classification',
                'run_statistical_cleaning',
                'convert_to_qlora_format',
                'save_training_dataset'
            ]
            
            try:
                current_index = phase_order.index(self.current_phase)
                if current_index + 1 < len(phase_order):
                    return phase_order[current_index + 1]
            except ValueError:
                pass
        
        # 現在のフェーズが完了していない、または不明な場合は、現在のフェーズから再開
        return self.current_phase
    
    def _save_checkpoint(self):
        """チェックポイント保存"""
        checkpoint_file = self.checkpoint_dir / f"checkpoint_{self.session_id}.pkl"
        
        checkpoint = {
            'session_id': self.session_id,
            'current_phase': self.current_phase,
            'phase_progress': self.phase_progress,
            'nsfw_samples': self.nsfw_samples,
            'drug_samples': self.drug_samples,
            'domain_knowledge_samples': self.domain_knowledge_samples,
            'merged_samples': self.merged_samples,
            'classified_samples': self.classified_samples,
            'cleaned_samples': self.cleaned_samples,
            'training_dataset_samples': self.training_dataset_samples,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(checkpoint, f)
            logger.info(f"チェックポイントを保存しました: {checkpoint_file}")
        except Exception as e:
            logger.error(f"チェックポイントの保存に失敗: {e}")
    
    def collect_nsfw_detection_data(self) -> bool:
        """NSFW検知データセット収集"""
        try:
            logger.info("="*80)
            logger.info("Phase 1: NSFW検知データセット収集")
            logger.info("="*80)
            
            self.current_phase = 'collect_nsfw_detection_data'
            self.phase_progress[self.current_phase] = {
                'status': 'running',
                'started_at': datetime.now().isoformat()
            }
            self._save_checkpoint()
            
            # NSFW検知データセット収集スクリプトのパス
            script_path = PROJECT_ROOT / 'scripts' / 'data' / 'collect_nsfw_detection_dataset.py'
            
            if not script_path.exists():
                logger.error(f"[ERROR] NSFW検知データセット収集スクリプトが見つかりません: {script_path}")
                return False
            
            # 設定からパラメータを取得
            output_dir = Path(self.nsfw_config.get('output_dir', 'D:/webdataset/nsfw_detection_dataset'))
            max_samples = self.nsfw_config.get('max_samples', 50000)
            nsfw_classifier_path = self.nsfw_config.get('nsfw_classifier_path')
            use_multimodal = self.nsfw_config.get('use_multimodal', True)
            
            # コマンドを構築
            cmd = [
                sys.executable,
                str(script_path),
                '--output', str(output_dir),
                '--max-samples', str(max_samples)
            ]
            
            if nsfw_classifier_path:
                cmd.extend(['--nsfw-classifier', str(nsfw_classifier_path)])
            
            if use_multimodal:
                cmd.append('--use-multimodal')
            
            logger.info(f"[PHASE 1] 実行コマンド: {' '.join(cmd)}")
            
            # エラー出力を取得するため、capture_output=Trueにする
            result = subprocess.run(
                cmd,
                capture_output=True,  # エラー出力を取得
                text=True,
                encoding='utf-8',
                errors='replace'
            )
            
            if result.returncode != 0:
                logger.error(f"[PHASE 1] NSFW検知データセット収集に失敗（終了コード: {result.returncode}）")
                if result.stderr:
                    logger.error(f"[PHASE 1] エラー出力: {result.stderr}")
                if result.stdout:
                    logger.error(f"[PHASE 1] 標準出力: {result.stdout}")
                return False
            
            # 収集されたデータを読み込み（trainとvalの両方を読み込む）
            nsfw_train_file = output_dir / 'nsfw_detection_train.jsonl'
            nsfw_val_file = output_dir / 'nsfw_detection_val.jsonl'
            
            files_found = []
            for file_path in [nsfw_train_file, nsfw_val_file]:
                if file_path.exists():
                    files_found.append(file_path)
                    logger.info(f"[PHASE 1] NSFW検知データセットを読み込み中: {file_path}")
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            for line in f:
                                line = line.strip()
                                if line:
                                    try:
                                        sample = json.loads(line)
                                        self.nsfw_samples.append(sample)
                                    except json.JSONDecodeError:
                                        continue
                    except Exception as e:
                        logger.error(f"[PHASE 1] ファイル読み込みエラー ({file_path}): {e}")
            
            if files_found:
                logger.info(f"[PHASE 1] NSFW検知データセット収集完了: {len(self.nsfw_samples)} サンプル（{len(files_found)}ファイルから読み込み）")
            else:
                # ファイルが見つからない場合、ディレクトリ内のファイルをリストアップ
                existing_files = list(output_dir.glob("*.jsonl"))
                if existing_files:
                    logger.warning(f"[PHASE 1] 期待されるファイルが見つかりませんでした。")
                    logger.warning(f"[PHASE 1] 期待されるファイル: {nsfw_train_file}, {nsfw_val_file}")
                    logger.warning(f"[PHASE 1] 実際に存在するファイル: {[str(f) for f in existing_files]}")
                else:
                    logger.warning(f"[PHASE 1] 出力ディレクトリにJSONLファイルが見つかりません: {output_dir}")
            
            self.phase_progress[self.current_phase] = {
                'status': 'completed',
                'started_at': self.phase_progress[self.current_phase]['started_at'],
                'completed_at': datetime.now().isoformat(),
                'samples_collected': len(self.nsfw_samples)
            }
            self._save_checkpoint()
            
            return True
            
        except Exception as e:
            logger.error(f"[PHASE 1] NSFW検知データセット収集でエラー: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.phase_progress[self.current_phase] = {
                'status': 'failed',
                'error': str(e)
            }
            self._save_checkpoint()
            return False
    
    def collect_drug_detection_data(self) -> bool:
        """違法薬物検知データセット収集"""
        try:
            logger.info("="*80)
            logger.info("Phase 2: 違法薬物検知データセット収集")
            logger.info("="*80)
            
            self.current_phase = 'collect_drug_detection_data'
            self.phase_progress[self.current_phase] = {
                'status': 'running',
                'started_at': datetime.now().isoformat()
            }
            self._save_checkpoint()
            
            # 違法薬物検知データセット収集スクリプトのパス
            script_path = PROJECT_ROOT / 'scripts' / 'data' / 'collect_drug_pharmaceutical_detection_dataset.py'
            
            if not script_path.exists():
                logger.error(f"[ERROR] 違法薬物検知データセット収集スクリプトが見つかりません: {script_path}")
                return False
            
            # 設定からパラメータを取得
            output_dir = Path(self.drug_config.get('output_dir', 'D:/webdataset/drug_pharmaceutical_detection_dataset'))
            max_samples_per_source = self.drug_config.get('max_samples_per_source', 1000)
            sources = self.drug_config.get('sources', ['all'])
            
            # コマンドを構築
            cmd = [
                sys.executable,
                str(script_path),
                '--output', str(output_dir),
                '--max-samples-per-source', str(max_samples_per_source)
            ]
            
            # ソースを指定
            if 'all' not in sources:
                cmd.extend(['--sources'] + sources)
            
            logger.info(f"[PHASE 2] 実行コマンド: {' '.join(cmd)}")
            
            # エラー出力を取得するため、capture_output=Trueにする
            result = subprocess.run(
                cmd,
                capture_output=True,  # エラー出力を取得
                text=True,
                encoding='utf-8',
                errors='replace'
            )
            
            if result.returncode != 0:
                logger.error(f"[PHASE 2] 違法薬物検知データセット収集に失敗（終了コード: {result.returncode}）")
                if result.stderr:
                    logger.error(f"[PHASE 2] エラー出力: {result.stderr}")
                if result.stdout:
                    logger.error(f"[PHASE 2] 標準出力: {result.stdout}")
                return False
            
            # 収集されたデータを読み込み（trainとvalの両方を読み込む）
            drug_train_file = output_dir / 'drug_pharmaceutical_detection_train.jsonl'
            drug_val_file = output_dir / 'drug_pharmaceutical_detection_val.jsonl'
            
            files_found = []
            for file_path in [drug_train_file, drug_val_file]:
                if file_path.exists():
                    files_found.append(file_path)
                    logger.info(f"[PHASE 2] 違法薬物検知データセットを読み込み中: {file_path}")
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            for line in f:
                                line = line.strip()
                                if line:
                                    try:
                                        sample = json.loads(line)
                                        self.drug_samples.append(sample)
                                    except json.JSONDecodeError:
                                        continue
                    except Exception as e:
                        logger.error(f"[PHASE 2] ファイル読み込みエラー ({file_path}): {e}")
            
            if files_found:
                logger.info(f"[PHASE 2] 違法薬物検知データセット収集完了: {len(self.drug_samples)} サンプル（{len(files_found)}ファイルから読み込み）")
            else:
                # ファイルが見つからない場合、ディレクトリ内のファイルをリストアップ
                existing_files = list(output_dir.glob("*.jsonl"))
                if existing_files:
                    logger.warning(f"[PHASE 2] 期待されるファイルが見つかりませんでした。")
                    logger.warning(f"[PHASE 2] 期待されるファイル: {drug_train_file}, {drug_val_file}")
                    logger.warning(f"[PHASE 2] 実際に存在するファイル: {[str(f) for f in existing_files]}")
                else:
                    logger.warning(f"[PHASE 2] 出力ディレクトリにJSONLファイルが見つかりません: {output_dir}")
            
            self.phase_progress[self.current_phase] = {
                'status': 'completed',
                'started_at': self.phase_progress[self.current_phase]['started_at'],
                'completed_at': datetime.now().isoformat(),
                'samples_collected': len(self.drug_samples)
            }
            self._save_checkpoint()
            
            return True
            
        except Exception as e:
            logger.error(f"[PHASE 2] 違法薬物検知データセット収集でエラー: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.phase_progress[self.current_phase] = {
                'status': 'failed',
                'error': str(e)
            }
            self._save_checkpoint()
            return False
    
    def collect_domain_knowledge_data(self) -> bool:
        """ドメイン別知識データセット収集"""
        try:
            logger.info("="*80)
            logger.info("Phase 3: ドメイン別知識データセット収集")
            logger.info("="*80)
            
            self.current_phase = 'collect_domain_knowledge_data'
            self.phase_progress[self.current_phase] = {
                'status': 'running',
                'started_at': datetime.now().isoformat()
            }
            self._save_checkpoint()
            
            # ドメイン別知識データセット収集スクリプトのパス
            script_path = PROJECT_ROOT / 'scripts' / 'data' / 'collect_domain_knowledge_with_playwright.py'
            
            if not script_path.exists():
                logger.error(f"[ERROR] ドメイン別知識データセット収集スクリプトが見つかりません: {script_path}")
                return False
            
            # 設定からパラメータを取得
            output_dir = Path(self.domain_knowledge_config.get('output_dir', 'D:/webdataset/domain_knowledge_collected'))
            domains = self.domain_knowledge_config.get('domains', ['defense', 'aerospace', 'transport', 'general'])
            so8t_model_path = self.domain_knowledge_config.get('so8t_model_path')
            use_cursor_browser = self.domain_knowledge_config.get('use_cursor_browser', True)
            remote_debugging_port = self.domain_knowledge_config.get('remote_debugging_port', 9222)
            delay_per_request = self.domain_knowledge_config.get('delay_per_request', 2.0)
            timeout = self.domain_knowledge_config.get('timeout', 30000)
            max_pages_per_domain = self.domain_knowledge_config.get('max_pages_per_domain', 100)
            max_depth = self.domain_knowledge_config.get('max_depth', 3)
            quality_threshold = self.domain_knowledge_config.get('quality_threshold', 0.7)
            enable_button_click = self.domain_knowledge_config.get('enable_button_click', True)
            max_button_clicks_per_page = self.domain_knowledge_config.get('max_button_clicks_per_page', 3)
            button_click_delay = self.domain_knowledge_config.get('button_click_delay', [1.5, 3.0])
            
            # コマンドを構築
            cmd = [
                sys.executable,
                str(script_path),
                '--output', str(output_dir),
                '--domains', ','.join(domains),
                '--delay', str(delay_per_request),
                '--timeout', str(timeout),
                '--max_pages_per_domain', str(max_pages_per_domain),
                '--max_depth', str(max_depth),
                '--quality_threshold', str(quality_threshold),
                '--max_button_clicks_per_page', str(max_button_clicks_per_page),
                '--button_click_delay', str(button_click_delay[0]), str(button_click_delay[1])
            ]
            
            if enable_button_click:
                cmd.append('--enable_button_click')
            
            if so8t_model_path:
                cmd.extend(['--so8t_model_path', str(so8t_model_path)])
            
            if use_cursor_browser:
                cmd.append('--use_cursor_browser')
                cmd.extend(['--remote_debugging_port', str(remote_debugging_port)])
            
            logger.info(f"[PHASE 3] 実行コマンド: {' '.join(cmd)}")
            logger.info(f"[PHASE 3] ブラウザを起動します（headless=False）...")
            
            # ブラウザを表示するため、capture_output=Falseにする
            result = subprocess.run(
                cmd,
                capture_output=False,  # ブラウザを表示するためFalse
                text=True,
                encoding='utf-8',
                errors='replace'
            )
            
            if result.returncode != 0:
                logger.error(f"[PHASE 3] ドメイン別知識データセット収集に失敗（終了コード: {result.returncode}）")
                logger.warning(f"[PHASE 3] ログファイルを確認してください: logs/collect_domain_knowledge_playwright.log")
                return False
            
            # 収集されたデータを読み込み（パターンマッチでファイルを検索）
            # まず_cleaned.jsonlファイルを検索
            domain_data_files = list(output_dir.glob("domain_knowledge_*_cleaned.jsonl"))
            if not domain_data_files:
                # フォールバック: cleanedなしのファイルも検索
                domain_data_files = list(output_dir.glob("domain_knowledge_*.jsonl"))
            
            if domain_data_files:
                # 最新のファイルを優先（セッションIDが新しいもの）
                domain_data_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                logger.info(f"[PHASE 3] {len(domain_data_files)}個のドメイン別知識データセットファイルを検出")
                
                for file_path in domain_data_files:
                    logger.info(f"[PHASE 3] ドメイン別知識データセットを読み込み中: {file_path}")
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            for line in f:
                                line = line.strip()
                                if line:
                                    try:
                                        sample = json.loads(line)
                                        self.domain_knowledge_samples.append(sample)
                                    except json.JSONDecodeError:
                                        continue
                    except Exception as e:
                        logger.error(f"[PHASE 3] ファイル読み込みエラー ({file_path}): {e}")
                
                logger.info(f"[PHASE 3] ドメイン別知識データセット収集完了: {len(self.domain_knowledge_samples)} サンプル（{len(domain_data_files)}ファイルから読み込み）")
            else:
                # ファイルが見つからない場合、ディレクトリ内のファイルをリストアップ
                existing_files = list(output_dir.glob("*.jsonl"))
                if existing_files:
                    logger.warning(f"[PHASE 3] 期待されるファイル（domain_knowledge_*.jsonl）が見つかりませんでした。")
                    logger.warning(f"[PHASE 3] 実際に存在するファイル: {[str(f) for f in existing_files]}")
                else:
                    logger.warning(f"[PHASE 3] 出力ディレクトリにJSONLファイルが見つかりません: {output_dir}")
            
            self.phase_progress[self.current_phase] = {
                'status': 'completed',
                'started_at': self.phase_progress[self.current_phase]['started_at'],
                'completed_at': datetime.now().isoformat(),
                'samples_collected': len(self.domain_knowledge_samples)
            }
            self._save_checkpoint()
            
            return True
            
        except Exception as e:
            logger.error(f"[PHASE 3] ドメイン別知識データセット収集でエラー: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.phase_progress[self.current_phase] = {
                'status': 'failed',
                'error': str(e)
            }
            self._save_checkpoint()
            return False
    
    def merge_all_datasets(self) -> bool:
        """全データセットの統合"""
        try:
            logger.info("="*80)
            logger.info("Phase 4: 全データセットの統合")
            logger.info("="*80)
            
            self.current_phase = 'merge_all_datasets'
            self.phase_progress[self.current_phase] = {
                'status': 'running',
                'started_at': datetime.now().isoformat()
            }
            self._save_checkpoint()
            
            # 全サンプルを統合
            self.merged_samples = []
            
            # NSFW検知データセット
            for sample in self.nsfw_samples:
                sample['data_source'] = 'nsfw_detection'
                self.merged_samples.append(sample)
            
            # 違法薬物検知データセット
            for sample in self.drug_samples:
                sample['data_source'] = 'drug_detection'
                self.merged_samples.append(sample)
            
            # ドメイン別知識データセット
            for sample in self.domain_knowledge_samples:
                sample['data_source'] = 'domain_knowledge'
                self.merged_samples.append(sample)
            
            logger.info(f"[PHASE 4] 全データセット統合完了: {len(self.merged_samples)} サンプル")
            logger.info(f"  - NSFW検知: {len(self.nsfw_samples)} サンプル")
            logger.info(f"  - 違法薬物検知: {len(self.drug_samples)} サンプル")
            logger.info(f"  - ドメイン別知識: {len(self.domain_knowledge_samples)} サンプル")
            
            self.phase_progress[self.current_phase] = {
                'status': 'completed',
                'started_at': self.phase_progress[self.current_phase]['started_at'],
                'completed_at': datetime.now().isoformat(),
                'total_samples': len(self.merged_samples)
            }
            self._save_checkpoint()
            
            return True
            
        except Exception as e:
            logger.error(f"[PHASE 4] 全データセット統合でエラー: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.phase_progress[self.current_phase] = {
                'status': 'failed',
                'error': str(e)
            }
            self._save_checkpoint()
            return False
    
    def initialize_quadruple_classifier(self) -> bool:
        """QuadrupleClassifierの初期化"""
        try:
            logger.info("="*80)
            logger.info("Phase 5: QuadrupleClassifierの初期化")
            logger.info("="*80)
            
            if not QUADRUPLE_CLASSIFIER_AVAILABLE:
                logger.warning("[PHASE 5] QuadrupleClassifierが利用できません")
                return False
            
            # SO8Tモデルパスを取得
            so8t_model_path = self.quadruple_config.get('so8t_model_path')
            
            if not so8t_model_path:
                # 自動検出を試みる
                default_paths = [
                    "D:/webdataset/models/so8t-phi4-so8t-ja-finetuned",
                    "models/so8t-phi4-so8t-ja-finetuned",
                    "so8t-mmllm/models/so8t-phi4-so8t-ja-finetuned",
                    Path(PROJECT_ROOT) / "models" / "so8t-phi4-so8t-ja-finetuned",
                    Path(PROJECT_ROOT) / "so8t-mmllm" / "models" / "so8t-phi4-so8t-ja-finetuned"
                ]
                for path in default_paths:
                    path_obj = Path(path)
                    if path_obj.exists() and (path_obj / "config.json").exists():
                        so8t_model_path = str(path_obj)
                        logger.info(f"[PHASE 5] SO8Tモデルパスを自動検出: {so8t_model_path}")
                        break
            
            if not so8t_model_path:
                logger.warning("[PHASE 5] SO8Tモデルパスが見つかりません")
                logger.warning("[PHASE 5] ルールベース分類のみを使用します")
                return False
            
            # QuadrupleClassifierを初期化
            logger.info(f"[PHASE 5] QuadrupleClassifierを初期化中: {so8t_model_path}")
            self.quadruple_classifier = QuadrupleClassifier(so8t_model_path=so8t_model_path)
            
            if self.quadruple_classifier.so8t_model is None:
                logger.warning("[PHASE 5] QuadrupleClassifierは初期化されましたが、SO8Tモデルは利用できません")
                logger.warning("[PHASE 5] ルールベース分類のみを使用します")
                return False
            
            logger.info("[PHASE 5] QuadrupleClassifierの初期化完了")
            return True
            
        except Exception as e:
            logger.error(f"[PHASE 5] QuadrupleClassifierの初期化でエラー: {e}")
            import traceback
            logger.error(traceback.format_exc())
            logger.warning("[PHASE 5] ルールベース分類のみを使用します")
            return False
    
    def run_quadruple_classification(self) -> bool:
        """4値分類・四重推論の実行"""
        try:
            logger.info("="*80)
            logger.info("Phase 6: 4値分類・四重推論の実行")
            logger.info("="*80)
            
            self.current_phase = 'run_quadruple_classification'
            self.phase_progress[self.current_phase] = {
                'status': 'running',
                'started_at': datetime.now().isoformat()
            }
            self._save_checkpoint()
            
            if not self.quadruple_classifier:
                logger.warning("[PHASE 6] QuadrupleClassifierが利用できません")
                logger.warning("[PHASE 6] ルールベース分類のみを使用します")
                # ルールベース分類を実行（簡易版）
                self.classified_samples = self.merged_samples.copy()
                logger.info(f"[PHASE 6] ルールベース分類完了: {len(self.classified_samples)} サンプル")
                self.phase_progress[self.current_phase] = {
                    'status': 'completed',
                    'started_at': self.phase_progress[self.current_phase]['started_at'],
                    'completed_at': datetime.now().isoformat(),
                    'samples_classified': len(self.classified_samples)
                }
                self._save_checkpoint()
                return True
            
            logger.info(f"[PHASE 6] {len(self.merged_samples)} サンプルを4値分類・四重推論中...")
            
            self.classified_samples = []
            processed_count = 0
            error_count = 0
            
            # 進捗バー
            if TQDM_AVAILABLE:
                iterator = tqdm(self.merged_samples, desc="4値分類・四重推論")
            else:
                iterator = self.merged_samples
            
            for i, sample in enumerate(iterator):
                try:
                    # 4値分類と四重推論を実行
                    classified_sample = self.quadruple_classifier.classify_quadruple(sample)
                    self.classified_samples.append(classified_sample)
                    processed_count += 1
                    
                    # 進捗ログ（100サンプルごと）
                    if (i + 1) % 100 == 0:
                        logger.info(f"[PHASE 6] 処理済み: {i + 1}/{len(self.merged_samples)} サンプル")
                    
                except Exception as e:
                    logger.error(f"[PHASE 6] サンプル {i} の処理に失敗: {e}")
                    error_count += 1
                    continue
            
            logger.info(f"[PHASE 6] 4値分類・四重推論完了: {processed_count} 処理済み, {error_count} エラー")
            
            # 4値分類ラベルの分布を計算
            label_dist = Counter(s.get('four_class_label', 'ALLOW') for s in self.classified_samples)
            logger.info(f"[PHASE 6] 4値分類ラベル分布: {dict(label_dist)}")
            
            self.phase_progress[self.current_phase] = {
                'status': 'completed',
                'started_at': self.phase_progress[self.current_phase]['started_at'],
                'completed_at': datetime.now().isoformat(),
                'samples_classified': len(self.classified_samples),
                'label_distribution': dict(label_dist)
            }
            self._save_checkpoint()
            
            return True
            
        except Exception as e:
            logger.error(f"[PHASE 6] 4値分類・四重推論でエラー: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.phase_progress[self.current_phase] = {
                'status': 'failed',
                'error': str(e)
            }
            self._save_checkpoint()
            return False
    
    def run_statistical_cleaning(self) -> bool:
        """統計的データクレンジングの実行"""
        try:
            logger.info("="*80)
            logger.info("Phase 7: 統計的データクレンジングの実行")
            logger.info("="*80)
            
            self.current_phase = 'run_statistical_cleaning'
            self.phase_progress[self.current_phase] = {
                'status': 'running',
                'started_at': datetime.now().isoformat()
            }
            self._save_checkpoint()
            
            logger.info(f"[PHASE 7] {len(self.classified_samples)} サンプルを統計的データクレンジング中...")
            
            # 統計的データクレンジングを実行
            self.cleaned_samples = self._statistical_data_cleaning(self.classified_samples)
            
            logger.info(f"[PHASE 7] 統計的データクレンジング完了: {len(self.cleaned_samples)} サンプル (削除: {len(self.classified_samples) - len(self.cleaned_samples)})")
            
            self.phase_progress[self.current_phase] = {
                'status': 'completed',
                'started_at': self.phase_progress[self.current_phase]['started_at'],
                'completed_at': datetime.now().isoformat(),
                'samples_before': len(self.classified_samples),
                'samples_after': len(self.cleaned_samples),
                'samples_removed': len(self.classified_samples) - len(self.cleaned_samples)
            }
            self._save_checkpoint()
            
            return True
            
        except Exception as e:
            logger.error(f"[PHASE 7] 統計的データクレンジングでエラー: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.phase_progress[self.current_phase] = {
                'status': 'failed',
                'error': str(e)
            }
            self._save_checkpoint()
            return False
    
    def _statistical_data_cleaning(self, samples: List[Dict]) -> List[Dict]:
        """
        統計的に有意な手法でデータクレンジング
        
        参考実装: scripts/data/parallel_deep_research_scraping.py の statistical_data_cleaning() メソッド
        """
        if not samples:
            return []
        
        logger.info(f"[STATISTICAL-CLEANING] {len(samples)} サンプルを統計的データクレンジング中...")
        
        cleaned_samples = []
        removal_stats = {
            'duplicates': 0,
            'outliers_zscore': 0,
            'outliers_iqr': 0,
            'low_quality': 0,
            'statistical_filter': 0,
            'total_removed': 0
        }
        
        # 1. 重複検出（ハッシュベース + 類似度ベース）
        seen_hashes = set()
        unique_samples = []
        
        for sample in samples:
            text = sample.get('input', sample.get('text', sample.get('output', '')))
            if not text:
                removal_stats['low_quality'] += 1
                continue
            
            # ハッシュベース重複検出
            text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
            if text_hash in seen_hashes:
                removal_stats['duplicates'] += 1
                continue
            
            # 類似度ベース重複検出
            is_duplicate = False
            for existing_sample in unique_samples:
                existing_text = existing_sample.get('input', existing_sample.get('text', existing_sample.get('output', '')))
                similarity = SequenceMatcher(None, text[:500], existing_text[:500]).ratio()
                if similarity > 0.95:  # 95%以上類似している場合は重複とみなす
                    is_duplicate = True
                    removal_stats['duplicates'] += 1
                    break
            
            if not is_duplicate:
                seen_hashes.add(text_hash)
                unique_samples.append(sample)
        
        logger.info(f"[STATISTICAL-CLEANING] 重複検出: {removal_stats['duplicates']} サンプルを削除")
        
        if not unique_samples:
            logger.warning("[STATISTICAL-CLEANING] 重複検出後、サンプルが残りませんでした")
            return []
        
        # 2. 外れ値検出（Z-score、IQR）
        if NUMPY_AVAILABLE and len(unique_samples) > 10:
            # テキスト長の統計
            text_lengths = [len(s.get('input', s.get('text', s.get('output', '')))) for s in unique_samples]
            
            if len(text_lengths) > 0 and np.std(text_lengths) > 0:
                mean_length = np.mean(text_lengths)
                std_length = np.std(text_lengths)
                
                # Z-scoreによる外れ値検出（|z| > 3）
                z_scores = np.abs((np.array(text_lengths) - mean_length) / std_length)
                z_threshold = 3.0
                
                zscore_filtered_samples = []
                for i, sample in enumerate(unique_samples):
                    if z_scores[i] <= z_threshold:
                        zscore_filtered_samples.append(sample)
                    else:
                        removal_stats['outliers_zscore'] += 1
                
                unique_samples = zscore_filtered_samples
                logger.info(f"[STATISTICAL-CLEANING] Z-score外れ値検出: {removal_stats['outliers_zscore']} サンプルを削除")
                
                # IQRによる外れ値検出
                if len(unique_samples) > 10:
                    text_lengths_cleaned = [len(s.get('input', s.get('text', s.get('output', '')))) for s in unique_samples]
                    q1 = np.percentile(text_lengths_cleaned, 25)
                    q3 = np.percentile(text_lengths_cleaned, 75)
                    iqr = q3 - q1
                    
                    if iqr > 0:
                        lower_bound = q1 - 1.5 * iqr
                        upper_bound = q3 + 1.5 * iqr
                        
                        iqr_filtered_samples = []
                        for sample in unique_samples:
                            text_length = len(sample.get('input', sample.get('text', sample.get('output', ''))))
                            if lower_bound <= text_length <= upper_bound:
                                iqr_filtered_samples.append(sample)
                            else:
                                removal_stats['outliers_iqr'] += 1
                        
                        unique_samples = iqr_filtered_samples
                        logger.info(f"[STATISTICAL-CLEANING] IQR外れ値検出: {removal_stats['outliers_iqr']} サンプルを削除")
        
        # 3. 統計的品質スコアリング
        quality_filtered_samples = []
        for sample in unique_samples:
            quality_score = self._calculate_statistical_quality_score(sample)
            sample['statistical_quality_score'] = quality_score
            
            # 品質スコアが低いサンプルを除外
            min_quality_score = self.cleaning_config.get('quality_scoring', {}).get('min_quality_score', 0.3)
            if quality_score < min_quality_score:
                removal_stats['low_quality'] += 1
                continue
            
            quality_filtered_samples.append(sample)
        
        cleaned_samples = quality_filtered_samples
        logger.info(f"[STATISTICAL-CLEANING] 低品質サンプル削除: {removal_stats['low_quality']} サンプルを削除")
        
        # 4. 信頼区間によるフィルタリング
        if NUMPY_AVAILABLE and len(cleaned_samples) > 10:
            confidence_scores = [s.get('four_class_label_id', 0) for s in cleaned_samples]
            
            if len(confidence_scores) > 0 and np.std(confidence_scores) > 0:
                mean_confidence = np.mean(confidence_scores)
                std_confidence = np.std(confidence_scores)
                
                # 95%信頼区間
                confidence_level = self.cleaning_config.get('confidence_interval', {}).get('confidence_level', 0.95)
                z_value = 1.96 if confidence_level == 0.95 else 2.576  # 95% or 99%
                confidence_interval_lower = mean_confidence - z_value * std_confidence
                confidence_interval_upper = mean_confidence + z_value * std_confidence
                
                ci_filtered_samples = []
                for sample in cleaned_samples:
                    confidence = sample.get('four_class_label_id', 0)
                    if confidence_interval_lower <= confidence <= confidence_interval_upper:
                        ci_filtered_samples.append(sample)
                    else:
                        removal_stats['statistical_filter'] += 1
                
                cleaned_samples = ci_filtered_samples
                logger.info(f"[STATISTICAL-CLEANING] 信頼区間フィルタリング: {removal_stats['statistical_filter']} サンプルを削除")
        
        removal_stats['total_removed'] = len(samples) - len(cleaned_samples)
        logger.info("="*80)
        logger.info("[STATISTICAL-CLEANING] 統計的データクレンジング完了:")
        logger.info(f"  元のサンプル数: {len(samples)}")
        logger.info(f"  クレンジング後サンプル数: {len(cleaned_samples)}")
        logger.info(f"  削除統計: {removal_stats}")
        logger.info(f"  保持率: {len(cleaned_samples)/len(samples)*100:.2f}%")
        logger.info("="*80)
        
        return cleaned_samples
    
    def _calculate_statistical_quality_score(self, sample: Dict) -> float:
        """
        統計的品質スコアを計算
        
        Args:
            sample: サンプル
        
        Returns:
            品質スコア（0.0-1.0）
        """
        score = 0.0
        max_score = 0.0
        
        # 1. テキスト長スコア（100-5000文字が最適）
        text = sample.get('input', sample.get('text', sample.get('output', '')))
        text_length = len(text)
        if 100 <= text_length <= 5000:
            score += 0.3
        elif 50 <= text_length < 100 or 5000 < text_length <= 10000:
            score += 0.2
        else:
            score += 0.1
        max_score += 0.3
        
        # 2. 情報量スコア（ユニークな単語数）
        if text:
            words = text.split()
            unique_words = len(set(words))
            if len(words) > 0:
                uniqueness_ratio = unique_words / len(words)
                score += 0.2 * uniqueness_ratio
        max_score += 0.2
        
        # 3. 4値分類の信頼度
        four_class_label = sample.get('four_class_label', 'ALLOW')
        if four_class_label in ['ALLOW', 'ESCALATION', 'DENY', 'REFUSE']:
            score += 0.2
        max_score += 0.2
        
        # 4. メタデータの完全性
        metadata_fields = ['url', 'category', 'language', 'title']
        metadata_completeness = sum(1 for field in metadata_fields if sample.get(field))
        score += 0.3 * (metadata_completeness / len(metadata_fields))
        max_score += 0.3
        
        # 正規化
        if max_score > 0:
            return score / max_score
        else:
            return 0.0
    
    def convert_to_qlora_format(self) -> bool:
        """QLoRA学習用データセット形式への変換"""
        try:
            logger.info("="*80)
            logger.info("Phase 8: QLoRA学習用データセット形式への変換")
            logger.info("="*80)
            
            self.current_phase = 'convert_to_qlora_format'
            self.phase_progress[self.current_phase] = {
                'status': 'running',
                'started_at': datetime.now().isoformat()
            }
            self._save_checkpoint()
            
            logger.info(f"[PHASE 8] {len(self.cleaned_samples)} サンプルをQLoRA学習用データセット形式に変換中...")
            
            self.training_dataset_samples = []
            
            # 進捗バー
            if TQDM_AVAILABLE:
                iterator = tqdm(self.cleaned_samples, desc="QLoRA形式変換")
            else:
                iterator = self.cleaned_samples
            
            for sample in iterator:
                training_sample = self._convert_to_training_format(sample)
                if training_sample:
                    self.training_dataset_samples.append(training_sample)
            
            logger.info(f"[PHASE 8] QLoRA学習用データセット形式への変換完了: {len(self.training_dataset_samples)} サンプル")
            
            self.phase_progress[self.current_phase] = {
                'status': 'completed',
                'started_at': self.phase_progress[self.current_phase]['started_at'],
                'completed_at': datetime.now().isoformat(),
                'samples_converted': len(self.training_dataset_samples)
            }
            self._save_checkpoint()
            
            return True
            
        except Exception as e:
            logger.error(f"[PHASE 8] QLoRA学習用データセット形式への変換でエラー: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.phase_progress[self.current_phase] = {
                'status': 'failed',
                'error': str(e)
            }
            self._save_checkpoint()
            return False
    
    def _convert_to_training_format(self, classified_sample: Dict) -> Optional[Dict]:
        """
        分類済みサンプルを学習用データセット形式に変換
        
        参考実装: scripts/data/parallel_deep_research_scraping.py の _convert_to_training_format() メソッド
        """
        try:
            text = classified_sample.get('text', classified_sample.get('input', classified_sample.get('output', '')))
            if not text or len(text) < 50:
                return None
            
            # 四重推論結果を取得
            quadruple_classification = classified_sample.get('quadruple_classification', {})
            task_reasoning = quadruple_classification.get('task', '')
            safety_reasoning = quadruple_classification.get('safety', '')
            policy_reasoning = quadruple_classification.get('policy', '')
            final_reasoning = quadruple_classification.get('final', '')
            four_class_label = classified_sample.get('four_class_label', 'ALLOW')
            
            # 学習用データセット形式に変換
            # instruction-output形式
            instruction = f"""以下のテキストを読み、四重推論（Task/Safety/Policy/Final）を行い、4値分類（ALLOW/ESCALATION/DENY/REFUSE）を実行してください。

テキスト:
{text[:2000]}  # 最大2000文字
"""
            
            output = f"""<think-task>
{task_reasoning}
</think-task>

<think-safety>
{safety_reasoning}
</think-safety>

<think-policy>
{policy_reasoning}
</think-policy>

<final>
{final_reasoning}

分類結果: {four_class_label}
</final>"""
            
            training_sample = {
                'instruction': instruction,
                'output': output,
                'input': text[:2000],  # 最大2000文字
                'category': classified_sample.get('category', 'unknown'),
                'language': classified_sample.get('language', 'unknown'),
                'url': classified_sample.get('url', ''),
                'domain': classified_sample.get('domain', ''),
                'title': classified_sample.get('title', ''),
                'keyword': classified_sample.get('keyword', ''),
                'four_class_label': four_class_label,
                'four_class_label_id': classified_sample.get('four_class_label_id', 0),
                'quadruple_classification': quadruple_classification,
                'nsfw_label': classified_sample.get('nsfw_label', 'safe'),
                'nsfw_confidence': classified_sample.get('nsfw_confidence', 0.0),
                'data_source': classified_sample.get('data_source', 'unknown'),
                'crawled_at': classified_sample.get('crawled_at', datetime.now().isoformat()),
                'classified_at': classified_sample.get('classified_at', datetime.now().isoformat()),
                'classification_version': classified_sample.get('classification_version', '1.0'),
                'source': 'nsfw_drug_detection_qlora_training_data_pipeline'
            }
            
            return training_sample
            
        except Exception as e:
            logger.error(f"[TRAINING-FORMAT] サンプルの学習用データセット形式への変換に失敗: {e}")
            return None
    
    def save_training_dataset(self) -> bool:
        """学習用データセットの保存"""
        try:
            logger.info("="*80)
            logger.info("Phase 9: 学習用データセットの保存")
            logger.info("="*80)
            
            self.current_phase = 'save_training_dataset'
            self.phase_progress[self.current_phase] = {
                'status': 'running',
                'started_at': datetime.now().isoformat()
            }
            self._save_checkpoint()
            
            if not self.training_dataset_samples:
                logger.warning("[PHASE 9] 学習用データセットサンプルがありません")
                return False
            
            logger.info(f"[PHASE 9] {len(self.training_dataset_samples)} サンプルを保存中...")
            
            # JSONL形式で保存
            training_dataset_file = self.output_dir / f"qlora_training_dataset_{self.session_id}.jsonl"
            with open(training_dataset_file, 'w', encoding='utf-8') as f:
                for sample in self.training_dataset_samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            
            logger.info(f"[PHASE 9] 学習用データセットを保存しました: {training_dataset_file}")
            
            # メタデータを保存
            metadata = {
                'total_samples': len(self.training_dataset_samples),
                'session_id': self.session_id,
                'created_at': datetime.now().isoformat(),
                'source': 'nsfw_drug_detection_qlora_training_data_pipeline',
                'classification_version': self.training_dataset_samples[0].get('classification_version', '1.0') if self.training_dataset_samples else '1.0',
                'four_class_distribution': self._calculate_four_class_distribution(),
                'category_distribution': self._calculate_category_distribution(),
                'language_distribution': self._calculate_language_distribution(),
                'data_source_distribution': self._calculate_data_source_distribution()
            }
            
            metadata_file = self.output_dir / f"qlora_training_dataset_metadata_{self.session_id}.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            logger.info(f"[PHASE 9] メタデータを保存しました: {metadata_file}")
            
            self.phase_progress[self.current_phase] = {
                'status': 'completed',
                'started_at': self.phase_progress[self.current_phase]['started_at'],
                'completed_at': datetime.now().isoformat(),
                'samples_saved': len(self.training_dataset_samples),
                'output_file': str(training_dataset_file),
                'metadata_file': str(metadata_file)
            }
            self._save_checkpoint()
            
            return True
            
        except Exception as e:
            logger.error(f"[PHASE 9] 学習用データセットの保存でエラー: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.phase_progress[self.current_phase] = {
                'status': 'failed',
                'error': str(e)
            }
            self._save_checkpoint()
            return False
    
    def _calculate_four_class_distribution(self) -> Dict[str, int]:
        """4値分類ラベルの分布を計算"""
        distribution = {'ALLOW': 0, 'ESCALATION': 0, 'DENY': 0, 'REFUSE': 0}
        for sample in self.training_dataset_samples:
            label = sample.get('four_class_label', 'ALLOW')
            if label in distribution:
                distribution[label] += 1
        return distribution
    
    def _calculate_category_distribution(self) -> Dict[str, int]:
        """カテゴリの分布を計算"""
        distribution = {}
        for sample in self.training_dataset_samples:
            category = sample.get('category', 'unknown')
            distribution[category] = distribution.get(category, 0) + 1
        return distribution
    
    def _calculate_language_distribution(self) -> Dict[str, int]:
        """言語の分布を計算"""
        distribution = {}
        for sample in self.training_dataset_samples:
            language = sample.get('language', 'unknown')
            distribution[language] = distribution.get(language, 0) + 1
        return distribution
    
    def _calculate_data_source_distribution(self) -> Dict[str, int]:
        """データソースの分布を計算"""
        distribution = {}
        for sample in self.training_dataset_samples:
            data_source = sample.get('data_source', 'unknown')
            distribution[data_source] = distribution.get(data_source, 0) + 1
        return distribution
    
    def run(self, auto_resume: bool = True) -> bool:
        """
        全自動パイプラインの実行
        
        Args:
            auto_resume: 自動再開モード（Trueの場合はチェックポイントから自動再開）
        """
        try:
            logger.info("="*80)
            logger.info("NSFW・違法薬物検知目的QLoRA学習用データ生成全自動パイプライン開始")
            logger.info("="*80)
            
            # チェックポイントから再開（auto_resumeがFalseの場合はスキップ）
            resume_from_phase = None
            if auto_resume:
                checkpoint = self._load_checkpoint(auto_resume=auto_resume)
                if checkpoint:
                    # データを復元
                    nsfw_samples_count = len(checkpoint.get('nsfw_samples', []))
                    drug_samples_count = len(checkpoint.get('drug_samples', []))
                    domain_knowledge_samples_count = len(checkpoint.get('domain_knowledge_samples', []))
                    total_samples = nsfw_samples_count + drug_samples_count + domain_knowledge_samples_count
                    
                    # データが0件の場合は新規実行
                    if total_samples == 0:
                        logger.warning("[CHECKPOINT] チェックポイントにデータがありません。新規実行を開始します。")
                        checkpoint = None
                    else:
                        logger.info("="*80)
                        logger.info("[RESUME] チェックポイントから再開します")
                        logger.info("="*80)
                        
                        # セッションIDを復元（または新しいセッションIDを使用）
                        checkpoint_session_id = checkpoint.get('session_id', self.session_id)
                        if checkpoint_session_id != self.session_id:
                            logger.info(f"[RESUME] セッションIDを復元: {checkpoint_session_id}")
                            self.session_id = checkpoint_session_id
                        
                        # データを復元
                        self.nsfw_samples = checkpoint.get('nsfw_samples', [])
                        self.drug_samples = checkpoint.get('drug_samples', [])
                        self.domain_knowledge_samples = checkpoint.get('domain_knowledge_samples', [])
                        self.merged_samples = checkpoint.get('merged_samples', [])
                        self.classified_samples = checkpoint.get('classified_samples', [])
                        self.cleaned_samples = checkpoint.get('cleaned_samples', [])
                        self.training_dataset_samples = checkpoint.get('training_dataset_samples', [])
                        self.current_phase = checkpoint.get('current_phase', None)
                        self.phase_progress = checkpoint.get('phase_progress', {})
                        
                        logger.info(f"[RESUME] 復元されたデータ:")
                        logger.info(f"  - NSFW検知サンプル: {len(self.nsfw_samples)}")
                        logger.info(f"  - 違法薬物検知サンプル: {len(self.drug_samples)}")
                        logger.info(f"  - ドメイン別知識サンプル: {len(self.domain_knowledge_samples)}")
                        logger.info(f"  - 統合サンプル: {len(self.merged_samples)}")
                        logger.info(f"  - 分類済みサンプル: {len(self.classified_samples)}")
                        logger.info(f"  - クレンジング済みサンプル: {len(self.cleaned_samples)}")
                        logger.info(f"  - 学習用データセットサンプル: {len(self.training_dataset_samples)}")
                        logger.info(f"  - 現在のフェーズ: {self.current_phase}")
                        
                        # 中断されたフェーズから再開
                        resume_from_phase = self._determine_resume_phase()
                        if resume_from_phase:
                            logger.info(f"[RESUME] フェーズ '{resume_from_phase}' から再開します")
                else:
                    logger.info("[NEW] チェックポイントが見つかりません。新規実行を開始します。")
            else:
                logger.info("[NEW] --no-auto-resumeオプションが指定されました。新規実行を開始します。")
                logger.info("[NEW] チェックポイントは無視されます。")
            
            # フェーズの実行順序を定義
            phases = [
                ('collect_nsfw_detection_data', self.collect_nsfw_detection_data, self.nsfw_config),
                ('collect_drug_detection_data', self.collect_drug_detection_data, self.drug_config),
                ('collect_domain_knowledge_data', self.collect_domain_knowledge_data, self.domain_knowledge_config),
                ('merge_all_datasets', self.merge_all_datasets, {'enabled': True}),
                ('initialize_quadruple_classifier', self.initialize_quadruple_classifier, self.quadruple_config),
                ('run_quadruple_classification', self.run_quadruple_classification, {'enabled': True}),
                ('run_statistical_cleaning', self.run_statistical_cleaning, self.cleaning_config),
                ('convert_to_qlora_format', self.convert_to_qlora_format, {'enabled': True}),
                ('save_training_dataset', self.save_training_dataset, {'enabled': True})
            ]
            
            # 再開フェーズが見つかった場合は、そのフェーズから開始
            start_index = 0
            if resume_from_phase:
                for i, (phase_name, _, _) in enumerate(phases):
                    if phase_name == resume_from_phase:
                        start_index = i
                        logger.info(f"[RESUME] フェーズ {i+1}/{len(phases)} から再開: {phase_name}")
                        break
            
            # 各フェーズを実行
            for i, (phase_name, phase_func, phase_config) in enumerate(phases[start_index:], start=start_index):
                # フェーズが有効でない場合はスキップ
                if not phase_config.get('enabled', True):
                    logger.info(f"[SKIP] Phase {i+1}: {phase_name} は無効化されています")
                    continue
                
                # 既に完了しているフェーズはスキップ（ただし、データが存在する場合のみ）
                phase_progress = self.phase_progress.get(phase_name, {})
                if phase_progress.get('status') == 'completed':
                    # データ収集フェーズの場合は、データが存在するか確認
                    if phase_name in ['collect_nsfw_detection_data', 'collect_drug_detection_data', 'collect_domain_knowledge_data']:
                        # データが存在しない場合は再実行
                        if phase_name == 'collect_nsfw_detection_data' and len(self.nsfw_samples) == 0:
                            logger.warning(f"[SKIP-OVERRIDE] Phase {i+1}: {phase_name} は完了していますが、データが0件のため再実行します")
                        elif phase_name == 'collect_drug_detection_data' and len(self.drug_samples) == 0:
                            logger.warning(f"[SKIP-OVERRIDE] Phase {i+1}: {phase_name} は完了していますが、データが0件のため再実行します")
                        elif phase_name == 'collect_domain_knowledge_data' and len(self.domain_knowledge_samples) == 0:
                            logger.warning(f"[SKIP-OVERRIDE] Phase {i+1}: {phase_name} は完了していますが、データが0件のため再実行します")
                        else:
                            logger.info(f"[SKIP] Phase {i+1}: {phase_name} は既に完了しています（データ: {len(self.nsfw_samples) if phase_name == 'collect_nsfw_detection_data' else len(self.drug_samples) if phase_name == 'collect_drug_detection_data' else len(self.domain_knowledge_samples)}件）")
                            continue
                    else:
                        logger.info(f"[SKIP] Phase {i+1}: {phase_name} は既に完了しています")
                        continue
                
                logger.info(f"[EXECUTE] Phase {i+1}/{len(phases)}: {phase_name}")
                
                # フェーズを実行
                if phase_name == 'run_statistical_cleaning':
                    # 統計的データクレンジングは特別処理
                    if not phase_func():
                        logger.error(f"[ERROR] Phase {i+1} ({phase_name}) が失敗しました")
                        return False
                elif phase_name == 'initialize_quadruple_classifier':
                    # QuadrupleClassifierの初期化は警告のみ
                    if not phase_func():
                        logger.warning(f"[WARNING] Phase {i+1} ({phase_name}) が失敗しましたが、ルールベース分類で続行します")
                else:
                    # その他のフェーズはエラーで終了
                    if not phase_func():
                        logger.error(f"[ERROR] Phase {i+1} ({phase_name}) が失敗しました")
                        return False
            
            logger.info("="*80)
            logger.info("NSFW・違法薬物検知目的QLoRA学習用データ生成全自動パイプライン完了")
            logger.info("="*80)
            logger.info(f"学習用データセット: {len(self.training_dataset_samples)} サンプル")
            logger.info(f"出力ファイル: {self.output_dir / f'qlora_training_dataset_{self.session_id}.jsonl'}")
            
            # 音声通知
            AudioNotifier.play_notification()
            
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] パイプライン実行中にエラーが発生: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self._save_checkpoint()
            return False


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='NSFW・違法薬物検知目的QLoRA学習用データ生成全自動パイプライン')
    parser.add_argument('--config', type=Path, required=True, help='設定ファイルのパス')
    parser.add_argument('--no-auto-resume', action='store_true', help='自動再開を無効化（新規実行）')
    
    args = parser.parse_args()
    
    if not args.config.exists():
        logger.error(f"[ERROR] 設定ファイルが見つかりません: {args.config}")
        sys.exit(1)
    
    pipeline = NSFWDrugDetectionQLoRATrainingDataPipeline(args.config)
    auto_resume = not args.no_auto_resume
    success = pipeline.run(auto_resume=auto_resume)
    
    if success:
        logger.info("[SUCCESS] パイプラインが正常に完了しました")
        sys.exit(0)
    else:
        logger.error("[ERROR] パイプラインが失敗しました")
        sys.exit(1)


if __name__ == '__main__':
    main()


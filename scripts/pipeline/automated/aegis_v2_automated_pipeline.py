#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AEGIS v2.0 全自動パイプライン（電源断リカバリー機能付き）

実行フロー:
1. Codex/Gemini CLIでDeep Researchデータ生成（日本のドメイン別知識優先）
2. データクレンジング
3. SO8T/thinking PPOモデルの学習
4. AEGIS v2.0として統合・保存

特徴:
- 3分間隔のチェックポイント自動保存
- 電源断からの自動リカバリー
- 電源投入時の自動再開
"""

import os
import sys
import json
import logging
import argparse
import subprocess
import threading
import time
import signal
import atexit
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ロギング設定
log_dir = PROJECT_ROOT / "logs"
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / "aegis_v2_pipeline.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# チェックポイント設定
CHECKPOINT_INTERVAL = 180  # 3分（秒）
MAX_CHECKPOINTS = 10  # 最大チェックポイント数
CHECKPOINT_DIR = Path("D:/webdataset/checkpoints/aegis_v2_pipeline")
SESSION_FILE = CHECKPOINT_DIR / "session.json"


class PipelineStage(Enum):
    """パイプラインステージ"""
    INIT = "init"
    DEEP_RESEARCH_DATA = "deep_research_data"
    DATA_CLEANSING = "data_cleansing"
    SO8T_PPO_TRAINING = "so8t_ppo_training"
    AEGIS_V2_INTEGRATION = "aegis_v2_integration"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class PipelineState:
    """パイプライン状態"""
    session_id: str
    stage: PipelineStage
    started_at: str
    last_checkpoint: float
    checkpoints: List[str]
    progress: Dict[str, Any]
    output_files: Dict[str, str]
    error: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """辞書に変換"""
        return {
            **asdict(self),
            "stage": self.stage.value
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'PipelineState':
        """辞書から復元"""
        data = data.copy()
        data["stage"] = PipelineStage(data["stage"])
        return cls(**data)


class PowerFailureRecovery:
    """電源断リカバリーシステム"""
    
    def __init__(
        self,
        checkpoint_dir: Path = CHECKPOINT_DIR,
        checkpoint_interval: float = CHECKPOINT_INTERVAL,
        max_checkpoints: int = MAX_CHECKPOINTS
    ):
        """初期化"""
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_interval = checkpoint_interval
        self.max_checkpoints = max_checkpoints
        
        self.session_file = self.checkpoint_dir / "session.json"
        self.state: Optional[PipelineState] = None
        
        # チェックポイントスレッド
        self.checkpoint_thread: Optional[threading.Thread] = None
        self.running = False
        
        # シグナルハンドラー設定
        self._setup_signal_handlers()
        
        logger.info("="*80)
        logger.info("Power Failure Recovery System Initialized")
        logger.info("="*80)
        logger.info(f"Checkpoint directory: {self.checkpoint_dir}")
        logger.info(f"Checkpoint interval: {checkpoint_interval} seconds ({checkpoint_interval/60:.1f} minutes)")
        logger.info(f"Max checkpoints: {max_checkpoints}")
    
    def _setup_signal_handlers(self):
        """シグナルハンドラー設定"""
        def signal_handler(signum, frame):
            logger.warning(f"[SIGNAL] Received signal {signum}, saving checkpoint...")
            self.save_checkpoint("emergency")
            logger.info("[SIGNAL] Emergency checkpoint saved. Exiting...")
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        atexit.register(self._cleanup)
    
    def _cleanup(self):
        """クリーンアップ"""
        if self.running:
            logger.info("[CLEANUP] Saving final checkpoint...")
            self.save_checkpoint("final")
            self.running = False
    
    def create_session(self, session_id: Optional[str] = None) -> PipelineState:
        """セッション作成"""
        if session_id is None:
            session_id = f"aegis_v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.state = PipelineState(
            session_id=session_id,
            stage=PipelineStage.INIT,
            started_at=datetime.now().isoformat(),
            last_checkpoint=time.time(),
            checkpoints=[],
            progress={},
            output_files={}
        )
        
        self.save_session()
        logger.info(f"[SESSION] Created session: {session_id}")
        return self.state
    
    def load_session(self) -> Optional[PipelineState]:
        """セッション復旧"""
        if not self.session_file.exists():
            logger.info("[RECOVERY] No session file found, starting fresh")
            return None
        
        try:
            with open(self.session_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.state = PipelineState.from_dict(data)
            logger.info("="*80)
            logger.info("[RECOVERY] Session restored from checkpoint")
            logger.info("="*80)
            logger.info(f"Session ID: {self.state.session_id}")
            logger.info(f"Stage: {self.state.stage.value}")
            logger.info(f"Started at: {self.state.started_at}")
            logger.info(f"Progress: {self.state.progress}")
            logger.info(f"Output files: {self.state.output_files}")
            
            if self.state.error:
                logger.warning(f"Previous error: {self.state.error}")
            
            return self.state
        except Exception as e:
            logger.error(f"[ERROR] Failed to load session: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None
    
    def save_session(self):
        """セッション保存"""
        if not self.state:
            return
        
        try:
            self.session_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.session_file, 'w', encoding='utf-8') as f:
                json.dump(self.state.to_dict(), f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"[ERROR] Failed to save session: {e}")
    
    def save_checkpoint(self, suffix: str = ""):
        """チェックポイント保存"""
        if not self.state:
            return
        
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            checkpoint_path = self.checkpoint_dir / f"checkpoint_{self.state.session_id}_{timestamp}_{suffix}.json"
            
            checkpoint_data = {
                "session_id": self.state.session_id,
                "stage": self.state.stage.value,
                "timestamp": timestamp,
                "progress": self.state.progress,
                "output_files": self.state.output_files,
                "checkpoint_time": time.time()
            }
            
            with open(checkpoint_path, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
            
            # チェックポイントリスト更新
            self.state.checkpoints.append(str(checkpoint_path))
            
            # 古いチェックポイント削除
            if len(self.state.checkpoints) > self.max_checkpoints:
                old_checkpoint = Path(self.state.checkpoints.pop(0))
                if old_checkpoint.exists():
                    old_checkpoint.unlink()
                    logger.debug(f"[CLEANUP] Deleted old checkpoint: {old_checkpoint}")
            
            self.state.last_checkpoint = time.time()
            self.save_session()
            
            logger.info(f"[CHECKPOINT] Saved: {checkpoint_path.name}")
        except Exception as e:
            logger.error(f"[ERROR] Failed to save checkpoint: {e}")
    
    def start_checkpoint_thread(self):
        """チェックポイントスレッド開始"""
        if self.running:
            return
        
        self.running = True
        
        def checkpoint_worker():
            while self.running:
                time.sleep(self.checkpoint_interval)
                if self.running and self.state:
                    current_time = time.time()
                    if (current_time - self.state.last_checkpoint) >= self.checkpoint_interval:
                        self.save_checkpoint("auto")
        
        self.checkpoint_thread = threading.Thread(target=checkpoint_worker, daemon=True)
        self.checkpoint_thread.start()
        logger.info(f"[CHECKPOINT] Auto-checkpoint thread started (interval: {self.checkpoint_interval}s)")
    
    def stop_checkpoint_thread(self):
        """チェックポイントスレッド停止"""
        self.running = False
        if self.checkpoint_thread:
            self.checkpoint_thread.join(timeout=5)
        logger.info("[CHECKPOINT] Auto-checkpoint thread stopped")
    
    def update_progress(self, stage: PipelineStage, progress: Dict[str, Any], output_files: Optional[Dict[str, str]] = None):
        """進捗更新"""
        if not self.state:
            return
        
        self.state.stage = stage
        self.state.progress.update(progress)
        if output_files:
            self.state.output_files.update(output_files)
        
        self.save_session()
        logger.info(f"[PROGRESS] Stage: {stage.value}, Progress: {progress}")


class AEGISV2AutomatedPipeline:
    """AEGIS v2.0 全自動パイプライン"""
    
    def __init__(
        self,
        prompts_file: Path,
        config_file: Path,
        output_dir: Path,
        api_type: str = "openai",
        api_key: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
        auto_resume: bool = True
    ):
        """初期化"""
        self.prompts_file = Path(prompts_file)
        if not self.prompts_file.is_absolute():
            self.prompts_file = PROJECT_ROOT / self.prompts_file
        
        self.config_file = Path(config_file)
        if not self.config_file.is_absolute():
            self.config_file = PROJECT_ROOT / self.config_file
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.api_type = api_type
        self.api_key = api_key or os.environ.get(f"{api_type.upper()}_API_KEY")
        self.gemini_api_key = gemini_api_key or os.environ.get("GEMINI_API_KEY")
        
        # リカバリーシステム初期化
        self.recovery = PowerFailureRecovery()
        
        # セッション復旧または新規作成
        if auto_resume:
            self.state = self.recovery.load_session()
            if self.state:
                logger.info("[RESUME] Resuming from previous session")
            else:
                self.state = self.recovery.create_session()
        else:
            self.state = self.recovery.create_session()
        
        # チェックポイントスレッド開始
        self.recovery.start_checkpoint_thread()
        
        logger.info("="*80)
        logger.info("AEGIS v2.0 Automated Pipeline Initialized")
        logger.info("="*80)
        logger.info(f"Prompts file: {self.prompts_file}")
        logger.info(f"Config file: {self.config_file}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Current stage: {self.state.stage.value}")
    
    def run_step1_deep_research_data(self) -> Path:
        """Step 1: Deep Researchデータ生成"""
        logger.info("="*80)
        logger.info("STEP 1: Deep Research Data Generation")
        logger.info("="*80)
        
        # 既に完了している場合はスキップ
        if self.state.progress.get("step1_completed", False):
            output_file = Path(self.state.output_files.get("deep_research_data"))
            if output_file and output_file.exists():
                logger.info(f"[SKIP] Step 1 already completed: {output_file}")
                return output_file
        
        # 既に完了している場合はスキップ
        if self.state.stage.value in [PipelineStage.DATA_CLEANSING.value, PipelineStage.SO8T_PPO_TRAINING.value, PipelineStage.AEGIS_V2_INTEGRATION.value, PipelineStage.COMPLETED.value]:
            output_file = Path(self.state.output_files.get("deep_research_data"))
            if output_file and output_file.exists():
                logger.info(f"[SKIP] Step 1 already completed: {output_file}")
                return output_file
        
        output_file = self.output_dir / "deep_research_thinking_dataset.jsonl"

        # 既存ファイルがある場合は再生成せずに再利用
        if output_file.exists() and output_file.stat().st_size > 0:
            logger.info(f"[SKIP] Step 1 dataset already exists: {output_file}")
            self.recovery.update_progress(
                PipelineStage.DEEP_RESEARCH_DATA,
                {"step1_completed": True, "output_file": str(output_file)},
                {"deep_research_data": str(output_file)},
            )
            return output_file
        
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "data" / "create_deep_research_thinking_dataset.py"),
            "--queries-file", str(self.prompts_file),
            "--output-file", str(output_file),
            "--use-codex",
            "--use-gemini",
            "--codex-api-type", self.api_type
        ]
        
        if self.api_key:
            os.environ[f"{self.api_type.upper()}_API_KEY"] = self.api_key
        if self.gemini_api_key:
            os.environ["GEMINI_API_KEY"] = self.gemini_api_key
        
        logger.info(f"[CMD] Running: {' '.join(cmd)}")
        # バッファリングを無効化してリアルタイムログ出力
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,  # 行バッファリング
            env=dict(os.environ, PYTHONUNBUFFERED="1"),  # Pythonのバッファリング無効化
        )
        
        # 標準出力と標準エラーをログに記録
        if result.stdout:
            logger.info(f"[STDOUT] {result.stdout}")
        if result.stderr:
            logger.warning(f"[STDERR] {result.stderr}")
        
        if result.returncode != 0:
            error_msg = f"Step 1 failed: {result.stderr}"
            logger.error(f"[ERROR] {error_msg}")
            self.state.error = error_msg
            self.state.stage = PipelineStage.FAILED
            self.recovery.save_session()
            raise RuntimeError(error_msg)
        
        self.recovery.update_progress(
            PipelineStage.DEEP_RESEARCH_DATA,
            {"step1_completed": True, "output_file": str(output_file)},
            {"deep_research_data": str(output_file)}
        )
        
        logger.info(f"[SUCCESS] Step 1 completed: {output_file}")
        return output_file
    
    def run_step2_data_cleansing(self, dataset_path: Path) -> Path:
        """Step 2: データクレンジング"""
        logger.info("="*80)
        logger.info("STEP 2: Data Cleansing")
        logger.info("="*80)
        
        # 既に完了している場合はスキップ
        if self.state.progress.get("step2_completed", False):
            cleansed_file = Path(self.state.output_files.get("cleansed_data"))
            if cleansed_file and cleansed_file.exists():
                logger.info(f"[SKIP] Step 2 already completed: {cleansed_file}")
                return cleansed_file
        
        # 既に完了している場合はスキップ
        if self.state.stage.value in [PipelineStage.SO8T_PPO_TRAINING.value, PipelineStage.AEGIS_V2_INTEGRATION.value, PipelineStage.COMPLETED.value]:
            cleansed_file = Path(self.state.output_files.get("cleansed_data"))
            if cleansed_file and cleansed_file.exists():
                logger.info(f"[SKIP] Step 2 already completed: {cleansed_file}")
                return cleansed_file
        
        cleansed_file = dataset_path.parent / f"{dataset_path.stem}_cleansed{dataset_path.suffix}"
        
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "data" / "cleanse_codex_pairwise_dataset.py"),
            "--dataset", str(dataset_path),
            "--min-quality-score", "0.7",
            "--balance-classes",
            "--remove-outliers"
        ]
        
        logger.info(f"[CMD] Running: {' '.join(cmd)}")
        # バッファリングを無効化してリアルタイムログ出力
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,  # 行バッファリング
            env=dict(os.environ, PYTHONUNBUFFERED="1"),  # Pythonのバッファリング無効化
        )
        
        # 標準出力と標準エラーをログに記録
        if result.stdout:
            logger.info(f"[STDOUT] {result.stdout}")
        if result.stderr:
            logger.warning(f"[STDERR] {result.stderr}")
        
        if result.returncode != 0:
            error_msg = f"Step 2 failed: {result.stderr}"
            logger.error(f"[ERROR] {error_msg}")
            self.state.error = error_msg
            self.state.stage = PipelineStage.FAILED
            self.recovery.save_session()
            raise RuntimeError(error_msg)
        
        self.recovery.update_progress(
            PipelineStage.DATA_CLEANSING,
            {"step2_completed": True, "cleansed_file": str(cleansed_file)},
            {"cleansed_data": str(cleansed_file)}
        )
        
        logger.info(f"[SUCCESS] Step 2 completed: {cleansed_file}")
        return cleansed_file
    
    def run_step3_so8t_ppo_training(self, dataset_path: Path = None) -> Path:
        """Step 3: SO8T PPO学習"""
        logger.info("="*80)
        logger.info("STEP 3: SO8T Quadruple PPO Training")
        logger.info("="*80)
        
        # 既に完了している場合はスキップ
        if self.state.stage.value in [PipelineStage.AEGIS_V2_INTEGRATION.value, PipelineStage.COMPLETED.value]:
            model_dir = Path(self.state.output_files.get("so8t_ppo_model"))
            if model_dir and model_dir.exists():
                logger.info(f"[SKIP] Step 3 already completed: {model_dir}")
                return model_dir
        
        # データセットパスが指定されていない、または空の場合は既存のペア比較データセットを使用
        if not dataset_path or not dataset_path.exists() or dataset_path.stat().st_size == 0:
            existing_pairwise = self.output_dir / "pairwise_dataset.jsonl"
            if existing_pairwise.exists() and existing_pairwise.stat().st_size > 0:
                logger.info(f"[INFO] Using existing pairwise dataset: {existing_pairwise}")
                dataset_path = existing_pairwise
            else:
                error_msg = f"Dataset file not found or empty: {dataset_path}"
                logger.error(f"[ERROR] {error_msg}")
                self.state.error = error_msg
                self.state.stage = PipelineStage.FAILED
                self.recovery.save_session()
                raise RuntimeError(error_msg)
        
        model_dir = self.output_dir / "so8t_ppo_model"
        model_dir.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "training" / "train_so8t_quadruple_ppo.py"),
            "--config", str(self.config_file),
            "--dataset", str(dataset_path),
            "--output-dir", str(model_dir),
            "--auto-resume"
        ]
        
        logger.info(f"[CMD] Running: {' '.join(cmd)}")
        logger.info(f"[INFO] Training log will be written to: logs/train_so8t_quadruple_ppo.log")
        logger.info(f"[INFO] This step may take a long time. Monitor the training log for progress.")
        
        # バッファリングを無効化してリアルタイムログ出力
        env = dict(os.environ)
        env['PYTHONUNBUFFERED'] = '1'
        
        # リアルタイムログ出力のためにPopenを使用
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
        )
        
        # リアルタイムでログを出力
        output_lines = []
        for line in process.stdout:
            line = line.rstrip()
            output_lines.append(line)
            logger.info(f"[TRAINING] {line}")
        
        # プロセス完了を待機
        returncode = process.wait()
        
        if returncode != 0:
            error_msg = f"Step 3 failed with return code {returncode}"
            logger.error(f"[ERROR] {error_msg}")
            logger.error(f"[ERROR] Last 20 lines of output:")
            for line in output_lines[-20:]:
                logger.error(f"[ERROR] {line}")
            self.state.error = error_msg
            self.state.stage = PipelineStage.FAILED
            self.recovery.save_session()
            raise RuntimeError(error_msg)
        
        final_model_dir = model_dir / "final_model"
        
        self.recovery.update_progress(
            PipelineStage.SO8T_PPO_TRAINING,
            {"step3_completed": True, "model_dir": str(final_model_dir)},
            {"so8t_ppo_model": str(final_model_dir)}
        )
        
        logger.info(f"[SUCCESS] Step 3 completed: {final_model_dir}")
        return final_model_dir
    
    def run_step4_aegis_v2_integration(self, model_dir: Path) -> Path:
        """Step 4: AEGIS v2.0統合"""
        logger.info("="*80)
        logger.info("STEP 4: AEGIS v2.0 Integration")
        logger.info("="*80)
        
        aegis_v2_dir = self.output_dir / "aegis_v2.0"
        aegis_v2_dir.mkdir(parents=True, exist_ok=True)
        
        # モデルファイルをコピー
        import shutil
        if model_dir.exists():
            shutil.copytree(model_dir, aegis_v2_dir / "model", dirs_exist_ok=True)
        
        # AEGIS v2.0メタデータ作成
        metadata = {
            "model_name": "AEGIS-v2.0",
            "version": "2.0",
            "base_model": "Borea-Phi3.5-instinct-jp",
            "architecture": "SO8T Quadruple PPO",
            "created_at": datetime.now().isoformat(),
            "session_id": self.state.session_id,
            "training_config": str(self.config_file),
            "features": [
                "SO(8) Rotation Gates",
                "Quadruple Thinking (Task/Safety/Policy/Final)",
                "Four-Class Classification",
                "QLoRA Weight Freezing",
                "Japanese Domain Knowledge Priority",
                "Deep Research Data Generation"
            ]
        }
        
        metadata_file = aegis_v2_dir / "metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        self.recovery.update_progress(
            PipelineStage.AEGIS_V2_INTEGRATION,
            {"step4_completed": True, "aegis_v2_dir": str(aegis_v2_dir)},
            {"aegis_v2.0": str(aegis_v2_dir)}
        )
        
        logger.info(f"[SUCCESS] Step 4 completed: {aegis_v2_dir}")
        return aegis_v2_dir
    
    def run_full_pipeline(self) -> Dict[str, Any]:
        """全パイプライン実行"""
        logger.info("="*80)
        logger.info("FULL PIPELINE: AEGIS v2.0 Creation")
        logger.info("="*80)
        
        try:
            # Step 1: Deep Researchデータ生成
            deep_research_data = self.run_step1_deep_research_data()
            
            # Step 2: データクレンジング
            cleansed_data = self.run_step2_data_cleansing(deep_research_data)
            
            # Step 3: SO8T PPO学習（既存のペア比較データセットを使用）
            # クレンジング済みデータが空の場合は、既存のpairwise_dataset.jsonlを使用
            pairwise_dataset = self.output_dir / "pairwise_dataset.jsonl"
            if pairwise_dataset.exists() and pairwise_dataset.stat().st_size > 0:
                logger.info(f"[INFO] Using existing pairwise dataset for Step 3: {pairwise_dataset}")
                so8t_model = self.run_step3_so8t_ppo_training(pairwise_dataset)
            else:
                so8t_model = self.run_step3_so8t_ppo_training(cleansed_data)
            
            # Step 4: AEGIS v2.0統合
            aegis_v2 = self.run_step4_aegis_v2_integration(so8t_model)
            
            # 完了
            self.state.stage = PipelineStage.COMPLETED
            self.state.error = None
            self.recovery.save_session()
            self.recovery.stop_checkpoint_thread()
            
            results = {
                "status": "success",
                "session_id": self.state.session_id,
                "aegis_v2_dir": str(aegis_v2),
                "output_files": self.state.output_files,
                "completed_at": datetime.now().isoformat()
            }
            
            logger.info("="*80)
            logger.info("[SUCCESS] AEGIS v2.0 pipeline completed successfully!")
            logger.info("="*80)
            logger.info(f"AEGIS v2.0 directory: {aegis_v2}")
            
            return results
            
        except Exception as e:
            logger.error(f"[ERROR] Pipeline failed: {e}")
            self.state.stage = PipelineStage.FAILED
            self.state.error = str(e)
            self.recovery.save_session()
            self.recovery.stop_checkpoint_thread()
            raise


def main():
    parser = argparse.ArgumentParser(
        description="AEGIS v2.0 Automated Pipeline with Power Failure Recovery"
    )
    parser.add_argument(
        "--prompts-file",
        type=Path,
        required=True,
        help="Prompts file path (JSONL format)"
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Training config file path (YAML)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("D:/webdataset/aegis_v2.0"),
        help="Output directory (default: D:/webdataset/aegis_v2.0)"
    )
    parser.add_argument(
        "--api-type",
        type=str,
        choices=["openai", "claude"],
        default="openai",
        help="API type (default: openai)"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key (if not provided, uses environment variable)"
    )
    parser.add_argument(
        "--gemini-api-key",
        type=str,
        default=None,
        help="Gemini API key (if not provided, uses environment variable)"
    )
    parser.add_argument(
        "--no-auto-resume",
        action="store_true",
        help="Do not auto-resume from checkpoint"
    )
    
    args = parser.parse_args()
    
    # パイプライン実行
    pipeline = AEGISV2AutomatedPipeline(
        prompts_file=args.prompts_file,
        config_file=args.config,
        output_dir=args.output_dir,
        api_type=args.api_type,
        api_key=args.api_key,
        gemini_api_key=args.gemini_api_key,
        auto_resume=not args.no_auto_resume
    )
    
    results = pipeline.run_full_pipeline()
    
    # 結果を保存
    results_file = args.output_dir / "pipeline_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"[SAVE] Pipeline results saved to {results_file}")


if __name__ == "__main__":
    main()


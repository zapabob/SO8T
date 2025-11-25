#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
学習ログ監視とパイプライン再開スクリプト

学習ログを監視し、完了を検出したらパイプラインを再開する。
"""

import os
import sys
import time
import logging
import subprocess
from pathlib import Path
from typing import Optional, List
from datetime import datetime

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 設定
TRAINING_LOG_FILE = PROJECT_ROOT / "logs" / "train_so8t_quadruple_ppo.log"
CHECKPOINT_DIR = Path("D:/webdataset/checkpoints/aegis_v2_pipeline")
SESSION_FILE = CHECKPOINT_DIR / "session.json"
PIPELINE_SCRIPT = PROJECT_ROOT / "scripts" / "pipelines" / "aegis_v2_automated_pipeline.py"

# 学習完了を示すキーワード
COMPLETION_KEYWORDS = [
    "Training completed",
    "Training finished",
    "Model saved",
    "SUCCESS",
    "completed successfully",
    "epoch",
    "loss",
    "saved checkpoint"
]

# エラーを示すキーワード
ERROR_KEYWORDS = [
    "ERROR",
    "Exception",
    "Traceback",
    "Failed",
    "failed",
    "Error"
]


def check_training_completion(log_file: Path) -> tuple[bool, Optional[str]]:
    """
    学習ログを確認して完了状態を判定
    
    Returns:
        (is_completed, status_message)
    """
    if not log_file.exists():
        return False, "Log file not found"
    
    try:
        # ログファイルの最後の100行を読み込む
        with open(log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            if not lines:
                return False, "Log file is empty"
            
            # 最後の100行を確認
            recent_lines = lines[-100:]
            recent_text = '\n'.join(recent_lines).lower()
            
            # エラーチェック（最新の10行のみを確認）
            recent_10_lines = lines[-10:]
            recent_10_text = '\n'.join(recent_10_lines).lower()
            
            # 最新のエラーを確認（インポート成功メッセージがある場合は無視）
            if "successfully imported" in recent_10_text:
                # インポート成功後はインポートエラーを無視
                for error_keyword in ERROR_KEYWORDS:
                    if error_keyword.lower() in recent_10_text:
                        # インポート関連のエラーは無視
                        if "import" in error_keyword.lower():
                            continue
                        # 最新のエラーを確認
                        for line in reversed(recent_10_lines):
                            if error_keyword.lower() in line.lower() and "import" not in line.lower():
                                return False, f"Error detected: {line.strip()}"
            else:
                # インポート成功前はすべてのエラーを確認
                for error_keyword in ERROR_KEYWORDS:
                    if error_keyword.lower() in recent_10_text:
                        for line in reversed(recent_10_lines):
                            if error_keyword.lower() in line.lower():
                                return False, f"Error detected: {line.strip()}"
            
            # 完了キーワードチェック
            for keyword in COMPLETION_KEYWORDS:
                if keyword.lower() in recent_text:
                    # 最新の完了メッセージを確認
                    for line in reversed(recent_lines):
                        if keyword.lower() in line.lower():
                            return True, f"Training completed: {line.strip()}"
            
            # モデル出力ディレクトリの確認
            model_dir = Path("D:/webdataset/aegis_v2.0/so8t_ppo_model")
            if model_dir.exists():
                # モデルファイルが存在するか確認
                model_files = list(model_dir.glob("*.pt")) + list(model_dir.glob("*.safetensors"))
                if model_files:
                    return True, f"Model files found in {model_dir}"
            
            # ログファイルが最近更新されているか確認
            log_mtime = log_file.stat().st_mtime
            current_time = time.time()
            time_since_update = current_time - log_mtime
            
            # 5分以上更新されていない場合は完了とみなす（ただし、エラーがない場合のみ）
            if time_since_update > 300:  # 5分
                return True, f"Log file not updated for {int(time_since_update/60)} minutes (assuming completed)"
            
            return False, f"Training in progress (last update: {int(time_since_update)}s ago)"
            
    except Exception as e:
        return False, f"Error reading log file: {e}"


def check_pipeline_state() -> dict:
    """パイプラインの状態を確認"""
    if not SESSION_FILE.exists():
        return {"exists": False}
    
    try:
        import json
        with open(SESSION_FILE, 'r', encoding='utf-8') as f:
            state = json.load(f)
        return {"exists": True, "state": state}
    except Exception as e:
        return {"exists": True, "error": str(e)}


def resume_pipeline() -> bool:
    """パイプラインを再開"""
    # セッションファイルから設定を読み込む
    if not SESSION_FILE.exists():
        logger.error("[ERROR] Session file not found. Cannot resume pipeline.")
        return False
    
    try:
        import json
        with open(SESSION_FILE, 'r', encoding='utf-8') as f:
            state = json.load(f)
        
        # 設定ファイルとプロンプトファイルのパスを取得
        config_file = Path(state.get("progress", {}).get("config_file", "configs/train_borea_phi35_so8t_thinking_frozen.yaml"))
        prompts_file = Path(state.get("progress", {}).get("prompts_file", "scripts/pipelines/thinking_prompts.txt"))
        output_dir = Path(state.get("output_files", {}).get("output_dir", "D:/webdataset/aegis_v2.0"))
        
        # パイプラインスクリプトを実行
        cmd = [
            sys.executable,
            str(PIPELINE_SCRIPT),
            "--prompts-file", str(prompts_file),
            "--config", str(config_file),
            "--output-dir", str(output_dir),
            "--auto-resume"
        ]
        
        logger.info(f"[RESUME] Resuming pipeline with command: {' '.join(cmd)}")
        
        # パイプラインを実行（バックグラウンドで実行）
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            bufsize=1
        )
        
        logger.info(f"[RESUME] Pipeline process started (PID: {process.pid})")
        return True
        
    except Exception as e:
        logger.error(f"[ERROR] Failed to resume pipeline: {e}")
        return False


def monitor_training(interval: int = 60, max_checks: Optional[int] = None) -> bool:
    """
    学習ログを監視
    
    Args:
        interval: チェック間隔（秒）
        max_checks: 最大チェック回数（Noneの場合は無制限）
    
    Returns:
        学習が完了したかどうか
    """
    logger.info("="*80)
    logger.info("Training Log Monitor")
    logger.info("="*80)
    logger.info(f"Log file: {TRAINING_LOG_FILE}")
    logger.info(f"Check interval: {interval}s")
    logger.info("="*80)
    
    check_count = 0
    
    while True:
        check_count += 1
        
        # 最大チェック回数に達した場合
        if max_checks and check_count > max_checks:
            logger.warning(f"[WARNING] Maximum check count ({max_checks}) reached. Stopping monitor.")
            return False
        
        # 学習完了を確認
        is_completed, status = check_training_completion(TRAINING_LOG_FILE)
        
        logger.info(f"[CHECK {check_count}] {status}")
        
        if is_completed:
            logger.info("="*80)
            logger.info("[SUCCESS] Training completed!")
            logger.info("="*80)
            return True
        
        # 次のチェックまで待機
        time.sleep(interval)


def main():
    """メイン処理"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Monitor training log and resume pipeline when complete"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Check interval in seconds (default: 60)"
    )
    parser.add_argument(
        "--max-checks",
        type=int,
        default=None,
        help="Maximum number of checks (default: unlimited)"
    )
    parser.add_argument(
        "--auto-resume",
        action="store_true",
        help="Automatically resume pipeline when training completes"
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check training status, do not monitor"
    )
    
    args = parser.parse_args()
    
    # チェックのみの場合
    if args.check_only:
        is_completed, status = check_training_completion(TRAINING_LOG_FILE)
        logger.info(f"[STATUS] {status}")
        
        if is_completed:
            logger.info("[INFO] Training is completed.")
            if args.auto_resume:
                logger.info("[INFO] Resuming pipeline...")
                resume_pipeline()
        else:
            logger.info("[INFO] Training is still in progress.")
        
        return
    
    # 監視モード
    is_completed = monitor_training(
        interval=args.interval,
        max_checks=args.max_checks
    )
    
    # 学習完了後、パイプラインを再開
    if is_completed and args.auto_resume:
        logger.info("[INFO] Resuming pipeline...")
        resume_pipeline()
    elif is_completed:
        logger.info("[INFO] Training completed. Use --auto-resume to resume pipeline.")


if __name__ == "__main__":
    main()


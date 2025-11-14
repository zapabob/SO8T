#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
時間ベースチェックポイントCallback

約 fixed_interval_sec ごとに Trainer のチェックポイント保存処理を実行する Callback。
Hugging Face Trainer の checkpoint-XXXX 形式をそのまま使うため、
Trainer._save_checkpoint(...) を呼び出す。

注意: _save_checkpoint は内部APIのため、transformers のバージョンが変わったら
壊れる可能性がある。
"""

import time
import logging
from pathlib import Path
from typing import Optional

from transformers import TrainerCallback, TrainerState, TrainerControl, TrainingArguments
from transformers.trainer import Trainer

logger = logging.getLogger(__name__)


class TimeBasedCheckpointCallback(TrainerCallback):
    """
    約 fixed_interval_sec ごとに Trainer のチェックポイント保存処理を実行する Callback。
    Hugging Face Trainer の checkpoint-XXXX 形式をそのまま使うため、
    Trainer._save_checkpoint(...) を呼び出す。
    """

    def __init__(self, fixed_interval_sec: int = 180, max_retries: int = 3, retry_delay_sec: float = 1.0):
        """
        初期化
        
        Args:
            fixed_interval_sec: チェックポイント保存間隔（秒）。デフォルトは180秒（3分）
            max_retries: チェックポイント保存失敗時の最大リトライ回数。デフォルトは3回
            retry_delay_sec: リトライ間の待機時間（秒）。デフォルトは1.0秒
        """
        self.fixed_interval_sec = fixed_interval_sec
        self._last_save_time: Optional[float] = None
        self.max_retries = max_retries
        self.retry_delay_sec = retry_delay_sec
        self._save_failures = 0  # 連続失敗回数を記録

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        学習開始時に呼ばれる。初期時刻を記録する。
        """
        self._last_save_time = time.time()
        logger.info(f"[TIMEBASED_CHECKPOINT] ================================================================================")
        logger.info(f"[TIMEBASED_CHECKPOINT] Training started. Checkpoints will be saved every {self.fixed_interval_sec}s ({self.fixed_interval_sec/60:.1f} minutes)")
        logger.info(f"[TIMEBASED_CHECKPOINT] Initial last_save_time: {self._last_save_time}")
        logger.info(f"[TIMEBASED_CHECKPOINT] First checkpoint will be saved after {self.fixed_interval_sec}s")
        logger.info(f"[TIMEBASED_CHECKPOINT] ================================================================================")
        return control

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        各ステップ終了時に呼ばれる。経過時間をチェックし、
        fixed_interval_sec 以上経過していたらチェックポイントを保存する。
        """
        if self._last_save_time is None:
            self._last_save_time = time.time()
            logger.debug(f"[TIMEBASED_CHECKPOINT] Initialized last_save_time at step {state.global_step}")
            return control

        now = time.time()
        elapsed = now - self._last_save_time
        
        # デバッグログ（最初の数ステップのみ、または10ステップごと）
        if state.global_step <= 10 or state.global_step % 10 == 0:
            logger.info(f"[TIMEBASED_CHECKPOINT] Step {state.global_step}: elapsed={elapsed:.1f}s, interval={self.fixed_interval_sec}s, remaining={self.fixed_interval_sec - elapsed:.1f}s")
        
        if elapsed < self.fixed_interval_sec:
            return control

        # Trainer インスタンスを取得
        trainer: Trainer = kwargs.get("trainer")
        if trainer is None:
            logger.error("[TIMEBASED_CHECKPOINT] Trainer instance not found in kwargs")
            return control

        # チェックポイント保存前のステップ数を記録
        current_step = state.global_step
        logger.info(f"[TIMEBASED_CHECKPOINT] Time interval reached ({elapsed:.1f}s >= {self.fixed_interval_sec}s). Saving checkpoint at step {current_step}...")
        
        # チェックポイントディレクトリの確認
        checkpoint_dir = Path(args.output_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"[TIMEBASED_CHECKPOINT] Checkpoint directory: {checkpoint_dir}")
        
        # Trainer の内部メソッドを使って checkpoint-XXXX を作成
        # trial=None, metrics=None で呼び出す
        # 注意: _save_checkpoint は内部APIのため、transformers のバージョンが
        # 変わったら壊れる可能性がある
        
        checkpoint_path = checkpoint_dir / f"checkpoint-{current_step}"
        save_success = False
        
        # リトライループ
        for attempt in range(self.max_retries):
            try:
                # チェックポイント保存を実行
                trainer._save_checkpoint(model=trainer.model, trial=None, metrics=None)  # type: ignore
                
                # チェックポイントが保存されたか確認
                # 少し待ってから確認（非同期保存の可能性があるため）
                import time as time_module
                time_module.sleep(0.5)
                
                if checkpoint_path.exists():
                    # チェックポイントファイルのサイズを確認
                    try:
                        checkpoint_size = sum(f.stat().st_size for f in checkpoint_path.rglob('*') if f.is_file())
                        checkpoint_size_mb = checkpoint_size / (1024 * 1024)
                        
                        # 保存成功
                        save_success = True
                        self._save_failures = 0  # 失敗カウントをリセット
                        
                        logger.info(f"[TIMEBASED_CHECKPOINT] Successfully saved checkpoint at step {current_step} "
                                  f"(elapsed: {elapsed:.1f}s, interval: {self.fixed_interval_sec}s, size: {checkpoint_size_mb:.2f}MB)")
                        logger.info(f"[TIMEBASED_CHECKPOINT] Checkpoint path: {checkpoint_path}")
                        
                        # 次の保存予定時刻を計算
                        next_save_time = now + self.fixed_interval_sec
                        from datetime import datetime
                        next_save_datetime = datetime.fromtimestamp(next_save_time)
                        logger.info(f"[TIMEBASED_CHECKPOINT] Next checkpoint scheduled at: {next_save_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
                        break  # 成功したらループを抜ける
                    except Exception as size_error:
                        logger.warning(f"[TIMEBASED_CHECKPOINT] Failed to get checkpoint size: {size_error}")
                        # ディレクトリは存在するので、保存は成功したとみなす
                        save_success = True
                        self._save_failures = 0
                        logger.info(f"[TIMEBASED_CHECKPOINT] Checkpoint saved at step {current_step} (size check failed)")
                        break
                else:
                    # チェックポイントが存在しない
                    if attempt < self.max_retries - 1:
                        logger.warning(f"[TIMEBASED_CHECKPOINT] Checkpoint not found after save (attempt {attempt + 1}/{self.max_retries}). Retrying...")
                        time_module.sleep(self.retry_delay_sec)
                    else:
                        logger.warning(f"[TIMEBASED_CHECKPOINT] Checkpoint save attempted but directory not found after {self.max_retries} attempts: {checkpoint_path}")
                        logger.warning(f"[TIMEBASED_CHECKPOINT] Output directory exists: {checkpoint_dir.exists()}")
                        logger.warning(f"[TIMEBASED_CHECKPOINT] Output directory contents: {list(checkpoint_dir.iterdir()) if checkpoint_dir.exists() else 'N/A'}")
                        
            except AttributeError as e:
                logger.error(f"[TIMEBASED_CHECKPOINT] _save_checkpoint method not found. Transformers version may have changed: {e}")
                logger.error(f"[TIMEBASED_CHECKPOINT] Available methods: {[m for m in dir(trainer) if 'checkpoint' in m.lower()]}")
                # AttributeErrorはリトライしても解決しないので、ループを抜ける
                break
            except Exception as e:
                self._save_failures += 1
                if attempt < self.max_retries - 1:
                    logger.warning(f"[TIMEBASED_CHECKPOINT] Failed to save checkpoint at step {current_step} (attempt {attempt + 1}/{self.max_retries}): {e}")
                    logger.warning(f"[TIMEBASED_CHECKPOINT] Retrying in {self.retry_delay_sec}s...")
                    time_module.sleep(self.retry_delay_sec)
                else:
                    logger.error(f"[TIMEBASED_CHECKPOINT] Failed to save checkpoint at step {current_step} after {self.max_retries} attempts: {e}", exc_info=True)
                    logger.error(f"[TIMEBASED_CHECKPOINT] Consecutive save failures: {self._save_failures}")
        
        # 最終的な失敗チェック
        if not save_success:
            logger.error(f"[TIMEBASED_CHECKPOINT] CRITICAL: Checkpoint save failed at step {current_step} after all retry attempts")
            logger.error(f"[TIMEBASED_CHECKPOINT] Total consecutive failures: {self._save_failures}")

        # 保存時刻を更新
        self._last_save_time = now

        return control


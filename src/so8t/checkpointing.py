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
from typing import Optional

from transformers import TrainerCallback, TrainerState, TrainerControl, TrainingArguments
from transformers.trainer import Trainer


class TimeBasedCheckpointCallback(TrainerCallback):
    """
    約 fixed_interval_sec ごとに Trainer のチェックポイント保存処理を実行する Callback。
    Hugging Face Trainer の checkpoint-XXXX 形式をそのまま使うため、
    Trainer._save_checkpoint(...) を呼び出す。
    """

    def __init__(self, fixed_interval_sec: int = 180):
        """
        初期化
        
        Args:
            fixed_interval_sec: チェックポイント保存間隔（秒）。デフォルトは180秒（3分）
        """
        self.fixed_interval_sec = fixed_interval_sec
        self._last_save_time: Optional[float] = None

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
            return control

        now = time.time()
        elapsed = now - self._last_save_time
        if elapsed < self.fixed_interval_sec:
            return control

        # Trainer インスタンスを取得
        trainer: Trainer = kwargs["trainer"]

        # Trainer の内部メソッドを使って checkpoint-XXXX を作成
        # trial=None, metrics=None で呼び出す
        # 注意: _save_checkpoint は内部APIのため、transformers のバージョンが
        # 変わったら壊れる可能性がある
        trainer._save_checkpoint(model=trainer.model, trial=None, metrics=None)  # type: ignore

        # 保存時刻を更新
        self._last_save_time = now

        return control


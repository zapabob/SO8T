#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rolling Checkpoint Manager
ローリング・チェックポイント・マネージャー

3分ごとの自動保存 + 最新5個だけ残すローリングストック機能
停電・再起動時の自動復旧をサポート

著者: 峯岸亮 (SO8Tプロジェクト)
"""

import os
import time
import shutil
import glob
from datetime import datetime
from pathlib import Path
from typing import Optional, Union, Any


class RollingCheckpointManager:
    """
    ローリング・チェックポイント・マネージャー

    機能:
    - 3分ごとの自動保存
    - 最新5個だけ残すローリング削除
    - 電源復旧時の自動再開サポート
    - ディスク容量節約

    使用例:
        ckpt_manager = RollingCheckpointManager("checkpoints_aegis", max_keep=5, save_interval_sec=180)

        # 学習ループ内で
        if ckpt_manager.should_save():
            ckpt_manager.save_checkpoint(model, tokenizer, step_info=f"epoch_{epoch}")
    """

    def __init__(self,
                 base_dir: Union[str, Path],
                 max_keep: int = 5,
                 save_interval_sec: int = 180,  # 3分
                 enable_logging: bool = True):
        """
        初期化

        Args:
            base_dir: チェックポイント保存先ディレクトリ
            max_keep: 保持するチェックポイントの最大数
            save_interval_sec: 保存間隔（秒）
            enable_logging: ログ出力有効化
        """
        self.base_dir = Path(base_dir)
        self.max_keep = max_keep
        self.save_interval_sec = save_interval_sec
        self.enable_logging = enable_logging
        self.last_save_time = time.time()
        self.save_count = 0

        # 保存ディレクトリ作成
        self.base_dir.mkdir(parents=True, exist_ok=True)

        if self.enable_logging:
            print(f"[INFO] RollingCheckpointManager initialized:")
            print(f"   Directory: {self.base_dir}")
            print(f"   Max keep: {self.max_keep}")
            print(f"   Save interval: {self.save_interval_sec}s")

    def should_save(self) -> bool:
        """
        前回の保存から指定時間が経過したかチェック

        Returns:
            保存が必要ならTrue
        """
        elapsed = time.time() - self.last_save_time
        return elapsed >= self.save_interval_sec

    def save_checkpoint(self,
                       model: Any,
                       tokenizer: Any,
                       step_info: str = "auto",
                       extra_info: Optional[dict] = None) -> str:
        """
        モデルを保存し、古いチェックポイントをローリング削除

        Args:
            model: 保存するモデル
            tokenizer: 保存するトークナイザー
            step_info: ステップ情報 (例: "epoch_10", "step_1000")
            extra_info: 追加のメタデータ

        Returns:
            保存されたチェックポイントのパス
        """
        # タイムスタンプ生成
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = self.base_dir / f"ckpt_{timestamp}_{step_info}"

        if self.enable_logging:
            print(f"💾 Saving checkpoint #{self.save_count + 1}: {save_path} ...")

        try:
            # モデル保存
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)

            # メタデータ保存
            metadata = {
                'timestamp': timestamp,
                'step_info': step_info,
                'save_count': self.save_count + 1,
                'time_saved': time.time(),
                'extra_info': extra_info or {}
            }

            import json
            with open(save_path / 'checkpoint_metadata.json', 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

            self.last_save_time = time.time()
            self.save_count += 1

            if self.enable_logging:
                print(f"✅ Checkpoint saved successfully: {save_path}")

        except Exception as e:
            print(f"❌ Error saving checkpoint: {e}")
            raise

        # ローリングクリーンアップ
        self._cleanup_old_checkpoints()

        return str(save_path)

    def _cleanup_old_checkpoints(self):
        """最新N個以外を削除"""
        # ckpt_ で始まるフォルダを全取得
        checkpoints = list(self.base_dir.glob("ckpt_*"))

        if len(checkpoints) <= self.max_keep:
            return  # 削除する必要なし

        # 作成日時順にソート（新しいのが後ろ）
        checkpoints.sort(key=lambda x: x.stat().st_mtime)

        # 保持数を超えている場合、古いものから削除
        to_delete = checkpoints[: -self.max_keep]

        for ckpt in to_delete:
            if self.enable_logging:
                print(f"🗑️ Removing old checkpoint: {ckpt}")
            try:
                shutil.rmtree(ckpt)
            except Exception as e:
                print(f"⚠️ Error deleting {ckpt}: {e}")

    def get_latest_checkpoint(self) -> Optional[str]:
        """
        再開用に最新のチェックポイントパスを取得

        Returns:
            最新チェックポイントのパス、なければNone
        """
        checkpoints = list(self.base_dir.glob("ckpt_*"))

        if not checkpoints:
            return None

        # 最新のものを返す
        latest = max(checkpoints, key=lambda x: x.stat().st_mtime)
        return str(latest)

    def get_all_checkpoints(self) -> list[str]:
        """
        全てのチェックポイントを取得（作成日時順）

        Returns:
            チェックポイントパスのリスト（古い順）
        """
        checkpoints = list(self.base_dir.glob("ckpt_*"))
        checkpoints.sort(key=lambda x: x.stat().st_mtime)
        return [str(ckpt) for ckpt in checkpoints]

    def get_checkpoint_info(self, checkpoint_path: Union[str, Path]) -> dict:
        """
        チェックポイントのメタデータを取得

        Args:
            checkpoint_path: チェックポイントのパス

        Returns:
            メタデータ辞書
        """
        metadata_path = Path(checkpoint_path) / 'checkpoint_metadata.json'

        if not metadata_path.exists():
            return {}

        try:
            import json
            with open(metadata_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error reading metadata {metadata_path}: {e}")
            return {}

    def force_save_now(self, model: Any, tokenizer: Any, step_info: str = "forced") -> str:
        """
        時間に関係なく強制保存

        Args:
            model: 保存するモデル
            tokenizer: 保存するトークナイザー
            step_info: ステップ情報

        Returns:
            保存されたチェックポイントのパス
        """
        old_time = self.last_save_time
        self.last_save_time = 0  # 強制的に保存可能にする

        try:
            return self.save_checkpoint(model, tokenizer, step_info)
        finally:
            self.last_save_time = old_time  # 元に戻す


class EmergencyCheckpointManager:
    """
    緊急チェックポイントマネージャー
    SIGINT/SIGTERM/異常終了時の自動保存
    """

    def __init__(self, checkpoint_manager: RollingCheckpointManager):
        self.ckpt_manager = checkpoint_manager
        self.model = None
        self.tokenizer = None
        self.setup_signal_handlers()

    def setup_signal_handlers(self):
        """シグナルハンドラーの設定"""
        import signal

        def emergency_save(signum, frame):
            """緊急保存ハンドラー"""
            print(f"\n🚨 Emergency save triggered by signal {signum}")
            if self.model is not None and self.tokenizer is not None:
                try:
                    self.ckpt_manager.force_save_now(
                        self.model, self.tokenizer,
                        step_info=f"emergency_sig{signum}"
                    )
                    print("✅ Emergency save completed")
                except Exception as e:
                    print(f"❌ Emergency save failed: {e}")
            else:
                print("⚠️ No model/tokenizer available for emergency save")

        # Windows対応のシグナル
        try:
            signal.signal(signal.SIGINT, emergency_save)   # Ctrl+C
            signal.signal(signal.SIGTERM, emergency_save)  # 終了要求
            # Windows固有のシグナル（利用可能なら）
            if hasattr(signal, 'SIGBREAK'):
                signal.signal(signal.SIGBREAK, emergency_save)
        except (OSError, ValueError) as e:
            print(f"⚠️ Signal handler setup failed: {e}")

    def register_model(self, model: Any, tokenizer: Any):
        """
        緊急保存用のモデルを登録

        Args:
            model: モデル
            tokenizer: トークナイザー
        """
        self.model = model
        self.tokenizer = tokenizer
        print("🛡️ Emergency checkpoint system armed")


# テスト関数
def test_checkpoint_manager():
    """テスト関数"""
    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        # マネージャー作成
        manager = RollingCheckpointManager(
            base_dir=temp_dir,
            max_keep=3,
            save_interval_sec=1  # テスト用に1秒
        )

        # モックモデル（実際にはtorch.nn.Module）
        class MockModel:
            def save_pretrained(self, path):
                Path(path).mkdir(exist_ok=True)
                (Path(path) / 'model.bin').write_text('mock model')

        class MockTokenizer:
            def save_pretrained(self, path):
                (Path(path) / 'tokenizer.json').write_text('{"mock": "tokenizer"}')

        model = MockModel()
        tokenizer = MockTokenizer()

        print("=== Testing RollingCheckpointManager ===")

        # 複数回保存テスト
        for i in range(5):
            time.sleep(1.1)  # 保存間隔を超える
            if manager.should_save():
                path = manager.save_checkpoint(model, tokenizer, f"test_{i}")
                print(f"Saved: {path}")

        # チェックポイント一覧確認
        checkpoints = manager.get_all_checkpoints()
        print(f"All checkpoints: {len(checkpoints)}")
        for ckpt in checkpoints:
            print(f"  {ckpt}")

        # 最新取得テスト
        latest = manager.get_latest_checkpoint()
        print(f"Latest: {latest}")

        print("✅ Test completed successfully!")


if __name__ == '__main__':
    test_checkpoint_manager()

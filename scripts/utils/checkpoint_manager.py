import os
import time
import shutil
import glob
from datetime import datetime

class RollingCheckpointManager:
    def __init__(self, base_dir, max_keep=5, save_interval_sec=180):
        self.base_dir = base_dir
        self.max_keep = max_keep
        self.save_interval_sec = save_interval_sec
        self.last_save_time = time.time()

        # 保存ディレクトリがなければ作成
        os.makedirs(self.base_dir, exist_ok=True)

    def should_save(self):
        """前回の保存から指定時間が経過したかチェック"""
        return (time.time() - self.last_save_time) >= self.save_interval_sec

    def save_checkpoint(self, model, tokenizer, step_info="auto"):
        """モデルを保存し、古いものを削除する"""
        # 1. 保存
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(self.base_dir, f"ckpt_{timestamp}_{step_info}")

        print(f"💾 Saving checkpoint: {save_path} ...")
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)

        self.last_save_time = time.time()
        # 2. ローリングストック（古い削除）
        self._cleanup_old_checkpoints()

    def _cleanup_old_checkpoints(self):
        """最新5個以外を削除"""
        # ckpt_ で始まるフォルダを全取得
        checkpoints = glob.glob(os.path.join(self.base_dir, "ckpt_*"))
        # 作成日時順にソート（新しいのが後ろ）
        checkpoints.sort(key=os.path.getmtime)
        # 保持数を超えている場合、古いものから削除
        if len(checkpoints) > self.max_keep:
            to_delete = checkpoints[: -self.max_keep]
            for ckpt in to_delete:
                print(f"🗑️ Removing old checkpoint: {ckpt}")
                try:
                    shutil.rmtree(ckpt) # ディレクトリごと削除
                except Exception as e:
                    print(f"Error deleting {ckpt}: {e}")

    def get_latest_checkpoint(self):
        """再開用に最新のチェックポイントパスを取得"""
        checkpoints = glob.glob(os.path.join(self.base_dir, "ckpt_*"))
        if not checkpoints:
            return None
        # 最新を返す
        return max(checkpoints, key=os.path.getmtime)

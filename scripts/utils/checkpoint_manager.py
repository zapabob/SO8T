import os
import time
import shutil
import glob
import pickle
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Callable

class RollingCheckpointManager:
    """
    汎用ローリングチェックポイントマネージャー
    すべての時間のかかる作業に適用可能
    """

    def __init__(self, base_dir: str, max_keep: int = 5, save_interval_sec: int = 180,
                 task_name: str = "generic_task"):
        self.base_dir = Path(base_dir)
        self.max_keep = max_keep
        self.save_interval_sec = save_interval_sec
        self.task_name = task_name
        self.last_save_time = time.time()

        # 作業状態保存用
        self.state_file = self.base_dir / "task_state.json"
        self.current_state = {
            "task_name": task_name,
            "start_time": datetime.now().isoformat(),
            "last_checkpoint": None,
            "total_checkpoints": 0,
            "is_completed": False
        }

        # 保存ディレクトリがなければ作成
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._load_state()

    def _load_state(self):
        """状態ファイルを読み込み"""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    self.current_state.update(json.load(f))
            except Exception as e:
                print(f"Warning: Could not load state file: {e}")

    def _save_state(self):
        """状態ファイルを保存"""
        try:
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump(self.current_state, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Warning: Could not save state file: {e}")

    def update_state(self, **kwargs):
        """状態を更新"""
        self.current_state.update(kwargs)
        self._save_state()

    def should_save(self) -> bool:
        """前回の保存から指定時間が経過したかチェック"""
        return (time.time() - self.last_save_time) >= self.save_interval_sec

    def save_checkpoint(self, data: Any = None, metadata: Dict = None,
                       step_info: str = "auto", custom_save_func: Callable = None):
        """汎用チェックポイント保存"""
        # 1. 保存
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f"ckpt_{timestamp}_{step_info}"
        save_path = self.base_dir / checkpoint_name

        print(f"[SAVE] [{self.task_name}] Saving checkpoint: {checkpoint_name} ...")

        # カスタム保存関数があれば使用
        if custom_save_func:
            custom_save_func(save_path, data, metadata)
        else:
            # デフォルト保存（モデル/トークナイザー）
            if hasattr(data, 'save_pretrained'):
                data.save_pretrained(str(save_path))
                if metadata and hasattr(metadata, 'save_pretrained'):
                    metadata.save_pretrained(str(save_path))

        # メタデータ保存
        if metadata:
            meta_file = save_path / "metadata.json"
            try:
                with open(meta_file, 'w', encoding='utf-8') as f:
                    json.dump(metadata if isinstance(metadata, dict) else {"info": str(metadata)},
                            f, indent=2, ensure_ascii=False)
            except:
                pass

        # 状態更新
        self.last_save_time = time.time()
        self.current_state["last_checkpoint"] = checkpoint_name
        self.current_state["total_checkpoints"] += 1
        self.update_state()

        # 2. ローリングストック（古い削除）
        self._cleanup_old_checkpoints()

        return save_path

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
                print(f"[CLEAN] Removing old checkpoint: {ckpt}")
                try:
                    shutil.rmtree(ckpt) # ディレクトリごと削除
                except Exception as e:
                    print(f"Error deleting {ckpt}: {e}")

    def get_latest_checkpoint(self) -> Optional[Path]:
        """再開用に最新のチェックポイントパスを取得"""
        checkpoints = list(self.base_dir.glob("ckpt_*"))
        checkpoints = [p for p in checkpoints if p.is_dir()]  # ディレクトリのみ

        if not checkpoints:
            return None

        # 最新を返す
        return max(checkpoints, key=lambda p: p.stat().st_mtime)

    def load_checkpoint(self, checkpoint_path: Path = None, custom_load_func: Callable = None) -> Any:
        """チェックポイントからデータを読み込み"""
        if checkpoint_path is None:
            checkpoint_path = self.get_latest_checkpoint()

        if checkpoint_path is None or not checkpoint_path.exists():
            print(f"[ERROR] No checkpoint found for {self.task_name}")
            return None

        print(f"[LOAD] [{self.task_name}] Loading checkpoint: {checkpoint_path.name}")

        # カスタム読み込み関数があれば使用
        if custom_load_func:
            return custom_load_func(checkpoint_path)

        # デフォルト読み込み（モデル/トークナイザー）
        try:
            # metadata読み込み
            meta_file = checkpoint_path / "metadata.json"
            metadata = None
            if meta_file.exists():
                with open(meta_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
            return checkpoint_path, metadata
        except Exception as e:
            print(f"Warning: Could not load checkpoint metadata: {e}")
            return checkpoint_path, None

    def auto_resume(self, resume_func: Callable) -> bool:
        """自動再開機能"""
        latest_ckpt = self.get_latest_checkpoint()
        if latest_ckpt and not self.current_state.get("is_completed", False):
            print(f"[RESUME] [{self.task_name}] Auto-resuming from: {latest_ckpt.name}")
            try:
                resume_func(latest_ckpt)
                return True
            except Exception as e:
                print(f"[ERROR] Auto-resume failed: {e}")
                return False
        return False

    def mark_completed(self):
        """タスク完了をマーク"""
        self.current_state["is_completed"] = True
        self.current_state["completion_time"] = datetime.now().isoformat()
        self.update_state()
        print(f"[SUCCESS] [{self.task_name}] Task completed!")

    def get_status(self) -> Dict:
        """現在の状態を取得"""
        status = self.current_state.copy()
        status["latest_checkpoint"] = self.get_latest_checkpoint()
        status["should_save"] = self.should_save()
        return status


# ============================================================================
# ユーティリティ関数群（すべてのスクリプトで使用可能）
# ============================================================================

def create_task_manager(task_name: str, output_dir: str = None, max_keep: int = 5,
                      save_interval_sec: int = 180) -> RollingCheckpointManager:
    """タスク用のチェックポイントマネージャーを作成"""
    if output_dir is None:
        output_dir = f"checkpoints/{task_name}"

    return RollingCheckpointManager(
        base_dir=output_dir,
        max_keep=max_keep,
        save_interval_sec=save_interval_sec,
        task_name=task_name
    )

def with_checkpointing(task_func: Callable, task_name: str, output_dir: str = None):
    """
    デコレータ: 任意の関数にチェックポイント機能を追加
    使用例:
    @with_checkpointing("my_task", "checkpoints/my_task")
    def my_long_running_function():
        pass
    """
    manager = create_task_manager(task_name, output_dir)

    def wrapper(*args, **kwargs):
        # 自動再開チェック
        def resume_func(checkpoint_path):
            print(f"Resuming {task_name} from {checkpoint_path}")
            # 本番実装: checkpointファイルをロードして状態を復元する例
            if checkpoint_path.endswith(".pkl"):
                import pickle
                with open(checkpoint_path, "rb") as f:
                    checkpoint_data = pickle.load(f)
                # 必要な変数・状態を復元（タスクごとに適切に変更すること）
                # 例: globals().update(checkpoint_data)
                print(f"[INFO] checkpoint内容: {checkpoint_data}")
            else:
                print("[WARNING] 未対応のcheckpoint形式")

        if manager.auto_resume(resume_func):
            return  # 再開成功したら終了

        # 通常実行
        try:
            result = task_func(*args, **kwargs)

            # 定期チェックポイント（ループ内で手動で呼ぶ）
            # manager.save_checkpoint(data=result, step_info="final")

            manager.mark_completed()
            return result

        except KeyboardInterrupt:
            print(f"\n[WARNING] {task_name} interrupted. Saving checkpoint...")
            manager.save_checkpoint(step_info="interrupted")
            raise
        except Exception as e:
            print(f"\n[ERROR] {task_name} failed: {e}. Saving checkpoint...")
            manager.save_checkpoint(step_info="error")
            raise

    return wrapper

def checkpoint_context(task_name: str, output_dir: str = None):
    """
    コンテキストマネージャー: with文でチェックポイント機能を有効化
    使用例:
    with checkpoint_context("my_task"):
        for i in range(1000):
            # 時間のかかる処理
            do_something()
            # 自動チェックポイント（3分ごと）
            if manager.should_save():
                manager.save_checkpoint(data={"step": i}, step_info=f"step_{i}")
    """
    manager = create_task_manager(task_name, output_dir)

    class CheckpointContext:
        def __enter__(self):
            return manager

        def __exit__(self, exc_type, exc_val, exc_tb):
            if exc_type is None:
                manager.mark_completed()
            else:
                print(f"\n[WARNING] Task {task_name} exited with exception. Saving checkpoint...")
                manager.save_checkpoint(step_info="error_exit")

    return CheckpointContext()

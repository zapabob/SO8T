from __future__ import annotations

import json
import time
import uuid
import signal
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Optional, Any

import torch
from torch.utils.data import DataLoader, Dataset

from .vocab import Vocabulary


@dataclass
class DialogueRecord:
    tokens: List[str]
    label: str
    scenario: str
    meta: Dict[str, str]


class DialogueDataset(Dataset):
    """JSONL dataset that maps ENV/CMD/SAFE sequences to token ids."""

    def __init__(
        self,
        path: Path,
        vocab: Vocabulary,
        label_to_id: Dict[str, int],
        max_seq_len: int,
    ) -> None:
        self.path = path
        self.vocab = vocab
        self.label_to_id = label_to_id
        self.max_seq_len = max_seq_len
        self.records: List[DialogueRecord] = []
        self._load()

    def _load(self) -> None:
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                payload = json.loads(line)
                tokens = payload["tokens"]
                record = DialogueRecord(
                    tokens=tokens,
                    label=payload["label"],
                    scenario=payload.get("scenario", "unknown"),
                    meta=payload.get("meta", {})
                )
                self.records.append(record)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        record = self.records[idx]
        
        # トークンをIDに変換
        input_ids = self.vocab.encode(record.tokens, include_special_tokens=False)
        
        # パディング/トリミング
        if len(input_ids) > self.max_seq_len:
            input_ids = input_ids[:self.max_seq_len]
        else:
            pad_id = self.vocab.pad_index
            input_ids.extend([pad_id] * (self.max_seq_len - len(input_ids)))
        
        # アテンションマスクを作成
        attention_mask = [1 if token_id != self.vocab.pad_index else 0 for token_id in input_ids]
        
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(self.label_to_id[record.label], dtype=torch.long)
        }


def build_dataloader(
    dataset: DialogueDataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Build DataLoader for DialogueDataset."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lambda batch: {
            key: torch.stack([item[key] for item in batch])
            for key in batch[0].keys()
        }
    )


def build_vocab_from_files(data_files: List[Path], min_freq: int = 1) -> Vocabulary:
    """Build vocabulary from multiple data files."""
    vocab = Vocabulary()
    
    for data_file in data_files:
        with data_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                payload = json.loads(line)
                tokens = payload["tokens"]
                vocab.build_from_iterator([tokens], min_freq=min_freq)
    
    return vocab


def default_labels() -> List[str]:
    """Default label list for SO8T tasks."""
    return ["COMPLY", "REFUSE", "ESCALATE"]


class SessionCheckpointManager:
    """セッション管理とオートセーブ機能を提供"""
    
    def __init__(self, output_dir: Path, session_id: Optional[str] = None, max_backups: int = 10):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.session_id = session_id or str(uuid.uuid4())[:8]
        self.max_backups = max_backups
        self.autosave_dir = self.output_dir / "autosave"
        self.autosave_dir.mkdir(exist_ok=True)
        
        # 緊急保存用のフラグ
        self._emergency_save_requested = False
        self._emergency_save_lock = threading.Lock()
        
        # シグナルハンドラを設定（Windows対応）
        self._setup_signal_handlers()
    
    def _setup_signal_handlers(self):
        """Windows対応のシグナルハンドラを設定"""
        def emergency_save_handler(signum, frame):
            with self._emergency_save_lock:
                self._emergency_save_requested = True
                print(f"\nEmergency save requested (signal {signum})")
        
        # Windowsで利用可能なシグナルを設定
        for sig in [signal.SIGINT, getattr(signal, 'SIGTERM', None), getattr(signal, 'SIGBREAK', None)]:
            if sig is not None:
                signal.signal(sig, emergency_save_handler)
    
    def save(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, 
             scaler: Any, scheduler: Any, meta: Dict[str, Any]) -> Path:
        """チェックポイントを保存"""
        timestamp = int(time.time())
        checkpoint_path = self.autosave_dir / f"autosave_{self.session_id}_{timestamp}.pt"
        meta_path = self.autosave_dir / f"autosave_{self.session_id}_{timestamp}.json"
        
        # モデル状態を保存
        checkpoint_data = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict() if hasattr(scaler, 'state_dict') else None,
            'scheduler_state_dict': scheduler.state_dict() if hasattr(scheduler, 'state_dict') else None,
            'session_id': self.session_id,
            'timestamp': timestamp,
            'meta': meta
        }
        
        torch.save(checkpoint_data, checkpoint_path)
        
        # メタデータをJSONで保存
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump({
                'session_id': self.session_id,
                'timestamp': timestamp,
                'checkpoint_path': str(checkpoint_path),
                'meta': meta
            }, f, indent=2, ensure_ascii=False)
        
        # 古いバックアップをローテーション
        self._rotate_backups()
        
        print(f"Checkpoint saved: {checkpoint_path}")
        return checkpoint_path
    
    def load_latest(self) -> Optional[Dict[str, Any]]:
        """最新のチェックポイントを読み込み"""
        # セッションIDに一致する最新のチェックポイントを検索
        pattern = f"autosave_{self.session_id}_*.pt"
        checkpoint_files = list(self.autosave_dir.glob(pattern))
        
        if not checkpoint_files:
            return None
        
        # タイムスタンプでソートして最新を取得
        latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.stem.split('_')[-1]))
        
        try:
            checkpoint_data = torch.load(latest_checkpoint, map_location='cpu')
            print(f"Loaded checkpoint: {latest_checkpoint}")
            return checkpoint_data
        except Exception as e:
            print(f"Failed to load checkpoint {latest_checkpoint}: {e}")
            return None
    
    def emergency_save(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, 
                       scaler: Any, scheduler: Any, meta: Dict[str, Any]) -> Optional[Path]:
        """緊急保存を実行"""
        with self._emergency_save_lock:
            if not self._emergency_save_requested:
                return None
            
            try:
                emergency_path = self.save(model, optimizer, scaler, scheduler, meta)
                print(f"Emergency save completed: {emergency_path}")
                self._emergency_save_requested = False
                return emergency_path
            except Exception as e:
                print(f"Emergency save failed: {e}")
                return None
    
    def check_emergency_save(self) -> bool:
        """緊急保存が必要かチェック"""
        with self._emergency_save_lock:
            return self._emergency_save_requested
    
    def _rotate_backups(self):
        """古いバックアップをローテーション"""
        pattern = f"autosave_{self.session_id}_*.pt"
        checkpoint_files = list(self.autosave_dir.glob(pattern))
        
        if len(checkpoint_files) <= self.max_backups:
            return
        
        # タイムスタンプでソートして古いものから削除
        sorted_files = sorted(checkpoint_files, key=lambda x: int(x.stem.split('_')[-1]))
        files_to_remove = sorted_files[:-self.max_backups]
        
        for file_path in files_to_remove:
            try:
                file_path.unlink()
                # 対応するJSONファイルも削除
                json_path = file_path.with_suffix('.json')
                if json_path.exists():
                    json_path.unlink()
                print(f"Removed old backup: {file_path}")
            except Exception as e:
                print(f"Failed to remove old backup {file_path}: {e}")
    
    def get_session_info(self) -> Dict[str, Any]:
        """セッション情報を取得"""
        latest_checkpoint = self.load_latest()
        if latest_checkpoint is None:
            return {
                'session_id': self.session_id,
                'has_checkpoint': False,
                'latest_timestamp': None
            }
        
        return {
            'session_id': self.session_id,
            'has_checkpoint': True,
            'latest_timestamp': latest_checkpoint.get('timestamp'),
            'latest_meta': latest_checkpoint.get('meta', {})
        }

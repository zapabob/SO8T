from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from .vocab import BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, Vocabulary


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
                    meta={
                        "label_reason": payload.get("label_reason", ""),
                        "env": payload.get("env", ""),
                        "cmd": payload.get("cmd", ""),
                        "safe": payload.get("safe", ""),
                    },
                )
                self.records.append(record)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        record = self.records[index]
        token_ids = self.vocab.encode(record.tokens)
        token_ids = token_ids[: self.max_seq_len]
        attention_mask = [1] * len(token_ids)
        return {
            "input_ids": torch.tensor(token_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(self.label_to_id[record.label], dtype=torch.long),
            "meta": record.meta,
        }


def collate_batch(batch: List[Dict[str, torch.Tensor]], pad_index: int) -> Dict[str, torch.Tensor]:
    max_len = max(item["input_ids"].shape[0] for item in batch)
    padded_ids: List[torch.Tensor] = []
    padded_mask: List[torch.Tensor] = []
    labels: List[torch.Tensor] = []
    metas: List[Dict[str, str]] = []
    for item in batch:
        seq = item["input_ids"]
        mask = item["attention_mask"]
        padding_needed = max_len - seq.shape[0]
        if padding_needed > 0:
            pad_tensor = torch.full((padding_needed,), pad_index, dtype=torch.long)
            mask_pad = torch.zeros(padding_needed, dtype=torch.long)
            seq = torch.cat([seq, pad_tensor], dim=0)
            mask = torch.cat([mask, mask_pad], dim=0)
        padded_ids.append(seq)
        padded_mask.append(mask)
        labels.append(item["labels"])
        metas.append(item["meta"])
    return {
        "input_ids": torch.stack(padded_ids, dim=0),
        "attention_mask": torch.stack(padded_mask, dim=0),
        "labels": torch.stack(labels, dim=0),
        "meta": metas,
    }


def make_collate_fn(pad_index: int):
    """Factory function to create collate_fn for DataLoader."""
    def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        return collate_batch(batch, pad_index)
    return collate_fn


def build_dataloader(
    dataset: DialogueDataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=make_collate_fn(dataset.vocab.pad_index),
        pin_memory=torch.cuda.is_available(),
    )


def build_vocab_from_files(paths: Iterable[Path], min_freq: int = 1) -> Vocabulary:
    vocab = Vocabulary()
    sequences: List[List[str]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                payload = json.loads(line)
                sequences.append(payload["tokens"])
    vocab.build_from_iterator(sequences, min_freq=min_freq)
    return vocab


def default_labels() -> Tuple[Dict[str, int], Dict[int, str]]:
    labels = [COMPLY, REFUSE, ESCALATE]
    label_to_id = {label: idx for idx, label in enumerate(labels)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    return label_to_id, id_to_label


# Late import to avoid circular dependency for the constants
COMPLY = "COMPLY"
REFUSE = "REFUSE"
ESCALATE = "ESCALATE"

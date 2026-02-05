from __future__ import annotations

from pathlib import Path

import random

from dataset_synth import generate_samples, partition_samples, write_jsonl
from shared.data import DialogueDataset, build_vocab_from_files, default_labels


def test_dataset_roundtrip(tmp_path: Path) -> None:
    rng_seed = 13
    samples = generate_samples(random.Random(rng_seed), 50)
    train, _, _ = partition_samples(samples, 0.8, 0.1)
    data_path = tmp_path / "train.jsonl"
    write_jsonl(data_path, train)

    vocab = build_vocab_from_files([data_path])
    label_to_id, _ = default_labels()
    dataset = DialogueDataset(data_path, vocab, label_to_id, max_seq_len=64)

    assert len(dataset) == len(train)
    sample = dataset[0]
    assert sample["input_ids"].ndim == 1
    assert sample["attention_mask"].sum() == len(sample["input_ids"])

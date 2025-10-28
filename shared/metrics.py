from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Tuple

import torch


def accuracy(preds: torch.Tensor, labels: torch.Tensor) -> float:
    correct = (preds == labels).sum().item()
    total = labels.numel()
    return correct / total if total else 0.0


def macro_f1(preds: torch.Tensor, labels: torch.Tensor, num_classes: int) -> float:
    eps = 1e-8
    f1_scores: List[float] = []
    for cls in range(num_classes):
        tp = ((preds == cls) & (labels == cls)).sum().item()
        fp = ((preds == cls) & (labels != cls)).sum().item()
        fn = ((preds != cls) & (labels == cls)).sum().item()
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        f1_scores.append(f1)
    return sum(f1_scores) / len(f1_scores) if f1_scores else 0.0


def confusion_matrix(preds: torch.Tensor, labels: torch.Tensor, num_classes: int) -> List[List[int]]:
    matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for pred, label in zip(preds.tolist(), labels.tolist()):
        matrix[label][pred] += 1
    return matrix

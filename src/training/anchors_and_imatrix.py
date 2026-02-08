import torch
import torch.nn as nn
import json
import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class StabilityAnchors:
    """
    Manages anchor sets for stability (KL-divergence) monitoring.
    Sets include general language ability and reasoning structure.
    """
    def __init__(self, tokenizer, anchor_path: Optional[str] = None):
        self.tokenizer = tokenizer
        self.anchors = []
        if anchor_path and os.path.exists(anchor_path):
            self._load_anchors(anchor_path)
        else:
            self._init_default_anchors()

    def _init_default_anchors(self):
        # Default safety anchors to prevent collapse
        self.anchors = [
            {"instruction": "こんにちは、自己紹介をしてください。", "output": "私はBorea-Phi-3.5、日本語に特化したAIアシスタントです。"},
            {"instruction": "1+1は？", "output": "2です。"},
            {"instruction": "SO8Tの四重推論タグをすべて挙げてください。", "output": "<think-task>, <think-analysis>, <think-safety>, <think-policy>の4つです。"}
        ]

    def _load_anchors(self, path: str):
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        self.anchors.append(json.loads(line))
        except Exception as e:
            logger.error(f"Failed to load anchors: {e}")

    def get_batch(self, batch_size: int = 4):
        # Prepare tokens for KL calculation
        prompts = [a["instruction"] for a in self.anchors[:batch_size]]
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to("cuda")
        return inputs

class IMatrixImportance:
    """
    Uses imatrix (importance matrix) data to guide evolutionary freezing.
    Higher importance means lower probability of unfreezing.
    """
    def __init__(self, imatrix_path: Optional[str] = None):
        self.importance_scores = {}
        if imatrix_path and os.path.exists(imatrix_path):
            self._load_imatrix(imatrix_path)

    def _load_imatrix(self, path: str):
        # Placeholder for loading actual imatrix binary/json
        # Format usually: layer.N.module -> importance score
        pass

    def get_score(self, module_name: str) -> float:
        # Default importance 1.0 if not found
        return self.importance_scores.get(module_name, 1.0)

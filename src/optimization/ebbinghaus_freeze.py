# -*- coding: utf-8 -*-
"""
EbbinghausFreeze: 忘却曲線ベースの動的冻结システム

R = exp(-t/S)
- R: 保持率 (retention)
- t: 経過時間 (時間)
- S: 記憶の強度係数 (デフォルト24時間)

エビングハウスの忘却曲線に基づいて、経過時間に応じて冻结パラメータを動的に調整。
時間が経つにつれて新しい知識が定着し、古い知識は冻結を維持（忘却防止）。
"""

from __future__ import annotations

import math
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MemoryNode:
    """記憶ノード: 学習した知識の一単位"""

    content: str
    domain: str  # science, math, world_events, etc.
    timestamp: datetime = field(default_factory=datetime.now)
    strength_coefficient: float = 24.0  # S値（時間）
    review_count: int = 0
    last_review: Optional[datetime] = None
    frozen: bool = True
    importance_score: float = 0.5

    def get_retention(self, current_time: Optional[datetime] = None) -> float:
        """
        現在の保持率を計算
        R = exp(-t/S)
        """
        if current_time is None:
            current_time = datetime.now()

        if self.last_review:
            elapsed_hours = (current_time - self.last_review).total_seconds() / 3600.0
        else:
            elapsed_hours = (current_time - self.timestamp).total_seconds() / 3600.0

        retention = math.exp(-elapsed_hours / self.strength_coefficient)
        return max(0.0, min(1.0, retention))

    def should_unfreeze(self, threshold: float = 0.5) -> bool:
        """
        冻结解除すべきかを判定
        保持率が閾値を下回ったら再学習が必要
        """
        retention = self.get_retention()
        return retention < threshold and self.importance_score > 0.3


@dataclass
class FreezeConfig:
    """冻结設定"""

    default_strength_hours: float = 24.0
    freeze_threshold: float = 0.7  # これ以下の保持率は冻结維持
    unfreeze_threshold: float = 0.5  # これ以下は再学習対象
    review_interval_hours: float = 1.0
    max_frozen_layers: int = 8
    min_trainable_layers: int = 2
    protection_domains: List[str] = field(
        default_factory=lambda: [
            "arxiv",
            "biorxiv",
            "domain_knowledge",
            "world_events_2024_2026",
            "science",
            "math",
            "quadruple_reasoning",
            "vssi",
        ]
    )


class EbbinghausFreeze:
    """
    忘却曲線ベースの動的冻结管理システム

    機能:
    - 各学習データの保持率を計算
    - 保護ドメイン（ArXiv/BioRxiv等）を冻结状態で維持
    - imatrix重要度と忘却曲線を組み合わせて冻结パラメータを決定
    - 統計的有意水準（95%）に基づいた冻结層選択
    """

    def __init__(
        self,
        config: Optional[FreezeConfig] = None,
        knowledge_base_path: Optional[str] = None,
        imatrix_path: Optional[str] = None,
    ):
        self.config = config or FreezeConfig()
        self.knowledge_base_path = (
            Path(knowledge_base_path) if knowledge_base_path else None
        )
        self.imatrix_path = Path(imatrix_path) if imatrix_path else None

        self.memory_nodes: Dict[str, MemoryNode] = {}
        self.layer_frozen_states: Dict[str, bool] = {}
        self.imatrix_scores: Dict[str, float] = {}

        self._load_knowledge_base()
        self._load_imatrix()

    def _load_knowledge_base(self) -> None:
        """知識ベースファイルからメモリノードを読み込み"""
        if self.knowledge_base_path and self.knowledge_base_path.exists():
            try:
                with open(self.knowledge_base_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for key, node_data in data.items():
                        self.memory_nodes[key] = MemoryNode(
                            content=node_data.get("content", ""),
                            domain=node_data.get("domain", "general"),
                            timestamp=datetime.fromisoformat(
                                node_data.get("timestamp", datetime.now().isoformat())
                            ),
                            strength_coefficient=node_data.get(
                                "strength_coefficient",
                                self.config.default_strength_hours,
                            ),
                            review_count=node_data.get("review_count", 0),
                            importance_score=node_data.get("importance_score", 0.5),
                        )
                logger.info(
                    f"Loaded {len(self.memory_nodes)} memory nodes from knowledge base"
                )
            except Exception as e:
                logger.warning(f"Failed to load knowledge base: {e}")

    def _load_imatrix(self) -> None:
        """imatrix重要度スコアを読み込み"""
        if self.imatrix_path and self.imatrix_path.exists():
            try:
                with open(self.imatrix_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.imatrix_scores = data.get("importance_scores", {})
                logger.info(f"Loaded {len(self.imatrix_scores)} imatrix scores")
            except Exception as e:
                logger.warning(f"Failed to load imatrix: {e}")

    def add_memory(
        self,
        content: str,
        domain: str,
        importance_score: float = 0.5,
        strength_hours: Optional[float] = None,
    ) -> str:
        """
        新しい記憶を追加

        Args:
            content: 記憶内容
            domain: ドメイン（arxiv, biorxiv, math, science等）
            importance_score: 重要度スコア (0-1)
            strength_hours: 記憶強度係数

        Returns:
            memory_id: 生成されたメモリID
        """
        memory_id = (
            f"mem_{len(self.memory_nodes)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        self.memory_nodes[memory_id] = MemoryNode(
            content=content,
            domain=domain,
            strength_coefficient=strength_hours or self.config.default_strength_hours,
            importance_score=importance_score,
            frozen=True,  # 新規記憶は初期状態冻结
        )

        logger.info(f"Added memory node: {memory_id} (domain: {domain})")
        return memory_id

    def calculate_layer_freeze_probability(
        self, layer_name: str, domain: str, imatrix_score: Optional[float] = None
    ) -> Tuple[bool, float]:
        """
        層の冻结確率を計算

        Args:
            layer_name: レイヤー名
            domain: 知識ドメイン
            imatrix_score: imatrix重要度スコア

        Returns:
            (should_freeze, confidence): 冻结すべきか、置信度
        """
        should_freeze = True
        confidence = 0.5

        if domain in self.config.protection_domains:
            should_freeze = True
            confidence = 0.95
        else:
            retention = 1.0
            for mem_id, node in self.memory_nodes.items():
                if node.domain == domain:
                    retention = min(retention, node.get_retention())

            if imatrix_score is None and layer_name in self.imatrix_scores:
                imatrix_score = self.imatrix_scores[layer_name]

            if imatrix_score is not None:
                combined_score = retention * imatrix_score
                should_freeze = combined_score > self.config.freeze_threshold
                confidence = min(0.95, combined_score + 0.1)

            if not should_freeze and retention < self.config.unfreeze_threshold:
                should_freeze = False
                confidence = retention

        return should_freeze, confidence

    def get_frozen_layers_95ci(
        self, layers: List[Dict[str, str]], domain_map: Dict[str, str]
    ) -> List[str]:
        """
        95%信頼区間で冻结すべき層を選択

        Args:
            layers: レイヤー情報のリスト [{"name": "...", "type": "..."}]
            domain_map: レイヤー名からドメインへのマッピング

        Returns:
            frozen_layers: 冻结すべきレイヤー名のリスト
        """
        freeze_scores = []
        layer_names = []

        for layer in layers:
            name = layer.get("name", "")
            domain = domain_map.get(name, "general")
            imatrix = self.imatrix_scores.get(name, None)

            should_freeze, confidence = self.calculate_layer_freeze_probability(
                name, domain, imatrix
            )

            freeze_scores.append(1.0 if should_freeze else 0.0)
            layer_names.append(name)

        if len(freeze_scores) < 3:
            return [
                name for name, score in zip(layer_names, freeze_scores) if score > 0.5
            ]

        mean_score = np.mean(freeze_scores)
        std_score = np.std(freeze_scores)

        if std_score == 0:
            return [
                name
                for name, score in zip(layer_names, freeze_scores)
                if score > mean_score
            ]

        z_scores = [(score - mean_score) / std_score for score in freeze_scores]

        frozen_layers = []
        for name, z_score in zip(layer_names, z_scores):
            if z_score < 1.96:  # 95% CI
                frozen_layers.append(name)

        max_frozen = min(self.config.max_frozen_layers, len(frozen_layers))
        return frozen_layers[:max_frozen]

    def review_memory(self, memory_id: str) -> float:
        """
        記憶をレビュー（再学習）し、保持率を更新

        Args:
            memory_id: レビューするメモリのID

        Returns:
            new_retention: 更新後の保持率
        """
        if memory_id not in self.memory_nodes:
            logger.warning(f"Memory {memory_id} not found")
            return 0.0

        node = self.memory_nodes[memory_id]
        node.review_count += 1
        node.last_review = datetime.now()

        new_retention = node.get_retention()
        node.strength_coefficient *= 1.0 + new_retention * 0.1

        logger.info(
            f"Reviewed memory {memory_id}: retention={new_retention:.3f}, reviews={node.review_count}"
        )
        return new_retention

    def save_state(self, output_path: str) -> None:
        """現在の状態をファイルに保存"""
        state = {
            "timestamp": datetime.now().isoformat(),
            "memory_nodes": {
                mem_id: {
                    "content": node.content,
                    "domain": node.domain,
                    "timestamp": node.timestamp.isoformat(),
                    "strength_coefficient": node.strength_coefficient,
                    "review_count": node.review_count,
                    "importance_score": node.importance_score,
                }
                for mem_id, node in self.memory_nodes.items()
            },
            "layer_frozen_states": self.layer_frozen_states,
            "config": {
                "default_strength_hours": self.config.default_strength_hours,
                "freeze_threshold": self.config.freeze_threshold,
                "protection_domains": self.config.protection_domains,
            },
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved EbbinghausFreeze state to {output_path}")

    def load_state(self, input_path: str) -> None:
        """状態ファイルを読み込み"""
        with open(input_path, "r", encoding="utf-8") as f:
            state = json.load(f)

        for mem_id, node_data in state.get("memory_nodes", {}).items():
            self.memory_nodes[mem_id] = MemoryNode(
                content=node_data["content"],
                domain=node_data["domain"],
                timestamp=datetime.fromisoformat(node_data["timestamp"]),
                strength_coefficient=node_data.get(
                    "strength_coefficient", self.config.default_strength_hours
                ),
                review_count=node_data.get("review_count", 0),
                importance_score=node_data.get("importance_score", 0.5),
            )

        self.layer_frozen_states = state.get("layer_frozen_states", {})
        logger.info(f"Loaded {len(self.memory_nodes)} memory nodes from {input_path}")

    def get_statistics(self) -> Dict[str, float]:
        """現在の統計情報を取得"""
        retentions = [node.get_retention() for node in self.memory_nodes.values()]
        return {
            "total_memories": len(self.memory_nodes),
            "mean_retention": np.mean(retentions) if retentions else 0.0,
            "std_retention": np.std(retentions) if retentions else 0.0,
            "frozen_layers": len(self.layer_frozen_states),
            "protection_domains": len(self.config.protection_domains),
        }

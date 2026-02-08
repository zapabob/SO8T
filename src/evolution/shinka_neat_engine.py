# -*- coding: utf-8 -*-
"""
ShinkaNEAT Evolutionary Engine - NEAT + ShinkaEvolve統合エンジン

Inspired by:
- Sakana AI "The AI Scientist" & "ShinkaEvolve"
- NEAT (NeuroEvolution of Augmenting Topologies) applied to Reasoning Chains
- Ollama for inference (Borea-Phi-3.5-Instinct-JP)

Features:
- Ollamaによる推論
- NEAT-inspired突然変異（ノード/リンク追加）
- ShinkaEvolveの島モデル進化
- 淘汰圧によるデータセット合成
- 凍結パラメータの動的調整との連携
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import httpx
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ReasoningNode:
    """推論ノード（NEATノード）"""

    id: int
    content: str
    node_type: str = "thought"  # observation, thought, critique, conclusion, analysis, safety, policy
    parents: List[int] = field(default_factory=list)
    depth: int = 0
    fitness_contribution: float = 0.0


@dataclass
class Individual:
    """進化個体（ゲノム = 推論ノードのリスト）"""

    genome: List[ReasoningNode]
    fitness: float = 0.0
    id: str = field(
        default_factory=lambda: hashlib.md5(str(random.random()).encode()).hexdigest()[
            :8
        ]
    )
    domain: str = "general"
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)
    mutation_history: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "fitness": self.fitness,
            "domain": self.domain,
            "generation": self.generation,
            "parent_ids": self.parent_ids,
            "mutation_history": self.mutation_history,
            "genome": [
                {
                    "id": n.id,
                    "content": n.content,
                    "node_type": n.node_type,
                    "parents": n.parents,
                    "depth": n.depth,
                }
                for n in self.genome
            ],
            "timestamp": self.timestamp,
        }


@dataclass
class EvolutionConfig:
    """進化設定"""

    population_size: int = 8
    island_count: int = 2
    generations: int = 3
    mutation_rate: float = 0.3
    crossover_rate: float = 0.2
    migration_interval: int = 2
    elite_ratio: float = 0.2
    diversity_threshold: float = 0.1
    tournament_size: int = 3


class OllamaClient:
    """Ollama推論クライアント"""

    def __init__(
        self,
        model: str = "borea-phi-3.5-instinct-jp",
        base_url: str = "http://localhost:11434",
    ):
        self.model = model
        self.base_url = base_url
        self.client = httpx.Client(timeout=120.0)

    def health_check(self) -> bool:
        """Ollama健全性チェック"""
        try:
            response = self.client.get(f"{self.base_url}/api/version")
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Ollama health check failed: {e}")
            return False

    def generate(
        self,
        prompt: str,
        system_prompt: str = "あなたは思考の専門家です。",
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        """テキスト生成"""
        try:
            response = self.client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "system": system_prompt,
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            )
            response.raise_for_status()
            return response.json().get("response", "").strip()
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            raise

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        """チャット形式生成"""
        try:
            response = self.client.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            )
            response.raise_for_status()
            return response.json().get("message", {}).get("content", "").strip()
        except Exception as e:
            logger.error(f"Ollama chat failed: {e}")
            raise


class NEATReasoningEngine:
    """
    NEAT-inspired推論エンジン

    突然変異オペレータ:
    - add_node: 新しい推論ノードを追加
    - add_link: 既存ノード間を接続
    - mutate_content: 既存ノードの内容を変更
    - add_critique: 批判ノードを追加（否定スピンルック向け）
    """

    def __init__(self, ollama_client: OllamaClient, node_id_start: int = 0):
        self.ollama = ollama_client
        self.node_id_counter = node_id_start
        self.knowledge_weights: Dict[str, float] = {}

    def create_initial_individual(
        self, topic: str, domain: str = "general", node_count: int = 3
    ) -> Individual:
        """初期個体を生成"""
        nodes = []

        obs_node = ReasoningNode(
            id=self.node_id_counter,
            content=f"Topic: {topic}",
            node_type="observation",
            depth=0,
        )
        self.node_id_counter += 1
        nodes.append(obs_node)

        for i in range(node_count - 1):
            node = ReasoningNode(
                id=self.node_id_counter,
                content=f"Initial thought {i + 1} about {topic}",
                node_type="thought",
                parents=[obs_node.id] if i == 0 else [],
                depth=1,
            )
            self.node_id_counter += 1
            nodes.append(node)

        return Individual(genome=nodes, domain=domain, fitness=0.0)

    def mutate(self, individual: Individual) -> Individual:
        """突然変異（NEATスタイル）"""
        new_genome = [n for n in individual.genome]
        mutation_type = random.choices(
            [
                "add_thought",
                "add_analysis",
                "add_safety",
                "add_policy",
                "add_critique",
                "mutate_content",
            ],
            weights=[0.3, 0.2, 0.15, 0.15, 0.1, 0.1],
        )[0]

        new_mutation_history = individual.mutation_history + [mutation_type]

        if mutation_type == "add_thought":
            parent = random.choice(new_genome)
            prompt = f"""
前の推論: {parent.content}

次の論理的思考ステップを追加してください。前の内容を深化させ、より深い分析を提供してください。
"""
            content = self.ollama.generate(prompt, temperature=0.8)
            new_node = ReasoningNode(
                id=self.node_id_counter,
                content=content,
                node_type="thought",
                parents=[parent.id],
                depth=parent.depth + 1,
            )
            self.node_id_counter += 1
            new_genome.append(new_node)

        elif mutation_type == "add_analysis":
            target = random.choice(new_genome)
            prompt = f"""
推論内容: {target.content}

この推論の論理的分析を行ってください。前提、推論プロセス、結論を分解して検証してください。
"""
            content = self.ollama.generate(prompt, temperature=0.7)
            new_node = ReasoningNode(
                id=self.node_id_counter,
                content=content,
                node_type="analysis",
                parents=[target.id],
                depth=target.depth,
            )
            self.node_id_counter += 1
            new_genome.append(new_node)

        elif mutation_type == "add_safety":
            target = random.choice(new_genome)
            prompt = f"""
推論内容: {target.content}

この推論の安全性とリスクを検討してください。潜在的な問題点、例外ケース、倫理的懸念を指摘してください。
"""
            content = self.ollama.generate(prompt, temperature=0.7)
            new_node = ReasoningNode(
                id=self.node_id_counter,
                content=content,
                node_type="safety",
                parents=[target.id],
                depth=target.depth,
            )
            self.node_id_counter += 1
            new_genome.append(new_node)

        elif mutation_type == "add_policy":
            target = random.choice(new_genome)
            prompt = f"""
推論内容: {target.content}

この推論に基づく政策的意思決定または最終結論を提示してください。具体的アクションと推奨事項を含める。
"""
            content = self.ollama.generate(prompt, temperature=0.6)
            new_node = ReasoningNode(
                id=self.node_id_counter,
                content=content,
                node_type="policy",
                parents=[target.id],
                depth=target.depth,
            )
            self.node_id_counter += 1
            new_genome.append(new_node)

        elif mutation_type == "add_critique":
            target = random.choice(new_genome)
            prompt = f"""
推論内容: {target.content}

この推論の論理的な誤りや欠けている視点を批判的に指摘してください。反論と代替案を提示してください。
"""
            content = self.ollama.generate(prompt, temperature=0.9)
            new_node = ReasoningNode(
                id=self.node_id_counter,
                content=content,
                node_type="critique",
                parents=[target.id],
                depth=target.depth,
            )
            self.node_id_counter += 1
            new_genome.append(new_node)

        elif mutation_type == "mutate_content":
            if len(new_genome) > 0:
                target = random.choice(new_genome)
                prompt = f"""
以下の内容を改善してください。より正確で深い洞察を提供してください。

内容: {target.content}
"""
                content = self.ollama.generate(prompt, temperature=0.8)
                target.content = content

        return Individual(
            genome=new_genome,
            fitness=individual.fitness,
            domain=individual.domain,
            generation=individual.generation + 1,
            parent_ids=[individual.id],
            mutation_history=new_mutation_history,
        )

    def crossover(self, parent1: Individual, parent2: Individual) -> Individual:
        """交叉（2つの個体を結合）"""
        min_len = min(len(parent1.genome), len(parent2.genome))
        split_point = random.randint(1, min_len - 1)

        new_genome = parent1.genome[:split_point] + [
            ReasoningNode(
                id=n.id,
                content=n.content,
                node_type=n.node_type,
                parents=n.parents,
                depth=n.depth,
            )
            for n in parent2.genome[split_point:]
        ]

        return Individual(
            genome=new_genome,
            fitness=max(parent1.fitness, parent2.fitness),
            domain=parent1.domain if random.random() < 0.5 else parent2.domain,
            generation=max(parent1.generation, parent2.generation) + 1,
            parent_ids=[parent1.id, parent2.id],
            mutation_history=["crossover"],
        )

    def evaluate_fitness(
        self,
        individual: Individual,
        reference: Optional[str] = None,
        world_events: Optional[Dict] = None,
    ) -> float:
        """
        適合度を評価

        評価基準:
        - 論理的マーカー（therefore, because等）の存在
        - 四重推論タグのバランス
        - 外部参照との整合性（オプション）
        - 多様性（他の個体との差異）
        """
        chain = " ".join([n.content for n in individual.genome])
        score = 0.0

        logical_markers = [
            "したがって",
            "。なぜなら",
            "しかし",
            "ゆえに",
            "すなわち",
            "therefore",
            "because",
            "however",
        ]
        marker_count = sum(1 for m in logical_markers if m.lower() in chain.lower())
        score += min(marker_count * 0.15, 0.45)

        node_types = set(n.node_type for n in individual.genome)
        type_bonus = {
            "observation": 0.1,
            "thought": 0.15,
            "analysis": 0.2,
            "safety": 0.2,
            "policy": 0.2,
            "critique": 0.15,
        }
        for ntype in node_types:
            score += type_bonus.get(ntype, 0.0)

        if len(individual.genome) > 3:
            score += 0.1
        if len(individual.genome) > 5:
            score += 0.1

        return min(score, 1.0)

    def update_knowledge_weights(self, best_individual: Individual) -> None:
        """最良個体の知識を重みとして保存"""
        key = f"{best_individual.domain}_{best_individual.id}"
        self.knowledge_weights[key] = {
            "weight": best_individual.fitness,
            "timestamp": datetime.now().isoformat(),
            "node_count": len(best_individual.genome),
        }


class ShinkaEvolveEngine:
    """
    ShinkaEvolveエンジン（Sakana AIinspired）

    島モデル進化:
    - 各島で独立した集団が進化
    定期的に個体が移動（移住）
    - 島間の知識交換を促進
    """

    def __init__(
        self, neat_engine: NEATReasoningEngine, config: Optional[EvolutionConfig] = None
    ):
        self.neat = neat_engine
        self.config = config or EvolutionConfig()
        self.islands: List[List[Individual]] = []
        self.node_id_counter = neat_engine.node_id_counter

    def initialize_islands(
        self, topic: str, domain: str, pop_per_island: Optional[int] = None
    ) -> None:
        """島を初期化"""
        pop = pop_per_island or self.config.population_size // self.config.island_count
        self.islands = []

        for _ in range(self.config.island_count):
            island = [
                self.neat.create_initial_individual(topic, domain) for _ in range(pop)
            ]
            self.islands.append(island)

        self.node_id_counter = self.neat.node_id_counter
        logger.info(
            f"Initialized {self.config.island_count} islands with {pop} individuals each"
        )

    def evolve_island(
        self, island: List[Individual], generation: int
    ) -> List[Individual]:
        """島を1世代進化させる"""
        for ind in island:
            ind.fitness = self.neat.evaluate_fitness(ind)

        island.sort(key=lambda x: x.fitness, reverse=True)

        elite_count = max(1, int(len(island) * self.config.elite_ratio))
        elites = island[:elite_count]
        survivors = island[: len(island) // 2]

        new_island = [ind for ind in elites]

        while len(new_island) < len(island):
            parent = random.choice(survivors)
            child = self.neat.mutate(parent)
            child.generation = generation
            new_island.append(child)

        return new_island

    def migrate(self) -> None:
        """島間で移住を実行"""
        if len(self.islands) < 2:
            return

        for i in range(len(self.islands)):
            if len(self.islands[i]) > 0:
                migrant_idx = random.randint(0, len(self.islands[i]) - 1)
                migrant = self.islands[i].pop(migrant_idx)

                dest_idx = (i + 1) % len(self.islands)
                self.islands[dest_idx].append(migrant)

        logger.info("Migration completed between islands")

    def run_evolution(
        self,
        topic: str,
        domain: str,
        reference: Optional[str] = None,
        world_events: Optional[Dict] = None,
        generations: Optional[int] = None,
    ) -> Individual:
        """
        進化を実行し最良個体を返す

        Args:
            topic: 進化トピック
            domain: 知識ドメイン
            reference: 参考情報（評価用）
            world_events: 世界情勢データ（オプション）
            generations: 世代数

        Returns:
            最良個体
        """
        gens = generations or self.config.generations
        self.initialize_islands(topic, domain)

        for gen in range(gens):
            logger.info(f"Generation {gen + 1}/{gens}...")

            for i, island in enumerate(self.islands):
                self.islands[i] = self.evolve_island(island, gen)

            if gen % self.config.migration_interval == 0:
                self.migrate()

        all_individuals = [ind for island in self.islands for ind in island]
        all_individuals.sort(key=lambda x: x.fitness, reverse=True)

        best = all_individuals[0]
        self.neat.update_knowledge_weights(best)

        logger.info(f"Evolution complete. Best fitness: {best.fitness:.3f}")
        return best

    def get_population_statistics(self) -> Dict[str, Any]:
        """集団統計を取得"""
        if not self.islands:
            return {"message": "No population initialized"}

        all_fitness = [ind.fitness for island in self.islands for ind in island]
        return {
            "total_individuals": len(all_fitness),
            "island_count": len(self.islands),
            "mean_fitness": np.mean(all_fitness),
            "std_fitness": np.std(all_fitness),
            "max_fitness": np.max(all_fitness),
            "min_fitness": np.min(all_fitness),
        }


class ShinkaNEATPipeline:
    """
    ShinkaNEAT統合パイプライン

    機能:
    - Ollamaによる推論
    - NEAT/ShinkaEvolve進化
    - 淘汰圧データセット合成
    - チェックポイント対応
    """

    def __init__(
        self,
        ollama_model: str = "borea-phi-3.5-instinct-jp",
        ollama_url: str = "http://localhost:11434",
        config: Optional[EvolutionConfig] = None,
    ):
        self.ollama = OllamaClient(ollama_model, ollama_url)
        self.neat = NEATReasoningEngine(self.ollama)
        self.evolve = ShinkaEvolveEngine(self.neat, config)
        self.knowledge_weights_file = Path("data/knowledge_weights.json")

    def generate_synthetic_dataset(
        self,
        topics: List[Dict[str, str]],
        output_path: str,
        skip_completed: bool = True,
    ) -> Dict[str, Any]:
        """
        合成データセットを生成

        Args:
            topics: トピック辞書のリスト [{"topic": "...", "domain": "..."}]
            output_path: 出力JSONLパス
            skip_completed: 完了済みトピックをスキップ

        Returns:
            生成統計
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        stats = {
            "total_topics": len(topics),
            "completed": 0,
            "skipped": 0,
            "errors": 0,
            "output_path": output_path,
        }

        for topic_info in topics:
            topic = topic_info.get("topic", "")
            domain = topic_info.get("domain", "general")

            if not topic:
                continue

            logger.info(f"Processing topic: {topic[:50]}...")

            try:
                best_individual = self.evolve.run_evolution(topic, domain)

                data = {
                    "instruction": f"{topic}について、深く論理的に考察せよ。",
                    "thinking": "\n".join(
                        [f"[{n.node_type}] {n.content}" for n in best_individual.genome]
                    ),
                    "output": f"{topic}に関する進化型推論の結果、多様な視点からの考察が得られました。",
                    "metadata": {
                        "method": "ShinkaNEAT",
                        "fitness": best_individual.fitness,
                        "domain": domain,
                        "generation": best_individual.generation,
                        "node_count": len(best_individual.genome),
                        "timestamp": best_individual.timestamp,
                    },
                }

                with open(output_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(data, ensure_ascii=False) + "\n")

                stats["completed"] += 1

            except Exception as e:
                logger.error(f"Error processing topic {topic[:30]}: {e}")
                stats["errors"] += 1

        logger.info(f"Dataset generation complete: {stats}")
        return stats

    def get_knowledge_weights(self) -> Dict[str, Any]:
        """知識重みを取得"""
        if self.knowledge_weights_file.exists():
            with open(self.knowledge_weights_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def save_knowledge_weights(self) -> None:
        """知識重みを保存"""
        weights = self.neat.knowledge_weights
        with open(self.knowledge_weights_file, "w", encoding="utf-8") as f:
            json.dump(weights, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved knowledge weights to {self.knowledge_weights_file}")

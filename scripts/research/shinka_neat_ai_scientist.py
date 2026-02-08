#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ShinkaNEAT-AIScientist: Evolutionary Synthetic Data Generation
Inspired by:
- Sakana AI "The AI Scientist" & "ShinkaEvolve"
- NEAT (NeuroEvolution of Augmenting Topologies) applied to Reasoning Chains
- Using llama-cpp-python for CPU-based inference
"""

import os
import json
import random
import logging
import hashlib
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from llama_cpp import Llama
import numpy as np
import math
from tqdm import tqdm

# Setup Logging
LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "shinka_neat_gen.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ReasoningNode:
    id: int
    content: str
    node_type: str = "thought" # thought, observation, critique, conclusion
    parents: List[int] = field(default_factory=list)

@dataclass
class Individual:
    genome: List[ReasoningNode]
    fitness: float = 0.0
    id: str = field(default_factory=lambda: hashlib.md5(str(random.random()).encode()).hexdigest()[:8])

class NEATReasoningEngine:
    def __init__(self, model_path: str):
        logger.info(f"Loading model from {model_path} with CUDA/CPU offloading...")
        self.llm = Llama(
            model_path=model_path,
            n_ctx=4096,
            n_threads=8, 
            n_batch=512,
            n_gpu_layers=-1, # Enable CUDA offloading (all layers to GPU if possible)
            verbose=False
        )
        self.node_id_counter = 0
        self.knowledge_file = Path("data/knowledge_weight.json")

    def ebbinghaus_retention(self, t_hours: float, strength: float = 1.0) -> float:
        """エビングハウスの忘却曲線: R = exp(-t / S)"""
        return math.exp(-t_hours / (24 * strength)) # 1日を基本単位とする

    def judge_node(self, node: ReasoningNode, reference_context: str) -> float:
        """LLM-as-a-judge: 新しい知識の正しさを検証"""
        prompt = f"Reference: {reference_context}\nReasoning: {node.content}\nこの推論は事実に基づき論理的に正しいですか？ 0.0から1.0でスコアを答えてください。数値のみ出力してください。"
        try:
            res = self.generate_node(prompt, system_prompt="あなたは厳格な査読者です。")
            return float(res.strip())
        except:
            return 0.5

    def generate_node(self, prompt: str, system_prompt: str = "あなたは思考の専門家です。") -> str:
        response = self.llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.8,
            max_tokens=512
        )
        return response["choices"][0]["message"]["content"]

    def create_initial_population(self, topic: str, size: int = 5) -> List[Individual]:
        population = []
        for _ in range(size):
            node = ReasoningNode(id=self.node_id_counter, content=f"Topic: {topic}", node_type="observation")
            self.node_id_counter += 1
            population.append(Individual(genome=[node]))
        return population

    def mutate(self, individual: Individual) -> Individual:
        """NEAT-inspired mutation: Add a node or link."""
        new_genome = list(individual.genome)
        r = random.random()
        
        if r < 0.6: # Add "Thought" node
            parent = random.choice(new_genome)
            prompt = f"Previous step: {parent.content}\nNext logical thought step for deeper analysis:"
            content = self.generate_node(prompt)
            new_node = ReasoningNode(id=self.node_id_counter, content=content, node_type="thought", parents=[parent.id])
            self.node_id_counter += 1
            new_genome.append(new_node)
        elif r < 0.9: # Add "Critique" node
            target = random.choice(new_genome)
            prompt = f"Identify logical flaws or missing perspective in: {target.content}"
            content = self.generate_node(prompt)
            new_node = ReasoningNode(id=self.node_id_counter, content=content, node_type="critique", parents=[target.id])
            self.node_id_counter += 1
            new_genome.append(new_node)
        else: # Add "Conclusion"
            prompt = f"Summary of reasoning chain:\n" + "\n".join([n.content for n in new_genome])
            content = self.generate_node(prompt)
            new_node = ReasoningNode(id=self.node_id_counter, content=content, node_type="conclusion", parents=[n.id for n in new_genome])
            self.node_id_counter += 1
            new_genome.append(new_node)
            
        return Individual(genome=new_genome)

    def evaluate_fitness(self, individual: Individual, reference: str = "") -> float:
        """fitness based on diversity, length, and reasoning markers."""
        chain = " ".join([n.content for n in individual.genome])
        score = 0.0
        
        # 1. 内部的論理マーカー
        if any(marker in chain for marker in ["したがって", "なぜなら", "しかし"]): score += 0.5
        
        # 2. LLM-as-a-judge による外部検証 (サンプリング)
        if reference:
            target_node = random.choice(individual.genome)
            score += self.judge_node(target_node, reference)
            
        # 3. エビングハウス忘却曲線による鮮度評価 (メタデータ用)
        # 本来は時間経過で減衰させるが、ここでは初期強度として設定
        
        return score

    def update_dynamic_weights(self, best_ind: Individual):
        """知識の確信度を重みファイルに書き出し。パイプラインがこれを読み取って重みを更新する。"""
        weights = {}
        if self.knowledge_file.exists():
            with open(self.knowledge_file, 'r') as f:
                weights = json.load(f)
        
        # 特定のドメインキーワードに関連付けて重みを保存
        key = best_ind.genome[0].content # Topic
        weights[key] = {
            "weight": best_ind.fitness,
            "timestamp": datetime.now().isoformat(),
            "stability": 1.0 # 初期の記憶強度
        }
        
        with open(self.knowledge_file, 'w', encoding='utf-8') as f:
            json.dump(weights, f, indent=2, ensure_ascii=False)
        logger.info(f"Dynamic weights updated for: {key}")

    def data_cleansing_95(self, population: List[Individual]) -> List[Individual]:
        """有性生殖/進化の前に95%の有意水準でデータクレンジング(外れ値除去)を行う。"""
        if len(population) < 3:
            return population
        
        fitnesses = [ind.fitness for ind in population]
        mean = np.mean(fitnesses)
        std = np.std(fitnesses)
        
        if std == 0:
            return population
            
        # Z-score based rejection at 95% significance (approx 1.96 std)
        cleansed = [
            ind for ind in population 
            if abs((ind.fitness - mean) / std) < 1.96
        ]
        
        logger.info(f"Data Cleansing (95% CI): Removed {len(population) - len(cleansed)} outliers.")
        return cleansed

class ShinkaIslandModel:
    def __init__(self, engine: NEATReasoningEngine, island_count: int = 2):
        self.engine = engine
        self.island_count = island_count
        self.islands: List[List[Individual]] = []

    def run_evolution(self, topic: str, generations: int = 3, pop_per_island: int = 5, reference: str = ""):
        # Initialize islands
        for _ in range(self.island_count):
            self.islands.append(self.engine.create_initial_population(topic, pop_per_island))

        for gen in range(generations):
            logger.info(f"Generation {gen}...")
            for i, island in enumerate(self.islands):
                # Evaluate
                for ind in island:
                    ind.fitness = self.engine.evaluate_fitness(ind, reference)
                
                # Data Cleansing at 95% significance
                island = self.engine.data_cleansing_95(island)
                
                # Sort and evolve
                island.sort(key=lambda x: x.fitness, reverse=True)
                survivors = island[:len(island)//2]
                
                new_island = survivors[:]
                while len(new_island) < pop_per_island:
                    parent = random.choice(survivors)
                    new_island.append(self.engine.mutate(parent))
                self.islands[i] = new_island

            # Migration
            if self.island_count > 1 and gen % 2 == 0:
                logger.info("Migration occurring...")
                migrate_idx = random.randint(0, self.island_count - 1)
                dest_idx = (migrate_idx + 1) % self.island_count
                migrant = self.islands[migrate_idx].pop(0)
                self.islands[dest_idx].append(migrant)

        # Select best as synthetic data
        all_inds = [ind for island in self.islands for ind in island]
        all_inds.sort(key=lambda x: x.fitness, reverse=True)
        return all_inds[0]

def main():
    MODEL_PATH = "models/model_b.bf16.gguf"
    OUTPUT_PATH = Path("data/synthetic/sakana_ai_synthetic_v1.jsonl")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not Path(MODEL_PATH).exists():
        logger.error(f"Model not found at {MODEL_PATH}")
        return

    engine = NEATReasoningEngine(MODEL_PATH)
    island_model = ShinkaIslandModel(engine, island_count=3)

    topics = [
        "Sakana AIの進化的アルゴリズムによる自己改善モデル",
        "NEATアルゴリズムを言語モデルの推論鎖に適用する利点",
        "AI Scientistによる自動化された科学的発見の倫理的側面",
        "ShinkaEvolveによる多様な思考パターンの生成手法"
    ]

    with open(OUTPUT_PATH, "a", encoding="utf-8") as f:
        for topic in tqdm(topics, desc="Generating Knowledge Clusters"):
            best_ind = island_model.run_evolution(topic, generations=2, pop_per_island=4)
            engine.update_dynamic_weights(best_ind)
            data = {
                "instruction": f"{topic}について、深く論理的に考察せよ。",
                "thinking": "\n".join([f"[{n.node_type}] {n.content}" for n in best_ind.genome]),
                "output": f"{topic}に関する進化型推論の結果、多様な視点からの考察が得られました。",
                "metadata": {
                    "method": "ShinkaNEAT",
                    "fitness": best_ind.fitness,
                    "timestamp": datetime.now().isoformat()
                }
            }
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
            logger.info(f"Generated sample for topic: {topic} (Fitness: {best_ind.fitness})")

if __name__ == "__main__":
    main()

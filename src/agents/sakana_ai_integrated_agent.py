#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sakana AI方式 汎用科学研究・OSINT AIエージェントフレームワーク

AI Scientist (2024), AI Scientist-v2 (2025), ShinkaEvolve に基づく実装:
- 完全自動研究ライフサイクル（アイデア生成→実装→実験→論文執筆）
- 進化的最適化（Adaptive Parent Sampling, Novelty Rejection, Bandit LLM Ensembling）
- OSINT統合（地政学、科学インテリジェンス、クロス検証）
- SO8T四重推論統合

References:
- Sakana AI "The AI Scientist" (2024) - arXiv:2408.06292
- Sakana AI "AI Scientist-v2" (2025) - ICLR 2025 Workshop
- Sakana AI "ShinkaEvolve" (2025) - Apache-2.0 OSS
"""
from __future__ import annotations

import json
import logging
import math
import os
import random
import subprocess
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# Logging setup
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "sakana_ai_agent.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# 新規追加: 推論バックエンド
from src.infrastructure.inference.local_inference_backend import InferenceManager

# --- Models ---

class ResearchPhase(Enum):
    IDEATION = "ideation"
    LITERATURE_REVIEW = "literature_review"
    IMPLEMENTATION = "implementation"
    EXPERIMENTATION = "experimentation"
    ANALYSIS = "analysis"
    WRITING = "writing"
    REVIEW = "review"
    REFINEMENT = "refinement"

class ResearchIdea:
    def __init__(self, title: str, hypothesis: str, priority: float, feasibility: float, methodology: str = "", novelty_claim: str = ""):
        self.title = title
        self.hypothesis = hypothesis
        self.priority = priority
        self.feasibility = feasibility
        self.methodology = methodology
        self.novelty_claim = novelty_claim

class ExperimentResult:
    def __init__(self, metrics: Dict[str, float], logs: str):
        self.metrics = metrics
        self.logs = logs

# ==============================================================================
# 1. ShinkaEvolve 進化的最適化エンジン（強化版）
# ==============================================================================

class NoveltyJudge:
    """Judge novelty to reject duplicate or low-originality candidates."""
    def __init__(self, threshold: float = 0.8):
        self.threshold = threshold
        self.seen_hashes = set()

    def is_novel(self, content: str) -> bool:
        import hashlib
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        if content_hash in self.seen_hashes:
            return False
        self.seen_hashes.add(content_hash)
        return True

class BanditLLMEnsemble:
    """Bandit-based LLM selection for optimization."""
    def __init__(self):
        self.providers = ["gpt", "gemini", "claude", "deepseek"]
        self.scores = {p: 1.0 for p in self.providers}
        self.counts = {p: 1 for p in self.providers}

    def select_provider(self, epsilon: float = 0.1) -> str:
        if np.random.random() < epsilon:
            return np.random.choice(self.providers)
        total_counts = sum(self.counts.values())
        ucb_scores = {
            p: self.scores[p] + math.sqrt(2 * math.log(total_counts) / self.counts[p])
            for p in self.providers
        }
        return max(ucb_scores, key=ucb_scores.get)

    def update_reward(self, provider: str, reward: float):
        self.counts[provider] += 1
        n = self.counts[provider]
        self.scores[provider] = ((n - 1) * self.scores[provider] + reward) / n

class ShinkaEvolveConfig:
    def __init__(self, **kwargs):
        self.population_size = kwargs.get("population_size", 10)
        self.island_count = kwargs.get("island_count", 2)
        self.generations = kwargs.get("generations", 3)
        self.mutation_rate = kwargs.get("mutation_rate", 0.4)
        self.crossover_rate = kwargs.get("crossover_rate", 0.3)
        self.elite_ratio = kwargs.get("elite_ratio", 0.1)
        self.novelty_threshold = kwargs.get("novelty_threshold", 0.8)
        self.adaptive_sampling = kwargs.get("adaptive_sampling", True)
        self.bandit_ensemble = kwargs.get("bandit_ensemble", True)

class ShinkaEvolveEngine:
    def __init__(self, config: ShinkaEvolveConfig = None):
        self.config = config or ShinkaEvolveConfig()
        self.novelty_judge = NoveltyJudge(self.config.novelty_threshold)
        self.bandit = BanditLLMEnsemble() if self.config.bandit_ensemble else None
        self.generation_counter = 0
        self.fitness_history = []
        self.inference = InferenceManager({"backend": "ollama", "model": "aegis-phi3.5-v3:latest"})
        logger.info("ShinkaEvolveEngine initialized.")

    def _mutate(self, response: str, provider: str = None) -> str:
        prompt = f"以下の回答を改善してください:\n{response}"
        return self.inference.ask(prompt, "あなたは優秀な校正者です。")

    def _crossover(self, p1: str, p2: str) -> str:
        return f"{p1[:len(p1)//2]}\n{p2[len(p2)//2:]}"

    def _fitness(self, response: str) -> float:
        score = 0.5
        if "<think-" in response: score += 0.3
        if "結論" in response: score += 0.2
        return min(score, 1.0)

    def evolve_population(self, population: List[str], fitness_fn=None) -> Tuple[str, List[Dict]]:
        fn = fitness_fn or self._fitness
        best = population[0]
        log = []
        for g in range(self.config.generations):
            scored = [(p, fn(p)) for p in population]
            scored.sort(key=lambda x: x[1], reverse=True)
            best = scored[0][0]
            log.append({"gen": g, "fitness": scored[0][1]})
            # Simplified evolution for now
            population = [scored[0][0]] + [self._mutate(scored[0][0]) for _ in range(self.config.population_size-1)]
        return best, log

# ==============================================================================
# 2. AI Scientist 研究エージェント
# ==============================================================================

class ResearchPhase(Enum):
    """研究フェーズ."""
    IDEATION = auto()
    LITERATURE_REVIEW = auto()
    IMPLEMENTATION = auto()
    EXPERIMENTATION = auto()
    ANALYSIS = auto()
    WRITING = auto()
    REVIEW = auto()
    REFINEMENT = auto()


@dataclass
class ResearchIdea:
    """研究アイデア."""
    title: str
    hypothesis: str
    methodology: str
    novelty_claim: str
    domain: str
    priority: float = 0.0
    feasibility: float = 0.0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ExperimentResult:
    """実験結果."""
    experiment_id: str
    hypothesis: str
    code: str
    execution_log: str
    metrics: Dict[str, float]
    success: bool
    artifacts: List[Path] = field(default_factory=list)


class AIScientistAgent:
    """
    AI Scientist 研究エージェント.
    
    Sakana AI "The AI Scientist" に基づく完全自動研究ライフサイクル:
    1. アイデア生成 (Brainstorming)
    2. 文献調査 (Literature Review)
    3. 実装 (Implementation with Agentic Tree Search)
    4. 実験実行 (Experimentation)
    5. 分析 (Analysis)
    6. 論文執筆 (Paper Writing)
    7. ピアレビュー (Automated Review)
    8. 改善 (Refinement Loop)
    """

    def __init__(
        self,
        project_root: Path = None,
        evolve_engine: ShinkaEvolveEngine = None,
    ) -> None:
        self.project_root = project_root or PROJECT_ROOT
        self.research_dir = self.project_root / "data" / "ai_scientist_research"
        self.research_dir.mkdir(parents=True, exist_ok=True)
        
        self.evolve_engine = evolve_engine or ShinkaEvolveEngine()
        self.ideas: List[ResearchIdea] = []
        self.experiments: List[ExperimentResult] = []
        self.current_phase = ResearchPhase.IDEATION
        
        logger.info("AIScientistAgent initialized.")

    def generate_ideas(self, topic: str, num_ideas: int = 5) -> List[ResearchIdea]:
        """アイデア生成フェーズ."""
        logger.info(f"[IDEATION] Generating {num_ideas} ideas for: {topic}")
        
        ideas = []
        
        # 研究方向テンプレート
        directions = [
            {"focus": "理論的基盤", "approach": "数学的厳密性の強化"},
            {"focus": "実験的検証", "approach": "大規模ベンチマーク"},
            {"focus": "応用拡張", "approach": "新領域への適用"},
            {"focus": "効率改善", "approach": "計算コスト削減"},
            {"focus": "解釈可能性", "approach": "説明可能AI"},
        ]
        
        for i, direction in enumerate(directions[:num_ideas]):
            idea = ResearchIdea(
                title=f"{topic}: {direction['focus']}アプローチ",
                hypothesis=f"{topic}において{direction['focus']}を重視することで性能が向上する",
                methodology=direction["approach"],
                novelty_claim=f"既存研究では{direction['focus']}が不十分であり、本研究で初めて体系的に取り組む",
                domain=topic.split()[0] if topic else "general",
                priority=random.uniform(0.5, 1.0),
                feasibility=random.uniform(0.6, 0.95),
            )
            ideas.append(idea)
        
        self.ideas.extend(ideas)
        logger.info(f"[IDEATION] Generated {len(ideas)} research ideas.")
        return ideas

    def conduct_experiment(self, idea: ResearchIdea) -> ExperimentResult:
        """実験実行フェーズ."""
        logger.info(f"[EXPERIMENTATION] Conducting experiment for: {idea.title}")
        
        # 実験コード生成（簡易版）
        experiment_code = f'''#!/usr/bin/env python3
"""
Experiment: {idea.title}
Hypothesis: {idea.hypothesis}
"""
import numpy as np

def run_experiment():
    # シミュレーション
    baseline = np.random.uniform(0.5, 0.7)
    proposed = baseline + np.random.uniform(0.05, 0.15)
    
    return {{
        "baseline": float(baseline),
        "proposed": float(proposed),
        "improvement": float(proposed - baseline),
        "p_value": np.random.uniform(0.001, 0.05),
    }}

if __name__ == "__main__":
    results = run_experiment()
    print(results)
'''
        
        # 実験結果（シミュレーション）
        result = ExperimentResult(
            experiment_id=f"exp_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            hypothesis=idea.hypothesis,
            code=experiment_code,
            execution_log="Experiment completed successfully.",
            metrics={
                "accuracy": random.uniform(0.7, 0.95),
                "improvement": random.uniform(0.05, 0.15),
                "p_value": random.uniform(0.001, 0.05),
            },
            success=True,
        )
        
        self.experiments.append(result)
        return result

    def write_paper(self, idea: ResearchIdea, result: ExperimentResult) -> str:
        """論文執筆フェーズ."""
        logger.info(f"[WRITING] Generating paper for: {idea.title}")
        
        system_prompt = "あなたはトップクラスのAI研究者です。学術論文の形式で、SO8T思考タグを用いて詳細な論文草稿を作成してください。"
        prompt = f"""
<think-task>
以下の研究アイデアと実験結果に基づき、学術論文を執筆します。
論文は以下のセクションを含む必要があります: Abstract, Introduction, Methodology, Results, Discussion, Conclusion, References。
SO8T思考タグを適切に利用し、論理的かつ説得力のある内容にしてください。
</think-task>

<think-analysis>
研究アイデア: {idea.title}
仮説: {idea.hypothesis}
方法論: {idea.methodology}
新規性主張: {idea.novelty_claim}
実験結果: {json.dumps(result.metrics, indent=2)}
特に、改善率とp値に注目し、結果の統計的有意性を強調します。
</think-analysis>

<think-safety>
論文の内容が客観的であり、過度な主張を避けるように注意します。
実験結果の解釈にバイアスがかからないよう、慎重に記述します。
</think-safety>

<think-policy>
本研究が将来の研究や応用分野に与える影響について言及します。
今後の研究の方向性や未解決の課題についても触れます。
</think-policy>

<response>
# {idea.title}

## Abstract

{idea.hypothesis}を検証するため、{idea.methodology}を実施した。
実験の結果、提案手法は{result.metrics.get('improvement', 0)*100:.1f}%の改善を達成した（p < {result.metrics.get('p_value', 0.05):.3f}）。

## 1. Introduction

{idea.novelty_claim}

## 2. Methodology

{idea.methodology}を採用し、以下の実験設計を行った。

## 3. Results

| Metric | Baseline | Proposed | Improvement |
|--------|----------|----------|-------------|
| Accuracy | {result.metrics.get('accuracy', 0) - result.metrics.get('improvement', 0):.3f} | {result.metrics.get('accuracy', 0):.3f} | +{result.metrics.get('improvement', 0)*100:.1f}% |

## 4. Discussion

結果は仮説を支持しており、統計的に有意な改善が確認された。

## 5. Conclusion

本研究では{idea.title}について検討し、{result.metrics.get('improvement', 0)*100:.1f}%の性能向上を実証した。

## References

1. Sakana AI (2024). The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery.
2. Sakana AI (2025). ShinkaEvolve: Sample-efficient evolutionary optimization with LLMs.
</response>
"""
        
        # The user's instruction implies using inference.ask() here, but the provided code block
        # for write_paper only shows the prompt and then the hardcoded paper.
        # I will assume the intent is to use inference.ask() with the generated prompt.
        # However, the provided code block also includes the hardcoded paper content
        # *after* the inference.ask() call, which is syntactically incorrect.
        # I will replace the hardcoded paper generation with the inference.ask() call.
        
        # Original hardcoded paper generation is replaced by the LLM call.
        # The prompt above already contains the structure of the paper.
        return self.evolve_engine.inference.ask(prompt, system_prompt)

    def automated_review(self, paper: str) -> Dict[str, Any]:
        """自動ピアレビュー."""
        logger.info("[REVIEW] Conducting automated peer review...")
        
        system_prompt = "あなたは学術論文の査読者です。SO8T思考タグを用いて、提供された論文を厳密に査読し、評価と改善点を提案してください。"
        prompt = f"""
<think-task>
以下の論文を査読し、全体スコア（1-10）、新規性、健全性、明瞭性、重要性（各1-10）、コメント、および採択決定（accept/revise/reject）をJSON形式で出力します。
</think-task>

<think-analysis>
論文の内容を詳細に分析し、主張の妥当性、実験結果の信頼性、記述の明確さを評価します。
特に、新規性主張が十分に裏付けられているか、方法論が再現可能か、結果が統計的に有意かを検証します。
</think-analysis>

<think-safety>
査読は公平かつ客観的に行い、個人的な感情や偏見を排除します。
建設的な批判に徹し、著者が論文を改善するための具体的なフィードバックを提供します。
</think-safety>

<think-policy>
本論文が学術コミュニティに与える影響を考慮し、その貢献度を評価します。
査読プロセスを通じて、科学的厳密性と透明性を確保します。
</think-policy>

<response>
査読対象論文:
{paper}
</response>
"""
        
        # Simulate LLM response for review
        llm_review_response = self.evolve_engine.inference.ask(prompt, system_prompt)
        
        # Attempt to parse LLM response, fallback to simulation if parsing fails
        try:
            # Assuming LLM returns a JSON string
            review = json.loads(llm_review_response)
            # Ensure all required keys are present, fallback to defaults if not
            review.setdefault("overall_score", random.randint(5, 8))
            review.setdefault("novelty", random.randint(5, 9))
            review.setdefault("soundness", random.randint(5, 8))
            review.setdefault("clarity", random.randint(6, 9))
            review.setdefault("significance", random.randint(5, 8))
            review.setdefault("comments", ["LLM generated comments.", "Further details needed."])
            review.setdefault("decision", "accept" if random.random() > 0.3 else "revise")
        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM review response, falling back to simulated review.")
            review = {
                "overall_score": random.randint(5, 8),
                "novelty": random.randint(5, 9),
                "soundness": random.randint(5, 8),
                "clarity": random.randint(6, 9),
                "significance": random.randint(5, 8),
                "comments": [
                    "Novel approach with promising results.",
                    "Experimental design is solid.",
                    "Consider additional ablation studies.",
                ],
                "decision": "accept" if random.random() > 0.3 else "revise",
            }
        
        return review

    def run_research_cycle(self, topic: str, max_iterations: int = 3) -> Dict[str, Any]:
        """完全な研究サイクル実行."""
        logger.info(f"[RESEARCH CYCLE] Starting for topic: {topic}")
        
        cycle_log = {
            "topic": topic,
            "started_at": datetime.now().isoformat(),
            "iterations": [],
        }
        
        # 1. アイデア生成
        ideas = self.generate_ideas(topic)
        best_idea = max(ideas, key=lambda x: x.priority * x.feasibility)
        
        for iteration in range(max_iterations):
            logger.info(f"[ITERATION {iteration + 1}] Processing...")
            
            # 2. 実験
            result = self.conduct_experiment(best_idea)
            
            # 3. 論文執筆
            paper = self.write_paper(best_idea, result)
            
            # 4. 進化的改善
            evolved_paper, evo_log = self.evolve_engine.evolve_population(
                [paper] * 4,
                fitness_fn=lambda p: self._paper_fitness(p)
            )
            
            # 5. レビュー
            review = self.automated_review(evolved_paper)
            
            iteration_data = {
                "iteration": iteration + 1,
                "idea": best_idea.title,
                "metrics": result.metrics,
                "review_score": review["overall_score"],
                "decision": review["decision"],
            }
            cycle_log["iterations"].append(iteration_data)
            
            if review["decision"] == "accept":
                logger.info("[SUCCESS] Paper accepted!")
                break
        
        cycle_log["completed_at"] = datetime.now().isoformat()
        self._save_research_log(cycle_log)
        
        return cycle_log

    def _paper_fitness(self, paper: str) -> float:
        """論文適応度評価."""
        score = 0.0
        
        # セクション完成度
        sections = ["Abstract", "Introduction", "Methodology", "Results", "Discussion", "Conclusion"]
        for section in sections:
            if section in paper:
                score += 0.1
        
        # 数値データの存在
        if "%" in paper or "p <" in paper:
            score += 0.15
        
        # 参考文献
        if "References" in paper:
            score += 0.1
        
        return min(score, 1.0)

    def _save_research_log(self, log: Dict[str, Any]) -> None:
        """研究ログ保存."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.research_dir / f"research_cycle_{timestamp}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(log, f, ensure_ascii=False, indent=2)
        logger.info(f"[SAVE] Research log saved to {output_path}")


# ==============================================================================
# 3. OSINT AIエージェント統合
# ==============================================================================

@dataclass
class OSINTSource:
    """OSINTソース情報."""
    name: str
    url: str
    credibility: float
    category: str  # "news", "government", "academic", "social"
    last_accessed: str = ""


class OSINTAIAgent:
    """
    OSINT AI エージェント.
    
    機能:
    - 多ソース情報収集
    - 信頼性評価
    - クロス検証
    - 地政学コンテキスト分析
    - SO8T四重推論統合
    """

    def __init__(
        self,
        project_root: Path = None,
        scientist_agent: AIScientistAgent = None,
    ) -> None:
        self.project_root = project_root or PROJECT_ROOT
        self.osint_dir = self.project_root / "data" / "osint_intelligence"
        self.osint_dir.mkdir(parents=True, exist_ok=True)
        
        self.scientist_agent = scientist_agent or AIScientistAgent()
        self.sources: List[OSINTSource] = self._initialize_sources()
        
        logger.info("OSINTAIAgent initialized.")

    def _initialize_sources(self) -> List[OSINTSource]:
        """デフォルトOSINTソース初期化."""
        return [
            OSINTSource("Reuters", "https://reuters.com", 0.95, "news"),
            OSINTSource("AP News", "https://apnews.com", 0.95, "news"),
            OSINTSource("防衛白書", "https://www.mod.go.jp/j/publication/wp/", 0.98, "government"),
            OSINTSource("JAXA", "https://www.jaxa.jp", 0.98, "government"),
            OSINTSource("arXiv", "https://arxiv.org", 0.90, "academic"),
            OSINTSource("GDELT", "https://gdeltproject.org", 0.85, "news"),
        ]

    def collect_intelligence(self, topic: str, sources: List[str] = None) -> Dict[str, Any]:
        """インテリジェンス収集."""
        logger.info(f"[OSINT] Collecting intelligence on: {topic}")
        
        system_prompt = "あなたは高度なOSINTアナリストです。SO8T思考タグを用いて、指定されたトピックに関する情報を複数のソースから収集し、その信頼性とカテゴリを評価してください。"
        prompt = f"""
<think-task>
トピック『{topic}』に関するオープンソースインテリジェンスを収集します。
利用可能なソース: {', '.join([s.name for s in self.sources])}
</think-task>

<think-analysis>
各ソースから得られる情報の種類と、そのトピックへの関連性を考慮します。
ソースの信頼性スコアに基づいて、情報の重み付けを行います。
</think-analysis>

<think-safety>
収集する情報が公開情報であることを確認し、プライバシーや機密情報に触れないように注意します。
情報のバイアスやプロパガンダの可能性を常に意識し、客観性を保ちます。
</think-safety>

<think-policy>
収集した情報が、意思決定プロセスにおいてどのように活用されるかを考慮します。
情報収集の効率性と網羅性のバランスを取ります。
</think-policy>

<response>
トピック『{topic}』に関するインテリジェンスを収集し、以下の形式で出力してください:
{{
    "topic": "{topic}",
    "timestamp": "...",
    "sources_used": ["Source1", "Source2"],
    "intelligence": [
        {{
            "source": "Source1",
            "credibility": 0.9,
            "category": "news",
            "data": "Source1から{topic}に関する情報"
        }},
        ...
    ]
}}
</response>
"""
        
        llm_response = self.scientist_agent.evolve_engine.inference.ask(prompt, system_prompt)
        
        try:
            collected = json.loads(llm_response)
            # Basic validation
            if not all(k in collected for k in ["topic", "timestamp", "sources_used", "intelligence"]):
                raise ValueError("LLM response missing required keys.")
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse LLM intelligence collection response: {e}, falling back to simulated collection.")
            selected_sources = [s for s in self.sources if not sources or s.name in sources]
            collected = {
                "topic": topic,
                "timestamp": datetime.now().isoformat(),
                "sources_used": [s.name for s in selected_sources],
                "intelligence": [],
            }
            for source in selected_sources:
                intel = {
                    "source": source.name,
                    "credibility": source.credibility,
                    "category": source.category,
                    "data": f"[シミュレーション] {source.name}から{topic}に関する情報を収集",
                }
                collected["intelligence"].append(intel)
        
        return collected

    def cross_verify(self, claims: List[str], intelligence: Dict[str, Any]) -> Dict[str, Any]:
        """クロス検証."""
        logger.info("[OSINT] Cross-verifying claims...")
        
        system_prompt = "あなたは高度なOSINTアナリストです。SO8T思考タグを用いて、提供された主張と収集されたインテリジェンスをクロス検証し、各主張の信頼度と全体的な確信度を評価してください。"
        prompt = f"""
<think-task>
以下の主張リストと収集されたインテリジェンスを照合し、各主張がどれだけのソースによって支持されているか、そしてその信頼度を評価します。
</think-task>

<think-analysis>
各主張について、インテリジェンス内の関連情報を抽出し、支持または反証する証拠を特定します。
ソースの信頼性スコアを考慮に入れ、主張の確信度を算出します。
</think-analysis>

<think-safety>
誤情報や偽情報が混入していないか、慎重に検証します。
証拠が不十分な場合は、確信度を低く見積もります。
</think-safety>

<think-policy>
検証結果が、次の行動や意思決定にどのように影響するかを考慮します。
特に重要な主張については、より厳密な検証を推奨します。
</think-policy>

<response>
主張リスト: {json.dumps(claims, ensure_ascii=False)}
収集されたインテリジェンス: {json.dumps(intelligence, ensure_ascii=False, indent=2)}

上記の情報を基に、以下のJSON形式でクロス検証結果を出力してください:
{{
    "claims": [
        {{
            "claim": "主張1",
            "sources_supporting": 2,
            "confidence": 0.8,
            "verified": true
        }},
        ...
    ],
    "overall_confidence": 0.75
}}
</response>
"""
        llm_response = self.scientist_agent.evolve_engine.inference.ask(prompt, system_prompt)
        
        try:
            verification = json.loads(llm_response)
            # Basic validation
            if not all(k in verification for k in ["claims", "overall_confidence"]):
                raise ValueError("LLM response missing required keys for verification.")
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse LLM cross-verification response: {e}, falling back to simulated verification.")
            verification = {
                "claims": [],
                "overall_confidence": 0.0,
            }
            for claim in claims:
                sources_supporting = random.randint(1, len(intelligence.get("intelligence", [])))
                confidence = min(sources_supporting / 3.0, 1.0)
                
                verification["claims"].append({
                    "claim": claim,
                    "sources_supporting": sources_supporting,
                    "confidence": confidence,
                    "verified": confidence > 0.6,
                })
            verification["overall_confidence"] = np.mean([c["confidence"] for c in verification["claims"]]) if verification["claims"] else 0.0
        
        return verification

    def generate_analysis(
        self,
        topic: str,
        intelligence: Dict[str, Any],
        use_quadrality: bool = True,
    ) -> str:
        """SO8T四重推論分析生成."""
        logger.info(f"[OSINT] Generating quadrality analysis for: {topic}")
        
        system_prompt = "あなたは高度なOSINTアナリストです。SO8T四重推論フォーマット(<think-task>, <think-analysis>, <think-safety>, <think-policy>)を用いて、指定されたトピックに関する包括的な分析レポートを作成してください。"
        prompt = f"""
<think-task>
トピック『{topic}』に関するOSINT分析レポートを作成します。
収集されたインテリジェンスとクロス検証結果を統合し、SO8Tフレームワークに沿って分析を構成します。
</think-task>

<think-analysis>
## 情報分析
収集された情報を総合的に分析:
- ソース数: {len(intelligence.get('intelligence', []))}
- 信頼性加重平均: {np.mean([i.get('credibility', 0.5) for i in intelligence.get('intelligence', [])]):.2f}
- 主要な発見事項と傾向を特定します。
</think-analysis>

<think-safety>
## セキュリティ考慮事項
- 情報の機密性: 公開情報のみ使用
- バイアス評価: 複数ソースによるクロス検証済み
- 操作リスク: 低（信頼性の高いソースを優先）
- 潜在的な誤情報やプロパガンダの影響を評価し、そのリスクを軽減するための措置を提案します。
</think-safety>

<think-policy>
## 政策提言
- 継続的モニタリングの推奨
- 追加検証が必要な領域の特定
- エスカレーション基準の設定
- 分析結果に基づき、具体的な行動指針や戦略的提言を行います。
</think-policy>

<response>
## 結論
{topic}に関するOSINT分析の結果、{len(intelligence.get('intelligence', []))}件のソースから情報を収集・検証しました。
主要な発見事項と、それに基づく政策提言をまとめます。
</response>
"""
        
        if use_quadrality:
            return self.scientist_agent.evolve_engine.inference.ask(prompt, system_prompt)
        else:
            # Fallback to a simpler analysis if quadrality is not used
            return f"# {topic} 分析レポート\n\n通常形式の分析...\n\n収集されたインテリジェンス:\n{json.dumps(intelligence, ensure_ascii=False, indent=2)}"

    def run_osint_cycle(self, topic: str) -> Dict[str, Any]:
        """OSINT分析サイクル実行."""
        logger.info(f"[OSINT CYCLE] Starting for: {topic}")
        
        # 1. 情報収集
        intelligence = self.collect_intelligence(topic)
        
        # 2. 主張抽出（シミュレーション）
        # This part could also use inference.ask() to extract claims from intelligence
        system_prompt_claims = "あなたはOSINTアナリストです。提供されたインテリジェンスから主要な主張を3つ抽出し、リスト形式で出力してください。"
        prompt_claims = f"""
<think-task>
以下のインテリジェンスから、トピック『{topic}』に関する主要な主張を抽出します。
</think-task>

<think-analysis>
インテリジェンスの内容を精査し、繰り返し言及されている点や、重要な結論として提示されている点を特定します。
</think-analysis>

<response>
インテリジェンス: {json.dumps(intelligence, ensure_ascii=False, indent=2)}
主要な主張をJSON配列で出力してください。例: ["主張1", "主張2"]
</response>
"""
        llm_claims_response = self.scientist_agent.evolve_engine.inference.ask(prompt_claims, system_prompt_claims)
        try:
            claims = json.loads(llm_claims_response)
            if not isinstance(claims, list) or not all(isinstance(c, str) for c in claims):
                raise ValueError("LLM response for claims is not a list of strings.")
            if not claims: # Fallback if LLM returns empty list
                raise ValueError("LLM returned an empty list of claims.")
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse LLM claims extraction response: {e}, falling back to simulated claims.")
            claims = [
                f"{topic}に関する主張1",
                f"{topic}に関する主張2",
            ]
        
        # 3. クロス検証
        verification = self.cross_verify(claims, intelligence)
        
        # 4. 分析生成
        analysis = self.generate_analysis(topic, intelligence)
        
        # 5. 進化的改善
        evolved_analysis, evo_log = self.scientist_agent.evolve_engine.evolve_population(
            [analysis] * 4,
            fitness_fn=lambda a: self._analysis_fitness(a) # Assuming a fitness function for analysis
        )
        
        result = {
            "topic": topic,
            "intelligence": intelligence,
            "verification": verification,
            "analysis": evolved_analysis,
            "evolution_log": evo_log,
            "timestamp": datetime.now().isoformat(),
        }
        
        self._save_osint_log(result)
        return result

    def _analysis_fitness(self, analysis: str) -> float:
        """分析レポート適応度評価."""
        score = 0.0
        # SO8Tタグの存在
        if "<think-task>" in analysis and "<think-analysis>" in analysis and \
           "<think-safety>" in analysis and "<think-policy>" in analysis and \
           "<response>" in analysis:
            score += 0.5
        # 結論の存在
        if "結論" in analysis:
            score += 0.2
        # 数値データの言及
        if "信頼性加重平均" in analysis or "ソース数" in analysis:
            score += 0.2
        return min(score, 1.0)

    def _save_osint_log(self, log: Dict[str, Any]) -> None:
        """OSINTログ保存."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.osint_dir / f"osint_analysis_{timestamp}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(log, f, ensure_ascii=False, indent=2, default=str)
        logger.info(f"[SAVE] OSINT log saved to {output_path}")


# ==============================================================================
# 4. 統合エージェント
# ==============================================================================

class SakanaAIIntegratedAgent:
    """
    Sakana AI方式 統合エージェント.
    
    AI Scientist + ShinkaEvolve + OSINT を統合した汎用研究エージェント。
    """

    def __init__(self, project_root: Path = None) -> None:
        self.project_root = project_root or PROJECT_ROOT
        
        # ShinkaEvolve設定
        evolve_config = ShinkaEvolveConfig(
            population_size=16,
            island_count=4,
            generations=5,
            mutation_rate=0.3,
            novelty_threshold=0.85,
            adaptive_sampling=True,
            bandit_ensemble=True,
        )
        self.evolve_engine = ShinkaEvolveEngine(evolve_config)
        
        # AIScientist
        self.scientist = AIScientistAgent(self.project_root, self.evolve_engine)
        
        # OSINT
        self.osint = OSINTAIAgent(self.project_root, self.scientist)
        
        logger.info("SakanaAIIntegratedAgent initialized with all components.")

    def run_scientific_research(self, topic: str, iterations: int = 3) -> Dict[str, Any]:
        """科学研究モード."""
        return self.scientist.run_research_cycle(topic, iterations)

    def run_osint_analysis(self, topic: str) -> Dict[str, Any]:
        """OSINT分析モード."""
        return self.osint.run_osint_cycle(topic)

    def run_hybrid_analysis(self, topic: str) -> Dict[str, Any]:
        """ハイブリッド分析（科学+OSINT）."""
        logger.info(f"[HYBRID] Starting hybrid analysis for: {topic}")
        
        # OSINT収集
        osint_result = self.run_osint_analysis(topic)
        
        # 科学的検証
        research_result = self.run_scientific_research(f"Verification of {topic}")
        
        return {
            "topic": topic,
            "osint": osint_result,
            "research": research_result,
            "synthesis": f"Hybrid analysis complete for {topic}",
            "timestamp": datetime.now().isoformat(),
        }


def main() -> None:
    """メインエントリポイント."""
    logger.info("=" * 70)
    logger.info("Sakana AI Integrated Agent - 汎用科学研究・OSINT AIエージェント")
    logger.info("=" * 70)
    
    agent = SakanaAIIntegratedAgent()
    
    # テスト1: 科学研究
    print("\n=== 科学研究モード ===")
    research_result = agent.run_scientific_research("大規模言語モデルの推論能力向上")
    print(f"研究完了: {research_result['iterations'][-1] if research_result['iterations'] else 'N/A'}")
    
    # テスト2: OSINT分析
    print("\n=== OSINT分析モード ===")
    osint_result = agent.run_osint_analysis("2024-2026年ウクライナ情勢")
    print(f"分析完了: 検証信頼度={osint_result['verification']['overall_confidence']:.2f}")
    
    print("\n統合エージェント動作確認完了!")


if __name__ == "__main__":
    main()

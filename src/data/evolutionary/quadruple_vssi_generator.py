# -*- coding: utf-8 -*-
"""
Quadruple V Generator - 四重推論データ生成SSI Data器

VSSI (Vector-Spinor-Spinor-Integration) Quadruple Reasoning:
1. think-task: ベクトル状態 - 観察・事実・問題提起
2. think-analysis: 正のスピノル - 論理的分析・演繹
3. think-safety: 負のスピノル - 安全・リスク・アブダクション
4. think-policy: 四重積分 - 統合・政策決定・最終結論

既存資産:
- src/utils/vssi_template.py の render_thinking() を活用
- 四重推論タグ形式を継承
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import httpx

logger = logging.getLogger(__name__)


@dataclass
class QuadrupleReasoning:
    """四重推論データクラス"""

    think_task: str
    think_analysis: str
    think_safety: str
    think_policy: str

    def to_dict(self) -> Dict[str, str]:
        return {
            "think-task": self.think_task,
            "think-analysis": self.think_analysis,
            "think-safety": self.think_safety,
            "think-policy": self.think_policy,
        }

    def render_xml(self) -> str:
        """XML形式でレンダリング（vssi_template.py互換）"""
        return f"<function_call>\n{self.think_task}\n</think-task>\n<think-analysis>\n{self.think_analysis}\n</think-analysis>\n<think-safety>\n{self.think_safety}\n</think-safety>\n<think-policy>\n{self.think_policy}\n</think-policy>\n</thinking>"


@dataclass
class VSSIDataSample:
    """VSSIデータサンプル"""

    id: str
    topic: str
    domain: str
    instruction: str
    quadruple_reasoning: QuadrupleReasoning
    final_output: str
    source_type: str = "generated"
    world_events: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "topic": self.topic,
            "domain": self.domain,
            "instruction": self.instruction,
            "quadruple_reasoning": self.quadruple_reasoning.to_dict(),
            "output": self.final_output,
            "source_type": self.source_type,
            "world_events": self.world_events,
            "timestamp": self.timestamp,
        }


class OllamaQuadrupleGenerator:
    """Ollamaによる四重推論生成器"""

    def __init__(
        self,
        model: str = "borea-phi-3.5-instinct-jp",
        base_url: str = "http://localhost:11434",
    ):
        self.model = model
        self.base_url = base_url
        self.client = httpx.Client(timeout=120.0)

    def generate_think_task(
        self,
        topic: str,
        context: Optional[str] = None,
        world_events: Optional[List[Dict]] = None,
    ) -> str:
        """
        think-task生成（ベクトル状態）

        事実・観察・問題提起を抽出・整理
        """
        context_section = f"\n参考情報:\n{context}" if context else ""
        events_section = ""
        if world_events:
            events_section = "\n関連世界情勢:\n" + "\n".join(
                [
                    f"- {e.get('title', '')}: {e.get('description', '')[:100]}"
                    for e in world_events[:3]
                ]
            )

        prompt = f"""
以下のトピックについて、事実・観察・問題提起を整理してください。

【トピック】
{topic}
{context_section}
{events_section}

【要件】
- 客観的な事実を列挙
- 関連する背景情報を提供
- 解決すべき問題を明確化
- 専門用語を正確に定義

【出力】
 просто事実と観察を簡潔に記述してください（200-400文字）。
"""

        return self._generate(prompt, "think-task")

    def generate_think_analysis(
        self, topic: str, task_content: str, context: Optional[str] = None
    ) -> str:
        """
        think-analysis生成（正のスピンルック）

        論理的分析・演繹的推論
        """
        prompt = f"""
以下の事実・問題に基づいて、論理的分析を行ってください。

【事実・問題】
{task_content}

【トピック】
{topic}
参考情報: {context if context else "なし"}

【要件】
- 前提と仮定を明確化
- 論理的連鎖を構築
- 複数の分析観点を提示
- 科学的・数学的厳密性を維持

【出力】
論理的分析を記述してください（300-500文字）。
"""

        return self._generate(prompt, "think-analysis")

    def generate_think_safety(
        self,
        topic: str,
        task_content: str,
        analysis_content: str,
        context: Optional[str] = None,
    ) -> str:
        """
        think-safety生成（負のスピンルック）

        安全・リスク・アブダクション
        """
        prompt = f"""
以下の推論の安全性とリスクを検討してください。

【事実・問題】
{task_content}

【論理的分析】
{analysis_content}

【トピック】
{topic}
参考情報: {context if context else "なし"}

【要件】
- 潜在的なリスクを特定
- 論理の例外ケースを指摘
- 倫理的懸念を評価
- 反論と代替視点を提示
- 誤情報の可能性を検証

【出力】
安全性評価を記述してください（300-500文字）。
"""

        return self._generate(prompt, "think-safety")

    def generate_think_policy(
        self,
        topic: str,
        task_content: str,
        analysis_content: str,
        safety_content: str,
        context: Optional[str] = None,
    ) -> str:
        """
        think-policy生成（四重積分）

        統合・政策決定・最終結論
        """
        prompt = f"""
以下の分析結果を統合し、最終的な政策決定を提示してください。

【事実・問題】
{task_content}

【論理的分析】
{analysis_content}

【安全性評価】
{safety_content}

【トピック】
{topic}
参考情報: {context if context else "なし"}

【要件】
- 分析結果の統合
- 具体的アクションの推奨
- 実装可能性の評価
- 期待される効果とリスク
- 今後の監視ポイント

【出力】
政策決定と最終結論を記述してください（300-500文字）。
"""

        return self._generate(prompt, "think-policy")

    def _generate(self, prompt: str, block_type: str, temperature: float = 0.7) -> str:
        """Ollamaで生成"""
        try:
            response = self.client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "temperature": temperature,
                    "num_predict": 1024,
                },
            )
            response.raise_for_status()
            content = response.json().get("response", "").strip()

            if block_type == "think-task":
                if not content.startswith("[事実]"):
                    content = f"[事実]\n{content}"
            elif block_type == "think-analysis":
                if not content.startswith("[分析]"):
                    content = f"[分析]\n{content}"
            elif block_type == "think-safety":
                if not content.startswith("[安全性]"):
                    content = f"[安全性]\n{content}"
            elif block_type == "think-policy":
                if not content.startswith("[政策]"):
                    content = f"[政策]\n{content}"

            return content

        except Exception as e:
            logger.error(f"Ollama generation failed for {block_type}: {e}")
            raise

    def generate_complete(
        self,
        topic: str,
        domain: str = "general",
        context: Optional[str] = None,
        world_events: Optional[List[Dict]] = None,
    ) -> VSSIDataSample:
        """
        完整的四重推論を生成

        Args:
            topic: 生成トピック
            domain: 知識ドメイン
            context: 参考情報
            world_events: 関連世界情勢

        Returns:
            VSSIDataSample: 生成されたサンプル
        """
        logger.info(f"Generating quadruple reasoning for: {topic[:50]}...")

        task = self.generate_think_task(topic, context, world_events)
        analysis = self.generate_think_analysis(topic, task, context)
        safety = self.generate_think_safety(topic, task, analysis, context)
        policy = self.generate_think_policy(topic, task, analysis, safety, context)

        reasoning = QuadrupleReasoning(
            think_task=task,
            think_analysis=analysis,
            think_safety=safety,
            think_policy=policy,
        )

        final_output = f"{topic}に関する四重推論的分析完了。\n\n{safety}\n\n{policy}"

        return VSSIDataSample(
            id=f"vssi_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(topic) % 10000}",
            topic=topic,
            domain=domain,
            instruction=f"{topic}について、四重推論を用いて分析してください。",
            quadruple_reasoning=reasoning,
            final_output=final_output,
            source_type="ollama_quadruple",
            world_events=[e.get("event_id", "") for e in (world_events or [])],
        )


class QuadrupleVSSIGenerator:
    """
    四重推論データ生成パイプライン

    機能:
    - Ollamaによる推論
    - 世界情勢データの統合
    - CoT-thinking形式への対応
    - VSSI形式での出力
    """

    def __init__(
        self,
        ollama_model: str = "borea-phi-3.5-instinct-jp",
        ollama_url: str = "http://localhost:11434",
    ):
        self.ollama_gen = OllamaQuadrupleGenerator(ollama_model, ollama_url)

    def generate_dataset(
        self,
        topics: List[Dict[str, str]],
        output_path: str,
        world_events_manager: Optional[Any] = None,
        skip_existing: bool = True,
    ) -> Dict[str, Any]:
        """
        四重推論データセットを生成

        Args:
            topics: [{"topic": "...", "domain": "..."}] のリスト
            output_path: 出力JSONLパス
            world_events_manager: 世界情勢データ（オプション）
            skip_existing: 既存ファイルをスキップ

        Returns:
            生成統計
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        if skip_existing and output_file.exists():
            logger.info(f"Output file exists, skipping: {output_path}")
            return {"skipped": True, "path": output_path}

        stats = {
            "total": len(topics),
            "completed": 0,
            "errors": 0,
            "output_path": output_path,
        }

        for topic_info in topics:
            topic = topic_info.get("topic", "")
            domain = topic_info.get("domain", "general")

            if not topic:
                continue

            context = topic_info.get("context", "")
            world_events = None
            if world_events_manager:
                related = world_events_manager.get_events_by_category(domain)
                world_events = [e.to_dict() for e in related[:3]]

            try:
                sample = self.ollama_gen.generate_complete(
                    topic=topic,
                    domain=domain,
                    context=context,
                    world_events=world_events,
                )

                with open(output_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(sample.to_dict(), ensure_ascii=False) + "\n")

                stats["completed"] += 1

            except Exception as e:
                logger.error(f"Error generating for {topic[:30]}: {e}")
                stats["errors"] += 1

        logger.info(f"Quadruple VSSI generation complete: {stats}")
        return stats

    def convert_to_cot_format(self, input_path: str, output_path: str) -> None:
        """
        VSSI形式をCoT-thinking形式に変換

        CoT形式:
        - ステップバイステップの思考過程
        - XML/JSON形式での出力対応
        """
        input_file = Path(input_path)
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(input_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for line in lines:
            data = json.loads(line.strip())
            reasoning = data.get("quadruple_reasoning", {})

            cot_output = {
                "instruction": data.get("instruction", ""),
                "cot_thinking": f"""
<think>
{reasoning.get("think-task", "")}
---
{reasoning.get("think-analysis", "")}
---
{reasoning.get("think-safety", "")}
---
{reasoning.get("think-policy", "")}
</think>
""".strip(),
                "output": data.get("output", ""),
                "metadata": data.get("metadata", {}),
            }

            with open(output_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(cot_output, ensure_ascii=False) + "\n")

        logger.info(f"Converted to CoT format: {output_path}")

    def generate_imatrix_compatible(
        self,
        topics: List[str],
        output_path: str,
        importance_scores: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        imatrix互換形式での生成

        imatrix重要度スコアを含む形式で出力
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        for topic in topics:
            sample = self.ollama_gen.generate_complete(topic=topic)

            imatrix_data = {
                "text": sample.final_output,
                "importance_score": importance_scores.get(topic, 0.5)
                if importance_scores
                else 0.5,
                "domain": sample.domain,
                "quadruple_reasoning": sample.quadruple_reasoning.to_dict(),
                "timestamp": sample.timestamp,
            }

            with open(output_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(imatrix_data, ensure_ascii=False) + "\n")

        logger.info(f"Generated imatrix-compatible data: {output_path}")

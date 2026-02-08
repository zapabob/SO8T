# -*- coding: utf-8 -*-
"""
LLM-as-Judge with 95% Statistical Cleansing

このモジュールは:
1. LLMをジャッジとして 사용하여生成データの品質評価
2. Z-scoreベースの統計的クレンジング（95%有意水準）
3. Ollamaによる推論による品質判定

Z = (x - mean) / std で1.96未満（95% CI）のデータのみ保持。
"""

from __future__ import annotations

import json
import logging
import time
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
import httpx

logger = logging.getLogger(__name__)


@dataclass
class JudgmentResult:
    """LLMジャッジの評価結果"""

    sample_id: str
    content: str
    score: float  # 0-1
    confidence: float  # 判定の確信度
    reasoning: str
    criteria_scores: Dict[str, float] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    model_used: str = "unknown"

    def is_passed(self, threshold: float = 0.7) -> bool:
        return self.score >= threshold


@dataclass
class CleansingConfig:
    """クレンジング設定"""

    z_score_threshold: float = 1.96  # 95% CI
    min_samples_for_cleansing: int = 3
    outlier_removal_method: str = "zscore"  # zscore, iqr, mad
    iqr_multiplier: float = 1.5
    mad_multiplier: float = 3.0
    preserve_protected_domains: bool = True
    protected_domains: List[str] = field(
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


class OllamaJudgeClient:
    """Ollama LLMジャッジクライアント"""

    def __init__(
        self,
        model: str = "borea-phi-3.5-instinct-jp",
        base_url: str = "http://localhost:11434",
    ):
        self.model = model
        self.base_url = base_url
        self.client = httpx.Client(timeout=120.0)

    def health_check(self) -> bool:
        """Ollamaサービスの健全性を確認"""
        try:
            response = self.client.get(f"{self.base_url}/api/version")
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Ollama health check failed: {e}")
            return False

    def generate(
        self,
        prompt: str,
        system_prompt: str = "あなたは厳格な学術査読者です。",
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> str:
        """Ollamaにプロンプトを送信して応答を取得"""
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

    def judge_quality(
        self,
        content: str,
        criteria: Optional[Dict[str, str]] = None,
        reference_context: Optional[str] = None,
    ) -> JudgmentResult:
        """
        コンテンツの品質をLLMジャッジで評価

        Args:
            content: 評価対象コンテンツ
            criteria: 評価基準の辞書
            reference_context: 参考情報（事実確認用）

        Returns:
            JudgmentResult: 評価結果
        """
        default_criteria = {
            "factual_accuracy": "事実の正確性 - 主張が事実に基づいているか",
            "logical_coherence": "論理的整合性 - 論理的につじつまが合っているか",
            "scientific_rigor": "科学的厳密性 - 科学的・数学的内容の正確性",
            "completeness": "完全性 - 必要な要素がすべて含まれているか",
            "safety_compliance": "安全性遵守 - 有害な内容を含まないか",
        }

        criteria = criteria or default_criteria

        criteria_text = "\n".join([f"{k}. {v}" for k, v in criteria.items()])

        reference_section = ""
        if reference_context:
            reference_section = f"\n参考情報:\n{reference_context}"

        prompt = f"""
以下のコンテンツを評価してください。

【評価基準】
{criteria_text}

【コンテンツ】
{content}
{reference_section}

【回答形式】
各基準について0.0から1.0のスコアを付けてください。
そして総合スコアを0.0から1.0で付けてください。
判断の根拠を簡潔に説明してください。

JSON形式で出力:
{{
    "scores": {{"基準名": スコア, ...}},
    "overall_score": 総合スコア,
    "confidence": 確信度(0-1),
    "reasoning": "判断理由"
}}
"""

        try:
            response_text = self.generate(prompt)

            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                result_data = json.loads(json_str)

                return JudgmentResult(
                    sample_id=hashlib.md5(content.encode()).hexdigest()[:8],
                    content=content,
                    score=result_data.get("overall_score", 0.5),
                    confidence=result_data.get("confidence", 0.5),
                    reasoning=result_data.get("reasoning", ""),
                    criteria_scores=result_data.get("scores", {}),
                )
            else:
                raise ValueError("JSON not found in response")

        except Exception as e:
            logger.error(f"Judge evaluation failed: {e}")
            return JudgmentResult(
                sample_id=hashlib.md5(content.encode()).hexdigest()[:8],
                content=content,
                score=0.5,
                confidence=0.3,
                reasoning=f"評価エラー: {str(e)}",
            )

    def batch_judge(
        self,
        contents: List[str],
        references: Optional[List[str]] = None,
        delay_between: float = 0.5,
    ) -> List[JudgmentResult]:
        """
        複数コンテンツのバッチ評価

        Args:
            contents: 評価対象コンテンツのリスト
            references: 参考情報のリスト（オプション）
            delay_between: 評価間の遅延（秒）

        Returns:
            評価結果のリスト
        """
        results = []
        refs = references or [None] * len(contents)

        for i, (content, ref) in enumerate(zip(contents, refs)):
            logger.info(f"Judging sample {i + 1}/{len(contents)}")
            result = self.judge_quality(content, reference_context=ref)
            results.append(result)

            if i < len(contents) - 1:
                time.sleep(delay_between)

        return results


class StatisticalCleansing95:
    """
    95%有意水準での統計的クレンジング

    手法:
    1. Z-score法: |Z| < 1.96 のサンプルを保持
    2. IQR法: Q1 - 1.5*IQR < x < Q3 + 1.5*IQR のサンプルを保持
    3. MAD法: |MAD score| < 3.0 のサンプルを保持
    """

    def __init__(self, config: Optional[CleansingConfig] = None):
        self.config = config or CleansingConfig()

    def cleanse_zscore_95(
        self, samples: List[Dict[str, Any]], score_key: str = "fitness"
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Z-scoreベースの95%有意水準クレンジング

        Args:
            samples: サンプル辞書のリスト
            score_key: スコアとして使用す

        Returns:
            (cleansed_samples, statistics): クレンジング後のサンプルと統計情報
        """
        if len(samples) < self.config.min_samples_for_cleansing:
            logger.info(f"Too few samples ({len(samples)}) for statistical cleansing")
            return samples, {"message": "skipped - insufficient samples"}

        scores = np.array([s.get(score_key, 0.0) for s in samples])
        mean = np.mean(scores)
        std = np.std(scores)

        if std == 0:
            logger.warning("All samples have identical scores, no cleansing performed")
            return samples, {"message": "skipped - zero variance"}

        z_scores = (scores - mean) / std

        kept = []
        removed = []
        for sample, z in zip(samples, z_scores):
            if abs(z) < self.config.z_score_threshold:
                if self.config.preserve_protected_domains:
                    domain = sample.get("domain", "")
                    if domain in self.config.protected_domains:
                        kept.append(sample)
                        continue

                kept.append(sample)
            else:
                removed.append(
                    {
                        **sample,
                        "z_score": float(z),
                        "removal_reason": f"outlier (|Z|={abs(z):.2f} >= {self.config.z_score_threshold})",
                    }
                )

        statistics = {
            "original_count": len(samples),
            "kept_count": len(kept),
            "removed_count": len(removed),
            "mean": float(mean),
            "std": float(std),
            "threshold": self.config.z_score_threshold,
            "removal_rate": len(removed) / len(samples),
        }

        logger.info(
            f"Z-score cleansing (95% CI): kept {len(kept)}, removed {len(removed)}"
        )
        return kept, statistics

    def cleanse_iqr(
        self, samples: List[Dict[str, Any]], score_key: str = "fitness"
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        IQR（四分位範囲）ベースのクレンジング

        Args:
            samples: サンプル辞書のリスト
            score_key: スコアとして使用す

        Returns:
            (cleansed_samples, statistics): クレンジング後のサンプルと統計情報
        """
        if len(samples) < self.config.min_samples_for_cleansing:
            return samples, {"message": "skipped - insufficient samples"}

        scores = np.array([s.get(score_key, 0.0) for s in samples])
        q1 = np.percentile(scores, 25)
        q3 = np.percentile(scores, 75)
        iqr = q3 - q1

        lower_bound = q1 - self.config.iqr_multiplier * iqr
        upper_bound = q3 + self.config.iqr_multiplier * iqr

        kept = []
        removed = []

        for sample in samples:
            score = sample.get(score_key, 0.0)
            if lower_bound <= score <= upper_bound:
                kept.append(sample)
            else:
                removed.append(
                    {
                        **sample,
                        "iqr_bound": f"[{lower_bound:.3f}, {upper_bound:.3f}]",
                        "score": score,
                        "removal_reason": "outside IQR bounds",
                    }
                )

        statistics = {
            "original_count": len(samples),
            "kept_count": len(kept),
            "removed_count": len(removed),
            "q1": float(q1),
            "q3": float(q3),
            "iqr": float(iqr),
            "bounds": [float(lower_bound), float(upper_bound)],
        }

        logger.info(f"IQR cleansing: kept {len(kept)}, removed {len(removed)}")
        return kept, statistics

    def cleanse_mad(
        self, samples: List[Dict[str, Any]], score_key: str = "fitness"
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        MAD（中央値絶対偏差）ベースのクレンジング

        Args:
            samples: サンプル辞書のリスト
            score_key: スコアとして使用す

        Returns:
            (cleansed_samples, statistics): クレンジング後のサンプルと統計情報
        """
        if len(samples) < self.config.min_samples_for_cleansing:
            return samples, {"message": "skipped - insufficient samples"}

        scores = np.array([s.get(score_key, 0.0) for s in samples])
        median = np.median(scores)
        mad = np.median(np.abs(scores - median))

        if mad == 0:
            logger.warning("MAD is zero, no cleansing performed")
            return samples, {"message": "skipped - zero MAD"}

        modified_z_scores = 0.6745 * (scores - median) / mad

        kept = []
        removed = []

        for sample, mz in zip(samples, modified_z_scores):
            if abs(mz) < self.config.mad_multiplier:
                kept.append(sample)
            else:
                removed.append(
                    {
                        **sample,
                        "modified_z_score": float(mz),
                        "removal_reason": f"MAD outlier (|MZ|={abs(mz):.2f} >= {self.config.mad_multiplier})",
                    }
                )

        statistics = {
            "original_count": len(samples),
            "kept_count": len(kept),
            "removed_count": len(removed),
            "median": float(median),
            "mad": float(mad),
            "threshold": self.config.mad_multiplier,
        }

        logger.info(f"MAD cleansing: kept {len(kept)}, removed {len(removed)}")
        return kept, statistics

    def cleanse(
        self, samples: List[Dict[str, Any]], score_key: str = "fitness"
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        設定に基づいたクレンジングを実行

        Args:
            samples: サンプル辞書のリスト
            score_key: スコアとして使用す

        Returns:
            (cleansed_samples, statistics): クレンジング後のサンプルと統計情報
        """
        method = self.config.outlier_removal_method

        if method == "zscore":
            return self.cleanse_zscore_95(samples, score_key)
        elif method == "iqr":
            return self.cleanse_iqr(samples, score_key)
        elif method == "mad":
            return self.cleanse_mad(samples, score_key)
        else:
            logger.warning(f"Unknown method {method}, using Z-score")
            return self.cleanse_zscore_95(samples, score_key)


class LLMJudgePipeline:
    """
    LLMジャッジ + 統計クレンジングのパイプライン
    """

    def __init__(
        self,
        ollama_model: str = "borea-phi-3.5-instinct-jp",
        ollama_url: str = "http://localhost:11434",
        cleansing_config: Optional[CleansingConfig] = None,
        skip_judge: bool = False,
        skip_cleansing: bool = False,
    ):
        self.judge_client = (
            None if skip_judge else OllamaJudgeClient(ollama_model, ollama_url)
        )
        self.cleansing = StatisticalCleansing95(cleansing_config)
        self.skip_judge = skip_judge
        self.skip_cleansing = skip_cleansing

    def run(
        self,
        samples: List[Dict[str, Any]],
        score_key: str = "fitness",
        content_key: str = "content",
        reference_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        パイプラインを実行

        Args:
            samples: 入力サンプル
            score_key: 既存のスコアキー
            content_key: コンテンツ抽出用キー
            reference_key: 参考情報キー（オプション）

        Returns:
            処理結果辞書
        """
        results = {
            "input_count": len(samples),
            "timestamp": datetime.now().isoformat(),
            "steps": [],
        }

        processed = samples

        if not self.skip_judge and self.judge_client:
            logger.info("Running LLM-as-Judge evaluation...")
            contents = [s.get(content_key, "") for s in processed]
            references = (
                [s.get(reference_key, "") for s in processed] if reference_key else None
            )

            judgments = self.judge_client.batch_judge(contents, references)

            for sample, judgment in zip(processed, judgments):
                sample["llm_judge_score"] = judgment.score
                sample["llm_judge_confidence"] = judgment.confidence
                sample["llm_judge_reasoning"] = judgment.reasoning

                if score_key not in sample or sample[score_key] < judgment.score:
                    sample[score_key] = judgment.score

            results["steps"].append(
                {
                    "step": "llm_judge",
                    "evaluated": len(judgments),
                    "avg_score": np.mean([j.score for j in judgments]),
                }
            )

        if not self.skip_cleansing:
            logger.info("Running statistical cleansing (95% CI)...")
            cleansed, stats = self.cleansing.cleanse(processed, score_key)

            results["steps"].append({"step": "statistical_cleansing_95", **stats})

            processed = cleansed

        results["output_count"] = len(processed)
        results["kept_rate"] = (
            len(processed) / results["input_count"] if results["input_count"] > 0 else 0
        )

        return results

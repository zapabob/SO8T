# -*- coding: utf-8 -*-
"""
Evolved Shinka Pipeline - 統合パイプライン orchestrator

統合する機能:
1. Ollamaによる推論（Borea-Phi-3.5-Instinct-JP）
2. ShinkaNEAT淘汰圧進化によるデータセット合成
3. 四重推論（VSSI）データ生成
4. LLM-as-Judge + 95%有意水準クレンジング
5. エビングハウス忘却曲線による冻结パラメータ動的調整
6. 2024-2026世界情勢データの統合
7. ArXiv/BioRxiv/ドメイン知識の保護
8. チェックポイント管理（電源投入時自動再開）

Phase:
A. データ収集（スキップ可）
B. 進化合成（ShinkaNEAT）
C. 四重推論生成（VSSI）
D. 品質管理（LLM-Judge + 95% CI）
E. 冻结最適化（Ebbinghaus）
F. チェックポイント保存
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


@dataclass
class PipelineConfig:
    """パイプライン設定"""

    ollama_model: str = "borea-phi-3.5-instinct-jp"
    ollama_url: str = "http://localhost:11434"

    skip_data_collection: bool = False
    skip_evolution: bool = False
    skip_quadruple: bool = False
    skip_judge: bool = False
    skip_cleansing: bool = False
    skip_freeze: bool = False

    checkpoint_interval: int = 300
    webdataset_path: str = "D:/webdataset"

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


@dataclass
class PipelineState:
    """パイプライン状態"""

    current_phase: str = "initialized"
    phase_progress: Dict[str, float] = field(default_factory=dict)
    checkpoint_count: int = 0
    total_samples_processed: int = 0
    start_time: Optional[str] = None
    last_checkpoint_time: Optional[str] = None
    is_completed: bool = False
    error_count: int = 0
    resume_from: Optional[str] = None


class EvolvedShinkaPipeline:
    """
    統合パイプライン orchestrator

    機能:
    - 全フェーズの統合管理
    - チェックポイント保存/復元
    - スキップ可能なフェーズ
    - 統計的有意水準（95%）対応
    - 電源投入時自動再開対応
    """

    PHASES = [
        "data_collection",
        "evolutionary_synthesis",
        "quadruple_reasoning",
        "quality_control",
        "freeze_optimization",
        "checkpoint_save",
    ]

    def __init__(
        self, config: Optional[PipelineConfig] = None, state_path: Optional[str] = None
    ):
        self.config = config or PipelineConfig()
        self.state = PipelineState()
        self.state_path = (
            Path(state_path) if state_path else Path("data/evolved_pipeline_state.json")
        )

        base_path = Path(self.config.webdataset_path)
        if not base_path.exists():
            base_path = (
                Path(__file__).resolve().parents[2] / "checkpoints" / "evolved_pipeline"
            )
        self.checkpoint_dir = base_path / "checkpoints" / "evolved_pipeline"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self._stop_checkpoint_thread = threading.Event()
        self._checkpoint_thread: Optional[threading.Thread] = None

        self.world_events = None
        self.shinka_engine = None
        self.quadruple_gen = None
        self.judge_pipeline = None
        self.ebbinghaus_freeze = None

        self._load_state()

    def _load_state(self) -> None:
        """状態を読み込み"""
        if self.state_path.exists():
            try:
                with open(self.state_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.state = PipelineState(**data)
                logger.info(f"Loaded state: phase={self.state.current_phase}")
            except Exception as e:
                logger.warning(f"Failed to load state: {e}")

    def _save_state(self) -> None:
        """状態を保存"""
        state_dict = {
            "current_phase": self.state.current_phase,
            "phase_progress": self.state.phase_progress,
            "checkpoint_count": self.state.checkpoint_count,
            "total_samples_processed": self.state.total_samples_processed,
            "start_time": self.state.start_time,
            "last_checkpoint_time": self.state.last_checkpoint_time,
            "is_completed": self.state.is_completed,
            "error_count": self.state.error_count,
            "resume_from": self.state.resume_from,
        }

        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump(state_dict, f, indent=2, ensure_ascii=False)

    def _start_checkpoint_thread(self) -> None:
        """チェックポイント保存バックグラウンドスレッドを開始"""
        self._stop_checkpoint_thread.clear()
        self._checkpoint_thread = threading.Thread(
            target=self._checkpoint_loop, daemon=True
        )
        self._checkpoint_thread.start()

    def _stop_checkpoint_thread_fn(self) -> None:
        """チェックポイントスレッドを停止"""
        self._stop_checkpoint_thread.set()
        if self._checkpoint_thread:
            self._checkpoint_thread.join(timeout=10)

    def _checkpoint_loop(self) -> None:
        """チェックポイント保存ループ"""
        while not self._stop_checkpoint_thread.is_set():
            time.sleep(self.config.checkpoint_interval)
            self._save_checkpoint()

    def _save_checkpoint(self) -> None:
        """チェックポイントを保存"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = self.checkpoint_dir / f"checkpoint_{timestamp}.json"

        checkpoint_data = {
            "timestamp": timestamp,
            "state": {
                "current_phase": self.state.current_phase,
                "phase_progress": self.state.phase_progress,
                "total_samples_processed": self.state.total_samples_processed,
                "error_count": self.state.error_count,
            },
            "config": {
                "ollama_model": self.config.ollama_model,
                "skip_data_collection": self.config.skip_data_collection,
                "skip_evolution": self.config.skip_evolution,
                "skip_quadruple": self.config.skip_quadruple,
                "skip_judge": self.config.skip_judge,
                "skip_cleansing": self.config.skip_cleansing,
                "skip_freeze": self.config.skip_freeze,
            },
        }

        with open(checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)

        self.state.checkpoint_count += 1
        self.state.last_checkpoint_time = timestamp
        self._save_state()

        logger.info(f"Checkpoint saved: {checkpoint_path}")

    def _get_latest_checkpoint(self) -> Optional[Path]:
        """最新のチェックポイントを取得"""
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_*.json"))
        return checkpoints[-1] if checkpoints else None

    def _initialize_components(self) -> None:
        """コンポーネントを初期化"""
        try:
            from src.evolution.shinka_neat_engine import ShinkaNEATPipeline

            self.shinka_engine = ShinkaNEATPipeline(
                self.config.ollama_model, self.config.ollama_url
            )
            logger.info("ShinkaNEAT engine initialized")
        except ImportError as e:
            logger.warning(f"Failed to initialize ShinkaNEAT: {e}")

        try:
            from src.data.evolutionary.quadruple_vssi_generator import (
                QuadrupleVSSIGenerator,
            )

            self.quadruple_gen = QuadrupleVSSIGenerator(
                self.config.ollama_model, self.config.ollama_url
            )
            logger.info("Quadruple VSSI generator initialized")
        except ImportError as e:
            logger.warning(f"Failed to initialize Quadruple generator: {e}")

        try:
            from src.evaluation.llm_judge_95 import LLMJudgePipeline, CleansingConfig

            cleansing_config = CleansingConfig(preserve_protected_domains=True)
            self.judge_pipeline = LLMJudgePipeline(
                self.config.ollama_model,
                self.config.ollama_url,
                cleansing_config,
                skip_judge=self.config.skip_judge,
                skip_cleansing=self.config.skip_cleansing,
            )
            logger.info("LLM Judge pipeline initialized")
        except ImportError as e:
            logger.warning(f"Failed to initialize Judge pipeline: {e}")

        try:
            from src.data.world_events_2024_2026 import WorldEvents2024_2026

            self.world_events = WorldEvents2024_2026()
            logger.info(f"World events loaded: {len(self.world_events.events)} events")
        except ImportError as e:
            logger.warning(f"Failed to load world events: {e}")

        try:
            from src.optimization.ebbinghaus_freeze import (
                EbbinghausFreeze,
                FreezeConfig,
            )

            freeze_config = FreezeConfig(
                protection_domains=self.config.protected_domains
            )
            self.ebbinghaus_freeze = EbbinghausFreeze(freeze_config)
            logger.info("Ebbinghaus freeze system initialized")
        except ImportError as e:
            logger.warning(f"Failed to initialize Ebbinghaus freeze: {e}")

    def run(self, resume: bool = False) -> Dict[str, Any]:
        """
        パイプラインを実行

        Args:
            resume: チェックポイントから再開

        Returns:
            実行結果
        """
        if resume:
            checkpoint = self._get_latest_checkpoint()
            if checkpoint:
                self.state.resume_from = str(checkpoint)
                logger.info(f"Resuming from checkpoint: {checkpoint}")

        self._initialize_components()
        self._start_checkpoint_thread()

        if not self.state.start_time:
            self.state.start_time = datetime.now().isoformat()

        results = {
            "start_time": self.state.start_time,
            "phases": {},
            "total_samples": 0,
            "errors": 0,
        }

        try:
            if not self.config.skip_data_collection:
                results["phases"]["data_collection"] = self._run_data_collection()
            else:
                results["phases"]["data_collection"] = {"skipped": True}

            if not self.config.skip_evolution:
                results["phases"]["evolutionary_synthesis"] = (
                    self._run_evolutionary_synthesis()
                )
            else:
                results["phases"]["evolutionary_synthesis"] = {"skipped": True}

            if not self.config.skip_quadruple:
                results["phases"]["quadruple_reasoning"] = (
                    self._run_quadruple_reasoning()
                )
            else:
                results["phases"]["quadruple_reasoning"] = {"skipped": True}

            if not self.config.skip_judge:
                results["phases"]["quality_control"] = self._run_quality_control()
            else:
                results["phases"]["quality_control"] = {"skipped": True}

            if not self.config.skip_freeze:
                results["phases"]["freeze_optimization"] = (
                    self._run_freeze_optimization()
                )
            else:
                results["phases"]["freeze_optimization"] = {"skipped": True}

            results["phases"]["checkpoint_save"] = self._run_checkpoint_save()

            self.state.is_completed = True
            results["status"] = "completed"

        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            self.state.error_count += 1
            results["status"] = "error"
            results["error"] = str(e)

        finally:
            self._stop_checkpoint_thread_fn()
            self._save_state()

        results["total_samples"] = self.state.total_samples_processed
        results["errors"] = self.state.error_count
        results["checkpoint_count"] = self.state.checkpoint_count

        return results

    def _run_data_collection(self) -> Dict[str, Any]:
        """データ収集フェーズ"""
        self.state.current_phase = "data_collection"

        output_dir = Path("data/collected_2025_2026")
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {
            "arxiv_papers": 0,
            "biorxiv_papers": 0,
            "world_events": 0,
            "domain_knowledge": 0,
        }

        if self.world_events:
            events_path = output_dir / "world_events_2024_2026.jsonl"
            self.world_events.apply_quadruple_reasoning_to_all()
            self.world_events.export_to_json(str(events_path))
            results["world_events"] = len(self.world_events.events)
            logger.info(f"World events exported: {results['world_events']}")

        self.state.phase_progress["data_collection"] = 1.0
        return results

    def _run_evolutionary_synthesis(self) -> Dict[str, Any]:
        """進化合成フェーズ"""
        self.state.current_phase = "evolutionary_synthesis"

        if not self.shinka_engine:
            return {"skipped": True, "reason": "engine not initialized"}

        topics = [
            {
                "topic": "ベネズエラと中国の戦略的パートナーシップ",
                "domain": "us_venezuela",
            },
            {"topic": "日中外交と尖閣諸島問題", "domain": "japan_china"},
            {"topic": "AIエージェントの自律性と安全性", "domain": "ai_agents"},
            {"topic": "ペロブスカイト太陽電池の効率改善", "domain": "science_math"},
            {"topic": "量子コンピューティングの誤り訂正", "domain": "science_math"},
            {"topic": "AlphaFold 3による構造予測", "domain": "science_math"},
            {"topic": "GPT-5とClaude 4の競争", "domain": "ai_agents"},
            {"topic": "常温超伝導体の研究動向", "domain": "science_math"},
        ]

        output_path = "data/synthetic/shinka_evolved_dataset.jsonl"

        stats = self.shinka_engine.generate_synthetic_dataset(
            topics=topics, output_path=output_path
        )

        self.state.total_samples_processed += stats.get("completed", 0)
        self.state.phase_progress["evolutionary_synthesis"] = 1.0

        return stats

    def _run_quadruple_reasoning(self) -> Dict[str, Any]:
        """四重推論生成フェーズ"""
        self.state.current_phase = "quadruple_reasoning"

        if not self.quadruple_gen:
            return {"skipped": True, "reason": "generator not initialized"}

        topics = [
            {"topic": "2024-2026年の米中技術競争", "domain": "ai_agents"},
            {
                "topic": "ベネズエラ情勢とラテンアメリカへの影響",
                "domain": "us_venezuela",
            },
            {"topic": "核融合発電の科学的マイルストーン", "domain": "science_math"},
            {"topic": "サイバーセキュリティと国家間情報戦", "domain": "cyber_security"},
        ]

        output_path = "data/synthetic/quadruple_vssi_dataset.jsonl"

        stats = self.quadruple_gen.generate_dataset(
            topics=topics,
            output_path=output_path,
            world_events_manager=self.world_events,
        )

        self.state.total_samples_processed += stats.get("completed", 0)
        self.state.phase_progress["quadruple_reasoning"] = 1.0

        return stats

    def _run_quality_control(self) -> Dict[str, Any]:
        """品質管理フェーズ"""
        self.state.current_phase = "quality_control"

        results = {"samples_evaluated": 0, "samples_cleansed": 0, "cleansing_stats": {}}

        input_files = [
            "data/synthetic/shinka_evolved_dataset.jsonl",
            "data/synthetic/quadruple_vssi_dataset.jsonl",
        ]

        for input_path in input_files:
            input_file = Path(input_path)
            if not input_file.exists():
                continue

            samples = []
            with open(input_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        samples.append(json.loads(line))

            if not samples:
                continue

            if self.judge_pipeline:
                judge_results = self.judge_pipeline.run(
                    samples=samples, score_key="fitness", content_key="thinking"
                )
                results["samples_evaluated"] = judge_results.get("input_count", 0)
                results["samples_cleansed"] = judge_results.get("output_count", 0)
                results["cleansing_stats"][input_file.stem] = judge_results
            else:
                results["samples_evaluated"] = len(samples)

        self.state.phase_progress["quality_control"] = 1.0
        return results

    def _run_freeze_optimization(self) -> Dict[str, Any]:
        """冻结最適化フェーズ"""
        self.state.current_phase = "freeze_optimization"

        if not self.ebbinghaus_freeze:
            return {"skipped": True, "reason": "freeze system not initialized"}

        results = {"memories_added": 0, "layers_analyzed": 0, "statistics": {}}

        if self.world_events:
            for event_id, event in (self.world_events.events or {}).items():
                self.ebbinghaus_freeze.add_memory(
                    content=event.description,
                    domain=event.category,
                    importance_score=event.impact_score,
                )
                results["memories_added"] += 1

        results["statistics"] = self.ebbinghaus_freeze.get_statistics()
        self.ebbinghaus_freeze.save_state("data/ebbinghaus_freeze_state.json")

        self.state.phase_progress["freeze_optimization"] = 1.0
        return results

    def _run_checkpoint_save(self) -> Dict[str, Any]:
        """チェックポイント保存フェーズ"""
        self.state.current_phase = "checkpoint_save"

        self._save_checkpoint()

        return {
            "checkpoint_saved": True,
            "checkpoint_path": str(self.checkpoint_dir),
            "total_checkpoints": self.state.checkpoint_count,
        }

    def get_status(self) -> Dict[str, Any]:
        """現在のステータスを取得"""
        return {
            "current_phase": self.state.current_phase,
            "phase_progress": self.state.phase_progress,
            "total_samples": self.state.total_samples_processed,
            "checkpoints": self.state.checkpoint_count,
            "is_completed": self.state.is_completed,
            "errors": self.state.error_count,
            "start_time": self.state.start_time,
            "last_checkpoint": self.state.last_checkpoint_time,
        }


def main():
    """メインエントリポイント"""
    import argparse

    parser = argparse.ArgumentParser(description="Evolved Shinka Pipeline")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--config", type=str, help="Config YAML path")

    args = parser.parse_args()

    config = PipelineConfig()

    if args.config and Path(args.config).exists():
        with open(args.config, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)
            for key, value in config_data.items():
                if hasattr(config, key):
                    setattr(config, key, value)

    pipeline = EvolvedShinkaPipeline(config)

    if args.resume:
        results = pipeline.run(resume=True)
    else:
        results = pipeline.run()

    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

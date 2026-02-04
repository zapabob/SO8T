# -*- coding: utf-8 -*-
"""
Integrated Moonshot Pipeline 2025–2026 (cleaned)

Phases:
1) Dataset discovery / HF CLI fetch
2) SFT/RLPO
3) Advanced techniques (mHC, GRPO, SO8T, GRAPE, imatrix)
4) Optional HF upload

Features:
- Rolling checkpoints every 5 minutes (configurable)
- Auto resume on restart
- DB logging (pipeline_runs / dataset_registry / checkpoint_registry)
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import subprocess

from experiments.enhanced_moonshot_pipeline import EnhancedMoonshotPipeline
from database.pipeline_db import PipelineDB, get_file_size_bytes
try:
    from subagents import SubagentManager, Task
except Exception:  # pragma: no cover - optional dependency
    SubagentManager = None
    Task = None

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class IntegratedMoonshotPipeline2025_2026:
    def __init__(self) -> None:
        self.project_root = Path(__file__).resolve().parents[2]
        self.data_dir = self.project_root / "data" / "collected_2025_2026"
        self.results_dir = self.project_root / "results" / "moonshot_2025_2026"
        self.models_dir = self.project_root / "models" / "moonshot_2025_2026"

        # DB logger
        self.db = PipelineDB(self.project_root / "so8t_memory.db")
        self.run_id = self.db.start_run(
            pipeline_name="moonshot_v3.0",
            model_name="zapabobouj-AEGIS-phi3.5-jp-v3.0",
            base_model="AXCXEPT/Borea-Phi-3.5-mini-Instruct-Jp",
            notes="Integrated pipeline (cleaned) with HF CLI + checkpoints",
        )

        # directories
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # checkpoint config
        self.checkpoint_interval = int(os.getenv("SO8T_CHECKPOINT_INTERVAL", "300"))
        self.rolling_count = int(os.getenv("SO8T_ROLLING_CHECKPOINTS", "5"))
        self.checkpoint_index_file = self.data_dir / "checkpoint_index.ptr"
        self.rolling_checkpoints = [
            self.data_dir / f"pipeline_checkpoint_{i+1}.json" for i in range(self.rolling_count)
        ]

        self._stop_checkpoint_thread = threading.Event()
        self._checkpoint_thread: Optional[threading.Thread] = None
        self._current_phase = "initialized"
        self._current_data: Dict[str, Any] = {}

        # subagent system (optional)
        self.subagent_manager: Optional[SubagentManager] = None
        self._init_subagents()

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def _get_webdataset_root(self) -> Path:
        webdataset_dir = Path("H:/from_D/webdataset")
        if not webdataset_dir.exists():
            webdataset_dir = Path("D:/webdataset")
        return webdataset_dir

    def ensure_grape_repo(self) -> Path:
        """Fetch GRAPE repository if missing."""
        grape_dir = self.project_root / "external" / "GRAPE"
        if grape_dir.exists():
            return grape_dir
        if os.getenv("SO8T_DRYRUN") == "1":
            logger.info("Dry-run mode: skipping GRAPE repo fetch")
            return grape_dir
        script = self.project_root / "scripts" / "setup" / "fetch_grape_repo.py"
        cmd = ["py", "-3", str(script), "--dest", str(grape_dir)]
        try:
            subprocess.run(cmd, check=True, cwd=self.project_root)
        except subprocess.CalledProcessError as exc:
            logger.warning("GRAPE repo fetch failed: %s", exc)
        return grape_dir

    def _init_subagents(self) -> None:
        if SubagentManager is None:
            return
        definitions_dir = self.project_root / "subagents" / "definitions"
        config_path = self.project_root / "config" / "subagents.yaml"
        manager = SubagentManager(definitions_dir=definitions_dir, config_path=config_path)
        manager.load()
        self.subagent_manager = manager

    def _serialize_routing(self, decision) -> Optional[Dict[str, Any]]:
        if decision is None:
            return None
        return {
            "strategy": decision.strategy,
            "reasoning": decision.reasoning,
            "assignments": [
                {
                    "subagent_name": assignment.subagent_name,
                    "task_portion": assignment.task_portion,
                    "capabilities": assignment.capabilities,
                    "configuration": assignment.configuration,
                }
                for assignment in decision.assignments
            ],
        }

    def _route_phase(self, phase: str, description: str, tags: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
        if self.subagent_manager is None or Task is None:
            return None
        task = Task(
            description=description,
            routing_strategy=os.getenv("SO8T_SUBAGENT_STRATEGY", "single_best"),
            tags=tags or [],
        )
        decision = self.subagent_manager.route(task)
        logger.info("[Subagents:%s] %s", phase, decision.reasoning)
        for assignment in decision.assignments:
            logger.info("  - %s (%s)", assignment.subagent_name, ", ".join(assignment.capabilities))
        return self._serialize_routing(decision)

    def _save_checkpoint(self, phase: str, data: Optional[Dict[str, Any]] = None) -> Path:
        if data is None:
            data = {}
        self._current_phase = phase
        self._current_data = data

        # rolling index
        idx = 0
        if self.checkpoint_index_file.exists():
            try:
                idx = int(self.checkpoint_index_file.read_text(encoding="utf-8").strip())
            except Exception:
                idx = 0
        idx = (idx + 1) % self.rolling_count
        self.checkpoint_index_file.write_text(str(idx), encoding="utf-8")

        ckpt_path = self.rolling_checkpoints[idx]
        payload = {
            "phase": phase,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "run_id": self.run_id,
            "data": data,
        }
        ckpt_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        self.db.log_checkpoint(self.run_id, phase=phase, checkpoint_path=str(ckpt_path), notes="rolling")
        logger.info("Checkpoint saved: %s", ckpt_path)
        return ckpt_path

    def _load_latest_checkpoint(self) -> Optional[Dict[str, Any]]:
        # Prefer pointer file
        if self.checkpoint_index_file.exists():
            try:
                idx = int(self.checkpoint_index_file.read_text(encoding="utf-8").strip())
                ckpt = self.rolling_checkpoints[idx]
                if ckpt.exists():
                    return json.loads(ckpt.read_text(encoding="utf-8"))
            except Exception:
                pass

        # fallback: newest file
        existing = [p for p in self.rolling_checkpoints if p.exists()]
        if not existing:
            return None
        latest = max(existing, key=lambda p: p.stat().st_mtime)
        try:
            return json.loads(latest.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _checkpoint_worker(self) -> None:
        while not self._stop_checkpoint_thread.wait(self.checkpoint_interval):
            try:
                self._save_checkpoint(self._current_phase, self._current_data)
            except Exception as exc:
                logger.warning("Checkpoint worker error: %s", exc)

    def _start_periodic_checkpoint(self) -> None:
        if self._checkpoint_thread and self._checkpoint_thread.is_alive():
            return
        self._checkpoint_thread = threading.Thread(target=self._checkpoint_worker, daemon=True)
        self._checkpoint_thread.start()

    def _stop_periodic_checkpoint(self) -> None:
        self._stop_checkpoint_thread.set()
        if self._checkpoint_thread:
            self._checkpoint_thread.join(timeout=5)

    # ------------------------------------------------------------------
    # Dataset discovery / HF CLI
    # ------------------------------------------------------------------
    def discover_existing_datasets(self) -> Dict[str, List[Path]]:
        datasets: Dict[str, List[Path]] = {
            "arxiv": [],
            "biorxiv": [],
            "nsfw_detection": [],
            "drug_detection": [],
            "integrated": [],
            "so8t": [],
        }

        data_dir = self.project_root / "data"
        if data_dir.exists():
            arxiv_dir = data_dir / "arxiv_biorxiv"
            if arxiv_dir.exists():
                datasets["arxiv"].extend(list(arxiv_dir.glob("*.jsonl")))

            integrated_dir = data_dir / "integrated"
            if integrated_dir.exists():
                datasets["integrated"].extend(list(integrated_dir.glob("*.jsonl")))

            nsfw_dir = data_dir / "nsfw_detection"
            if nsfw_dir.exists():
                datasets["nsfw_detection"].extend(list(nsfw_dir.glob("*.jsonl")))

            so8t_dirs = list(data_dir.glob("so8t_*"))
            for so8t_dir in so8t_dirs:
                datasets["so8t"].extend(list(so8t_dir.glob("*.jsonl")))

        webdataset_dir = self._get_webdataset_root()
        if webdataset_dir.exists():
            nsfw_web = webdataset_dir / "nsfw_detection_dataset"
            if nsfw_web.exists():
                datasets["nsfw_detection"].extend(list(nsfw_web.glob("*.jsonl")))

            drug_web = webdataset_dir / "drug_pharmaceutical_detection_dataset"
            if drug_web.exists():
                datasets["drug_detection"].extend(list(drug_web.glob("*.jsonl")))

            processed_dir = webdataset_dir / "processed"
            if processed_dir.exists():
                datasets["integrated"].extend(list(processed_dir.glob("**/*.jsonl")))

        return datasets

    def collect_hf_cli_datasets(self) -> Path:
        logger.info("Phase HF: HF CLI download")
        base_dir = self._get_webdataset_root() / "hf_selected"
        base_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = self.data_dir / "hf_cli_manifest.json"
        if os.getenv("SO8T_DRYRUN") == "1":
            logger.info("Dry-run mode: skipping HF CLI dataset fetch")
            return manifest_path

        cmd = [
            "py",
            "-3",
            str(self.project_root / "scripts" / "data_processing" / "hf_cli_dataset_fetch.py"),
            "--base-dir",
            str(base_dir),
            "--manifest",
            str(manifest_path),
        ]
        logger.info("HF CLI command: %s", " ".join(cmd))
        try:
            subprocess.run(cmd, check=True, cwd=self.project_root)
        except subprocess.CalledProcessError as exc:
            logger.error("HF CLI failed: %s", exc)
            return manifest_path

        # DB logging
        try:
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                for item in manifest.get("datasets", []):
                    self.db.log_dataset(
                        run_id=self.run_id,
                        dataset_id=item.get("dataset_id", ""),
                        source_type="huggingface",
                        category=item.get("category"),
                        local_path=item.get("local_dir"),
                        file_size_bytes=item.get("size_bytes"),
                        acquired_via="hf_cli",
                    )
        except Exception as exc:
            logger.warning("Manifest parse failed: %s", exc)

        return manifest_path

    def verify_nsfw_drug_datasets(self) -> bool:
        datasets = self.discover_existing_datasets()
        found_nsfw = len(datasets["nsfw_detection"]) > 0
        found_drug = len(datasets["drug_detection"]) > 0
        if found_nsfw and found_drug:
            logger.info("NSFW / Drug datasets available.")
            return True
        logger.warning("NSFW or Drug datasets missing. Please provide data.")
        return False

    def build_quadrality_think_dataset(self, input_path: Path, output_path: Path) -> Optional[Path]:
        """Generate <think> quadrality dataset from an existing JSONL file."""
        script = self.project_root / "scripts" / "data_processing" / "build_quadrality_think_dataset.py"
        if not script.exists():
            logger.warning("Quadrality builder script not found: %s", script)
            return None
        if not input_path.exists():
            logger.warning("Quadrality input dataset not found: %s", input_path)
            return None
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = ["py", "-3", str(script), "--input", str(input_path), "--output", str(output_path)]
        try:
            subprocess.run(cmd, check=True, cwd=self.project_root)
            logger.info("Quadrality <think> dataset created: %s", output_path)
            return output_path
        except subprocess.CalledProcessError as exc:
            logger.warning("Quadrality dataset build failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Phases
    # ------------------------------------------------------------------
    def execute_sft(self, dataset_paths: List[Path]) -> Path:
        logger.info("Phase SFT/RLPO")
        pipeline = EnhancedMoonshotPipeline(boreas_model_path="AXCXEPT/Borea-Phi-3.5-mini-Instruct-Jp")
        pipeline.load_boreas_model()
        pipeline.execute_sft_rlpo_integration(target_datasets=dataset_paths)
        pipeline._cleanup_resources()
        out_dir = Path("models/aegis_v25_rlpo")
        return out_dir

    def execute_advanced_techniques_integration(self, sft_model_path: Path) -> Path:
        logger.info("Phase Advanced Techniques")
        pipeline = EnhancedMoonshotPipeline(boreas_model_path=str(sft_model_path))
        pipeline.load_boreas_model()
        self.ensure_grape_repo()
        pipeline.execute_so8_residual_adapter_retraining()
        pipeline.execute_deepseek_grpo_integration()
        pipeline.execute_mhc_manifold_integration()
        grape_variant = os.getenv("SO8T_GRAPE_VARIANT", "multiplicative")
        pipeline.execute_grape_position_encoding(variant=grape_variant)
        pipeline.execute_geometric_scaling_integration()
        pipeline.execute_so8t_imatrix_quantization()
        pipeline.execute_bf16_gguf_conversion()
        pipeline._cleanup_resources()
        return self.models_dir / "final_model_with_advanced_techniques"

    def execute_hf_upload_automation(self) -> None:
        pipeline = EnhancedMoonshotPipeline(boreas_model_path="AXCXEPT/Borea-Phi-3.5-mini-Instruct-Jp")
        pipeline.execute_hf_upload_automation()

    # ------------------------------------------------------------------
    # Pipeline runner
    # ------------------------------------------------------------------
    def execute_full_pipeline(self, use_existing_datasets: bool = True) -> None:
        self._start_periodic_checkpoint()
        checkpoint = self._load_latest_checkpoint()
        resume_phase = checkpoint.get("phase") if checkpoint else None

        phases = ["collect", "sft", "advanced", "upload"]
        start_idx = phases.index(resume_phase) if resume_phase in phases else 0

        datasets = self.discover_existing_datasets()
        dataset_paths: List[Path] = []
        for items in datasets.values():
            dataset_paths.extend(items)

        if start_idx <= 0:
            routing = self._route_phase(
                "collect",
                "Collect and validate datasets (HF CLI, arxiv/biorxiv, exams, safety)",
                tags=["dataset", "ingestion", "quality"],
            )
            self._save_checkpoint(
                "collect",
                {
                    "datasets": [str(p) for p in dataset_paths],
                    "subagent_routing": routing,
                },
            )
            if not use_existing_datasets:
                self.collect_hf_cli_datasets()

            # Build quadrality <think> dataset from first integrated file if available
            if datasets["integrated"]:
                quad_output = self.project_root / "data" / "integrated" / "quadrality_think.jsonl"
                created = self.build_quadrality_think_dataset(datasets["integrated"][0], quad_output)
                if created:
                    dataset_paths.append(created)
                    self.db.log_dataset(
                        run_id=self.run_id,
                        dataset_id="quadrality_think",
                        source_type="synthetic",
                        category="so8t_quadrality",
                        local_path=str(created),
                        file_size_bytes=get_file_size_bytes(str(created)),
                        acquired_via="build_quadrality_think_dataset",
                    )

        if start_idx <= 1:
            self._route_phase(
                "sft",
                "Run SFT/RLPO training for AEGIS model",
                tags=["sft", "training", "rlpo"],
            )
            sft_path = self.execute_sft(dataset_paths)
            self._save_checkpoint("sft", {"sft_model_path": str(sft_path)})
        else:
            sft_path = Path(checkpoint.get("data", {}).get("sft_model_path", "models/aegis_v25_rlpo"))

        if start_idx <= 2:
            routing = self._route_phase(
                "advanced",
                "Integrate GRPO, mHC, SO8T, GRAPE, imatrix, and GGUF conversion",
                tags=["grpo", "mhc", "so8t", "grape", "imatrix"],
            )
            final_path = self.execute_advanced_techniques_integration(sft_path)
            self._save_checkpoint(
                "advanced",
                {
                    "final_model_path": str(final_path),
                    "subagent_routing": routing,
                },
            )

        if start_idx <= 3:
            routing = self._route_phase(
                "upload",
                "Publish model artifacts to Hugging Face and Ollama",
                tags=["deploy", "huggingface", "ollama"],
            )
            self.execute_hf_upload_automation()
            self._save_checkpoint("upload", {"subagent_routing": routing})

        self._stop_periodic_checkpoint()
        self.db.end_run(self.run_id, status="completed")
        logger.info("Pipeline completed.")

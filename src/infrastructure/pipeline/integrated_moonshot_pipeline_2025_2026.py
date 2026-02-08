# -*- coding: utf-8 -*-
"""
Integrated Moonshot Pipeline 2025–2026 (AEGIS-v3.0 Edition)

Phases:
1) Dataset discovery / HF CLI fetch
2) SFT/RLPO (Unsloth)
3) Advanced techniques (mHC, GRPO, SO8T, GRAPE, imatrix, GGUF)
4) Autonomous Research (Sakana AI Integrated Agent)
5) Advanced HF CLI upload (Plots, GGUF, Stats)

Features:
- Rolling checkpoints (3 generations) every 5 minutes
- Power-on Auto-Resume
- Real-time progress monitoring
- DB logging (SQLite)
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import subprocess
import yaml
from tqdm import tqdm

# Add project root and source subdirectories to sys.path
# Since this file is in src/infrastructure/pipeline/, parents[3] is the root.
project_root = Path(__file__).resolve().parents[3]
src_root = project_root / "src"
for p in [project_root, src_root]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from core.experiments.enhanced_moonshot_pipeline import EnhancedMoonshotPipeline
from infrastructure.database.pipeline_db import PipelineDB, get_file_size_bytes

try:
    from agents.manager import SubagentManager, Task
except Exception:
    SubagentManager = None
    Task = None

from agents.sakana_ai_integrated_agent import SakanaAIIntegratedAgent
from data.research.evolutionary_optimizer import EvolutionaryOptimizer
from infrastructure.documentation.generate_model_card import ModelCardGenerator

# Statistical Benchmark integration
try:
    from evaluation.phase6_statistical_benchmark import IndustryStandardBenchmark
    BENCHMARK_AVAILABLE = True
except ImportError:
    BENCHMARK_AVAILABLE = False

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")



class IntegratedMoonshotPipeline2025_2026:
    def __init__(self) -> None:
        self.project_root = Path(__file__).resolve().parents[3]
        self.data_dir = self.project_root / "data" / "collected_2025_2026"
        self.results_dir = self.project_root / "results" / "moonshot_2025_2026"
        self.models_dir = self.project_root / "models" / "moonshot_2025_2026"

        # DB logger
        self.pipeline_name = os.getenv("SO8T_PIPELINE_NAME", "Moonshot")
        self.db = PipelineDB(self.project_root / "so8t_memory.db")
        self.run_id = self.db.start_run(
            pipeline_name=f"{self.pipeline_name.lower()}_v3.0",
            model_name=f"zapabobouj-AEGIS-{self.pipeline_name.lower()}-v3.0",
            base_model=os.getenv("SO8T_BASE_MODEL", "AXCXEPT/Borea-Phi-3.5-mini-Instruct-Jp"),
            notes="Integrated pipeline (cleaned) with HF CLI + checkpoints",
        )

        # directories
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # checkpoint config
        self.checkpoint_interval = 300  # 5 minutes
        self.rolling_count = 3          # 3 generations as requested
        self.checkpoint_index_file = self.data_dir / f"{self.pipeline_name.lower()}_checkpoint_index.ptr"
        self.rolling_checkpoints = [
            self.data_dir / f"{self.pipeline_name.lower()}_checkpoint_{i+1}.json" for i in range(self.rolling_count)
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
        try:
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
        except Exception as e:
            logger.warning(f"[Subagents] Routing failed for {phase}: {e}")
            return None

    def _generate_subagent_schedule(self) -> Optional[Path]:
        if self.subagent_manager is None or Task is None:
            return None
        tasks_file = self.project_root / "config" / "subagent_tasks.yaml"
        if not tasks_file.exists():
            return None

        try:
            tasks_payload = yaml.safe_load(tasks_file.read_text(encoding="utf-8")) or {}
        except Exception as exc:
            logger.warning("Failed to read subagent tasks: %s", exc)
            return None

        schedule = []
        for task_entry in tasks_payload.get("tasks", []):
            description = task_entry.get("description", "")
            if not description:
                continue
            task = Task(
                description=description,
                routing_strategy=task_entry.get("routing_strategy", "single_best"),
                required_capabilities=task_entry.get("required_capabilities", []) or [],
                tags=task_entry.get("tags", []) or [],
            )
            decision = self.subagent_manager.route(task)
            schedule.append(
                {
                    "id": task_entry.get("id"),
                    "description": description,
                    "routing": self._serialize_routing(decision),
                }
            )

        output_path = self.results_dir / "subagent_schedule.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(schedule, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Subagent schedule written: %s", output_path)
        return output_path

    def _save_checkpoint(self, phase: str, data: Optional[Dict[str, Any]] = None) -> Path:
        if data is None:
            data = {}
        self._current_phase = phase
        self._current_data = data
 
        # 3世代ローリングインデックス (1, 2, 3)
        idx = 1
        if self.checkpoint_index_file.exists():
            try:
                prev_idx = int(self.checkpoint_index_file.read_text(encoding="utf-8").strip())
                idx = (prev_idx % self.rolling_count) + 1
            except Exception:
                idx = 1
        self.checkpoint_index_file.write_text(str(idx), encoding="utf-8")
 
        ckpt_path = self.data_dir / f"pipeline_checkpoint_{idx}.json"
        payload = {
            "phase": phase,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "run_id": self.run_id,
            "data": data,
        }
        ckpt_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        self.db.log_checkpoint(self.run_id, phase=phase, checkpoint_path=str(ckpt_path), notes=f"rolling_{idx}")
        logger.info("[CHECKPOINT] Phase: %s saved to generation %d", phase, idx)
        return ckpt_path

    def _load_latest_checkpoint(self) -> Optional[Dict[str, Any]]:
        # Prefer pointer file
        if self.checkpoint_index_file.exists():
            try:
                idx = int(self.checkpoint_index_file.read_text(encoding="utf-8").strip())
                ckpt = self.rolling_checkpoints[idx - 1]  # idx is 1-based, list is 0-based
                if ckpt.exists():
                    return json.loads(ckpt.read_text(encoding="utf-8"))
            except (IndexError, ValueError) as e:
                logger.warning("[CHECKPOINT] Index lookup failed: %s, falling back to latest file", e)

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
            "integrated": [],
            "nsfw_detection": [],
            "drug_detection": [],
            "so8t": [],
            "tool_calling": [],
            "enrichment": [],
            "reward_strategy": [],
        }

        data_dir = self.project_root / "data"
        if data_dir.exists():
            arxiv_dir = data_dir / "arxiv_biorxiv"
            if arxiv_dir.exists():
                datasets["arxiv"].extend(list(arxiv_dir.glob("**/*.jsonl")))

            tool_dir = data_dir / "tool_calling"
            if tool_dir.exists():
                datasets["tool_calling"].extend(list(tool_dir.glob("*.jsonl")))

            integrated_dir = data_dir / "integrated"
            if integrated_dir.exists():
                datasets["integrated"].extend(list(integrated_dir.glob("*.jsonl")))

            nsfw_dir = data_dir / "nsfw_detection"
            if nsfw_dir.exists():
                datasets["nsfw_detection"].extend(list(nsfw_dir.glob("*.jsonl")))

            so8t_dirs = list(data_dir.glob("so8t_*"))
            for so8t_dir in so8t_dirs:
                datasets["so8t"].extend(list(so8t_dir.glob("*.jsonl")))

            enrichment_dir = data_dir / "multi_domain_enrichment"
            if enrichment_dir.exists():
                datasets["enrichment"].extend(list(enrichment_dir.glob("*.jsonl")))

            reward_dir = data_dir / "reward_strategy"
            if reward_dir.exists():
                datasets["reward_strategy"].extend(list(reward_dir.glob("*.jsonl")))

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
            str(self.project_root / "src" / "data" / "processing" / "hf_cli_dataset_fetch.py"),
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

    def collect_new_datasets(self) -> Dict[str, Path]:
        """新規データセット収集 (設計書準拠)
        
        収集対象:
        - Arxiv/BioRxiv 論文 (2024-2026 高引用)
        - OSINT ソース (ポップカルチャー、世界情勢)
        - MCP/スキルデータセット
        - WebResearch/DeepResearch データ
        """
        logger.info("Phase: New Dataset Collection (AEGIS v3.0 Spec)")
        
        collected = {}
        output_dir = self.data_dir / "new_collected"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if os.getenv("SO8T_DRYRUN") == "1":
            logger.info("Dry-run mode: skipping new dataset collection")
            return collected
        
        # 1. Arxiv/BioRxiv 論文収集 (設計書準拠: 100k papers)
        # Force strict 50k + 50k count
        if os.getenv("SO8T_COLLECT_ARXIV", "1") == "1":
            try:
                arxiv_output = output_dir / "arxiv_biorxiv_vssi.jsonl"
                
                # Strict enforcement of 50k targets per spec
                arxiv_count = os.getenv("SO8T_ARXIV_COUNT", "50000")
                biorxiv_count = os.getenv("SO8T_BIORXIV_COUNT", "50000")
                
                logger.info(f"[ARXIV] Enforcing VSSI Quadruple Reasoning collection: Arxiv={arxiv_count}, BioRxiv={biorxiv_count}")

                cmd = [
                    "py", "-3",
                    str(self.project_root / "src" / "data" / "processing" / "process_arxiv_biorxiv.py"),
                    "--arxiv-count", arxiv_count,
                    "--biorxiv-count", biorxiv_count,
                    "--export-vssi",
                    "--vssi-output", str(arxiv_output),
                ]
                
                subprocess.run(cmd, check=False, cwd=self.project_root, 
                               env={**os.environ, "PYTHONPATH": str(self.project_root)})
                if arxiv_output.exists():
                    collected["arxiv_biorxiv"] = arxiv_output
                    logger.info(f"[ARXIV] Collected VSSI dataset to {arxiv_output}")
            except Exception as e:
                logger.warning(f"Arxiv collection failed: {e}")

        # 1b. Semantic Scholar 論文収集 (Parallel)
        if os.getenv("SO8T_COLLECT_SEMANTIC_SCHOLAR", "1") == "1":
            try:
                ss_output = output_dir / "semanticscholar_vssi.jsonl"
                ss_query = os.getenv("SO8T_SS_QUERY", "deep learning transformer architecture")
                ss_count = os.getenv("SO8T_SS_COUNT", "1000")
                
                logger.info(f"[SEMANTIC_SCHOLAR] Collecting papers for query: {ss_query}, limit: {ss_count}")
                
                cmd = [
                    "py", "-3",
                    str(self.project_root / "src" / "data" / "collection" / "semanticscholar_fetcher.py"),
                    "--query", ss_query,
                    "--max-papers", ss_count,
                    "--output", str(ss_output),
                ]
                
                # 並行実行を想定しているが、シンプルに逐次実行するか Popen を使う
                subprocess.run(cmd, check=False, cwd=self.project_root,
                               env={**os.environ, "PYTHONPATH": str(self.project_root)})
                
                if ss_output.exists():
                    collected["semanticscholar"] = ss_output
                    logger.info(f"[SEMANTIC_SCHOLAR] Collected VSSI dataset to {ss_output}")
            except Exception as e:
                logger.warning(f"Semantic Scholar collection failed: {e}")
        
        # 2. OSINT ソース収集 (Pop-culture & World Affairs via Script, NO OLLAMA)
        if os.getenv("SO8T_COLLECT_OSINT", "1") == "1":
            try:
                osint_base = output_dir / "osint"
                # Ensure no Ollama flags are passed implicitly
                cmd = [
                    "py", "-3",
                    str(self.project_root / "src" / "data" / "processing" / "osint_source_collector.py"),
                    "--domain", "all",
                    "--output-dir", str(osint_base),
                ]
                logger.info("Collecting OSINT sources (Script-based)...")
                subprocess.run(cmd, check=False, cwd=self.project_root,
                               env={**os.environ, "PYTHONPATH": str(self.project_root)})
                
                # 集約されたファイルを代表として登録
                osint_output = osint_base / "world_affairs" / "sources.jsonl"
                if osint_output.exists():
                    collected["osint"] = osint_output
                    logger.info(f"[OSINT] Collected world_affairs to {osint_output}")
            except Exception as e:
                logger.warning(f"OSINT collection failed: {e}")
        
        # 3. 日本大学入試問題 (ローカルデータ統合)
        japan_exam_dir = self.project_root / "data" / "japan_entrance_exams"
        if japan_exam_dir.exists():
            exam_files = list(japan_exam_dir.glob("*.jsonl"))
            if exam_files:
                collected["japan_exams"] = exam_files[0]
                logger.info(f"[JAPAN_EXAMS] Found {len(exam_files)} exam datasets")
        
        # 4. MCP/スキルデータセット統合
        mcp_skill_dir = self.project_root / "data" / "mcp_skills"
        if mcp_skill_dir.exists():
            mcp_files = list(mcp_skill_dir.glob("*.jsonl"))
            if mcp_files:
                collected["mcp_skills"] = mcp_files[0]
                logger.info(f"[MCP_SKILLS] Found {len(mcp_files)} skill datasets")
        
        # 5. WebResearch/DeepResearch データ
        research_dir = self.project_root / "data" / "research_datasets"
        if research_dir.exists():
            research_files = list(research_dir.glob("*.jsonl"))
            if research_files:
                collected["research"] = research_files[0]
                logger.info(f"[RESEARCH] Found {len(research_files)} research datasets")
        
        # DB logging
        for name, path in collected.items():
            try:
                self.db.log_dataset(
                    run_id=self.run_id,
                    dataset_id=f"new_{name}",
                    source_type=name,
                    category="new_collection",
                    local_path=str(path),
                    file_size_bytes=get_file_size_bytes(path) if path.exists() else 0,
                    acquired_via="collect_new_datasets",
                )
            except Exception:
                pass
        
        logger.info(f"[NEW_DATASETS] Collected {len(collected)} new dataset types")
        return collected

    def run_multi_domain_enrichment(self) -> List[Path]:
        """Phase 4: Multi-domain data enrichment."""
        if all(os.getenv(flag, "1") == "0" for flag in ["SO8T_ENRICH_ACADEMIC", "SO8T_ENRICH_POP", "SO8T_ENRICH_WORLD", "SO8T_ENRICH_PHARMA"]):
            logger.info("All enrichment toggles disabled; skipping enrichment run.")
            return []
        script = self.project_root / "scripts" / "data_processing" / "multi_domain_enrichment.py"
        if not script.exists():
            logger.warning("Enrichment script not found: %s", script)
            return []
        if os.getenv("SO8T_DRYRUN") == "1":
            logger.info("Dry-run mode: skipping multi-domain enrichment")
            return []

        cmd = [
            "py",
            "-3",
            str(script),
        ]
        if os.getenv("SO8T_QUADRUPLE_TOKENS", "0") == "1":
            cmd.append("--quadruple")
        if os.getenv("SO8T_OSINT_AUTO_SOURCES", "0") == "1":
            cmd.append("--auto-sources")
        if os.getenv("SO8T_THINK_TAG_STYLE"):
            cmd.extend(["--think-tag-style", os.getenv("SO8T_THINK_TAG_STYLE")])
        if os.getenv("SO8T_MAX_PAPERS"):
            cmd.extend(["--max-papers", os.getenv("SO8T_MAX_PAPERS")])

        logger.info("Enrichment command: %s", " ".join(cmd))
        try:
            subprocess.run(cmd, check=True, cwd=self.project_root)
        except subprocess.CalledProcessError as exc:
            logger.error("Enrichment failed: %s", exc)
            return []

        manifest_path = self.project_root / "results" / "multi_domain_enrichment_manifest.json"
        outputs: List[Path] = []
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                for _, path_str in (manifest.get("outputs") or {}).items():
                    if path_str:
                        outputs.append(Path(path_str))
            except Exception as exc:
                logger.warning("Failed to parse enrichment manifest: %s", exc)

        # fallback: scan output directory
        if not outputs:
            out_dir = self.project_root / "data" / "multi_domain_enrichment"
            outputs = list(out_dir.glob("*.jsonl")) if out_dir.exists() else []

        for output in outputs:
            self.db.log_dataset(
                run_id=self.run_id,
                dataset_id=output.stem,
                source_type="enrichment",
                category="multi_domain",
                local_path=str(output),
                file_size_bytes=get_file_size_bytes(str(output)),
                acquired_via="multi_domain_enrichment",
            )

        return outputs

    def apply_reward_strategy(self) -> Optional[Path]:
        """Phase 5: Apply Quadrality reward strategy to datasets."""
        if os.getenv("SO8T_REWARD_STRATEGY", "1") != "1":
            logger.info("Reward strategy disabled via SO8T_REWARD_STRATEGY")
            return None
        script = self.project_root / "scripts" / "rl" / "quadrality_reward_strategy.py"
        if not script.exists():
            logger.warning("Reward strategy script not found: %s", script)
            return None

        input_paths = []
        env_inputs = os.getenv("SO8T_REWARD_INPUTS")
        if env_inputs:
            input_paths = [Path(p.strip()) for p in env_inputs.split(",") if p.strip()]
        else:
            default_path = self.project_root / "data" / "multi_domain_enrichment" / "pharma_safety_vssi.jsonl"
            if default_path.exists():
                input_paths = [default_path]

        if not input_paths:
            logger.warning("No reward input datasets found")
            return None

        output_dir = self.project_root / "data" / "reward_strategy"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "quadrality_reward.jsonl"

        cmd = [
            "py",
            "-3",
            str(script),
            "--output",
            str(output_path),
            "--input",
        ] + [str(p) for p in input_paths]

        logger.info("Reward strategy command: %s", " ".join(cmd))
        try:
            subprocess.run(cmd, check=True, cwd=self.project_root)
        except subprocess.CalledProcessError as exc:
            logger.error("Reward strategy failed: %s", exc)
            return None

        if output_path.exists():
            self.db.log_dataset(
                run_id=self.run_id,
                dataset_id="quadrality_reward",
                source_type="reward_strategy",
                category="grpo",
                local_path=str(output_path),
                file_size_bytes=get_file_size_bytes(str(output_path)),
                acquired_via="quadrality_reward_strategy",
            )
            return output_path
        return None

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
        if os.environ.get("SO8T_QUADRUPLE_TOKENS", "").strip().lower() in {"1", "true", "yes"}:
            cmd.append("--quadruple")
        tag_style = os.environ.get("SO8T_THINK_TAG_STYLE")
        if tag_style:
            cmd += ["--think-tag-style", tag_style]
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
        """Phase SFT: Unsloth直接呼び出しによる実GPU学習
        
        Features:
        - 5分ローリングチェックポイント (3世代)
        - 電源投入時自動再開
        - リアルタイム進捗監視
        """
        logger.info("=" * 60)
        logger.info("Phase SFT: Starting Unsloth GPU Training")
        logger.info("=" * 60)
        
        output_dir = self.project_root / "models" / f"aegis_v3_{self.pipeline_name.lower()}_sft"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # チェックポイントからの再開確認
        resume_checkpoint = None
        checkpoint_dir = self.project_root / "checkpoints" / "aegis_v3_sft"
        if checkpoint_dir.exists():
            checkpoints = sorted(checkpoint_dir.glob("checkpoint-*"), key=lambda p: p.stat().st_mtime, reverse=True)
            if checkpoints:
                resume_checkpoint = checkpoints[0]
                logger.info(f"[RESUME] Found checkpoint: {resume_checkpoint}")
        
        # Unsloth トレーナースクリプトを呼び出し
        trainer_script = self.project_root / "src" / "training" / "train_unsloth_so8t.py"
        config_path = self.project_root / "src" / "infrastructure" / "config" / "borea_training.json"
        
        # データセットパスを結合
        dataset_args = []
        for dp in dataset_paths:
            if dp and dp.exists():
                dataset_args.extend(["--dataset", str(dp)])
        
        cmd = [
            "py", "-3", str(trainer_script),
            "--config", str(config_path),
            "--output-dir", str(output_dir),
            "--checkpoint-interval", str(int(os.getenv("SO8T_CHECKPOINT_INTERVAL", "300"))),
            "--rolling-checkpoints", str(int(os.getenv("SO8T_CHECKPOINT_ROLLING", "3"))),
        ]
        if resume_checkpoint:
            cmd.extend(["--resume-from", str(resume_checkpoint)])
        cmd.extend(dataset_args)
        
        logger.info(f"[SFT] Command: {' '.join(cmd[:8])}...")
        logger.info(f"[SFT] Output dir: {output_dir}")
        logger.info(f"[SFT] Checkpoint interval: 5 minutes, rolling: 3 generations")
        
        # 進捗ログファイル
        progress_log = self.project_root / "logs" / "sft_progress.log"
        progress_log.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(progress_log, "w", encoding="utf-8") as log_file:
                log_file.write(f"[{datetime.now().isoformat()}] SFT Training Started\n")
                log_file.write(f"Output: {output_dir}\n")
                log_file.write(f"Datasets: {len(dataset_args)//2}\n")
                log_file.flush()
                
                process = subprocess.Popen(
                    cmd,
                    cwd=self.project_root,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                
                # リアルタイムログ出力
                for line in process.stdout:
                    logger.info(f"[SFT] {line.rstrip()}")
                    log_file.write(line)
                    log_file.flush()
                    # tqdm進捗の抽出（PowerShell可視化用）
                    if "loss" in line.lower() or "step" in line.lower() or "%" in line:
                        print(line.rstrip())  # PowerShell用に標準出力
                
                process.wait()
                
                if process.returncode != 0:
                    logger.error(f"[SFT] Training failed with code {process.returncode}")
                    log_file.write(f"[ERROR] Exit code: {process.returncode}\n")
                else:
                    logger.info("[SFT] Training completed successfully")
                    log_file.write(f"[{datetime.now().isoformat()}] Training Completed\n")
                    # 完了フラグ (Pipeline specific)
                    (self.project_root / "models" / f"sft_{self.pipeline_name.lower()}_v3.done").touch()
                    
        except Exception as e:
            logger.error(f"[SFT] Exception: {e}")
            with open(progress_log, "a", encoding="utf-8") as f:
                f.write(f"[CRITICAL ERROR] {e}\n")
            raise
        
        return output_dir

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
        """Enhanced HF CLI upload including GGUF, Safetensors, and Plots."""
        logger.info("Phase: Advanced HF CLI Upload")
        repo_id = "zapabobouj/AEGIS-phi3.5-jp-v3.0"
        
        # 1. Safetensors (Local model directory)
        model_dir = "models/aegis_v3_borea_final"
        if os.path.exists(model_dir):
            logger.info("Uploading Safetensors weights...")
            subprocess.run(["py", "-m", "huggingface_hub.commands.huggingface_cli", "upload", repo_id, model_dir, ".", "--repo-type", "model"], check=False)
            
        # 2. BF16 GGUF
        gguf_path = "models/aegis_v3_borea_final/zapabobouj-AEGIS-phi3.5-jp-v3.0.gguf"
        if os.path.exists(gguf_path):
            logger.info("Uploading BF16 GGUF model...")
            subprocess.run(["py", "-m", "huggingface_hub.commands.huggingface_cli", "upload", repo_id, gguf_path, "zapabobouj-AEGIS-phi3.5-jp-v3.0.gguf"], check=False)
            
        # 3. Plots and Benchmark Stats
        stats_dir = "src/evaluation/results/phase6_industry"
        if os.path.exists(stats_dir):
            logger.info("Uploading Error Bar graphs and Summary Statistics...")
            subprocess.run(["py", "-m", "huggingface_hub.commands.huggingface_cli", "upload", repo_id, stats_dir, "evaluation_results"], check=False)
            
        # 4. Model Card
        readme_path = self.models_dir / "README.md"
        if readme_path.exists():
            subprocess.run(["py", "-m", "huggingface_hub.commands.huggingface_cli", "upload", repo_id, str(readme_path), "README.md"], check=False)
            
        logger.info("HF CLI Upload process finished.")

    def execute_autonomous_research(self, topic: str) -> Dict[str, Any]:
        """Execute autonomous research phase using SakanaAIIntegratedAgent."""
        logger.info("Phase: Autonomous Research (Sakana AI Style)")
        agent = SakanaAIIntegratedAgent(project_root=self.project_root)
        
        # ハイブリッド分析（OSINT + 科学研究サイクル）を実行
        results = agent.run_hybrid_analysis(topic)
        
        # DB logging
        self.db.log_event(self.run_id, event_type="autonomous_research_sakana", 
                         details={"topic": topic, "status": "completed"})
        
        return results

    def execute_statistical_benchmark(self) -> Dict[str, Any]:
        """Execute statistical benchmark phase using IndustryStandardBenchmark.
        
        Runs lm-eval-harness benchmarks, computes ANOVA/Cohen's d statistics,
        and generates error bar plots.
        """
        logger.info("Phase: Statistical Benchmark (ANOVA/Cohen's d)")
        
        if not BENCHMARK_AVAILABLE:
            logger.warning("IndustryStandardBenchmark not available. Skipping benchmark phase.")
            return {"status": "skipped", "reason": "benchmark module not available"}
        
        if os.getenv("SO8T_SKIP_BENCHMARK") == "1":
            logger.info("Benchmark skipped via SO8T_SKIP_BENCHMARK=1")
            return {"status": "skipped", "reason": "SO8T_SKIP_BENCHMARK=1"}
        
        try:
            output_dir = self.project_root / "src" / "evaluation" / "results" / "phase6_industry"
            benchmark = IndustryStandardBenchmark(
                output_dir=output_dir,
                use_vllm=os.getenv("SO8T_USE_VLLM", "0") == "1",
                batch_size=int(os.getenv("SO8T_BENCHMARK_BATCH_SIZE", "8")),
            )
            
            # Run full benchmark pipeline
            benchmark.run()
            
            # Collect results
            results = {
                "status": "completed",
                "output_dir": str(output_dir),
                "summary_stats": str(output_dir / "summary_statistics.json") if (output_dir / "summary_statistics.json").exists() else None,
                "error_bar_plot": str(output_dir / "error_bar_plot.png") if (output_dir / "error_bar_plot.png").exists() else None,
            }
            
            # DB logging
            self.db.log_event(self.run_id, event_type="statistical_benchmark",
                             details=results)
            
            logger.info("Statistical benchmark completed: %s", output_dir)
            return results
            
        except Exception as exc:
            logger.error("Statistical benchmark failed: %s", exc)
            return {"status": "failed", "error": str(exc)}

    # ------------------------------------------------------------------
    # Pipeline runner
    # ------------------------------------------------------------------
    def execute_full_pipeline(self, use_existing_datasets: bool = True) -> None:
        self._start_periodic_checkpoint()
        
        # Power-on Auto-Resume logic
        checkpoint = self._load_latest_checkpoint()
        last_completed_phase = checkpoint.get("phase") if checkpoint else None
        
        phases_order = ["collect", "enrich", "reward", "research", "sft", "advanced", "benchmark", "upload"]
        
        # Determine start index (start from the phase AFTER the last completed one)
        start_idx = 0
        if last_completed_phase in phases_order:
            start_idx = phases_order.index(last_completed_phase) + 1
            if start_idx < len(phases_order):
                logger.info(f"[RECOVERY] Phase '{last_completed_phase}' completed. Resuming from '{phases_order[start_idx]}'.")
            else:
                logger.info(f"[RECOVERY] All phases completed (last: {last_completed_phase}).")
        else:
            logger.info("[START] Starting new pipeline run")

        datasets = self.discover_existing_datasets()
        dataset_paths: List[Path] = []
        for items in datasets.values():
            dataset_paths.extend(items)

        # ----------------------------------------------------------------
        # Phase 1: Collect
        # ----------------------------------------------------------------
        current_idx = 0
        if start_idx <= current_idx:
            phase = "collect"
            self._current_phase = phase
            logger.info(f"\n[PHASE START] {phase}: Collecting & Validating Datasets")
            
            routing = self._route_phase("collect", "Collect datasets", tags=["dataset"])
            if not use_existing_datasets and os.getenv("SO8T_SKIP_HF_FETCH", "0") != "1":
                self.collect_hf_cli_datasets()
            elif os.getenv("SO8T_SKIP_HF_FETCH", "0") == "1":
                logger.info("[SKIP] HF CLI dataset fetch skipped via SO8T_SKIP_HF_FETCH")
            
            # 新規データセット収集 (設計書準拠)
            try:
                new_datasets = self.collect_new_datasets()
                for name, path in new_datasets.items():
                    if path not in dataset_paths:
                        dataset_paths.append(path)
            except Exception as e:
                logger.error(f"Phase collect failed: {e}")
                # Continue if possible or raise
            
            if datasets["integrated"]:
                try:
                    quad_output = self.project_root / "data" / "integrated" / "quadrality_think.jsonl"
                    created = self.build_quadrality_think_dataset(datasets["integrated"][0], quad_output)
                    if created: dataset_paths.append(created)
                except Exception as e:
                    logger.warning(f"Quadrality dataset build failed: {e}")

            self._save_checkpoint("collect", {"datasets": [str(p) for p in dataset_paths], "new_datasets": [], "subagent_routing": routing})
        
        # ----------------------------------------------------------------
        # Phase 2: Enrich
        # ----------------------------------------------------------------
        current_idx = 1
        if start_idx <= current_idx:
            phase = "enrich"
            self._current_phase = phase
            logger.info(f"\n[PHASE START] {phase}: Multi-domain Data Enrichment")
            
            routing = self._route_phase("enrich", "Multi-domain enrichment", tags=["enrichment"])
            try:
                enriched_paths = self.run_multi_domain_enrichment()
                for p in enriched_paths:
                    if p not in dataset_paths: dataset_paths.append(p)
            except Exception as e:
                logger.error(f"Enrichment failed: {e}")
                enriched_paths = []

            self._save_checkpoint("enrich", {"enriched_paths": [str(p) for p in enriched_paths], "subagent_routing": routing})

        # ----------------------------------------------------------------
        # Phase 3: Reward
        # ----------------------------------------------------------------
        current_idx = 2
        if start_idx <= current_idx:
            phase = "reward"
            self._current_phase = phase
            logger.info(f"\n[PHASE START] {phase}: Applying Quadrality Reward Strategy")
            
            routing = self._route_phase("reward", "Apply reward strategy", tags=["reward"])
            try:
                reward_path = self.apply_reward_strategy()
                if reward_path and reward_path not in dataset_paths: dataset_paths.append(reward_path)
            except Exception as e:
                logger.error(f"Reward strategy failed: {e}")
                reward_path = None

            self._save_checkpoint("reward", {"reward_path": str(reward_path) if reward_path else None, "subagent_routing": routing})

        # ----------------------------------------------------------------
        # Phase 4: Research
        # ----------------------------------------------------------------
        current_idx = 3
        if start_idx <= current_idx:
            phase = "research"
            self._current_phase = phase
            logger.info(f"\n[PHASE START] {phase}: Sakana AI Autonomous Research")
            
            routing = self._route_phase("research", "Autonomous research", tags=["research"])
            research_topic = os.getenv("SO8T_RESEARCH_TOPIC", "Advanced Mathematical Reasoning for LLMs")
            try:
                research_results = self.execute_autonomous_research(research_topic)
            except Exception as e:
                logger.error(f"Research failed: {e}")
                research_results = {}

            self._save_checkpoint("research", {"research_results": research_results, "subagent_routing": routing})

        # ----------------------------------------------------------------
        # Phase 5: SFT (GPU Training)
        # ----------------------------------------------------------------
        current_idx = 4
        if start_idx <= current_idx:
            phase = "sft"
            self._current_phase = phase
            logger.info(f"\n[PHASE START] {phase}: Unsloth SFT/RLPO Training")
            
            self._route_phase("sft", "Run SFT training", tags=["sft"])
            try:
                # Ensure we have datasets
                if not dataset_paths:
                    logger.warning("No datasets found for SFT! Using default discovery.")
                    d = self.discover_existing_datasets()
                    for v in d.values(): dataset_paths.extend(v)
                
                sft_path = self.execute_sft(dataset_paths)
                self._save_checkpoint("sft", {"sft_model_path": str(sft_path)})
            except Exception as e:
                logger.critical(f"SFT Training failed: {e}")
                raise e

        # ----------------------------------------------------------------
        # Phase 6: Advanced (Integration)
        # ----------------------------------------------------------------
        current_idx = 5
        if start_idx <= current_idx:
            phase = "advanced"
            self._current_phase = phase
            logger.info(f"\n[PHASE START] {phase}: mHC/GRPO/GGUF/imatrix Integration")
            
            routing = self._route_phase("advanced", "Advanced integration", tags=["advanced"])
            
            # SFT path recovery from checkpoint if needed
            current_ckpt = self._load_latest_checkpoint()
            sft_path_str = None
            if current_ckpt:
                sft_path_str = current_ckpt.get("data", {}).get("sft_model_path")
            
            if not sft_path_str:
                # Fallback to expected path
                sft_path_str = str(self.project_root / "models" / "aegis_v3_borea_sft")
            
            try:
                final_path = self.execute_advanced_techniques_integration(Path(sft_path_str))
                self._save_checkpoint("advanced", {"final_model_path": str(final_path), "subagent_routing": routing})
            except Exception as e:
                logger.error(f"Advanced integration failed: {e}")

        # ----------------------------------------------------------------
        # Phase 7: Benchmark
        # ----------------------------------------------------------------
        current_idx = 6
        if start_idx <= current_idx:
            phase = "benchmark"
            self._current_phase = phase
            logger.info(f"\n[PHASE START] {phase}: Statistical Benchmark (ANOVA/Cohen's d)")
            
            routing = self._route_phase("benchmark", "Statistical benchmark", tags=["evaluation"])
            try:
                benchmark_results = self.execute_statistical_benchmark()
                self._save_checkpoint("benchmark", {"benchmark_results": benchmark_results, "subagent_routing": routing})
            except Exception as e:
                logger.error(f"Benchmark failed: {e}")

        # ----------------------------------------------------------------
        # Phase 8: Upload
        # ----------------------------------------------------------------
        current_idx = 7
        if start_idx <= current_idx:
            phase = "upload"
            self._current_phase = phase
            logger.info(f"\n[PHASE START] {phase}: HF CLI Advanced Upload")
            
            routing = self._route_phase("upload", "Advanced HF Upload", tags=["upload"])
            try:
                # Generate final Model Card
                gen = ModelCardGenerator(self.project_root)
                card = gen.generate("SO8T-AEGIS-phi3.5-v3.0", "3.0.0")
                gen.save(card, self.models_dir / "README.md")
                
                # Unified Advanced Upload
                self.execute_hf_upload_automation()
                self._save_checkpoint("upload", {"subagent_routing": routing})
            except Exception as e:
                logger.error(f"Upload failed: {e}")

        self._stop_periodic_checkpoint()
        self.db.end_run(self.run_id, status="completed")
        logger.info("Pipeline completed successfully.")

if __name__ == "__main__":
    try:
        pipeline = IntegratedMoonshotPipeline2025_2026()
        pipeline.execute_full_pipeline()
    except Exception as e:
        logger.critical(f"Pipeline execution failed: {e}")
        sys.exit(1)

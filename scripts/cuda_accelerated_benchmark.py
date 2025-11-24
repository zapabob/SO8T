#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CUDA最適化 lm-evaluation-harness オーケストレーター"""

import argparse
import json
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import psutil
from tqdm import tqdm

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore

try:
    import GPUtil
except ImportError:  # pragma: no cover
    GPUtil = None  # type: ignore

DEFAULT_TASKS = ["gsm8k", "mmlu", "hellaswag"]
LM_EVAL_SCRIPT = Path(__file__).resolve().parent / "evaluation" / "lm_eval_benchmark.py"
RESULTS_ROOT = Path(r"D:/webdataset/benchmark_results/cuda_lm_eval")
AGGREGATE_PATH = RESULTS_ROOT / "cuda_lm_eval_aggregate.json"
PLAY_AUDIO_SCRIPT = Path("scripts/utils/play_audio_notification.ps1")

DEFAULT_MODELS = [
    {
        "alias": "modela_phi35",
        "runner": "hf",
        "model_name": "microsoft/Phi-3.5-mini-instruct",
        "tasks": DEFAULT_TASKS,
        "batch_size": 4,
        "limit": 120,
    },
    {
        "alias": "aegis_gguf_q8",
        "runner": "llama.cpp",
        "model_name": r"D:/webdataset/gguf_models/aegis-borea-phi35/aegis-borea-phi35_Q8_0.gguf",
        "tasks": DEFAULT_TASKS,
        "batch_size": 2,
        "limit": 120,
        "model_args": "n_gpu_layers=40",
    },
]


def load_model_matrix(config_path: Optional[Path]) -> List[Dict[str, str]]:
    if config_path and config_path.exists():
        with config_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return DEFAULT_MODELS


def gather_hardware_info() -> Dict[str, str]:
    info: Dict[str, str] = {
        "cpu_percent": f"{psutil.cpu_percent(interval=None):.1f}",
        "ram_gb": f"{psutil.virtual_memory().total / (1024 ** 3):.1f}",
    }
    if torch and torch.cuda.is_available():
        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)
        info.update(
            {
                "gpu_name": props.name,
                "total_gpu_memory_gb": f"{props.total_memory / (1024 ** 3):.1f}",
                "cuda_version": torch.version.cuda,
            }
        )
    else:
        info["gpu_name"] = "CPU / CUDA unavailable"
    return info


def gpu_snapshot() -> Dict[str, float]:
    snapshot: Dict[str, float] = {}
    if GPUtil:
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]
            snapshot = {
                "utilization": round(gpu.load * 100, 2),
                "memory_used_gb": round(gpu.memoryUsed / 1024, 2),
                "memory_total_gb": round(gpu.memoryTotal / 1024, 2),
                "temperature_c": gpu.temperature,
            }
    return snapshot


def ensure_audio_notification() -> None:
    if not PLAY_AUDIO_SCRIPT.exists():
        return
    try:
        subprocess.run(
            [
                "powershell",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(PLAY_AUDIO_SCRIPT),
            ],
            check=False,
        )
    except Exception:
        pass


class CudaLmEvalOrchestrator:
    def __init__(self, models: List[Dict[str, str]], args: argparse.Namespace) -> None:
        self.models = models
        self.args = args
        self.results_root = Path(args.output_root or RESULTS_ROOT)
        self.results_root.mkdir(parents=True, exist_ok=True)
        self.state_path = self.results_root / "cuda_lm_eval_state.json"
        self.state = {"completed": []}
        if not args.fresh:
            self._load_state()
        self.logger = self._configure_logger()

    def _configure_logger(self) -> logging.Logger:
        logger = logging.getLogger("cuda_lm_eval")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
            logger.addHandler(handler)
        return logger

    def _load_state(self) -> None:
        if self.state_path.exists():
            with self.state_path.open("r", encoding="utf-8") as f:
                self.state = json.load(f)

    def _save_state(self) -> None:
        with self.state_path.open("w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=2)

    def _load_aggregate(self) -> Dict:
        if AGGREGATE_PATH.exists():
            with AGGREGATE_PATH.open("r", encoding="utf-8") as f:
                return json.load(f)
        return {"runs": []}

    def _save_aggregate(self, data: Dict) -> None:
        data["generated_at"] = datetime.utcnow().isoformat() + "Z"
        data["hardware"] = gather_hardware_info()
        with AGGREGATE_PATH.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def _build_command(self, config: Dict[str, str], tasks: List[str], limit: Optional[int], run_dir: Path) -> List[str]:
        cmd = [
            sys.executable,
            str(LM_EVAL_SCRIPT),
            "--model-runner",
            config["runner"],
            "--model-name",
            config["model_name"],
            "--batch-size",
            str(config.get("batch_size", self.args.batch_size)),
            "--device",
            self.args.device,
            "--output-root",
            str(run_dir.parent),
            "--run-dir",
            str(run_dir),
        ]
        cmd.append("--tasks")
        cmd.extend(tasks)
        if limit is not None:
            cmd.extend(["--limit", str(limit)])
        if config.get("model_args"):
            cmd.extend(["--model-args", config["model_args"]])
        return cmd

    def _run_single(self, config: Dict[str, str]) -> Optional[Dict]:
        alias = config["alias"]
        if self.args.models and alias not in self.args.models:
            return None
        tasks = self.args.tasks or config.get("tasks") or DEFAULT_TASKS
        limit = self.args.limit if self.args.limit is not None else config.get("limit")

        if config["runner"] == "llama.cpp":
            model_path = Path(config["model_name"])
            if not model_path.exists():
                self.logger.warning("[SKIP] %s: GGUF not found (%s)", alias, model_path)
                return None

        if alias in self.state.get("completed", []) and not self.args.fresh:
            self.logger.info("[SKIP] %s already completed. Use --fresh to rerun.", alias)
            return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.results_root / alias / f"lm_eval_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)

        cmd = self._build_command(config, tasks, limit, run_dir)
        self.logger.info("[RUN] %s -> %s", alias, " ".join(cmd))

        before_gpu = gpu_snapshot()
        before_cpu = psutil.cpu_percent(interval=None)
        before_ram = psutil.virtual_memory().used / (1024 ** 3)
        start_time = time.time()

        process = subprocess.run(cmd, check=False)
        duration = time.time() - start_time
        after_gpu = gpu_snapshot()
        after_cpu = psutil.cpu_percent(interval=None)
        after_ram = psutil.virtual_memory().used / (1024 ** 3)

        summary_path = run_dir / "summary.json"
        summary = {}
        if summary_path.exists():
            with summary_path.open("r", encoding="utf-8") as f:
                summary = json.load(f)

        metadata_path = run_dir / "metadata.json"
        metadata = {}
        if metadata_path.exists():
            with metadata_path.open("r", encoding="utf-8") as f:
                metadata = json.load(f)

        result = {
            "alias": alias,
            "runner": config["runner"],
            "model_name": config["model_name"],
            "tasks": tasks,
            "limit": limit,
            "run_dir": str(run_dir),
            "exit_code": process.returncode,
            "duration_sec": round(duration, 2),
            "gpu_before": before_gpu,
            "gpu_after": after_gpu,
            "cpu_before_percent": before_cpu,
            "cpu_after_percent": after_cpu,
            "ram_before_gb": round(before_ram, 2),
            "ram_after_gb": round(after_ram, 2),
            "summary": summary,
            "metadata": metadata,
        }

        self.logger.info("[DONE] %s exit=%s duration=%.2fs", alias, process.returncode, duration)

        completed = set(self.state.get("completed", []))
        completed.add(alias)
        self.state["completed"] = sorted(completed)
        self._save_state()

        aggregate = self._load_aggregate()
        aggregate.setdefault("runs", []).append(result)
        self._save_aggregate(aggregate)

        ensure_audio_notification()
        return result

    def run(self) -> List[Dict]:
        results: List[Dict] = []
        for config in tqdm(self.models, desc="CUDA lm-eval benchmarks"):
            run_result = self._run_single(config)
            if run_result:
                results.append(run_result)
        return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CUDA加速 lm-evaluation-harness パイプライン")
    parser.add_argument("--config", type=Path, default=None, help="モデル設定JSON (任意)")
    parser.add_argument("--models", nargs="*", help="実行するモデルalias (未指定は全て)")
    parser.add_argument("--tasks", nargs="*", help="タスク上書き (例: gsm8k mmlu)")
    parser.add_argument("--limit", type=int, default=None, help="各タスクの評価サンプル上限")
    parser.add_argument("--batch-size", type=int, default=4, help="デフォルトbatch size")
    parser.add_argument("--device", default="cuda:0", help="lm_eval_benchmark.pyに渡すdevice")
    parser.add_argument("--output-root", type=Path, default=RESULTS_ROOT, help="結果保存ディレクトリ")
    parser.add_argument("--fresh", action="store_true", help="過去状態を無視して全再実行")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models = load_model_matrix(args.config)
    orchestrator = CudaLmEvalOrchestrator(models, args)
    results = orchestrator.run()

    if results:
        print("\n[CUDA-LM-EVAL] Completed runs:")
        for item in results:
            summary = item.get("summary", {})
            short_metrics = []
            for task, metrics in summary.items():
                acc = metrics.get("acc")
                if acc is not None:
                    short_metrics.append(f"{task}:{acc:.3f}")
            metric_text = ", ".join(short_metrics) if short_metrics else "No metrics"
            print(f"  - {item['alias']} ({item['runner']}): {metric_text}")
        print(f"\n[CUDA-LM-EVAL] Aggregate summary -> {AGGREGATE_PATH}")
    else:
        print("[CUDA-LM-EVAL] No runs executed (check filters / paths).")

    ensure_audio_notification()


if __name__ == "__main__":
    main()

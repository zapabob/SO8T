#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lm-evaluation-harness ラッパースクリプト

- Hugging Face / GGUF(Ollama) モデル双方に対応
- GSM8K / MMLU / HellaSwag など主要タスクを一括実行
- すべての結果を D:/webdataset/benchmark_results/lm_eval/ 以下へ保存
- 実行コマンドと結果をJSONで記録
"""

import argparse
import json
import os
import subprocess
import sys
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore


DEFAULT_OUTPUT_ROOT = Path(r"D:/webdataset/benchmark_results/lm_eval")
PLAY_AUDIO_SCRIPT = Path("scripts/utils/play_audio_notification.ps1")
DEFAULT_TASKS = ["gsm8k", "mmlu", "hellaswag"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run industry-standard benchmarks via lm-evaluation-harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
            Examples:
              # Hugging Faceモデル
              python scripts/evaluation/lm_eval_benchmark.py ^
                  --model-runner hf ^
                  --model-name microsoft/Phi-3.5-mini-instruct ^
                  --tasks gsm8k mmlu hellaswag

              # Ollama用GGUF（llama.cppバックエンド）
              python scripts/evaluation/lm_eval_benchmark.py ^
                  --model-runner llama.cpp ^
                  --model-name D:/webdataset/gguf_models/aegis/aegis_Q8_0.gguf ^
                  --n-gpu-layers 40
            """
        ),
    )

    parser.add_argument(
        "--model-runner",
        choices=["hf", "llama.cpp", "vllm", "ollama"],
        default="hf",
        help="lm-evaluation-harnessの--model値",
    )
    parser.add_argument(
        "--model-name",
        required=True,
        help="Hugging Faceリポジトリ名またはGGUFファイルへのフルパス",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=DEFAULT_TASKS,
        help="評価タスク一覧（スペース区切り）",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="lm-evalの--batch_size",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="各タスクの最大サンプル数。Noneならフルサイズ",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="lm-evalの--device",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="結果保存先ルートディレクトリ",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="結果保存ディレクトリ（指定がなければ自動生成）",
    )
    parser.add_argument(
        "--model-args",
        default=None,
        help="lm-eval --model_args にそのまま渡す値（指定しなければ自動生成）",
    )
    parser.add_argument(
        "--n-gpu-layers",
        type=int,
        default=32,
        help="llama.cpp利用時のn_gpu_layers（デフォルト32）",
    )
    parser.add_argument(
        "--tensor-split",
        default=None,
        help="llama.cpp利用時のtensor_split指定 (例: 4,4)",
    )
    parser.add_argument(
        "--ollama-url",
        default="http://127.0.0.1:11434",
        help="Ollama API URL (lm-evalがollamaバックエンドに対応している場合のみ使用)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Ollama実行時の温度",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="コマンドだけ表示して実行しない",
    )

    return parser.parse_args()


def auto_model_args(args: argparse.Namespace) -> str:
    if args.model_args:
        return args.model_args

    if args.model_runner == "hf":
        device_map = "auto" if args.device.startswith("cuda") else "cpu"
        return ",".join(
            [
                f"pretrained={args.model_name}",
                "dtype=float16",
                "trust_remote_code=True",
                f"device_map={device_map}",
                "use_accelerate=True",
            ]
        )

    if args.model_runner == "llama.cpp":
        parts = [f"model={args.model_name}", f"n_gpu_layers={args.n_gpu_layers}"]
        if args.tensor_split:
            parts.append(f"tensor_split={args.tensor_split}")
        return ",".join(parts)

    if args.model_runner == "vllm":
        return ",".join(
            [
                f"pretrained={args.model_name}",
                "tensor_parallel_size=1",
                "dtype=float16",
            ]
        )

    if args.model_runner == "ollama":
        return ",".join(
            [
                f"model={args.model_name}",
                f"base_url={args.ollama_url}",
                f"temperature={args.temperature}",
            ]
        )

    raise ValueError(f"Unsupported model runner: {args.model_runner}")


def gather_hardware_info() -> Dict[str, str]:
    info: Dict[str, str] = {}
    if torch and torch.cuda.is_available():
        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)
        info["gpu_name"] = props.name
        info["total_memory_gb"] = f"{props.total_memory / (1024**3):.1f}"
        info["cuda_version"] = torch.version.cuda
    else:
        info["gpu_name"] = "CPU / CUDA unavailable"
    return info


def build_command(args: argparse.Namespace, run_dir: Path) -> List[str]:
    cmd = [
        sys.executable,
        "-m",
        "lm_eval",
        "--model",
        args.model_runner,
        "--tasks",
        ",".join(args.tasks),
        "--batch_size",
        str(args.batch_size),
        "--device",
        args.device,
        "--output_path",
        str(run_dir),
        "--log_samples",
    ]

    if args.limit is not None:
        cmd.extend(["--limit", str(args.limit)])

    model_args_value = auto_model_args(args)
    cmd.extend(["--model_args", model_args_value])

    return cmd


def run_command(cmd: List[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
        process.wait()
        return process.returncode


def summarize_results(results_path: Path) -> Dict[str, Dict[str, float]]:
    if not results_path.exists():
        return {}

    with results_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    summary: Dict[str, Dict[str, float]] = {}
    for task, metrics in data.get("results", {}).items():
        numeric_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                numeric_metrics[key] = float(value)
        summary[task] = numeric_metrics
    return summary


def save_run_metadata(
    run_dir: Path,
    cmd: List[str],
    args: argparse.Namespace,
    exit_code: int,
) -> None:
    metadata = {
        "command": cmd,
        "model_runner": args.model_runner,
        "model_name": args.model_name,
        "tasks": args.tasks,
        "batch_size": args.batch_size,
        "limit": args.limit,
        "device": args.device,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "hardware": gather_hardware_info(),
        "exit_code": exit_code,
    }
    with (run_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def play_completion_audio() -> None:
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


def main() -> None:
    args = parse_args()
    args.tasks = [task.strip() for task in args.tasks if task.strip()]

    if args.run_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sanitized_name = (
            args.model_name.replace("/", "_")
            .replace(":", "_")
            .replace("\\", "_")
            .replace(" ", "_")
        )
        run_dir = args.output_root / f"lm_eval_{sanitized_name}_{timestamp}"
    else:
        run_dir = args.run_dir

    run_dir.mkdir(parents=True, exist_ok=True)

    cmd = build_command(args, run_dir)

    print("[LM-EVAL] Command:")
    print(" ".join(cmd))
    if args.dry_run:
        save_run_metadata(run_dir, cmd, args, exit_code=0)
        print("[LM-EVAL] Dry-run completed.")
        return

    log_path = run_dir / "lm_eval_stdout.log"
    exit_code = run_command(cmd, log_path)
    save_run_metadata(run_dir, cmd, args, exit_code)

    summary = summarize_results(run_dir / "results.json")
    if summary:
        summary_path = run_dir / "summary.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print("\n[LM-EVAL] Summary:")
        for task, metrics in summary.items():
            acc = metrics.get("acc", metrics.get("exact_match", None))
            acc_text = f"{acc:.3f}" if acc is not None else "N/A"
            print(f"  - {task}: acc={acc_text} ({metrics})")
        print(f"\n[LM-EVAL] Summary saved to {summary_path}")
    else:
        print("[LM-EVAL] No summary data found. Check lm_eval_stdout.log for details.")

    if exit_code != 0:
        print(f"[LM-EVAL] Benchmark failed with exit code {exit_code}")
    else:
        print(f"[LM-EVAL] Benchmark finished. Results stored in {run_dir}")

    play_completion_audio()


if __name__ == "__main__":
    main()


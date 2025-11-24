#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GGUFモデル統合ベンチマークスイート

MMLU、GSM8K、HellaSwagなどの業界標準ベンチマーク + ELYZA-100を
GGUF化したモデルで実行し、結果を統合レポートとして出力
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

DEFAULT_CONFIG = Path("configs/gguf_benchmark_models.json")
DEFAULT_OUTPUT_ROOT = Path(r"D:/webdataset/benchmark_results/gguf_benchmark")
LM_EVAL_SCRIPT = Path("scripts/evaluation/lm_eval_benchmark.py")
ELYZA_SCRIPT = Path("scripts/evaluation/elyza_benchmark.py")
PLAY_AUDIO_SCRIPT = Path("scripts/utils/play_audio_notification.ps1")


def get_worktree_name() -> str:
    """Get current git worktree name"""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--git-dir"],
            capture_output=True,
            text=True,
            check=True,
        )
        git_dir = Path(result.stdout.strip())
        if "worktrees" in git_dir.parts:
            idx = git_dir.parts.index("worktrees")
            if idx + 1 < len(git_dir.parts):
                return git_dir.parts[idx + 1]
        return "main"
    except Exception:
        return "main"


def load_model_config(config_path: Path) -> List[Dict]:
    """Load GGUF model configuration"""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    
    return config.get("gguf_models", [])


def check_gguf_file(gguf_path: str) -> bool:
    """Check if GGUF file exists"""
    path = Path(gguf_path)
    if not path.exists():
        print(f"[WARNING] GGUF file not found: {gguf_path}")
        return False
    if path.stat().st_size == 0:
        print(f"[WARNING] GGUF file is empty: {gguf_path}")
        return False
    return True


def run_lm_eval_benchmark(
    model_config: Dict,
    output_dir: Path,
    dry_run: bool = False,
) -> Dict[str, Optional[int]]:
    """Run lm-evaluation-harness benchmark for GGUF model"""
    alias = model_config["alias"]
    gguf_path = model_config["gguf_path"]
    
    if not check_gguf_file(gguf_path):
        return {
            "name": f"lm_eval_{alias}",
            "status": "skipped",
            "exit_code": None,
            "reason": "GGUF file not found or empty",
        }
    
    tasks = model_config.get("tasks", ["gsm8k", "mmlu", "hellaswag"])
    batch_size = model_config.get("batch_size", 2)
    n_gpu_layers = model_config.get("n_gpu_layers", 40)
    limit = model_config.get("limit", None)
    
    cmd = [
        sys.executable,
        str(LM_EVAL_SCRIPT),
        "--model-runner",
        "llama.cpp",
        "--model-name",
        gguf_path,
        "--tasks",
    ] + tasks + [
        "--batch-size",
        str(batch_size),
        "--n-gpu-layers",
        str(n_gpu_layers),
        "--output-root",
        str(output_dir),
    ]
    
    if limit:
        cmd.extend(["--limit", str(limit)])
    
    print(f"\n{'='*60}")
    print(f"[LM-EVAL] Running benchmark for {alias}")
    print(f"GGUF: {gguf_path}")
    print(f"Tasks: {', '.join(tasks)}")
    print(f"{'='*60}")
    
    if dry_run:
        print(f"[DRY-RUN] Command: {' '.join(cmd)}")
        return {
            "name": f"lm_eval_{alias}",
            "status": "dry_run",
            "exit_code": 0,
            "command": cmd,
        }
    
    log_path = output_dir / f"lm_eval_{alias}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with log_path.open("w", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="")
                log_file.write(line)
                log_file.flush()
            process.wait()
            exit_code = process.returncode
        
        return {
            "name": f"lm_eval_{alias}",
            "status": "completed" if exit_code == 0 else "failed",
            "exit_code": exit_code,
            "log_path": str(log_path),
            "command": cmd,
        }
    except Exception as e:
        return {
            "name": f"lm_eval_{alias}",
            "status": "error",
            "exit_code": 1,
            "error": str(e),
        }


def run_elyza_benchmark(
    model_config: Dict,
    output_dir: Path,
    elyza_limit: Optional[int] = None,
    dry_run: bool = False,
) -> Dict[str, Optional[int]]:
    """Run ELYZA-100 benchmark via Ollama"""
    alias = model_config["alias"]
    ollama_name = model_config.get("ollama_name")
    
    if not ollama_name:
        return {
            "name": f"elyza_{alias}",
            "status": "skipped",
            "exit_code": None,
            "reason": "ollama_name not specified",
        }
    
    # Check if model is available in Ollama
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if ollama_name.split(":")[0] not in result.stdout:
            print(f"[WARNING] Model {ollama_name} not found in Ollama. Please import it first.")
            return {
                "name": f"elyza_{alias}",
                "status": "skipped",
                "exit_code": None,
                "reason": f"Model {ollama_name} not found in Ollama",
            }
    except Exception as e:
        print(f"[WARNING] Failed to check Ollama models: {e}")
        return {
            "name": f"elyza_{alias}",
            "status": "skipped",
            "exit_code": None,
            "reason": f"Ollama check failed: {e}",
        }
    
    elyza_output_dir = output_dir / "elyza"
    limit = elyza_limit or model_config.get("elyza_limit", None)
    
    cmd = [
        sys.executable,
        "-u",
        str(ELYZA_SCRIPT),
        "--model-name",
        ollama_name,
        "--output-dir",
        str(elyza_output_dir),
    ]
    
    if limit:
        cmd.extend(["--limit", str(limit)])
    
    print(f"\n{'='*60}")
    print(f"[ELYZA-100] Running benchmark for {alias}")
    print(f"Ollama model: {ollama_name}")
    print(f"{'='*60}")
    
    if dry_run:
        print(f"[DRY-RUN] Command: {' '.join(cmd)}")
        return {
            "name": f"elyza_{alias}",
            "status": "dry_run",
            "exit_code": 0,
            "command": cmd,
        }
    
    log_path = output_dir / f"elyza_{alias}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with log_path.open("w", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="")
                log_file.write(line)
                log_file.flush()
            process.wait()
            exit_code = process.returncode
        
        return {
            "name": f"elyza_{alias}",
            "status": "completed" if exit_code == 0 else "failed",
            "exit_code": exit_code,
            "log_path": str(log_path),
            "command": cmd,
        }
    except Exception as e:
        return {
            "name": f"elyza_{alias}",
            "status": "error",
            "exit_code": 1,
            "error": str(e),
        }


def generate_summary_report(
    output_dir: Path,
    model_configs: List[Dict],
    results: List[Dict],
    worktree: str,
) -> None:
    """Generate summary Markdown report"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"gguf_benchmark_report_{timestamp}.md"
    
    lines = [
        "# GGUFモデル統合ベンチマークレポート",
        "",
        f"- **実行日時**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- **Worktree**: {worktree}",
        f"- **出力ディレクトリ**: `{output_dir}`",
        "",
        "## 評価モデル",
        "",
    ]
    
    for config in model_configs:
        lines.append(f"### {config['alias']}")
        lines.append(f"- **説明**: {config.get('description', 'N/A')}")
        lines.append(f"- **GGUFパス**: `{config['gguf_path']}`")
        lines.append(f"- **Ollama名**: `{config.get('ollama_name', 'N/A')}`")
        lines.append("")
    
    lines.extend([
        "## 実行結果",
        "",
    ])
    
    for result in results:
        status_icon = {
            "completed": "[OK]",
            "failed": "[NG]",
            "skipped": "[SKIP]",
            "error": "[ERROR]",
            "dry_run": "[DRY-RUN]",
        }.get(result.get("status", "unknown"), "[?]")
        
        lines.append(f"### {result['name']} {status_icon}")
        if "exit_code" in result and result["exit_code"] is not None:
            lines.append(f"- **Exit code**: {result['exit_code']}")
        if "log_path" in result:
            lines.append(f"- **Log**: `{result['log_path']}`")
        if "reason" in result:
            lines.append(f"- **Reason**: {result['reason']}")
        if "error" in result:
            lines.append(f"- **Error**: {result['error']}")
        lines.append("")
    
    lines.extend([
        "## 結果ファイル",
        "",
        "- lm-eval結果: `lm_eval_*/` ディレクトリ",
        "- ELYZA-100結果: `elyza/` ディレクトリ",
        "- ログファイル: `*.log`",
        "",
    ])
    
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[REPORT] Summary report saved to: {report_path}")


def play_audio() -> None:
    """Play completion audio notification"""
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GGUFモデル統合ベンチマークスイート",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="GGUFモデル設定ファイル",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="結果出力ルートディレクトリ",
    )
    parser.add_argument(
        "--skip-lm-eval",
        action="store_true",
        help="lm-evalベンチマークをスキップ",
    )
    parser.add_argument(
        "--skip-elyza",
        action="store_true",
        help="ELYZA-100ベンチマークをスキップ",
    )
    parser.add_argument(
        "--elyza-limit",
        type=int,
        default=None,
        help="ELYZA-100タスク数制限",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="実行するモデルのエイリアス（指定しなければすべて実行）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="コマンドのみ表示して実行しない",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    # Load model configurations
    model_configs = load_model_config(args.config)
    
    # Filter models if specified
    if args.models:
        model_configs = [
            config for config in model_configs
            if config["alias"] in args.models
        ]
    
    if not model_configs:
        print("[ERROR] No models to evaluate")
        return
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_root / f"gguf_benchmark_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"GGUFモデル統合ベンチマークスイート")
    print(f"{'='*60}")
    print(f"出力ディレクトリ: {output_dir}")
    print(f"評価モデル数: {len(model_configs)}")
    print(f"{'='*60}\n")
    
    results: List[Dict] = []
    
    # Run benchmarks for each model
    for model_config in model_configs:
        alias = model_config["alias"]
        print(f"\n[PROCESSING] {alias}")
        
        # Run lm-eval benchmark
        if not args.skip_lm_eval:
            lm_result = run_lm_eval_benchmark(
                model_config,
                output_dir,
                dry_run=args.dry_run,
            )
            results.append(lm_result)
        
        # Run ELYZA-100 benchmark
        if not args.skip_elyza:
            elyza_result = run_elyza_benchmark(
                model_config,
                output_dir,
                elyza_limit=args.elyza_limit,
                dry_run=args.dry_run,
            )
            results.append(elyza_result)
    
    # Generate summary report
    worktree = get_worktree_name()
    generate_summary_report(output_dir, model_configs, results, worktree)
    
    # Save results metadata
    metadata = {
        "timestamp": timestamp,
        "worktree": worktree,
        "output_dir": str(output_dir),
        "model_configs": model_configs,
        "results": results,
    }
    metadata_path = output_dir / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"\n[COMPLETE] All benchmarks finished. Results saved to: {output_dir}")
    
    if not args.dry_run:
        play_audio()


if __name__ == "__main__":
    main()


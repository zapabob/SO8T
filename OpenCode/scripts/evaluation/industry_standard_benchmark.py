#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Industry-standard benchmark orchestrator

Runs:
1. CUDA-accelerated lm-evaluation-harness sweep
2. DeepEval ethics / logic tests
3. ELYZA-100 Japanese language capability benchmark
4. promptfoo A/B comparison
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

DEFAULT_RUN_ROOT = Path(r"D:/webdataset/benchmark_results/industry_standard")
DOCS_REPORT_ROOT = Path("_docs/benchmark_results/industry_standard")
PLAY_AUDIO_SCRIPT = Path("scripts/utils/play_audio_notification.ps1")


def get_worktree_name() -> str:
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
    except Exception:
        pass
    return "main"


def play_audio() -> None:
    if not PLAY_AUDIO_SCRIPT.exists():
        return
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


def run_and_log(name: str, cmd: List[str], log_path: Path) -> Dict[str, Optional[int]]:
    print(f"[RUN] {name}: {' '.join(cmd)}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Use unbuffered Python for better real-time logging
    if cmd[0] in ["py", "python", "python3"] and "-3" not in cmd[:2]:
        # Insert -u flag for unbuffered output
        cmd = [cmd[0], "-u"] + cmd[1:]
    elif cmd[0] == "py" and cmd[1] == "-3":
        # Replace with unbuffered version
        cmd = ["py", "-u", "-3"] + cmd[2:]
    
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,  # Line buffering
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
            log_file.flush()  # Force flush to disk
        process.wait()
        exit_code = process.returncode
        print(f"[DONE] {name} exit={exit_code}")
        return {"name": name, "command": cmd, "exit_code": exit_code, "log_path": str(log_path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Industry-standard benchmark orchestrator",
    )
    parser.add_argument("--lm-config", type=Path, default=None, help="JSON config for CUDA lm-eval script")
    parser.add_argument("--deepeval-model", default="aegis-borea-phi35-instinct-jp:q8_0")
    parser.add_argument("--deepeval-runner", choices=["ollama", "hf"], default="ollama")
    parser.add_argument(
        "--elyza-models",
        nargs="+",
        default=["model-a:q8_0", "aegis-phi3.5-fixed-0.8:latest"],
        help="Ollama model names for ELYZA-100 benchmark",
    )
    parser.add_argument("--elyza-limit", type=int, default=None, help="Limit number of ELYZA tasks (for testing)")
    parser.add_argument("--promptfoo-config", type=Path, default=Path("configs/promptfoo_config.yaml"))
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--skip-lm", action="store_true")
    parser.add_argument("--skip-deepeval", action="store_true")
    parser.add_argument("--skip-elyza", action="store_true")
    parser.add_argument("--skip-promptfoo", action="store_true")
    parser.add_argument("--label", default=None)
    return parser.parse_args()


def generate_markdown(report_path: Path, metadata: Dict[str, str], steps: List[Dict[str, Optional[int]]]) -> None:
    lines = [
        "# Industry-standard Benchmark Report",
        "",
        f"- **Timestamp**: {metadata['timestamp']}",
        f"- **Worktree**: {metadata['worktree']}",
        f"- **Run Directory**: `{metadata['run_dir']}`",
        "",
        "## Step Results",
        "",
    ]
    for step in steps:
        status = "[OK]" if step["exit_code"] == 0 else "[NG]"
        lines.append(f"### {step['name']} {status}")
        lines.append(f"- Command: `{' '.join(step['command'])}`")
        lines.append(f"- Exit code: {step['exit_code']}")
        lines.append(f"- Log: `{step['log_path']}`")
        lines.append("")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_root = args.run_root
    run_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    label = args.label or timestamp
    run_dir = run_root / f"industry_{label}"
    run_dir.mkdir(parents=True, exist_ok=True)

    steps: List[Dict[str, Optional[int]]] = []

    if not args.skip_lm:
        lm_cmd = [
            "py",
            "-3",
            "scripts/cuda_accelerated_benchmark.py",
        ]
        if args.lm_config:
            lm_cmd.extend(["--config", str(args.lm_config)])
        steps.append(run_and_log("lm_eval", lm_cmd, run_dir / "01_lm_eval.log"))

    if not args.skip_deepeval:
        deepeval_cmd = [
            "py",
            "-3",
            "scripts/evaluation/deepeval_ethics_test.py",
            "--model-runner",
            args.deepeval_runner,
            "--model-name",
            args.deepeval_model,
        ]
        steps.append(run_and_log("deepeval", deepeval_cmd, run_dir / "02_deepeval.log"))

    if not args.skip_elyza:
        elyza_output_dir = run_dir / "elyza"
        elyza_output_dir.mkdir(parents=True, exist_ok=True)
        for model_name in args.elyza_models:
            elyza_cmd = [
                "py",
                "-3",
                "scripts/evaluation/elyza_benchmark.py",
                "--model-name",
                model_name,
                "--output-dir",
                str(elyza_output_dir),
            ]
            if args.elyza_limit:
                elyza_cmd.extend(["--limit", str(args.elyza_limit)])
            step_name = f"elyza_{model_name.replace(':', '_').replace('/', '_')}"
            steps.append(run_and_log(step_name, elyza_cmd, run_dir / f"03_{step_name}.log"))

    if not args.skip_promptfoo:
        promptfoo_cmd = [
            "py",
            "-3",
            "scripts/evaluation/promptfoo_ab_test.py",
            "--config",
            str(args.promptfoo_config),
            "--output-root",
            str(run_dir / "promptfoo"),
            "--html",
            "--json",
            "--use-npx",
        ]
        steps.append(run_and_log("promptfoo", promptfoo_cmd, run_dir / "04_promptfoo.log"))

    metadata = {
        "timestamp": timestamp,
        "worktree": get_worktree_name(),
        "run_dir": str(run_dir),
    }
    with (run_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                **metadata,
                "steps": steps,
            },
            f,
            indent=2,
        )

    doc_report = DOCS_REPORT_ROOT / f"{timestamp}_{metadata['worktree']}_industry_standard_report.md"
    generate_markdown(doc_report, metadata, steps)

    overall_exit = max((step["exit_code"] or 0) for step in steps) if steps else 0
    print(f"[SUMMARY] Logs stored in {run_dir}")
    print(f"[SUMMARY] Markdown report: {doc_report}")
    play_audio()
    if overall_exit != 0:
        raise SystemExit(overall_exit)


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
promptfoo A/B テストラッパー

- configs/promptfoo_config.yaml を利用
- HTML/JSONレポートを D:/webdataset/benchmark_results/promptfoo/ に保存
- Node.js 環境を自動チェック
"""

import argparse
import json
import subprocess
import textwrap
from datetime import datetime
from pathlib import Path
from typing import List

DEFAULT_CONFIG = Path("configs/promptfoo_config.yaml")
DEFAULT_OUTPUT_ROOT = Path(r"D:/webdataset/benchmark_results/promptfoo")
CHECK_NODE_SCRIPT = Path("scripts/utils/check_nodejs.bat")
PLAY_AUDIO_SCRIPT = Path("scripts/utils/play_audio_notification.ps1")


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


def ensure_node() -> None:
    if CHECK_NODE_SCRIPT.exists():
        subprocess.run(
            [str(CHECK_NODE_SCRIPT)],
            check=False,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="promptfoo A/B benchmark runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
            Examples:
              python scripts/evaluation/promptfoo_ab_test.py
              python scripts/evaluation/promptfoo_ab_test.py --config configs/custom.yaml --use-npx
            """
        ),
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="promptfoo設定ファイル")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="結果保存ディレクトリ")
    parser.add_argument("--label", default=None, help="ラン識別子 (未指定はtimestamp)")
    parser.add_argument("--skip-node-check", action="store_true", help="Node.jsチェックをスキップ")
    parser.add_argument("--use-npx", action="store_true", help="npx promptfoo@latest を使用")
    parser.add_argument("--html", action="store_true", help="HTMLレポートを生成")
    parser.add_argument("--json", action="store_true", help="JSONレポートを生成")
    parser.add_argument("--dry-run", action="store_true", help="コマンドのみ表示して実行しない")
    return parser.parse_args()


def build_command(args: argparse.Namespace, output_dir: Path) -> List[str]:
    cmd: List[str]
    if args.use_npx:
        cmd = ["npx", "--yes", "promptfoo@latest", "eval"]
    else:
        cmd = ["promptfoo", "eval"]

    cmd.extend(["--config", str(args.config)])
    cmd.extend(["--output", str(output_dir)])
    if args.html:
        cmd.extend(["--format", "html"])
    if args.json:
        cmd.extend(["--format", "json"])
    return cmd


def main() -> None:
    args = parse_args()
    if not args.config.exists():
        raise FileNotFoundError(f"Config not found: {args.config}")

    if not args.skip_node_check:
        ensure_node()

    args.output_root.mkdir(parents=True, exist_ok=True)
    label = args.label or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_root / f"promptfoo_{label}"
    run_dir.mkdir(parents=True, exist_ok=True)

    cmd = build_command(args, run_dir)

    print("[PROMPTFOO] Command:")
    print(" ".join(cmd))
    if args.dry_run:
        print("[PROMPTFOO] Dry-run complete.")
        return

    completed = subprocess.run(cmd, check=False)
    if completed.returncode != 0:
        print(f"[PROMPTFOO] Failed with exit code {completed.returncode}")
        play_audio()
        raise SystemExit(completed.returncode)

    summary = {
        "config": str(args.config),
        "output_dir": str(run_dir),
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "command": cmd,
    }
    with (run_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[PROMPTFOO] Results stored at {run_dir}")
    play_audio()


if __name__ == "__main__":
    main()


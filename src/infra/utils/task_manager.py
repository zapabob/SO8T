#!/usr/bin/env python3
"""
汎用タスクマネージャー
すべての時間のかかる作業にチェックポイント機能を適用
"""

import os
import sys
import time
import signal
import argparse
from pathlib import Path
from typing import Callable, Any, Dict
from src.utils.checkpoint_manager import create_task_manager, checkpoint_context, with_checkpointing


def run_with_checkpointing(task_func: Callable, task_name: str, output_dir: str = None, **kwargs):
    """
    任意のタスク関数にチェックポイント機能を適用して実行

    Args:
        task_func: 実行するタスク関数
        task_name: タスク名（チェックポイント識別用）
        output_dir: チェックポイント保存ディレクトリ
        **kwargs: タスク関数に渡す引数
    """
    manager = create_task_manager(task_name, output_dir)

    def signal_handler(signum, frame):
        """シグナルハンドラー（Ctrl+Cなどで中断された場合）"""
        print(f"\n⚠️  Task {task_name} interrupted! Saving checkpoint...")
        manager.save_checkpoint(step_info="interrupted")
        sys.exit(1)

    # シグナルハンドラーを設定
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # 自動再開チェック
        def resume_func(checkpoint_path):
            print(f"Resuming {task_name} from {checkpoint_path}")
            # 実際の再開ロジックはタスク依存

        if manager.auto_resume(resume_func):
            return

        # タスク実行
        print(f"🚀 Starting task: {task_name}")
        start_time = time.time()

        result = task_func(manager=manager, **kwargs)

        elapsed = time.time() - start_time
        print(f"Task {task_name} completed in {elapsed:.2f} seconds")
        # 最終チェックポイント保存
        manager.save_checkpoint(data=result, step_info="completed")
        manager.mark_completed()
        print(f"Result: {result.get('status', 'unknown') if result else 'unknown'}")

        return result

    except KeyboardInterrupt:
        print(f"\n⚠️  Task {task_name} interrupted by user")
        manager.save_checkpoint(step_info="user_interrupt")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Task {task_name} failed: {e}")
        manager.save_checkpoint(step_info="error")
        raise


# ============================================================================
# 特定のタスク用のラッパー関数
# ============================================================================

def run_rlpo_training(**kwargs):
    """RLPO学習実行"""
    from src.training.rlpo_science_nsfw_automated import main as rlpo_main
    # コマンドライン引数をシミュレート
    sys.argv = ['rlpo_training'] + [f'--{k}={v}' for k, v in kwargs.items()]
    rlpo_main()

def run_dataset_creation(dataset_type: str, **kwargs):
    """データセット作成実行"""
    if dataset_type == "science":
        from create_science_dataset import main as create_main
    elif dataset_type == "nsfw_drug":
        from src.data.create_nsfw_drug_dataset import main as create_main
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    create_main()

def run_evaluation(**kwargs):
    """評価実行"""
    from run_evaluation import main as eval_main
    eval_main()

def run_benchmark(**kwargs):
    """ベンチマーク実行"""
    from src.benchmark import main as bench_main
    bench_main()

def run_gguf_conversion(model_path: str, **kwargs):
    """GGUF変換実行"""
    import subprocess
    import sys

    cmd = [
        sys.executable,
        "scripts/conversion/convert_hf_to_gguf.py",
        model_path,
        "--outfile", kwargs.get('output_file', f"{model_path}.gguf"),
        "--outtype", kwargs.get('quantization', 'q8_0')
    ]

    # 追加の引数を処理
    if kwargs.get('vocab_only'):
        cmd.append('--vocab-only')
    if kwargs.get('verbose'):
        cmd.append('--verbose')

    subprocess.run(cmd, check=True)


def run_sunshine_pipeline(**kwargs):
    """サンシャインパイプライン実行（SO8T実験）"""
    import subprocess
    import sys

    cmd = [
        sys.executable,
        "scripts/pipeline/sunshine_pipeline.py"
    ]

    # 引数処理
    if kwargs.get('skip_baseline'):
        cmd.append('--skip_baseline')
    if kwargs.get('model_name'):
        cmd.extend(['--model_name', kwargs['model_name']])

    subprocess.run(cmd, check=True)


def run_data_creation(dataset_type: str, **kwargs):
    """データセット作成実行"""
    import subprocess
    import sys

    if dataset_type == "science":
        script = "create_science_dataset.py"
    elif dataset_type == "nsfw_drug":
        script = "scripts/data/create_nsfw_drug_dataset.py"
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    cmd = [sys.executable, script]
    subprocess.run(cmd, check=True)


def run_evaluation_pipeline(**kwargs):
    """評価パイプライン実行"""
    import subprocess
    import sys

    cmd = [sys.executable, "run_evaluation.py"]

    if kwargs.get('model_path'):
        cmd.extend(['--model_path', kwargs['model_path']])
    if kwargs.get('output_dir'):
        cmd.extend(['--output_dir', kwargs['output_dir']])

    subprocess.run(cmd, check=True)


def run_report_generation(**kwargs):
    """レポート生成実行"""
    import subprocess
    import sys

    cmd = [sys.executable, "generate_training_report.py"]

    if kwargs.get('input_dir'):
        cmd.extend(['--input_dir', kwargs['input_dir']])
    if kwargs.get('output_file'):
        cmd.extend(['--output_file', kwargs['output_file']])

    subprocess.run(cmd, check=True)


# ============================================================================
# メイン実行関数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Universal Task Manager with Checkpointing')
    parser.add_argument('task', help='Task to run (rlpo, dataset, evaluation, benchmark, gguf, sunshine, data, report)')
    parser.add_argument('--task_name', default=None, help='Custom task name')
    parser.add_argument('--output_dir', default=None, help='Output directory')
    parser.add_argument('--dataset_type', default='science', help='Dataset type for dataset creation')
    parser.add_argument('--model_path', default=None, help='Model path for GGUF conversion or evaluation')
    parser.add_argument('--quantization', default='q8_0', help='Quantization type for GGUF conversion')
    parser.add_argument('--output_file', default=None, help='Output file for GGUF conversion or reports')
    parser.add_argument('--skip_baseline', action='store_true', help='Skip baseline run in sunshine pipeline')
    parser.add_argument('--input_dir', default=None, help='Input directory for report generation')

    # RLPO固有の引数
    parser.add_argument('--model_name', default='microsoft/phi-3.5-mini-instruct')
    parser.add_argument('--max_steps', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=2)

    args = parser.parse_args()

    # タスク名設定
    task_name = args.task_name or f"{args.task}_{int(time.time())}"
    output_dir = args.output_dir or f"checkpoints/{task_name}"

    # タスク実行
    if args.task == "rlpo":
        run_with_checkpointing(
            run_rlpo_training,
            task_name,
            output_dir,
            model_name=args.model_name,
            max_steps=args.max_steps,
            batch_size=args.batch_size
        )
    elif args.task == "dataset" or args.task == "data":
        run_with_checkpointing(
            lambda **kwargs: run_data_creation(args.dataset_type, **kwargs),
            task_name,
            output_dir
        )
    elif args.task == "evaluation":
        run_with_checkpointing(
            run_evaluation_pipeline,
            task_name,
            output_dir,
            model_path=args.model_path
        )
    elif args.task == "benchmark":
        run_with_checkpointing(
            run_benchmark,
            task_name,
            output_dir
        )
    elif args.task == "gguf":
        if not args.model_path:
            print("❌ --model_path required for GGUF conversion")
            sys.exit(1)

        run_with_checkpointing(
            lambda **kwargs: run_gguf_conversion(args.model_path, **kwargs),
            task_name,
            output_dir,
            quantization=args.quantization,
            output_file=args.output_file
        )
    elif args.task == "sunshine":
        run_with_checkpointing(
            run_sunshine_pipeline,
            task_name,
            output_dir,
            skip_baseline=args.skip_baseline,
            model_name=args.model_name
        )
    elif args.task == "report":
        run_with_checkpointing(
            run_report_generation,
            task_name,
            output_dir,
            input_dir=args.input_dir,
            output_file=args.output_file
        )
    else:
        print(f"❌ Unknown task: {args.task}")
        print("Available tasks: rlpo, dataset/data, evaluation, benchmark, gguf, sunshine, report")
        sys.exit(1)


if __name__ == "__main__":
    main()

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
from scripts.utils.checkpoint_manager import create_task_manager, checkpoint_context, with_checkpointing


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
        print(".1f"
        # 最終チェックポイント保存
        manager.save_checkpoint(data=result, step_info="completed")
        manager.mark_completed()

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
    from scripts.training.rlpo_science_nsfw_automated import main as rlpo_main
    # コマンドライン引数をシミュレート
    sys.argv = ['rlpo_training'] + [f'--{k}={v}' for k, v in kwargs.items()]
    rlpo_main()

def run_dataset_creation(dataset_type: str, **kwargs):
    """データセット作成実行"""
    if dataset_type == "science":
        from create_science_dataset import main as create_main
    elif dataset_type == "nsfw_drug":
        from scripts.data.create_nsfw_drug_dataset import main as create_main
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    create_main()

def run_evaluation(**kwargs):
    """評価実行"""
    from run_evaluation import main as eval_main
    eval_main()

def run_benchmark(**kwargs):
    """ベンチマーク実行"""
    from scripts.benchmark import main as bench_main
    bench_main()


# ============================================================================
# メイン実行関数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Universal Task Manager with Checkpointing')
    parser.add_argument('task', help='Task to run (rlpo, dataset, evaluation, benchmark)')
    parser.add_argument('--task_name', default=None, help='Custom task name')
    parser.add_argument('--output_dir', default=None, help='Output directory')
    parser.add_argument('--dataset_type', default='science', help='Dataset type for dataset creation')

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
    elif args.task == "dataset":
        run_with_checkpointing(
            lambda **kwargs: run_dataset_creation(args.dataset_type, **kwargs),
            task_name,
            output_dir
        )
    elif args.task == "evaluation":
        run_with_checkpointing(
            run_evaluation,
            task_name,
            output_dir
        )
    elif args.task == "benchmark":
        run_with_checkpointing(
            run_benchmark,
            task_name,
            output_dir
        )
    else:
        print(f"❌ Unknown task: {args.task}")
        sys.exit(1)


if __name__ == "__main__":
    main()

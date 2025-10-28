from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List

from dataset_synth import generate_samples, partition_samples, write_jsonl
from shared.utils import notify_task_completion


ROOT = Path(__file__).resolve().parent.parent


def _run_python(script: str, extra_args: List[str]) -> None:
    cmd = [sys.executable, str(ROOT / script), *extra_args]
    subprocess.run(cmd, check=True)


def cmd_generate_data(args: argparse.Namespace) -> None:
    from random import Random

    rng = Random(args.seed)
    samples = generate_samples(rng, args.count)
    rng.shuffle(samples)
    train, val, test = partition_samples(samples, args.train_ratio, args.val_ratio)
    output_dir = Path(args.output_dir)
    write_jsonl(output_dir / "train.jsonl", train)
    write_jsonl(output_dir / "val.jsonl", val)
    write_jsonl(output_dir / "test.jsonl", test)
    metadata = {
        "total": len(samples),
        "train": len(train),
        "val": len(val),
        "test": len(test),
        "seed": args.seed,
    }
    (output_dir / "metadata.json").write_text(
        __import__("json").dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print(__import__("json").dumps(metadata, indent=2))
    
    # データ生成完了の音声通知
    notify_task_completion("データ生成")


def cmd_train(args: argparse.Namespace) -> None:
    extra = ["--config", str(args.config)]
    _run_python("train.py", extra)
    
    # 訓練完了の音声通知
    notify_task_completion("モデル訓練")


def cmd_eval(args: argparse.Namespace) -> None:
    extra = [
        "--checkpoint",
        str(args.checkpoint),
        "--config",
        str(args.config),
        "--split",
        args.split,
        "--output",
        str(args.output),
    ]
    if args.batch_size:
        extra.extend(["--batch-size", str(args.batch_size)])
    _run_python("eval.py", extra)
    
    # 評価完了の音声通知
    notify_task_completion("モデル評価")


def cmd_infer(args: argparse.Namespace) -> None:
    extra = ["--checkpoint", str(args.checkpoint)]
    if args.env:
        extra.extend(["--env", args.env])
    if args.cmd:
        extra.extend(["--cmd", args.cmd])
    if args.safe:
        extra.extend(["--safe", args.safe])
    if args.input_json:
        extra.extend(["--input-json", str(args.input_json)])
    _run_python("infer.py", extra)
    
    # 推論完了の音声通知
    notify_task_completion("モデル推論")


def cmd_report(args: argparse.Namespace) -> None:
    _run_python("scripts/summarize_results.py", [])
    
    # レポート生成完了の音声通知
    notify_task_completion("レポート生成")


def cmd_train_safety(args: argparse.Namespace) -> None:
    extra = ["--config", str(args.config)]
    if args.data_dir:
        extra.extend(["--data_dir", str(args.data_dir)])
    if args.output_dir:
        extra.extend(["--output_dir", str(args.output_dir)])
    if args.seed:
        extra.extend(["--seed", str(args.seed)])
    if args.no_resume:
        extra.append("--no_resume")
    _run_python("train_safety.py", extra)
    
    # 安全重視訓練完了の音声通知
    notify_task_completion("安全重視モデル訓練")


def cmd_visualize_safety(args: argparse.Namespace) -> None:
    extra = []
    if args.log_file:
        extra.extend(["--log_file", str(args.log_file)])
    if args.output_dir:
        extra.extend(["--output_dir", str(args.output_dir)])
    _run_python("visualize_safety_training.py", extra)
    
    # 安全可視化完了の音声通知
    notify_task_completion("安全可視化")


def cmd_test_safety(args: argparse.Namespace) -> None:
    extra = ["--checkpoint", str(args.checkpoint)]
    if args.vocab:
        extra.extend(["--vocab", str(args.vocab)])
    if args.output_dir:
        extra.extend(["--output_dir", str(args.output_dir)])
    _run_python("test_safety_inference.py", extra)
    
    # 安全テスト完了の音声通知
    notify_task_completion("安全テスト")


def cmd_demonstrate_safety(args: argparse.Namespace) -> None:
    extra = ["--checkpoint", str(args.checkpoint)]
    if args.vocab:
        extra.extend(["--vocab", str(args.vocab)])
    if args.output_dir:
        extra.extend(["--output_dir", str(args.output_dir)])
    _run_python("demonstrate_safety_inference.py", extra)
    
    # 安全実証完了の音声通知
    notify_task_completion("安全実証")


def cmd_pipeline_safety(args: argparse.Namespace) -> None:
    """安全重視SO8Tの完全パイプライン実行"""
    print("🚀 Starting Safety-Aware SO8T Pipeline...")
    print("=" * 60)
    
    # 1. 訓練
    print("📚 Step 1: Training safety-aware model...")
    try:
        cmd_train_safety(args)
        print("✅ Training completed successfully!")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return
    
    # 2. 可視化
    print("\n📊 Step 2: Creating visualizations...")
    try:
        cmd_visualize_safety(args)
        print("✅ Visualization completed successfully!")
    except Exception as e:
        print(f"❌ Visualization failed: {e}")
        return
    
    # 3. テスト
    print("\n🧪 Step 3: Running safety tests...")
    try:
        # デフォルトのチェックポイントパスを設定
        checkpoint_path = args.output_dir / "safety_model_best.pt"
        test_args = argparse.Namespace(
            checkpoint=checkpoint_path,
            vocab=args.data_dir / "vocab.json" if args.data_dir else Path("data/vocab.json"),
            output_dir=args.output_dir / "safety_test_results"
        )
        cmd_test_safety(test_args)
        print("✅ Safety testing completed successfully!")
    except Exception as e:
        print(f"❌ Safety testing failed: {e}")
        return
    
    # 4. 実証テスト
    print("\n🎭 Step 4: Running safety demonstration...")
    try:
        # デフォルトのチェックポイントパスを設定
        checkpoint_path = args.output_dir / "safety_model_best.pt"
        demo_args = argparse.Namespace(
            checkpoint=checkpoint_path,
            vocab=args.data_dir / "vocab.json" if args.data_dir else Path("data/vocab.json"),
            output_dir=args.output_dir / "safety_demonstration_results"
        )
        cmd_demonstrate_safety(demo_args)
        print("✅ Safety demonstration completed successfully!")
    except Exception as e:
        print(f"❌ Safety demonstration failed: {e}")
        return
    
    # 5. 実装ログ生成
    print("\n📝 Step 5: Generating implementation log...")
    try:
        from scripts.impl_logger import generate_impl_log
        generate_impl_log(
            feature_name="安全重視SO8T",
            summary_file=args.output_dir / "safety_training_log.jsonl",
            output_dir=Path("_docs")
        )
        print("✅ Implementation log generated successfully!")
    except Exception as e:
        print(f"⚠️ Implementation log generation failed: {e}")
        print("Continuing without log generation...")
    
    print("\n🎉 Safety-Aware SO8T Pipeline completed successfully!")
    print("=" * 60)
    
    # パイプライン完了の音声通知
    notify_task_completion("安全重視SO8Tパイプライン")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SO8T experiment orchestrator.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    gen = subparsers.add_parser("generate-data", help="Generate synthetic dataset.")
    gen.add_argument("--count", type=int, default=3000)
    gen.add_argument("--train-ratio", type=float, default=0.8)
    gen.add_argument("--val-ratio", type=float, default=0.1)
    gen.add_argument("--seed", type=int, default=7)
    gen.add_argument("--output-dir", type=str, default="data")
    gen.set_defaults(func=cmd_generate_data)

    train_parser = subparsers.add_parser("train", help="Train the SO8T model.")
    train_parser.add_argument("--config", type=Path, default=ROOT / "configs/train_default.yaml")
    train_parser.set_defaults(func=cmd_train)

    eval_parser = subparsers.add_parser("eval", help="Evaluate a checkpoint.")
    eval_parser.add_argument("--checkpoint", type=Path, default=ROOT / "chk/so8t_default.pt")
    eval_parser.add_argument("--config", type=Path, default=ROOT / "configs/train_default.yaml")
    eval_parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    eval_parser.add_argument("--output", type=Path, default=ROOT / "chk/eval_summary.json")
    eval_parser.add_argument("--batch-size", type=int, default=64)
    eval_parser.set_defaults(func=cmd_eval)

    infer_parser = subparsers.add_parser("infer", help="Run single-sample inference.")
    infer_parser.add_argument("--checkpoint", type=Path, required=True)
    infer_parser.add_argument("--env", type=str)
    infer_parser.add_argument("--cmd", type=str)
    infer_parser.add_argument("--safe", type=str)
    infer_parser.add_argument("--input-json", type=Path)
    infer_parser.set_defaults(func=cmd_infer)

    report_parser = subparsers.add_parser("report", help="Summarize experiment outputs.")
    report_parser.set_defaults(func=cmd_report)

    # Safety-aware commands
    train_safety_parser = subparsers.add_parser("train-safety", help="Train safety-aware SO8T model.")
    train_safety_parser.add_argument("--config", type=Path, default=ROOT / "configs/train_safety.yaml")
    train_safety_parser.add_argument("--data_dir", type=Path, default=ROOT / "data")
    train_safety_parser.add_argument("--output_dir", type=Path, default=ROOT / "chk")
    train_safety_parser.add_argument("--seed", type=int, default=42)
    train_safety_parser.add_argument("--no_resume", action="store_true", help="Disable auto-resume from checkpoint")
    train_safety_parser.set_defaults(func=cmd_train_safety)

    visualize_safety_parser = subparsers.add_parser("visualize-safety", help="Visualize safety training results.")
    visualize_safety_parser.add_argument("--log_file", type=Path, default=ROOT / "chk/safety_training_log.jsonl")
    visualize_safety_parser.add_argument("--output_dir", type=Path, default=ROOT / "safety_visualizations")
    visualize_safety_parser.set_defaults(func=cmd_visualize_safety)

    test_safety_parser = subparsers.add_parser("test-safety", help="Test safety-aware SO8T model.")
    test_safety_parser.add_argument("--checkpoint", type=Path, default=ROOT / "chk/safety_model_best.pt")
    test_safety_parser.add_argument("--vocab", type=Path, default=ROOT / "data/vocab.json")
    test_safety_parser.add_argument("--output_dir", type=Path, default=ROOT / "safety_test_results")
    test_safety_parser.set_defaults(func=cmd_test_safety)

    demonstrate_safety_parser = subparsers.add_parser("demonstrate-safety", help="Demonstrate safety-aware SO8T model.")
    demonstrate_safety_parser.add_argument("--checkpoint", type=Path, default=ROOT / "chk/safety_model_best.pt")
    demonstrate_safety_parser.add_argument("--vocab", type=Path, default=ROOT / "data/vocab.json")
    demonstrate_safety_parser.add_argument("--output_dir", type=Path, default=ROOT / "safety_demonstration_results")
    demonstrate_safety_parser.set_defaults(func=cmd_demonstrate_safety)

    pipeline_safety_parser = subparsers.add_parser("pipeline-safety", help="Run complete safety-aware SO8T pipeline.")
    pipeline_safety_parser.add_argument("--config", type=Path, default=ROOT / "configs/train_safety.yaml")
    pipeline_safety_parser.add_argument("--data_dir", type=Path, default=ROOT / "data")
    pipeline_safety_parser.add_argument("--output_dir", type=Path, default=ROOT / "chk")
    pipeline_safety_parser.add_argument("--seed", type=int, default=42)
    pipeline_safety_parser.add_argument("--no_resume", action="store_true", help="Disable auto-resume from checkpoint")
    pipeline_safety_parser.set_defaults(func=cmd_pipeline_safety)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

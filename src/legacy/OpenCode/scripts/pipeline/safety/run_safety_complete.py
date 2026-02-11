#!/usr/bin/env python3
"""
Safety-Aware SO8T Complete Pipeline Runner
CLIなしで学習推論実証を完全実行するスクリプト
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime


def print_banner():
    """バナーを表示"""
    print("=" * 80)
    print("Safety-Aware SO8T Complete Pipeline Runner")
    print("   学習推論実証の完全実行システム")
    print("=" * 80)


def print_step(step_num: int, total_steps: int, title: str, description: str):
    """ステップを表示"""
    print(f"\nStep {step_num}/{total_steps}: {title}")
    print(f"   {description}")
    print("-" * 60)


def run_command(command: list, step_name: str) -> bool:
    """コマンドを実行"""
    print(f"Executing: {' '.join(command)}")
    start_time = time.time()
    
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8')
        elapsed_time = time.time() - start_time
        print(f"SUCCESS: {step_name} completed successfully! (took {elapsed_time:.1f}s)")
        
        # 出力を表示（重要な部分のみ）
        if result.stdout:
            lines = result.stdout.strip().split('\n')
            for line in lines[-10:]:  # 最後の10行のみ表示
                if line.strip():
                    print(f"   {line}")
        
        return True
    except subprocess.CalledProcessError as e:
        elapsed_time = time.time() - start_time
        print(f"FAILED: {step_name} failed after {elapsed_time:.1f}s")
        print(f"   Exit code: {e.returncode}")
        if e.stdout:
            print(f"   stdout: {e.stdout}")
        if e.stderr:
            print(f"   stderr: {e.stderr}")
        return False
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"FAILED: {step_name} failed after {elapsed_time:.1f}s: {e}")
        return False


def check_required_files():
    """必要なファイルの存在確認"""
    required_files = [
        "train_safety.py",
        "visualize_safety_training.py", 
        "test_safety_inference.py",
        "demonstrate_safety_inference.py",
        "scripts/impl_logger.py",
        "configs/train_safety.yaml"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print("ERROR: Missing required files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        return False
    
    print("SUCCESS: All required files found!")
    return True


def check_and_build_vocab(data_dir: Path, output_dir: Path):
    """語彙ファイルの存在確認と構築"""
    vocab_file = data_dir / "vocab.json"
    
    if vocab_file.exists():
        print(f"SUCCESS: Vocabulary file found: {vocab_file}")
        return True
    
    print(f"WARNING: Vocabulary file not found: {vocab_file}")
    print("Building vocabulary from training data...")
    
    # 語彙構築コマンド
    command = [
        sys.executable, "build_vocab.py",
        "--data_dir", str(data_dir),
        "--output_file", str(vocab_file)
    ]
    
    success = run_command(command, "Vocabulary Building")
    return success


def main():
    parser = argparse.ArgumentParser(description="Run complete safety-aware SO8T pipeline")
    parser.add_argument("--config", type=Path, default=Path("configs/train_safety.yaml"),
                       help="Training config path")
    parser.add_argument("--data_dir", type=Path, default=Path("data"),
                       help="Data directory")
    parser.add_argument("--output_dir", type=Path, default=Path("chk"),
                       help="Output directory")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--no_resume", action="store_true",
                       help="Disable auto-resume from checkpoint")
    parser.add_argument("--skip_training", action="store_true",
                       help="Skip training step (use existing model)")
    parser.add_argument("--skip_visualization", action="store_true",
                       help="Skip visualization step")
    parser.add_argument("--skip_testing", action="store_true",
                       help="Skip testing step")
    parser.add_argument("--skip_demonstration", action="store_true",
                       help="Skip demonstration step")
    
    args = parser.parse_args()
    
    print_banner()
    
    # 必要なファイルの確認
    print("\nChecking required files...")
    if not check_required_files():
        print("\nERROR: Please ensure all required files are present before running.")
        sys.exit(1)
    
    # 語彙ファイルの確認と構築
    print("\nChecking vocabulary...")
    if not check_and_build_vocab(args.data_dir, args.output_dir):
        print("\nERROR: Failed to build vocabulary. Please check the data files.")
        sys.exit(1)
    
    # 出力ディレクトリを作成
    args.output_dir.mkdir(exist_ok=True)
    
    # 実行ステップを決定
    steps = []
    if not args.skip_training:
        steps.append(("training", "Safety-Aware Model Training"))
    if not args.skip_visualization:
        steps.append(("visualization", "Training Results Visualization"))
    if not args.skip_testing:
        steps.append(("testing", "Safety Inference Testing"))
    if not args.skip_demonstration:
        steps.append(("demonstration", "Safety Demonstration"))
    steps.append(("logging", "Implementation Log Generation"))
    
    total_steps = len(steps)
    print(f"\nPipeline Steps: {total_steps}")
    for i, (step, desc) in enumerate(steps, 1):
        print(f"   {i}. {desc}")
    
    # 各ステップを実行
    start_time = time.time()
    
    for i, (step, description) in enumerate(steps, 1):
        print_step(i, total_steps, description, f"Executing {step} step...")
        
        if step == "training":
            command = [
                sys.executable, "train_safety.py",
                "--config", str(args.config),
                "--data_dir", str(args.data_dir),
                "--output_dir", str(args.output_dir),
                "--seed", str(args.seed)
            ]
            if args.no_resume:
                command.append("--no_resume")
            
            success = run_command(command, "Training")
            
        elif step == "visualization":
            command = [
                sys.executable, "visualize_safety_training.py",
                "--log_file", str(args.output_dir / "safety_training_log.jsonl"),
                "--output_dir", str(args.output_dir / "safety_visualizations")
            ]
            success = run_command(command, "Visualization")
            
        elif step == "testing":
            command = [
                sys.executable, "test_safety_inference.py",
                "--checkpoint", str(args.output_dir / "safety_model_best.pt"),
                "--vocab", str(args.data_dir / "vocab.json"),
                "--output_dir", str(args.output_dir / "safety_test_results")
            ]
            success = run_command(command, "Testing")
            
        elif step == "demonstration":
            command = [
                sys.executable, "demonstrate_safety_inference.py",
                "--checkpoint", str(args.output_dir / "safety_model_best.pt"),
                "--vocab", str(args.data_dir / "vocab.json"),
                "--output_dir", str(args.output_dir / "safety_demonstration_results")
            ]
            success = run_command(command, "Demonstration")
            
        elif step == "logging":
            command = [
                sys.executable, "scripts/impl_logger.py",
                "--feature", "安全重視SO8T",
                "--summary-file", str(args.output_dir / "safety_training_log.jsonl"),
                "--output-dir", "_docs"
            ]
            success = run_command(command, "Logging")
        
        if not success:
            print(f"\nFAILED: Pipeline failed at step {i}: {description}")
            print("   Check the error messages above for details.")
            sys.exit(1)
    
    # 完了サマリー
    total_time = time.time() - start_time
    print(f"\nSUCCESS: Pipeline completed successfully!")
    print("=" * 80)
    print(f"Total execution time: {total_time:.1f} seconds")
    print(f"Output directory: {args.output_dir}")
    
    # 生成されたファイルを確認
    print("\nGenerated Files:")
    result_files = [
        ("safety_model_best.pt", "Trained model checkpoint"),
        ("safety_training_log.jsonl", "Training log"),
        ("safety_visualizations/", "Training visualizations"),
        ("safety_test_results/", "Test results"),
        ("safety_demonstration_results/", "Demonstration results"),
        ("_docs/", "Implementation logs")
    ]
    
    for file_path, description in result_files:
        full_path = args.output_dir / file_path if not file_path.endswith('/') else args.output_dir / file_path
        if full_path.exists():
            print(f"   SUCCESS: {file_path} - {description}")
        else:
            print(f"   MISSING: {file_path} - {description} (not found)")
    
    # 実装ログの確認
    log_files = list(Path("_docs").glob("*安全重視SO8T*.md"))
    if log_files:
        print(f"\nImplementation Logs:")
        for log_file in log_files:
            print(f"   {log_file.name}")
    
    print(f"\nSafety-Aware SO8T Complete Pipeline finished!")
    print("   Check the output files for detailed results.")


if __name__ == "__main__":
    main()

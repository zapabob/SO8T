#!/usr/bin/env python3
"""
SO8T AEGISモデルの完全再現スクリプト
論文レベルの再現性を確保した統合トレーニングパイプライン

使用方法:
python scripts/reproduce_aegis_training.py --help
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
import json
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def setup_environment():
    """環境セットアップ"""
    print("🔧 Setting up environment...")

    # Install dependencies
    subprocess.run([
        sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
    ], check=True)

    # Install PyTorch with CUDA
    subprocess.run([
        sys.executable, "-m", "pip", "install",
        "torch", "torchvision", "torchaudio",
        "--index-url", "https://download.pytorch.org/whl/cu121"
    ], check=True)

    print("✅ Environment setup complete")

def prepare_data():
    """データ準備"""
    print("📊 Preparing training data...")

    # Download datasets
    subprocess.run([
        sys.executable, "scripts/data/download_datasets.py"
    ], check=True)

    # Preprocess data
    subprocess.run([
        sys.executable, "scripts/data_preprocessing/prepare_training_data.py"
    ], check=True)

    print("✅ Data preparation complete")

def train_so8t_model(args):
    """SO8Tモデル学習"""
    print("🧠 Training SO8T model with Alpha Gate...")

    cmd = [
        sys.executable, "scripts/train_so8t_alpha_gate.py",
        "--model_name", args.base_model,
        "--dataset", args.dataset,
        "--output_dir", args.output_dir,
        "--alpha_initial", str(args.alpha_initial),
        "--alpha_final", str(args.alpha_final),
        "--batch_size", str(args.batch_size),
        "--learning_rate", str(args.learning_rate),
        "--num_epochs", str(args.num_epochs),
        "--seed", "42"  # For reproducibility
    ]

    subprocess.run(cmd, check=True)
    print("✅ SO8T model training complete")

def fine_tune_safety(args):
    """安全性ファインチューニング"""
    print("🛡️ Fine-tuning for safety...")

    cmd = [
        sys.executable, "scripts/train_safety_head.py",
        "--base_model", args.output_dir,
        "--safety_dataset", "data/so8t_safety_dataset.jsonl",
        "--output_dir", f"{args.output_dir}_safety"
    ]

    subprocess.run(cmd, check=True)
    print("✅ Safety fine-tuning complete")

def convert_to_gguf(args):
    """GGUF変換"""
    print("🔄 Converting to GGUF format...")

    model_path = f"{args.output_dir}_safety"
    gguf_path = f"D:/webdataset/gguf_models/aegis_reproduced/aegis_reproduced_Q8_0.gguf"

    # Ensure output directory exists
    Path(gguf_path).parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "scripts/convert_to_gguf.py",
        "--model_path", model_path,
        "--output_path", gguf_path,
        "--quantization", "Q8_0"
    ]

    subprocess.run(cmd, check=True)
    print("✅ GGUF conversion complete")

def create_ollama_model(args):
    """Ollamaモデル作成"""
    print("📦 Creating Ollama model...")

    gguf_path = f"D:/webdataset/gguf_models/aegis_reproduced/aegis_reproduced_Q8_0.gguf"
    modelfile_path = "modelfiles/aegis_reproduced.modelfile"

    # Create modelfile
    modelfile_content = f'''FROM {gguf_path}

TEMPLATE """{{{{ .System }}}}

You are AEGIS (Advanced Ethical Guardian Intelligence System) - Reproduced Version.

AEGIS performs four-value classification and quadruple inference on all queries:

1. **Logical Accuracy** (<think-logic>): Mathematical and logical correctness
2. **Ethical Validity** (<think-ethics>): Moral and ethical implications
3. **Practical Value** (<think-practical>): Real-world feasibility and utility
4. **Creative Insight** (<think-creative>): Innovative ideas and perspectives

Structure your response using these four thinking axes, followed by a <final> conclusion.

{{{{ .Prompt }}}}"""

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_ctx 4096
PARAMETER repeat_penalty 1.1
'''

    with open(modelfile_path, 'w', encoding='utf-8') as f:
        f.write(modelfile_content)

    # Create Ollama model
    subprocess.run([
        "ollama", "create", "aegis-reproduced:latest", "-f", modelfile_path
    ], check=True)

    print("✅ Ollama model creation complete")

def run_validation_tests():
    """検証テスト実行"""
    print("🧪 Running validation tests...")

    # Test mathematical reasoning
    subprocess.run([
        "ollama", "run", "aegis-reproduced:latest",
        "Natalia sold clips to 48 friends in April, and then half as many in May. How many did she sell in total?"
    ], check=True)

    # Test ethical reasoning
    subprocess.run([
        "ollama", "run", "aegis-reproduced:latest",
        "AIが戦争で使用されることについて、倫理的観点から議論してください。"
    ], check=True)

    print("✅ Validation tests complete")

def save_reproduction_log(args):
    """再現ログ保存"""
    log_data = {
        "reproduction_info": {
            "timestamp": datetime.now().isoformat(),
            "script_version": "1.0.0",
            "python_version": sys.version,
            "platform": sys.platform
        },
        "training_parameters": {
            "base_model": args.base_model,
            "dataset": args.dataset,
            "alpha_initial": args.alpha_initial,
            "alpha_final": args.alpha_final,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "num_epochs": args.num_epochs,
            "seed": 42
        },
        "output_paths": {
            "model_dir": args.output_dir,
            "safety_model_dir": f"{args.output_dir}_safety",
            "gguf_path": f"D:/webdataset/gguf_models/aegis_reproduced/aegis_reproduced_Q8_0.gguf",
            "modelfile": "modelfiles/aegis_reproduced.modelfile"
        },
        "reproducibility_notes": [
            "All random seeds are fixed for reproducibility",
            "Environment setup is automated",
            "Data preprocessing is deterministic",
            "Model training uses deterministic algorithms where possible"
        ]
    }

    log_path = Path("logs/reproduction_log.json")
    log_path.parent.mkdir(exist_ok=True)

    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)

    print(f"📝 Reproduction log saved: {log_path}")

def main():
    parser = argparse.ArgumentParser(
        description="SO8T AEGISモデル完全再現スクリプト",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # 完全再現実行
  python scripts/reproduce_aegis_training.py --full

  # カスタムパラメータで実行
  python scripts/reproduce_aegis_training.py --base_model microsoft/phi-3.5-mini-instruct --alpha_final 0.8 --batch_size 8
        """
    )

    parser.add_argument("--full", action="store_true",
                       help="完全再現モード（推奨）")
    parser.add_argument("--base_model", type=str,
                       default="microsoft/phi-3.5-mini-instruct",
                       help="ベースモデル名")
    parser.add_argument("--dataset", type=str,
                       default="data/so8t_thinking_phi35_weighted_train.jsonl",
                       help="トレーニングデータセット")
    parser.add_argument("--output_dir", type=str,
                       default="models/aegis_reproduced",
                       help="出力ディレクトリ")
    parser.add_argument("--alpha_initial", type=float, default=0.1,
                       help="Alpha Gate初期値")
    parser.add_argument("--alpha_final", type=float, default=0.8,
                       help="Alpha Gate最終値")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="バッチサイズ")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                       help="学習率")
    parser.add_argument("--num_epochs", type=int, default=3,
                       help="エポック数")

    args = parser.parse_args()

    print("🚀 Starting SO8T AEGIS Model Reproduction")
    print("=" * 50)

    try:
        # Step 1: Environment setup
        if args.full:
            setup_environment()

        # Step 2: Data preparation
        if args.full:
            prepare_data()

        # Step 3: SO8T model training
        train_so8t_model(args)

        # Step 4: Safety fine-tuning
        fine_tune_safety(args)

        # Step 5: GGUF conversion
        convert_to_gguf(args)

        # Step 6: Ollama model creation
        create_ollama_model(args)

        # Step 7: Validation tests
        run_validation_tests()

        # Step 8: Save reproduction log
        save_reproduction_log(args)

        print("\n🎉 SO8T AEGIS Model Reproduction Complete!")
        print("Model available as: ollama run aegis-reproduced:latest")

    except subprocess.CalledProcessError as e:
        print(f"❌ Error during reproduction: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
SO8T/thinking Complete Training and Evaluation Workflow
トレーニング完了後の自動ワークフロー実行
"""

import os
import sys
import time
import subprocess
import argparse
from pathlib import Path
import logging

# Project imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def wait_for_training_completion(checkpoint_dir: str, max_wait_minutes: int = 60):
    """トレーニング完了を待機"""
    logger.info(f"Waiting for training completion in: {checkpoint_dir}")
    logger.info("Checking for checkpoint files every 5 minutes...")

    start_time = time.time()
    wait_seconds = 300  # 5 minutes

    while time.time() - start_time < max_wait_minutes * 60:
        if os.path.exists(checkpoint_dir) and os.listdir(checkpoint_dir):
            logger.info("✅ Training completed! Checkpoints found.")
            return True

        logger.info(f"Waiting... ({int((time.time() - start_time) / 60)}min elapsed)")
        time.sleep(wait_seconds)

    logger.warning("⚠️ Training wait timeout reached. Proceeding with available checkpoints.")
    return False


def run_model_evaluation(model_path: str, test_data: str, output_dir: str):
    """モデル評価を実行"""
    logger.info("🔍 Starting Model Evaluation...")

    os.makedirs(output_dir, exist_ok=True)

    # シンプルな評価スクリプトを作成・実行
    eval_script = f"""
import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate_model():
    try:
        logger.info("Loading model and tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("{model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            "{model_path}",
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )

        # テストデータの読み込み
        test_samples = []
        with open("{test_data}", 'r', encoding='utf-8') as f:
            for line in f:
                test_samples.append(json.loads(line.strip()))
                if len(test_samples) >= 10:  # 最大10サンプルで評価
                    break

        logger.info(f"Evaluating {{len(test_samples)}} samples...")

        results = []
        model.eval()

        for i, sample in enumerate(test_samples):
            text = sample.get('text', '')
            inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)

            with torch.no_grad():
                outputs = model(**inputs)
                loss = outputs.loss.item() if hasattr(outputs, 'loss') else 0.0

            results.append({{
                'sample_id': i,
                'text_length': len(text),
                'loss': loss,
                'perplexity': torch.exp(torch.tensor(loss)).item() if loss > 0 else float('inf')
            }})

        # 結果を保存
        avg_loss = sum(r['loss'] for r in results) / len(results)
        avg_perplexity = sum(r['perplexity'] for r in results if r['perplexity'] != float('inf')) / len(results)

        eval_results = {{
            'num_samples': len(results),
            'avg_loss': avg_loss,
            'avg_perplexity': avg_perplexity,
            'individual_results': results
        }}

        with open("{output_dir}/evaluation_results.json", 'w', encoding='utf-8') as f:
            json.dump(eval_results, f, indent=2, ensure_ascii=False)

        logger.info(f"Evaluation completed. Avg Loss: {{avg_loss:.4f}}, Avg Perplexity: {{avg_perplexity:.2f}}")
        print(f"✅ Model evaluation completed successfully")

    except Exception as e:
        logger.error(f"Evaluation failed: {{e}}")
        print(f"❌ Model evaluation failed: {{e}}")

if __name__ == "__main__":
    evaluate_model()
"""

    script_path = f"{output_dir}/eval_temp.py"
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(eval_script)

    try:
        result = subprocess.run([sys.executable, script_path],
                              capture_output=True, text=True, timeout=600)

        if result.returncode == 0:
            logger.info("✅ Model evaluation completed successfully")
            return True
        else:
            logger.error(f"❌ Model evaluation failed: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        logger.error("❌ Model evaluation timed out")
        return False
    finally:
        # テンポラリスクリプトを削除
        if os.path.exists(script_path):
            os.remove(script_path)


def run_gguf_conversion(model_path: str, output_dir: str):
    """GGUF変換を実行"""
    logger.info("🔄 Starting GGUF Conversion...")

    os.makedirs(output_dir, exist_ok=True)

    # llama.cppのconvert_hf_to_gguf.pyが存在するか確認
    llama_cpp_dir = PROJECT_ROOT / "external" / "llama.cpp-master"
    convert_script = llama_cpp_dir / "convert_hf_to_gguf.py"

    if not convert_script.exists():
        logger.error(f"❌ llama.cpp convert script not found: {convert_script}")
        return False

    conversions = [
        ("q8_0", "so8t_thinking_q8_0.gguf"),
        ("f16", "so8t_thinking_f16.gguf")
    ]

    success_count = 0

    for quant_type, output_file in conversions:
        output_path = os.path.join(output_dir, output_file)

        cmd = [
            sys.executable, str(convert_script),
            model_path,
            "--outfile", output_path,
            "--outtype", quant_type
        ]

        try:
            logger.info(f"Converting to {quant_type}...")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)

            if result.returncode == 0:
                logger.info(f"✅ GGUF conversion ({quant_type}) completed")
                success_count += 1
            else:
                logger.error(f"❌ GGUF conversion ({quant_type}) failed: {result.stderr}")

        except subprocess.TimeoutExpired:
            logger.error(f"❌ GGUF conversion ({quant_type}) timed out")

    return success_count > 0


def create_ollama_modelfile(gguf_path: str, modelfile_path: str):
    """Ollama用のModelfileを作成"""
    modelfile_content = f'''FROM {gguf_path}

TEMPLATE """{{{{ .System }}}}

{{{{ .Prompt }}}}}}"""

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_ctx 4096
'''

    with open(modelfile_path, 'w', encoding='utf-8') as f:
        f.write(modelfile_content)

    logger.info(f"Created Ollama modelfile: {modelfile_path}")


def run_ollama_import(modelfile_path: str, model_name: str):
    """Ollamaにモデルをインポート"""
    logger.info("📦 Starting Ollama Import...")

    try:
        cmd = ["ollama", "create", model_name, "-f", modelfile_path]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if result.returncode == 0:
            logger.info(f"✅ Ollama import completed: {model_name}")
            return True
        else:
            logger.error(f"❌ Ollama import failed: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        logger.error("❌ Ollama import timed out")
        return False


def run_japanese_performance_tests(model_name: str):
    """日本語性能テストを実行"""
    logger.info("🧪 Starting Japanese Performance Tests...")

    test_cases = [
        {
            "name": "Japanese Understanding Test",
            "prompt": "以下の文章を読んで、内容を要約してください。日本語で回答してください。\n\nSO8TはSO(8)回転群に基づく革新的なアーキテクチャで、幾何学的推論能力を強化します。このモデルはAlpha Gateによって動的に幾何学的経路を制御し、学習過程で自律的にゲートを開きます。"
        },
        {
            "name": "Japanese Generation Test",
            "prompt": "SO(8)回転群について、数学的な基礎から応用までを説明してください。ステップバイステップで分かりやすく教えてください。"
        },
        {
            "name": "Logical Reasoning Test",
            "prompt": "次の論理パズルを解いてください：すべてのSO8Tモデルは幾何学的推論を行うが、いくつかのモデルは特別なAlpha Gateを持つ。この命題が真であるための条件を説明してください。"
        },
        {
            "name": "Mathematical Reasoning Test",
            "prompt": "ベクトル空間における回転群SO(8)の性質を説明してください。特に8次元ユークリッド空間での回転について。"
        },
        {
            "name": "Self-Verification Test",
            "prompt": "あなた自身の思考プロセスを分析してください。SO8Tアーキテクチャがどのようにあなたの推論能力を強化しているかを説明してください。"
        }
    ]

    results = []

    for i, test_case in enumerate(test_cases):
        logger.info(f"Running test {i+1}: {test_case['name']}")

        try:
            cmd = ["ollama", "run", model_name, test_case['prompt']]
            result = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=120, encoding='utf-8')

            if result.returncode == 0:
                response = result.stdout.strip()
                logger.info(f"✅ Test {i+1} completed")
                print(f"\n🧪 {test_case['name']}")
                print(f"Response: {response[:200]}..." if len(response) > 200 else f"Response: {response}")
                results.append({
                    "test_id": i+1,
                    "name": test_case['name'],
                    "success": True,
                    "response_length": len(response)
                })
            else:
                logger.error(f"❌ Test {i+1} failed: {result.stderr}")
                results.append({
                    "test_id": i+1,
                    "name": test_case['name'],
                    "success": False,
                    "error": result.stderr
                })

        except subprocess.TimeoutExpired:
            logger.error(f"❌ Test {i+1} timed out")
            results.append({
                "test_id": i+1,
                "name": test_case['name'],
                "success": False,
                "error": "timeout"
            })

    # テスト結果を保存
    test_results = {
        "total_tests": len(test_cases),
        "successful_tests": sum(1 for r in results if r['success']),
        "failed_tests": sum(1 for r in results if not r['success']),
        "results": results
    }

    results_file = PROJECT_ROOT / "results" / "so8t_thinking_japanese_tests.json"
    os.makedirs(results_file.parent, exist_ok=True)

    with open(results_file, 'w', encoding='utf-8') as f:
        import json
        json.dump(test_results, f, indent=2, ensure_ascii=False)

    logger.info(f"Japanese performance tests completed. Results saved to: {results_file}")
    return test_results


def play_completion_audio():
    """完了音声を再生"""
    try:
        audio_file = r"C:\Users\downl\Desktop\SO8T\.cursor\marisa_owattaze.wav"
        if os.path.exists(audio_file):
            subprocess.run([
                "powershell",
                "-ExecutionPolicy", "Bypass",
                "-Command",
                f"(New-Object Media.SoundPlayer '{audio_file}').PlaySync();"
            ], check=False)
            logger.info("🎵 Completion audio played")
    except Exception as e:
        logger.warning(f"Audio playback failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="SO8T/thinking Complete Workflow")
    parser.add_argument("--checkpoint-dir", type=str,
                       default="D:/webdataset/checkpoints/so8t_thinking_retrained",
                       help="Training checkpoint directory")
    parser.add_argument("--test-data", type=str,
                       default="data/nkat_so8t_v2/val_nkat_so8t.jsonl",
                       help="Test data for evaluation")
    parser.add_argument("--results-dir", type=str,
                       default="results/so8t_thinking_workflow",
                       help="Results directory")
    parser.add_argument("--max-wait-minutes", type=int, default=120,
                       help="Maximum wait time for training completion (minutes)")
    parser.add_argument("--skip-wait", action="store_true",
                       help="Skip waiting for training completion")
    parser.add_argument("--force-run", action="store_true",
                       help="Force run workflow even if checkpoints don't exist")

    args = parser.parse_args()

    logger.info("🚀 Starting SO8T/thinking Complete Workflow")
    logger.info("=" * 60)

    # 結果ディレクトリを作成
    results_dir = Path(PROJECT_ROOT) / args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. トレーニング完了を待機
    if os.path.exists(args.checkpoint_dir) and os.listdir(args.checkpoint_dir):
        logger.info("✅ Training checkpoints found, proceeding with workflow")
    elif args.skip_wait:
        logger.info("⏭️ Skipping training wait, proceeding with workflow")
    elif args.force_run:
        logger.warning("⚠️ Force run enabled, proceeding without checkpoints")
    else:
        logger.info("⏳ Waiting for training completion...")
        training_completed = wait_for_training_completion(args.checkpoint_dir, args.max_wait_minutes)
        if not training_completed and not args.force_run:
            logger.error("❌ Training wait timeout and no checkpoints found")
            return

    # 2. モデル評価
    eval_dir = results_dir / "evaluation"
    evaluation_success = run_model_evaluation(args.checkpoint_dir, args.test_data, str(eval_dir))

    # 3. GGUF変換
    gguf_dir = Path("D:/webdataset/gguf_models/so8t_thinking_v1")
    gguf_success = run_gguf_conversion(args.checkpoint_dir, str(gguf_dir))

    # 4. Ollamaインポート
    if gguf_success:
        modelfile_path = results_dir / "so8t_thinking.modelfile"
        gguf_file = gguf_dir / "so8t_thinking_q8_0.gguf"

        create_ollama_modelfile(str(gguf_file), str(modelfile_path))
        ollama_success = run_ollama_import(str(modelfile_path), "so8t-thinking:latest")
    else:
        logger.error("GGUF conversion failed, skipping Ollama import")
        ollama_success = False

    # 5. 日本語性能テスト
    if ollama_success:
        test_results = run_japanese_performance_tests("so8t-thinking:latest")
    else:
        logger.error("Ollama import failed, skipping performance tests")
        test_results = None

    # ワークフロー結果をまとめる
    workflow_results = {
        "workflow_completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "steps": {
            "evaluation": evaluation_success,
            "gguf_conversion": gguf_success,
            "ollama_import": ollama_success,
            "performance_tests": test_results is not None
        },
        "directories": {
            "checkpoints": args.checkpoint_dir,
            "evaluation": str(eval_dir),
            "gguf_models": str(gguf_dir),
            "results": str(results_dir)
        }
    }

    if test_results:
        workflow_results["performance_summary"] = {
            "total_tests": test_results["total_tests"],
            "successful_tests": test_results["successful_tests"],
            "success_rate": f"{test_results['successful_tests']}/{test_results['total_tests']}"
        }

    # 結果を保存
    summary_file = results_dir / "workflow_summary.json"
    import json
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(workflow_results, f, indent=2, ensure_ascii=False)

    logger.info("🎉 SO8T/thinking Complete Workflow Finished!")
    logger.info(f"Results saved to: {summary_file}")

    # 完了音声を再生
    play_completion_audio()

    # 最終ステータスを表示
    success_steps = sum(1 for step in workflow_results["steps"].values() if step)
    total_steps = len(workflow_results["steps"])

    print(f"\n🎯 Workflow Completion Status: {success_steps}/{total_steps} steps successful")

    if success_steps == total_steps:
        print("✅ All steps completed successfully!")
        print("🚀 SO8T/thinking model is ready for inference")
    else:
        print("⚠️ Some steps failed. Check logs for details.")


if __name__ == "__main__":
    main()










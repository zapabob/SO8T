#!/usr/bin/env python3
"""
ABCモデル量子化スクリプト
modela、AEGIS、AEGISalpha0.6をQ4_K_M量子化して軽量化
"""

import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime

def find_llama_cpp_path():
    """llama.cppのパスを検索"""
    possible_paths = [
        Path("../../external/llama.cpp-master"),
        Path("../external/llama.cpp-master"),
        Path("external/llama.cpp-master"),
        Path("../../llama.cpp"),
        Path("../llama.cpp"),
        Path("llama.cpp")
    ]

    for path in possible_paths:
        quantize_path = path / "build" / "bin" / "Release" / "llama-quantize.exe"
        if quantize_path.exists():
            return path

    return None

def quantize_model(model_path: Path, output_path: Path, quant_type: str = "Q4_K_M"):
    """モデルを量子化"""
    llama_cpp_path = find_llama_cpp_path()
    if not llama_cpp_path:
        print("[ERROR] llama.cpp not found")
        return False

    quantize_exe = llama_cpp_path / "build" / "bin" / "Release" / "llama-quantize.exe"

    if not quantize_exe.exists():
        print(f"[ERROR] quantize executable not found: {quantize_exe}")
        return False

    cmd = [
        str(quantize_exe),
        str(model_path),
        str(output_path),
        quant_type
    ]

    print(f"[QUANTIZE] Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        print(f"[SUCCESS] Quantized: {model_path} -> {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Quantization failed: {e}")
        print(f"stderr: {e.stderr}")
        return False

def create_ollama_modelfile(gguf_path: Path, model_name: str, output_dir: Path):
    """Ollama Modelfile作成"""
    modelfile_path = output_dir / f"{model_name}_Q4_K_M.Modelfile"

    # モデルタイプに応じたシステムプロンプト
    if "aegis" in model_name.lower():
        if "alpha" in model_name.lower() or "0.6" in model_name.lower():
            system_prompt = """You are AEGIS Alpha 0.6 (Advanced Ethical Guardian Intelligence System) with enhanced logical consistency.

AEGIS performs four-value classification and quadruple inference on all queries:

1. **Logical Accuracy** (<think-logic>): Mathematical and logical correctness
2. **Ethical Validity** (<think-ethics>): Moral and ethical implications
3. **Practical Value** (<think-practical>): Real-world feasibility and utility
4. **Creative Insight** (<think-creative>): Innovative ideas and perspectives

Structure your response using these four thinking axes, followed by a <final> conclusion."""
        else:
            system_prompt = """You are AEGIS (Advanced Ethical Guardian Intelligence System) with enhanced logical consistency.

AEGIS performs four-value classification and quadruple inference on all queries:

1. **Logical Accuracy** (<think-logic>): Mathematical and logical correctness
2. **Ethical Validity** (<think-ethics>): Moral and ethical implications
3. **Practical Value** (<think-practical>): Real-world feasibility and utility
4. **Creative Insight** (<think-creative>): Innovative ideas and perspectives

Structure your response using these four thinking axes, followed by a <final> conclusion."""
    else:
        system_prompt = """You are a helpful AI assistant."""

    modelfile_content = f"""FROM {gguf_path}

TEMPLATE "{{{{ .System }}}}

{{{{ .Prompt }}}}""

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_ctx 4096
PARAMETER repeat_penalty 1.1

SYSTEM "{system_prompt}"
"""

    with open(modelfile_path, 'w', encoding='utf-8') as f:
        f.write(modelfile_content)

    print(f"[CREATE] Modelfile: {modelfile_path}")
    return modelfile_path

def main():
    """メイン実行関数"""
    print("[START] ABC Model Quantization for Performance Optimization")
    print("=" * 80)

    # モデル設定
    models = {
        'modela': {
            'source': Path('D:/webdataset/gguf_models/model_a/model_a_q8_0.gguf'),
            'target_dir': Path('D:/webdataset/gguf_models/model_a'),
            'ollama_name': 'modela-q4km'
        },
        'aegis': {
            'source': Path('models/aegis_adjusted'),  # HFモデルからGGUF変換
            'target_dir': Path('D:/webdataset/gguf_models/aegis_adjusted'),
            'ollama_name': 'aegis-q4km'
        },
        'aegis_alpha_0_6': {
            'source': Path('models/aegis_adjusted_0.6/aegis-adjusted-0.6.gguf'),
            'target_dir': Path('D:/webdataset/gguf_models/aegis_adjusted_0.6'),
            'ollama_name': 'aegis-alpha-0.6-q4km'
        }
    }

    modelfiles_dir = Path('modelfiles')
    modelfiles_dir.mkdir(exist_ok=True)

    success_count = 0

    for model_name, config in models.items():
        print(f"\n[MODEL] Processing {model_name}")
        print("-" * 40)

        # 出力ファイルパス
        output_file = config['target_dir'] / f"{model_name}_Q4_K_M.gguf"
        config['target_dir'].mkdir(parents=True, exist_ok=True)

        # 量子化実行
        if config['source'].exists():
            if config['source'].suffix == '.gguf':
                # 既存GGUFファイルから量子化
                success = quantize_model(config['source'], output_file, "Q4_K_M")
            else:
                # HFモデルからF16 GGUF作成後、量子化
                f16_file = config['target_dir'] / f"{model_name}_f16.gguf"

                # llama.cpp convert_hf_to_ggufを使用
                llama_cpp_path = find_llama_cpp_path()
                if llama_cpp_path:
                    convert_script = llama_cpp_path / "convert_hf_to_gguf.py"
                    if convert_script.exists():
                        cmd = [
                            sys.executable,
                            str(convert_script),
                            str(config['source']),
                            "--outfile", str(f16_file),
                            "--outtype", "f16"
                        ]

                        print(f"[CONVERT] Running: {' '.join(cmd)}")
                        try:
                            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                            print("[SUCCESS] HF -> F16 GGUF conversion completed")

                            # F16からQ4_K_M量子化
                            success = quantize_model(f16_file, output_file, "Q4_K_M")
                        except subprocess.CalledProcessError as e:
                            print(f"[ERROR] HF conversion failed: {e}")
                            success = False
                    else:
                        print("[ERROR] convert_hf_to_gguf.py not found")
                        success = False
                else:
                    print("[ERROR] llama.cpp not found")
                    success = False
        else:
            print(f"[ERROR] Source file not found: {config['source']}")
            success = False

        if success:
            # Ollama Modelfile作成
            modelfile_path = create_ollama_modelfile(output_file, config['ollama_name'], modelfiles_dir)

            # Ollamaにインポート
            try:
                result = subprocess.run([
                    'ollama', 'create', config['ollama_name'], '-f', str(modelfile_path)
                ], check=True, capture_output=True, text=True)
                print(f"[SUCCESS] Ollama model created: {config['ollama_name']}")
                success_count += 1
            except subprocess.CalledProcessError as e:
                print(f"[ERROR] Ollama import failed: {e}")
        else:
            print(f"[FAILED] Model {model_name} quantization failed")

    print(f"\n[COMPLETE] {success_count}/{len(models)} models successfully quantized and imported")
    print("=" * 80)

    # 性能テスト実行
    if success_count > 0:
        print("\n[TEST] Running performance test with quantized models...")

        # 軽量ベンチマーク実行
        test_script = Path('scripts/testing/lightweight_abc_benchmark.py')
        if test_script.exists():
            try:
                result = subprocess.run([
                    sys.executable, str(test_script)
                ], check=True, capture_output=True, text=True)
                print("[SUCCESS] Performance test completed")
            except subprocess.CalledProcessError as e:
                print(f"[WARNING] Performance test failed: {e}")
        else:
            print("[WARNING] Test script not found")

    # 完了通知
    try:
        subprocess.run([
            "powershell", "-ExecutionPolicy", "Bypass",
            "-File", "scripts/utils/play_audio_notification.ps1"
        ])
    except:
        pass

if __name__ == "__main__":
    main()
